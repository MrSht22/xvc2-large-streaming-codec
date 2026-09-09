import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch

from xvc2_codec.cache import temporal_cache, vector_cache
from xvc2_codec.audit import audit_manifests
from xvc2_codec.config import CodecConfig
from xvc2_codec.data import PairDataset
from xvc2_codec.preprocess import (
    cache_fields,
    command_finalize,
    command_plan,
    extract_shard,
    extraction_batches,
    lagged_frame_metrics,
    stable_item_id,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_wav(path: Path, seconds: float = 0.2) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * round(seconds * 16_000))


def test_sharded_cache_reads_only_requested_offset(tmp_path: Path) -> None:
    path = tmp_path / "cache.bin"
    values = np.arange(60, dtype=np.float16).reshape(20, 3)
    values.tofile(path)
    row = {
        "student_hidden_path": str(path),
        "student_hidden_offset_frames": 5,
        "student_hidden_frames": 10,
        "student_hidden_dim": 3,
        "student_hidden_dtype": "float16",
    }
    cache = temporal_cache(row, "student_hidden", "student_hidden")
    assert cache is not None
    torch.testing.assert_close(cache.read(2, 4), torch.from_numpy(values[7:11]).float())


def test_indexed_vector_cache(tmp_path: Path) -> None:
    path = tmp_path / "vectors.pt"
    torch.save(torch.arange(12).view(3, 4), path)
    value = vector_cache(
        {"speaker_target_path": str(path), "speaker_target_index": 1},
        "speaker_target",
        "speaker_target",
    )
    torch.testing.assert_close(value, torch.arange(4, 8).float())


def test_plan_builds_inventory_and_speaker_references(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    sa = tmp_path / "sa.wav"
    write_wav(audio)
    write_wav(sa)
    source_manifest = tmp_path / "source.jsonl"
    pair_manifest = tmp_path / "pairs.jsonl"
    source_row = {
        "utterance_id": "u1",
        "speaker_id": "s1",
        "audio_path": str(audio),
        "duration_seconds": 0.2,
    }
    pair_row = {
        "utterance_id": "u1",
        "speaker_id": "s1",
        "normalized_source_audio_path": str(audio),
        "anonymized_audio_path": str(sa),
        "source_audio": {"duration_seconds": 0.2},
        "anonymized_audio": {"duration_seconds": 0.2},
    }
    write_jsonl(source_manifest, [source_row])
    write_jsonl(pair_manifest, [pair_row])
    output = tmp_path / "plan"
    command_plan(
        argparse.Namespace(
            source_manifest=source_manifest,
            pair_manifest=pair_manifest,
            output_dir=output,
            references_per_speaker=3,
            space_safety_factor=1.0,
        )
    )
    report = json.loads((output / "plan.json").read_text())
    assert report["status"] == "PASS"
    assert report["inventory_items"] == 3
    assert report["pair_duration_alignment"]["status"] == "PASS"
    assert report["pair_duration_alignment"]["fraction_within_one_50hz_frame"] == 1.0
    assert (output / "speaker_references" / "wav.scp").is_file()


def test_extraction_batches_limit_padded_audio() -> None:
    rows = [{"duration_seconds": value} for value in (1, 1, 4, 4, 4)]
    batches = extraction_batches(rows, batch_size=4, maximum_batch_samples=8 * 16_000)
    assert batches == [[0, 1], [2, 3], [4]]


def test_lagged_frame_metrics_recovers_known_shift() -> None:
    generator = torch.Generator().manual_seed(3)
    source = torch.randn(30, 8, generator=generator)
    anonymized = torch.cat((torch.randn(3, 8, generator=generator), source[:-3]))
    metrics = lagged_frame_metrics(source, anonymized, maximum_lag=5)
    assert metrics["best_lag_frames"] == 3
    assert metrics["best_lag_cosine"] > 0.99


class FakeStudent(torch.nn.Module):
    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        output_lengths = torch.div(lengths, 320, rounding_mode="floor")
        frames = int(output_lengths.max())
        return {
            "hidden_states": torch.ones(len(lengths), frames, 768),
            "phone_logits": torch.full((len(lengths), frames, 40), 2.0),
            "output_lengths": output_lengths,
        }


def test_extract_shard_is_resumable(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    checkpoint = tmp_path / "student.pt"
    write_wav(audio)
    torch.save({}, checkpoint)
    rows = [
        {
            "item_id": "source:u1:x",
            "audio_path": str(audio),
            "duration_seconds": 0.2,
            "estimated_frames": 10,
        }
    ]
    output = tmp_path / "cache"
    arguments = (FakeStudent(), rows, output, 0, 0, checkpoint, torch.device("cpu"), 2, 32_000, 0)
    assert extract_shard(*arguments) == 1
    first_mtime = (output / "hidden-r00-s00000.bin").stat().st_mtime_ns
    assert extract_shard(*arguments) == 1
    assert (output / "hidden-r00-s00000.bin").stat().st_mtime_ns == first_mtime
    index = json.loads((output / "index-r00-s00000.jsonl").read_text())
    cache = temporal_cache(index, "student_hidden", "student_hidden")
    assert cache is not None and cache.shape == (10, 768)


def test_finalize_joins_source_and_sa_views(tmp_path: Path) -> None:
    audio = tmp_path / "source.wav"
    sa = tmp_path / "sa.wav"
    write_wav(audio)
    write_wav(sa)
    source_row = {
        "utterance_id": "u1",
        "speaker_id": "s1",
        "audio_path": str(audio),
        "duration_seconds": 0.2,
    }
    pair_row = {
        "utterance_id": "u1",
        "speaker_id": "s1",
        "normalized_source_audio_path": str(audio),
        "anonymized_audio_path": str(sa),
    }
    source_manifest, pair_manifest = tmp_path / "source.jsonl", tmp_path / "pair.jsonl"
    write_jsonl(source_manifest, [source_row])
    write_jsonl(pair_manifest, [pair_row])
    student_dir = tmp_path / "student"
    student_dir.mkdir()
    cache_path = student_dir / "hidden.bin"
    np.zeros((30, 768), dtype=np.float16).tofile(cache_path)
    phone_path = student_dir / "phone.bin"
    np.zeros((30, 40), dtype=np.float16).tofile(phone_path)
    base_cache = {
        "student_hidden_path": str(cache_path),
        "student_hidden_offset_frames": 0,
        "student_hidden_frames": 10,
        "student_hidden_dim": 768,
        "student_hidden_dtype": "float16",
        "phone_target_path": str(phone_path),
        "phone_target_offset_frames": 0,
        "phone_target_frames": 10,
        "phone_target_dim": 40,
        "phone_target_dtype": "float16",
    }
    index_rows = []
    for role, path, offset in (
        ("source", audio, 0),
        ("pair-source", audio, 10),
        ("pair-sa", sa, 20),
    ):
        fields = dict(base_cache)
        fields["student_hidden_offset_frames"] = offset
        fields["phone_target_offset_frames"] = offset
        index_rows.append(
            {"item_id": stable_item_id(role, "u1", str(path)), "audio_path": str(path), **fields}
        )
    write_jsonl(student_dir / "index-r00-s00000.jsonl", index_rows)
    speaker_dir = tmp_path / "speakers"
    source_store = speaker_dir / "source" / "codec_source" / "spk-level"
    source_store.mkdir(parents=True)
    (source_store / "id2idx").write_text("s1 0\n")
    torch.save(torch.zeros(1, 128), source_store / "speaker_vectors.pt")
    speaker_dir.mkdir(exist_ok=True)
    (speaker_dir / "sa_id2idx").write_text("u1 0\n")
    torch.save(torch.ones(1, 128), speaker_dir / "sa_speaker_vectors.pt")
    output = tmp_path / "final"
    command_finalize(
        argparse.Namespace(
            source_manifest=source_manifest,
            pair_manifest=pair_manifest,
            student_cache_dir=student_dir,
            speaker_cache_dir=speaker_dir,
            output_dir=output,
        )
    )
    source = json.loads((output / "source_train_cache.jsonl").read_text())
    pair = json.loads((output / "pair_train_cache.jsonl").read_text())
    assert source["speaker_target_index"] == 0
    assert pair["source"]["student_hidden_offset_frames"] == 10
    assert pair["sa"]["student_hidden_offset_frames"] == 20
    assert set(cache_fields(pair["sa"])) == set(base_cache)
    report = audit_manifests(
        output / "source_train_cache.jsonl",
        output / "pair_train_cache.jsonl",
        CodecConfig(),
        speaker_target_dim=128,
    )
    assert report["status"] == "PASS"
    loaded = PairDataset([pair], hop_length=320, segment_frames=5).load(0, crop_seed=3)
    assert loaded["source"]["student_hidden"].shape == (5, 768)
    assert loaded["sa"]["speaker_target"].shape == (128,)
