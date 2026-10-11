import json
import sys
from pathlib import Path

import pytest
import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader

from xvc2_codec.vc_data import VCPairDataConfig, VCPairDataset, main


def make_manifest(tmp_path: Path, source_frames: int = 8, target_frames: int = 8) -> Path:
    source, target = tmp_path / "generated.flac", tmp_path / "natural.wav"
    for path, frames, sign in ((source, source_frames, 1), (target, target_frames, -1)):
        waveform = torch.linspace(0.05, 0.3, frames * 320).unsqueeze(0) * sign
        torchaudio.save(str(path), waveform, 16_000)
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "index": 17,
                "source_utt": "utt_11",
                "target_utt": "utt_01",
                "source_wav_path": str(source),
                "target_wav_path": str(target),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def config(**kwargs) -> VCPairDataConfig:
    return VCPairDataConfig(
        segment_duration=0.06,
        max_val_duration=0.1,
        volume_normalize=False,
        highpass_cutoff_freq=0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "draw,role,source_name,target_name",
    [
        (0.1, "reconstruction", "natural.wav", "natural.wav"),
        (0.3, "reversed", "natural.wav", "generated.flac"),
        (0.9, "standard", "generated.flac", "natural.wav"),
    ],
)
def test_xvc_roles_route_both_crops_and_conditions(
    tmp_path, monkeypatch, draw, role, source_name, target_name
):
    manifest = make_manifest(tmp_path)
    monkeypatch.setattr(torch, "rand", lambda *args: torch.tensor(draw))
    monkeypatch.setattr(torch, "randint", lambda *args: torch.tensor(2))
    row = VCPairDataset(manifest, config())[0]
    assert row["role_assignment_mode"] == role
    assert Path(row["source_wav_path"]).name == source_name
    assert Path(row["target_wav_path"]).name == target_name
    assert row["index"] == 17
    assert row["manifest_target_utt"] == "utt_01"
    expected_ids = {
        "standard": ("utt_11", "utt_01"),
        "reversed": ("utt_01", "utt_11"),
        "reconstruction": ("utt_01", "utt_01"),
    }
    assert (row["source_utt"], row["target_utt"]) == expected_ids[role]
    assert row["start_frame"] == 2
    for view in ("source", "target"):
        original, _ = torchaudio.load(row[f"{view}_wav_path"])
        torch.testing.assert_close(row[view]["waveform"], original[:, 640:1600])
        assert row[view]["frames"] == 3
        condition = row[f"{view}_wav_cond"]
        assert torch.count_nonzero(condition[:, 640:1600]) == 0
        torch.testing.assert_close(condition[:, :640], original[:, :640])
        torch.testing.assert_close(condition[:, 1600:], original[:, 1600:])
        assert torch.count_nonzero(row[view]["waveform"]) == 960


def test_unequal_pair_uses_real_shared_frames_and_short_crop_padding(tmp_path):
    manifest = make_manifest(tmp_path, source_frames=2, target_frames=7)
    row = VCPairDataset(manifest, config(reconstruction_ratio=0, reversed_ratio=0))[0]
    assert row["source_num_samples"] == 640
    assert row["target_num_samples"] == 2240
    assert row["condition_frames"] == 2
    for view in ("source", "target"):
        assert row[view]["frames"] == 2
        assert row[view]["waveform"].shape == (1, 960)
        assert torch.count_nonzero(row[view]["waveform"][:, :640]) == 640
        assert torch.count_nonzero(row[view]["waveform"][:, 640:]) == 0
        assert row[f"{view}_wav_cond"].shape == (1, 640)
        assert torch.count_nonzero(row[f"{view}_wav_cond"]) == 0


def test_unmasked_condition_keeps_full_pair_audio(tmp_path):
    manifest = make_manifest(tmp_path)
    row = VCPairDataset(manifest, config(mask_target_condition=False))[0]
    for view in ("source", "target"):
        original, _ = torchaudio.load(row[f"{view}_wav_path"])
        torch.testing.assert_close(row[f"{view}_wav_cond"], original)


def test_validation_direction_is_standard_and_batch_padding_keeps_lengths(tmp_path):
    manifest = make_manifest(tmp_path)
    dataset = VCPairDataset(manifest, config(reconstruction_ratio=1, reversed_ratio=0), False)
    long = dataset[0]
    assert long["role_assignment_mode"] == "standard"
    assert long["start_frame"] == 0
    assert long["source"]["waveform"].shape == (1, 1600)
    short_dir = tmp_path / "short"
    short_dir.mkdir()
    short = VCPairDataset(make_manifest(short_dir, 2, 2), config(), False)[0]
    batch = dataset.collate([long, short])
    for view in ("source", "target"):
        assert batch[view]["waveform"].shape == (2, 1, 1600)
        assert batch[view]["frames"].tolist() == [5, 2]
        assert torch.count_nonzero(batch[view]["waveform"][1, :, 640:]) == 0
    assert batch["target_wav_cond"].shape == (2, 1, 2560)
    assert batch["condition_frames"].tolist() == [8, 2]
    assert torch.count_nonzero(batch["target_wav_cond"][1]) == 0
    torch.testing.assert_close(long["source"]["waveform"], dataset[0]["source"]["waveform"])


def test_stereo_and_non_16k_audio_are_codec_shaped(tmp_path):
    manifest = make_manifest(tmp_path)
    rows = [json.loads(manifest.read_text())]
    stereo = tmp_path / "stereo.wav"
    torchaudio.save(
        str(stereo), torch.stack((torch.ones(1280) * 0.3, torch.ones(1280) * 0.1)), 8000
    )
    rows[0]["source_wav_path"] = str(stereo)
    manifest.write_text(json.dumps(rows[0]) + "\n")
    row = VCPairDataset(manifest, config(), False)[0]
    assert row["source_num_samples"] == 2560
    assert row["source"]["waveform"].shape == (1, 1600)
    assert row["source"]["waveform"][0, 320:1280].mean().item() == pytest.approx(0.2, abs=1e-3)


def test_xvc_preprocessing_normalizes_volume_and_highpasses_dc(tmp_path):
    manifest = make_manifest(tmp_path)
    audio = tmp_path / "generated.flac"
    torchaudio.save(str(audio), torch.ones(1, 2560) * 0.5, 16_000)
    normalized = VCPairDataset(manifest, VCPairDataConfig(highpass_cutoff_freq=0), False)[0][
        "source"
    ]["waveform"]
    assert normalized.mean().item() == pytest.approx(0.2, abs=1e-4)
    filtered = VCPairDataset(manifest, VCPairDataConfig(), False)[0]["source"]["waveform"]
    assert filtered[:, 1600:].abs().mean().item() < 1e-3


@pytest.mark.parametrize(
    "options",
    [
        {"reconstruction_ratio": 0.8, "reversed_ratio": 0.4},
        {"reversed_ratio": -0.1},
        {"segment_duration": 0.001},
        {"highpass_cutoff_freq": 8000},
    ],
)
def test_invalid_sampling_config_is_rejected(options):
    with pytest.raises(ValueError):
        VCPairDataConfig(**options)


def test_subframe_pair_is_rejected_instead_of_training_on_padding(tmp_path):
    manifest = make_manifest(tmp_path)
    torchaudio.save(str(tmp_path / "generated.flac"), torch.ones(1, 100), 16_000)
    with pytest.raises(ValueError, match="No complete shared"):
        VCPairDataset(manifest, config(), False)[0]


def test_loader_seed_reproduces_roles_and_crops(tmp_path):
    manifest = make_manifest(tmp_path, 20, 20)
    row = json.loads(manifest.read_text())
    manifest.write_text("".join(json.dumps({**row, "index": index}) + "\n" for index in range(32)))
    dataset = VCPairDataset(manifest, config())

    def sample():
        torch.manual_seed(42)
        batches = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate)
        return [(batch["role_assignment_mode"], batch["start_frame"]) for batch in batches]

    first, second = sample(), sample()
    assert first == second
    assert len({role for roles, _ in first for role in roles}) == 3
    assert len({start for _, starts in first for start in starts}) > 1


def test_io_command_reads_original_jsonl_and_writes_role_report(tmp_path, monkeypatch):
    manifest = make_manifest(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "reconstruction_ratio": 0,
                "reversed_ratio": 1,
                "volume_normalize": False,
                "highpass_cutoff_freq": 0,
            }
        )
    )
    output = tmp_path / "report" / "io.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vc_data",
            "--manifest",
            str(manifest),
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--max-pairs",
            "1",
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert report["roles"] == {"reversed": 1}
    assert report["sample_rate"] == 16000
    assert report["hop_length"] == 320
    record = report["records"][0]
    assert record["source_wav_path"].endswith("natural.wav")
    assert record["target_wav_path"].endswith("generated.flac")
    assert record["valid_frames"] == 8
    assert record["batch_samples"] == 38400
    assert record["target_condition_nonzero_samples"] == 0
