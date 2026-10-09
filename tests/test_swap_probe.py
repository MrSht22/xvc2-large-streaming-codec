import argparse
import json
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

from xvc2_codec.config import ExperimentConfig
from xvc2_codec.model import LargeStreamingCodec
from xvc2_codec.smoke import tiny_config
from xvc2_codec.swap_probe import (
    decode_swaps,
    frame_mapping,
    generate,
    load_codec,
    map_latent,
    normalize_pairs,
)
from xvc2_codec.swap_report import (
    REQUIRED,
    evaluate,
    intervention_rows,
    matrix_summary,
    prosody_distances,
    select_examples,
)


def test_single_swaps_use_only_requested_donor_branch():
    class StubCodec:
        def keep_head(self, hidden):
            return {"z_inv": hidden, "z_dyn": hidden + 10}

        def acoustic_encoder(self, wave):
            return wave + 20

        def fusion(self, inv, dyn, edit):
            return inv.transpose(1, 2) + 100 * dyn.transpose(1, 2) + 10000 * edit

        def decoder(self, fused):
            return fused

    outputs = decode_swaps(
        StubCodec(),
        torch.ones(1, 3),
        torch.ones(3, 1),
        torch.full((1, 4), 2.0),
        torch.full((4, 1), 2.0),
        frame_mapping(3, 4, None),
    )
    assert set(outputs) == {"AAA", "BBB", "BBB_mapped", "BAA", "ABA", "AAB"}
    torch.testing.assert_close(outputs["BAA"] - outputs["AAA"], torch.ones(1, 3))
    torch.testing.assert_close(outputs["ABA"] - outputs["AAA"], torch.full((1, 3), 100.0))
    torch.testing.assert_close(outputs["AAB"] - outputs["AAA"], torch.full((1, 3), 10000.0))


def test_mapping_interpolates_all_latent_layouts_and_rejects_invalid_maps(tmp_path):
    inv = torch.tensor([[[0.0], [10.0], [20.0]]])
    mapping = torch.tensor([0.0, 0.5, 2.0])
    expected = torch.tensor([[[0.0], [5.0], [20.0]]])
    torch.testing.assert_close(map_latent(inv, mapping, "z_dyn"), expected)
    torch.testing.assert_close(
        map_latent(inv.transpose(1, 2), mapping, "z_edit"), expected.transpose(1, 2)
    )
    for values in ([0, 2, 1], [0, 1, 3], [0, float("nan"), 2], [0, 1]):
        path = tmp_path / "map.npy"
        np.save(path, np.array(values))
        with pytest.raises(ValueError, match="Frame map"):
            frame_mapping(3, 3, str(path))


def test_seed_export_reads_libritts_text_and_selects_other_source_reference(tmp_path):
    sources = [
        tmp_path / chapter / f"source{i}.wav"
        for i, chapter in enumerate(("chapter1", "chapter1", "chapter2"))
    ]
    for source in sources:
        source.parent.mkdir(exist_ok=True)
        source.with_suffix(".normalized.txt").write_text("The normalized transcript.")
    rows = [
        {
            "index": i + 1,
            "source_speaker": "8455",
            "target_speaker": "4077",
            "conversion": "same" if i % 2 == 0 else "cross",
            "source_audio_path": str(source),
            "converted_audio_path": str(tmp_path / f"vc{i}.wav"),
            "target_reference_audio_path": str(tmp_path / "target.wav"),
        }
        for i, source in enumerate(sources)
    ]
    normalized = normalize_pairs(rows)
    assert normalized[0]["pair_id"] == "seedvc_001"
    assert normalized[0]["a"]["audio_path"] == rows[0]["source_audio_path"]
    assert normalized[0]["b"]["audio_path"] == rows[0]["converted_audio_path"]
    assert normalized[0]["b_identity_reference_path"] == rows[0]["target_reference_audio_path"]
    assert normalized[0]["a_identity_reference_path"] == str(sources[2])
    assert normalized[0]["text"] == "The normalized transcript."
    assert normalized[0]["conversion"] == "same"
    assert "a" not in rows[0]  # Do not mutate the user's original manifest rows.
    sources[0].with_suffix(".normalized.txt").write_text("")
    sources[0].with_suffix(".original.txt").write_text("Original transcript.")
    assert normalize_pairs(rows)[0]["text"] == "Original transcript."
    sources[0].with_suffix(".original.txt").unlink()
    with pytest.raises(FileNotFoundError, match="No LibriTTS transcript"):
        normalize_pairs(rows)
    rows[0]["transcript"] = "An explicit transcript."
    assert normalize_pairs(rows)[0]["text"] == "An explicit transcript."


def test_prosody_masks_unvoiced_and_does_not_score_empty_pitch():
    first = np.zeros((8, 4))
    second = first.copy()
    second[:, 0] = 100
    distances = prosody_distances(first, second)
    assert distances["f0"] is None
    assert distances["delta_f0"] is None
    first[2:6, 1] = second[2:6, 1] = 1
    second[2:6, 0] = 2
    assert prosody_distances(first, second)["f0"] == 2
    with pytest.raises(ValueError, match="matching"):
        prosody_distances(first, second[:4])


def test_matrix_uses_AAA_differences_and_masks_insufficient_contrast():
    records = []
    for branch in REQUIRED:
        records.append(
            {
                "pair_id": "pair",
                "branch": branch,
                "text": "same words",
                "wer": 0.2 if branch == "BAA" else 0.1,
                "sim_a": 0.8,
                "sim_b": 0.7 if branch == "AAB" else 0.3,
                "rms": 0.2,
                "clip_fraction": 0,
                "identity_reference_kind": "provided",
                "prosody_to_b": {
                    "f0": 0.2 if branch == "ABA" else 1.0,
                    "vuv": 0.1,
                    "energy": 0.2,
                    "delta_f0": None,
                },
            }
        )
    interventions = intervention_rows(records)
    matrix = matrix_summary(interventions)
    assert len(matrix) == 9
    scores = {(row["branch"], row["attribute"]): row["mean"] for row in matrix}
    assert scores["BAA", "content"] == pytest.approx(10)
    assert scores["ABA", "prosody"] == pytest.approx(0.8)
    assert scores["AAB", "speaker"] == pytest.approx(0.4)
    next(row for row in records if row["branch"] == "AAA")["prosody_to_b"]["f0"] = 0.01
    assert all(row["prosody"] is None for row in intervention_rows(records))
    examples = [{"pair_id": str(i), "content": float(i)} for i in range(9)]
    assert [row["pair_id"] for row in select_examples(examples, "content", 3)] == ["0", "4", "8"]


def make_fixture(tmp_path):
    config = ExperimentConfig(
        model=replace(
            tiny_config(),
            encoder_rates=(2, 4, 5, 8),
            decoder_rates=(8, 5, 4, 2),
            decoder_channels=(32, 24, 16, 8),
        )
    )
    codec = LargeStreamingCodec(config.model)
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    state = {f"codec.{name}": value for name, value in codec.state_dict().items()}
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"config": config.to_dict(), "step": 123, "model": state, "ema": {"shadow": state}},
        checkpoint,
    )
    views = {}
    for name, frames, frequency in (("a", 40, 120), ("b", 48, 180)):
        path = tmp_path / f"{name}.wav"
        times = torch.arange(frames * 320) / 16000
        torchaudio.save(str(path), (0.2 * torch.sin(2 * torch.pi * frequency * times))[None], 16000)
        hidden = tmp_path / f"{name}.pt"
        torch.save(torch.randn(frames, config.model.student_dim), hidden)
        views[name] = {"audio_path": str(path), "student_hidden_path": str(hidden)}
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        json.dumps({"pair_id": "pair1", "donor_method": "seed-vc", "text": "hello world", **views})
        + "\n"
    )
    return argparse.Namespace(
        manifest=manifest,
        config=config_path,
        checkpoint=checkpoint,
        weights="ema",
        student_checkpoint=None,
        device="cpu",
        max_pairs=None,
        output_dir=tmp_path / "generated",
    )


def test_real_codec_generate_and_waveform_report(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    args = make_fixture(tmp_path)
    generate(args)
    records = [
        json.loads(line) for line in (args.output_dir / "branches.jsonl").read_text().splitlines()
    ]
    assert len(records) == 8
    for row in records:
        waveform, rate = torchaudio.load(row["audio_path"])
        assert rate == 16000
        assert waveform.shape[-1] == (48 if row["branch"] in ("B", "BBB") else 40) * 320
    # Only the expensive external ASR/SV models are stubbed; use real Praat extraction,
    # real generated WAVs, metric aggregation, HTML generation and copied audio.
    speaker = types.ModuleType("sim.evaluate")
    speaker.load_base_plus_model = lambda *args: (None, None, "cpu")

    def embed(paths, processor, model, device, batch_size, backend):
        assert backend == "wavlm-base-plus-sv"
        return [np.array([1.0, float(index + 1)]) for index in range(len(paths))]

    speaker.embed_paths = embed
    speaker.cosine_similarity = lambda a, b: float(
        np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    )
    asr = types.ModuleType("wer.evaluate")
    asr.load_transcriber = lambda *args: lambda path: "hello world"
    text = types.ModuleType("common.text")
    text.word_errors = lambda ref, hyp: {"wer": 0.0, "reference_words": 2, "errors": 0}
    for name, module in (("sim.evaluate", speaker), ("wer.evaluate", asr), ("common.text", text)):
        monkeypatch.setitem(__import__("sys").modules, name, module)
    evaluate(
        argparse.Namespace(
            manifest=args.output_dir / "branches.jsonl",
            evaluator_dir=tmp_path,
            models_dir=tmp_path,
            device="cpu",
            compute_type="float32",
            batch_size=2,
            examples_per_cell=3,
            output_dir=tmp_path / "report",
        )
    )
    report = tmp_path / "report"
    matrix = json.loads((report / "matrix.json").read_text())
    assert len(matrix) == 9
    assert all(cell["mean"] == 0 for cell in matrix if cell["attribute"] == "content")
    html = (report / "report.html").read_text()
    assert html.count("<td ") == 9
    assert html.count("<audio controls") == 45
    assert len(list((report / "audio").glob("*.wav"))) == 7
    assert "same-text fallback" in html
    with pytest.raises(FileExistsError):
        generate(args)


def test_flat_seed_export_generates_with_on_the_fly_student(tmp_path, monkeypatch):
    args = make_fixture(tmp_path)
    cached = json.loads(args.manifest.read_text())
    source = cached["a"]["audio_path"]
    Path(source).with_suffix(".normalized.txt").write_text("hello world")
    row = {
        "index": 1,
        "source_speaker": "8455",
        "target_speaker": "4077",
        "conversion": "same",
        "source_audio_path": source,
        "converted_audio_path": cached["b"]["audio_path"],
        "target_reference_audio_path": cached["b"]["audio_path"],
    }
    args.manifest.write_text(json.dumps(row) + "\n")
    args.student_checkpoint = tmp_path / "student.pt"

    class Student:
        def __call__(self, waveform, lengths):
            frames = waveform.shape[-1] // 320
            return {
                "hidden_states": torch.zeros(1, frames, 16),
                "output_lengths": torch.tensor([frames]),
            }

    monkeypatch.setattr("xvc2_codec.swap_probe.load_student", lambda *args: Student())
    generate(args)
    normalized = json.loads((args.output_dir / "pairs.jsonl").read_text())
    assert normalized["text"] == "hello world"
    records = [
        json.loads(line) for line in (args.output_dir / "branches.jsonl").read_text().splitlines()
    ]
    assert len(records) == 8
    assert all(
        record["pair_id"] == "seedvc_001"
        and record["source_speaker"] == "8455"
        and record["conversion"] == "same"
        for record in records
    )


def test_checkpoint_selects_requested_weights(tmp_path):
    args = make_fixture(tmp_path)
    payload = torch.load(args.checkpoint, weights_only=False)
    payload["ema"]["shadow"] = dict(payload["ema"]["shadow"])
    name = next(iter(payload["ema"]["shadow"]))
    payload["ema"]["shadow"][name] = torch.full_like(payload["model"][name], 3)
    torch.save(payload, args.checkpoint)
    model, step = load_codec(args.config, args.checkpoint, "model", torch.device("cpu"))
    ema, _ = load_codec(args.config, args.checkpoint, "ema", torch.device("cpu"))
    assert step == 123
    assert torch.all(ema.state_dict()[name.removeprefix("codec.")] == 3)
    assert not torch.equal(
        model.state_dict()[name.removeprefix("codec.")],
        ema.state_dict()[name.removeprefix("codec.")],
    )
