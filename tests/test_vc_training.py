import json
import sys
from pathlib import Path

import pytest
import torch
import torchaudio
import yaml
from test_vc_streaming import codec_config, latents, make_pipeline
from torch.utils.data import DistributedSampler

from xvc2_codec.losses import ReconstructionLoss
from xvc2_codec.vc import StreamingLatentConverter
from xvc2_codec.vc_condition import ReferenceFrontend
from xvc2_codec.vc_data import VCPairDataConfig, VCPairDataset
from xvc2_codec.vc_infer import convert_stream
from xvc2_codec.vc_train import (
    CursorDistributedSampler,
    SeededVCPairDataset,
    build_optimizer,
    build_training_loader,
    compute_losses,
    move_vc_batch,
    restore_vc,
    valid_reconstruction_loss,
    vc_state,
)

WEIGHTS = {"dyn": 1.0, "edit": 1.0, "paired_reconstruction": 1.0, "natural_reconstruction": 1.0}
TRAINING = {"converter_learning_rate": 1e-3, "decoder_learning_rate": 2e-5, "weight_decay": 0.0}


class TinySpeaker(torch.nn.Module):
    embedding_dim = 4

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 4)

    def forward(self, audio):
        return self.projection(torch.stack((audio.mean(-1), audio.square().mean(-1)), dim=-1))


def paired_batch():
    return {
        name: {"waveform": torch.randn(2, 1, 3840), "frames": torch.tensor([12, 5])}
        for name in ("source", "target", "natural")
    }


def test_decoder_stage_updates_only_converter_and_decoder_and_preserves_teacher():
    vc = make_pipeline().train()
    batch, condition = paired_batch(), torch.randn(2, 4)
    frozen = {name: value.clone() for name, value in vc.codec.state_dict().items()}
    teacher = vc._encode(batch["target"]["waveform"], batch["target"]["frames"] * 320)
    optimizer = build_optimizer(vc, TRAINING)
    paired = vc.forward_pair(batch, condition)
    compute_losses(paired, batch, ReconstructionLoss(), WEIGHTS)[0].backward()
    optimizer.step()
    for name, value in vc.codec.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    vc.set_training_stage("decoder")
    optimizer = build_optimizer(vc, TRAINING)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 2e-5]
    assert vc.codec.decoder.training
    assert not vc.codec.fusion.training and not vc.codec.keep_head.training
    optimizer.zero_grad(set_to_none=True)
    paired = vc.forward_pair(batch, condition, reconstruct_natural=True)
    # Ordinary natural reconstruction trains Decoder directly, bypassing Converter.
    paired["natural_reconstruction"].square().mean().backward()
    assert all(p.grad is None for p in vc.converter.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in vc.codec.decoder.parameters())
    optimizer.zero_grad(set_to_none=True)
    paired = vc.forward_pair(batch, condition, reconstruct_natural=True)
    compute_losses(paired, batch, ReconstructionLoss(), WEIGHTS)[0].backward()
    optimizer.step()
    assert any(
        not torch.equal(value, frozen[name])
        for name, value in vc.codec.state_dict().items()
        if name.startswith("decoder.")
    )
    for name, value in vc.codec.state_dict().items():
        if not name.startswith("decoder."):
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    after = vc._encode(batch["target"]["waveform"], batch["target"]["frames"] * 320)
    torch.testing.assert_close(teacher["z_dyn"], after["z_dyn"], rtol=0, atol=0)
    vc.set_training_stage("converter")
    assert all(not p.requires_grad for p in vc.codec.parameters())


def test_reference_attention_is_static_causal_padding_safe_and_differentiable():
    model = StreamingLatentConverter(
        codec_config(), condition_dim=4, channels=8, dilations=(1, 2), reference_mels=80
    ).eval()
    torch.nn.init.normal_(model.output.weight, std=0.1)
    inv, dyn, edit, speaker = latents()
    mel = torch.randn(1, 80, 15)
    condition = model.prepare_condition(speaker, mel, torch.tensor([15]))
    expected = model(inv, dyn, edit, condition)
    state = model.init_streaming_state(condition)
    parts = []
    for start, end in ((0, 3), (3, 8), (8, 19)):
        result, state = model.forward_chunk(
            inv[:, start:end], dyn[:, start:end], edit[:, :, start:end], state, end == 19
        )
        parts.append(result["z_edit"])
    torch.testing.assert_close(torch.cat(parts, -1), expected["z_edit"], rtol=1e-5, atol=1e-6)
    # Preparing the reference has no source input; changing future source cannot change a prefix.
    changed_edit = edit.clone()
    changed_edit[:, :, 8:] += 100
    changed = model(inv, dyn, changed_edit, condition)
    torch.testing.assert_close(expected["z_edit"][:, :, :8], changed["z_edit"][:, :, :8])
    padded = model.prepare_condition(
        speaker, torch.nn.functional.pad(mel, (0, 10)), torch.tensor([15])
    )
    torch.testing.assert_close(model(inv, dyn, edit, padded)["z_edit"], expected["z_edit"])
    expected["z_edit"].square().mean().backward()
    assert model.reference.encoder[0].weight.grad.abs().sum() > 0
    # Per-stream condition/cache owns its tensors.
    condition.keys.zero_()
    assert state.condition.keys.abs().sum() > 0


def test_adapted_decoder_and_mel_reference_keep_raw_stream_equivalence():
    vc = make_pipeline()
    vc.converter = StreamingLatentConverter(
        codec_config(), condition_dim=4, channels=8, dilations=(1, 2), reference_mels=80
    )
    torch.nn.init.normal_(vc.converter.output.weight, std=0.1)
    vc.set_training_stage("decoder")
    vc.eval()
    waveform = torch.randn(1, 1, 3911)
    condition = vc.converter.prepare_condition(
        torch.randn(1, 4), torch.randn(1, 80, 11), torch.tensor([11])
    )
    with torch.no_grad():
        expected = vc(waveform, torch.tensor([3911]), condition)["reconstruction"]
        streamed = convert_stream(vc, waveform, condition, 317)
    torch.testing.assert_close(streamed, expected, rtol=1e-4, atol=2e-6)
    restored = make_pipeline()
    restored.converter = StreamingLatentConverter(
        codec_config(), condition_dim=4, channels=8, dilations=(1, 2), reference_mels=80
    )
    restore_vc(restored, vc_state(vc))
    restored.eval()
    other_condition = restored.converter.prepare_condition(
        condition.speaker, torch.randn(1, 80, 11), torch.tensor([11])
    )
    # State restore includes the adapted Decoder and the learned reference encoder.
    for name, value in restored.codec.decoder.state_dict().items():
        torch.testing.assert_close(value, vc.codec.decoder.state_dict()[name])
    assert convert_stream(restored, waveform, other_condition, 641).shape == streamed.shape


def test_reference_frontend_excludes_batch_padding_and_freezes_speaker():
    frontend = ReferenceFrontend(TinySpeaker()).train()
    first, second = torch.randn(1, 1, 8000), torch.randn(1, 1, 9600)
    packed = torch.cat((torch.nn.functional.pad(first, (0, 1600)), second))
    embedding, mel, lengths = frontend(packed, torch.tensor([8000, 9600]))
    alone, alone_mel, _ = frontend(first, torch.tensor([8000]))
    torch.testing.assert_close(embedding[:1], alone)
    torch.testing.assert_close(mel[:1, :, : lengths[0]], alone_mel)
    assert not frontend.speaker_encoder.training and not embedding.requires_grad
    assert all(not p.requires_grad for p in frontend.speaker_encoder.parameters())
    with pytest.raises(ValueError, match="0.5s"):
        frontend(first, torch.tensor([7999]))
    with pytest.raises(ValueError, match="silent"):
        frontend(torch.zeros_like(first), torch.tensor([8000]))


def test_reference_frontend_batches_equal_lengths_without_reordering_or_reading_padding():
    class CountingSpeaker(TinySpeaker):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, audio):
            self.batch_sizes.append(audio.shape[0])
            return super().forward(audio), audio.mean(-1)

    frontend = ReferenceFrontend(CountingSpeaker())
    lengths = torch.tensor([9600, 8000, 9600, 12800])
    audio = torch.randn(4, 1, 12800)
    for index, length in enumerate(lengths):
        audio[index, :, length:] = float("nan")
    embeddings, mels, mel_lengths = frontend(audio, lengths)
    assert frontend.speaker_encoder.batch_sizes == [2, 1, 1]
    for index, length in enumerate(lengths):
        embedding, mel, frames = frontend(
            audio[index : index + 1, :, :length], lengths[index : index + 1]
        )
        torch.testing.assert_close(embeddings[index : index + 1], embedding)
        torch.testing.assert_close(mels[index : index + 1, :, : frames[0]], mel)
        assert mel_lengths[index] == frames[0]
    audio[2, :, : lengths[2]] = 0
    with pytest.raises(ValueError, match="silent"):
        frontend(audio, lengths)


def test_short_audio_reconstruction_loss_cannot_read_batch_padding():
    lengths = torch.tensor([320, 1600])
    predicted, target = torch.randn(2, 1, 1600), torch.randn(2, 1, 1600)
    first = valid_reconstruction_loss(predicted, target, lengths, ReconstructionLoss())
    predicted[0, :, 320:] += 100
    target[0, :, 320:] -= 100
    second = valid_reconstruction_loss(predicted, target, lengths, ReconstructionLoss())
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def make_training_data(tmp_path):
    reference = tmp_path / "reference.wav"
    torchaudio.save(str(reference), torch.randn(1, 9600) * 0.05, 16_000)
    manifests = []
    for split in ("train", "val"):
        path = tmp_path / f"{split}.wav"
        torchaudio.save(str(path), torch.randn(1, 3200) * 0.05, 16_000)
        manifest = tmp_path / f"{split}.jsonl"
        rows = [
            {
                "index": index,
                "source_utt": f"{split}_source",
                "target_utt": f"{split}_target",
                "source_wav_path": str(path),
                "target_wav_path": str(path),
                "source_reference_wav_path": str(reference),
                "target_reference_wav_path": str(reference),
            }
            for index in range(3)
        ]
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifests.append(manifest)
    return manifests


def test_references_follow_roles_and_natural_reconstruction_keeps_original_target(
    tmp_path, monkeypatch
):
    train, _ = make_training_data(tmp_path)
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    generated = tmp_path / "generated.wav"
    torchaudio.save(str(generated), torch.ones(1, 3200) * 0.2, 16_000)
    reverse_ref = tmp_path / "generated_identity.wav"
    torchaudio.save(str(reverse_ref), torch.ones(1, 9600) * 0.3, 16_000)
    rows[0].update(source_wav_path=str(generated), source_reference_wav_path=str(reverse_ref))
    train.write_text(json.dumps(rows[0]) + "\n")
    config = VCPairDataConfig(segment_duration=0.12, volume_normalize=False, highpass_cutoff_freq=0)
    for draw, role in ((0.1, "reconstruction"), (0.3, "reversed"), (0.9, "standard")):
        monkeypatch.setattr(torch, "rand", lambda *args, draw=draw: torch.tensor(draw))
        item = VCPairDataset(train, config)[0]
        assert item["role_assignment_mode"] == role
        expected_path = (
            str(reverse_ref) if role == "reversed" else rows[0]["target_reference_wav_path"]
        )
        assert item["reference_wav_path"] == expected_path
        assert item["reference_kind"] == "independent"
        natural, _ = torchaudio.load(rows[0]["target_wav_path"])
        start = item["start_frame"] * 320
        torch.testing.assert_close(item["natural"]["waveform"], natural[:, start : start + 1920])


def test_training_cli_checkpoint_resume_and_decoder_transition(tmp_path, monkeypatch):
    from xvc2_codec import vc_train

    train, val = make_training_data(tmp_path)
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/vc_train.yaml").read_text())
    config["data"].update(
        segment_duration=0.12, max_val_duration=0.2, volume_normalize=False, highpass_cutoff_freq=0
    )
    config["converter"].update(condition_dim=4, channels=8, dilations=[1, 2])
    config["training"].update(amp="none", log_interval=1, save_interval=2, val_interval=2)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(vc_train, "load_codec", lambda *args: (make_pipeline().codec, 300000))
    monkeypatch.setattr(vc_train, "load_student", lambda *args: make_pipeline().student)
    monkeypatch.setattr(vc_train, "load_speaker_encoder", lambda *args: TinySpeaker())
    output = tmp_path / "converter"
    base_args = [
        "vc_train",
        "--config",
        str(config_path),
        "--codec-config",
        "codec.yaml",
        "--codec-checkpoint",
        "codec.pt",
        "--student-checkpoint",
        "student.pt",
        "--speaker-model-dir",
        "speaker",
        "--xvc-root",
        "X-VC",
        "--train-manifest",
        str(train),
        "--val-manifest",
        str(val),
        "--device",
        "cpu",
        "--batch-size",
        "1",
        "--audio-examples",
        "1",
    ]

    def run(extra):
        monkeypatch.setattr(sys, "argv", base_args + extra)
        vc_train.main()

    run(["--output-dir", str(output), "--steps", "2"])
    checkpoint = output / "step-000002.pt"
    assert checkpoint.is_file()
    assert (output / "validation-000002/000-converted.wav").is_file()
    run(
        [
            "--output-dir",
            str(output),
            "--resume",
            str(checkpoint),
            "--steps",
            "2",
            "--num-workers",
            "1",
            "--prefetch-factor",
            "3",
        ]
    )
    resumed = torch.load(output / "step-000004.pt", weights_only=False)
    uninterrupted = tmp_path / "uninterrupted"
    run(["--output-dir", str(uninterrupted), "--steps", "4"])
    direct = torch.load(uninterrupted / "step-000004.pt", weights_only=False)
    for name, value in resumed["vc"]["converter"].items():
        torch.testing.assert_close(value, direct["vc"]["converter"][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="batch/world size differs"):
        run(
            [
                "--output-dir",
                str(tmp_path / "rejected"),
                "--resume",
                str(checkpoint),
                "--batch-size",
                "2",
                "--steps",
                "1",
            ]
        )
    # Old checkpoints have only a batch cursor; convert it using the old batch size.
    legacy = torch.load(checkpoint, weights_only=False)
    legacy.pop("sample_offset")
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    consumed = []
    original_getitem = SeededVCPairDataset.__getitem__

    def record_item(dataset, index):
        consumed.append((dataset.epoch, index))
        return original_getitem(dataset, index)

    monkeypatch.setattr(SeededVCPairDataset, "__getitem__", record_item)
    legacy_resume = tmp_path / "legacy_resume"
    run(
        [
            "--output-dir",
            str(legacy_resume),
            "--resume",
            str(legacy_path),
            "--steps",
            "2",
        ]
    )
    restored = torch.load(legacy_resume / "step-000004.pt", weights_only=False)
    expected_sampler = DistributedSampler(range(3), num_replicas=1, rank=0, seed=1)
    first_epoch = list(expected_sampler)
    expected_sampler.set_epoch(1)
    assert consumed == [(0, first_epoch[2]), (1, next(iter(expected_sampler)))]
    assert restored["step"] == 4 and restored["sample_offset"] == 1 and restored["epoch"] == 1
    assert all(int(state["step"]) == 4 for state in restored["optimizer"]["state"].values())
    for name, value in restored["vc"]["converter"].items():
        torch.testing.assert_close(value, direct["vc"]["converter"][name], rtol=0, atol=0)
    for name, value in restored["vc"]["decoder"].items():
        torch.testing.assert_close(value, legacy["vc"]["decoder"][name], rtol=0, atol=0)
    consumed.clear()
    run(
        [
            "--output-dir",
            str(legacy_resume),
            "--resume",
            str(legacy_resume / "step-000004.pt"),
            "--steps",
            "1",
        ]
    )
    assert consumed == [(1, list(expected_sampler)[1])]
    again = torch.load(legacy_resume / "step-000005.pt", weights_only=False)
    assert again["step"] == 5 and again["sample_offset"] == 2
    adapted = tmp_path / "decoder"
    run(
        [
            "--output-dir",
            str(adapted),
            "--stage",
            "decoder",
            "--initialize-vc",
            str(checkpoint),
            "--steps",
            "1",
        ]
    )
    payload = torch.load(adapted / "step-000001.pt", weights_only=False)
    assert payload["stage"] == "decoder" and payload["step"] == 1
    assert len(payload["optimizer"]["param_groups"]) == 2
    assert "natural_reconstruction" in json.loads(
        (adapted / "validation-000001/metrics.json").read_text()
    )
    assert all(not key.startswith("student") for key in payload["vc"])


def test_sample_cursor_preserves_each_rank_order_without_replaying_consumed_items():
    for rank in (0, 1):
        sampler = CursorDistributedSampler(range(11), num_replicas=2, rank=rank, seed=42)
        for epoch in (0, 1):
            sampler.sample_offset = 0
            sampler.set_epoch(epoch)
            full = list(sampler)
            # Consuming a batch of two starts the next iterator at exactly sample2.
            sampler.sample_offset = 2
            assert list(sampler) == full[2:] and len(sampler) == len(full) - 2
            sampler.sample_offset = len(full)
            assert list(sampler) == [] and len(sampler) == 0


@pytest.mark.parametrize("independent", [True, False])
def test_training_omits_legacy_conditions_without_changing_roles_crops_or_references(
    tmp_path, independent
):
    train, _ = make_training_data(tmp_path)
    if not independent:
        rows = [json.loads(line) for line in train.read_text().splitlines()]
        for row in rows:
            row.pop("source_reference_wav_path")
            row.pop("target_reference_wav_path")
        train.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = VCPairDataConfig(segment_duration=0.12, volume_normalize=False, highpass_cutoff_freq=0)
    original = SeededVCPairDataset(train, config)
    compact = SeededVCPairDataset(train, config, include_pair_conditions=False)
    for epoch in (0, 1):
        original.epoch = compact.epoch = epoch
        full = original.collate([original[index] for index in range(len(original))])
        trimmed = compact.collate([compact[index] for index in range(len(compact))])
        assert not {"source_wav_cond", "target_wav_cond", "condition_frames"} & trimmed.keys()
        for key, value in trimmed.items():
            if isinstance(value, dict):
                for name, tensor in value.items():
                    torch.testing.assert_close(tensor, full[key][name], rtol=0, atol=0)
            elif torch.is_tensor(value):
                torch.testing.assert_close(value, full[key], rtol=0, atol=0)
            else:
                assert value == full[key]


def test_training_transfer_keeps_reference_lengths_and_unused_natural_crop_on_cpu():
    batch = paired_batch()
    batch.update(
        reference_waveform=torch.randn(2, 1, 9600), reference_samples=torch.tensor([9600, 8000])
    )
    converted = move_vc_batch(batch, torch.device("meta"), "converter")
    assert converted["source"]["waveform"].device.type == "meta"
    assert converted["reference_waveform"].device.type == "meta"
    assert converted["reference_samples"] is batch["reference_samples"]
    assert converted["natural"] is batch["natural"]
    adapted = move_vc_batch(batch, torch.device("meta"), "decoder")
    assert adapted["natural"]["waveform"].device.type == "meta"


def test_persistent_workers_observe_epoch_and_match_zero_worker_sampling(tmp_path):
    train, _ = make_training_data(tmp_path)
    config = VCPairDataConfig(segment_duration=0.12, volume_normalize=False, highpass_cutoff_freq=0)

    def sample(workers):
        dataset = SeededVCPairDataset(train, config)
        dataset.seed = 1
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=1)
        loader = build_training_loader(dataset, sampler, 1, workers, 3, torch.device("cpu"))
        assert loader.persistent_workers == bool(workers)
        assert loader.prefetch_factor == (3 if workers else None)
        epochs = []
        for epoch in (0, 1, 2):
            dataset.epoch = epoch
            sampler.set_epoch(epoch)
            epochs.append(
                [
                    (batch["index"], batch["start_frame"], batch["role_assignment_mode"])
                    for batch in loader
                ]
            )
        return epochs

    expected = sample(0)
    assert sample(1) == expected
    assert expected[0] != expected[1]
