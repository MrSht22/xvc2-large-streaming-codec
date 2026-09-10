import json
import wave
from pathlib import Path

import torch
import torchaudio

from xvc2_codec.audit import audit_manifests
from xvc2_codec.config import LossConfig, ScheduleConfig
from xvc2_codec.data import PairDataset, TrainingStepDataset, _load_audio_crop
from xvc2_codec.losses import ReconstructionLoss
from xvc2_codec.model import LargeStreamingCodec
from xvc2_codec.schedule import weights_at
from xvc2_codec.smoke import tiny_config
from xvc2_codec.train import discriminator_batch, forward_views, warmup_learning_rate


class StubSourceDataset:
    def __len__(self) -> int:
        return 32

    def load(self, index: int, crop_seed: int) -> dict[str, object]:
        value = float(index * 1000 + crop_seed % 997)
        return {
            "waveform": torch.full((1, 16), value),
            "student_hidden": torch.full((1, 2), value),
            "speaker_target": torch.tensor([value]),
            "frames": 1,
        }


class StubPairDataset(StubSourceDataset):
    def load(self, index: int, crop_seed: int) -> dict[str, dict[str, object]]:
        source = super().load(index, crop_seed)
        return {"source": source, "sa": {**source, "waveform": source["waveform"] + 1}}


class CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, waveform: torch.Tensor, student_hidden: torch.Tensor):
        self.calls += 1
        return {"reconstruction": waveform + student_hidden[:, :1, :1]}


def test_codec_forward_backward() -> None:
    config = tiny_config()
    model = LargeStreamingCodec(config)
    frames = 8
    waveform = torch.randn(2, 1, frames * config.hop_length)
    hidden = torch.randn(2, frames, config.student_dim)
    result = model(waveform, hidden)
    result["reconstruction"].square().mean().backward()
    assert result["z_inv"].shape == (2, frames, config.inv_dim)
    assert result["z_dyn"].shape == (2, frames, config.dyn_dim)


def test_loss_schedule() -> None:
    schedule = ScheduleConfig()
    loss = LossConfig()
    assert weights_at(0, schedule, loss).adversarial == 0
    assert weights_at(10_000, schedule, loss).sa_inv == 0
    assert weights_at(30_000, schedule, loss).adversarial == loss.adversarial
    assert weights_at(60_000, schedule, loss).sa_inv == loss.sa_inv


def test_learning_rate_warmup() -> None:
    assert warmup_learning_rate(1e-4, 0, 1_000) == 0
    assert warmup_learning_rate(1e-4, 500, 1_000) == 5e-5
    assert warmup_learning_rate(1e-4, 2_000, 1_000) == 1e-4
    assert warmup_learning_rate(1e-4, 1, 0) == 1e-4


def test_discriminator_batch_contains_both_pair_views() -> None:
    batches = {
        "source": {"waveform": torch.zeros(2, 1, 16)},
        "sa": {"waveform": torch.ones(2, 1, 16)},
    }
    outputs = {
        "source": {"reconstruction": torch.full((2, 1, 16), 2.0)},
        "sa": {"reconstruction": torch.full((2, 1, 16), 3.0)},
    }
    real, fake = discriminator_batch(outputs, batches, ("source", "sa"))
    assert real.shape == fake.shape == (4, 1, 16)
    torch.testing.assert_close(real[:2], batches["source"]["waveform"])
    torch.testing.assert_close(real[2:], batches["sa"]["waveform"])
    torch.testing.assert_close(fake[:2], outputs["source"]["reconstruction"])
    torch.testing.assert_close(fake[2:], outputs["sa"]["reconstruction"])


def test_pair_views_share_one_model_forward() -> None:
    model = CountingModel()
    batches = {
        "source": {
            "waveform": torch.zeros(2, 1, 4),
            "student_hidden": torch.zeros(2, 1, 1),
        },
        "sa": {
            "waveform": torch.ones(2, 1, 4),
            "student_hidden": torch.zeros(2, 1, 1),
        },
    }
    outputs = forward_views(model, batches, ("source", "sa"))
    assert model.calls == 1
    torch.testing.assert_close(outputs["source"]["reconstruction"], torch.zeros(2, 1, 4))
    torch.testing.assert_close(outputs["sa"]["reconstruction"], torch.ones(2, 1, 4))


def test_reconstruction_loss_ignores_padded_samples() -> None:
    loss = ReconstructionLoss(fft_sizes=(256, 512, 1024))
    target = torch.randn(2, 1, 2048)
    predicted = target.clone()
    predicted[1, :, 1600:] = 100
    total, metrics = loss(predicted, target, torch.tensor([2048, 1600]))
    torch.testing.assert_close(total, torch.zeros_like(total), atol=1e-6, rtol=0)
    for value in metrics.values():
        torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-6, rtol=0)


def test_vectorized_reconstruction_matches_per_item_loss() -> None:
    loss = ReconstructionLoss(fft_sizes=(256, 512, 1024))
    lengths = torch.tensor([2048, 1600])
    target = torch.randn(2, 1, 2048)
    predicted = torch.randn(2, 1, 2048)
    batched_total, batched_metrics = loss(predicted, target, lengths)

    per_item = [
        loss(
            predicted[index : index + 1, :, :length],
            target[index : index + 1, :, :length],
            torch.tensor([length]),
        )
        for index, length in enumerate(lengths.tolist())
    ]
    expected_total = torch.stack([item[0] for item in per_item]).mean()
    expected_metrics = {
        name: torch.stack([item[1][name] for item in per_item]).mean() for name in batched_metrics
    }
    torch.testing.assert_close(batched_total, expected_total)
    for name, value in batched_metrics.items():
        torch.testing.assert_close(value, expected_metrics[name])


def test_training_step_dataset_is_stable_across_resume() -> None:
    options = {
        "source": StubSourceDataset(),
        "pair": StubPairDataset(),
        "end_step": 10,
        "batch_size": 2,
        "rank": 1,
        "world_size": 2,
        "seed": 7,
        "pair_probability": 0.5,
    }
    uninterrupted = TrainingStepDataset(start_step=0, **options)[4]
    resumed = TrainingStepDataset(start_step=4, **options)[0]
    assert uninterrupted["step"] == resumed["step"] == 5
    assert uninterrupted["kind"] == resumed["kind"]
    first = uninterrupted["batch"]
    second = resumed["batch"]
    views = ("source", "sa") if uninterrupted["kind"] == "pair" else (None,)
    for view in views:
        first_view = first if view is None else first[view]
        second_view = second if view is None else second[view]
        for key in first_view:
            if torch.is_tensor(first_view[key]):
                torch.testing.assert_close(first_view[key], second_view[key])


def test_non_16khz_crop_matches_full_resample(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    source = torch.linspace(-0.5, 0.5, 8000)[None]
    torchaudio.save(str(audio), source, 8000)
    full, sample_rate = torchaudio.load(str(audio))
    resampled = torchaudio.functional.resample(full, sample_rate, 16_000)
    cropped = _load_audio_crop({"audio_path": str(audio)}, 321, 2048)
    torch.testing.assert_close(cropped, resampled[:, 321 : 321 + 2048])


def test_manifest_audit_accepts_aligned_cache(tmp_path: Path, capsys) -> None:
    config = tiny_config()
    audio = tmp_path / "audio.wav"
    frames = 8
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * frames * config.hop_length)
    hidden = tmp_path / "hidden.pt"
    speaker = tmp_path / "speaker.pt"
    torch.save(torch.zeros(frames, config.student_dim), hidden)
    torch.save(torch.zeros(6), speaker)
    view = {
        "audio_path": str(audio),
        "student_hidden_path": str(hidden),
        "speaker_target_path": str(speaker),
    }
    source = tmp_path / "source.jsonl"
    pair = tmp_path / "pair.jsonl"
    source.write_text(json.dumps(view) + "\n")
    pair.write_text(json.dumps({"source": view, "sa": view}) + "\n")
    report = audit_manifests(source, pair, config, speaker_target_dim=6, progress_every=1)
    assert report["status"] == "PASS"
    progress = capsys.readouterr().out
    assert "audit_progress stage=source items=1/1 failures=0" in progress
    assert "audit_progress stage=pair items=1/1 failures=0" in progress


def test_pair_dataset_uses_shared_crop(tmp_path: Path) -> None:
    config = tiny_config()
    frames = 20
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * frames * config.hop_length)
    hidden = tmp_path / "hidden.pt"
    speaker = tmp_path / "speaker.pt"
    values = torch.arange(frames).float()[:, None].expand(frames, config.student_dim)
    torch.save(values, hidden)
    torch.save(torch.zeros(6), speaker)
    view = {
        "audio_path": str(audio),
        "student_hidden_path": str(hidden),
        "speaker_target_path": str(speaker),
    }
    dataset = PairDataset([{"source": view, "sa": view}], config.hop_length, 8)
    item = dataset.load(0, crop_seed=123)
    torch.testing.assert_close(item["source"]["student_hidden"], item["sa"]["student_hidden"])
    assert item["source"]["frames"] == item["sa"]["frames"] == 8
