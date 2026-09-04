import json
import wave
from pathlib import Path

import torch

from xvc2_codec.audit import audit_manifests
from xvc2_codec.config import LossConfig, ScheduleConfig
from xvc2_codec.model import LargeStreamingCodec
from xvc2_codec.schedule import weights_at
from xvc2_codec.smoke import tiny_config


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


def test_manifest_audit_accepts_aligned_cache(tmp_path: Path) -> None:
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
    report = audit_manifests(source, pair, config, speaker_target_dim=6)
    assert report["status"] == "PASS"
