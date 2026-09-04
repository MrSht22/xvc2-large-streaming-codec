import torch

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
