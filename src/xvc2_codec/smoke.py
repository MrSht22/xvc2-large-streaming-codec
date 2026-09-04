from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import CodecConfig, load_config
from .model import LargeStreamingCodec, parameter_breakdown


def tiny_config() -> CodecConfig:
    return CodecConfig(
        student_dim=16,
        vocab_size=10,
        keep_trunk_dim=16,
        inv_dim=8,
        dyn_dim=4,
        edit_dim=8,
        encoder_channels=4,
        encoder_rates=(2, 2),
        inv_projection_dim=8,
        dyn_projection_dim=4,
        edit_projection_dim=8,
        fusion_channels=16,
        fusion_kernel_size=3,
        fusion_dilations=(1, 2),
        decoder_rates=(2, 2),
        decoder_channels=(16, 8),
        residual_kernels=(3, 5),
        residual_dilations=(1, 2),
        dropout=0.0,
    )


def run(config_path: Path | None = None) -> dict[str, int]:
    torch.manual_seed(11)
    config = load_config(config_path).model if config_path else tiny_config()
    model = LargeStreamingCodec(config)
    frames = 12
    waveform = torch.randn(2, 1, frames * config.hop_length)
    hidden = torch.randn(2, frames, config.student_dim)
    outputs = model(waveform, hidden)
    F.l1_loss(outputs["reconstruction"], waveform).backward()
    assert outputs["z_edit"].shape == (2, config.edit_dim, frames)
    counts = parameter_breakdown(model)
    print(json.dumps(counts, sort_keys=True))
    print("large_streaming_codec_smoke=PASS")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
