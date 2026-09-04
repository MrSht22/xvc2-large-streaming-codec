from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


@dataclass(frozen=True)
class CodecConfig:
    student_dim: int = 768
    vocab_size: int = 40
    keep_trunk_dim: int = 512
    inv_dim: int = 128
    dyn_dim: int = 32
    edit_dim: int = 96
    encoder_channels: int = 48
    encoder_rates: tuple[int, ...] = (2, 4, 5, 8)
    inv_projection_dim: int = 128
    dyn_projection_dim: int = 64
    edit_projection_dim: int = 128
    fusion_channels: int = 640
    fusion_kernel_size: int = 7
    fusion_dilations: tuple[int, ...] = (1, 3, 9, 1)
    decoder_rates: tuple[int, ...] = (8, 5, 4, 2)
    decoder_channels: tuple[int, ...] = (768, 512, 384, 256)
    residual_kernels: tuple[int, ...] = (3, 7, 11)
    residual_dilations: tuple[int, ...] = (1, 3, 5)
    dropout: float = 0.1

    def __post_init__(self) -> None:
        scalar_names = (
            "student_dim",
            "vocab_size",
            "keep_trunk_dim",
            "inv_dim",
            "dyn_dim",
            "edit_dim",
            "encoder_channels",
            "inv_projection_dim",
            "dyn_projection_dim",
            "edit_projection_dim",
            "fusion_channels",
            "fusion_kernel_size",
        )
        if any(getattr(self, name) <= 0 for name in scalar_names):
            raise ValueError("Codec dimensions must be positive")
        if len(self.decoder_rates) != len(self.decoder_channels):
            raise ValueError("decoder_rates and decoder_channels must have equal length")
        if not self.encoder_rates or _product(self.encoder_rates) != _product(self.decoder_rates):
            raise ValueError("Encoder and decoder rates must have the same non-zero product")
        sequences = (
            self.encoder_rates,
            self.fusion_dilations,
            self.decoder_rates,
            self.decoder_channels,
            self.residual_kernels,
            self.residual_dilations,
        )
        if any(value <= 0 for values in sequences for value in values):
            raise ValueError("Rates, channels, kernels, and dilations must be positive")
        if len(self.residual_kernels) != len(self.residual_dilations):
            raise ValueError("residual_kernels and residual_dilations must match")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def hop_length(self) -> int:
        return _product(self.encoder_rates)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CodecConfig":
        values = dict(values)
        for name in (
            "encoder_rates",
            "fusion_dilations",
            "decoder_rates",
            "decoder_channels",
            "residual_kernels",
            "residual_dilations",
        ):
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class DiscriminatorConfig:
    fft_sizes: tuple[int, ...] = (256, 512, 1024)
    base_channels: int = 16
    maximum_channels: int = 128

    def __post_init__(self) -> None:
        if not self.fft_sizes or any(value < 8 or value % 4 for value in self.fft_sizes):
            raise ValueError("fft_sizes must be divisible by four and at least eight")
        if min(self.base_channels, self.maximum_channels) <= 0:
            raise ValueError("Discriminator channels must be positive")


@dataclass(frozen=True)
class ScheduleConfig:
    reconstruction_end: int = 10_000
    gan_ramp_end: int = 30_000
    sa_ramp_end: int = 60_000
    max_steps: int = 300_000

    def __post_init__(self) -> None:
        if not 0 < self.reconstruction_end < self.gan_ramp_end < self.sa_ramp_end <= self.max_steps:
            raise ValueError("Expected reconstruction < GAN < SA <= max_steps")


@dataclass(frozen=True)
class LossConfig:
    reconstruction: float = 1.0
    adversarial: float = 0.1
    feature_matching: float = 1.0
    edit_style: float = 0.5
    sa_inv: float = 1.0
    sa_dyn: float = 0.05
    phone_anchor: float = 0.25
    dyn_anchor: float = 0.25
    prosody_anchor: float = 0.1

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("Loss weights cannot be negative")


@dataclass(frozen=True)
class TrainingConfig:
    generator_learning_rate: float = 1e-4
    discriminator_learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    amp: str = "bf16"
    ema_decay: float = 0.999
    seed: int = 1
    log_interval: int = 20
    save_interval: int = 10_000

    def __post_init__(self) -> None:
        if (
            min(
                self.generator_learning_rate,
                self.discriminator_learning_rate,
                self.gradient_clip,
            )
            <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid training scalar")
        if self.amp not in {"bf16", "fp16", "none"}:
            raise ValueError("amp must be bf16, fp16, or none")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    model: CodecConfig = field(default_factory=CodecConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path) -> ExperimentConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or values.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    discriminator = dict(values.get("discriminator", {}))
    if "fft_sizes" in discriminator:
        discriminator["fft_sizes"] = tuple(discriminator["fft_sizes"])
    return ExperimentConfig(
        model=CodecConfig.from_dict(values["model"]),
        discriminator=DiscriminatorConfig(**discriminator),
        schedule=ScheduleConfig(**values.get("schedule", {})),
        loss=LossConfig(**values.get("loss", {})),
        training=TrainingConfig(**values.get("training", {})),
    )
