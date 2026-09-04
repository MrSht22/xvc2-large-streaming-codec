from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import DiscriminatorConfig


class STFTDiscriminator(torch.nn.Module):
    def __init__(self, fft_size: int, base_channels: int, maximum_channels: int) -> None:
        super().__init__()
        self.fft_size = fft_size
        channels = [
            base_channels,
            min(base_channels * 2, maximum_channels),
            min(base_channels * 4, maximum_channels),
            min(base_channels * 8, maximum_channels),
        ]
        self.blocks = torch.nn.ModuleList(
            torch.nn.Conv2d(inputs, outputs, (3, 5), stride=stride, padding=(1, 2))
            for inputs, outputs, stride in zip(
                (3, *channels[:-1]), channels, ((1, 2), (2, 2), (2, 2), (2, 1))
            )
        )
        self.output = torch.nn.Conv2d(channels[-1], 1, 3, padding=1)
        self.register_buffer("window", torch.hann_window(fft_size), persistent=False)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if waveform.shape[-1] < self.fft_size:
            waveform = F.pad(waveform, (0, self.fft_size - waveform.shape[-1]))
        with torch.autocast(waveform.device.type, enabled=False):
            spectrum = torch.stft(
                waveform[:, 0].float(),
                n_fft=self.fft_size,
                hop_length=self.fft_size // 4,
                window=self.window.float(),
                center=False,
                return_complex=True,
            )
            magnitude = spectrum.abs()
            scale = magnitude.mean((1, 2), keepdim=True).clamp_min(1e-5)
            values = torch.stack(
                (spectrum.real / scale, spectrum.imag / scale, torch.log1p(magnitude / scale)),
                dim=1,
            )
        features = []
        for block in self.blocks:
            values = F.leaky_relu(block(values), 0.1)
            features.append(values)
        return self.output(values), features


class MultiScaleSTFTDiscriminator(torch.nn.Module):
    def __init__(self, config: DiscriminatorConfig) -> None:
        super().__init__()
        self.config = config
        self.scales = torch.nn.ModuleList(
            STFTDiscriminator(size, config.base_channels, config.maximum_channels)
            for size in config.fft_sizes
        )

    def forward(self, waveform: torch.Tensor):
        return [scale(waveform) for scale in self.scales]


def discriminator_loss(real, fake) -> torch.Tensor:
    return torch.stack(
        [
            F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
            for (real_score, _), (fake_score, _) in zip(real, fake)
        ]
    ).mean()


def generator_adversarial_loss(fake) -> torch.Tensor:
    return -torch.stack([score.mean() for score, _ in fake]).mean()


def feature_matching_loss(real, fake) -> torch.Tensor:
    losses = []
    for (_, real_features), (_, fake_features) in zip(real, fake):
        losses.extend(
            F.l1_loss(fake_feature, real_feature.detach())
            for real_feature, fake_feature in zip(real_features, fake_features)
        )
    return torch.stack(losses).mean()
