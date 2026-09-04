from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class ReconstructionLoss:
    def __init__(
        self, sample_rate: int = 16_000, fft_sizes: tuple[int, ...] = (256, 512, 1024)
    ) -> None:
        self.sample_rate = sample_rate
        self.fft_sizes = fft_sizes
        self.windows: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.mel_filters: dict[torch.device, torch.Tensor] = {}

    def _magnitude(self, waveform: torch.Tensor, fft_size: int) -> torch.Tensor:
        if waveform.numel() < fft_size:
            waveform = F.pad(waveform, (0, fft_size - waveform.numel()))
        key = (fft_size, waveform.device)
        if key not in self.windows:
            self.windows[key] = torch.hann_window(fft_size, device=waveform.device)
        return (
            torch.stft(
                waveform.float(),
                fft_size,
                hop_length=fft_size // 4,
                window=self.windows[key],
                center=False,
                return_complex=True,
            )
            .abs()
            .clamp_min(1e-5)
        )

    def _mel_filter(
        self, device: torch.device, fft_size: int = 1024, mels: int = 80
    ) -> torch.Tensor:
        if device in self.mel_filters:
            return self.mel_filters[device]
        frequencies = torch.linspace(0, self.sample_rate / 2, fft_size // 2 + 1, device=device)
        mel_max = 2595 * math.log10(1 + self.sample_rate / 1400)
        mel_points = torch.linspace(0, mel_max, mels + 2, device=device)
        hz = 700 * (torch.pow(10.0, mel_points / 2595) - 1)
        lower = (frequencies[None] - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None]).clamp_min(
            1e-8
        )
        upper = (hz[2:, None] - frequencies[None]) / (hz[2:, None] - hz[1:-1, None]).clamp_min(1e-8)
        self.mel_filters[device] = torch.minimum(lower, upper).clamp_min(0)
        return self.mel_filters[device]

    def __call__(
        self, predicted: torch.Tensor, target: torch.Tensor, sample_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        wave_losses, log_losses, convergence_losses, mel_losses = [], [], [], []
        for index, length in enumerate(sample_lengths.tolist()):
            predicted_item = predicted[index, 0, :length]
            target_item = target[index, 0, :length]
            wave_losses.append(F.l1_loss(predicted_item, target_item))
            current_log, current_convergence = [], []
            for size in self.fft_sizes:
                predicted_magnitude = self._magnitude(predicted_item, size)
                target_magnitude = self._magnitude(target_item, size)
                current_log.append(F.l1_loss(predicted_magnitude.log(), target_magnitude.log()))
                current_convergence.append(
                    torch.linalg.vector_norm(predicted_magnitude - target_magnitude)
                    / torch.linalg.vector_norm(target_magnitude).clamp_min(1e-5)
                )
            log_losses.append(torch.stack(current_log).mean())
            convergence_losses.append(torch.stack(current_convergence).mean())
            mel_filter = self._mel_filter(predicted.device)
            predicted_mel = mel_filter @ self._magnitude(predicted_item, 1024)
            target_mel = mel_filter @ self._magnitude(target_item, 1024)
            mel_losses.append(F.l1_loss(predicted_mel.log(), target_mel.log()))
        metrics = {
            "waveform_l1": torch.stack(wave_losses).mean(),
            "log_spectral": torch.stack(log_losses).mean(),
            "spectral_convergence": torch.stack(convergence_losses).mean(),
            "log_mel": torch.stack(mel_losses).mean(),
        }
        total = (
            metrics["log_spectral"]
            + metrics["spectral_convergence"]
            + metrics["log_mel"]
            + 0.1 * metrics["waveform_l1"]
        )
        return total, metrics


def masked_smooth_l1(
    predicted: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    frames = min(predicted.shape[1], target.shape[1])
    mask = torch.arange(frames, device=predicted.device)[None] < lengths[:, None].clamp_max(frames)
    values = F.smooth_l1_loss(
        predicted[:, :frames].float(), target[:, :frames].float(), reduction="none"
    ).mean(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def trajectory_correlation_loss(
    predicted: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    losses = []
    for index, length in enumerate(lengths.tolist()):
        length = min(int(length), predicted.shape[1], target.shape[1])
        first = predicted[index, :length].float().flatten()
        second = target[index, :length].float().flatten()
        first = first - first.mean()
        second = second - second.mean()
        cosine = F.cosine_similarity(first[None], second[None]).squeeze(0)
        losses.append(1 - cosine)
    return torch.stack(losses).mean()
