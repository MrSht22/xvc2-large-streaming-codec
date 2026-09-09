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
        if waveform.shape[-1] < fft_size:
            waveform = F.pad(waveform, (0, fft_size - waveform.shape[-1]))
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
        predicted = predicted[:, 0].float()
        target = target[:, 0].float()
        sample_mask = (
            torch.arange(predicted.shape[-1], device=predicted.device)[None]
            < sample_lengths[:, None]
        )
        waveform_l1 = (
            ((predicted - target).abs() * sample_mask).sum(-1) / sample_lengths.clamp_min(1)
        ).mean()

        log_losses, convergence_losses = [], []
        magnitudes = {}
        for size in self.fft_sizes:
            predicted_magnitude = self._magnitude(predicted, size)
            target_magnitude = self._magnitude(target, size)
            magnitudes[size] = (predicted_magnitude, target_magnitude)
            hop = size // 4
            valid_frames = ((sample_lengths - size).clamp_min(0) // hop + 1).clamp_max(
                predicted_magnitude.shape[-1]
            )
            frame_mask = (
                torch.arange(predicted_magnitude.shape[-1], device=predicted.device)[None, None]
                < valid_frames[:, None, None]
            )
            denominator = frame_mask.sum((1, 2)).clamp_min(1) * predicted_magnitude.shape[1]
            log_losses.append(
                ((predicted_magnitude.log() - target_magnitude.log()).abs() * frame_mask).sum(
                    (1, 2)
                )
                / denominator
            )
            difference = (predicted_magnitude - target_magnitude) * frame_mask
            convergence_losses.append(
                torch.linalg.vector_norm(difference, dim=(1, 2))
                / torch.linalg.vector_norm(target_magnitude * frame_mask, dim=(1, 2)).clamp_min(
                    1e-5
                )
            )

        if 1024 in magnitudes:
            predicted_magnitude, target_magnitude = magnitudes[1024]
        else:
            predicted_magnitude = self._magnitude(predicted, 1024)
            target_magnitude = self._magnitude(target, 1024)
        valid_mel_frames = ((sample_lengths - 1024).clamp_min(0) // 256 + 1).clamp_max(
            predicted_magnitude.shape[-1]
        )
        mel_mask = (
            torch.arange(predicted_magnitude.shape[-1], device=predicted.device)[None, None]
            < valid_mel_frames[:, None, None]
        )
        mel_filter = self._mel_filter(predicted.device)
        predicted_mel = torch.matmul(mel_filter, predicted_magnitude).clamp_min(1e-5)
        target_mel = torch.matmul(mel_filter, target_magnitude).clamp_min(1e-5)
        mel_denominator = mel_mask.sum((1, 2)).clamp_min(1) * predicted_mel.shape[1]
        log_mel = (
            ((predicted_mel.log() - target_mel.log()).abs() * mel_mask).sum((1, 2))
            / mel_denominator
        ).mean()
        metrics = {
            "waveform_l1": waveform_l1,
            "log_spectral": torch.stack(log_losses).mean(),
            "spectral_convergence": torch.stack(convergence_losses).mean(),
            "log_mel": log_mel,
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
