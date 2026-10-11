"""Offline reference frontend; target conditions enter only the latent Converter."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


def load_speaker_encoder(xvc_root: Path, model_dir: Path, device: torch.device):
    """Reuse X-VC's ERes2Net and its exact FBank/checkpoint layout."""
    if not (xvc_root / "models/codec/sac/modules/speaker_encoder.py").is_file():
        raise FileNotFoundError(f"Not an X-VC checkout: {xvc_root}")
    sys.path.insert(0, str(xvc_root.resolve()))
    from models.codec.sac.modules.speaker_encoder import SpeakerEmbedder

    return SpeakerEmbedder(str(model_dir), freeze=True).eval().requires_grad_(False).to(device)


class ReferenceFrontend(torch.nn.Module):
    def __init__(self, speaker_encoder: torch.nn.Module, mels: int = 80) -> None:
        super().__init__()
        self.speaker_encoder = speaker_encoder.eval().requires_grad_(False)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=512,
            win_length=400,
            hop_length=320,
            n_mels=mels,
            center=False,
            power=1.0,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.speaker_encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor):
        if (lengths < 8000).any():
            raise ValueError(
                "Reference has less than 0.5s outside the supervised crop. "
                "Provide target_reference_wav_path (and source_reference_wav_path for reverse), "
                "preferably a 2-6s same-speaker different-utterance reference."
            )
        embeddings, mels = [], []
        # Trim before ERes2Net pooling and Mel extraction: batch padding is not a reference.
        for index, length in enumerate(lengths.tolist()):
            audio = waveform[index, 0, :length].float()
            if not torch.isfinite(audio).all() or audio.abs().max() < 1e-6:
                raise ValueError("Reference audio is silent or nonfinite")
            output = self.speaker_encoder(audio[None])
            embedding = output[0] if isinstance(output, tuple) else output
            embeddings.append(embedding[0])
            mels.append(self.mel(audio).clamp_min(1e-5).log())
        mel_lengths = torch.tensor([mel.shape[-1] for mel in mels], device=waveform.device)
        maximum = int(mel_lengths.max())
        return (
            torch.stack(embeddings),
            torch.stack([F.pad(mel, (0, maximum - mel.shape[-1])) for mel in mels]),
            mel_lengths,
        )
