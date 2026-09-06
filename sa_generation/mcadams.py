"""VoicePrivacy B2 McAdams anonymization.

The LPC pole-rotation algorithm is derived from the official VoicePrivacy
implementation at commit f59d6282fddd6c09d5dba5c261a180334ce2797d:
anonymization/modules/mcadams/anonymise_dir_mcadams_rand_seed.py

The wrapper keeps the algorithm parameters used by VoicePrivacy 2022 B2 and
uses stable SHA256 identity hashing for reproducible speaker-level mapping.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import numpy as np
import scipy.signal
import soundfile as sf


def coefficient_for_identity(
    identity: str, minimum: float = 0.5, maximum: float = 0.9
) -> float:
    seed = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16)
    return float(np.random.default_rng(seed).uniform(minimum, maximum))


def anonymize_samples(
    samples: np.ndarray,
    sample_rate: int,
    coefficient: float,
    window_ms: float = 20.0,
    shift_ms: float = 10.0,
    lpc_order: int = 20,
) -> np.ndarray:
    if samples.ndim != 1:
        raise ValueError(f"Expected mono samples, received shape {samples.shape}")
    window_length = int(np.floor(window_ms * 0.001 * sample_rate))
    shift_length = int(np.floor(shift_ms * 0.001 * sample_rate))
    if samples.size < window_length:
        raise ValueError(
            f"Audio must contain at least {window_length} samples for a {window_ms} ms window"
        )

    eps = np.finfo(np.float32).eps
    samples = samples.astype(np.float64, copy=False) + eps
    analysis_window = np.hanning(window_length)
    overlap_scale = np.sum(analysis_window) / shift_length
    window = np.sqrt(analysis_window / overlap_scale)

    frames = librosa.util.frame(
        samples, frame_length=window_length, hop_length=shift_length
    ).T
    windowed_frames = frames * window
    lpc_coefficients = librosa.lpc(windowed_frames + eps, order=lpc_order, axis=1)
    poles = np.array(
        [scipy.signal.tf2zpk(np.array([1.0]), coefficients)[1] for coefficients in lpc_coefficients]
    )

    old_angles = np.angle(poles)
    new_angles = old_angles.copy()
    complex_poles = ~np.isreal(poles)
    negative = complex_poles & (old_angles < 0.0)
    positive = complex_poles & (old_angles > 0.0)
    new_angles[negative] = -((-old_angles[negative]) ** coefficient)
    new_angles[positive] = old_angles[positive] ** coefficient
    new_poles = np.abs(poles) * np.exp(1j * new_angles)

    reconstructed = []
    for old_lpc, frame, frame_poles in zip(
        lpc_coefficients, windowed_frames, new_poles
    ):
        new_lpc = np.real(np.poly(frame_poles))
        residual = scipy.signal.lfilter(old_lpc, np.array([1.0]), frame)
        reconstructed.append(
            scipy.signal.lfilter(np.array([1.0]), new_lpc, residual) * window
        )

    output = np.zeros_like(samples)
    for frame_index, frame in enumerate(reconstructed):
        start = frame_index * shift_length
        output[start : start + window_length] += frame
    peak = float(np.max(np.abs(output)))
    if peak > 0.0:
        output = output / peak * 0.999
    return output.astype(np.float32)


def anonymize_file(source: Path, destination: Path, coefficient: float) -> None:
    samples, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    output = anonymize_samples(mono, sample_rate, coefficient)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, output, sample_rate, subtype="PCM_16")

