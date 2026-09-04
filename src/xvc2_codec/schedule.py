from __future__ import annotations

from dataclasses import dataclass

from .config import LossConfig, ScheduleConfig


@dataclass(frozen=True)
class ActiveWeights:
    reconstruction: float
    adversarial: float
    feature_matching: float
    edit_style: float
    sa_inv: float
    sa_dyn: float
    phone_anchor: float
    dyn_anchor: float
    prosody_anchor: float


def _ramp(step: int, start: int, end: int) -> float:
    return min(max((step - start) / max(end - start, 1), 0.0), 1.0)


def weights_at(step: int, schedule: ScheduleConfig, loss: LossConfig) -> ActiveWeights:
    gan = _ramp(step, schedule.reconstruction_end, schedule.gan_ramp_end)
    sa = _ramp(step, schedule.gan_ramp_end, schedule.sa_ramp_end)
    return ActiveWeights(
        reconstruction=loss.reconstruction,
        adversarial=loss.adversarial * gan,
        feature_matching=loss.feature_matching * gan,
        edit_style=loss.edit_style,
        sa_inv=loss.sa_inv * sa,
        sa_dyn=loss.sa_dyn * sa,
        phone_anchor=loss.phone_anchor,
        dyn_anchor=loss.dyn_anchor,
        prosody_anchor=loss.prosody_anchor,
    )
