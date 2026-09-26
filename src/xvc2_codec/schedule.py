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
    normalized_f0: float
    voicing: float
    relative_energy: float
    f0_delta: float
    dyn_phone_adversary: float
    edit_phone_adversary: float


def _ramp(step: int, start: int, end: int) -> float:
    return min(max((step - start) / max(end - start, 1), 0.0), 1.0)


def weights_at(
    step: int,
    schedule: ScheduleConfig,
    loss: LossConfig,
    phase_start_step: int | None = None,
) -> ActiveWeights:
    gan = _ramp(step, schedule.reconstruction_end, schedule.gan_ramp_end)
    sa = _ramp(step, schedule.gan_ramp_end, schedule.sa_ramp_end)
    disentanglement_start = (
        schedule.gan_ramp_end if phase_start_step is None else phase_start_step
    )
    disentanglement = _ramp(
        step,
        disentanglement_start,
        disentanglement_start + schedule.disentanglement_ramp_steps,
    )
    return ActiveWeights(
        reconstruction=loss.reconstruction,
        adversarial=loss.adversarial * gan,
        feature_matching=loss.feature_matching * gan,
        edit_style=loss.edit_style,
        sa_inv=loss.sa_inv * sa,
        sa_dyn=loss.sa_dyn * sa,
        phone_anchor=loss.phone_anchor,
        dyn_anchor=loss.dyn_anchor,
        normalized_f0=loss.normalized_f0 * disentanglement,
        voicing=loss.voicing * disentanglement,
        relative_energy=loss.relative_energy * disentanglement,
        f0_delta=loss.f0_delta * disentanglement,
        dyn_phone_adversary=loss.dyn_phone_adversary * disentanglement,
        edit_phone_adversary=loss.edit_phone_adversary * disentanglement,
    )
