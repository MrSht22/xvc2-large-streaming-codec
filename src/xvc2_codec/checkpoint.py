from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch


def restore_cuda_rng_state(states: list[torch.Tensor]) -> None:
    if not states or not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    if device < len(states):
        torch.cuda.set_rng_state(states[device], device=device)


def save_checkpoint(path: Path, **payload: Any) -> None:
    payload = dict(payload)
    payload["format_version"] = 1
    payload["rng"] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, *, restore_rng: bool = True) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not restore_rng:
        return payload
    random.setstate(payload["rng"]["python"])
    torch.set_rng_state(payload["rng"]["torch"])
    if "cuda" in payload["rng"] and torch.cuda.is_available():
        restore_cuda_rng_state(payload["rng"]["cuda"])
    return payload
