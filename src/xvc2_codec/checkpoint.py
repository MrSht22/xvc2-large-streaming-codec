from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch


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


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(payload["rng"]["python"])
    torch.set_rng_state(payload["rng"]["torch"])
    if "cuda" in payload["rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return payload
