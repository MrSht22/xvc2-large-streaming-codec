from __future__ import annotations

import torch


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state["decay"])
        loaded_shadow = state["shadow"]
        if not isinstance(loaded_shadow, dict):
            raise TypeError("EMA shadow state must be a dictionary")
        missing = sorted(set(self.shadow) - set(loaded_shadow))
        unexpected = sorted(set(loaded_shadow) - set(self.shadow))
        if missing or unexpected:
            raise ValueError(f"EMA shadow keys differ: missing={missing}, unexpected={unexpected}")
        for name, current in self.shadow.items():
            loaded = loaded_shadow[name]
            if not torch.is_tensor(loaded):
                raise TypeError(f"EMA shadow value {name!r} must be a Tensor")
            if loaded.shape != current.shape:
                raise ValueError(
                    f"EMA shadow shape differs for {name}: "
                    f"checkpoint={list(loaded.shape)}, model={list(current.shape)}"
                )
            current.copy_(loaded)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)
