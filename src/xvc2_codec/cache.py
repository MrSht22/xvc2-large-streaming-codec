from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch


DTYPES = {"float16": np.float16, "float32": np.float32}


def _torch_tensor(path: str, key: str | None) -> torch.Tensor:
    value = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        if key and key in value:
            value = value[key]
        elif "tensor" in value:
            value = value["tensor"]
        else:
            raise ValueError(f"Tensor dictionary at {path} requires key {key!r}")
    if not torch.is_tensor(value):
        raise TypeError(f"Cache at {path} is not a Tensor")
    return value


@lru_cache(maxsize=16)
def _vector_store(path: str, key: str | None) -> torch.Tensor:
    return _torch_tensor(path, key)


@dataclass(frozen=True)
class TemporalCache:
    path: str
    frames: int
    dimension: int
    offset_frames: int = 0
    dtype: str = "float16"
    legacy_tensor: torch.Tensor | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.frames, self.dimension

    def read(self, start: int = 0, frames: int | None = None) -> torch.Tensor:
        count = self.frames - start if frames is None else frames
        if start < 0 or count < 0 or start + count > self.frames:
            raise IndexError(
                f"Invalid cache crop start={start}, frames={count}, shape={self.shape}"
            )
        if self.legacy_tensor is not None:
            return self.legacy_tensor[start : start + count].float()
        numpy_dtype = DTYPES.get(self.dtype)
        if numpy_dtype is None:
            raise ValueError(f"Unsupported cache dtype: {self.dtype}")
        byte_offset = (self.offset_frames + start) * self.dimension * np.dtype(numpy_dtype).itemsize
        values = np.memmap(
            Path(self.path).expanduser(),
            mode="r",
            dtype=numpy_dtype,
            offset=byte_offset,
            shape=(count, self.dimension),
        )
        return torch.from_numpy(np.asarray(values).copy()).float()


def temporal_cache(row: dict[str, Any], prefix: str, key: str) -> TemporalCache | None:
    path_field = f"{prefix}_path"
    path = row.get(path_field)
    if not path:
        return None
    frames_field = f"{prefix}_frames"
    dimension_field = f"{prefix}_dim"
    if frames_field in row or dimension_field in row or f"{prefix}_offset_frames" in row:
        required = (frames_field, dimension_field)
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"Sharded {prefix} cache is missing {missing}")
        return TemporalCache(
            path=str(path),
            frames=int(row[frames_field]),
            dimension=int(row[dimension_field]),
            offset_frames=int(row.get(f"{prefix}_offset_frames", 0)),
            dtype=str(row.get(f"{prefix}_dtype", "float16")),
        )
    value = _torch_tensor(str(path), key)
    if value.ndim != 2:
        raise ValueError(f"Expected rank-2 {prefix} cache, got {list(value.shape)}")
    return TemporalCache(str(path), value.shape[0], value.shape[1], legacy_tensor=value)


def vector_cache(row: dict[str, Any], prefix: str, key: str | None = None) -> torch.Tensor | None:
    path = row.get(f"{prefix}_path")
    if not path:
        return None
    value = _vector_store(str(Path(path).expanduser().resolve()), key)
    index = row.get(f"{prefix}_index")
    if index is not None:
        if value.ndim != 2:
            raise ValueError(f"Indexed {prefix} cache must be rank 2, got {list(value.shape)}")
        value = value[int(index)]
    if value.ndim != 1:
        raise ValueError(f"Expected rank-1 {prefix} cache, got {list(value.shape)}")
    return value.float()
