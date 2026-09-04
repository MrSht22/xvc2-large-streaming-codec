from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _load_tensor(path: str, key: str | None = None) -> torch.Tensor:
    value = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        if key and key in value:
            value = value[key]
        elif "tensor" in value:
            value = value["tensor"]
        else:
            raise ValueError(f"Tensor dictionary at {path} requires key {key!r}")
    return value.float()


def load_view(
    row: dict[str, Any], hop_length: int, segment_frames: int, random_crop: bool
) -> dict[str, Any]:
    waveform, sample_rate = torchaudio.load(row["audio_path"])
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    if waveform.shape[0] != 1:
        waveform = waveform.mean(0, keepdim=True)
    hidden = _load_tensor(row["student_hidden_path"], "student_hidden")
    available = min(hidden.shape[0], waveform.shape[-1] // hop_length)
    frames = min(available, segment_frames)
    if frames <= 0:
        raise RuntimeError(f"No aligned frames for {row['audio_path']}")
    start = int(torch.randint(available - frames + 1, (1,))) if random_crop else 0
    result = {
        "waveform": waveform[:, start * hop_length : (start + frames) * hop_length],
        "student_hidden": hidden[start : start + frames],
        "frames": frames,
    }
    optional = {
        "speaker_target_path": ("speaker_target", None),
        "phone_target_path": ("phone_target", "phone_logits"),
        "dyn_target_path": ("dyn_target", "dyn_anchor"),
        "prosody_target_path": ("prosody_target", "prosody"),
    }
    for field, (name, key) in optional.items():
        if row.get(field):
            value = _load_tensor(row[field], key)
            result[name] = value if name == "speaker_target" else value[start : start + frames]
    return result


class SourceDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], hop_length: int, segment_frames: int) -> None:
        self.rows, self.hop_length, self.segment_frames = rows, hop_length, segment_frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return load_view(self.rows[index], self.hop_length, self.segment_frames, True)


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], hop_length: int, segment_frames: int) -> None:
        self.rows, self.hop_length, self.segment_frames = rows, hop_length, segment_frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "source": load_view(row["source"], self.hop_length, self.segment_frames, True),
            "sa": load_view(row["sa"], self.hop_length, self.segment_frames, True),
        }


def collate_views(items: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "waveform": pad_sequence(
            [item["waveform"].T for item in items], batch_first=True
        ).transpose(1, 2),
        "student_hidden": pad_sequence(
            [item["student_hidden"] for item in items], batch_first=True
        ),
        "frames": torch.tensor([item["frames"] for item in items]),
    }
    for name in ("speaker_target", "phone_target", "dyn_target", "prosody_target"):
        if all(name in item for item in items):
            values = [item[name] for item in items]
            result[name] = (
                torch.stack(values)
                if name == "speaker_target"
                else pad_sequence(values, batch_first=True)
            )
    return result


def collate_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {view: collate_views([item[view] for item in items]) for view in ("source", "sa")}
