from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .cache import temporal_cache, vector_cache


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _load_view_tensors(row: dict[str, Any]) -> dict[str, Any]:
    hidden = temporal_cache(row, "student_hidden", "student_hidden")
    if hidden is None:
        raise ValueError("Every row requires student_hidden_path")
    result = {"student_hidden": hidden}
    speaker = vector_cache(row, "speaker_target", "speaker_target")
    if speaker is not None:
        result["speaker_target"] = speaker
    for prefix, name, key in (
        ("phone_target", "phone_target", "phone_logits"),
        ("dyn_target", "dyn_target", "dyn_anchor"),
        ("prosody_target", "prosody_target", "prosody"),
    ):
        value = temporal_cache(row, prefix, key)
        if value is not None:
            result[name] = value
    return result


def _available_frames(
    row: dict[str, Any], tensors: dict[str, Any], hop_length: int, info: Any
) -> int:
    audio_samples = info.num_frames * 16_000 // info.sample_rate
    available = min(tensors["student_hidden"].frames, audio_samples // hop_length)
    if available <= 0:
        raise RuntimeError(f"No aligned frames for {row['audio_path']}")
    return available


def _load_audio_crop(
    row: dict[str, Any], start_sample: int, num_samples: int, info: Any | None = None
) -> torch.Tensor:
    info = torchaudio.info(row["audio_path"]) if info is None else info
    if info.sample_rate == 16_000:
        waveform, _ = torchaudio.load(
            row["audio_path"], frame_offset=start_sample, num_frames=num_samples
        )
    else:
        waveform, sample_rate = torchaudio.load(row["audio_path"])
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        waveform = waveform[:, start_sample : start_sample + num_samples]
    if waveform.shape[0] != 1:
        waveform = waveform.mean(0, keepdim=True)
    if waveform.shape[-1] < num_samples:
        raise RuntimeError(f"Partial audio read was shorter than requested: {row['audio_path']}")
    return waveform[:, :num_samples]


def _crop_view(
    row: dict[str, Any],
    tensors: dict[str, Any],
    hop_length: int,
    start: int,
    frames: int,
    info: Any,
) -> dict[str, Any]:
    result = {
        "waveform": _load_audio_crop(row, start * hop_length, frames * hop_length, info=info),
        "student_hidden": tensors["student_hidden"].read(start, frames),
        "frames": frames,
    }
    for name in ("speaker_target", "phone_target", "dyn_target", "prosody_target"):
        if name in tensors:
            value = tensors[name]
            result[name] = value if name == "speaker_target" else value.read(start, frames)
    return result


def _crop_start(available: int, frames: int, seed: int | None) -> int:
    if seed is None:
        return int(torch.randint(available - frames + 1, (1,)))
    return random.Random(seed).randrange(available - frames + 1)


def load_view(
    row: dict[str, Any],
    hop_length: int,
    segment_frames: int,
    random_crop: bool,
    crop_seed: int | None = None,
) -> dict[str, Any]:
    tensors = _load_view_tensors(row)
    info = torchaudio.info(row["audio_path"])
    available = _available_frames(row, tensors, hop_length, info)
    frames = min(available, segment_frames)
    start = _crop_start(available, frames, crop_seed) if random_crop else 0
    return _crop_view(row, tensors, hop_length, start, frames, info)


class SourceDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], hop_length: int, segment_frames: int) -> None:
        self.rows, self.hop_length, self.segment_frames = rows, hop_length, segment_frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return load_view(self.rows[index], self.hop_length, self.segment_frames, True)

    def load(self, index: int, crop_seed: int) -> dict[str, Any]:
        return load_view(self.rows[index], self.hop_length, self.segment_frames, True, crop_seed)


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], hop_length: int, segment_frames: int) -> None:
        self.rows, self.hop_length, self.segment_frames = rows, hop_length, segment_frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.load(index, int(torch.randint(2**31, (1,))))

    def load(self, index: int, crop_seed: int) -> dict[str, Any]:
        row = self.rows[index]
        source_tensors = _load_view_tensors(row["source"])
        sa_tensors = _load_view_tensors(row["sa"])
        source_info = torchaudio.info(row["source"]["audio_path"])
        sa_info = torchaudio.info(row["sa"]["audio_path"])
        available = min(
            _available_frames(row["source"], source_tensors, self.hop_length, source_info),
            _available_frames(row["sa"], sa_tensors, self.hop_length, sa_info),
        )
        frames = min(available, self.segment_frames)
        start = _crop_start(available, frames, crop_seed)
        return {
            "source": _crop_view(
                row["source"], source_tensors, self.hop_length, start, frames, source_info
            ),
            "sa": _crop_view(row["sa"], sa_tensors, self.hop_length, start, frames, sa_info),
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


class TrainingStepDataset(Dataset):
    """Build deterministic per-rank batches ahead of GPU consumption."""

    def __init__(
        self,
        source: SourceDataset,
        pair: PairDataset,
        *,
        start_step: int,
        end_step: int,
        batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        pair_probability: float,
    ) -> None:
        self.source = source
        self.pair = pair
        self.start_step = start_step
        self.end_step = end_step
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.pair_probability = pair_probability

    def __len__(self) -> int:
        return self.end_step - self.start_step

    def __getitem__(self, offset: int) -> dict[str, Any]:
        step = self.start_step + offset + 1
        choose_pair = random.Random(self.seed + step).random() < self.pair_probability
        dataset = self.pair if choose_pair else self.source
        selection_seed = self.seed * 1_000_003 + step
        global_size = self.batch_size * self.world_size
        indices = random.Random(selection_seed).sample(range(len(dataset)), global_size)
        begin = self.rank * self.batch_size
        indices = indices[begin : begin + self.batch_size]
        items = [
            dataset.load(index, selection_seed * 1_000_003 + begin + position)
            for position, index in enumerate(indices)
        ]
        return {
            "step": step,
            "kind": "pair" if choose_pair else "source",
            "batch": collate_pairs(items) if choose_pair else collate_views(items),
        }
