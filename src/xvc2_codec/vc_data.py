"""X-VC paired-audio sampling on the large Codec's 16 kHz / 50 Hz axis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from .data import _load_audio_with_retries, read_jsonl

SAMPLE_RATE = 16_000
HOP_LENGTH = 320


@dataclass(frozen=True)
class VCPairDataConfig:
    segment_duration: float = 2.4
    max_val_duration: float = 12.0
    reconstruction_ratio: float = 0.2
    reversed_ratio: float = 0.4
    mask_target_condition: bool = True
    volume_normalize: bool = True
    highpass_cutoff_freq: float = 40.0
    max_reference_duration: float = 6.0

    def __post_init__(self) -> None:
        if (
            min(self.segment_duration, self.max_val_duration, self.max_reference_duration)
            * SAMPLE_RATE
            < HOP_LENGTH
        ):
            raise ValueError("Segment durations must contain at least one 50 Hz frame")
        if (
            not 0 <= self.reconstruction_ratio <= 1
            or not 0 <= self.reversed_ratio <= 1
            or self.reconstruction_ratio + self.reversed_ratio > 1
        ):
            raise ValueError("Sampling ratios must be nonnegative and sum to at most one")
        if not 0 <= self.highpass_cutoff_freq < SAMPLE_RATE / 2:
            raise ValueError("Highpass cutoff must be in [0, 8000) Hz")


def _normalize_volume(waveform: torch.Tensor) -> torch.Tensor:
    """Use X-VC's 0.2 upper-amplitude normalization, including its quiet-audio rule."""
    amplitudes = waveform.abs().flatten().sort().values
    if amplitudes[-1] < 0.1:
        waveform = waveform * (0.1 / amplitudes[-1].clamp_min(1e-3))
    significant = amplitudes[amplitudes > 0.01]
    if significant.numel() <= 10:
        return waveform
    count = significant.numel()
    volume = significant[int(0.9 * count) : int(0.99 * count)].mean()
    waveform = waveform * (0.2 / volume).clamp(0.1, 10.0)
    return waveform / waveform.abs().max().clamp_min(1.0)


def _load_waveform(path: str, config: VCPairDataConfig) -> torch.Tensor:
    waveform, sample_rate = _load_audio_with_retries(path)
    waveform = waveform.float().mean(0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Empty or nonfinite VC audio: {path}")
    if config.volume_normalize:
        waveform = _normalize_volume(waveform)
    if config.highpass_cutoff_freq:
        waveform = torchaudio.functional.highpass_biquad(
            waveform, SAMPLE_RATE, config.highpass_cutoff_freq
        )
    return waveform


class VCPairDataset(Dataset):
    """Read the original X-VC JSONL; emit Codec-shaped source/target waveform views.

    Training uses PyTorch's worker RNG for role sampling and shared random crops.
    Validation always uses the standard direction and a crop starting at zero.
    This assumes that source and target already share a content/time axis.
    """

    def __init__(self, manifest: Path, config: VCPairDataConfig, random_crop: bool = True) -> None:
        self.rows = read_jsonl(manifest)
        if not self.rows:
            raise ValueError(f"Empty VC manifest: {manifest}")
        self.config = config
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        source_path, target_path = row["source_wav_path"], row["target_wav_path"]
        source_utt, target_utt = row.get("source_utt"), row["target_utt"]
        reference_path = row.get("target_reference_wav_path")
        role = "standard"
        if self.random_crop:
            draw = float(torch.rand(()))
            if draw < self.config.reconstruction_ratio:
                role = "reconstruction"
                source_path, source_utt = target_path, target_utt
            elif draw < self.config.reconstruction_ratio + self.config.reversed_ratio:
                role = "reversed"
                source_path, target_path = target_path, source_path
                source_utt, target_utt = target_utt, source_utt
                reference_path = row.get("source_reference_wav_path")

        source = _load_waveform(source_path, self.config)
        target = (
            source.clone()
            if source_path == target_path
            else _load_waveform(target_path, self.config)
        )
        source_samples, target_samples = source.shape[-1], target.shape[-1]
        available = min(source_samples, target_samples) // HOP_LENGTH
        if available < 1:
            raise ValueError(f"No complete shared 50 Hz frame for VC pair {row['target_utt']}")
        duration = (
            self.config.segment_duration if self.random_crop else self.config.max_val_duration
        )
        requested = int(duration * SAMPLE_RATE / HOP_LENGTH)
        crop_frames = requested if self.random_crop else min(requested, available)
        valid_frames = min(crop_frames, available)
        start_frame = (
            int(torch.randint(available - crop_frames + 1, ()))
            if self.random_crop and available > crop_frames
            else 0
        )
        start, stop = start_frame * HOP_LENGTH, (start_frame + crop_frames) * HOP_LENGTH
        source = source[:, : available * HOP_LENGTH]
        target = target[:, : available * HOP_LENGTH]
        natural = source if role == "reversed" else target
        source_condition, target_condition = source.clone(), target.clone()
        if self.config.mask_target_condition:
            source_condition[:, start:stop] = 0
            target_condition[:, start:stop] = 0

        views = {}
        for name, waveform in (("source", source), ("target", target)):
            crop = waveform[:, start:stop]
            views[name] = {
                "waveform": F.pad(crop, (0, crop_frames * HOP_LENGTH - crop.shape[-1])),
                "frames": valid_frames,
            }
        natural_crop = natural[:, start:stop]
        views["natural"] = {
            "waveform": F.pad(natural_crop, (0, crop_frames * HOP_LENGTH - natural_crop.shape[-1])),
            "frames": valid_frames,
        }
        if reference_path:
            if Path(reference_path).resolve() in {
                Path(source_path).resolve(),
                Path(target_path).resolve(),
            }:
                raise ValueError("Independent reference must differ from both paired audio paths")
            reference = _load_waveform(reference_path, self.config)
            reference_kind = "independent"
        else:
            # Remove the supervised crop rather than feeding long runs of artificial silence.
            reference = (
                torch.cat((target[:, :start], target[:, stop:]), dim=-1)
                if self.config.mask_target_condition
                else target
            )
            reference_kind = "paired_masked" if self.config.mask_target_condition else "paired_full"
        reference = reference[:, : int(self.config.max_reference_duration * SAMPLE_RATE)]
        return {
            "index": row.get("index", index),
            "manifest_target_utt": row["target_utt"],
            "source_utt": source_utt,
            "target_utt": target_utt,
            "source_wav_path": source_path,
            "target_wav_path": target_path,
            "role_assignment_mode": role,
            "start_frame": start_frame,
            "source_num_samples": source_samples,
            "target_num_samples": target_samples,
            "condition_frames": available,
            "source_wav_cond": source_condition,
            "target_wav_cond": target_condition,
            "reference_waveform": reference,
            "reference_samples": reference.shape[-1],
            "reference_kind": reference_kind,
            "reference_wav_path": reference_path or target_path,
            **views,
        }

    @staticmethod
    def collate(rows: list[dict]) -> dict:
        result = {
            key: [row[key] for row in rows]
            for key in rows[0]
            if key
            not in {
                "source",
                "target",
                "natural",
                "source_wav_cond",
                "target_wav_cond",
                "reference_waveform",
            }
        }
        for name in ("source", "target", "natural"):
            samples = max(row[name]["waveform"].shape[-1] for row in rows)
            result[name] = {
                "waveform": torch.stack(
                    [
                        F.pad(row[name]["waveform"], (0, samples - row[name]["waveform"].shape[-1]))
                        for row in rows
                    ]
                ),
                "frames": torch.tensor([row[name]["frames"] for row in rows]),
            }
        for key in ("source_wav_cond", "target_wav_cond", "reference_waveform"):
            samples = max(row[key].shape[-1] for row in rows)
            result[key] = torch.stack(
                [F.pad(row[key], (0, samples - row[key].shape[-1])) for row in rows]
            )
        result["condition_frames"] = torch.tensor([row["condition_frames"] for row in rows])
        result["reference_samples"] = torch.tensor([row["reference_samples"] for row in rows])
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check X-VC pair IO without loading any model")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "val"), default="train")
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_pairs, args.batch_size) < 1:
        parser.error("--max-pairs and --batch-size must be positive")
    config = VCPairDataConfig(**yaml.safe_load(args.config.read_text(encoding="utf-8")))
    torch.manual_seed(args.seed)
    dataset = VCPairDataset(args.manifest, config, random_crop=args.mode == "train")
    loader = DataLoader(
        Subset(dataset, range(min(args.max_pairs, len(dataset)))),
        batch_size=args.batch_size,
        collate_fn=dataset.collate,
        num_workers=0,
    )
    records = []
    for batch in loader:
        for index, role in enumerate(batch["role_assignment_mode"]):
            records.append(
                {
                    "index": batch["index"][index],
                    "role": role,
                    "source_wav_path": batch["source_wav_path"][index],
                    "target_wav_path": batch["target_wav_path"][index],
                    "start_frame": batch["start_frame"][index],
                    "valid_frames": int(batch["source"]["frames"][index]),
                    "batch_samples": batch["source"]["waveform"].shape[-1],
                    "source_num_samples": batch["source_num_samples"][index],
                    "target_num_samples": batch["target_num_samples"][index],
                    "condition_frames": int(batch["condition_frames"][index]),
                    "reference_kind": batch["reference_kind"][index],
                    "reference_wav_path": batch["reference_wav_path"][index],
                    "reference_samples": int(batch["reference_samples"][index]),
                    "reference_has_0_5s": bool(batch["reference_samples"][index] >= 8000),
                    "target_condition_nonzero_samples": int(
                        torch.count_nonzero(batch["target_wav_cond"][index])
                    ),
                }
            )
    report = {
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "mode": args.mode,
        "seed": args.seed,
        "pairs": len(records),
        "roles": dict(Counter(row["role"] for row in records)),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"vc_pair_io=PASS pairs={len(records)} report={args.output}", flush=True)


if __name__ == "__main__":
    main()
