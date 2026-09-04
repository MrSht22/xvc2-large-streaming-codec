from __future__ import annotations

import argparse
import json
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .config import CodecConfig, load_config
from .data import read_jsonl


def audio_metadata(path: Path) -> tuple[int, int]:
    try:
        metadata = torchaudio.info(path)
        return metadata.sample_rate, metadata.num_frames
    except RuntimeError:
        if path.suffix.lower() != ".wav":
            raise
        with wave.open(str(path), "rb") as stream:
            return stream.getframerate(), stream.getnframes()


def load_tensor(path: str, preferred_key: str) -> torch.Tensor:
    value = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        if preferred_key in value:
            value = value[preferred_key]
        elif "tensor" in value:
            value = value["tensor"]
        else:
            raise ValueError(f"missing tensor key {preferred_key!r}")
    if not torch.is_tensor(value):
        raise TypeError("cache is not a Tensor")
    return value


def audit_view(
    row: dict[str, Any],
    label: str,
    config: CodecConfig,
    speaker_target_dim: int,
    alignment_tolerance: int,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    required = {"audio_path", "student_hidden_path", "speaker_target_path"}
    missing = sorted(required - row.keys())
    if missing:
        return [f"{label}:missing={missing}"], {}
    details: dict[str, Any] = {}
    try:
        sample_rate, num_frames = audio_metadata(Path(row["audio_path"]).expanduser())
        audio_frames = round(num_frames * 16_000 / sample_rate) // config.hop_length
        details.update(
            {
                "audio_seconds": num_frames / sample_rate,
                "sample_rate": sample_rate,
                "audio_frames_50hz": audio_frames,
            }
        )
    except Exception as error:
        failures.append(f"{label}:audio={type(error).__name__}")
        audio_frames = None
    cache_contracts = {
        "student_hidden_path": ("student_hidden", config.student_dim, True),
        "speaker_target_path": ("speaker_target", speaker_target_dim, False),
        "phone_target_path": ("phone_logits", config.vocab_size, True),
        "dyn_target_path": ("dyn_anchor", config.dyn_dim, True),
        "prosody_target_path": ("prosody", 4, True),
    }
    for field, (key, dimension, temporal) in cache_contracts.items():
        if field not in row or not row[field]:
            continue
        try:
            tensor = load_tensor(row[field], key)
        except Exception as error:
            failures.append(f"{label}:{field}={type(error).__name__}:{error}")
            continue
        details[field] = list(tensor.shape)
        expected_rank = 2 if temporal else 1
        if tensor.ndim != expected_rank or tensor.shape[-1] != dimension:
            failures.append(f"{label}:{field}:shape={list(tensor.shape)}:expected_last={dimension}")
        if temporal and audio_frames is not None and tensor.ndim == 2:
            if abs(tensor.shape[0] - audio_frames) > alignment_tolerance:
                failures.append(
                    f"{label}:{field}:frames={tensor.shape[0]}:audio_frames={audio_frames}"
                )
    return failures, details


def audit_manifests(
    source_path: Path,
    pair_path: Path,
    config: CodecConfig,
    speaker_target_dim: int,
    max_items: int | None = None,
    alignment_tolerance: int = 1,
) -> dict[str, Any]:
    failures: list[str] = []
    counters: Counter[str] = Counter()
    seen: set[str] = set()
    source_rows = read_jsonl(source_path)
    pair_rows = read_jsonl(pair_path)
    source_selected = source_rows[:max_items] if max_items is not None else source_rows
    pair_selected = pair_rows[:max_items] if max_items is not None else pair_rows
    for index, row in enumerate(source_selected):
        label = f"source:{index}"
        item_failures, _ = audit_view(row, label, config, speaker_target_dim, alignment_tolerance)
        failures.extend(item_failures)
        key = str(row.get("utterance_id") or row.get("audio_path"))
        if key in seen:
            failures.append(f"{label}:duplicate={key}")
        seen.add(key)
        counters["source_views_scanned"] += 1
    for index, row in enumerate(pair_selected):
        if not isinstance(row.get("source"), dict) or not isinstance(row.get("sa"), dict):
            failures.append(f"pair:{index}:requires_source_and_sa_objects")
            continue
        pair_details = []
        for view in ("source", "sa"):
            label = f"pair:{index}:{view}"
            item_failures, details = audit_view(
                row[view], label, config, speaker_target_dim, alignment_tolerance
            )
            failures.extend(item_failures)
            pair_details.append(details)
            counters["pair_views_scanned"] += 1
        if len(pair_details) == 2:
            source_hidden = pair_details[0].get("student_hidden_path")
            sa_hidden = pair_details[1].get("student_hidden_path")
            if (
                source_hidden
                and sa_hidden
                and abs(source_hidden[0] - sa_hidden[0]) > alignment_tolerance
            ):
                failures.append(
                    f"pair:{index}:source_sa_frame_mismatch={source_hidden[0]}:{sa_hidden[0]}"
                )
    return {
        "source_manifest": str(source_path),
        "pair_manifest": str(pair_path),
        "source_rows_total": len(source_rows),
        "pair_rows_total": len(pair_rows),
        **counters,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Codec source and SA-pair manifests")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--speaker-target-dim", type=int, required=True)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--alignment-tolerance-frames", type=int, default=1)
    args = parser.parse_args()
    report = audit_manifests(
        args.source_manifest.resolve(),
        args.pair_manifest.resolve(),
        load_config(args.config).model,
        args.speaker_target_dim,
        args.max_items,
        args.alignment_tolerance_frames,
    )
    print(json.dumps(report, sort_keys=True))
    print(f"codec_manifest_audit={report['status']}")
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
