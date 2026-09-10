from __future__ import annotations

import argparse
import json
import time
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torchaudio

from .cache import temporal_cache, vector_cache
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


def audit_view(
    row: dict[str, Any],
    label: str,
    config: CodecConfig,
    speaker_target_dim: int,
    alignment_tolerance: int,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    required = {"audio_path", "student_hidden_path", "speaker_target_path"}
    missing = sorted(name for name in required if not row.get(name))
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
        "student_hidden": ("student_hidden", config.student_dim, True),
        "speaker_target": ("speaker_target", speaker_target_dim, False),
        "phone_target": ("phone_logits", config.vocab_size, True),
        "dyn_target": ("dyn_anchor", config.dyn_dim, True),
        "prosody_target": ("prosody", 4, True),
    }
    for prefix, (key, dimension, temporal) in cache_contracts.items():
        field = f"{prefix}_path"
        if field not in row or not row[field]:
            continue
        try:
            cache = temporal_cache(row, prefix, key) if temporal else vector_cache(row, prefix, key)
            assert cache is not None
            shape = cache.shape if temporal else tuple(cache.shape)
            if temporal:
                cache.read(0, min(1, cache.frames))
        except Exception as error:
            failures.append(f"{label}:{field}={type(error).__name__}:{error}")
            continue
        details[field] = list(shape)
        expected_rank = 2 if temporal else 1
        if len(shape) != expected_rank or shape[-1] != dimension:
            failures.append(f"{label}:{field}:shape={list(shape)}:expected_last={dimension}")
        if temporal and audio_frames is not None and len(shape) == 2:
            if abs(shape[0] - audio_frames) > alignment_tolerance:
                failures.append(f"{label}:{field}:frames={shape[0]}:audio_frames={audio_frames}")
    return failures, details


def audit_manifests(
    source_path: Path,
    pair_path: Path,
    config: CodecConfig,
    speaker_target_dim: int,
    max_items: int | None = None,
    alignment_tolerance: int = 1,
    progress_every: int | None = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    failures: list[str] = []
    counters: Counter[str] = Counter()
    seen: set[str] = set()
    source_rows = read_jsonl(source_path)
    pair_rows = read_jsonl(pair_path)
    source_selected = source_rows[:max_items] if max_items is not None else source_rows
    pair_selected = pair_rows[:max_items] if max_items is not None else pair_rows
    stage_started = time.monotonic()

    def progress(stage: str, completed: int, total: int, force: bool = False) -> None:
        if progress_every is None or (not force and completed % progress_every):
            return
        elapsed = max(time.monotonic() - stage_started, 1e-6)
        print(
            f"audit_progress stage={stage} items={completed}/{total} "
            f"failures={len(failures)} elapsed_seconds={elapsed:.1f} "
            f"items_per_second={completed / elapsed:.2f}",
            flush=True,
        )

    def audit_source(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any], list[str]]:
        index, row = item
        label = f"source:{index}"
        item_failures, _ = audit_view(row, label, config, speaker_target_dim, alignment_tolerance)
        return index, row, item_failures

    def audit_pair(item: tuple[int, dict[str, Any]]) -> tuple[int, list[str], int]:
        index, row = item
        item_failures = []
        views_scanned = 0
        if not isinstance(row.get("source"), dict) or not isinstance(row.get("sa"), dict):
            return index, [f"pair:{index}:requires_source_and_sa_objects"], views_scanned
        pair_details = []
        for view in ("source", "sa"):
            label = f"pair:{index}:{view}"
            view_failures, details = audit_view(
                row[view], label, config, speaker_target_dim, alignment_tolerance
            )
            item_failures.extend(view_failures)
            pair_details.append(details)
            views_scanned += 1
        source_hidden = pair_details[0].get("student_hidden_path")
        sa_hidden = pair_details[1].get("student_hidden_path")
        if source_hidden and sa_hidden:
            if "alignment_lag_frames" in row:
                lag = int(row["alignment_lag_frames"])
                overlap = min(source_hidden[0] - max(-lag, 0), sa_hidden[0] - max(lag, 0))
                if overlap <= 0:
                    item_failures.append(f"pair:{index}:no_overlap_after_lag={lag}")
            elif abs(source_hidden[0] - sa_hidden[0]) > alignment_tolerance:
                item_failures.append(
                    f"pair:{index}:source_sa_frame_mismatch={source_hidden[0]}:{sa_hidden[0]}"
                )
        return index, item_failures, views_scanned

    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="codec-audit") as executor:
        def bounded_map(function: Any, rows: list[dict[str, Any]]) -> Any:
            window = num_workers * 8
            for begin in range(0, len(rows), window):
                items = enumerate(rows[begin : begin + window], start=begin)
                yield from executor.map(function, items)

        source_results = bounded_map(audit_source, source_selected)
        for index, row, item_failures in source_results:
            failures.extend(item_failures)
            key = str(row.get("utterance_id") or row.get("audio_path"))
            if key in seen:
                failures.append(f"source:{index}:duplicate={key}")
            seen.add(key)
            counters["source_views_scanned"] += 1
            progress("source", index + 1, len(source_selected), index + 1 == len(source_selected))

        stage_started = time.monotonic()
        pair_results = bounded_map(audit_pair, pair_selected)
        for index, item_failures, views_scanned in pair_results:
            failures.extend(item_failures)
            counters["pair_views_scanned"] += views_scanned
            progress("pair", index + 1, len(pair_selected), index + 1 == len(pair_selected))
    return {
        "source_manifest": str(source_path),
        "pair_manifest": str(pair_path),
        "source_rows_total": len(source_rows),
        "pair_rows_total": len(pair_rows),
        "num_workers": num_workers,
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
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    if args.progress_every <= 0 or args.num_workers <= 0:
        parser.error("--progress-every and --num-workers must be positive")
    report = audit_manifests(
        args.source_manifest.resolve(),
        args.pair_manifest.resolve(),
        load_config(args.config).model,
        args.speaker_target_dim,
        args.max_items,
        args.alignment_tolerance_frames,
        args.progress_every,
        args.num_workers,
    )
    print(json.dumps(report, sort_keys=True))
    print(f"codec_manifest_audit={report['status']}")
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
