#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if "utterance_id" not in row or "speaker_id" not in row:
                raise ValueError(
                    f"Manifest line {line_number} needs utterance_id and speaker_id"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    utterance_ids = [str(row["utterance_id"]) for row in rows]
    if len(utterance_ids) != len(set(utterance_ids)):
        raise ValueError("Manifest contains duplicate utterance_id values")
    return rows


def partition_by_speaker(rows: list[dict], shard_count: int) -> list[list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["speaker_id"])].append(row)
    if shard_count > len(groups):
        raise ValueError(
            f"Requested {shard_count} workers for only {len(groups)} speakers"
        )

    shards: list[list[dict]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for _, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        shard_index = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[shard_index].extend(group)
        loads[shard_index] += len(group)
    return shards


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def shard_output_is_complete(shard_manifest: Path, shard_output: Path) -> bool:
    pair_path = shard_output / "pairs.jsonl"
    if not pair_path.is_file():
        return False
    try:
        expected_ids = {
            str(row["utterance_id"]) for row in read_jsonl(shard_manifest)
        }
        actual_ids = {str(row["utterance_id"]) for row in read_jsonl(pair_path)}
    except (OSError, ValueError):
        return False
    return actual_ids == expected_ids


def merge_pairs(source_rows: list[dict], shard_dirs: list[Path], output: Path) -> None:
    pairs = {}
    for shard_dir in shard_dirs:
        pair_path = shard_dir / "pairs.jsonl"
        if not pair_path.is_file():
            raise FileNotFoundError(f"Missing shard output: {pair_path}")
        for row in read_jsonl(pair_path):
            utterance_id = str(row["utterance_id"])
            if utterance_id in pairs:
                raise RuntimeError(f"Duplicate shard output for {utterance_id}")
            pairs[utterance_id] = row

    source_ids = [str(row["utterance_id"]) for row in source_rows]
    missing = [utterance_id for utterance_id in source_ids if utterance_id not in pairs]
    extras = sorted(set(pairs) - set(source_ids))
    if missing or extras:
        raise RuntimeError(
            f"Cannot merge shard outputs: missing={len(missing)}, extras={len(extras)}"
        )
    write_jsonl([pairs[utterance_id] for utterance_id in source_ids], output)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run speaker-safe STTTS manifest shards in parallel on multiple GPUs."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--anonymization-level", choices=("spk", "utt"), default="spk")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--launch-delay-seconds", type=float, default=5.0)
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=ROOT / "third_party" / "Voice-Privacy-Challenge-2026",
    )
    args = parser.parse_args()

    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU ID")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be at least 1")
    if args.launch_delay_seconds < 0:
        raise ValueError("--launch-delay-seconds cannot be negative")

    manifest = args.manifest.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()
    vendor_dir = args.vendor_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = read_jsonl(manifest)
    worker_count = len(gpus) * args.workers_per_gpu
    shards = partition_by_speaker(rows, worker_count)

    metadata = {
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "models_dir": str(models_dir),
        "vendor_dir": str(vendor_dir),
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "anonymization_level": args.anonymization_level,
        "seed": args.seed,
        "shard_utterance_counts": [len(shard) for shard in shards],
        "shard_speaker_counts": [
            len({str(row["speaker_id"]) for row in shard}) for shard in shards
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "parallel_run_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous != metadata:
            raise RuntimeError(
                f"Output directory belongs to a different parallel run: {output_dir}"
            )
    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    manifest_dir = output_dir / "shard_manifests"
    log_dir = output_dir / "logs"
    shard_root = output_dir / "shards"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    log_streams = []
    shard_dirs = []

    try:
        for shard_index, shard_rows in enumerate(shards):
            gpu = gpus[shard_index % len(gpus)]
            shard_name = f"shard-{shard_index:03d}"
            shard_manifest = manifest_dir / f"{shard_name}.jsonl"
            shard_output = shard_root / shard_name
            log_path = log_dir / f"{shard_name}.log"
            write_jsonl(shard_rows, shard_manifest)
            shard_dirs.append(shard_output)

            command = [
                sys.executable,
                "-u",
                str(ROOT / "anonymize_sa.py"),
                "--manifest",
                str(shard_manifest),
                "--backend",
                "sttts",
                "--models-dir",
                str(models_dir),
                "--vendor-dir",
                str(vendor_dir),
                "--output-dir",
                str(shard_output),
                "--anonymization-level",
                args.anonymization_level,
                "--gpus",
                gpu,
                "--seed",
                str(args.seed + shard_index),
            ]
            log_stream = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            processes.append((shard_name, gpu, process, log_path))
            log_streams.append(log_stream)
            print(
                f"launched={shard_name},gpu={gpu},pid={process.pid},"
                f"utterances={len(shard_rows)},log={log_path}",
                flush=True,
            )
            if shard_index + 1 < len(shards) and args.launch_delay_seconds:
                time.sleep(args.launch_delay_seconds)

        failures = []
        for shard_name, gpu, process, log_path in processes:
            return_code = process.wait()
            print(
                f"finished={shard_name},gpu={gpu},return_code={return_code},log={log_path}",
                flush=True,
            )
            if return_code != 0:
                failures.append((shard_name, return_code, log_path))
        if failures:
            details = "; ".join(
                f"{name}:code={code},log={log}" for name, code, log in failures
            )
            raise RuntimeError(f"Parallel STTTS workers failed: {details}")
    except KeyboardInterrupt:
        for _, _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, _, process, _ in processes:
            process.wait()
        raise
    finally:
        for stream in log_streams:
            stream.close()

    pair_manifest = output_dir / "pairs.jsonl"
    merge_pairs(rows, shard_dirs, pair_manifest)
    print(f"workers={worker_count}")
    print(f"utterances={len(rows)}")
    print(f"pair_manifest={pair_manifest}")
    print("parallel_anonymization=PASS")


if __name__ == "__main__":
    main()
