#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from parallel_anonymize_sa import (
    GENERATION_CONTRACT,
    ROOT,
    file_sha256,
    merge_pairs,
    read_jsonl,
    shard_output_is_complete,
    worker_environment,
)


def create_resume_log_dir(log_root: Path) -> Path:
    log_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1000):
        candidate = log_root / f"resume-{attempt:03d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Too many resume attempts under {log_root}")


def load_parallel_run(output_dir: Path) -> tuple[dict, list[dict], list[Path], list[Path]]:
    metadata_path = output_dir / "parallel_run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing parallel run metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    required = {
        "generation_contract",
        "manifest",
        "manifest_sha256",
        "models_dir",
        "vendor_dir",
        "gpus",
        "anonymization_level",
        "seed",
        "transcript_mode",
        "online_aligner_fine_tune",
        "shard_utterance_counts",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Parallel run metadata is missing keys: {', '.join(missing)}")
    if metadata["generation_contract"] != GENERATION_CONTRACT:
        raise RuntimeError(
            "Parallel run uses an incompatible generation contract; use a new output directory"
        )

    manifest = Path(metadata["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(f"Original manifest is missing: {manifest}")
    if file_sha256(manifest) != metadata["manifest_sha256"]:
        raise RuntimeError(f"Original manifest has changed since launch: {manifest}")
    source_rows = read_jsonl(manifest)

    shard_manifests = []
    shard_dirs = []
    for shard_index, expected_count in enumerate(metadata["shard_utterance_counts"]):
        shard_name = f"shard-{shard_index:03d}"
        shard_manifest = output_dir / "shard_manifests" / f"{shard_name}.jsonl"
        if not shard_manifest.is_file():
            raise FileNotFoundError(f"Missing shard manifest: {shard_manifest}")
        actual_count = len(read_jsonl(shard_manifest))
        if actual_count != expected_count:
            raise RuntimeError(
                f"Shard manifest changed: {shard_manifest} "
                f"(expected {expected_count}, found {actual_count})"
            )
        shard_manifests.append(shard_manifest)
        shard_dirs.append(output_dir / "shards" / shard_name)
    return metadata, source_rows, shard_manifests, shard_dirs


def incomplete_shard_indexes(shard_manifests: list[Path], shard_dirs: list[Path]) -> list[int]:
    return [
        index
        for index, (manifest, output) in enumerate(zip(shard_manifests, shard_dirs))
        if not shard_output_is_complete(manifest, output)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume an interrupted parallel STTTS run without repartitioning it."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        help="Optional comma-separated GPU override; defaults to the original run.",
    )
    parser.add_argument("--launch-delay-seconds", type=float, default=5.0)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Inspect resumable state without launching workers or merging outputs.",
    )
    args = parser.parse_args()
    if args.launch_delay_seconds < 0:
        raise ValueError("--launch-delay-seconds cannot be negative")

    output_dir = args.output_dir.expanduser().resolve()
    metadata, source_rows, shard_manifests, shard_dirs = load_parallel_run(output_dir)
    original_gpus = [str(value) for value in metadata["gpus"]]
    gpus = (
        [value.strip() for value in args.gpus.split(",") if value.strip()]
        if args.gpus is not None
        else original_gpus
    )
    if not gpus:
        raise ValueError("At least one GPU ID is required")

    incomplete = incomplete_shard_indexes(shard_manifests, shard_dirs)
    complete_count = len(shard_dirs) - len(incomplete)
    print(f"shards_total={len(shard_dirs)}")
    print(f"shards_complete={complete_count}")
    print(f"shards_to_resume={len(incomplete)}")
    print(
        "resume_shards="
        + (",".join(f"shard-{index:03d}" for index in incomplete) or "none")
    )
    if args.check_only:
        print("resume_check=PASS")
        return

    if not incomplete:
        pair_manifest = output_dir / "pairs.jsonl"
        merge_pairs(source_rows, shard_dirs, pair_manifest)
        print(f"pair_manifest={pair_manifest}")
        print("parallel_resume=PASS")
        return

    log_dir = create_resume_log_dir(output_dir / "logs")
    processes = []
    log_streams = []
    try:
        for shard_index in incomplete:
            shard_name = f"shard-{shard_index:03d}"
            gpu = gpus[shard_index % len(gpus)]
            log_path = log_dir / f"{shard_name}.log"
            command = [
                sys.executable,
                "-u",
                str(ROOT / "anonymize_sa.py"),
                "--manifest",
                str(shard_manifests[shard_index]),
                "--backend",
                "sttts",
                "--models-dir",
                str(Path(metadata["models_dir"])),
                "--vendor-dir",
                str(Path(metadata["vendor_dir"])),
                "--output-dir",
                str(shard_dirs[shard_index]),
                "--anonymization-level",
                str(metadata["anonymization_level"]),
                "--gpus",
                "0",
                "--seed",
                str(int(metadata["seed"]) + shard_index),
                "--transcript-mode",
                str(metadata["transcript_mode"]),
            ]
            if metadata["online_aligner_fine_tune"]:
                command.append("--online-aligner-fine-tune")
            log_stream = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=worker_environment(gpu),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            processes.append((shard_name, gpu, process, log_path))
            log_streams.append(log_stream)
            print(
                f"resumed={shard_name},gpu={gpu},pid={process.pid},log={log_path}",
                flush=True,
            )
            if shard_index != incomplete[-1] and args.launch_delay_seconds:
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
            raise RuntimeError(f"Resumed STTTS workers failed: {details}")
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

    remaining = incomplete_shard_indexes(shard_manifests, shard_dirs)
    if remaining:
        names = ", ".join(f"shard-{index:03d}" for index in remaining)
        raise RuntimeError(f"Shard outputs are still incomplete: {names}")

    pair_manifest = output_dir / "pairs.jsonl"
    merge_pairs(source_rows, shard_dirs, pair_manifest)
    print(f"resumed_shards={len(incomplete)}")
    print(f"utterances={len(source_rows)}")
    print(f"pair_manifest={pair_manifest}")
    print("parallel_resume=PASS")


if __name__ == "__main__":
    main()
