"""Split a large VC JSONL by natural target; attach sibling references to validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a"}


def path_key(value: str) -> str:
    # Lexical normalization avoids resolving millions of paths on network storage.
    return os.path.abspath(value)


def manifest_rows(manifest: Path):
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                for field in ("source_wav_path", "target_wav_path", "target_utt"):
                    if not isinstance(row[field], str) or not row[field]:
                        raise ValueError(f"Missing/non-string {field}")
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid VC row at {manifest}:{line_number}: {error}") from error
            yield row, line


def find_reference(target: Path, blocked: set[str], directory_cache: dict, seed: int):
    """Emilia: speaker/mp3/target.mp3 -> another file in that same mp3 directory."""
    directory = target.parent
    if directory not in directory_cache:
        directory_cache[directory] = (
            sorted(
                path
                for path in directory.iterdir()
                if path.suffix.lower() in AUDIO_SUFFIXES and path.is_file()
            )
            if directory.is_dir()
            else []
        )
    candidates = [
        path
        for path in directory_cache[directory]
        if path.stem != target.stem and path_key(str(path)) not in blocked
    ]
    if not target.is_file() or not candidates:
        return None
    digest = hashlib.sha256(f"{seed}:{target}".encode()).digest()
    reference = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
    return str(reference)


def split_manifest(
    manifest: Path, train_manifest: Path, val_manifest: Path, val_size: int = 256, seed: int = 42
) -> dict:
    if val_size <= 0:
        raise ValueError("val_size must be positive")
    paths = [path_key(str(path)) for path in (manifest, train_manifest, val_manifest)]
    if len(set(paths)) != 3:
        raise ValueError("Input, train and val manifest paths must be distinct")
    if train_manifest.exists() or val_manifest.exists():
        raise FileExistsError("Output manifest already exists; choose new output paths")

    counts, identities, blocked_sources = {}, {}, {}
    for row, _ in manifest_rows(manifest):
        target = path_key(row["target_wav_path"])
        previous = identities.setdefault(row["target_utt"], target)
        if previous != target:
            raise ValueError(f"target_utt {row['target_utt']!r} maps to multiple target paths")
        counts[target] = counts.get(target, 0) + 1
        source = path_key(row["source_wav_path"])
        if Path(source).parent == Path(target).parent:
            blocked_sources.setdefault(target, set()).add(source)
    if len(counts) <= val_size:
        raise ValueError(
            f"Need more than {val_size} unique targets to leave nonempty train; found {len(counts)}"
        )

    targets = sorted(counts)
    random.Random(seed).shuffle(targets)
    references, directory_cache = {}, {}
    skipped = 0
    for target in targets:
        reference = find_reference(
            Path(target), blocked_sources.get(target, set()), directory_cache, seed
        )
        if reference is None:
            skipped += 1
            continue
        references[target] = reference
        if len(references) == val_size:
            break
    if len(references) != val_size:
        raise ValueError(
            f"Only {len(references)} targets have a sibling reference; "
            f"need {val_size}. Check that target paths/directories are mounted."
        )

    # Selection and input validation finish before creating either output.
    for output in (train_manifest, val_manifest):
        output.parent.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = 0, 0
    with (
        train_manifest.open("x", encoding="utf-8") as train_stream,
        val_manifest.open("x", encoding="utf-8") as val_stream,
    ):
        for row, line in manifest_rows(manifest):
            target = path_key(row["target_wav_path"])
            if target in references:
                row["target_reference_wav_path"] = references[target]
                val_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                val_rows += 1
            else:
                train_stream.write(line if line.endswith("\n") else line + "\n")
                train_rows += 1
    return {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "seed": seed,
        "train_targets": len(counts) - val_size,
        "val_targets": val_size,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "skipped_targets_without_reference": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument(
        "--val-size", type=int, default=256, help="Number of unique natural targets"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = split_manifest(
        args.manifest, args.train_manifest, args.val_manifest, args.val_size, args.seed
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
