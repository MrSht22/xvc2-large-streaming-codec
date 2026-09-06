#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


INVALID_ID_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
DEFAULT_EXTENSIONS = (".wav", ".flac")


def normalize_extension(value: str) -> str:
    value = value.lower()
    return value if value.startswith(".") else f".{value}"


def sanitize_id(value: str) -> str:
    sanitized = INVALID_ID_CHARS.sub("-", value).strip("-.")
    if not sanitized:
        raise ValueError(f"Cannot create a valid ID from {value!r}")
    return sanitized


def build_rows(
    audio_root: Path,
    extensions: tuple[str, ...],
    speaker_parent_depth: int,
    limit: int | None,
) -> list[dict]:
    audio_root = audio_root.expanduser().resolve()
    if not audio_root.is_dir():
        raise NotADirectoryError(f"Audio root does not exist: {audio_root}")
    if speaker_parent_depth < 1:
        raise ValueError("--speaker-parent-depth must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")

    allowed = {normalize_extension(value) for value in extensions}
    audio_paths = sorted(
        path.resolve()
        for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    )
    if limit is not None:
        audio_paths = audio_paths[:limit]
    if not audio_paths:
        raise ValueError(f"No matching audio files found under {audio_root}")

    rows = []
    seen_ids: dict[str, Path] = {}
    for audio_path in audio_paths:
        relative = audio_path.relative_to(audio_root)
        parent_parts = relative.parent.parts
        if len(parent_parts) < speaker_parent_depth:
            raise ValueError(
                f"{relative} has no parent at depth {speaker_parent_depth}; "
                "adjust --speaker-parent-depth"
            )
        speaker_component = parent_parts[-speaker_parent_depth]
        utterance_components = (*relative.parent.parts, relative.stem)
        utterance_id = sanitize_id("__".join(utterance_components))
        speaker_id = sanitize_id(speaker_component)
        previous = seen_ids.get(utterance_id)
        if previous is not None:
            raise ValueError(
                f"Generated duplicate utterance_id {utterance_id!r}: "
                f"{previous} and {audio_path}"
            )
        seen_ids[utterance_id] = audio_path
        rows.append(
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "audio_path": str(audio_path),
                "relative_audio_path": relative.as_posix(),
                "gender": "u",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic JSONL manifest for batch SA inference."
    )
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--speaker-parent-depth",
        type=int,
        default=1,
        help="1 uses the immediate parent directory; LibriTTS uses 2",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Audio filename extensions to include",
    )
    parser.add_argument("--limit", type=int, help="Keep only the first N sorted files")
    args = parser.parse_args()

    rows = build_rows(
        audio_root=args.audio_root,
        extensions=tuple(args.extensions),
        speaker_parent_depth=args.speaker_parent_depth,
        limit=args.limit,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    extension_counts = Counter(Path(row["audio_path"]).suffix.lower() for row in rows)
    print(f"audio_root={args.audio_root.expanduser().resolve()}")
    print(f"utterances={len(rows)}")
    print(f"speakers={len({row['speaker_id'] for row in rows})}")
    print(f"extension_counts={dict(sorted(extension_counts.items()))}")
    print(f"manifest={output}")
    print("manifest_creation=PASS")


if __name__ == "__main__":
    main()
