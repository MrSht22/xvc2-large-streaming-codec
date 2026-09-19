from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        lines = tqdm(stream, desc=f"read {path.name}", unit="lines", file=sys.stderr)
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        values = list(rows)
        for row in tqdm(values, desc=f"write {label}", unit="rows", file=sys.stderr):
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def _utterance_id(row: dict[str, Any]) -> str:
    if row.get("utterance_id") is not None:
        return str(row["utterance_id"])
    source = row.get("source")
    if isinstance(source, dict) and source.get("utterance_id") is not None:
        return str(source["utterance_id"])
    path = row.get("audio_path")
    if path:
        return str(path)
    if isinstance(source, dict) and source.get("audio_path"):
        return str(source["audio_path"])
    raise ValueError("Manifest row has no utterance_id or audio_path")


def _speaker_id(row: dict[str, Any]) -> str:
    if row.get("speaker_id") is not None:
        return str(row["speaker_id"])
    source = row.get("source")
    if isinstance(source, dict) and source.get("speaker_id") is not None:
        return str(source["speaker_id"])
    raise ValueError(f"Row {_utterance_id(row)!r} has no speaker_id")


def _audio_key(row: dict[str, Any]) -> str:
    paths = [row.get("audio_path")]
    source = row.get("source")
    if isinstance(source, dict):
        paths.append(source.get("audio_path"))
    return "|".join(sorted(str(path) for path in paths if path))


def _chapter_id(row: dict[str, Any]) -> str:
    if row.get("chapter_id") is not None:
        return str(row["chapter_id"])
    utterance = _utterance_id(row)
    return utterance.rsplit("-", 1)[0] if "-" in utterance else utterance


def _has_phone_target(row: dict[str, Any]) -> bool:
    return bool(row.get("phone_target_path"))


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _validate_unique(rows: list[dict[str, Any]], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        item = _utterance_id(row)
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    if duplicates:
        examples = sorted(duplicates)[:5]
        raise ValueError(f"{label} contains duplicate utterance IDs: {examples}")


def _exclusion_keys(paths: list[Path]) -> tuple[set[str], set[str]]:
    utterances: set[str] = set()
    audio: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            utterances.add(_utterance_id(row))
            audio.add(_audio_key(row))
    return utterances, audio


def _filter_pool(
    rows: list[dict[str, Any]], excluded_utterances: set[str], excluded_audio: set[str]
) -> list[dict[str, Any]]:
    result = []
    for row in tqdm(rows, desc="filter candidate pool", unit="rows", file=sys.stderr):
        if _utterance_id(row) not in excluded_utterances and _audio_key(row) not in excluded_audio:
            result.append(row)
    _validate_unique(result, "candidate pool")
    return result


def _group_by_speaker(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_speaker_id(row)].append(row)
    return groups


def _shuffle_group(rows: list[dict[str, Any]], seed: int, speaker_id: str) -> list[dict[str, Any]]:
    result = list(rows)
    random.Random(_stable_seed(seed, speaker_id)).shuffle(result)
    return result


def select_content_rows(
    rows: list[dict[str, Any]],
    *,
    speakers: int,
    utterances_per_speaker: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a speaker-balanced phone-labeled set for content probes."""
    candidates = [row for row in rows if _has_phone_target(row)]
    groups = _group_by_speaker(candidates)
    speaker_ids = sorted(groups)
    random.Random(seed).shuffle(speaker_ids)
    if speakers > 0:
        speaker_ids = speaker_ids[:speakers]
    selected: list[dict[str, Any]] = []
    for speaker_id in tqdm(
        sorted(speaker_ids), desc="select content speakers", unit="speakers", file=sys.stderr
    ):
        group = _shuffle_group(groups[speaker_id], seed, speaker_id)
        selected.extend(group[:utterances_per_speaker])
    if not selected:
        raise RuntimeError("Content selection produced no phone-labeled rows")
    return sorted(selected, key=_utterance_id)


def _chapter_balanced_rows(
    rows: list[dict[str, Any]], count: int, seed: int, speaker_id: str
) -> list[dict[str, Any]]:
    chapters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        chapters[_chapter_id(row)].append(row)
    chapter_ids = sorted(chapters)
    random.Random(_stable_seed(seed, f"chapters:{speaker_id}")).shuffle(chapter_ids)
    for chapter_id in chapter_ids:
        chapters[chapter_id] = _shuffle_group(chapters[chapter_id], seed, chapter_id)
    selected: list[dict[str, Any]] = []
    while len(selected) < count and chapter_ids:
        progressed = False
        for chapter_id in chapter_ids:
            if chapters[chapter_id]:
                selected.append(chapters[chapter_id].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def select_leakage_rows(
    rows: list[dict[str, Any]],
    *,
    speakers: int,
    utterances_per_speaker: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select repeated, chapter-balanced utterances for speaker leakage probes."""
    groups = _group_by_speaker(rows)
    eligible = [
        speaker_id
        for speaker_id, group in groups.items()
        if len({_chapter_id(row) for row in group}) >= min(utterances_per_speaker, 2)
        and len(group) >= utterances_per_speaker
    ]
    if len(eligible) < speakers:
        raise RuntimeError(
            f"Only {len(eligible)} speakers have {utterances_per_speaker} usable utterances; "
            f"requested {speakers}"
        )
    random.Random(seed).shuffle(eligible)
    selected: list[dict[str, Any]] = []
    for speaker_id in tqdm(
        sorted(eligible[:speakers]), desc="select leakage speakers", unit="speakers", file=sys.stderr
    ):
        selected.extend(
            _chapter_balanced_rows(groups[speaker_id], utterances_per_speaker, seed, speaker_id)
        )
    return sorted(selected, key=lambda row: (_speaker_id(row), _utterance_id(row)))


def _pair_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    _validate_unique(rows, "pair manifest")
    return {_utterance_id(row): row for row in rows}


def _select_pairs(
    source_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]] | None, label: str
) -> tuple[list[dict[str, Any]], list[str]]:
    if pair_rows is None:
        return [], []
    indexed = _pair_index(pair_rows)
    selected = []
    missing = []
    for row in tqdm(source_rows, desc=f"match {label} pairs", unit="rows", file=sys.stderr):
        item = indexed.get(_utterance_id(row))
        if item is None:
            missing.append(_utterance_id(row))
        else:
            selected.append(item)
    if missing:
        raise RuntimeError(
            f"{label} has {len(missing)} source rows without a pair row; first={missing[:5]}"
        )
    return selected, missing


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = _group_by_speaker(rows)
    counts = [len(group) for group in groups.values()]
    return {
        "rows": len(rows),
        "speakers": len(groups),
        "min_utterances_per_speaker": min(counts) if counts else 0,
        "max_utterances_per_speaker": max(counts) if counts else 0,
        "mean_utterances_per_speaker": sum(counts) / len(counts) if counts else 0.0,
        "chapters": len({_chapter_id(row) for row in rows}),
    }


def build_eval_caches(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = read_jsonl(args.source_manifest)
    pair_rows = read_jsonl(args.pair_manifest) if args.pair_manifest else None
    excluded_utterances, excluded_audio = _exclusion_keys(args.exclude_manifest)
    source_pool = _filter_pool(source_rows, excluded_utterances, excluded_audio)
    content_rows = select_content_rows(
        source_pool,
        speakers=args.content_speakers,
        utterances_per_speaker=args.content_utterances_per_speaker,
        seed=args.seed,
    )
    leakage_rows = select_leakage_rows(
        source_pool,
        speakers=args.leakage_speakers,
        utterances_per_speaker=args.leakage_utterances_per_speaker,
        seed=args.seed + 1,
    )
    content_pairs, _ = _select_pairs(content_rows, pair_rows, "content selection")
    leakage_pairs, _ = _select_pairs(leakage_rows, pair_rows, "leakage selection")

    output_dir = args.output_dir.expanduser().resolve()
    write_jsonl(
        content_rows,
        output_dir / "content" / "source_eval_cache.jsonl",
        "content source manifest",
    )
    write_jsonl(
        leakage_rows,
        output_dir / "leakage" / "source_eval_cache.jsonl",
        "leakage source manifest",
    )
    if pair_rows is not None:
        write_jsonl(
            content_pairs,
            output_dir / "content" / "pair_eval_cache.jsonl",
            "content pair manifest",
        )
        write_jsonl(
            leakage_pairs,
            output_dir / "leakage" / "pair_eval_cache.jsonl",
            "leakage pair manifest",
        )
    report = {
        "status": "PASS",
        "source_manifest": str(args.source_manifest.expanduser().resolve()),
        "pair_manifest": (
            str(args.pair_manifest.expanduser().resolve()) if args.pair_manifest else None
        ),
        "excluded_manifests": [str(path.expanduser().resolve()) for path in args.exclude_manifest],
        "candidate_pool": _summary(source_pool),
        "content": {
            "source": _summary(content_rows),
            "pair": _summary(content_pairs) if pair_rows is not None else None,
            "phone_labeled_rows": sum(_has_phone_target(row) for row in content_rows),
        },
        "leakage": {
            "source": _summary(leakage_rows),
            "pair": _summary(leakage_pairs) if pair_rows is not None else None,
        },
        "selection": {
            "seed": args.seed,
            "content_speakers": args.content_speakers,
            "content_utterances_per_speaker": args.content_utterances_per_speaker,
            "leakage_speakers": args.leakage_speakers,
            "leakage_utterances_per_speaker": args.leakage_utterances_per_speaker,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build speaker-balanced Codec evaluation caches")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--content-speakers", type=int, default=128)
    parser.add_argument("--content-utterances-per-speaker", type=int, default=4)
    parser.add_argument("--leakage-speakers", type=int, default=64)
    parser.add_argument("--leakage-utterances-per-speaker", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    positive = (
        args.content_speakers,
        args.content_utterances_per_speaker,
        args.leakage_speakers,
        args.leakage_utterances_per_speaker,
    )
    if any(value <= 0 for value in positive):
        parser.error("selection sizes must be positive")
    report = build_eval_caches(args)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    print("codec_eval_cache_build=PASS")


if __name__ == "__main__":
    main()
