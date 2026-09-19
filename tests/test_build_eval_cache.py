import json
from pathlib import Path

from xvc2_codec.build_eval_cache import build_eval_caches, select_content_rows, select_leakage_rows


def source_row(speaker: str, chapter: int, utterance: int) -> dict[str, object]:
    item = f"{speaker}-{chapter:04d}-{utterance:04d}"
    return {
        "utterance_id": item,
        "speaker_id": speaker,
        "audio_path": f"/audio/{item}.wav",
        "student_hidden_path": f"/hidden/{item}.pt",
        "phone_target_path": f"/phone/{item}.pt",
    }


def pair_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "utterance_id": row["utterance_id"],
        "speaker_id": row["speaker_id"],
        "source": {**row, "audio_path": row["audio_path"]},
        "sa": {**row, "audio_path": f"/sa/{row['utterance_id']}.wav"},
    }


def test_select_content_is_speaker_balanced() -> None:
    rows = [source_row(str(speaker), chapter, utterance) for speaker in range(3) for chapter in range(2) for utterance in range(2)]
    selected = select_content_rows(rows, speakers=2, utterances_per_speaker=3, seed=3)
    counts = {}
    for row in selected:
        counts[row["speaker_id"]] = counts.get(row["speaker_id"], 0) + 1
    assert len(selected) == 6
    assert sorted(counts.values()) == [3, 3]


def test_select_leakage_uses_repeated_speakers_and_chapters() -> None:
    rows = [source_row(str(speaker), chapter, utterance) for speaker in range(3) for chapter in range(4) for utterance in range(2)]
    selected = select_leakage_rows(rows, speakers=2, utterances_per_speaker=4, seed=4)
    groups = {}
    for row in selected:
        groups.setdefault(row["speaker_id"], []).append(row)
    assert len(groups) == 2
    assert all(len(items) == 4 for items in groups.values())
    assert all(len({_chapter.rsplit("-", 1)[0] for _chapter in [item["utterance_id"] for item in items]}) >= 2 for items in groups.values())


def test_build_eval_caches_writes_source_and_pair_manifests(tmp_path: Path) -> None:
    rows = [source_row(str(speaker), chapter, utterance) for speaker in range(3) for chapter in range(4) for utterance in range(2)]
    source_path = tmp_path / "source.jsonl"
    pair_path = tmp_path / "pair.jsonl"
    source_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    pair_path.write_text("\n".join(json.dumps(pair_row(row)) for row in rows) + "\n", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "source_manifest": source_path,
            "pair_manifest": pair_path,
            "exclude_manifest": [],
            "output_dir": tmp_path / "out",
            "content_speakers": 2,
            "content_utterances_per_speaker": 2,
            "leakage_speakers": 2,
            "leakage_utterances_per_speaker": 4,
            "seed": 1,
        },
    )()
    report = build_eval_caches(args)
    assert report["status"] == "PASS"
    assert (tmp_path / "out/content/source_eval_cache.jsonl").exists()
    assert (tmp_path / "out/content/pair_eval_cache.jsonl").exists()
    assert (tmp_path / "out/leakage/source_eval_cache.jsonl").exists()
    assert (tmp_path / "out/leakage/pair_eval_cache.jsonl").exists()


def test_build_eval_caches_selects_from_source_pair_intersection(tmp_path: Path) -> None:
    paired_rows = [
        source_row(str(speaker), chapter, utterance)
        for speaker in range(2)
        for chapter in range(4)
        for utterance in range(2)
    ]
    source_rows = paired_rows + [source_row("unpaired", chapter, utterance) for chapter in range(4) for utterance in range(2)]
    source_path = tmp_path / "source.jsonl"
    pair_path = tmp_path / "pair.jsonl"
    source_path.write_text("\n".join(json.dumps(row) for row in source_rows) + "\n", encoding="utf-8")
    pair_path.write_text("\n".join(json.dumps(pair_row(row)) for row in paired_rows) + "\n", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "source_manifest": source_path,
            "pair_manifest": pair_path,
            "exclude_manifest": [],
            "output_dir": tmp_path / "out",
            "content_speakers": 2,
            "content_utterances_per_speaker": 2,
            "leakage_speakers": 2,
            "leakage_utterances_per_speaker": 4,
            "seed": 1,
        },
    )()
    report = build_eval_caches(args)
    assert report["candidate_pool_before_pair_filter"] == len(source_rows)
    assert report["candidate_pool"]["rows"] == len(paired_rows)


def test_build_eval_caches_falls_back_to_nested_pair_source(tmp_path: Path) -> None:
    pair_source_rows = [
        source_row(str(speaker), chapter, utterance)
        for speaker in range(2)
        for chapter in range(4)
        for utterance in range(2)
    ]
    unrelated_source_rows = [source_row("unrelated", chapter, utterance) for chapter in range(4) for utterance in range(2)]
    source_path = tmp_path / "source.jsonl"
    pair_path = tmp_path / "pair.jsonl"
    source_path.write_text(
        "\n".join(json.dumps(row) for row in unrelated_source_rows) + "\n", encoding="utf-8"
    )
    pair_path.write_text(
        "\n".join(json.dumps(pair_row(row)) for row in pair_source_rows) + "\n", encoding="utf-8"
    )
    args = type(
        "Args",
        (),
        {
            "source_manifest": source_path,
            "pair_manifest": pair_path,
            "exclude_manifest": [],
            "output_dir": tmp_path / "out",
            "content_speakers": 2,
            "content_utterances_per_speaker": 2,
            "leakage_speakers": 2,
            "leakage_utterances_per_speaker": 4,
            "seed": 1,
        },
    )()
    report = build_eval_caches(args)
    assert report["candidate_pool_source"] == "pair_manifest_nested_source"
    assert report["candidate_pool"]["rows"] == len(pair_source_rows)
