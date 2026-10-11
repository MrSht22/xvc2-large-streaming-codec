import json
from pathlib import Path

import pytest

from xvc2_codec.vc_split import split_manifest


def make_manifest(root: Path, targets: int = 260, variants: int = 2):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(targets):
        directory = root / f"EN_B00006_S{index:05d}" / "mp3"
        directory.mkdir(parents=True)
        target = directory / f"target_{index}.mp3"
        target.touch()
        # Same utterance in another encoding must not become its reference.
        target.with_suffix(".wav").touch()
        (directory / "reference.flac").touch()
        (directory / "not_audio.txt").touch()
        for variant in range(variants):
            rows.append(
                {
                    "index": len(rows),
                    "source_utt": f"src_{index}_{variant}",
                    "source_wav_path": str(root / "generated" / f"{index}_{variant}.flac"),
                    "target_utt": f"utt_{index}",
                    "target_wav_path": str(target),
                    "extra": {"transcript": "keep 原文"},
                }
            )
    manifest = root / "all.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return manifest, rows


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_default_256_targets_hold_out_all_variants_and_preserve_every_row(tmp_path):
    manifest, original = make_manifest(tmp_path)
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    report = split_manifest(manifest, train, val)
    training, validation = read(train), read(val)
    train_targets = {row["target_wav_path"] for row in training}
    val_targets = {row["target_wav_path"] for row in validation}
    assert len(val_targets) == 256 and len(validation) == 512
    assert train_targets.isdisjoint(val_targets)
    assert len(training) + len(validation) == len(original)
    assert report["val_targets"] == 256 and report["val_rows"] == 512
    assert report["train_targets"] == 4
    indexed = {row["index"]: row for row in original}
    for row in training:
        assert row == indexed[row["index"]]
        assert "target_reference_wav_path" not in row
    for row in validation:
        reference = Path(row["target_reference_wav_path"])
        assert reference.parent == Path(row["target_wav_path"]).parent
        assert reference.name == "reference.flac" and reference.is_file()
        assert {
            key: value for key, value in row.items() if key != "target_reference_wav_path"
        } == indexed[row["index"]]
    second_train, second_val = tmp_path / "train2.jsonl", tmp_path / "val2.jsonl"
    split_manifest(manifest, second_train, second_val)
    assert train.read_bytes() == second_train.read_bytes()
    assert val.read_bytes() == second_val.read_bytes()


def test_missing_reference_target_stays_in_train_and_source_cannot_be_reference(tmp_path):
    manifest, rows = make_manifest(tmp_path, targets=5, variants=1)
    Path(rows[0]["target_wav_path"]).with_name("reference.flac").unlink()
    # A same-directory source file must not be selected as target reference.
    source = Path(rows[1]["target_wav_path"]).with_name("source.wav")
    source.touch()
    rows[1]["source_wav_path"] = str(source)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    split_manifest(manifest, train, val, val_size=4)
    assert read(train)[0]["target_wav_path"] == rows[0]["target_wav_path"]
    assert len(read(val)) == 4
    assert all(Path(row["target_reference_wav_path"]).name == "reference.flac" for row in read(val))


def test_ineligible_or_insufficient_groups_fail_before_creating_outputs(tmp_path):
    manifest, rows = make_manifest(tmp_path, targets=3, variants=1)
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    with pytest.raises(ValueError, match="nonempty train"):
        split_manifest(manifest, train, val, val_size=3)
    for row in rows:
        Path(row["target_wav_path"]).with_name("reference.flac").unlink()
    with pytest.raises(ValueError, match="sibling reference"):
        split_manifest(manifest, train, val, val_size=2)
    assert not train.exists() and not val.exists()


def test_existing_outputs_and_ambiguous_target_ids_are_rejected(tmp_path):
    manifest, rows = make_manifest(tmp_path, targets=3, variants=1)
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.write_text("keep existing")
    with pytest.raises(FileExistsError):
        split_manifest(manifest, train, val, val_size=1)
    assert train.read_text() == "keep existing" and not val.exists()
    rows[1]["target_utt"] = rows[0]["target_utt"]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="multiple target paths"):
        split_manifest(manifest, tmp_path / "other.jsonl", val, val_size=1)


def test_invalid_row_is_reported_with_line_number_without_partial_outputs(tmp_path):
    manifest, _ = make_manifest(tmp_path, targets=3, variants=1)
    with manifest.open("a") as stream:
        stream.write('{"target_utt":"broken"}\n')
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    with pytest.raises(ValueError, match=r"all.jsonl:4"):
        split_manifest(manifest, train, val, val_size=1)
    assert not train.exists() and not val.exists()
