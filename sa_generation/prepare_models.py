#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from model_assets import (
    ARCHIVE_NAMES,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_ARCHIVE_SIZES,
    REQUIRED_STTTS_MODELS,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        bad_member = zip_file.testzip()
        if bad_member is not None:
            raise RuntimeError(f"CRC verification failed for {archive}: {bad_member}")
        for member in zip_file.infolist():
            target = (root / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member in {archive}: {member.filename}")
        zip_file.extractall(root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and extract public VoicePrivacy STTTS v2.0 model archives."
    )
    parser.add_argument("--archives-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing extracted model directory"
    )
    args = parser.parse_args()

    archives_dir = args.archives_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    archives = [archives_dir / name for name in ARCHIVE_NAMES]
    missing = [str(path) for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model archives: " + ", ".join(missing))

    archive_manifest = {}
    for archive in archives:
        actual_size = archive.stat().st_size
        expected_size = EXPECTED_ARCHIVE_SIZES[archive.name]
        if actual_size != expected_size:
            raise RuntimeError(
                f"Unexpected size for {archive.name}: {actual_size}, expected {expected_size}"
            )
        actual_sha256 = sha256(archive)
        if actual_sha256 != EXPECTED_ARCHIVE_SHA256[archive.name]:
            raise RuntimeError(
                f"Unexpected SHA256 for {archive.name}: {actual_sha256}"
            )
        archive_manifest[archive.name] = {
            "bytes": actual_size,
            "sha256": actual_sha256,
        }

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise RuntimeError(
                f"Output model directory is not empty: {output_dir}. Pass --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        print(f"extracting={archive.name}")
        safe_extract(archive, output_dir)

    missing_models = [
        name for name in REQUIRED_STTTS_MODELS if not (output_dir / name).is_file()
    ]
    if missing_models:
        raise RuntimeError(
            "Extracted model layout is incomplete: " + ", ".join(missing_models)
        )

    model_files = {
        name: {
            "bytes": (output_dir / name).stat().st_size,
            "sha256": sha256(output_dir / name),
        }
        for name in REQUIRED_STTTS_MODELS
    }
    manifest = {
        "source": "DigitalPhonetics/speaker-anonymization release v2.0",
        "source_url": "https://github.com/DigitalPhonetics/speaker-anonymization/releases/tag/v2.0",
        "archives": archive_manifest,
        "required_model_files": model_files,
    }
    manifest_path = output_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"model_manifest={manifest_path}")
    print("model_preparation=PASS")


if __name__ == "__main__":
    main()
