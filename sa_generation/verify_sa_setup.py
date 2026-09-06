#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path

from model_assets import REQUIRED_STTTS_MODELS
from sa_common import configure_espeak_library


ROOT = Path(__file__).resolve().parent
VENDOR_RUNTIME_FILES = (
    "LICENSE",
    "anonymization/pipelines/sttts/sttts_pipeline.py",
    "anonymization/modules/sttts/speaker_embeddings/speaker_extraction.py",
    "anonymization/modules/sttts/speaker_embeddings/speaker_embeddings.py",
    "anonymization/modules/sttts/speaker_embeddings/speechbrain_vectors.py",
    "anonymization/modules/sttts/speaker_embeddings/utils.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a VoicePrivacy SA runtime.")
    parser.add_argument("--backend", required=True, choices=("mcadams", "sttts"))
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=ROOT / "third_party" / "Voice-Privacy-Challenge-2026",
    )
    args = parser.parse_args()

    print(f"python={sys.version.split()[0]}")
    for package in ("numpy", "scipy", "librosa", "soundfile"):
        print(f"{package}={package_version(package)}")

    vendor_dir = args.vendor_dir.expanduser().resolve()
    missing_vendor_files = [
        name for name in VENDOR_RUNTIME_FILES if not (vendor_dir / name).is_file()
    ]
    if missing_vendor_files:
        raise FileNotFoundError(
            "Missing vendored runtime files: " + ", ".join(missing_vendor_files)
        )
    print(f"vendor_dir={vendor_dir}")
    print("vendor_commit=f59d6282fddd6c09d5dba5c261a180334ce2797d")
    print(f"vendor_runtime_files=PASS,files={len(VENDOR_RUNTIME_FILES)}")

    if args.backend == "sttts":
        if args.models_dir is None:
            raise ValueError("--models-dir is required for sttts")
        models_dir = args.models_dir.expanduser().resolve()
        missing = [name for name in REQUIRED_STTTS_MODELS if not (models_dir / name).is_file()]
        if missing:
            raise FileNotFoundError("Missing STTTS models: " + ", ".join(missing))
        manifest_path = models_dir / "model_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing model_manifest.json; run prepare_models.py first: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_models = manifest.get("required_model_files", {})
        for name in REQUIRED_STTTS_MODELS:
            record = recorded_models.get(name)
            if not isinstance(record, dict):
                raise RuntimeError(f"Model manifest has no record for {name}")
            model_path = models_dir / name
            if model_path.stat().st_size != record.get("bytes"):
                raise RuntimeError(f"Model size does not match manifest: {name}")
            if sha256(model_path) != record.get("sha256"):
                raise RuntimeError(f"Model SHA256 does not match manifest: {name}")
        print(f"model_integrity=PASS,files={len(REQUIRED_STTTS_MODELS)}")

        import torch
        import torchaudio
        from phonemizer.backend import EspeakBackend

        print(f"torch={torch.__version__}")
        print(f"torchaudio={torchaudio.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu={torch.cuda.get_device_name(0)}")
        for package in (
            "espnet",
            "espnet_model_zoo",
            "speechbrain",
            "silero_vad",
            "phonemizer",
            "parselmouth",
        ):
            print(f"{package}={package_version(package)}")
        configure_espeak_library()
        if shutil.which("espeak-ng") is None or not EspeakBackend.is_available():
            raise RuntimeError("espeak-ng executable or shared library is unavailable")
        print(f"espeak_ng={shutil.which('espeak-ng')}")
        print(f"espeak_library={EspeakBackend.library()}")
        print(f"models_dir={models_dir}")

    print("setup_verification=PASS")


if __name__ == "__main__":
    main()
