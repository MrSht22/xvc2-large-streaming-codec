from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def configure_espeak_library() -> str | None:
    """Expose Homebrew's espeak-ng library when macOS cannot discover it."""
    configured = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
    if configured or sys.platform != "darwin":
        return configured
    for candidate in (
        Path("/opt/homebrew/lib/libespeak-ng.dylib"),
        Path("/usr/local/lib/libespeak-ng.dylib"),
    ):
        if candidate.is_file():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(candidate)
            return str(candidate)
    return None


@dataclass(frozen=True)
class SourceItem:
    utterance_id: str
    speaker_id: str
    audio_path: str
    gender: str = "u"


def _validate_id(value: str, field: str) -> str:
    if not ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field} must match {ID_PATTERN.pattern!r}; received {value!r}"
        )
    return value


def _item_from_dict(data: dict, line_number: int | None = None) -> SourceItem:
    where = f" on manifest line {line_number}" if line_number is not None else ""
    if "audio_path" not in data:
        raise ValueError(f"Missing audio_path{where}")
    audio_path = Path(data["audio_path"]).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist{where}: {audio_path}")

    utterance_id = str(data.get("utterance_id") or audio_path.stem)
    speaker_id = str(data.get("speaker_id") or utterance_id)
    gender = str(data.get("gender") or "u").lower()
    if gender not in {"f", "m", "u"}:
        raise ValueError(f"gender must be f, m, or u{where}; received {gender!r}")
    return SourceItem(
        utterance_id=_validate_id(utterance_id, "utterance_id"),
        speaker_id=_validate_id(speaker_id, "speaker_id"),
        audio_path=str(audio_path),
        gender=gender,
    )


def load_items(
    manifest: Path | None,
    audio: Path | None,
    utterance_id: str | None,
    speaker_id: str | None,
    gender: str,
) -> list[SourceItem]:
    if (manifest is None) == (audio is None):
        raise ValueError("Specify exactly one of --manifest or --audio")

    if manifest is not None:
        items = []
        with manifest.expanduser().resolve().open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on manifest line {line_number}: {error}"
                    ) from error
                items.append(_item_from_dict(data, line_number))
    else:
        assert audio is not None
        data = {
            "audio_path": str(audio),
            "utterance_id": utterance_id,
            "speaker_id": speaker_id,
            "gender": gender,
        }
        items = [_item_from_dict(data)]

    if not items:
        raise ValueError("Input contains no utterances")
    utterance_ids = [item.utterance_id for item in items]
    duplicates = sorted({value for value in utterance_ids if utterance_ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate utterance_id values: {', '.join(duplicates)}")
    return items


def normalize_audio(source: Path, destination: Path, sample_rate: int = 16000) -> None:
    audio, source_rate = sf.read(source, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if not np.isfinite(mono).all():
        raise ValueError(f"Audio contains NaN or Inf values: {source}")
    if mono.size == 0:
        raise ValueError(f"Audio is empty: {source}")
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, source_rate // divisor)
    peak = float(np.max(np.abs(mono)))
    if peak > 1.0:
        mono = mono / peak
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, mono, sample_rate, subtype="PCM_16")


def prepare_normalized_audio(
    items: Iterable[SourceItem], output_dir: Path, sample_rate: int = 16000
) -> dict[str, Path]:
    normalized = {}
    for item in items:
        destination = output_dir / "normalized" / f"{item.utterance_id}.wav"
        normalize_audio(Path(item.audio_path), destination, sample_rate=sample_rate)
        normalized[item.utterance_id] = destination.resolve()
    return normalized


def prepare_kaldi_data(
    items: Iterable[SourceItem], normalized: dict[str, Path], data_dir: Path
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    item_list = list(items)
    speaker_to_utterances: dict[str, list[str]] = {}
    speaker_to_gender = {}
    for item in item_list:
        speaker_to_utterances.setdefault(item.speaker_id, []).append(item.utterance_id)
        existing = speaker_to_gender.setdefault(item.speaker_id, item.gender)
        if existing != item.gender:
            raise ValueError(f"Conflicting gender values for speaker {item.speaker_id}")

    files = {
        "wav.scp": [
            f"{item.utterance_id} {normalized[item.utterance_id]}\n" for item in item_list
        ],
        "utt2spk": [
            f"{item.utterance_id} {item.speaker_id}\n" for item in item_list
        ],
        "spk2utt": [
            f"{speaker} {' '.join(utterances)}\n"
            for speaker, utterances in sorted(speaker_to_utterances.items())
        ],
        "spk2gender": [
            f"{speaker} {speaker_to_gender[speaker]}\n"
            for speaker in sorted(speaker_to_gender)
        ],
    }
    for name, lines in files.items():
        (data_dir / name).write_text("".join(lines), encoding="utf-8")


def input_fingerprint(items: Iterable[SourceItem], settings: dict) -> str:
    payload = {
        "items": [asdict(item) for item in items],
        "settings": settings,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def audio_info(path: Path) -> dict:
    info = sf.info(path)
    return {
        "sample_rate": info.samplerate,
        "num_frames": info.frames,
        "duration_seconds": info.frames / info.samplerate,
        "channels": info.channels,
    }


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
