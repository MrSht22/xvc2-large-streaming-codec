#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mcadams import anonymize_file, coefficient_for_identity
from model_assets import REQUIRED_STTTS_MODELS
from sa_common import (
    SourceItem,
    audio_info,
    configure_espeak_library,
    input_fingerprint,
    load_items,
    prepare_kaldi_data,
    prepare_normalized_audio,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_VENDOR = ROOT / "third_party" / "Voice-Privacy-Challenge-2026"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate normalized source/SA waveform pairs with a citable VoicePrivacy method."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="JSONL with audio_path and optional IDs")
    source.add_argument("--audio", type=Path, help="Single input audio file")
    parser.add_argument("--utterance-id")
    parser.add_argument("--speaker-id")
    parser.add_argument("--gender", default="u", choices=("f", "m", "u"))
    parser.add_argument("--backend", required=True, choices=("mcadams", "sttts"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anonymization-level", choices=("spk", "utt"), default="spk")
    parser.add_argument("--jobs", type=int, default=1, help="McAdams worker count")
    parser.add_argument("--gpus", default="0", help="STTTS visible GPU IDs, or cpu")
    parser.add_argument("--seed", type=int, default=2026, help="STTTS pseudo-speaker seed")
    parser.add_argument("--models-dir", type=Path, help="Extracted STTTS v2.0 model directory")
    parser.add_argument("--vendor-dir", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs, source snapshot, and model layout without inference",
    )
    return parser


def validate_vendor(vendor_dir: Path) -> Path:
    vendor_dir = vendor_dir.expanduser().resolve()
    required = (
        vendor_dir / "LICENSE",
        vendor_dir / "anonymization" / "pipelines" / "sttts" / "sttts_pipeline.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing vendored VoicePrivacy files: " + ", ".join(missing))
    return vendor_dir


def validate_models(models_dir: Path | None) -> Path:
    if models_dir is None:
        raise ValueError("--models-dir is required for the sttts backend")
    models_dir = models_dir.expanduser().resolve()
    missing = [name for name in REQUIRED_STTTS_MODELS if not (models_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"STTTS model directory is incomplete ({models_dir}); missing: " + ", ".join(missing)
        )
    return models_dir


def _mcadams_job(arguments: tuple[Path, Path, float]) -> None:
    anonymize_file(*arguments)


def run_mcadams(
    items: list[SourceItem],
    normalized: dict[str, Path],
    output_dir: Path,
    level: str,
    jobs: int,
) -> tuple[dict[str, Path], dict[str, float]]:
    if jobs < 1:
        raise ValueError("--jobs must be at least 1")
    outputs = {}
    coefficients = {}
    work = []
    for item in items:
        identity = item.speaker_id if level == "spk" else item.utterance_id
        coefficient = coefficient_for_identity(identity)
        destination = (output_dir / "anonymized" / f"{item.utterance_id}.wav").resolve()
        outputs[item.utterance_id] = destination
        coefficients[item.utterance_id] = coefficient
        work.append((normalized[item.utterance_id], destination, coefficient))
    if jobs == 1:
        for arguments in work:
            _mcadams_job(arguments)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            list(executor.map(_mcadams_job, work))
    return outputs, coefficients


def build_sttts_config(
    models_dir: Path,
    runtime_vectors: Path,
    output_dir: Path,
    dataset_name: str,
    level: str,
) -> dict:
    intermediate = (output_dir / "work" / "sttts_intermediate").resolve()
    level_key = "anon_level_spk" if level == "spk" else "anon_level_utt"
    other_key = "anon_level_utt" if level == "spk" else "anon_level_spk"
    return {
        "data_dir": (output_dir / "work" / "kaldi").resolve(),
        "results_dir": Path("wav"),
        "models_dir": models_dir,
        "vectors_dir": (intermediate / "original_speaker_embeddings"),
        "save_intermediate": True,
        "save_output": True,
        "intermediate_dir": intermediate,
        "anon_suffix": "_sttts",
        "modules": {
            "asr": {
                "recognizer": "ims",
                "force_compute_recognition": False,
                "model_path": models_dir / "asr" / "asr_branchformer_tts-phn_en.zip",
                "ctc_weight": 0.2,
                "utt_start_token": "~",
                "utt_end_token": "~#",
                "results_path": intermediate / "transcription" / "asr_branchformer_tts-phn_en",
            },
            "speaker_embeddings": {
                "anonymizer": "ims",
                "force_compute_extraction": False,
                "force_compute_anonymization": False,
                "vec_type": "style-embed",
                "emb_model_path": models_dir / "tts" / "Embedding" / "embedding_function.pt",
                "anon_settings": {
                    "anon_method": "gan",
                    "vectors_file": runtime_vectors,
                    "gan_model_path": models_dir
                    / "anonymization"
                    / "gan_style-embed"
                    / "style-embed_wgan.pt",
                    "num_sampled": 5000,
                    "sim_threshold": 0.7,
                    "save_intermediate": True,
                },
                "extraction_results_path": intermediate
                / "original_speaker_embeddings"
                / "style-embed",
                "anon_results_path": intermediate
                / "anon_speaker_embeddings"
                / "style-embed",
                level_key: [dataset_name],
                other_key: [],
            },
            # No prosody anonymizer: durations, pitch, and energy are cloned from input.
            "prosody": {
                "extractor_type": "ims",
                "force_compute_extraction": False,
                "aligner_model_path": models_dir / "tts" / "Aligner" / "aligner.pt",
                "extraction_results_path": intermediate / "original_prosody" / "ims_extractor",
            },
            "tts": {
                "synthesizer": "ims",
                "force_compute_synthesis": False,
                "fastspeech_path": models_dir
                / "tts"
                / "FastSpeech2_Multi"
                / "prosody_cloning.pt",
                "hifigan_path": models_dir / "tts" / "HiFiGAN_combined" / "best.pt",
                "embeddings_path": models_dir / "tts" / "Embedding" / "embedding_function.pt",
                "output_sr": 16000,
                "results_path": intermediate / "anon_speech" / "ims_sttts_pc",
            },
        },
    }


def run_sttts(
    items: list[SourceItem],
    output_dir: Path,
    models_dir: Path,
    vendor_dir: Path,
    level: str,
    gpus: str,
    seed: int,
) -> dict[str, Path]:
    configure_espeak_library()
    if gpus.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    sys.path.insert(0, str(vendor_dir))
    import numpy as np
    import torch
    import torchaudio
    from silero_vad import (
        VADIterator,
        collect_chunks,
        get_speech_timestamps,
        load_silero_vad,
        read_audio,
        save_audio,
    )

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile", "sox", "ffmpeg"]

    # The upstream pipeline calls torch.hub for Silero VAD. Route that exact
    # request to the installed package so inference remains offline.
    original_hub_load = torch.hub.load

    def offline_hub_load(repo_or_dir, model, *hub_args, **hub_kwargs):
        if repo_or_dir == "snakers4/silero-vad" and model == "silero_vad":
            vad_model = load_silero_vad(onnx=hub_kwargs.get("onnx", False))
            utilities = (
                get_speech_timestamps,
                save_audio,
                read_audio,
                VADIterator,
                collect_chunks,
            )
            return vad_model, utilities
        return original_hub_load(repo_or_dir, model, *hub_args, **hub_kwargs)

    torch.hub.load = offline_hub_load

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        visible_count = len([value for value in gpus.split(",") if value.strip()])
        devices = [torch.device(f"cuda:{index}") for index in range(visible_count)]
    else:
        devices = [torch.device("cpu")]

    dataset_name = "sa_input"
    runtime_model_dir = output_dir / "work" / "model_runtime"
    runtime_model_dir.mkdir(parents=True, exist_ok=True)
    runtime_vectors = runtime_model_dir / "style-embed_wgan.pt"

    config = build_sttts_config(
        models_dir=models_dir,
        runtime_vectors=runtime_vectors,
        output_dir=output_dir,
        dataset_name=dataset_name,
        level=level,
    )
    from anonymization.pipelines.sttts import STTTSPipeline

    pipeline = STTTSPipeline(config=config, force_compute=False, devices=devices)
    results = pipeline.run_anonymization_pipeline(
        {dataset_name: output_dir / "work" / "kaldi" / dataset_name}
    )
    outputs = {key: Path(value).resolve() for key, value in results[dataset_name].items()}
    missing = [item.utterance_id for item in items if item.utterance_id not in outputs]
    if missing:
        raise RuntimeError("STTTS did not produce output for: " + ", ".join(missing))
    return outputs


def make_pair_rows(
    items: list[SourceItem],
    normalized: dict[str, Path],
    anonymized: dict[str, Path],
    backend: str,
    level: str,
    coefficients: dict[str, float] | None = None,
) -> list[dict]:
    rows = []
    for item in items:
        source_info = audio_info(normalized[item.utterance_id])
        anon_info = audio_info(anonymized[item.utterance_id])
        row = {
            "utterance_id": item.utterance_id,
            "speaker_id": item.speaker_id,
            "original_audio_path": item.audio_path,
            "normalized_source_audio_path": str(normalized[item.utterance_id]),
            "anonymized_audio_path": str(anonymized[item.utterance_id]),
            "backend": backend,
            "anonymization_level": level,
            "source_audio": source_info,
            "anonymized_audio": anon_info,
        }
        if coefficients is not None:
            row["mcadams_coefficient"] = coefficients[item.utterance_id]
        rows.append(row)
    return rows


def main() -> None:
    args = build_parser().parse_args()
    items = load_items(
        manifest=args.manifest,
        audio=args.audio,
        utterance_id=args.utterance_id,
        speaker_id=args.speaker_id,
        gender=args.gender,
    )
    output_dir = args.output_dir.expanduser().resolve()
    settings = {
        "backend": args.backend,
        "anonymization_level": args.anonymization_level,
        "seed": args.seed if args.backend == "sttts" else None,
        "sample_rate": 16000,
    }
    fingerprint = input_fingerprint(items, settings)
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("input_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Output directory belongs to a different run: {output_dir}. Use a new directory."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    vendor_dir = validate_vendor(args.vendor_dir)
    models_dir = validate_models(args.models_dir) if args.backend == "sttts" else None
    metadata = {
        "input_fingerprint": fingerprint,
        "settings": settings,
        "utterance_count": len(items),
        "vendor_commit": "f59d6282fddd6c09d5dba5c261a180334ce2797d",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    if args.check_only:
        print(f"input_validation=PASS,utterances={len(items)}")
        print(f"vendor_validation=PASS,path={vendor_dir}")
        if models_dir is not None:
            print(f"model_layout_validation=PASS,path={models_dir}")
        print("check_only=PASS")
        return

    normalized = prepare_normalized_audio(items, output_dir)
    kaldi_dir = output_dir / "work" / "kaldi" / "sa_input"
    prepare_kaldi_data(items, normalized, kaldi_dir)

    if args.backend == "mcadams":
        anonymized, coefficients = run_mcadams(
            items,
            normalized,
            output_dir,
            args.anonymization_level,
            args.jobs,
        )
    else:
        assert models_dir is not None
        anonymized = run_sttts(
            items,
            output_dir,
            models_dir,
            vendor_dir,
            args.anonymization_level,
            args.gpus,
            args.seed,
        )
        coefficients = None

    rows = make_pair_rows(
        items,
        normalized,
        anonymized,
        args.backend,
        args.anonymization_level,
        coefficients,
    )
    manifest_path = output_dir / "pairs.jsonl"
    write_jsonl(rows, manifest_path)
    print(f"backend={args.backend}")
    print(f"utterances={len(rows)}")
    print(f"pair_manifest={manifest_path}")
    print("anonymization=PASS")


if __name__ == "__main__":
    main()
