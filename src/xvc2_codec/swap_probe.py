"""Frozen single-branch swaps for original / same-text Seed-VC audio pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio

from .cache import temporal_cache
from .config import load_config
from .data import read_jsonl
from .model import LargeStreamingCodec
from .preprocess import load_student, write_jsonl


SWAPS = {"BAA": "z_inv", "ABA": "z_dyn", "AAB": "z_edit"}


def normalize_pairs(rows: list[dict]) -> list[dict]:
    """Accept the Seed-VC export directly, retaining the original experiment labels."""
    normalized = []
    for original in rows:
        row = dict(original)
        if "source_audio_path" in row and "converted_audio_path" in row:
            row.setdefault("pair_id", f"seedvc_{int(row['index']):03d}")
            row.setdefault("donor_method", "seed-vc")
            row.setdefault("a", {"audio_path": row["source_audio_path"]})
            row.setdefault("b", {"audio_path": row["converted_audio_path"]})
            row.setdefault("b_identity_reference_path", row["target_reference_audio_path"])
            candidates = [
                candidate["source_audio_path"]
                for candidate in rows
                if candidate.get("source_speaker") == row.get("source_speaker")
                and candidate.get("source_audio_path")
                and candidate["source_audio_path"] != row["source_audio_path"]
            ]
            if candidates:
                # Prefer a different chapter, then select deterministically within that pool.
                candidates.sort(
                    key=lambda path: (
                        Path(path).parent == Path(row["source_audio_path"]).parent,
                        path,
                    )
                )
                row.setdefault("a_identity_reference_path", candidates[0])
        text = str(row.get("text") or row.get("transcript") or "").strip()
        if not text:
            source = Path(row["a"]["audio_path"])
            paths = [source.with_suffix(".normalized.txt"), source.with_suffix(".original.txt")]
            for path in paths:
                if path.is_file():
                    text = path.read_text(encoding="utf-8-sig").strip()
                    if text:
                        row["text_path"] = str(path)
                        break
            if not text:
                raise FileNotFoundError(
                    f"No LibriTTS transcript for {source}; expected {paths}. "
                    "Keep dataset sidecar text files or provide text/transcript in JSONL."
                )
        row["text"] = text
        normalized.append(row)
    return normalized


def load_codec(config_path: Path, checkpoint: Path, weights: str, device: torch.device):
    config = load_config(config_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["config"]["model"] != config.model.to_dict():
        raise ValueError("Checkpoint architecture differs from --config")
    state = payload["ema"]["shadow"] if weights == "ema" else payload["model"]
    state = {
        key.removeprefix("codec."): value
        for key, value in state.items()
        if key.startswith("codec.")
    }
    codec = LargeStreamingCodec(config.model)
    codec.load_state_dict(state, strict=True)
    return codec.eval().requires_grad_(False).to(device), int(payload["step"])


def load_view(view: dict, codec: LargeStreamingCodec, student, device: torch.device):
    waveform, rate = torchaudio.load(view["audio_path"])
    waveform = waveform.mean(0, keepdim=True)
    if rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, rate, 16_000)
    hidden_cache = temporal_cache(view, "student_hidden", "student_hidden")
    if hidden_cache is not None:
        hidden = hidden_cache.read()
    elif student is not None:
        with torch.inference_mode():
            output = student(waveform.to(device), torch.tensor([waveform.shape[-1]], device=device))
        hidden = output["hidden_states"][0, : int(output["output_lengths"][0])].cpu()
    else:
        raise ValueError("Each a/b view needs Student hidden or --student-checkpoint")
    frames = waveform.shape[-1] // codec.config.hop_length
    # Only discard a final partial hop, never silently truncate words to a short cache.
    if hidden.ndim != 2 or hidden.shape[1] != codec.config.student_dim or hidden.shape[0] < frames:
        raise ValueError(f"Invalid/short Student hidden for {view['audio_path']}")
    if frames < 2 or not torch.isfinite(hidden).all() or not torch.isfinite(waveform).all():
        raise ValueError(f"Invalid/short input: {view['audio_path']}")
    return waveform[:, : frames * codec.config.hop_length], hidden[:frames].float()


def frame_mapping(a_frames: int, b_frames: int, map_path: str | None) -> torch.Tensor:
    mapping = (
        torch.from_numpy(np.load(map_path)).float()
        if map_path
        else torch.linspace(0, b_frames - 1, a_frames)
    )
    if (
        mapping.shape != (a_frames,)
        or not torch.isfinite(mapping).all()
        or mapping.min() < 0
        or mapping.max() > b_frames - 1
        or (mapping[1:] < mapping[:-1]).any()
    ):
        raise ValueError("Frame map must be finite, monotonic, and within B; one index per A frame")
    return mapping


def map_latent(values: torch.Tensor, mapping: torch.Tensor, name: str) -> torch.Tensor:
    values = values.transpose(1, 2) if name == "z_edit" else values
    mapping = mapping.to(values.device)
    left, right = mapping.floor().long(), mapping.ceil().long()
    fraction = (mapping - left).reshape(1, -1, 1)
    result = values[:, left] * (1 - fraction) + values[:, right] * fraction
    return result.transpose(1, 2) if name == "z_edit" else result


@torch.inference_mode()
def decode_swaps(codec, a_wave, a_hidden, b_wave, b_hidden, mapping):
    def encode(wave, hidden):
        return {
            **codec.keep_head(hidden.unsqueeze(0)),
            "z_edit": codec.acoustic_encoder(wave.unsqueeze(0)),
        }

    def decode(latents):
        return (
            codec.decoder(codec.fusion(*(latents[name] for name in ("z_inv", "z_dyn", "z_edit"))))[
                0
            ]
            .float()
            .cpu()
        )

    a, b = encode(a_wave, a_hidden), encode(b_wave, b_hidden)
    mapped = {name: map_latent(b[name], mapping, name) for name in ("z_inv", "z_dyn", "z_edit")}
    outputs = {"AAA": decode(a), "BBB": decode(b), "BBB_mapped": decode(mapped)}
    for branch, name in SWAPS.items():
        outputs[branch] = decode({**a, name: mapped[name]})
    return outputs


def generate(args: argparse.Namespace) -> None:
    rows = normalize_pairs(read_jsonl(args.manifest))
    if args.max_pairs is not None:
        rows = rows[: args.max_pairs]
    if not rows or len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("Need non-empty pairs with unique pair_id")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    codec, step = load_codec(args.config, args.checkpoint, args.weights, device)
    if codec.config.hop_length != 320:
        raise ValueError("This speech evaluation expects 16 kHz / 50 Hz (hop 320)")
    student = load_student(args.student_checkpoint, device) if args.student_checkpoint else None
    # A new directory avoids accidentally combining outputs from different experiments.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    request = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "checkpoint": str(args.checkpoint.resolve()),
        "step": step,
        "weights": args.weights,
        "config": str(args.config.resolve()),
        "student_checkpoint": str(args.student_checkpoint) if student is not None else None,
        "max_pairs": args.max_pairs,
        "sample_rate": 16_000,
        "swaps": SWAPS,
    }
    (args.output_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    write_jsonl(rows, args.output_dir / "pairs.jsonl")
    records = []
    for index, row in enumerate(rows, 1):
        pair_id = str(row["pair_id"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", pair_id) or pair_id in (".", ".."):
            raise ValueError("pair_id must be a safe filename")
        if row.get("donor_method") != "seed-vc" or not str(row.get("text", "")).strip():
            raise ValueError("Each pair requires donor_method=seed-vc and shared text")
        a_wave, a_hidden = load_view(row["a"], codec, student, device)
        b_wave, b_hidden = load_view(row["b"], codec, student, device)
        lag = int(row.get("alignment_lag_frames", 0))
        a_offset, b_offset = max(-lag, 0), max(lag, 0)
        if lag and not row.get("aligned_text"):
            raise ValueError("Lag cropping requires aligned_text for the retained content")
        a_wave, a_hidden = a_wave[:, a_offset * 320 :], a_hidden[a_offset:]
        b_wave, b_hidden = b_wave[:, b_offset * 320 :], b_hidden[b_offset:]
        if min(len(a_hidden), len(b_hidden)) < 2:
            raise ValueError("No usable frames after lag cropping")
        mapping = frame_mapping(len(a_hidden), len(b_hidden), row.get("donor_frame_map_path"))
        outputs = decode_swaps(
            codec,
            a_wave.to(device),
            a_hidden.to(device),
            b_wave.to(device),
            b_hidden.to(device),
            mapping,
        )
        outputs.update({"A": a_wave, "B": b_wave})
        folder = args.output_dir / "audio" / pair_id
        folder.mkdir(parents=True)
        np.save(folder / "donor_frame_map.npy", mapping.numpy())
        paths = {}
        for branch, waveform in outputs.items():
            expected = len(b_hidden) if branch in ("B", "BBB") else len(a_hidden)
            if waveform.shape != (1, expected * 320) or not torch.isfinite(waveform).all():
                raise ValueError(f"Invalid output {pair_id}/{branch}")
            path = folder / f"{branch}.wav"
            torchaudio.save(str(path), waveform, 16_000, encoding="PCM_F", bits_per_sample=32)
            paths[branch] = str(path.resolve())
        identity_a = row.get("a_identity_reference_path", paths["A"])
        identity_b = row.get("b_identity_reference_path", paths["B"])
        for branch, waveform in outputs.items():
            records.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "index",
                            "source_speaker",
                            "target_speaker",
                            "conversion",
                            "text_path",
                            "source_audio_path",
                            "converted_audio_path",
                            "target_reference_audio_path",
                        )
                        if key in row
                    },
                    "pair_id": pair_id,
                    "branch": branch,
                    "text": row.get("aligned_text") or row["text"],
                    "audio_path": paths[branch],
                    "a_identity_reference_path": str(Path(identity_a).resolve()),
                    "b_identity_reference_path": str(Path(identity_b).resolve()),
                    "identity_reference_kind": "provided"
                    if row.get("a_identity_reference_path") and row.get("b_identity_reference_path")
                    else "same-text fallback",
                    "alignment": "explicit_frame_map"
                    if row.get("donor_frame_map_path")
                    else "normalized_time",
                    "alignment_lag_frames": lag,
                    "a_frames": len(a_hidden),
                    "b_frames": len(b_hidden),
                    "rms": float(waveform.square().mean().sqrt()),
                    "peak": float(waveform.abs().max()),
                    "clip_fraction": float((waveform.abs() >= 1).float().mean()),
                }
            )
        print(f"codec_swap_generate={index}/{len(rows)}", flush=True)
    write_jsonl(records, args.output_dir / "branches.jsonl")
    print("codec_swap_generate=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate", help="Frozen BAA / ABA / AAB inference")
    generate_parser.add_argument("--manifest", type=Path, required=True)
    generate_parser.add_argument("--config", type=Path, required=True)
    generate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser.add_argument("--student-checkpoint", type=Path)
    generate_parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    generate_parser.add_argument("--device", default="auto")
    generate_parser.add_argument("--max-pairs", type=int)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser = commands.add_parser("evaluate", help="Metrics and 3x3 listening report")
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument("--evaluator-dir", type=Path, required=True)
    evaluate_parser.add_argument("--models-dir", type=Path, required=True)
    evaluate_parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate_parser.add_argument("--compute-type", default="float16")
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_parser.add_argument("--examples-per-cell", type=int, default=3)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if args.max_pairs is not None and args.max_pairs <= 0:
            parser.error("--max-pairs must be positive")
        generate(args)
    else:
        if min(args.batch_size, args.examples_per_cell) <= 0:
            parser.error("batch size and examples per cell must be positive")
        from .swap_report import evaluate

        evaluate(args)


if __name__ == "__main__":
    main()
