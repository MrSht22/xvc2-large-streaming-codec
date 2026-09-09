from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .data import read_jsonl


SAMPLE_RATE = 16_000
HIDDEN_DIM = 768
PHONE_DIM = 40
SPEAKER_DIM = 128
CACHE_DTYPE = "float16"


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    os.replace(temporary, path)


def stable_item_id(role: str, utterance_id: str, audio_path: str) -> str:
    digest = hashlib.sha256(str(Path(audio_path).expanduser()).encode()).hexdigest()[:16]
    return f"{role}:{utterance_id}:{digest}"


def pair_paths(row: dict[str, Any]) -> tuple[str, str]:
    source = row.get("normalized_source_audio_path")
    anonymized = row.get("anonymized_audio_path")
    if not source or not anonymized:
        raise ValueError(
            "SA pair rows require normalized_source_audio_path and anonymized_audio_path"
        )
    return str(source), str(anonymized)


def duration_seconds(row: dict[str, Any], path: str, nested: str | None = None) -> float:
    metadata = row.get(nested, {}) if nested else row
    if isinstance(metadata, dict) and metadata.get("duration_seconds") is not None:
        return float(metadata["duration_seconds"])
    if not nested and row.get("duration_seconds") is not None:
        return float(row["duration_seconds"])
    info = torchaudio.info(path)
    return info.num_frames / info.sample_rate


def inventory_rows(
    source_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(role: str, row: dict[str, Any], path: str, nested: str | None = None) -> None:
        utterance_id = str(row["utterance_id"])
        item_id = stable_item_id(role, utterance_id, path)
        if item_id in seen:
            return
        seen.add(item_id)
        duration = duration_seconds(row, path, nested)
        inventory.append(
            {
                "item_id": item_id,
                "utterance_id": utterance_id,
                "speaker_id": str(row["speaker_id"]),
                "role": role,
                "audio_path": str(Path(path).expanduser().resolve()),
                "duration_seconds": duration,
                "estimated_frames": max(1, round(duration * 50)),
            }
        )

    for row in source_rows:
        add("source", row, str(row["audio_path"]))
    for row in pair_rows:
        source, anonymized = pair_paths(row)
        add("pair-source", row, source, "source_audio")
        add("pair-sa", row, anonymized, "anonymized_audio")
    return inventory


def pair_duration_report(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = np.asarray(
        [duration_seconds(row, pair_paths(row)[0], "source_audio") for row in pair_rows],
        dtype=np.float64,
    )
    anonymized = np.asarray(
        [duration_seconds(row, pair_paths(row)[1], "anonymized_audio") for row in pair_rows],
        dtype=np.float64,
    )
    absolute = np.abs(anonymized - source)
    relative = absolute / np.maximum(source, 1e-9)
    signed = anonymized - source
    quantiles = (0.5, 0.9, 0.95, 0.99)
    return {
        "status": "PASS" if float(np.quantile(relative, 0.95)) <= 0.01 else "NEEDS_ATTENTION",
        "rows": len(pair_rows),
        "source_hours": float(source.sum() / 3600),
        "anonymized_hours": float(anonymized.sum() / 3600),
        "mean_signed_drift_ms": float(signed.mean() * 1000),
        "mean_absolute_drift_ms": float(absolute.mean() * 1000),
        "absolute_drift_ms_quantiles": {
            f"p{round(quantile * 100)}": float(np.quantile(absolute, quantile) * 1000)
            for quantile in quantiles
        },
        "relative_drift_percent_quantiles": {
            f"p{round(quantile * 100)}": float(np.quantile(relative, quantile) * 100)
            for quantile in quantiles
        },
        "fraction_within_one_50hz_frame": float((absolute <= 0.02).mean()),
        "fraction_within_five_percent": float((relative <= 0.05).mean()),
    }


def write_speaker_references(
    source_rows: list[dict[str, Any]], output_dir: Path, references_per_speaker: int
) -> int:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[str(row["speaker_id"])].append(row)
    selected = []
    for speaker_id in sorted(grouped):
        candidates = sorted(grouped[speaker_id], key=lambda row: str(row["utterance_id"]))
        selected.extend(candidates[:references_per_speaker])
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_scp, utt2spk = [], []
    speakers: dict[str, list[str]] = defaultdict(list)
    for row in selected:
        audio_path = str(Path(row["audio_path"]).expanduser().resolve())
        reference_hash = hashlib.sha256(
            f"{row['utterance_id']}\0{audio_path}".encode()
        ).hexdigest()[:20]
        utterance_id = f"ref-{reference_hash}"
        speaker_id = str(row["speaker_id"])
        wav_scp.append(f"{utterance_id} {audio_path}\n")
        utt2spk.append(f"{utterance_id} {speaker_id}\n")
        speakers[speaker_id].append(utterance_id)
    (output_dir / "wav.scp").write_text("".join(wav_scp), encoding="utf-8")
    (output_dir / "utt2spk").write_text("".join(utt2spk), encoding="utf-8")
    (output_dir / "spk2utt").write_text(
        "".join(f"{speaker} {' '.join(items)}\n" for speaker, items in speakers.items()),
        encoding="utf-8",
    )
    (output_dir / "spk2gender").write_text(
        "".join(f"{speaker} u\n" for speaker in speakers), encoding="utf-8"
    )
    return len(selected)


def command_plan(args: argparse.Namespace) -> None:
    source_rows = read_jsonl(args.source_manifest)
    pair_rows = read_jsonl(args.pair_manifest)
    inventory = inventory_rows(source_rows, pair_rows)
    pair_durations = pair_duration_report(pair_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(inventory, args.output_dir / "inventory.jsonl")
    reference_count = write_speaker_references(
        source_rows, args.output_dir / "speaker_references", args.references_per_speaker
    )
    total_frames = sum(row["estimated_frames"] for row in inventory)
    cache_bytes = total_frames * (HIDDEN_DIM + PHONE_DIM) * np.dtype(np.float16).itemsize
    required_bytes = int(cache_bytes * args.space_safety_factor)
    free_bytes = shutil.disk_usage(args.output_dir).free
    hours_by_role = {
        role: sum(row["duration_seconds"] for row in inventory if row["role"] == role) / 3600
        for role in ("source", "pair-source", "pair-sa")
    }
    report = {
        "status": "PASS" if free_bytes >= required_bytes else "FAIL",
        "source_rows": len(source_rows),
        "pair_rows": len(pair_rows),
        "pair_duration_alignment": pair_durations,
        "inventory_items": len(inventory),
        "inventory_hours": sum(row["duration_seconds"] for row in inventory) / 3600,
        "inventory_hours_by_role": hours_by_role,
        "source_speakers": len({str(row["speaker_id"]) for row in source_rows}),
        "speaker_reference_items": reference_count,
        "estimated_cache_bytes": cache_bytes,
        "estimated_cache_tib": cache_bytes / 1024**4,
        "required_free_bytes": required_bytes,
        "required_free_tib": required_bytes / 1024**4,
        "available_free_bytes": free_bytes,
        "available_free_tib": free_bytes / 1024**4,
        "space_safety_factor": args.space_safety_factor,
        "inventory_path": str((args.output_dir / "inventory.jsonl").resolve()),
        "speaker_reference_dir": str((args.output_dir / "speaker_references").resolve()),
    }
    (args.output_dir / "plan.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    print(f"codec_preprocess_plan={report['status']}", flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


class AudioDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, sample_rate = torchaudio.load(row["audio_path"])
        if waveform.shape[0] != 1:
            waveform = waveform.mean(0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        return {"row": row, "waveform": waveform[0]}


def collate_audio(items: list[dict[str, Any]]) -> dict[str, Any]:
    waveforms = [item["waveform"] for item in items]
    return {
        "rows": [item["row"] for item in items],
        "waveforms": pad_sequence(waveforms, batch_first=True),
        "lengths": torch.tensor([waveform.numel() for waveform in waveforms]),
    }


def partition_shards(
    rows: list[dict[str, Any]], rank: int, world_size: int, maximum_frames: int
) -> list[list[dict[str, Any]]]:
    assigned = rows[rank::world_size]
    shards: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_frames = 0
    for row in assigned:
        frames = int(row["estimated_frames"])
        if current and current_frames + frames > maximum_frames:
            shards.append(current)
            current, current_frames = [], 0
        current.append(row)
        current_frames += frames
    if current:
        shards.append(current)
    return shards


def shard_fingerprint(rows: list[dict[str, Any]], checkpoint: Path) -> str:
    payload = [(row["item_id"], row["audio_path"]) for row in rows]
    stat = checkpoint.stat()
    checkpoint_identity = [str(checkpoint.resolve()), stat.st_size, stat.st_mtime_ns]
    encoded = json.dumps([checkpoint_identity, payload], separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def extraction_batches(
    rows: list[dict[str, Any]], batch_size: int, maximum_batch_samples: int
) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    maximum_samples = 0
    for index, row in enumerate(rows):
        samples = max(1, round(float(row["duration_seconds"]) * SAMPLE_RATE))
        proposed_maximum = max(maximum_samples, samples)
        if current and (
            len(current) >= batch_size
            or proposed_maximum * (len(current) + 1) > maximum_batch_samples
        ):
            batches.append(current)
            current, maximum_samples = [], 0
        current.append(index)
        maximum_samples = max(maximum_samples, samples)
    if current:
        batches.append(current)
    return batches


def load_student(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    try:
        from xvc2_student.config import ModelConfig
        from xvc2_student.model import StreamingPhoneEncoder
    except ImportError as error:
        raise RuntimeError(
            "xvc2_student is unavailable; add xvc2-large-streaming-student/src to PYTHONPATH"
        ) from error
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = StreamingPhoneEncoder(ModelConfig.from_dict(payload["config"]["model"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False).to(device)
    if model.config.model_dim != HIDDEN_DIM or model.config.vocab_size != PHONE_DIM:
        raise ValueError("Student checkpoint dimensions do not match Codec cache schema")
    return model


def extract_shard(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    output_dir: Path,
    rank: int,
    shard_index: int,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    maximum_batch_samples: int,
    num_workers: int,
) -> int:
    stem = f"r{rank:02d}-s{shard_index:05d}"
    hidden_path = output_dir / f"hidden-{stem}.bin"
    phone_path = output_dir / f"phone-{stem}.bin"
    index_path = output_dir / f"index-{stem}.jsonl"
    complete_path = output_dir / f"complete-{stem}.json"
    fingerprint = shard_fingerprint(rows, checkpoint)
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Completed shard fingerprint differs: {complete_path}")
        print(f"rank={rank} shard={shard_index} status=SKIP_COMPLETE", flush=True)
        return int(complete["items"])

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp-{os.getpid()}"
    hidden_tmp, phone_tmp = Path(str(hidden_path) + suffix), Path(str(phone_path) + suffix)
    index_tmp = Path(str(index_path) + suffix)
    rows = sorted(rows, key=lambda row: (int(row["estimated_frames"]), row["item_id"]))
    loader = DataLoader(
        AudioDataset(rows),
        batch_sampler=extraction_batches(rows, batch_size, maximum_batch_samples),
        num_workers=num_workers,
        collate_fn=collate_audio,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    offset = 0
    processed = 0
    processed_seconds = 0.0
    started = time.monotonic()
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        with (
            hidden_tmp.open("wb") as hidden_stream,
            phone_tmp.open("wb") as phone_stream,
            index_tmp.open("w", encoding="utf-8") as index_stream,
        ):
            with torch.inference_mode():
                for batch in loader:
                    waveforms = batch["waveforms"].to(device, non_blocking=True)
                    lengths = batch["lengths"].to(device, non_blocking=True)
                    with torch.autocast(
                        device.type, dtype=amp_dtype, enabled=device.type == "cuda"
                    ):
                        outputs = model(waveforms, lengths)
                    for item_index, row in enumerate(batch["rows"]):
                        frames = int(outputs["output_lengths"][item_index])
                        hidden = outputs["hidden_states"][item_index, :frames].half().cpu().numpy()
                        phone = outputs["phone_logits"][item_index, :frames].half().cpu().numpy()
                        hidden.tofile(hidden_stream)
                        phone.tofile(phone_stream)
                        cache = {
                            **row,
                            "student_hidden_path": str(hidden_path.resolve()),
                            "student_hidden_offset_frames": offset,
                            "student_hidden_frames": frames,
                            "student_hidden_dim": HIDDEN_DIM,
                            "student_hidden_dtype": CACHE_DTYPE,
                            "phone_target_path": str(phone_path.resolve()),
                            "phone_target_offset_frames": offset,
                            "phone_target_frames": frames,
                            "phone_target_dim": PHONE_DIM,
                            "phone_target_dtype": CACHE_DTYPE,
                        }
                        index_stream.write(json.dumps(cache, sort_keys=True) + "\n")
                        offset += frames
                        processed += 1
                        processed_seconds += float(row["duration_seconds"])
                    if processed % 100 <= batch_size:
                        elapsed = max(time.monotonic() - started, 1e-6)
                        print(
                            f"rank={rank} shard={shard_index} items={processed}/{len(rows)} "
                            f"audio_hours={processed_seconds / 3600:.3f} "
                            f"items_per_second={processed / elapsed:.2f}",
                            flush=True,
                        )
        os.replace(hidden_tmp, hidden_path)
        os.replace(phone_tmp, phone_path)
        os.replace(index_tmp, index_path)
    finally:
        for path in (hidden_tmp, phone_tmp, index_tmp):
            path.unlink(missing_ok=True)
    complete = {"fingerprint": fingerprint, "items": processed, "frames": offset}
    temporary = Path(str(complete_path) + suffix)
    temporary.write_text(json.dumps(complete, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, complete_path)
    print(
        f"rank={rank} shard={shard_index} status=PASS items={processed} frames={offset}", flush=True
    )
    return processed


def command_extract_student(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required but unavailable")
    rows = read_jsonl(args.inventory)
    shards = partition_shards(rows, rank, world_size, args.shard_max_frames)
    model = load_student(args.student_checkpoint, device)
    processed = 0
    for shard_index, shard in enumerate(shards):
        processed += extract_shard(
            model,
            shard,
            args.output_dir,
            rank,
            shard_index,
            args.student_checkpoint,
            device,
            args.batch_size,
            round(args.max_batch_seconds * SAMPLE_RATE),
            args.num_workers,
        )
    print(f"codec_student_extraction=PASS,rank={rank},items={processed},shards={len(shards)}")


def load_audio(path: str, maximum_seconds: float | None = None) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] != 1:
        waveform = waveform.mean(0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    if maximum_seconds is not None:
        waveform = waveform[:, : round(maximum_seconds * SAMPLE_RATE)]
    return waveform


def lagged_frame_metrics(
    source: torch.Tensor, anonymized: torch.Tensor, maximum_lag: int
) -> dict[str, float | int]:
    source = torch.nn.functional.normalize(source.float(), dim=-1)
    anonymized = torch.nn.functional.normalize(anonymized.float(), dim=-1)
    scores = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            left, right = source[-lag:], anonymized[:lag]
        elif lag > 0:
            left, right = source[:-lag], anonymized[lag:]
        else:
            frames = min(source.shape[0], anonymized.shape[0])
            left, right = source[:frames], anonymized[:frames]
        frames = min(left.shape[0], right.shape[0])
        score = float((left[:frames] * right[:frames]).sum(-1).mean()) if frames else -1.0
        scores.append((score, lag))
    zero_score = scores[maximum_lag][0]
    best_score, best_lag = max(scores)
    return {
        "zero_lag_cosine": zero_score,
        "best_lag_cosine": best_score,
        "best_lag_frames": best_lag,
        "best_lag_improvement": best_score - zero_score,
    }


def quantile_report(values: list[float]) -> dict[str, float]:
    return {
        name: float(np.quantile(values, quantile))
        for name, quantile in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99))
    }


def command_probe_pair_alignment(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    rows = read_jsonl(args.pair_manifest)
    if args.max_items < len(rows):
        indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.max_items))
        rows = [rows[index] for index in indices]
    model = load_student(args.student_checkpoint, device)
    items = []
    previous_sa_hidden = None
    mismatched_cosines = []
    started = time.monotonic()
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            source_path, sa_path = pair_paths(row)
            source = load_audio(source_path, args.max_seconds)
            anonymized = load_audio(sa_path, args.max_seconds)
            waveforms = pad_sequence([source[0], anonymized[0]], batch_first=True).to(device)
            lengths = torch.tensor([source.shape[1], anonymized.shape[1]], device=device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                outputs = model(waveforms, lengths)
            source_frames, sa_frames = (int(value) for value in outputs["output_lengths"])
            source_hidden = outputs["hidden_states"][0, :source_frames]
            sa_hidden = outputs["hidden_states"][1, :sa_frames]
            metrics = lagged_frame_metrics(source_hidden, sa_hidden, args.max_lag_frames)
            source_phone = outputs["phone_logits"][0, :source_frames].argmax(-1)
            sa_phone = outputs["phone_logits"][1, :sa_frames].argmax(-1)
            zero_frames = min(source_phone.numel(), sa_phone.numel())
            zero_phone_agreement = float(
                (source_phone[:zero_frames] == sa_phone[:zero_frames]).float().mean()
            )
            lag = int(metrics["best_lag_frames"])
            if lag < 0:
                source_phone, sa_phone = source_phone[-lag:], sa_phone[:lag]
            elif lag > 0:
                source_phone, sa_phone = source_phone[:-lag], sa_phone[lag:]
            frames = min(source_phone.numel(), sa_phone.numel())
            phone_agreement = float((source_phone[:frames] == sa_phone[:frames]).float().mean())
            if previous_sa_hidden is not None:
                mismatch_frames = min(source_hidden.shape[0], previous_sa_hidden.shape[0])
                mismatch = torch.nn.functional.cosine_similarity(
                    source_hidden[:mismatch_frames].float(),
                    previous_sa_hidden[:mismatch_frames].float(),
                    dim=-1,
                ).mean()
                mismatched_cosines.append(float(mismatch))
            previous_sa_hidden = sa_hidden
            items.append(
                {
                    "utterance_id": str(row["utterance_id"]),
                    "source_frames": source_frames,
                    "sa_frames": sa_frames,
                    "phone_argmax_agreement_zero_lag": zero_phone_agreement,
                    "phone_argmax_agreement_at_best_lag": phone_agreement,
                    **metrics,
                }
            )
            if index % 10 == 0 or index == len(rows):
                print(
                    f"pair_alignment_probe={index}/{len(rows)},"
                    f"elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
    absolute_lags = [abs(int(item["best_lag_frames"])) for item in items]
    improvements = [float(item["best_lag_improvement"]) for item in items]
    within_one = sum(lag <= 1 for lag in absolute_lags) / len(absolute_lags)
    status = (
        "PASS"
        if within_one >= 0.9
        and float(np.quantile(absolute_lags, 0.95)) <= 2
        and float(np.quantile(improvements, 0.5)) <= 0.01
        else "NEEDS_ATTENTION"
    )
    report = {
        "status": status,
        "checkpoint": str(args.student_checkpoint),
        "items_sampled": len(items),
        "max_seconds": args.max_seconds,
        "max_lag_frames": args.max_lag_frames,
        "fraction_best_lag_within_one_frame": within_one,
        "absolute_best_lag_frames_quantiles": quantile_report(absolute_lags),
        "zero_lag_cosine_quantiles": quantile_report(
            [float(item["zero_lag_cosine"]) for item in items]
        ),
        "best_lag_cosine_quantiles": quantile_report(
            [float(item["best_lag_cosine"]) for item in items]
        ),
        "best_lag_improvement_quantiles": quantile_report(improvements),
        "phone_argmax_agreement_zero_lag_quantiles": quantile_report(
            [float(item["phone_argmax_agreement_zero_lag"]) for item in items]
        ),
        "phone_argmax_agreement_best_lag_quantiles": quantile_report(
            [float(item["phone_argmax_agreement_at_best_lag"]) for item in items]
        ),
        "phone_argmax_agreement_gain_quantiles": quantile_report(
            [
                float(item["phone_argmax_agreement_at_best_lag"])
                - float(item["phone_argmax_agreement_zero_lag"])
                for item in items
            ]
        ),
        "mismatched_pair_cosine_quantiles": quantile_report(mismatched_cosines),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "items"}
    print(json.dumps(summary, sort_keys=True), flush=True)
    print(f"codec_pair_alignment_probe={status}", flush=True)
    if status != "PASS":
        raise SystemExit(1)


def read_id2idx(path: Path) -> dict[str, int]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            identifier, index = line.split()
            result[identifier] = int(index)
    return result


def embedding_dirs(shard_dir: Path, level: str) -> tuple[Path, Path]:
    root = shard_dir / "work" / "sttts_intermediate"
    source = root / "original_speaker_embeddings" / "style-embed" / "sa_input"
    source = source / ("spk-level" if level == "spk" else "utt-level")
    anonymous = root / "anon_speaker_embeddings" / "style-embed" / "sa_input"
    return source, anonymous


def consolidate_sa_vectors(sa_run_dir: Path, pair_manifest: Path, output_dir: Path) -> int:
    metadata = json.loads((sa_run_dir / "parallel_run_metadata.json").read_text(encoding="utf-8"))
    level = str(metadata["anonymization_level"])
    pair_rows = read_jsonl(pair_manifest)
    requested = {str(row["utterance_id"]): row for row in pair_rows}
    utterance_to_shard = {}
    for path in sorted((sa_run_dir / "shard_manifests").glob("shard-*.jsonl")):
        shard_dir = sa_run_dir / "shards" / path.stem
        for row in read_jsonl(path):
            if str(row["utterance_id"]) in requested:
                utterance_to_shard[str(row["utterance_id"])] = shard_dir
    missing = sorted(set(requested) - set(utterance_to_shard))
    if missing:
        raise RuntimeError(f"No SA shard mapping for {len(missing)} utterances")
    stores: dict[Path, tuple[dict[str, int], torch.Tensor]] = {}
    vectors = []
    identifiers = []
    for row in pair_rows:
        utterance_id = str(row["utterance_id"])
        _, store_dir = embedding_dirs(utterance_to_shard[utterance_id], level)
        if store_dir not in stores:
            stores[store_dir] = (
                read_id2idx(store_dir / "id2idx"),
                torch.load(store_dir / "speaker_vectors.pt", map_location="cpu", weights_only=True),
            )
        mapping, store = stores[store_dir]
        identifier = str(row["speaker_id"] if level == "spk" else utterance_id)
        if identifier not in mapping:
            raise KeyError(f"No anonymous vector for {identifier!r} in {store_dir}")
        vectors.append(store[mapping[identifier]].float())
        identifiers.append(utterance_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = torch.stack(vectors)
    torch.save(matrix, output_dir / "sa_speaker_vectors.pt")
    (output_dir / "sa_id2idx").write_text(
        "".join(f"{identifier} {index}\n" for index, identifier in enumerate(identifiers)),
        encoding="utf-8",
    )
    return int(matrix.shape[1])


def command_extract_speakers(args: argparse.Namespace) -> None:
    vendor = args.vendor_dir.expanduser().resolve()
    model_path = args.models_dir.expanduser().resolve() / "tts/Embedding/embedding_function.pt"
    sys.path.insert(0, str(vendor))
    from anonymization.modules.sttts.speaker_embeddings.speaker_extraction import SpeakerExtraction

    devices = [torch.device(value.strip()) for value in args.devices.split(",")]
    results = args.output_dir / "source"
    settings = {
        "emb_model_path": model_path,
        "vec_type": "style-embed",
        "emb_level": "spk",
    }
    extractor = SpeakerExtraction(
        devices=devices,
        settings=settings,
        results_dir=results,
        model_dir=model_path,
        save_intermediate=True,
        force_compute=False,
    )
    embeddings = extractor.extract_speakers(
        dataset_path=args.reference_dir, dataset_name="codec_source", emb_level="spk"
    )
    if embeddings.vectors is None or embeddings.vectors.ndim != 2:
        raise RuntimeError("Speaker extraction produced no vector matrix")
    if embeddings.vectors.shape[1] != SPEAKER_DIM:
        raise ValueError(f"Expected {SPEAKER_DIM}-d GST vectors, got {embeddings.vectors.shape[1]}")
    sa_dimension = consolidate_sa_vectors(args.sa_run_dir, args.pair_manifest, args.output_dir)
    if sa_dimension != SPEAKER_DIM:
        raise ValueError(f"Expected {SPEAKER_DIM}-d anonymous GST vectors, got {sa_dimension}")
    print(
        f"codec_speaker_extraction=PASS,source_speakers={len(embeddings)},"
        f"embedding_dim={embeddings.vectors.shape[1]}",
        flush=True,
    )


def load_student_index(cache_dir: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(cache_dir.glob("index-r*-s*.jsonl")):
        for row in read_jsonl(path):
            item_id = str(row["item_id"])
            if item_id in result:
                raise ValueError(f"Duplicate Student cache item: {item_id}")
            result[item_id] = row
    return result


def cache_fields(row: dict[str, Any]) -> dict[str, Any]:
    names = (
        "student_hidden_path",
        "student_hidden_offset_frames",
        "student_hidden_frames",
        "student_hidden_dim",
        "student_hidden_dtype",
        "phone_target_path",
        "phone_target_offset_frames",
        "phone_target_frames",
        "phone_target_dim",
        "phone_target_dtype",
    )
    return {name: row[name] for name in names}


def command_finalize(args: argparse.Namespace) -> None:
    source_rows = read_jsonl(args.source_manifest)
    pair_rows = read_jsonl(args.pair_manifest)
    student = load_student_index(args.student_cache_dir)
    source_store = args.speaker_cache_dir / "source" / "codec_source" / "spk-level"
    source_mapping = read_id2idx(source_store / "id2idx")
    source_vectors = torch.load(
        source_store / "speaker_vectors.pt", map_location="cpu", weights_only=True
    )
    sa_mapping = read_id2idx(args.speaker_cache_dir / "sa_id2idx")
    sa_vectors = torch.load(
        args.speaker_cache_dir / "sa_speaker_vectors.pt", map_location="cpu", weights_only=True
    )
    if (
        source_vectors.ndim != 2
        or sa_vectors.ndim != 2
        or source_vectors.shape[1] != sa_vectors.shape[1]
    ):
        raise ValueError("Source and SA speaker stores must be rank 2 with the same dimension")
    if source_vectors.shape[1] != SPEAKER_DIM:
        raise ValueError(f"Expected {SPEAKER_DIM}-d GST speaker targets")

    def view(
        row: dict[str, Any], role: str, audio_path: str, speaker_path: Path, speaker_index: int
    ) -> dict[str, Any]:
        item_id = stable_item_id(role, str(row["utterance_id"]), audio_path)
        if item_id not in student:
            raise KeyError(f"Missing Student cache for {item_id}")
        return {
            "utterance_id": str(row["utterance_id"]),
            "speaker_id": str(row["speaker_id"]),
            "audio_path": str(Path(audio_path).expanduser().resolve()),
            **cache_fields(student[item_id]),
            "speaker_target_path": str(speaker_path.resolve()),
            "speaker_target_index": speaker_index,
        }

    source_output = []
    source_path = source_store / "speaker_vectors.pt"
    for row in source_rows:
        speaker_id = str(row["speaker_id"])
        if speaker_id not in source_mapping:
            raise KeyError(f"Missing source speaker vector for {speaker_id}")
        source_output.append(
            view(row, "source", str(row["audio_path"]), source_path, source_mapping[speaker_id])
        )
    pair_output = []
    sa_path = args.speaker_cache_dir / "sa_speaker_vectors.pt"
    for row in pair_rows:
        utterance_id, speaker_id = str(row["utterance_id"]), str(row["speaker_id"])
        source_audio, sa_audio = pair_paths(row)
        if speaker_id not in source_mapping or utterance_id not in sa_mapping:
            raise KeyError(f"Missing speaker target for pair {utterance_id}")
        pair_output.append(
            {
                "utterance_id": utterance_id,
                "source": view(
                    row, "pair-source", source_audio, source_path, source_mapping[speaker_id]
                ),
                "sa": view(row, "pair-sa", sa_audio, sa_path, sa_mapping[utterance_id]),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(source_output, args.output_dir / "source_train_cache.jsonl")
    write_jsonl(pair_output, args.output_dir / "pair_train_cache.jsonl")
    report = {
        "status": "PASS",
        "source_rows": len(source_output),
        "pair_rows": len(pair_output),
        "student_cache_items": len(student),
        "speaker_target_dim": int(source_vectors.shape[1]),
        "source_manifest": str((args.output_dir / "source_train_cache.jsonl").resolve()),
        "pair_manifest": str((args.output_dir / "pair_train_cache.jsonl").resolve()),
    }
    (args.output_dir / "finalize.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    print("codec_preprocess_finalize=PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build sharded caches for X-VC2 Codec training")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--source-manifest", type=Path, required=True)
    plan.add_argument("--pair-manifest", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--references-per-speaker", type=int, default=3)
    plan.add_argument("--space-safety-factor", type=float, default=1.10)
    plan.set_defaults(function=command_plan)

    student = commands.add_parser("extract-student")
    student.add_argument("--inventory", type=Path, required=True)
    student.add_argument("--student-checkpoint", type=Path, required=True)
    student.add_argument("--output-dir", type=Path, required=True)
    student.add_argument("--batch-size", type=int, default=8)
    student.add_argument("--max-batch-seconds", type=float, default=240.0)
    student.add_argument("--num-workers", type=int, default=2)
    student.add_argument("--shard-max-frames", type=int, default=500_000)
    student.add_argument("--require-cuda", action="store_true")
    student.set_defaults(function=command_extract_student)

    probe = commands.add_parser("probe-pair-alignment")
    probe.add_argument("--pair-manifest", type=Path, required=True)
    probe.add_argument("--student-checkpoint", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--device", default="cuda:0")
    probe.add_argument("--max-items", type=int, default=256)
    probe.add_argument("--max-seconds", type=float, default=20.0)
    probe.add_argument("--max-lag-frames", type=int, default=30)
    probe.add_argument("--seed", type=int, default=1)
    probe.set_defaults(function=command_probe_pair_alignment)

    speakers = commands.add_parser("extract-speakers")
    speakers.add_argument("--reference-dir", type=Path, required=True)
    speakers.add_argument("--models-dir", type=Path, required=True)
    speakers.add_argument("--vendor-dir", type=Path, required=True)
    speakers.add_argument("--sa-run-dir", type=Path, required=True)
    speakers.add_argument("--pair-manifest", type=Path, required=True)
    speakers.add_argument("--output-dir", type=Path, required=True)
    speakers.add_argument("--devices", default="cuda:0")
    speakers.set_defaults(function=command_extract_speakers)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--source-manifest", type=Path, required=True)
    finalize.add_argument("--pair-manifest", type=Path, required=True)
    finalize.add_argument("--student-cache-dir", type=Path, required=True)
    finalize.add_argument("--speaker-cache-dir", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.set_defaults(function=command_finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if getattr(args, "references_per_speaker", 1) <= 0:
        raise ValueError("references-per-speaker must be positive")
    if getattr(args, "space_safety_factor", 1.0) < 1.0:
        raise ValueError("space-safety-factor must be at least 1")
    for name in (
        "batch_size",
        "shard_max_frames",
        "max_batch_seconds",
        "max_items",
        "max_seconds",
        "max_lag_frames",
    ):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if hasattr(args, "num_workers") and args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    args.function(args)


if __name__ == "__main__":
    main()
