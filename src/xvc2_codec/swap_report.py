"""Waveform evaluation and a portable 3x3 report for the three single swaps."""

from __future__ import annotations

import csv
import html
import json
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio

from .data import read_jsonl
from .preprocess import praat_f0, prosody_target, write_jsonl


SWAPS = {"BAA": "Z_inv", "ABA": "Z_dyn", "AAB": "Z_edit"}
ATTRIBUTES = ("content", "prosody", "speaker")
REQUIRED = {"A", "B", "AAA", "BBB", "BBB_mapped", *SWAPS}


def prosody_distances(first: np.ndarray, second: np.ndarray) -> dict:
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 4:
        raise ValueError("Prosody comparisons require matching [frames, 4] trajectories")
    voiced = (first[:, 1] > 0.5) & (second[:, 1] > 0.5)
    consecutive = voiced.copy()
    consecutive[0] = False
    consecutive[1:] &= voiced[:-1]

    def rmse(column: int, mask: np.ndarray):
        return (
            float(np.sqrt(np.mean((first[mask, column] - second[mask, column]) ** 2)))
            if mask.sum() >= 3
            else None
        )

    return {
        "f0": rmse(0, voiced),
        "vuv": float(np.mean(first[:, 1] != second[:, 1])),
        "energy": rmse(2, np.ones(len(first), dtype=bool)),
        "delta_f0": rmse(3, consecutive),
        "joint_voiced_fraction": float(voiced.mean()),
    }


def grouped_records(records: list[dict]) -> dict[str, dict[str, dict]]:
    groups = defaultdict(dict)
    for row in records:
        pair_id, branch = row["pair_id"], row["branch"]
        if branch in groups[pair_id]:
            raise ValueError(f"Duplicate branch {pair_id}/{branch}")
        groups[pair_id][branch] = row
    if not groups or any(set(group) != REQUIRED for group in groups.values()):
        raise ValueError(f"Each pair must contain exactly {sorted(REQUIRED)}")
    for pair_id, group in groups.items():
        if len({row["text"] for row in group.values()}) != 1:
            raise ValueError(f"Inconsistent same-text reference: {pair_id}")
    return dict(groups)


def intervention_rows(records: list[dict]) -> list[dict]:
    result = []
    for pair_id, group in grouped_records(records).items():
        baseline = group["AAA"]
        for branch, latent in SWAPS.items():
            row = group[branch]
            values = {
                "pair_id": pair_id,
                "branch": branch,
                "latent": latent,
                "content": 100 * (row["wer"] - baseline["wer"]),
                "speaker": (row["sim_b"] - row["sim_a"]) - (baseline["sim_b"] - baseline["sim_a"]),
                "rms_ratio_to_AAA": row["rms"] / max(baseline["rms"], 1e-8),
                "clip_fraction": row["clip_fraction"],
                "identity_reference_kind": row["identity_reference_kind"],
            }
            for metric in ("f0", "vuv", "energy", "delta_f0"):
                before, after = baseline["prosody_to_b"][metric], row["prosody_to_b"][metric]
                values[f"prosody_{metric}"] = (
                    before - after if (before is not None and after is not None) else None
                )
            # A/B too similar cannot test transfer. Keep raw diagnostics, mask heatmap score.
            gap = baseline["prosody_to_b"]["f0"]
            values["prosody"] = values["prosody_f0"] if gap is not None and gap >= 0.1 else None
            values["prosody_control_gap"] = gap
            values["prosody_control_note"] = (
                "insufficient A/B F0 contrast or voiced overlap"
                if values["prosody"] is None
                else "normalized F0 only; see V/UV, energy and delta below"
            )
            result.append(values)
    return result


def matrix_summary(interventions: list[dict]) -> list[dict]:
    cells = []
    for branch, latent in SWAPS.items():
        rows = [row for row in interventions if row["branch"] == branch]
        for attribute in ATTRIBUTES:
            values = [row[attribute] for row in rows if row[attribute] is not None]
            submetrics = {}
            for metric in ("vuv", "energy", "delta_f0"):
                available = [
                    row[f"prosody_{metric}"]
                    for row in rows
                    if attribute == "prosody" and row[f"prosody_{metric}"] is not None
                ]
                submetrics[f"prosody_{metric}_mean"] = (
                    statistics.fmean(available) if available else None
                )
            cells.append(
                {
                    "branch": branch,
                    "latent": latent,
                    "attribute": attribute,
                    "mean": statistics.fmean(values) if values else None,
                    "median": statistics.median(values) if values else None,
                    "valid_pairs": len(values),
                    "total_pairs": len(rows),
                    **submetrics,
                }
            )
    return cells


def select_examples(rows: list[dict], attribute: str, count: int) -> list[dict]:
    # A spread through the distribution (including extremes), not only best cases.
    ordered = sorted(
        rows, key=lambda row: (row[attribute] is None, row[attribute] or 0, row["pair_id"])
    )
    positions = np.linspace(0, len(ordered) - 1, min(count, len(ordered))).round().astype(int)
    return [ordered[position] for position in positions]


def format_number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.4f}"


def write_report(records: list[dict], output: Path, examples_per_cell: int) -> None:
    groups = grouped_records(records)
    interventions = intervention_rows(records)
    cells = matrix_summary(interventions)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, output / "metrics.jsonl")
    write_jsonl(interventions, output / "interventions.jsonl")
    (output / "matrix.json").write_text(json.dumps(cells, indent=2, allow_nan=False) + "\n")
    with (output / "matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    assets = output / "audio"
    assets.mkdir(exist_ok=True)
    copied = {}

    def audio(pair_id: str, branch: str) -> str:
        key = (pair_id, branch)
        if key not in copied:
            destination = assets / f"example-{len(copied):04d}.wav"
            shutil.copyfile(groups[pair_id][branch]["audio_path"], destination)
            copied[key] = f"audio/{destination.name}"
        return (
            f'<label>{html.escape(branch)}<audio controls preload="none" src="{copied[key]}">'
            "</audio></label>"
        )

    labels = {
        "content": "Content: ΔWER (percentage points)",
        "prosody": "Local prosody: normalized-F0 movement toward B",
        "speaker": "Speaker: Δ(SIM_B − SIM_A)",
    }
    parts = [
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
        "<title>Codec single-swap 3×3</title><style>",
        "body{font:15px system-ui;margin:24px;color:#202630}table{border-collapse:collapse;"
        "width:100%;table-layout:fixed}th,td{border:1px solid #ccd2dc;padding:14px;"
        "vertical-align:top}th:first-child{width:75px}audio{display:block;width:100%;"
        "height:32px}label{display:block;margin:8px 0}details{margin-top:12px}"
        ".score{font-size:25px;font-weight:650}small{color:#4c5563}"
        "p{max-width:1100px}code{overflow-wrap:anywhere}</style><body>",
        "<h1>Codec · 三路单独 swap</h1>",
        "<p>A 为原音，B 为 Seed-VC 同句转换音。只干预 BAA / ABA / AAB；AAA 是共同基线，"
        "BBB_mapped 是 B→A 时间映射后的完整重建对照。每格样本覆盖该指标的低、中、高位置。</p>",
        "<p>Content 正值表示 WER 恶化，不能区分同句内容来自哪路。Prosody 正值表示 normalized "
        "F0 更接近 BBB_mapped；A/B F0 RMSE &lt; 0.1 或 voiced 交集不足时记 N/A。Speaker "
        "正值表示相对 AAA 更偏向 B 的身份参考。不同列的数值、颜色不可横向比较。</p>",
        "<p>时间映射可能改变局部韵律；同句身份参考有内容混杂。这里展示 Decoder 的干预响应，"
        "不等同于完整解耦证明。无混合输出的精确 waveform 真值，不计算 PESQ/STOI。</p>",
        "<table><tr><th>Swap</th>"
        + "".join(f"<th>{label}</th>" for label in labels.values())
        + "</tr>",
    ]
    for branch, latent in SWAPS.items():
        parts.append(f"<tr><th>{latent}<br>{branch}</th>")
        rows = [row for row in interventions if row["branch"] == branch]
        for attribute in ATTRIBUTES:
            cell = next(
                cell
                for cell in cells
                if cell["branch"] == branch and cell["attribute"] == attribute
            )
            scale = max(
                [
                    abs(c["mean"])
                    for c in cells
                    if c["attribute"] == attribute and c["mean"] is not None
                ]
                or [1]
            )
            opacity = min(abs(cell["mean"] or 0) / max(scale, 1e-8), 1) * 0.18
            color = "180,60,40" if attribute == "content" else "30,100,160"
            parts.append(
                f'<td style="background:rgba({color},{opacity:.3f})">'
                f'<div class="score">{format_number(cell["mean"])}</div>'
                f"<small>有效 {cell['valid_pairs']}/{cell['total_pairs']} 对；"
                f"中位数 {format_number(cell['median'])}</small>"
            )
            if attribute == "prosody":
                parts.append(
                    "<p>其他局部轨迹的平均距离改善：<br>"
                    + "；".join(
                        f"{metric} {format_number(cell[f'prosody_{metric}_mean'])}"
                        for metric in ("vuv", "energy", "delta_f0")
                    )
                    + "</p>"
                )
            for example in select_examples(rows, attribute, examples_per_cell):
                pair_id = example["pair_id"]
                group, row = groups[pair_id], groups[pair_id][branch]
                text = html.escape(row["text"])
                parts.append(
                    f"<details><summary>{html.escape(pair_id)} · "
                    f"{format_number(example[attribute])}</summary><p>{text}</p>"
                )
                parts.append(
                    f"<p>输出 ASR: {html.escape(row['asr_text'])}<br>"
                    f"WER {row['wer']:.4f}；SIM_A {row['sim_a']:.4f}；"
                    f"SIM_B {row['sim_b']:.4f}<br>"
                    f"与 B 对照共同 voiced 帧比例 "
                    f"{row['prosody_to_b']['joint_voiced_fraction']:.3f}<br>"
                    f"RMS/AAA {example['rms_ratio_to_AAA']:.3f}；"
                    f"|幅度|≥1 比例 {row['clip_fraction']:.4f}<br>"
                    f"身份参考: {html.escape(row['identity_reference_kind'])}；"
                    f"映射: {html.escape(row['alignment'])}<br>"
                    f"源/目标 speaker：{html.escape(str(row.get('source_speaker', '未提供')))} / "
                    f"{html.escape(str(row.get('target_speaker', '未提供')))}；"
                    f"conversion：{html.escape(str(row.get('conversion', '未提供')))}</p>"
                )
                parts.append(
                    "<p>向 B 的距离改善："
                    + "；".join(
                        f"{metric} {format_number(example[f'prosody_{metric}'])}"
                        for metric in ("f0", "vuv", "energy", "delta_f0")
                    )
                    + "<br>"
                    + html.escape(example["prosody_control_note"])
                    + "</p>"
                )
                for audio_branch in ("A", "B", "AAA", "BBB_mapped", branch):
                    parts.append(audio(pair_id, audio_branch))
                parts.append(
                    f"<small>A/B 原音 WER：{group['A']['wer']:.4f} / "
                    f"{group['B']['wer']:.4f}；BBB 原生 / 映射 WER："
                    f"{group['BBB']['wer']:.4f} / "
                    f"{group['BBB_mapped']['wer']:.4f}</small></details>"
                )
            parts.append("</td>")
        parts.append("</tr>")
    parts.append("</table></body></html>")
    (output / "report.html").write_text("\n".join(parts), encoding="utf-8")


def evaluate(args) -> None:
    records = read_jsonl(args.manifest)
    groups = grouped_records(records)
    # Reuse the current codec-eval backend, without copying evaluator/model implementations.
    sys.path.insert(0, str(args.evaluator_dir.resolve()))
    from common.text import word_errors
    from sim.evaluate import cosine_similarity, embed_paths, load_base_plus_model
    from wer.evaluate import load_transcriber

    transcribe = load_transcriber(
        args.models_dir / "faster-whisper-large-v3", args.device, args.compute_type, 5
    )
    processor, model, device = load_base_plus_model(
        args.models_dir / "wavlm-base-plus-sv", args.device
    )
    paths = sorted(
        {
            str(row[field])
            for row in records
            for field in ("audio_path", "a_identity_reference_path", "b_identity_reference_path")
        }
    )
    embeddings = dict(
        zip(
            paths,
            embed_paths(paths, processor, model, device, args.batch_size, "wavlm-base-plus-sv"),
            strict=True,
        )
    )
    results = []
    for index, (pair_id, group) in enumerate(groups.items(), 1):
        features = {}
        for branch, row in group.items():
            waveform, rate = torchaudio.load(row["audio_path"])
            if rate != 16_000 or waveform.shape[0] != 1 or waveform.shape[-1] % 320:
                raise ValueError(
                    f"Evaluation requires mono 16 kHz whole frames: {pair_id}/{branch}"
                )
            if not torch.isfinite(waveform).all():
                raise ValueError(f"Non-finite audio: {pair_id}/{branch}")
            frames = waveform.shape[-1] // 320
            features[branch] = prosody_target(waveform, frames, praat_f0(waveform, frames, 50, 600))
        for branch, row in group.items():
            path = row["audio_path"]
            hypothesis = transcribe(path)
            word = word_errors(row["text"], hypothesis)
            if not word["reference_words"]:
                raise ValueError("WER requires a non-empty English transcript")
            result = {
                **row,
                "asr_text": hypothesis,
                "wer": word["wer"],
                "word_errors": word["errors"],
                "reference_words": word["reference_words"],
                "sim_a": cosine_similarity(
                    embeddings[path], embeddings[row["a_identity_reference_path"]]
                ),
                "sim_b": cosine_similarity(
                    embeddings[path], embeddings[row["b_identity_reference_path"]]
                ),
            }
            if branch not in ("B", "BBB"):
                result["prosody_to_a"] = prosody_distances(features[branch], features["AAA"])
                result["prosody_to_b"] = prosody_distances(features[branch], features["BBB_mapped"])
            results.append(result)
        print(f"codec_swap_evaluate={index}/{len(groups)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_report(results, args.output_dir, args.examples_per_cell)
    request = {
        "manifest": str(args.manifest.resolve()),
        "evaluator_dir": str(args.evaluator_dir.resolve()),
        "asr_model": str((args.models_dir / "faster-whisper-large-v3").resolve()),
        "speaker_model": str((args.models_dir / "wavlm-base-plus-sv").resolve()),
        "device": args.device,
        "compute_type": args.compute_type,
        "pitch_floor": 50,
        "pitch_ceiling": 600,
        "examples_per_cell": args.examples_per_cell,
    }
    (args.output_dir / "evaluation_request.json").write_text(json.dumps(request, indent=2) + "\n")
    print(f"codec_swap_evaluate=PASS report={args.output_dir / 'report.html'}")
