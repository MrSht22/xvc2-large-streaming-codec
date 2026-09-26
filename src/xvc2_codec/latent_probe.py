from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .checkpoint import load_checkpoint
from .config import load_config
from .data import PairDataset, SourceDataset, collate_views, read_jsonl
from .model import LargeStreamingCodec
from .train import TrainableCodec, _load_initial_model_state


@dataclass
class LatentItem:
    item_id: str
    speaker_id: str | None
    z_inv: torch.Tensor
    z_dyn: torch.Tensor
    z_edit: torch.Tensor
    phone: torch.Tensor | None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _split_indices(
    item_ids: list[str], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    grouped: dict[str, list[int]] = {}
    for index, item_id in enumerate(item_ids):
        grouped.setdefault(item_id, []).append(index)
    order = list(grouped)
    random.Random(seed).shuffle(order)
    val_count = max(1, round(len(order) * val_fraction)) if len(order) > 1 else 0
    val = sorted(index for item_id in order[:val_count] for index in grouped[item_id])
    train = sorted(index for item_id in order[val_count:] for index in grouped[item_id])
    if not train and val:
        train, val = val[:1], val[1:]
    return train, val


def _split_stratified(
    labels: torch.Tensor, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        grouped.setdefault(int(label), []).append(index)
    train: list[int] = []
    val: list[int] = []
    for label, indices in sorted(grouped.items()):
        random.Random(f"{seed}:speaker:{label}").shuffle(indices)
        if len(indices) < 2:
            raise ValueError(f"Speaker class {label} has fewer than two utterances")
        val_count = min(max(1, round(len(indices) * val_fraction)), len(indices) - 1)
        val.extend(indices[:val_count])
        train.extend(indices[val_count:])
    return sorted(train), sorted(val)


def _label_map(labels: Iterable[str]) -> dict[str, int]:
    return {label: index for index, label in enumerate(sorted(set(labels)))}


def _sample_rows(rows: list[dict[str, Any]], max_items: int | None, seed: int) -> list[dict[str, Any]]:
    if max_items is None or max_items >= len(rows):
        return rows
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[index] for index in sorted(indices[:max_items])]


def _load_model(config_path: Path, checkpoint_path: Path, weights: str, device: torch.device) -> nn.Module:
    config = load_config(config_path)
    payload = load_checkpoint(checkpoint_path, restore_rng=False)
    checkpoint_config = payload.get("config", {})
    if checkpoint_config.get("model") != config.model.to_dict():
        raise RuntimeError("Checkpoint model architecture differs from the supplied config")
    model = TrainableCodec(
        LargeStreamingCodec(config.model),
        _speaker_target_dim(payload),
        config.loss.dyn_phone_grl_scale,
        config.loss.edit_phone_grl_scale,
    )
    state = payload["model"] if weights == "model" else payload["ema"]["shadow"]
    _load_initial_model_state(model, state)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _speaker_target_dim(payload: dict[str, Any]) -> int:
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state")
    weight = state.get("style_head.2.weight")
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise ValueError("Cannot infer speaker target dimension from style_head")
    return int(weight.shape[0])


def _extract_source_items(
    model: nn.Module,
    rows: list[dict[str, Any]],
    config: Any,
    device: torch.device,
    segment_frames: int,
    batch_size: int,
    seed: int,
) -> list[LatentItem]:
    dataset = SourceDataset(rows, config.model.hop_length, segment_frames)
    result: list[LatentItem] = []
    for begin in range(0, len(rows), batch_size):
        items = [
            dataset.load(index, seed * 1_000_003 + begin + position)
            for position, index in enumerate(range(begin, min(begin + batch_size, len(rows))))
        ]
        batch = collate_views(items)
        waveform = batch["waveform"].to(device, non_blocking=True)
        hidden = batch["student_hidden"].to(device, non_blocking=True)
        with torch.inference_mode():
            outputs = model(waveform, hidden)
        z_edit = outputs["z_edit"].transpose(1, 2)
        for position, row in enumerate(rows[begin : begin + len(items)]):
            frames = min(
                int(items[position]["frames"]),
                int(outputs["z_inv"].shape[1]),
                int(outputs["z_dyn"].shape[1]),
                int(z_edit.shape[1]),
            )
            phone = batch.get("phone_target")
            if phone is not None:
                frames = min(frames, int(phone.shape[1]))
            phone_item = None if phone is None else phone[position, :frames].cpu().float()
            result.append(
                LatentItem(
                    item_id=str(row.get("utterance_id") or row.get("audio_path")),
                    speaker_id=str(row["speaker_id"]) if row.get("speaker_id") is not None else None,
                    z_inv=outputs["z_inv"][position, :frames].cpu().float(),
                    z_dyn=outputs["z_dyn"][position, :frames].cpu().float(),
                    z_edit=z_edit[position, :frames].cpu().float(),
                    phone=phone_item,
                )
            )
    return result


def _pair_consistency(
    model: nn.Module,
    rows: list[dict[str, Any]],
    config: Any,
    device: torch.device,
    segment_frames: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    dataset = PairDataset(rows, config.model.hop_length, segment_frames)
    values: dict[str, list[float]] = {"z_inv_mse": [], "z_inv_cosine": [], "z_dyn_cosine": [], "z_edit_cosine": []}
    for begin in range(0, len(rows), batch_size):
        items = [
            dataset.load(index, seed * 1_000_003 + begin + position)
            for position, index in enumerate(range(begin, min(begin + batch_size, len(rows))))
        ]
        source = collate_views([item["source"] for item in items])
        sa = collate_views([item["sa"] for item in items])
        with torch.inference_mode():
            source_output = model(
                source["waveform"].to(device, non_blocking=True),
                source["student_hidden"].to(device, non_blocking=True),
            )
            sa_output = model(
                sa["waveform"].to(device, non_blocking=True),
                sa["student_hidden"].to(device, non_blocking=True),
            )
        source_latents = {
            "z_inv": source_output["z_inv"],
            "z_dyn": source_output["z_dyn"],
            "z_edit": source_output["z_edit"].transpose(1, 2),
        }
        sa_latents = {
            "z_inv": sa_output["z_inv"],
            "z_dyn": sa_output["z_dyn"],
            "z_edit": sa_output["z_edit"].transpose(1, 2),
        }
        for position, item in enumerate(items):
            frames = int(item["source"]["frames"])
            for name in ("z_inv", "z_dyn", "z_edit"):
                frames = min(
                    frames,
                    int(source_latents[name].shape[1]),
                    int(sa_latents[name].shape[1]),
                )
                left = source_latents[name][position, :frames].float()
                right = sa_latents[name][position, :frames].float()
                cosine = F.cosine_similarity(left, right, dim=-1).mean().item()
                values[f"{name}_cosine"].append(cosine)
            values["z_inv_mse"].append(
                (
                    source_latents["z_inv"][position, :frames].float()
                    - sa_latents["z_inv"][position, :frames].float()
                )
                .square()
                .mean()
                .item()
            )
    return {
        "pairs": len(rows),
        "mean": {name: sum(items) / max(len(items), 1) for name, items in values.items()},
        "per_pair": values,
    }


def _prepare_content(
    items: list[LatentItem], latent_name: str, max_frames: int
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    groups: list[str] = []
    for item in items:
        if item.phone is None:
            continue
        features.append(getattr(item, latent_name))
        labels.append(item.phone.argmax(dim=-1))
        groups.extend([item.item_id] * len(item.z_inv))
    if not features:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long), []
    x = torch.cat(features, dim=0)
    y = torch.cat(labels, dim=0).long()
    if len(x) != len(y) or len(x) != len(groups):
        raise RuntimeError(
            "Content probe alignment mismatch: "
            f"features={len(x)}, labels={len(y)}, groups={len(groups)}"
        )
    if len(x) > max_frames:
        generator = torch.Generator().manual_seed(17)
        indices = torch.randperm(len(x), generator=generator)[:max_frames]
        x, y = x[indices], y[indices]
        groups = [groups[index] for index in indices.tolist()]
    return x, y, groups


def _prepare_speaker(items: list[LatentItem], latent_name: str) -> tuple[torch.Tensor, list[str]]:
    selected = [item for item in items if item.speaker_id is not None]
    if not selected:
        return torch.empty(0, 0), []
    return torch.stack([getattr(item, latent_name).mean(dim=0) for item in selected]), [
        str(item.speaker_id) for item in selected
    ]


def _train_linear_probe(
    features: torch.Tensor,
    labels: list[int] | torch.Tensor,
    item_ids: list[str],
    seed: int,
    val_fraction: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    stratified: bool = False,
) -> dict[str, Any]:
    if len(features) == 0:
        return {"status": "SKIP", "reason": "no_labels"}
    if isinstance(labels, list):
        labels_tensor = torch.tensor(labels, dtype=torch.long)
    else:
        labels_tensor = labels.long()
    if len(features) != len(labels_tensor) or len(features) != len(item_ids):
        raise RuntimeError(
            "Linear probe alignment mismatch: "
            f"features={len(features)}, labels={len(labels_tensor)}, groups={len(item_ids)}"
        )
    label_count = int(labels_tensor.max().item()) + 1
    if label_count < 2:
        return {"status": "SKIP", "reason": "fewer_than_two_classes"}
    train_items, val_items = (
        _split_stratified(labels_tensor, val_fraction, seed)
        if stratified
        else _split_indices(item_ids, val_fraction, seed)
    )
    train_set = set(train_items)
    val_set = set(val_items)
    train_mask = torch.tensor([index in train_set for index in range(len(features))])
    val_mask = torch.tensor([index in val_set for index in range(len(features))])
    if not train_mask.any() or not val_mask.any():
        return {"status": "SKIP", "reason": "empty_split"}
    train_x, train_y = features[train_mask], labels_tensor[train_mask]
    val_x, val_y = features[val_mask], labels_tensor[val_mask]
    means = train_x.mean(dim=0, keepdim=True)
    scales = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    train_x = (train_x - means) / scales
    val_x = (val_x - means) / scales
    model = nn.Linear(features.shape[-1], label_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(train_x), generator=generator)
        for begin in range(0, len(order), batch_size):
            indices = order[begin : begin + batch_size]
            loss = F.cross_entropy(model(train_x[indices]), train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.inference_mode():
        predictions = model(val_x).argmax(dim=-1)
    accuracy = float((predictions == val_y).float().mean())
    majority = float((val_y == torch.bincount(train_y).argmax()).float().mean())
    return {
        "status": "PASS",
        "train_items": int(train_mask.sum()),
        "val_items": int(val_mask.sum()),
        "classes": label_count,
        "accuracy": accuracy,
        "majority_baseline": majority,
        "uniform_chance_baseline": 1.0 / label_count,
        "accuracy_over_majority": accuracy - majority,
    }


def _content_probes(items: list[LatentItem], args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("z_inv", "z_dyn", "z_edit"):
        features, labels, groups = _prepare_content(items, name, args.max_frames)
        if len(features) == 0:
            result[name] = {"status": "SKIP", "reason": "phone_target_missing"}
            continue
        result[name] = _train_linear_probe(
            features,
            labels,
            groups,
            args.seed,
            args.val_fraction,
            args.epochs,
            args.probe_batch_size,
            args.learning_rate,
        )
    return result


def _speaker_probes(items: list[LatentItem], args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("z_inv", "z_dyn", "z_edit"):
        features, labels = _prepare_speaker(items, name)
        if len(features) == 0:
            result[name] = {"status": "SKIP", "reason": "speaker_id_missing"}
            continue
        mapping = _label_map(labels)
        encoded = [mapping[label] for label in labels]
        result[name] = _train_linear_probe(
            features,
            encoded,
            [item.item_id for item in items if item.speaker_id is not None],
            args.seed + 100,
            args.val_fraction,
            args.epochs,
            args.probe_batch_size,
            args.learning_rate,
            stratified=True,
        )
    return result


def _write_csv(path: Path, content: dict[str, Any], leakage: dict[str, Any]) -> None:
    rows = []
    for family, values in (("content", content), ("speaker_leakage", leakage)):
        for latent, metrics in values.items():
            row = {"family": family, "latent": latent}
            row.update({name: value for name, value in metrics.items() if not isinstance(value, dict)})
            rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Codec latent content and leakage probes")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--segment-seconds", type=float, default=3.2)
    parser.add_argument("--max-source-items", type=int, default=512)
    parser.add_argument("--max-pair-items", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-latents", action="store_true")
    args = parser.parse_args()
    if args.max_source_items <= 0 or args.max_pair_items <= 0 or args.extract_batch_size <= 0:
        parser.error("item and batch limits must be positive")
    if args.max_frames <= 0 or args.epochs <= 0 or args.probe_batch_size <= 0:
        parser.error("probe limits must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    config = load_config(args.config)
    model = _load_model(args.config, args.checkpoint, args.weights, device)
    segment_frames = round(args.segment_seconds * 16_000 / config.model.hop_length)
    source_rows = _sample_rows(read_jsonl(args.source_manifest), args.max_source_items, args.seed)
    source_items = _extract_source_items(
        model,
        source_rows,
        config,
        device,
        segment_frames,
        args.extract_batch_size,
        args.seed,
    )
    content = _content_probes(source_items, args)
    leakage = _speaker_probes(source_items, args)
    pairs = None
    if args.pair_manifest:
        pair_rows = _sample_rows(read_jsonl(args.pair_manifest), args.max_pair_items, args.seed + 1)
        pairs = _pair_consistency(
            model,
            pair_rows,
            config,
            device,
            segment_frames,
            args.extract_batch_size,
            args.seed + 1,
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASS",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "weights": args.weights,
        "source_manifest": str(args.source_manifest.expanduser().resolve()),
        "source_items": len(source_items),
        "phone_labeled_items": sum(item.phone is not None for item in source_items),
        "content_probe": content,
        "speaker_leakage_probe": leakage,
        "pair_consistency": pairs,
        "interpretation": {
            "content_probe": "higher accuracy indicates more recoverable phone content",
            "speaker_leakage_probe": "higher accuracy indicates more recoverable source speaker identity",
            "pair_consistency": "higher z_inv cosine and lower z_inv MSE indicate source/SA invariance",
        },
    }
    _write_json(output_dir / "report.json", report)
    _write_csv(output_dir / "probe_metrics.csv", content, leakage)
    if args.save_latents:
        torch.save(
            {
                "item_ids": [item.item_id for item in source_items],
                "speaker_ids": [item.speaker_id for item in source_items],
                "z_inv": [item.z_inv for item in source_items],
                "z_dyn": [item.z_dyn for item in source_items],
                "z_edit": [item.z_edit for item in source_items],
                "phone": [item.phone for item in source_items],
            },
            output_dir / "source_latents.pt",
        )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, default=_json_default))
    print("codec_latent_probe=PASS")


if __name__ == "__main__":
    main()
