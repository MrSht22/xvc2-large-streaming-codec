from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config
from .data import PairDataset, SourceDataset, TrainingStepDataset, read_jsonl
from .discriminator import (
    MultiScaleSTFTDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)
from .ema import ExponentialMovingAverage
from .losses import ReconstructionLoss, masked_smooth_l1, trajectory_correlation_loss
from .model import LargeStreamingCodec, parameter_breakdown
from .schedule import weights_at


class TrainableCodec(torch.nn.Module):
    """Codec plus the training-only Z_edit speaker/style projection."""

    def __init__(self, codec: LargeStreamingCodec, speaker_target_dim: int) -> None:
        super().__init__()
        self.codec = codec
        self.style_head = torch.nn.Sequential(
            torch.nn.Linear(codec.config.edit_dim, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, speaker_target_dim),
        )

    def forward(
        self, waveform: torch.Tensor, student_hidden: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        output = self.codec(waveform, student_hidden)
        output["style_embedding"] = self.style_head(output["z_edit"].mean(-1))
        return output


def initialize_runtime(device_arg: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return torch.device(f"cuda:{local_rank}"), rank, world_size, local_rank
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_arg), rank, world_size, local_rank


def move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {name: move(item, device) for name, item in value.items()}
    return value


def initialize_data_worker(_: int) -> None:
    torch.set_num_threads(1)


def warmup_learning_rate(base: float, update: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base
    return base * min(max(update, 0) / warmup_steps, 1.0)


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def discriminator_batch(outputs, batches, views):
    return (
        torch.cat([batches[view]["waveform"] for view in views], dim=0),
        torch.cat([outputs[view]["reconstruction"] for view in views], dim=0),
    )


def anchor_losses(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    frames = batch["frames"]
    zero = output["reconstruction"].new_zeros(())
    phone = zero
    if "phone_target" in batch:
        length = min(output["phone_logits"].shape[1], batch["phone_target"].shape[1])
        mask = torch.arange(length, device=frames.device)[None] < frames[:, None].clamp_max(length)
        values = F.kl_div(
            output["phone_logits"][:, :length].float().log_softmax(-1),
            batch["phone_target"][:, :length].float().softmax(-1),
            reduction="none",
        ).sum(-1)
        phone = (values * mask).sum() / mask.sum().clamp_min(1)
    dyn = (
        masked_smooth_l1(output["dyn_anchor"], batch["dyn_target"], frames)
        if "dyn_target" in batch
        else zero
    )
    prosody = (
        masked_smooth_l1(output["prosody"], batch["prosody_target"], frames)
        if "prosody_target" in batch
        else zero
    )
    return {"phone_anchor": phone, "dyn_anchor": dyn, "prosody_anchor": prosody}


def view_metrics(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    reconstruction_loss: ReconstructionLoss,
    hop_length: int,
) -> dict[str, torch.Tensor]:
    samples = batch["frames"] * hop_length
    reconstruction, reconstruction_metrics = reconstruction_loss(
        output["reconstruction"], batch["waveform"], samples
    )
    if "speaker_target" not in batch:
        raise ValueError("Every training row requires speaker_target_path")
    style = (
        1
        - F.cosine_similarity(
            output["style_embedding"].float(), batch["speaker_target"].float(), dim=-1
        ).mean()
    )
    anchors = anchor_losses(output, batch)
    return {
        "reconstruction": reconstruction,
        "edit_style": style,
        **anchors,
        **reconstruction_metrics,
    }


def forward_views(
    model: torch.nn.Module,
    batches: dict[str, dict[str, torch.Tensor]],
    views: tuple[str, ...],
) -> dict[str, dict[str, torch.Tensor]]:
    batch_size = batches[views[0]]["waveform"].shape[0]
    waveform = torch.cat([batches[view]["waveform"] for view in views], dim=0)
    student_hidden = torch.cat([batches[view]["student_hidden"] for view in views], dim=0)
    combined = model(waveform, student_hidden)
    return {
        view: {
            name: value[index * batch_size : (index + 1) * batch_size]
            for name, value in combined.items()
        }
        for index, view in enumerate(views)
    }


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the large unified X-VC2 Codec")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speaker-target-dim", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-rank item or pair batch")
    parser.add_argument("--segment-seconds", type=float, default=3.2)
    parser.add_argument("--pair-probability", type=float, default=0.15)
    parser.add_argument("--steps", type=int, help="Stop after this many steps in this invocation")
    parser.add_argument("--num-workers", type=int, default=2, help="Loader workers per rank")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--no-pin-memory", action="store_true", help="Disable pinned host-memory batches"
    )
    args = parser.parse_args()
    if not 0 <= args.pair_probability <= 1:
        raise ValueError("pair_probability must be in [0, 1]")
    if min(args.batch_size, args.prefetch_factor) <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch or DataLoader configuration")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("steps must be positive")

    config = load_config(args.config)
    device, rank, world_size, local_rank = initialize_runtime(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    random.seed(config.training.seed + rank)
    torch.manual_seed(config.training.seed + rank)
    segment_frames = round(args.segment_seconds * 16_000 / config.model.hop_length)
    source_dataset = SourceDataset(
        read_jsonl(args.source_manifest), config.model.hop_length, segment_frames
    )
    pair_dataset = PairDataset(
        read_jsonl(args.pair_manifest), config.model.hop_length, segment_frames
    )
    codec = LargeStreamingCodec(config.model)
    model = TrainableCodec(codec, args.speaker_target_dim).to(device)
    discriminator = MultiScaleSTFTDiscriminator(config.discriminator).to(device)
    generator_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.generator_learning_rate,
        weight_decay=config.training.weight_decay,
        fused=device.type == "cuda",
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.training.discriminator_learning_rate,
        betas=(0.8, 0.99),
        weight_decay=config.training.weight_decay,
        fused=device.type == "cuda",
    )
    ema = ExponentialMovingAverage(model, config.training.ema_decay)
    use_amp = device.type == "cuda" and config.training.amp != "none"
    amp_dtype = torch.bfloat16 if config.training.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and config.training.amp == "fp16")
    step = 0
    if args.resume:
        payload = load_checkpoint(args.resume)
        if payload["config"] != config.to_dict():
            raise RuntimeError("Resume configuration differs")
        model.load_state_dict(payload["model"], strict=True)
        discriminator.load_state_dict(payload["discriminator"], strict=True)
        generator_optimizer.load_state_dict(payload["generator_optimizer"])
        discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
        ema.load_state_dict(payload["ema"])
        scaler.load_state_dict(payload.get("scaler", {}))
        step = int(payload["step"])
    training_model: torch.nn.Module = model
    training_discriminator: torch.nn.Module = discriminator
    if world_size > 1:
        training_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
        training_discriminator = DistributedDataParallel(
            discriminator,
            device_ids=[local_rank],
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    reconstruction_loss = ReconstructionLoss()
    end_step = min(
        config.schedule.max_steps,
        step + args.steps if args.steps is not None else config.schedule.max_steps,
    )
    step_dataset = TrainingStepDataset(
        source_dataset,
        pair_dataset,
        start_step=step,
        end_step=end_step,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        seed=config.training.seed,
        pair_probability=args.pair_probability,
    )
    if min(len(source_dataset), len(pair_dataset)) < args.batch_size * world_size:
        raise ValueError("Each dataset must contain at least one global batch")
    loader_options = {
        "dataset": step_dataset,
        "batch_size": None,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda" and not args.no_pin_memory,
    }
    if args.num_workers:
        loader_options.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
            worker_init_fn=initialize_data_worker,
            multiprocessing_context="spawn" if device.type == "cuda" else None,
        )
    loader = DataLoader(**loader_options)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print(json.dumps({"parameters": parameter_breakdown(codec), "config": config.to_dict()}))

    interval_started = time.monotonic()
    interval_audio_seconds = 0.0
    interval_data_wait = 0.0
    interval_steps = 0
    iterator = iter(loader)
    while step < end_step:
        wait_started = time.monotonic()
        prepared = next(iterator)
        interval_data_wait += time.monotonic() - wait_started
        next_step = int(prepared["step"])
        weights = weights_at(next_step, config.schedule, config.loss)
        choose_pair = prepared["kind"] == "pair"
        views = ("source", "sa") if choose_pair else ("source",)
        cpu_batch = prepared["batch"]
        cpu_batches = cpu_batch if choose_pair else {"source": cpu_batch}
        local_audio_seconds = sum(
            float(cpu_batches[view]["frames"].sum()) * config.model.hop_length / 16_000
            for view in views
        )
        batch = move(cpu_batch, device)
        batches = batch if choose_pair else {"source": batch}

        generator_learning_rate = warmup_learning_rate(
            config.training.generator_learning_rate,
            next_step,
            config.training.generator_warmup_steps,
        )
        set_learning_rate(generator_optimizer, generator_learning_rate)

        generator_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = forward_views(training_model, batches, views)
            metrics = {}
            for view in views:
                metrics[view] = view_metrics(
                    outputs[view],
                    batches[view],
                    reconstruction_loss,
                    config.model.hop_length,
                )
            reconstruction = torch.stack([metrics[view]["reconstruction"] for view in views]).mean()
            edit_style = torch.stack([metrics[view]["edit_style"] for view in views]).mean()
            phone_anchor = torch.stack([metrics[view]["phone_anchor"] for view in views]).mean()
            dyn_anchor = torch.stack([metrics[view]["dyn_anchor"] for view in views]).mean()
            prosody_anchor = torch.stack([metrics[view]["prosody_anchor"] for view in views]).mean()
            sa_inv = reconstruction.new_zeros(())
            sa_dyn = reconstruction.new_zeros(())
            if choose_pair:
                pair_lengths = torch.minimum(batches["source"]["frames"], batches["sa"]["frames"])
                sa_inv = masked_smooth_l1(
                    outputs["source"]["z_inv"], outputs["sa"]["z_inv"].detach(), pair_lengths
                )
                sa_dyn = trajectory_correlation_loss(
                    outputs["source"]["z_dyn"], outputs["sa"]["z_dyn"].detach(), pair_lengths
                )
            adversarial = reconstruction.new_zeros(())
            feature_matching = reconstruction.new_zeros(())
            if weights.adversarial:
                set_requires_grad(discriminator, False)
                real_waveform, fake_waveform = discriminator_batch(outputs, batches, views)
                fake = training_discriminator(fake_waveform)
                with torch.no_grad():
                    real = training_discriminator(real_waveform)
                adversarial = generator_adversarial_loss(fake)
                feature_matching = feature_matching_loss(real, fake)
            objective = (
                weights.reconstruction * reconstruction
                + weights.edit_style * edit_style
                + weights.phone_anchor * phone_anchor
                + weights.dyn_anchor * dyn_anchor
                + weights.prosody_anchor * prosody_anchor
                + weights.sa_inv * sa_inv
                + weights.sa_dyn * sa_dyn
                + weights.adversarial * adversarial
                + weights.feature_matching * feature_matching
            )
        scaler.scale(objective).backward()
        scaler.unscale_(generator_optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip
        )
        scaler.step(generator_optimizer)
        scaler.update()

        discriminator_value = objective.new_zeros(())
        discriminator_learning_rate = 0.0
        if weights.adversarial:
            set_requires_grad(discriminator, True)
            discriminator_learning_rate = warmup_learning_rate(
                config.training.discriminator_learning_rate,
                next_step - config.schedule.reconstruction_end,
                config.training.discriminator_warmup_steps,
            )
            set_learning_rate(discriminator_optimizer, discriminator_learning_rate)
            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                real_waveform, fake_waveform = discriminator_batch(outputs, batches, views)
                real = training_discriminator(real_waveform)
                fake = training_discriminator(fake_waveform.detach())
                discriminator_value = discriminator_loss(real, fake)
            discriminator_value.backward()
            discriminator_optimizer.step()
        step = next_step
        interval_audio_seconds += local_audio_seconds
        interval_steps += 1
        if step >= config.schedule.reconstruction_end:
            ema.update(model)
        if step == 1 or step % config.training.log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - interval_started
            performance = torch.tensor(
                [interval_audio_seconds, interval_data_wait], device=device, dtype=torch.float64
            )
            elapsed_tensor = torch.tensor(elapsed, device=device, dtype=torch.float64)
            peak_memory = torch.tensor(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                device=device,
                dtype=torch.float64,
            )
            if world_size > 1:
                dist.all_reduce(performance, op=dist.ReduceOp.SUM)
                dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "batch_kind": "pair" if choose_pair else "source",
                            "objective": float(objective.detach()),
                            "reconstruction": float(reconstruction.detach()),
                            "sa_inv": float(sa_inv.detach()),
                            "sa_dyn": float(sa_dyn.detach()),
                            "adversarial": float(adversarial.detach()),
                            "feature_matching": float(feature_matching.detach()),
                            "discriminator": float(discriminator_value.detach()),
                            "gradient_norm": float(gradient_norm),
                            "generator_learning_rate": generator_learning_rate,
                            "discriminator_learning_rate": discriminator_learning_rate,
                            "global_audio_seconds_per_second": float(
                                performance[0] / elapsed_tensor
                            ),
                            "mean_data_wait_seconds_per_rank_step": float(
                                performance[1] / (world_size * interval_steps)
                            ),
                            "maximum_allocated_gib": float(peak_memory / 1024**3),
                            "weights": weights.__dict__,
                        }
                    )
                )
            interval_started = time.monotonic()
            interval_audio_seconds = 0.0
            interval_data_wait = 0.0
            interval_steps = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        if rank == 0 and (step % config.training.save_interval == 0 or step == end_step):
            save_checkpoint(
                args.output_dir / f"step-{step:06d}.pt",
                step=step,
                config=config.to_dict(),
                model=model.state_dict(),
                discriminator=discriminator.state_dict(),
                generator_optimizer=generator_optimizer.state_dict(),
                discriminator_optimizer=discriminator_optimizer.state_dict(),
                ema=ema.state_dict(),
                scaler=scaler.state_dict(),
            )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
