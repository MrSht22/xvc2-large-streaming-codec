from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .config import DiscriminatorConfig, load_config
from .discriminator import (
    MultiScaleSTFTDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)
from .model import LargeStreamingCodec, parameter_breakdown
from .smoke import tiny_config


def initialize(device_arg: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        use_cuda = torch.cuda.is_available() and device_arg != "cpu"
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if use_cuda else "gloo")
        return (
            torch.device(f"cuda:{local_rank}" if use_cuda else "cpu"),
            rank,
            world_size,
            local_rank,
        )
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_arg), rank, world_size, local_rank


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def run(args: argparse.Namespace) -> dict[str, object]:
    device, rank, world_size, local_rank = initialize(args.device)
    torch.manual_seed(args.seed + rank)
    experiment = None if args.tiny else load_config(args.config)
    config = tiny_config() if args.tiny else experiment.model
    discriminator_config = (
        DiscriminatorConfig(fft_sizes=(16, 32), base_channels=4, maximum_channels=16)
        if args.tiny
        else experiment.discriminator
    )
    model = LargeStreamingCodec(config).to(device)
    discriminator = MultiScaleSTFTDiscriminator(discriminator_config).to(device)
    generator_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-4)
    training_model: torch.nn.Module = model
    training_discriminator: torch.nn.Module = discriminator
    if world_size > 1:
        device_ids = [local_rank] if device.type == "cuda" else None
        training_model = DistributedDataParallel(
            model, device_ids=device_ids, find_unused_parameters=True
        )
        training_discriminator = DistributedDataParallel(
            discriminator, device_ids=device_ids, broadcast_buffers=False
        )
    frames = max(2, round(args.audio_seconds * 16_000 / config.hop_length))
    waveform = torch.randn(args.batch_size, 1, frames * config.hop_length, device=device)
    hidden = torch.randn(args.batch_size, frames, config.student_dim, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    measured = []
    total_steps = args.warmup_steps + args.steps
    for index in range(total_steps):
        started = time.perf_counter()
        generator_optimizer.zero_grad(set_to_none=True)
        output = training_model(waveform, hidden)
        reconstruction = F.l1_loss(output["reconstruction"], waveform)
        objective = reconstruction
        if args.with_discriminator:
            set_requires_grad(discriminator, False)
            fake_outputs = training_discriminator(output["reconstruction"])
            with torch.no_grad():
                real_outputs = training_discriminator(waveform)
            objective = (
                objective
                + 0.1 * generator_adversarial_loss(fake_outputs)
                + feature_matching_loss(real_outputs, fake_outputs)
            )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        generator_optimizer.step()
        if args.with_discriminator:
            set_requires_grad(discriminator, True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            real_outputs = training_discriminator(waveform)
            fake_outputs = training_discriminator(output["reconstruction"].detach())
            discriminator_value = discriminator_loss(real_outputs, fake_outputs)
            discriminator_value.backward()
            discriminator_optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if index >= args.warmup_steps:
            measured.append(elapsed)
    maximum_elapsed = sum(measured)
    if world_size > 1:
        value = torch.tensor(maximum_elapsed, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        maximum_elapsed = float(value)
    report = {
        "status": "PASS",
        "device": str(device),
        "world_size": world_size,
        "batch_size_per_rank": args.batch_size,
        "audio_seconds_per_item": frames * config.hop_length / 16_000,
        "steps": args.steps,
        "with_discriminator": args.with_discriminator,
        "mean_step_seconds": maximum_elapsed / args.steps,
        "global_audio_seconds_per_second": (
            world_size
            * args.batch_size
            * args.steps
            * frames
            * config.hop_length
            / 16_000
            / maximum_elapsed
        ),
        "peak_memory_bytes_per_rank": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "parameters": parameter_breakdown(model),
        "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
    }
    if rank == 0:
        print(json.dumps(report, sort_keys=True))
        print("codec_training_benchmark=PASS")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Codec train/DDP resource benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/codec_63m.yaml"))
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-seconds", type=float, default=3.2)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--with-discriminator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.steps) <= 0 or min(args.audio_seconds, args.warmup_steps) < 0:
        parser.error("batch size and steps must be positive; durations cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
