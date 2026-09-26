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
from .losses import (
    ReconstructionLoss,
    masked_phone_kl,
    masked_smooth_l1,
    prosody_losses,
    trajectory_correlation_loss,
)
from .model import LargeStreamingCodec, PhoneAdversary, parameter_breakdown
from .schedule import weights_at


NEW_ADVERSARY_PREFIXES = ("dyn_phone_adversary.", "edit_phone_adversary.")


class TrainableCodec(torch.nn.Module):
    """Codec plus the training-only Z_edit speaker/style projection."""

    def __init__(
        self,
        codec: LargeStreamingCodec,
        speaker_target_dim: int,
        dyn_phone_grl_scale: float = 0.0,
        edit_phone_grl_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.codec = codec
        self.style_head = torch.nn.Sequential(
            torch.nn.Linear(codec.config.edit_dim, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, speaker_target_dim),
        )
        self.dyn_phone_adversary = PhoneAdversary(
            codec.config.dyn_dim, codec.config.vocab_size, dyn_phone_grl_scale
        )
        self.edit_phone_adversary = PhoneAdversary(
            codec.config.edit_dim, codec.config.vocab_size, edit_phone_grl_scale
        )

    def forward(
        self, waveform: torch.Tensor, student_hidden: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        output = self.codec(waveform, student_hidden)
        output["style_embedding"] = self.style_head(output["z_edit"].mean(-1))
        output["dyn_phone_logits"] = self.dyn_phone_adversary(output["z_dyn"])
        output["edit_phone_logits"] = self.edit_phone_adversary(
            output["z_edit"].transpose(1, 2)
        )
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
    phone = (
        masked_phone_kl(output["phone_logits"], batch["phone_target"], frames)
        if "phone_target" in batch
        else zero
    )
    dyn = (
        masked_smooth_l1(output["dyn_anchor"], batch["dyn_target"], frames)
        if "dyn_target" in batch
        else zero
    )
    prosody = (
        prosody_losses(output["prosody"], batch["prosody_target"], frames)
        if "prosody_target" in batch
        else {
            "normalized_f0": zero,
            "voicing": zero,
            "relative_energy": zero,
            "f0_delta": zero,
        }
    )
    dyn_phone = (
        masked_phone_kl(output["dyn_phone_logits"], batch["phone_target"], frames)
        if "phone_target" in batch
        else zero
    )
    edit_phone = (
        masked_phone_kl(output["edit_phone_logits"], batch["phone_target"], frames)
        if "phone_target" in batch
        else zero
    )
    return {
        "phone_anchor": phone,
        "dyn_anchor": dyn,
        "dyn_phone_adversary": dyn_phone,
        "edit_phone_adversary": edit_phone,
        **prosody,
    }


def view_metrics(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    reconstruction_loss: ReconstructionLoss,
    hop_length: int,
    auxiliary: bool = True,
) -> dict[str, torch.Tensor]:
    samples = batch["frames"] * hop_length
    reconstruction, reconstruction_metrics = reconstruction_loss(
        output["reconstruction"], batch["waveform"], samples
    )
    if not auxiliary:
        return {"reconstruction": reconstruction, **reconstruction_metrics}
    if "speaker_target" not in batch:
        raise ValueError("Every training row requires speaker_target_path")
    if "phone_target" not in batch:
        raise ValueError("Disentanglement training requires phone_target_path")
    if "prosody_target" not in batch:
        raise ValueError("Disentanglement training requires prosody_target_path")
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


def configure_training_mode(model: TrainableCodec, mode: str) -> list[torch.nn.Parameter]:
    set_requires_grad(model, mode == "joint")
    if mode == "quality":
        set_requires_grad(model.codec.decoder, True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"Training mode {mode!r} has no trainable parameters")
    return parameters


def _validate_initialization_config(payload: dict[str, Any], config: Any) -> None:
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Initialization checkpoint does not contain a config")
    for section, expected in (
        ("model", config.model.to_dict()),
        ("discriminator", config.to_dict()["discriminator"]),
    ):
        if checkpoint_config.get(section) != expected:
            raise RuntimeError(f"Initialization checkpoint {section} architecture differs")


def _load_initial_model_state(
    model: TrainableCodec, state: dict[str, torch.Tensor]
) -> list[str]:
    result = model.load_state_dict(state, strict=False)
    invalid_missing = [
        name
        for name in result.missing_keys
        if not name.startswith(NEW_ADVERSARY_PREFIXES)
    ]
    if invalid_missing or result.unexpected_keys:
        raise RuntimeError(
            "Initialization model state differs: "
            f"missing={invalid_missing}, unexpected={result.unexpected_keys}"
        )
    return list(result.missing_keys)


def initialize_from_checkpoint(
    model: TrainableCodec,
    discriminator: torch.nn.Module,
    ema: ExponentialMovingAverage,
    payload: dict[str, Any],
    config: Any,
) -> list[str]:
    _validate_initialization_config(payload, config)
    missing = _load_initial_model_state(model, payload["model"])
    discriminator.load_state_dict(payload["discriminator"], strict=True)
    loaded_shadow = payload.get("ema", {}).get("shadow", {})
    if not isinstance(loaded_shadow, dict):
        raise TypeError("Initialization checkpoint EMA shadow must be a dictionary")
    invalid_missing = [
        name
        for name in ema.shadow
        if name not in loaded_shadow and not name.startswith(NEW_ADVERSARY_PREFIXES)
    ]
    unexpected = sorted(set(loaded_shadow) - set(ema.shadow))
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Initialization EMA state differs: "
            f"missing={invalid_missing}, unexpected={unexpected}"
        )
    for name, current in ema.shadow.items():
        loaded = loaded_shadow.get(name)
        if loaded is None:
            continue
        if not torch.is_tensor(loaded) or loaded.shape != current.shape:
            raise RuntimeError(f"Initialization EMA state differs for {name}")
        current.copy_(loaded)
    return missing


def require_disentanglement_caches(
    source_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]
) -> None:
    def check(label: str, row: dict[str, Any]) -> None:
        missing = [
            name for name in ("phone_target_path", "prosody_target_path") if not row.get(name)
        ]
        if missing:
            raise ValueError(f"Joint disentanglement training requires {missing}: {label}")

    for index, row in enumerate(source_rows):
        check(f"source:{index}", row)
    for index, row in enumerate(pair_rows):
        for name in ("source", "sa"):
            check(f"pair:{index}:{name}", row[name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the large unified X-VC2 Codec")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speaker-target-dim", type=int, required=True)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--resume", type=Path)
    checkpoints.add_argument("--initialize-from", type=Path)
    parser.add_argument("--mode", choices=("joint", "quality"), default="joint")
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
    source_rows = read_jsonl(args.source_manifest)
    pair_rows = read_jsonl(args.pair_manifest)
    if args.mode == "joint":
        require_disentanglement_caches(source_rows, pair_rows)
    source_dataset = SourceDataset(source_rows, config.model.hop_length, segment_frames)
    pair_dataset = PairDataset(pair_rows, config.model.hop_length, segment_frames)
    codec = LargeStreamingCodec(config.model)
    model = TrainableCodec(
        codec,
        args.speaker_target_dim,
        config.loss.dyn_phone_grl_scale,
        config.loss.edit_phone_grl_scale,
    ).to(device)
    discriminator = MultiScaleSTFTDiscriminator(config.discriminator).to(device)
    ema = ExponentialMovingAverage(model, config.training.ema_decay)
    step = 0
    phase_start_step = config.schedule.gan_ramp_end
    initialized_missing: list[str] = []
    initialization_payload = None
    if args.initialize_from:
        initialization_payload = load_checkpoint(args.initialize_from, restore_rng=False)
        initialized_missing = initialize_from_checkpoint(
            model, discriminator, ema, initialization_payload, config
        )
        step = int(initialization_payload["step"])
        phase_start_step = step
    trainable_parameters = configure_training_mode(model, args.mode)
    generator_optimizer = torch.optim.AdamW(
        trainable_parameters,
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
    use_amp = device.type == "cuda" and config.training.amp != "none"
    amp_dtype = torch.bfloat16 if config.training.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and config.training.amp == "fp16")
    if args.resume:
        payload = load_checkpoint(args.resume)
        if payload["config"] != config.to_dict():
            raise RuntimeError("Resume configuration differs")
        if payload.get("mode", "joint") != args.mode:
            raise RuntimeError("Resume training mode differs")
        model.load_state_dict(payload["model"], strict=True)
        discriminator.load_state_dict(payload["discriminator"], strict=True)
        generator_optimizer.load_state_dict(payload["generator_optimizer"])
        discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
        ema.load_state_dict(payload["ema"])
        scaler.load_state_dict(payload.get("scaler", {}))
        step = int(payload["step"])
        phase_start_step = int(payload.get("phase_start_step", config.schedule.gan_ramp_end))
    training_model: torch.nn.Module = model
    training_discriminator: torch.nn.Module = discriminator
    if world_size > 1:
        training_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
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
    if end_step <= step:
        raise ValueError(
            f"Checkpoint step {step} has reached configured max_steps "
            f"{config.schedule.max_steps}"
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
        print(
            json.dumps(
                {
                    "parameters": parameter_breakdown(codec),
                    "config": config.to_dict(),
                    "mode": args.mode,
                    "start_step": step,
                    "phase_start_step": phase_start_step,
                    "initialized_missing_keys": initialized_missing,
                }
            )
        )

    interval_started = time.monotonic()
    interval_audio_seconds = 0.0
    interval_data_wait = 0.0
    interval_maximum_data_wait = 0.0
    interval_source_steps = 0
    interval_pair_steps = 0
    interval_steps = 0
    iterator = iter(loader)
    while step < end_step:
        wait_started = time.monotonic()
        prepared = next(iterator)
        data_wait = time.monotonic() - wait_started
        interval_data_wait += data_wait
        interval_maximum_data_wait = max(interval_maximum_data_wait, data_wait)
        next_step = int(prepared["step"])
        weights = weights_at(next_step, config.schedule, config.loss, phase_start_step)
        choose_pair = prepared["kind"] == "pair"
        interval_pair_steps += int(choose_pair)
        interval_source_steps += int(not choose_pair)
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
                    auxiliary=args.mode == "joint",
                )
            reconstruction = torch.stack([metrics[view]["reconstruction"] for view in views]).mean()
            zero = reconstruction.new_zeros(())
            auxiliary_names = (
                "edit_style",
                "phone_anchor",
                "dyn_anchor",
                "normalized_f0",
                "voicing",
                "relative_energy",
                "f0_delta",
                "dyn_phone_adversary",
                "edit_phone_adversary",
            )
            auxiliary = {
                name: (
                    torch.stack([metrics[view][name] for view in views]).mean()
                    if args.mode == "joint"
                    else zero
                )
                for name in auxiliary_names
            }
            sa_inv = reconstruction.new_zeros(())
            sa_dyn = reconstruction.new_zeros(())
            if choose_pair and args.mode == "joint":
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
                + weights.edit_style * auxiliary["edit_style"]
                + weights.phone_anchor * auxiliary["phone_anchor"]
                + weights.dyn_anchor * auxiliary["dyn_anchor"]
                + weights.normalized_f0 * auxiliary["normalized_f0"]
                + weights.voicing * auxiliary["voicing"]
                + weights.relative_energy * auxiliary["relative_energy"]
                + weights.f0_delta * auxiliary["f0_delta"]
                + weights.dyn_phone_adversary * auxiliary["dyn_phone_adversary"]
                + weights.edit_phone_adversary * auxiliary["edit_phone_adversary"]
                + weights.sa_inv * sa_inv
                + weights.sa_dyn * sa_dyn
                + weights.adversarial * adversarial
                + weights.feature_matching * feature_matching
            )
        scaler.scale(objective).backward()
        scaler.unscale_(generator_optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, config.training.gradient_clip
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
            maximum_data_wait = torch.tensor(
                interval_maximum_data_wait, device=device, dtype=torch.float64
            )
            peak_memory = torch.tensor(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                device=device,
                dtype=torch.float64,
            )
            if world_size > 1:
                dist.all_reduce(performance, op=dist.ReduceOp.SUM)
                dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                dist.all_reduce(maximum_data_wait, op=dist.ReduceOp.MAX)
                dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "batch_kind": "pair" if choose_pair else "source",
                            "objective": float(objective.detach()),
                            "reconstruction": float(reconstruction.detach()),
                            **{
                                name: float(value.detach())
                                for name, value in auxiliary.items()
                            },
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
                            "maximum_data_wait_seconds_per_rank_step": float(
                                maximum_data_wait
                            ),
                            "source_steps_in_interval": interval_source_steps,
                            "pair_steps_in_interval": interval_pair_steps,
                            "maximum_allocated_gib": float(peak_memory / 1024**3),
                            "weights": weights.__dict__,
                        }
                    )
                )
            interval_started = time.monotonic()
            interval_audio_seconds = 0.0
            interval_data_wait = 0.0
            interval_maximum_data_wait = 0.0
            interval_source_steps = 0
            interval_pair_steps = 0
            interval_steps = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        if rank == 0 and (step % config.training.save_interval == 0 or step == end_step):
            save_checkpoint(
                args.output_dir / f"step-{step:06d}.pt",
                step=step,
                mode=args.mode,
                phase_start_step=phase_start_step,
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
