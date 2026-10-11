"""Two-stage paired latent VC training with a fixed Student/Codec coordinate system."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torchaudio
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import load_checkpoint, save_checkpoint
from .losses import ReconstructionLoss, masked_smooth_l1
from .preprocess import load_student
from .swap_probe import load_codec
from .train import initialize_data_worker, initialize_runtime, move
from .vc import StreamingLatentConverter, StreamingVC
from .vc_condition import ReferenceFrontend, load_speaker_encoder
from .vc_data import VCPairDataConfig, VCPairDataset


class SeededVCPairDataset(VCPairDataset):
    """Replay roles/crops at an epoch+item cursor without consuming model RNG."""

    seed = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Persistent worker copies must observe the parent's sampler epoch.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    @property
    def epoch(self):
        return int(self._epoch)

    @epoch.setter
    def epoch(self, value):
        self._epoch.fill_(value)

    def __getitem__(self, index):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed + self.epoch * len(self) + index)
            return super().__getitem__(index)


def build_training_loader(dataset, sampler, batch_size, num_workers, prefetch_factor, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers else None,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collate,
        worker_init_fn=initialize_data_worker,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(dataset.seed),
    )


class VCTrainingModel(torch.nn.Module):
    def __init__(self, vc: StreamingVC, frontend: ReferenceFrontend) -> None:
        super().__init__()
        self.vc = vc
        self.frontend = frontend

    def condition(self, batch):
        speaker, mel, lengths = self.frontend(
            batch["reference_waveform"], batch["reference_samples"]
        )
        return self.vc.converter.prepare_condition(speaker, mel, lengths)

    def forward(self, batch):
        if (batch["source"]["frames"] < 2).any():
            raise ValueError("VC training audio needs at least two complete 320-sample frames")
        return self.vc.forward_pair(
            batch,
            self.condition(batch),
            reconstruct_natural=self.vc.training_stage == "decoder",
        )


def build_optimizer(vc, training):
    groups = [
        {
            "params": list(vc.converter.parameters()),
            "lr": training["converter_learning_rate"],
            "name": "converter",
        }
    ]
    if vc.training_stage == "decoder":
        groups.append(
            {
                "params": list(vc.codec.decoder.parameters()),
                "lr": training["decoder_learning_rate"],
                "name": "decoder",
            }
        )
    return torch.optim.AdamW(groups, weight_decay=training["weight_decay"])


def valid_reconstruction_loss(predicted, target, lengths, reconstruction):
    # Trim before STFT, including the special case length < FFT size. Merely masking
    # STFT frames would let the first padded FFT read invalid batch samples.
    total = predicted.new_zeros(())
    for length in lengths.unique().tolist():
        selected = lengths == length
        value, _ = reconstruction(
            predicted[selected, :, :length], target[selected, :, :length], lengths[selected]
        )
        total = total + value * selected.sum() / lengths.numel()
    return total


def compute_losses(paired, batch, reconstruction, weights):
    prediction, target = paired["prediction"], paired["target_latents"]
    frames = paired["frames"]
    if (frames <= 0).any():
        raise ValueError("Pair contains no mutually supported Student/Codec frame")
    spectral = valid_reconstruction_loss(
        prediction["reconstruction"],
        paired["target_waveform"],
        prediction["output_lengths"],
        reconstruction,
    )
    metrics = {
        "paired_reconstruction": spectral,
        "dyn": masked_smooth_l1(prediction["z_dyn"], target["z_dyn"], frames),
        "edit": masked_smooth_l1(
            prediction["z_edit"].transpose(1, 2), target["z_edit"].transpose(1, 2), frames
        ),
    }
    if "natural_reconstruction" in paired:
        waveform = paired["natural_reconstruction"]
        metrics["natural_reconstruction"] = valid_reconstruction_loss(
            waveform,
            batch["natural"]["waveform"][:, :, : waveform.shape[-1]],
            paired["natural_lengths"],
            reconstruction,
        )
    objective = sum(weights[name] * value for name, value in metrics.items())
    return objective, metrics


def vc_state(vc):
    # Student and latent encoders are loaded from the pinned base checkpoints.
    return {"converter": vc.converter.state_dict(), "decoder": vc.codec.decoder.state_dict()}


def restore_vc(vc, state):
    vc.converter.load_state_dict(state["converter"], strict=True)
    vc.codec.decoder.load_state_dict(state["decoder"], strict=True)


@torch.no_grad()
def validate(model, loader, device, reconstruction, weights, output_dir, audio_examples):
    was_training = model.training
    model.eval()
    totals, count = {}, 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        batch = move(batch, device)
        paired = model(batch)
        _, metrics = compute_losses(paired, batch, reconstruction, weights)
        size = batch["source"]["waveform"].shape[0]
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * size
        if count < audio_examples:
            for index in range(min(size, audio_examples - count)):
                length = int(paired["prediction"]["output_lengths"][index])
                reference_samples = int(batch["reference_samples"][index])
                target_latents = paired["target_latents"]
                codec_target = model.vc.codec.decoder(
                    model.vc.codec.fusion(
                        target_latents["z_inv"][index : index + 1],
                        target_latents["z_dyn"][index : index + 1],
                        target_latents["z_edit"][index : index + 1],
                    )
                )[0]
                waves = {
                    "source": batch["source"]["waveform"][index, :, :length],
                    "target": paired["target_waveform"][index, :, :length],
                    "codec_target": codec_target[:, :length],
                    "converted": paired["prediction"]["reconstruction"][index, :, :length],
                    "reference": batch["reference_waveform"][index, :, :reference_samples],
                }
                for name, wave in waves.items():
                    torchaudio.save(
                        str(output_dir / f"{count + index:03d}-{name}.wav"),
                        wave.float().cpu(),
                        16_000,
                    )
                (output_dir / f"{count + index:03d}.json").write_text(
                    json.dumps(
                        {
                            "source": batch["source_wav_path"][index],
                            "target": batch["target_wav_path"][index],
                            "reference": batch["reference_wav_path"][index],
                            "reference_kind": batch["reference_kind"][index],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        count += size
    model.train(was_training)
    metrics = {name: value / count for name, value in totals.items()}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def validate_config(config):
    VCPairDataConfig(**config["data"])
    training = config["training"]
    if training["amp"] not in {"none", "bf16", "fp16"}:
        raise ValueError("amp must be none, bf16 or fp16")
    for name in (
        "max_steps",
        "log_interval",
        "save_interval",
        "val_interval",
        "converter_learning_rate",
        "decoder_learning_rate",
        "gradient_clip",
    ):
        if training[name] <= 0:
            raise ValueError(f"training.{name} must be positive")
    if training["decoder_learning_rate"] > training["converter_learning_rate"]:
        raise ValueError("Decoder learning rate must not exceed Converter learning rate")
    if set(config["loss"]) != {"dyn", "edit", "paired_reconstruction", "natural_reconstruction"}:
        raise ValueError(
            "VC loss keys must be dyn/edit/paired_reconstruction/natural_reconstruction"
        )
    if min(config["loss"].values()) < 0 or config["loss"]["natural_reconstruction"] <= 0:
        raise ValueError(
            "Loss weights must be nonnegative; natural reconstruction must be positive"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--codec-config", type=Path, required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--speaker-model-dir", type=Path, required=True)
    parser.add_argument("--xvc-root", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("converter", "decoder"), default="converter")
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--resume", type=Path)
    checkpoints.add_argument("--initialize-vc", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0, help="Audio loader workers per rank")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Prefetched batches per worker; used only when num-workers > 0",
    )
    parser.add_argument("--steps", type=int, help="Additional optimizer steps this invocation")
    parser.add_argument("--val-pairs", type=int, default=16)
    parser.add_argument("--audio-examples", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.val_pairs) <= 0 or min(args.num_workers, args.audio_examples) < 0:
        parser.error("Invalid batch/worker/validation sizes")
    if args.prefetch_factor <= 0:
        parser.error("--prefetch-factor must be positive")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.stage == "decoder" and not (args.initialize_vc or args.resume):
        parser.error("Decoder adaptation requires --initialize-vc or --resume")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    training = config["training"]
    device, rank, world_size, local_rank = initialize_runtime(args.device)
    random.seed(training["seed"] + rank)
    torch.manual_seed(training["seed"] + rank)
    data_config = VCPairDataConfig(**config["data"])
    dataset = SeededVCPairDataset(args.train_manifest, data_config)
    dataset.seed = training["seed"]
    validation = VCPairDataset(args.val_manifest, data_config, random_crop=False)
    # All generated variants of one natural utterance must stay in one split.
    for key in ("target_utt", "target_wav_path"):
        if {row[key] for row in dataset.rows} & {row[key] for row in validation.rows}:
            raise ValueError(f"Train/validation overlap in original {key}")
    validation.rows = validation.rows[: args.val_pairs]
    codec, codec_step = load_codec(args.codec_config, args.codec_checkpoint, args.weights, device)
    student = load_student(args.student_checkpoint, device)
    converter_config = dict(config["converter"])
    converter_config["dilations"] = tuple(converter_config["dilations"])
    vc = StreamingVC(student, codec, StreamingLatentConverter(codec.config, **converter_config))
    vc.set_training_stage(args.stage)
    speaker = load_speaker_encoder(args.xvc_root, args.speaker_model_dir, device)
    if speaker.embedding_dim != vc.converter.condition_dim:
        raise ValueError("Speaker encoder and Converter condition dimensions differ")
    model = (
        VCTrainingModel(
            vc, ReferenceFrontend(speaker, mels=converter_config["reference_mels"] or 80)
        )
        .to(device)
        .train()
    )
    base = {
        "codec_checkpoint": str(args.codec_checkpoint.resolve()),
        "codec_step": codec_step,
        "student_checkpoint": str(args.student_checkpoint.resolve()),
        "weights": args.weights,
        "codec_model": codec.config.to_dict(),
        "speaker_model_dir": str(args.speaker_model_dir.resolve()),
    }
    step, epoch, offset = 0, 0, 0
    payload = None
    if args.resume or args.initialize_vc:
        payload = load_checkpoint(args.resume or args.initialize_vc, restore_rng=bool(args.resume))
        if payload["base"] != base or payload["config"]["converter"] != config["converter"]:
            raise ValueError("Pinned base checkpoints or Converter architecture differ")
        restore_vc(vc, payload["vc"])
    optimizer = build_optimizer(vc, training)
    use_amp = device.type == "cuda" and training["amp"] != "none"
    dtype = torch.bfloat16 if training["amp"] == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and training["amp"] == "fp16")
    data_identity = {
        "train_manifest": str(args.train_manifest.resolve()),
        "val_manifest": str(args.val_manifest.resolve()),
        "batch_size": args.batch_size,
        "world_size": world_size,
    }
    if args.resume:
        if (
            payload["stage"] != args.stage
            or payload["config"] != config
            or payload["data"] != data_identity
        ):
            raise ValueError("Resume stage/config/data/batch/world size differs")
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        step, epoch, offset = payload["step"], payload["epoch"], payload["offset"]
    end = step + args.steps if args.steps else training["max_steps"]
    if end <= step:
        raise ValueError("No steps remaining; use --steps for an explicit extension")
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, seed=training["seed"], drop_last=False
    )
    loader = build_training_loader(
        dataset, sampler, args.batch_size, args.num_workers, args.prefetch_factor, device
    )
    val_loader = DataLoader(validation, batch_size=1, collate_fn=validation.collate, num_workers=0)
    wrapped = (
        DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        if world_size > 1
        else model
    )
    reconstruction = ReconstructionLoss()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "base": base,
                    "loader": {
                        "num_workers_per_rank": args.num_workers,
                        "prefetch_factor": loader.prefetch_factor,
                        "persistent_workers": loader.persistent_workers,
                    },
                    "trainable": {
                        group["name"]: sum(p.numel() for p in group["params"])
                        for group in optimizer.param_groups
                    },
                }
            ),
            flush=True,
        )
    interval_seconds, interval_data_wait, interval_samples, interval_steps = 0.0, 0.0, 0, 0
    while step < end:
        sampler.set_epoch(epoch)
        dataset.epoch = epoch
        previous_end = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            data_wait = time.perf_counter() - previous_end
            if batch_index < offset:
                previous_end = time.perf_counter()
                continue
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                paired = wrapped(batch)
                objective, metrics = compute_losses(paired, batch, reconstruction, config["loss"])
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Nonfinite VC objective at step {step + 1}")
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, training["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            step += 1
            offset = batch_index + 1
            logging_step = step == 1 or step % training["log_interval"] == 0
            if logging_step and device.type == "cuda":
                torch.cuda.synchronize(device)
            interval_seconds += time.perf_counter() - previous_end
            interval_data_wait += data_wait
            interval_samples += batch["source"]["waveform"].shape[0]
            interval_steps += 1
            if logging_step:
                values = torch.stack([objective.detach(), *[v.detach() for v in metrics.values()]])
                performance = torch.tensor(
                    [interval_seconds, interval_data_wait, interval_samples],
                    dtype=torch.float64,
                    device=device,
                )
                reports = [torch.empty_like(performance) for _ in range(world_size)]
                if world_size > 1:
                    dist.all_reduce(values)
                    values /= world_size
                    dist.all_gather(reports, performance)
                else:
                    reports[0] = performance
                if rank == 0:
                    timings = torch.stack(reports).cpu()
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "stage": args.stage,
                                "objective": float(values[0]),
                                **dict(zip(metrics, map(float, values[1:]))),
                                "gradient_norm": float(gradient_norm),
                                "learning_rates": {
                                    g["name"]: g["lr"] for g in optimizer.param_groups
                                },
                                "reference_kinds": batch["reference_kind"],
                                "mean_step_seconds_by_rank": (
                                    timings[:, 0] / interval_steps
                                ).tolist(),
                                "mean_data_wait_seconds_by_rank": (
                                    timings[:, 1] / interval_steps
                                ).tolist(),
                                "global_pairs_per_second": float(
                                    timings[:, 2].sum() / timings[:, 0].max()
                                ),
                            }
                        ),
                        flush=True,
                    )
                interval_seconds, interval_data_wait, interval_samples, interval_steps = (
                    0.0,
                    0.0,
                    0,
                    0,
                )
            if step % training["val_interval"] == 0 or step == end:
                if rank == 0:
                    scores = validate(
                        model,
                        val_loader,
                        device,
                        reconstruction,
                        config["loss"],
                        args.output_dir / f"validation-{step:06d}",
                        args.audio_examples,
                    )
                    print(json.dumps({"step": step, "validation": scores}), flush=True)
                if world_size > 1:
                    dist.barrier()
            if rank == 0 and (step % training["save_interval"] == 0 or step == end):
                save_checkpoint(
                    args.output_dir / f"step-{step:06d}.pt",
                    step=step,
                    epoch=epoch,
                    offset=offset,
                    stage=args.stage,
                    base=base,
                    config=config,
                    data=data_identity,
                    vc=vc_state(vc),
                    optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(),
                )
            if step >= end:
                break
            # Exclude validation, checkpoint writes and log collectives from training timing.
            previous_end = time.perf_counter()
        if step < end:
            epoch += 1
            offset = 0
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
