"""Run a trained VC checkpoint with cached raw-audio chunks and a static reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio

from .checkpoint import load_checkpoint
from .preprocess import load_student
from .swap_probe import load_codec
from .vc import StreamingLatentConverter, StreamingVC
from .vc_condition import ReferenceFrontend, load_speaker_encoder
from .vc_data import VCPairDataConfig, _load_waveform
from .vc_train import restore_vc


@torch.no_grad()
def convert_stream(vc, waveform, condition, chunk_samples):
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    vc.eval()
    state = vc.init_streaming_state(condition)
    parts = []
    for position in range(0, waveform.shape[-1], chunk_samples):
        part, state = vc.forward_chunk(waveform[:, :, position : position + chunk_samples], state)
        parts.append(part)
    tail, _ = vc.forward_chunk(waveform[:, :, :0], state, is_final=True)
    return torch.cat([*parts, tail], dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vc-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-config", type=Path, required=True)
    parser.add_argument("--xvc-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-samples", type=int, default=3200)
    args = parser.parse_args()
    if args.chunk_samples <= 0:
        parser.error("--chunk-samples must be positive")
    device = torch.device(args.device)
    payload = load_checkpoint(args.vc_checkpoint, restore_rng=False)
    base, config = payload["base"], payload["config"]
    codec, _ = load_codec(
        args.codec_config, Path(base["codec_checkpoint"]), base["weights"], device
    )
    student = load_student(Path(base["student_checkpoint"]), device)
    converter_config = dict(config["converter"])
    converter_config["dilations"] = tuple(converter_config["dilations"])
    vc = StreamingVC(student, codec, StreamingLatentConverter(codec.config, **converter_config))
    restore_vc(vc, payload["vc"])
    vc.to(device).eval()
    frontend = (
        ReferenceFrontend(
            load_speaker_encoder(args.xvc_root, Path(base["speaker_model_dir"]), device),
            mels=converter_config["reference_mels"] or 80,
        )
        .to(device)
        .eval()
    )
    data_config = VCPairDataConfig(**config["data"])
    # File demo uses the training frontend once on the whole file, never per chunk.
    waveform = _load_waveform(str(args.source), data_config)[None].to(device)
    reference = _load_waveform(str(args.reference), data_config)[
        :, : int(data_config.max_reference_duration * 16_000)
    ][None].to(device)
    with torch.no_grad():
        speaker, mel, lengths = frontend(
            reference, torch.tensor([reference.shape[-1]], device=device)
        )
        condition = vc.converter.prepare_condition(speaker, mel, lengths)
        output = convert_stream(vc, waveform, condition, args.chunk_samples)
    if output.shape[-1] == 0:
        raise ValueError("Source is shorter than the Student receptive field")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), output[0].float().cpu(), 16_000)
    print(
        f"vc_stream=PASS input_samples={waveform.shape[-1]} output_samples={output.shape[-1]} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
