"""Causal latent conversion and a single-stream Student/Codec VC pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .causal import CausalConv1d, ChannelLayerNorm
from .config import CodecConfig
from .model import LargeStreamingCodec


class ConditionalCausalBlock(torch.nn.Module):
    def __init__(self, channels: int, condition_dim: int, dilation: int) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm(channels)
        self.condition = torch.nn.Linear(condition_dim, channels * 2)
        self.conv = CausalConv1d(channels, channels, 3, dilation=dilation)
        self.output = torch.nn.Conv1d(channels, channels, 1)

    def _modulate(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        return F.silu(self.norm(values) * (1 + scale[:, :, None]) + shift[:, :, None])

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return values + self.output(F.silu(self.conv(self._modulate(values, condition))))

    def forward_chunk(self, values, condition, state):
        delta, state = self.conv.forward_chunk(self._modulate(values, condition), state)
        return values + self.output(F.silu(delta)), state


@dataclass
class ReferenceCondition:
    speaker: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor


class ReferenceAttention(torch.nn.Module):
    """Source queries attend only to static reference memory, never updated source tokens."""

    def __init__(self, channels: int, mels: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(mels, channels, 3, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv1d(channels, channels, 1),
        )
        self.query = torch.nn.Linear(channels, channels)
        self.key = torch.nn.Linear(channels, channels)
        self.value = torch.nn.Linear(channels, channels)
        self.output = torch.nn.Linear(channels, channels)

    def prepare(self, speaker, mel, lengths):
        memory = self.encoder(mel).transpose(1, 2)
        valid = (
            torch.arange(memory.shape[1], device=mel.device)[None] < ((lengths + 1) // 2)[:, None]
        )
        if not valid.any(-1).all():
            raise ValueError("Each reference must contain at least one valid Mel frame")
        return ReferenceCondition(speaker, self.key(memory), self.value(memory), valid)

    def forward(self, values, condition):
        query = self.query(values.transpose(1, 2))
        attended = F.scaled_dot_product_attention(
            query[:, None],
            condition.keys[:, None],
            condition.values[:, None],
            attn_mask=condition.valid[:, None, None],
            dropout_p=0.0,
        )[:, 0]
        return values + self.output(attended).transpose(1, 2)


@dataclass
class ConverterState:
    condition: torch.Tensor | ReferenceCondition
    blocks: list[torch.Tensor | None]
    finalized: bool = False


class StreamingLatentConverter(torch.nn.Module):
    """Keep inv unchanged; predict conditional dyn/edit residuals at the source frame rate."""

    def __init__(
        self,
        config: CodecConfig,
        condition_dim: int = 192,
        channels: int = 256,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        reference_mels: int = 0,
    ) -> None:
        super().__init__()
        if min(condition_dim, channels) <= 0 or not dilations or min(dilations) <= 0:
            raise ValueError("Converter dimensions and dilations must be positive")
        self.config = config
        self.condition_dim = condition_dim
        self.reference = ReferenceAttention(channels, reference_mels) if reference_mels else None
        self.input = torch.nn.Conv1d(config.inv_dim + config.dyn_dim + config.edit_dim, channels, 1)
        self.blocks = torch.nn.ModuleList(
            ConditionalCausalBlock(channels, condition_dim, dilation) for dilation in dilations
        )
        self.output = torch.nn.Conv1d(channels, config.dyn_dim + config.edit_dim, 1)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def _input(self, z_inv, z_dyn, z_edit, condition):
        batch, frames, _ = z_inv.shape
        expected = (
            (batch, frames, self.config.inv_dim),
            (batch, frames, self.config.dyn_dim),
            (batch, self.config.edit_dim, frames),
            (batch, self.condition_dim),
        )
        actual = tuple(tuple(value.shape) for value in (z_inv, z_dyn, z_edit, condition))
        if actual != expected:
            raise ValueError(f"Expected latent/condition shapes {expected}, got {actual}")
        values = torch.cat((z_inv.transpose(1, 2), z_dyn.transpose(1, 2), z_edit), dim=1)
        return self.input(values)

    def _output(self, values, z_inv, z_dyn, z_edit):
        dyn_delta, edit_delta = self.output(F.silu(values)).split(
            (self.config.dyn_dim, self.config.edit_dim), dim=1
        )
        return {
            "z_inv": z_inv,
            "z_dyn": z_dyn + dyn_delta.transpose(1, 2),
            "z_edit": z_edit + edit_delta,
        }

    def forward(self, z_inv, z_dyn, z_edit, condition):
        if z_inv.shape[1] == 0:
            return {"z_inv": z_inv, "z_dyn": z_dyn, "z_edit": z_edit}
        speaker = condition.speaker if isinstance(condition, ReferenceCondition) else condition
        values = self._input(z_inv, z_dyn, z_edit, speaker)
        for block in self.blocks:
            values = block(values, speaker)
        if self.reference is not None:
            if not isinstance(condition, ReferenceCondition):
                raise ValueError("Mel-conditioned Converter requires a prepared reference")
            values = self.reference(values, condition)
        return self._output(values, z_inv, z_dyn, z_edit)

    def prepare_condition(self, speaker, mel, lengths):
        if self.reference is None:
            return speaker
        return self.reference.prepare(speaker, mel, lengths)

    def init_streaming_state(self, condition) -> ConverterState:
        speaker = condition.speaker if isinstance(condition, ReferenceCondition) else condition
        if speaker.ndim != 2 or speaker.shape[-1] != self.condition_dim:
            raise ValueError(f"Expected condition [batch, {self.condition_dim}]")
        if isinstance(condition, ReferenceCondition):
            condition = ReferenceCondition(
                *(
                    value.detach().clone()
                    for value in (
                        condition.speaker,
                        condition.keys,
                        condition.values,
                        condition.valid,
                    )
                )
            )
        else:
            condition = condition.detach().clone()
        return ConverterState(condition, [None] * len(self.blocks))

    @torch.no_grad()
    def forward_chunk(self, z_inv, z_dyn, z_edit, state: ConverterState, is_final=False):
        if self.training:
            raise RuntimeError("forward_chunk requires eval mode")
        if state.finalized:
            raise RuntimeError("Converter state has already been finalized")
        if z_inv.shape[1] == 0:
            state.finalized = is_final
            return {"z_inv": z_inv, "z_dyn": z_dyn, "z_edit": z_edit}, state
        speaker = (
            state.condition.speaker
            if isinstance(state.condition, ReferenceCondition)
            else state.condition
        )
        values = self._input(z_inv, z_dyn, z_edit, speaker)
        for index, block in enumerate(self.blocks):
            values, state.blocks[index] = block.forward_chunk(values, speaker, state.blocks[index])
        if self.reference is not None:
            if not isinstance(state.condition, ReferenceCondition):
                raise ValueError("Mel-conditioned Converter requires a prepared reference")
            values = self.reference(values, state.condition)
        state.finalized = is_final
        return self._output(values, z_inv, z_dyn, z_edit), state


@dataclass
class VCStreamingState:
    student: Any
    converter: ConverterState
    waveform_buffer: torch.Tensor
    edit_buffer: torch.Tensor
    acoustic_encoder: dict | None = None
    fusion: dict | None = None
    decoder: dict | None = None
    finalized: bool = False


class StreamingVC(torch.nn.Module):
    """Frozen latent encoders + causal Converter, with optional Decoder adaptation.

    A speaker encoder supplies the global target condition before streaming starts.
    Raw streaming accepts one 16 kHz mono stream and arbitrary sample boundaries.
    Output covers only complete frames supported by both Student and Codec.
    """

    def __init__(
        self,
        student: torch.nn.Module,
        codec: LargeStreamingCodec,
        converter: StreamingLatentConverter,
    ) -> None:
        super().__init__()
        if student.stride_samples != codec.config.hop_length:
            raise ValueError("Student stride must match Codec hop length")
        if student.config.model_dim != codec.config.student_dim:
            raise ValueError("Student hidden dimension must match Codec student_dim")
        if converter.config != codec.config:
            raise ValueError("Converter and Codec configurations must match")
        self.student = student.eval().requires_grad_(False)
        self.codec = codec.eval().requires_grad_(False)
        self.converter = converter
        self.training_stage = "converter"

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"converter", "decoder"}:
            raise ValueError("VC stage must be converter or decoder")
        self.training_stage = stage
        self.student.requires_grad_(False)
        self.codec.requires_grad_(False)
        self.converter.requires_grad_(True)
        self.codec.decoder.requires_grad_(stage == "decoder")
        self.train(self.training)

    def train(self, mode: bool = True) -> StreamingVC:
        super().train(mode)
        self.student.eval()
        self.codec.eval()
        self.codec.decoder.train(mode and self.training_stage == "decoder")
        return self

    @torch.no_grad()
    def _encode(self, waveform: torch.Tensor, lengths: torch.Tensor):
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("Expected mono waveform [batch, 1, samples]")
        if (
            lengths.shape != (waveform.shape[0],)
            or (lengths < 0).any()
            or (lengths > waveform.shape[-1]).any()
        ):
            raise ValueError("Expected valid sample lengths [batch] within the waveform")
        if int(lengths.max()) < self.student.receptive_field_samples:
            return {
                "z_inv": waveform.new_empty(waveform.shape[0], 0, self.codec.config.inv_dim),
                "z_dyn": waveform.new_empty(waveform.shape[0], 0, self.codec.config.dyn_dim),
                "z_edit": waveform.new_empty(waveform.shape[0], self.codec.config.edit_dim, 0),
                "frames": torch.zeros_like(lengths),
            }
        waveform = waveform[:, :, : int(lengths.max())]
        student = self.student(waveform[:, 0], lengths)
        hidden = student["hidden_states"]
        frames = min(hidden.shape[1], waveform.shape[-1] // self.codec.config.hop_length)
        keep = self.codec.keep_head(hidden[:, :frames])
        edit = self.codec.acoustic_encoder(waveform[:, :, : frames * self.codec.config.hop_length])
        return {
            "z_inv": keep["z_inv"],
            "z_dyn": keep["z_dyn"],
            "z_edit": edit,
            "frames": torch.minimum(
                student["output_lengths"], lengths // self.codec.config.hop_length
            ),
        }

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor, condition: torch.Tensor):
        source = self._encode(waveform, lengths)
        output = self.converter(source["z_inv"], source["z_dyn"], source["z_edit"], condition)
        output["reconstruction"] = (
            self.codec.decoder(
                self.codec.fusion(output["z_inv"], output["z_dyn"], output["z_edit"])
            )
            if output["z_inv"].shape[1]
            else waveform[:, :, :0]
        )
        output["output_lengths"] = source["frames"] * self.codec.config.hop_length
        return output

    def forward_pair(self, batch: dict, condition, *, reconstruct_natural: bool = False) -> dict:
        """Convert the assigned source; encode the assigned target as a frozen teacher."""
        hop = self.codec.config.hop_length
        prediction = self(batch["source"]["waveform"], batch["source"]["frames"] * hop, condition)
        target = self._encode(batch["target"]["waveform"], batch["target"]["frames"] * hop)
        shared = min(prediction["z_inv"].shape[1], target["z_inv"].shape[1])
        for name in ("z_inv", "z_dyn", "z_edit"):
            axis = 2 if name == "z_edit" else 1
            prediction[name] = prediction[name].narrow(axis, 0, shared)
            target[name] = target[name].narrow(axis, 0, shared)
        prediction["reconstruction"] = prediction["reconstruction"][:, :, : shared * hop]
        frames = torch.minimum(prediction["output_lengths"] // hop, target["frames"]).clamp_max(
            shared
        )
        prediction["output_lengths"] = frames * hop
        target["frames"] = frames
        result = {
            "prediction": prediction,
            "target_latents": target,
            "target_waveform": batch["target"]["waveform"][:, :, : shared * hop],
            "frames": frames,
        }
        if reconstruct_natural:
            natural = batch["natural"]
            encoded = self._encode(natural["waveform"], natural["frames"] * hop)
            result["natural_reconstruction"] = self.codec.decoder(
                self.codec.fusion(encoded["z_inv"], encoded["z_dyn"], encoded["z_edit"])
            )
            result["natural_lengths"] = encoded["frames"] * hop
        return result

    def init_streaming_state(self, condition) -> VCStreamingState:
        speaker = condition.speaker if isinstance(condition, ReferenceCondition) else condition
        if speaker.ndim != 2 or speaker.shape[0] != 1:
            raise ValueError("Raw streaming supports one target condition [1, condition_dim]")
        parameter = next(self.codec.parameters())
        return VCStreamingState(
            student=self.student.init_streaming_state(speaker.device, parameter.dtype),
            converter=self.converter.init_streaming_state(condition),
            waveform_buffer=torch.empty(1, 1, 0, device=speaker.device, dtype=parameter.dtype),
            edit_buffer=torch.empty(
                1, self.codec.config.edit_dim, 0, device=speaker.device, dtype=parameter.dtype
            ),
        )

    @torch.no_grad()
    def forward_chunk(self, waveform: torch.Tensor, state: VCStreamingState, is_final=False):
        if self.training:
            raise RuntimeError("forward_chunk requires eval mode")
        if state.finalized:
            raise RuntimeError("VC state has already been finalized")
        if waveform.ndim != 3 or tuple(waveform.shape[:2]) != (1, 1):
            raise ValueError("Expected one mono stream [1, 1, samples]")
        hop = self.codec.config.hop_length
        state.waveform_buffer = torch.cat((state.waveform_buffer, waveform), dim=-1)
        samples = state.waveform_buffer.shape[-1] // hop * hop
        if samples:
            edit, state.acoustic_encoder = self.codec.acoustic_encoder.forward_chunk(
                state.waveform_buffer[:, :, :samples], state.acoustic_encoder
            )
            state.waveform_buffer = state.waveform_buffer[:, :, samples:]
            state.edit_buffer = torch.cat((state.edit_buffer, edit), dim=-1)
        student, state.student = self.student.forward_chunk(waveform[:, 0], state.student, is_final)
        hidden = student["hidden_states"]
        frames = hidden.shape[1]
        if frames > state.edit_buffer.shape[-1]:
            raise RuntimeError(
                "Student emitted more frames than the acoustic encoder has available"
            )
        if frames:
            keep = self.codec.keep_head(hidden)
            output, state.converter = self.converter.forward_chunk(
                keep["z_inv"],
                keep["z_dyn"],
                state.edit_buffer[:, :, :frames],
                state.converter,
                is_final,
            )
            state.edit_buffer = state.edit_buffer[:, :, frames:]
            fused, state.fusion = self.codec.fusion.forward_chunk(
                output["z_inv"], output["z_dyn"], output["z_edit"], state.fusion
            )
            reconstruction, state.decoder = self.codec.decoder.forward_chunk(fused, state.decoder)
        else:
            reconstruction = waveform.new_empty(1, 1, 0)
        if is_final:
            state.waveform_buffer = state.waveform_buffer[:, :, :0]
            state.edit_buffer = state.edit_buffer[:, :, :0]
            state.converter.finalized = True
            state.finalized = True
        return reconstruction, state
