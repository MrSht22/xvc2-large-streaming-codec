from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .causal import CausalConv1d, ChannelLayerNorm, MultiKernelResidual, ResidualUnit
from .config import CodecConfig


class KeepHead(torch.nn.Module):
    """Frame-local split of frozen Student states into content and expressive dynamics."""

    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        self.input_norm = torch.nn.LayerNorm(config.student_dim)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(config.student_dim, config.keep_trunk_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.keep_trunk_dim, config.keep_trunk_dim),
            torch.nn.GELU(),
        )
        self.inv = torch.nn.Linear(config.keep_trunk_dim, config.inv_dim)
        self.dyn = torch.nn.Linear(config.keep_trunk_dim, config.dyn_dim)
        self.inv_norm = torch.nn.LayerNorm(config.inv_dim)
        self.dyn_norm = torch.nn.LayerNorm(config.dyn_dim)
        self.phone_head = torch.nn.Linear(config.inv_dim, config.vocab_size)
        self.dyn_anchor = torch.nn.Linear(config.dyn_dim, config.dyn_dim)
        self.prosody_head = torch.nn.Linear(config.dyn_dim, 4)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        trunk = self.trunk(self.input_norm(hidden))
        z_inv = self.inv_norm(self.inv(trunk))
        z_dyn = self.dyn_norm(self.dyn(trunk))
        return {
            "z_inv": z_inv,
            "z_dyn": z_dyn,
            "phone_logits": self.phone_head(z_inv),
            "dyn_anchor": self.dyn_anchor(z_dyn),
            "prosody": self.prosody_head(z_dyn),
        }


class EncoderBlock(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int, rate: int) -> None:
        super().__init__()
        self.residuals = torch.nn.ModuleList(
            ResidualUnit(input_channels, 7, dilation) for dilation in (1, 3, 9)
        )
        self.downsample = CausalConv1d(input_channels, output_channels, 2 * rate, stride=rate)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for residual in self.residuals:
            values = residual(values)
        return self.downsample(F.silu(values))

    def forward_chunk(
        self, values: torch.Tensor, state: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        state = state or {}
        states = state.get("residuals", [None] * len(self.residuals))
        next_states = []
        for residual, item in zip(self.residuals, states):
            values, item = residual.forward_chunk(values, item)
            next_states.append(item)
        values, downsample = self.downsample.forward_chunk(F.silu(values), state.get("downsample"))
        return values, {"residuals": next_states, "downsample": downsample}


class AcousticEncoder(torch.nn.Module):
    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        channels = config.encoder_channels
        self.input = CausalConv1d(1, channels, 7)
        blocks = []
        for rate in config.encoder_rates:
            blocks.append(EncoderBlock(channels, channels * 2, rate))
            channels *= 2
        self.blocks = torch.nn.ModuleList(blocks)
        self.output = CausalConv1d(channels, config.edit_dim, 3)
        self.norm = ChannelLayerNorm(config.edit_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        values = self.input(waveform)
        for block in self.blocks:
            values = block(values)
        return self.norm(self.output(F.silu(values)))

    def forward_chunk(
        self, waveform: torch.Tensor, state: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        state = state or {}
        values, input_state = self.input.forward_chunk(waveform, state.get("input"))
        states = state.get("blocks", [None] * len(self.blocks))
        next_states = []
        for block, item in zip(self.blocks, states):
            values, item = block.forward_chunk(values, item)
            next_states.append(item)
        values, output_state = self.output.forward_chunk(F.silu(values), state.get("output"))
        return self.norm(values), {
            "input": input_state,
            "blocks": next_states,
            "output": output_state,
        }


class Fusion(torch.nn.Module):
    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        self.inv = torch.nn.Linear(config.inv_dim, config.inv_projection_dim)
        self.dyn = torch.nn.Linear(config.dyn_dim, config.dyn_projection_dim)
        self.edit = torch.nn.Linear(config.edit_dim, config.edit_projection_dim)
        input_dim = (
            config.inv_projection_dim + config.dyn_projection_dim + config.edit_projection_dim
        )
        self.input = CausalConv1d(input_dim, config.fusion_channels, 1)
        self.residuals = torch.nn.ModuleList(
            ResidualUnit(config.fusion_channels, config.fusion_kernel_size, dilation)
            for dilation in config.fusion_dilations
        )

    def _project(
        self, z_inv: torch.Tensor, z_dyn: torch.Tensor, z_edit: torch.Tensor
    ) -> torch.Tensor:
        if len({z_inv.shape[1], z_dyn.shape[1], z_edit.shape[2]}) != 1:
            raise ValueError("Z_inv, Z_dyn, and Z_edit frame counts must match")
        values = torch.cat(
            (self.inv(z_inv), self.dyn(z_dyn), self.edit(z_edit.transpose(1, 2))), dim=-1
        )
        return values.transpose(1, 2)

    def forward(
        self, z_inv: torch.Tensor, z_dyn: torch.Tensor, z_edit: torch.Tensor
    ) -> torch.Tensor:
        values = self.input(self._project(z_inv, z_dyn, z_edit))
        for residual in self.residuals:
            values = residual(values)
        return values

    def forward_chunk(
        self,
        z_inv: torch.Tensor,
        z_dyn: torch.Tensor,
        z_edit: torch.Tensor,
        state: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        state = state or {}
        values, input_state = self.input.forward_chunk(
            self._project(z_inv, z_dyn, z_edit), state.get("input")
        )
        states = state.get("residuals", [None] * len(self.residuals))
        next_states = []
        for residual, item in zip(self.residuals, states):
            values, item = residual.forward_chunk(values, item)
            next_states.append(item)
        return values, {"input": input_state, "residuals": next_states}


class DecoderBlock(torch.nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, rate: int, config: CodecConfig
    ) -> None:
        super().__init__()
        self.rate = rate
        self.upsample = CausalConv1d(input_channels, output_channels, 2 * rate)
        self.residual = MultiKernelResidual(
            output_channels, config.residual_kernels, config.residual_dilations
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.repeat_interleave(self.rate, dim=-1)
        return self.residual(self.upsample(F.silu(values)))

    def forward_chunk(
        self, values: torch.Tensor, state: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        state = state or {}
        values = values.repeat_interleave(self.rate, dim=-1)
        values, upsample = self.upsample.forward_chunk(F.silu(values), state.get("upsample"))
        values, residual = self.residual.forward_chunk(values, state.get("residual"))
        return values, {"upsample": upsample, "residual": residual}


class Decoder(torch.nn.Module):
    """Unconditional causal waveform Decoder; target speaker never enters here."""

    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        input_channels = config.fusion_channels
        blocks = []
        for rate, channels in zip(config.decoder_rates, config.decoder_channels):
            blocks.append(DecoderBlock(input_channels, channels, rate, config))
            input_channels = channels
        self.blocks = torch.nn.ModuleList(blocks)
        self.output = CausalConv1d(input_channels, 1, 7)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            values = block(values)
        return torch.tanh(self.output(F.silu(values)))

    def forward_chunk(
        self, values: torch.Tensor, state: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        state = state or {}
        states = state.get("blocks", [None] * len(self.blocks))
        next_states = []
        for block, item in zip(self.blocks, states):
            values, item = block.forward_chunk(values, item)
            next_states.append(item)
        values, output = self.output.forward_chunk(F.silu(values), state.get("output"))
        return torch.tanh(values), {"blocks": next_states, "output": output}


class LargeStreamingCodec(torch.nn.Module):
    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        self.config = config
        self.keep_head = KeepHead(config)
        self.acoustic_encoder = AcousticEncoder(config)
        self.fusion = Fusion(config)
        self.decoder = Decoder(config)

    def forward(
        self, waveform: torch.Tensor, student_hidden: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("Expected waveform [batch, 1, samples]")
        if waveform.shape[-1] % self.config.hop_length:
            raise ValueError("Waveform length must be divisible by hop_length")
        keep = self.keep_head(student_hidden)
        z_edit = self.acoustic_encoder(waveform)
        reconstruction = self.decoder(self.fusion(keep["z_inv"], keep["z_dyn"], z_edit))
        if reconstruction.shape != waveform.shape:
            raise RuntimeError("Reconstruction shape differs from waveform")
        return {**keep, "z_edit": z_edit, "reconstruction": reconstruction}


def parameter_breakdown(model: LargeStreamingCodec) -> dict[str, int]:
    result = {
        "keep_head": sum(p.numel() for p in model.keep_head.parameters()),
        "acoustic_encoder": sum(p.numel() for p in model.acoustic_encoder.parameters()),
        "fusion": sum(p.numel() for p in model.fusion.parameters()),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
    }
    result["acoustic_total"] = result["acoustic_encoder"] + result["fusion"] + result["decoder"]
    result["total"] = sum(p.numel() for p in model.parameters())
    return result
