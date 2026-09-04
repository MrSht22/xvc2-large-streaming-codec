from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class ChannelLayerNorm(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.left_context = dilation * (kernel_size - 1)
        self.stride = stride
        self.conv = torch.nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_context, 0)))

    def forward_chunk(
        self, values: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.shape[-1] % self.stride:
            raise ValueError("Chunk length must be divisible by convolution stride")
        expected = (values.shape[0], values.shape[1], self.left_context)
        if state is None:
            state = values.new_zeros(expected)
        if state.shape != expected:
            raise ValueError(f"Expected state {expected}, got {tuple(state.shape)}")
        combined = torch.cat((state, values), dim=-1)
        output = self.conv(combined)
        next_state = combined[..., -self.left_context :] if self.left_context else combined[..., :0]
        return output, next_state


class ResidualUnit(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.first = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.second = CausalConv1d(channels, channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.first(F.silu(values))
        return residual + self.second(F.silu(values))

    def forward_chunk(
        self,
        values: torch.Tensor,
        state: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        first_state, second_state = state or (None, None)
        residual = values
        values, first_state = self.first.forward_chunk(F.silu(values), first_state)
        values, second_state = self.second.forward_chunk(F.silu(values), second_state)
        return residual + values, (first_state, second_state)


class MultiKernelResidual(torch.nn.Module):
    def __init__(self, channels: int, kernels: tuple[int, ...], dilations: tuple[int, ...]) -> None:
        super().__init__()
        self.paths = torch.nn.ModuleList(
            ResidualUnit(channels, kernel, dilation) for kernel, dilation in zip(kernels, dilations)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack([path(values) for path in self.paths]).mean(0)

    def forward_chunk(
        self, values: torch.Tensor, state: list[Any] | None = None
    ) -> tuple[torch.Tensor, list[Any]]:
        states = state or [None] * len(self.paths)
        results = [path.forward_chunk(values, item) for path, item in zip(self.paths, states)]
        return torch.stack([item[0] for item in results]).mean(0), [item[1] for item in results]
