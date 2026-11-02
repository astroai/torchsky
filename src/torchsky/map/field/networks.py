"""Neural-network building blocks shared by field models."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.act(self.norm(residual + x))


class ResidualStack(nn.Module):
    def __init__(self, hidden_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim) for _ in range(max(int(num_blocks), 0))]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def silu_mlp_sequential(
    in_dim: int, hidden_dim: int, hidden_layers: int
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(max(int(hidden_layers), 1)):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.SiLU())
        current_dim = hidden_dim
    return nn.Sequential(*layers)


def linear_at(container: nn.Sequential | nn.ModuleList, index: int) -> nn.Linear:
    """Return ``container[index]`` typed as :class:`nn.Linear` (torch types items as Module)."""
    layer = container[index]
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"expected nn.Linear at index {index}, got {type(layer).__name__}")
    return layer


def linear_layers(container: nn.Sequential | nn.ModuleList) -> list[nn.Linear]:
    """Return every item of ``container`` typed as :class:`nn.Linear`."""
    return [linear_at(container, index) for index in range(len(container))]


def reset_silu_mlp_linears(sequence: nn.Sequential) -> None:
    for module in sequence:
        if type(module) is nn.Linear:
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)


def reset_logflux_readout_linear(linear: nn.Linear) -> None:
    nn.init.uniform_(linear.weight, -1e-2, 1e-2)
    nn.init.constant_(linear.bias, -3.0)


__all__ = [
    "ResidualMLPBlock",
    "ResidualStack",
    "linear_at",
    "linear_layers",

    "reset_logflux_readout_linear",
    "reset_silu_mlp_linears",
    "silu_mlp_sequential",
]
