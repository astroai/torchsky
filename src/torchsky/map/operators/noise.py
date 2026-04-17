"""Noise-model helpers and internal interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .base import NoiseOp


@dataclass
class ExplicitNoiseOp(NoiseOp):
    def __init__(self, *, name: str = "explicit_noise") -> None:
        super().__init__(name=name, learned=False)


@dataclass
class LearnedNoiseOp(NoiseOp):
    def __init__(self, *, name: str = "learned_noise") -> None:
        super().__init__(name=name, learned=True)


def sanitize_inverse_variance(weight: Tensor) -> Tensor:
    return weight.where(weight > 0.0, weight.new_zeros(()).expand_as(weight))

