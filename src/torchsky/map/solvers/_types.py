"""Shared callable aliases for solver operators."""

from __future__ import annotations

from typing import Callable

from torch import Tensor

from ..operators.bundle import ObservationOperatorBundle
from ..scenes import SceneRepresentation
from ..target import TargetRenderSpec

ApplyOperatorFn = Callable[[SceneRepresentation, TargetRenderSpec], Tensor]
AdjointOperatorFn = Callable[[Tensor], Tensor]
ApplyNoiseFn = Callable[[Tensor], Tensor]
PreconditionerFn = Callable[[Tensor], Tensor]
NormalOperatorFn = Callable[[Tensor], Tensor]
BindTargetFn = Callable[
    [ObservationOperatorBundle, TargetRenderSpec],
    tuple[ApplyOperatorFn | None, AdjointOperatorFn | None, ApplyNoiseFn | None],
]
