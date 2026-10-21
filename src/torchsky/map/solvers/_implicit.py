"""Policy placeholder for future implicit solver differentiation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._foundation import SolverResult, StateTree


class SolverDifferentiationMode(str, Enum):
    UNROLLED = "unrolled"
    IMPLICIT = "implicit"


@dataclass(frozen=True)
class ImplicitDifferentiationPolicy:
    mode: SolverDifferentiationMode = SolverDifferentiationMode.UNROLLED
    require_converged: bool = True

    def validate(self, result: SolverResult[StateTree]) -> None:
        if self.mode is SolverDifferentiationMode.IMPLICIT:
            raise NotImplementedError(
                "implicit solver differentiation is policy-only until a custom backward is implemented"
            )
        if (
            self.require_converged
            and result.diagnostics is not None
            and not result.diagnostics.converged
        ):
            raise RuntimeError("unrolled differentiation requires a converged solve")


__all__ = ["ImplicitDifferentiationPolicy", "SolverDifferentiationMode"]
