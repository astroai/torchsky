"""Fixed Gaussian likelihood evidence and requested sampling uncertainty.

These objects reuse bound operators rather than rasterizing a new scene or
assuming that measurements are images. Response and precision callbacks must
be fixed, linear, and adjoint-consistent. Correlated observations or shared
eliminated nuisances belong in one joint factor. They are not made independent
by assigning different product names.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import Literal

import torch
from torch import Tensor

from .evidence import EvidenceSnapshot, _GuardedEvidence
from .solvers import (
    BoundObservationOperator,
    BoundObservationOperatorSet,
    LinearSolveResult,
)


NuisanceTreatment = Literal["conditional", "profiled", "calibrated"]


def _identifiers(values: tuple[str, ...], *, name: str, required: bool) -> None:
    if not isinstance(values, tuple) or (required and not values):
        raise ValueError(f"{name} must be a {'nonempty ' if required else ''}tuple")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain nonempty identifiers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate identifiers")


def _finite_tensor(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor) or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty tensor")
    if not (value.is_floating_point() or value.is_complex()):
        raise TypeError(f"{name} must have floating-point or complex dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite; zero unsupported samples explicitly")


def _compatible(value: Tensor, reference: Tensor, *, name: str) -> None:
    _finite_tensor(value, name=name)
    if (
        value.shape != reference.shape
        or value.dtype != reference.dtype
        or value.device != reference.device
    ):
        raise ValueError(f"{name} must preserve shape, dtype and device")


def _real_inner(left: Tensor, right: Tensor) -> Tensor:
    return torch.sum(left.conj() * right).real


@dataclass(frozen=True)
class GaussianLikelihoodFactor:
    """One fixed Gaussian quadratic, with centered native data and lineage.

    ``measurement_ids`` identify disjoint native sample groups, including the
    constituent groups of a derived product. A mosaic and its inputs therefore
    share identifiers. Callers own the truthful lineage and sampling partition.
    ``eliminated_nuisance_ids`` identify unknowns profiled or integrated in this
    factor; common unknowns must be eliminated jointly, not separately twice.

    Supply data centered by the declared calibrated additive mean. Precision
    callbacks own observed-support and real/complex noise conventions. This is
    a quadratic likelihood kernel. For a conditional or calibrated Gaussian
    density, supply its scalar ``log_normalization`` (e.g. half of
    ``n log(2 pi) + logdet(C)`` in real coordinates). Otherwise absolute density
    evaluation fails closed. Profiled nuisance objectives are not normalized
    predictive densities. Rebuild after changing response, calibration, noise,
    or data; tensors are not copied.

    Data and normalization versions are tracked automatically. Supply an
    ``EvidenceSnapshot`` in ``dependencies`` for callback matrices, module
    parameters, noise, calibration and support, plus a revision getter for
    mutable non-tensor state or replaced tensor bindings. Callback closures are
    not inspected. Inference tensors require an explicit revision; callers own
    revision changes for writes that bypass Torch mutation versions.
    """

    operator: BoundObservationOperator
    data: Tensor
    measurement_ids: tuple[str, ...]
    calibration_id: str
    provenance_id: str
    nuisance_treatment: NuisanceTreatment = "conditional"
    eliminated_nuisance_ids: tuple[str, ...] = ()
    log_normalization: Tensor | None = None
    dependencies: EvidenceSnapshot | None = None
    _snapshot: EvidenceSnapshot = field(init=False, repr=False, compare=False)
    _operator_binding: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.operator, BoundObservationOperator):
            raise TypeError("operator must be a BoundObservationOperator")
        _finite_tensor(self.data, name="data")
        _identifiers(self.measurement_ids, name="measurement_ids", required=True)
        _identifiers(
            self.eliminated_nuisance_ids, name="eliminated_nuisance_ids", required=False
        )
        for name in ("calibration_id", "provenance_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty identifier")
        if self.nuisance_treatment not in {"conditional", "profiled", "calibrated"}:
            raise ValueError("unknown nuisance_treatment")
        if self.nuisance_treatment == "conditional" and self.eliminated_nuisance_ids:
            raise ValueError("conditional factors do not eliminate nuisance variables")
        if (
            self.nuisance_treatment != "conditional"
            and not self.eliminated_nuisance_ids
        ):
            raise ValueError(
                "eliminated nuisance variables require explicit identifiers"
            )
        if self.log_normalization is not None:
            normalization = self.log_normalization
            _finite_tensor(normalization, name="log_normalization")
            if normalization.ndim != 0 or normalization.is_complex():
                raise ValueError("log_normalization must be a real scalar tensor")
            if (
                normalization.dtype != self.data.real.dtype
                or normalization.device != self.data.device
            ):
                raise ValueError(
                    "log_normalization must match data real dtype and device"
                )
        if self.dependencies is not None and not isinstance(
            self.dependencies, EvidenceSnapshot
        ):
            raise TypeError("dependencies must be an EvidenceSnapshot")
        tensors = {"data": self.data}
        if self.log_normalization is not None:
            tensors["log_normalization"] = self.log_normalization
        snapshot = (
            EvidenceSnapshot.capture(tensors)
            if self.dependencies is None
            else self.dependencies.with_tensors(tensors)
        )
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_operator_binding", self._binding())

    def _binding(self) -> tuple[int, ...]:
        return tuple(
            id(getattr(self.operator, name))
            for name in (
                "operators",
                "target",
                "apply_operator",
                "apply_adjoint",
                "apply_noise",
            )
        )

    def validate_evidence(self) -> None:
        """Reject changed data or declared physics before reusing this factor.

        Mutable callback state needs ``dependencies``; direct callback/target
        replacement is also detected. Calibration/provenance IDs are lineage,
        not mutation guards.
        """
        self._snapshot.validate()
        if self._binding() != self._operator_binding:
            raise RuntimeError(
                "stale evidence operator binding; rebuild the factor/system/query"
            )

    def precision(self, value: Tensor) -> Tensor:
        self.validate_evidence()
        _compatible(value, self.data, name="measurement input")
        weighted = self.operator.apply_noise_model(value)
        _compatible(weighted, value, name="precision output")
        self.validate_evidence()
        return weighted

    def residual(self, state: Tensor) -> Tensor:
        self.validate_evidence()
        prediction = self.operator.apply(state)
        _compatible(prediction, self.data, name="response output")
        self.validate_evidence()
        return self.data - prediction


@dataclass(frozen=True)
class GaussianNormalEquations(_GuardedEvidence):
    """Full quadratic evidence ``x^T H x - 2 x^T b + c`` for fixed callbacks.

    The full information action is retained. No diagonal approximation or
    reconstructed image is presented as sufficient evidence. ``c`` excludes
    determinant terms; optional ``log_normalization`` must be supplied by the
    caller from the declared covariance and observed support.

    Statistics returned by a likelihood system validate its declared sources
    on public reads and information calls. A previously retrieved tensor is
    an ordinary tensor; its later use cannot be intercepted.
    """

    rhs: Tensor
    data_quadratic: Tensor
    information: Callable[[Tensor], Tensor]
    measurement_ids: tuple[str, ...]
    log_normalization: Tensor | None = None
    _evidence_guard: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )
    _guarded_fields = frozenset(
        {"rhs", "data_quadratic", "information", "log_normalization"}
    )

    def __post_init__(self) -> None:
        if self._evidence_guard is not None:
            information = self.information
            guard = self._evidence_guard

            def checked_information(state: Tensor) -> Tensor:
                guard()
                result = information(state)
                guard()
                return result

            object.__setattr__(self, "information", checked_information)

    def quadratic(self, state: Tensor) -> Tensor:
        _compatible(state, self.rhs, name="scene state")
        return (
            _real_inner(state, self.information(state))
            - 2 * _real_inner(state, self.rhs)
            + self.data_quadratic
        )

    def negative_log_likelihood(self, state: Tensor) -> Tensor:
        if self.log_normalization is None:
            raise ValueError(
                "normalized conditional/calibrated density constants are unavailable"
            )
        return self.quadratic(state) * 0.5 + self.log_normalization


@dataclass(frozen=True)
class FunctionalSamplingUncertainty(_GuardedEvidence):
    """Fixed-model sampling variance, mean transfer and solve diagnostics.

    For ridge strength d, ``mean_response`` is ``H (H+d Id)^-1 g``. It equals
    g for an accurately solved unregularized estimable query. Otherwise its
    difference from g describes possible bias and is not a confidence bound.
    This result supplies no conformal or misspecification coverage guarantee.
    Public numeric reads validate the factor evidence and query direction;
    rebuild after changing either source.
    """

    sampling_variance: Tensor
    mean_response: Tensor
    solve_residual_norm: Tensor
    damping: float
    nuisance_treatments: tuple[NuisanceTreatment, ...]
    _evidence_guard: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )
    _guarded_fields = frozenset(
        {"sampling_variance", "mean_response", "solve_residual_norm", "damping"}
    )


@dataclass(frozen=True)
class GaussianLikelihoodSystem:
    """Independent fixed factors over a common scene, without a field solve.

    Operators may emit native pixels, vectors, timelines or complex arrays.
    Nonlinear/Poisson likelihoods cannot use this quadratic compression. The
    system requires independent factors under their declared nuisance treatment;
    unrecorded correlation cannot be detected from metadata alone.
    """

    factors: tuple[GaussianLikelihoodFactor, ...]
    exposure_chunk_size: int = 1
    _operators: BoundObservationOperatorSet = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.factors, tuple) or not self.factors:
            raise ValueError("factors must be a nonempty tuple")
        measurements: set[str] = set()
        nuisances: set[str] = set()
        for factor in self.factors:
            if not isinstance(factor, GaussianLikelihoodFactor):
                raise TypeError("factors must contain GaussianLikelihoodFactor objects")
            factor.validate_evidence()
            repeated = measurements.intersection(factor.measurement_ids)
            if repeated:
                raise ValueError(
                    f"duplicate native measurement evidence: {sorted(repeated)}"
                )
            repeated_nuisance = nuisances.intersection(factor.eliminated_nuisance_ids)
            if repeated_nuisance:
                raise ValueError(
                    f"shared eliminated nuisances require a joint factor: {sorted(repeated_nuisance)}"
                )
            measurements.update(factor.measurement_ids)
            nuisances.update(factor.eliminated_nuisance_ids)
        object.__setattr__(
            self,
            "_operators",
            BoundObservationOperatorSet(
                tuple(factor.operator for factor in self.factors),
                exposure_chunk_size=self.exposure_chunk_size,
            ),
        )

    @property
    def measurement_ids(self) -> tuple[str, ...]:
        return tuple(
            identifier
            for factor in self.factors
            for identifier in factor.measurement_ids
        )

    def validate_evidence(self) -> None:
        """Validate all fixed factor dependencies without reading tensor contents."""
        for factor in self.factors:
            factor.validate_evidence()

    def rhs(self) -> Tensor:
        self.validate_evidence()
        total: Tensor | None = None
        for factor in self.factors:
            value = factor.operator.adjoint(factor.precision(factor.data))
            _finite_tensor(value, name="adjoint output")
            if total is not None:
                _compatible(value, total, name="shared scene adjoint")
            total = value if total is None else total + value
        assert total is not None
        self.validate_evidence()
        return total

    def information(self, state: Tensor) -> Tensor:
        self.validate_evidence()
        _finite_tensor(state, name="scene state")
        total: Tensor | None = None
        for factor in self.factors:
            prediction = factor.operator.apply(state)
            value = factor.operator.adjoint(factor.precision(prediction))
            _compatible(value, state, name="information output")
            total = value if total is None else total + value
        assert total is not None
        self.validate_evidence()
        return total

    def quadratic(self, state: Tensor) -> Tensor:
        total: Tensor | None = None
        for factor in self.factors:
            residual = factor.residual(state)
            value = _real_inner(residual, factor.precision(residual))
            total = value if total is None else total + value
        assert total is not None
        return total

    def normal_equations(self) -> GaussianNormalEquations:
        self.validate_evidence()
        constant = sum(
            (
                _real_inner(factor.data, factor.precision(factor.data))
                for factor in self.factors
            ),
            start=self.factors[0].data.real.new_zeros(()),
        )
        normalization = None
        if all(
            factor.nuisance_treatment != "profiled"
            and factor.log_normalization is not None
            for factor in self.factors
        ):
            normalization = sum(
                (
                    factor.log_normalization
                    for factor in self.factors
                    if factor.log_normalization is not None
                ),
                start=self.factors[0].data.real.new_zeros(()),
            )
        return GaussianNormalEquations(
            self.rhs(),
            constant,
            self.information,
            self.measurement_ids,
            normalization,
            _evidence_guard=self.validate_evidence,
        )

    def negative_log_likelihood(self, state: Tensor) -> Tensor:
        """Evaluate a normalized density only with declared Gaussian constants."""
        self.validate_evidence()
        normalization = self.factors[0].data.real.new_zeros(())
        for factor in self.factors:
            if (
                factor.nuisance_treatment == "profiled"
                or factor.log_normalization is None
            ):
                raise ValueError(
                    "normalized conditional/calibrated density constants are unavailable"
                )
            normalization = normalization + factor.log_normalization
        return self.quadratic(state) * 0.5 + normalization

    def solve(
        self,
        *,
        iterations: int = 64,
        damping: float = 0.0,
        preconditioner: Callable[[Tensor], Tensor] | Tensor | None = None,
    ) -> LinearSolveResult:
        """Use the shared matrix-free solver; this does not certify convergence."""
        result = self._operators.solve_normal_system_cg(
            self.rhs(),
            iterations=iterations,
            damping=damping,
            preconditioner=preconditioner,
        )
        self.validate_evidence()
        return result

    def functional_uncertainty(
        self,
        functional: Tensor,
        *,
        iterations: int = 64,
        damping: float = 0.0,
        rtol: float = 1e-5,
        atol: float = 0.0,
        preconditioner: Callable[[Tensor], Tensor] | Tensor | None = None,
        dependencies: EvidenceSnapshot | None = None,
    ) -> FunctionalSamplingUncertainty:
        """Compute g^T K^-1 H K^-1 g, with K=H+d Id, using one solve.

        A failed/inconsistent solve raises instead of reporting finite certainty
        for an unmeasured query. Passing residual checks does not prove that all
        sky modes are identifiable, or bound ill-conditioned forward error.
        Direction versions are tracked automatically; an inference direction
        requires a caller revision in ``dependencies``. This optional snapshot
        also covers other mutable query state not represented by the factors.
        """
        self.validate_evidence()
        _finite_tensor(functional, name="functional")
        if dependencies is not None and not isinstance(dependencies, EvidenceSnapshot):
            raise TypeError("dependencies must be an EvidenceSnapshot")
        snapshot = (
            EvidenceSnapshot.capture({"functional": functional})
            if dependencies is None
            else dependencies.with_tensors({"functional": functional})
        )

        def validate_uncertainty() -> None:
            self.validate_evidence()
            snapshot.validate()

        if (
            not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or iterations < 1
        ):
            raise ValueError("iterations must be a positive integer")
        for value, name in ((damping, "damping"), (rtol, "rtol"), (atol, "atol")):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if rtol == 0 and atol == 0:
            raise ValueError("at least one solve tolerance must be positive")
        solved = self._operators.solve_normal_system_cg(
            functional,
            iterations=iterations,
            damping=damping,
            preconditioner=preconditioner,
        ).solution
        response = self.information(solved)
        residual_norm = torch.linalg.vector_norm(
            response + damping * solved - functional
        )
        threshold = atol + rtol * torch.linalg.vector_norm(functional)
        if not bool(torch.isfinite(residual_norm)) or bool(residual_norm > threshold):
            raise RuntimeError(
                "functional solve did not converge; query may be unidentifiable or require a better solver"
            )
        variance = _real_inner(solved, response)
        if not bool(torch.isfinite(variance)) or bool(variance < 0):
            raise RuntimeError(
                "sampling variance is invalid; precision must be positive semidefinite"
            )
        validate_uncertainty()
        return FunctionalSamplingUncertainty(
            variance,
            response,
            residual_norm,
            float(damping),
            tuple(factor.nuisance_treatment for factor in self.factors),
            _evidence_guard=validate_uncertainty,
        )


__all__ = [
    "FunctionalSamplingUncertainty",
    "GaussianLikelihoodFactor",
    "GaussianLikelihoodSystem",
    "GaussianNormalEquations",
]
