"""Internal solver hooks for field fitting and discrete inverse mapmaking."""

from __future__ import annotations

from ._types import (
    AdjointOperatorFn,
    ApplyNoiseFn,
    ApplyOperatorFn,
    NormalOperatorFn,
    PreconditionerFn,
)
from ._foundation import (
    DiagonalPreconditioner,
    GaugeProjection,
    LinearSolveResult,
    Preconditioner,
    conjugate_gradient,
    lsqr,
)

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Iterator, Sequence

import torch
from torch import Tensor, nn

from ..operators.bundle import ObservationOperatorBundle
from ..scenes import (
    SceneRepresentation,
    as_scene_representation,
)
from ..target import TargetRenderSpec


def _relative_norm(error: Tensor, reference: Tensor) -> float:
    """Relative L2 norm of ``error`` with respect to ``reference``.

    A zero reference falls back to the absolute error norm, so diagnostics never
    divide by zero.
    """
    error_norm = torch.linalg.vector_norm(torch.as_tensor(error).reshape(-1))
    ref_norm = torch.linalg.vector_norm(torch.as_tensor(reference).reshape(-1))
    if float(ref_norm.item()) == 0.0:
        return 0.0 if float(error_norm.item()) == 0.0 else float(error_norm.item())
    return float((error_norm / ref_norm).item())


def _subspace_leakage(error: Tensor, basis: Tensor, reference: Tensor) -> float:
    """Fraction of ``error`` that lies in the span of ``basis``.

    Used by component-decomposition diagnostics to report how much of a private
    residual leaks into the shared-state subspace. Empty or all-zero bases have
    no leakage by definition.
    """
    basis_t = torch.as_tensor(basis)
    if int(basis_t.numel()) == 0:
        return 0.0
    basis_rows = basis_t.reshape(int(basis_t.shape[0]), -1)
    nonzero = torch.any(basis_rows != 0, dim=1)
    basis_rows = basis_rows[nonzero]
    if int(basis_rows.shape[0]) == 0:
        return 0.0
    vector = torch.as_tensor(error).reshape(-1)
    dtype = torch.promote_types(vector.dtype, basis_rows.dtype)
    matrix = basis_rows.to(dtype=dtype, device=vector.device).transpose(0, 1)
    rhs = vector.to(dtype=dtype, device=vector.device)
    coeffs = torch.linalg.lstsq(matrix, rhs).solution
    projected = matrix @ coeffs
    proj_norm = torch.linalg.vector_norm(projected)
    ref_norm = torch.linalg.vector_norm(torch.as_tensor(reference).reshape(-1))
    if float(ref_norm.item()) == 0.0:
        return 0.0 if float(proj_norm.item()) == 0.0 else float(proj_norm.item())
    return float((proj_norm / ref_norm).item())


def default_apply_operator(
    scene: SceneRepresentation,
    target: TargetRenderSpec,
    *,
    operators: ObservationOperatorBundle | None = None,
) -> Tensor:
    return scene.render_to_target(target, operators=operators)


def identity_noise(tensor: Tensor) -> Tensor:
    return tensor


@dataclass(frozen=True)
class NormalEquationProbe:
    """Native pieces needed by future discrete GLS/MAP assembly."""

    prediction: Tensor
    weighted_prediction: Tensor
    weighted_data: Tensor
    lhs_probe: Tensor
    rhs: Tensor


@dataclass(frozen=True)
class MultiObservationNormalEquationProbe:
    """Normal-equation probes for a shared scene seen by multiple observations."""

    predictions: tuple[Tensor, ...]
    weighted_predictions: tuple[Tensor, ...]
    weighted_data: tuple[Tensor, ...]
    lhs_probe: Tensor
    rhs: Tensor


def _normalized_probe(probe: Tensor) -> Tensor:
    probe_t = torch.as_tensor(probe)
    norm = torch.linalg.vector_norm(probe_t.reshape(-1))
    if not torch.isfinite(norm) or float(norm.item()) == 0.0:
        return torch.ones_like(probe_t)
    return probe_t / norm


def _solver_complex_dtype(*tensors: Tensor) -> torch.dtype:
    for tensor in tensors:
        if tensor.device.type == "mps":
            return torch.complex64
    return torch.complex128


def _rayleigh_quotient(state: Tensor, image: Tensor) -> float:
    dtype = _solver_complex_dtype(state, image)
    state_flat = state.reshape(-1).to(dtype)
    image_flat = image.reshape(-1).to(dtype)
    denom = torch.sum(torch.conj(state_flat) * state_flat).real
    if float(denom.item()) <= 0.0:
        return 0.0
    numer = torch.sum(torch.conj(state_flat) * image_flat).real
    value = float((numer / denom).item())
    return max(value, 0.0)


def _inner_product(lhs: Tensor, rhs: Tensor) -> Tensor:
    dtype = _solver_complex_dtype(lhs, rhs)
    lhs_flat = torch.as_tensor(lhs).reshape(-1).to(dtype)
    rhs_flat = torch.as_tensor(rhs).reshape(-1).to(dtype)
    return torch.sum(torch.conj(lhs_flat) * rhs_flat)


def make_diagonal_preconditioner(diagonal: Tensor) -> DiagonalPreconditioner:
    """Wrap a diagonal as a Jacobi-style :class:`DiagonalPreconditioner`."""
    return DiagonalPreconditioner(diagonal)


def _apply_preconditioner(
    preconditioner: Preconditioner[Tensor] | PreconditionerFn | Tensor | None,
    vector: Tensor,
) -> Tensor:
    if preconditioner is None:
        return vector
    if isinstance(preconditioner, Preconditioner):
        return torch.as_tensor(preconditioner.apply(vector))
    if callable(preconditioner):
        return torch.as_tensor(preconditioner(vector))
    return (
        torch.as_tensor(preconditioner).to(dtype=vector.dtype, device=vector.device)
        * vector
    )


def _solve_normal_system_cg(
    normal_action: NormalOperatorFn,
    rhs: Tensor,
    *,
    initial_solution: Tensor | None = None,
    iterations: int = 16,
    damping: float = 0.0,
    preconditioner: Preconditioner[Tensor] | PreconditionerFn | Tensor | None = None,
    rtol: float = 1.0e-11,
    atol: float = 0.0,
    projection: GaugeProjection[Tensor] | Callable[[Tensor], Tensor] | None = None,
    operator_is_hpd: bool = True,
    record_residual_history: bool = True,
) -> LinearSolveResult:
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise TypeError("iterations must be an integer")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not math.isfinite(float(damping)) or damping < 0.0:
        raise ValueError("damping must be finite and non-negative")
    rhs_t = torch.as_tensor(rhs)
    if not (torch.is_floating_point(rhs_t) or torch.is_complex(rhs_t)):
        raise TypeError("rhs must have a floating-point or complex dtype")
    def _action(state: Tensor) -> Tensor:
        image = torch.as_tensor(normal_action(state)).to(
            dtype=state.dtype, device=state.device
        )
        if tuple(image.shape) != tuple(rhs_t.shape):
            raise ValueError("normal operator output shape must match rhs")
        if damping > 0.0:
            image = image + float(damping) * state
        return image

    result = conjugate_gradient(
        _action,
        rhs_t,
        operator_is_hpd=operator_is_hpd,
        initial_solution=(
            None
            if initial_solution is None
            else torch.as_tensor(initial_solution).to(
                dtype=rhs_t.dtype, device=rhs_t.device
            )
        ),
        max_iterations=iterations,
        rtol=rtol,
        atol=atol,
        preconditioner=preconditioner,
        projection=projection,
        record_residual_history=record_residual_history,
    )
    return result


def _estimate_inverse_diagonal_hutchinson(
    solve: Callable[[Tensor], Tensor],
    reference_state: Tensor,
    *,
    num_probes: int,
    seed: int,
    normal_action: NormalOperatorFn,
    damping: float = 0.0,
    rtol: float = 1.0e-5,
    atol: float = 0.0,
) -> Tensor:
    """Return a signed inverse-curvature diagonal estimate after checked solves."""
    if not isinstance(num_probes, int) or isinstance(num_probes, bool):
        raise TypeError("num_probes must be an integer")
    if num_probes < 1:
        raise ValueError("num_probes must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    for name, value in (("damping", damping), ("rtol", rtol), ("atol", atol)):
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    state = torch.as_tensor(reference_state)
    if not (torch.is_floating_point(state) or torch.is_complex(state)):
        raise TypeError("reference_state must have a floating-point or complex dtype")
    if state.numel() == 0:
        raise ValueError("reference_state must not be empty")
    if not torch.all(torch.isfinite(state)):
        raise ValueError("reference_state must be finite")
    real_dtype = state.real.dtype if torch.is_complex(state) else state.dtype
    generator_device = (
        state.device if state.device.type in {"cpu", "cuda"} else torch.device("cpu")
    )
    # ponytail: MPS has no local Generator; remove the CPU transfer when torch adds it.
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    estimate = torch.zeros(state.shape, dtype=real_dtype, device=state.device)

    def _checked_output(output: Tensor, name: str) -> Tensor:
        if not isinstance(output, Tensor):
            raise TypeError(f"{name} must return a tensor")
        if output.shape != state.shape:
            raise ValueError(f"{name} output shape must match reference_state")
        if output.dtype != state.dtype or output.device != state.device:
            raise ValueError(
                f"{name} output dtype and device must match reference_state"
            )
        if not torch.all(torch.isfinite(output)):
            raise RuntimeError(f"{name} produced a non-finite inverse probe")
        return output

    for probe_index in range(num_probes):
        probe = torch.randint(
            0,
            2,
            state.shape,
            dtype=torch.int8,
            device=generator_device,
            generator=generator,
        ).to(dtype=real_dtype, device=state.device)
        probe = (probe * 2.0 - 1.0).to(dtype=state.dtype)
        solution = _checked_output(solve(probe), "normal-system solve")
        image = _checked_output(normal_action(solution), "normal action")
        residual = image + damping * solution - probe
        residual_norm = torch.linalg.vector_norm(residual)
        allowed_residual = torch.maximum(
            residual_norm.new_tensor(atol), rtol * torch.linalg.vector_norm(probe)
        )
        if not torch.isfinite(residual_norm) or bool(residual_norm > allowed_residual):
            raise RuntimeError(
                f"inverse probe {probe_index} failed the actual residual check: "
                f"norm={float(residual_norm.item()):.6g}, "
                f"allowed={float(allowed_residual.item()):.6g}; "
                "the system may be singular or the solve incomplete"
            )
        contribution = (torch.conj(probe) * solution).real.to(dtype=real_dtype)
        if not torch.all(torch.isfinite(contribution)):
            raise RuntimeError(
                "normal-system solve produced a non-finite inverse probe"
            )
        estimate.add_(contribution)
    estimate.div_(float(num_probes))
    if not torch.all(torch.isfinite(estimate)):
        raise RuntimeError("inverse probe accumulation produced non-finite values")
    return estimate


@dataclass
class BoundObservationOperator:
    """Target-bound observation operator shared by discrete and INR scene types."""

    operators: ObservationOperatorBundle
    target: TargetRenderSpec
    apply_operator: ApplyOperatorFn | None = None
    apply_adjoint: AdjointOperatorFn | None = None
    apply_noise: ApplyNoiseFn | None = None

    def apply(self, scene: SceneRepresentation | nn.Module | Tensor) -> Tensor:
        scene_representation = as_scene_representation(scene)
        if self.apply_operator is not None:
            return self.apply_operator(scene_representation, self.target)
        return default_apply_operator(
            scene_representation,
            self.target,
            operators=self.operators,
        )

    def apply_noise_model(self, tensor: Tensor) -> Tensor:
        fn = self.apply_noise or identity_noise
        return fn(tensor)

    def adjoint(self, tensor: Tensor) -> Tensor:
        if self.apply_adjoint is None:
            raise NotImplementedError("adjoint operator hook is not configured")
        return self.apply_adjoint(tensor)

    def normal_equation_probe(
        self,
        scene: SceneRepresentation | nn.Module | Tensor,
        data: Tensor,
    ) -> NormalEquationProbe:
        prediction = self.apply(scene)
        weighted_prediction = self.apply_noise_model(prediction)
        weighted_data = self.apply_noise_model(data)
        return NormalEquationProbe(
            prediction=prediction,
            weighted_prediction=weighted_prediction,
            weighted_data=weighted_data,
            lhs_probe=self.adjoint(weighted_prediction),
            rhs=self.adjoint(weighted_data),
        )

    def normal_operator_action(
        self,
        scene: SceneRepresentation | nn.Module | Tensor,
    ) -> Tensor:
        """Apply ``A^T N^-1 A`` without materializing the normal matrix."""
        return self.adjoint(self.apply_noise_model(self.apply(scene)))

    def solve_normal_system_cg(
        self,
        rhs: Tensor,
        *,
        initial_solution: Tensor | None = None,
        iterations: int = 16,
        damping: float = 0.0,
        preconditioner: Preconditioner[Tensor] | PreconditionerFn | Tensor | None = None,
        rtol: float = 1.0e-11,
        atol: float = 0.0,
        projection: GaugeProjection[Tensor] | Callable[[Tensor], Tensor] | None = None,
        operator_is_hpd: bool = True,
    ) -> LinearSolveResult:
        """Solve ``(A^T N^-1 A + damping I) x = rhs`` matrix-free."""
        return _solve_normal_system_cg(
            self.normal_operator_action,
            rhs,
            initial_solution=initial_solution,
            iterations=iterations,
            damping=damping,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            projection=projection,
            operator_is_hpd=operator_is_hpd,
        )

    def solve_least_squares_lsqr(
        self,
        data: Tensor,
        *,
        initial_solution: Tensor | None = None,
        iterations: int = 100,
        rtol: float = 1.0e-5,
        atol: float = 0.0,
        damping: float = 0.0,
        projection: GaugeProjection[Tensor] | Callable[[Tensor], Tensor] | None = None,
    ) -> LinearSolveResult:
        return lsqr(
            self.apply,
            self.adjoint,
            torch.as_tensor(data),
            initial_solution=initial_solution,
            max_iterations=iterations,
            rtol=rtol,
            atol=atol,
            damping=damping,
            projection=projection,
        )

    def estimate_posterior_variance_diagonal(
        self,
        reference_state: Tensor,
        *,
        num_probes: int = 32,
        seed: int = 0,
        iterations: int = 32,
        damping: float = 0.0,
        preconditioner: PreconditionerFn | Tensor | None = None,
        rtol: float = 1.0e-5,
        atol: float = 0.0,
    ) -> Tensor:
        """Estimate the inverse-curvature diagonal with signed Hutchinson probes.

        The historical name is retained. The target is
        ``diag((A^T N^-1 A + damping I)^-1)``; damping changes curvature and
        does not give the sampling covariance of a regularized estimator.
        Negative finite-probe estimates remain signed, without a positive floor.
        Every solve must satisfy the actual residual criterion
        ``|| (H + damping I) x - z || <= max(atol, rtol ||z||)`` or raise.
        The declared normal system must be fixed, Hermitian positive definite
        on the chosen state space, with any unmeasured gauges removed explicitly.
        A finite set of checked probes does not certify global identifiability
        or establish a bound on inverse error for an ill-conditioned system.
        A posterior interpretation additionally requires a declared Gaussian
        likelihood and prior; inverse curvature alone supplies neither.
        """
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be positive")

        def _solve(rhs: Tensor) -> Tensor:
            return self.solve_normal_system_cg(
                rhs,
                iterations=iterations,
                damping=damping,
                preconditioner=preconditioner,
            ).solution

        return _estimate_inverse_diagonal_hutchinson(
            _solve,
            reference_state,
            num_probes=num_probes,
            seed=seed,
            normal_action=self.normal_operator_action,
            damping=damping,
            rtol=rtol,
            atol=atol,
        )

    def estimate_normal_operator_scale(
        self,
        data: Tensor,
        *,
        initial_probe: Tensor | None = None,
        iterations: int = 8,
    ) -> float:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        data_t = torch.as_tensor(data)
        probe = initial_probe
        if probe is None:
            probe = self.adjoint(self.apply_noise_model(data_t))
        state = _normalized_probe(torch.as_tensor(probe))
        estimate = 0.0
        for _ in range(int(iterations)):
            image = self.adjoint(self.apply_noise_model(self.apply(state)))
            estimate = _rayleigh_quotient(state, image)
            state = _normalized_probe(image)
        return estimate

    def recommended_step_size(
        self,
        data: Tensor,
        *,
        initial_probe: Tensor | None = None,
        iterations: int = 8,
        safety: float = 0.9,
    ) -> float:
        if safety <= 0.0 or safety > 1.0:
            raise ValueError("safety must satisfy 0 < safety <= 1")
        scale = self.estimate_normal_operator_scale(
            data,
            initial_probe=initial_probe,
            iterations=iterations,
        )
        return float(safety / max(scale, 1.0e-12))

    def estimate_normal_operator_diagonal(
        self,
        data: Tensor,
        *,
        initial_probe: Tensor | None = None,
        damping: float = 0.0,
        num_probes: int = 16,
    ) -> Tensor:
        prototype = torch.as_tensor(
            initial_probe
            if initial_probe is not None
            else self.adjoint(self.apply_noise_model(torch.as_tensor(data)))
        )
        diagonal = torch.zeros_like(prototype)
        for _ in range(int(num_probes)):
            if torch.is_complex(prototype):
                re = (
                    torch.rand(
                        prototype.shape,
                        dtype=prototype.real.dtype,
                        device=prototype.device,
                    )
                    > 0.5
                ).to(prototype.real.dtype) * 2 - 1
                z = re.to(prototype.dtype)
            else:
                z = (torch.rand_like(prototype) > 0.5).to(prototype.dtype) * 2 - 1
            Az = self.apply(z)
            NAz = self.apply_noise_model(Az)
            ANAz = self.adjoint(NAz)
            if damping > 0.0:
                ANAz = ANAz + float(damping) * z
            diagonal = diagonal + (z.conj() * ANAz).real.to(diagonal.dtype)
        diagonal = diagonal / float(num_probes)
        if torch.is_complex(diagonal):
            diagonal = diagonal.real.clamp(min=0.0)
        else:
            diagonal = diagonal.clamp(min=0.0)
        return diagonal

    def solve_weighted_least_squares(
        self,
        data: Tensor,
        *,
        initial_solution: Tensor | None = None,
        iterations: int = 16,
        step_size: float | None = None,
        scale_iterations: int = 8,
        step_safety: float = 0.9,
    ) -> LinearSolveResult:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        history: list[float] = []
        data_t = torch.as_tensor(data)
        if initial_solution is None:
            prototype = self.adjoint(self.apply_noise_model(data_t))
            solution = torch.zeros_like(prototype)
        else:
            solution = torch.as_tensor(initial_solution).clone()
        step = (
            float(step_size)
            if step_size is not None
            else self.recommended_step_size(
                data_t,
                initial_probe=solution,
                iterations=scale_iterations,
                safety=step_safety,
            )
        )
        if step <= 0.0:
            raise ValueError("step_size must be positive")
        for _ in range(int(iterations)):
            prediction = self.apply(solution)
            residual = data_t.to(dtype=prediction.dtype) - prediction
            history.append(float(torch.linalg.vector_norm(residual).item()))
            weighted_residual = self.apply_noise_model(residual)
            update = self.adjoint(weighted_residual).to(dtype=solution.dtype)
            solution = solution + step * update
        return LinearSolveResult(
            solution=solution,
            residual_norm_history=tuple(history),
        )

    def solve_normal_equation_cg(
        self,
        data: Tensor,
        *,
        initial_solution: Tensor | None = None,
        iterations: int = 16,
        damping: float = 0.0,
        preconditioner: Preconditioner[Tensor] | PreconditionerFn | Tensor | None = None,
        rtol: float = 1.0e-11,
        atol: float = 0.0,
        projection: GaugeProjection[Tensor] | Callable[[Tensor], Tensor] | None = None,
        operator_is_hpd: bool = True,
    ) -> LinearSolveResult:
        data_t = torch.as_tensor(data)
        rhs = self.adjoint(self.apply_noise_model(data_t))
        return self.solve_normal_system_cg(
            rhs,
            initial_solution=initial_solution,
            iterations=iterations,
            damping=damping,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            projection=projection,
            operator_is_hpd=operator_is_hpd,
        )


@dataclass
class BoundObservationOperatorSet:
    """Shared-scene operator assembled from multiple target-bound observations."""

    operators: tuple[BoundObservationOperator, ...]
    exposure_chunk_size: int = 1

    def __post_init__(self) -> None:
        if len(self.operators) == 0:
            raise ValueError(
                "BoundObservationOperatorSet requires at least one operator"
            )
        if not isinstance(self.exposure_chunk_size, int) or isinstance(
            self.exposure_chunk_size, bool
        ):
            raise TypeError("exposure_chunk_size must be an integer")
        if self.exposure_chunk_size < 1:
            raise ValueError("exposure_chunk_size must be positive")

    def _iter_operator_chunks(self) -> Iterator[tuple[BoundObservationOperator, ...]]:
        # ponytail: schedule chunks serially; add cross-exposure fusion only after profiling.
        for start in range(0, len(self.operators), self.exposure_chunk_size):
            yield self.operators[start : start + self.exposure_chunk_size]

    def iter_apply_chunks(
        self,
        scene: SceneRepresentation | nn.Module | Tensor,
    ) -> Iterator[tuple[Tensor, ...]]:
        """Yield predictions in bounded, deterministic exposure-order chunks."""
        for operators in self._iter_operator_chunks():
            yield tuple(operator.apply(scene) for operator in operators)

    def apply(
        self,
        scene: SceneRepresentation | nn.Module | Tensor,
    ) -> tuple[Tensor, ...]:
        return tuple(
            prediction
            for chunk in self.iter_apply_chunks(scene)
            for prediction in chunk
        )

    def apply_noise_model(self, tensors: Sequence[Tensor]) -> tuple[Tensor, ...]:
        if len(tensors) != len(self.operators):
            raise ValueError("tensor count must match operator count")
        return tuple(
            operator.apply_noise_model(tensor)
            for operator, tensor in zip(self.operators, tensors, strict=True)
        )

    def adjoint(self, tensors: Sequence[Tensor]) -> Tensor:
        if len(tensors) != len(self.operators):
            raise ValueError("tensor count must match operator count")
        total: Tensor | None = None
        for operator, tensor in zip(self.operators, tensors, strict=True):
            contribution = operator.adjoint(tensor)
            total = contribution if total is None else total + contribution
        assert total is not None
        return total

    def weighted_adjoint(self, tensors: Sequence[Tensor]) -> Tensor:
        """Apply ``A^T N^-1`` without materializing weighted exposures."""
        if len(tensors) != len(self.operators):
            raise ValueError("tensor count must match operator count")
        total: Tensor | None = None
        for operator, tensor in zip(self.operators, tensors, strict=True):
            contribution = operator.adjoint(operator.apply_noise_model(tensor))
            total = contribution if total is None else total + contribution
        assert total is not None
        return total

    def normal_operator_action(
        self,
        scene: SceneRepresentation | nn.Module | Tensor,
    ) -> Tensor:
        """Apply ``A^T N^-1 A`` with chunk-bounded prediction intermediates."""
        total: Tensor | None = None
        for operators in self._iter_operator_chunks():
            predictions = tuple(operator.apply(scene) for operator in operators)
            for operator, prediction in zip(operators, predictions, strict=True):
                weighted_prediction = operator.apply_noise_model(prediction)
                contribution = operator.adjoint(weighted_prediction)
                total = contribution if total is None else total + contribution
        assert total is not None
        return total

    def solve_normal_system_cg(
        self,
        rhs: Tensor,
        *,
        initial_solution: Tensor | None = None,
        iterations: int = 16,
        damping: float = 0.0,
        preconditioner: Preconditioner[Tensor] | PreconditionerFn | Tensor | None = None,
        rtol: float = 1.0e-11,
        atol: float = 0.0,
        projection: GaugeProjection[Tensor] | Callable[[Tensor], Tensor] | None = None,
        operator_is_hpd: bool = True,
    ) -> LinearSolveResult:
        """Solve ``(A^T N^-1 A + damping I) x = rhs`` matrix-free."""
        return _solve_normal_system_cg(
            self.normal_operator_action,
            rhs,
            initial_solution=initial_solution,
            iterations=iterations,
            damping=damping,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            projection=projection,
            operator_is_hpd=operator_is_hpd,
        )

    def solve_least_squares_lsqr(
        self,
        data: Sequence[Tensor],
        *,
        initial_solution: Tensor | None = None,
        iterations: int = 100,
        rtol: float = 1.0e-5,
        atol: float = 0.0,
        damping: float = 0.0,
        projection: GaugeProjection[Tensor] | Callable[[Tensor], Tensor] | None = None,
    ) -> LinearSolveResult:
        if len(data) != len(self.operators):
            raise ValueError("data count must match operator count")
        data_t = tuple(torch.as_tensor(item) for item in data)
        return lsqr(
            self.apply,
            self.adjoint,
            data_t,
            initial_solution=initial_solution,
            max_iterations=iterations,
            rtol=rtol,
            atol=atol,
            damping=damping,
            projection=projection,
        )

    def estimate_posterior_variance_diagonal(
        self,
        reference_state: Tensor,
        *,
        num_probes: int = 32,
        seed: int = 0,
        iterations: int = 32,
        damping: float = 0.0,
        preconditioner: PreconditionerFn | Tensor | None = None,
        rtol: float = 1.0e-5,
        atol: float = 0.0,
    ) -> Tensor:
        """Estimate inverse curvature with signed, residual-checked probes.

        Uses the same contract as
        ``BoundObservationOperator.estimate_posterior_variance_diagonal``:
        no positive floor, failure on an inadequate actual solve residual,
        and no claim that damped inverse curvature is sampling covariance.
        Checked finite probes do not certify global identifiability.
        """
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be positive")

        def _solve(rhs: Tensor) -> Tensor:
            return self.solve_normal_system_cg(
                rhs,
                iterations=iterations,
                damping=damping,
                preconditioner=preconditioner,
            ).solution

        return _estimate_inverse_diagonal_hutchinson(
            _solve,
            reference_state,
            num_probes=num_probes,
            seed=seed,
            normal_action=self.normal_operator_action,
            damping=damping,
            rtol=rtol,
            atol=atol,
        )

    def _residual_norm(
        self,
        scene: SceneRepresentation | nn.Module | Tensor,
        data: Sequence[Tensor],
    ) -> Tensor:
        if len(data) != len(self.operators):
            raise ValueError("data count must match operator count")
        squared_norm: Tensor | None = None
        offset = 0
        for operators in self._iter_operator_chunks():
            predictions = tuple(operator.apply(scene) for operator in operators)
            for prediction in predictions:
                datum = torch.as_tensor(data[offset])
                residual = datum.to(dtype=prediction.dtype) - prediction
                squared = torch.sum(torch.abs(residual).square())
                squared_norm = (
                    squared if squared_norm is None else squared_norm + squared
                )
                offset += 1
        assert squared_norm is not None
        return torch.sqrt(squared_norm)

    def _residual_adjoint_and_norm(
        self,
        scene: SceneRepresentation | nn.Module | Tensor,
        data: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if len(data) != len(self.operators):
            raise ValueError("data count must match operator count")
        total: Tensor | None = None
        squared_norm: Tensor | None = None
        offset = 0
        for operators in self._iter_operator_chunks():
            predictions = tuple(operator.apply(scene) for operator in operators)
            for operator, prediction in zip(operators, predictions, strict=True):
                datum = torch.as_tensor(data[offset])
                residual = datum.to(dtype=prediction.dtype) - prediction
                squared = torch.sum(torch.abs(residual).square())
                squared_norm = (
                    squared if squared_norm is None else squared_norm + squared
                )
                contribution = operator.adjoint(operator.apply_noise_model(residual))
                total = contribution if total is None else total + contribution
                offset += 1
        assert total is not None
        assert squared_norm is not None
        return total, torch.sqrt(squared_norm)

    def normal_equation_probe(
        self,
        scene: SceneRepresentation | nn.Module | Tensor,
        data: Sequence[Tensor],
    ) -> MultiObservationNormalEquationProbe:
        if len(data) != len(self.operators):
            raise ValueError("data count must match operator count")
        predictions = self.apply(scene)
        weighted_predictions = self.apply_noise_model(predictions)
        weighted_data = self.apply_noise_model(
            tuple(torch.as_tensor(item) for item in data)
        )
        return MultiObservationNormalEquationProbe(
            predictions=predictions,
            weighted_predictions=weighted_predictions,
            weighted_data=weighted_data,
            lhs_probe=self.adjoint(weighted_predictions),
            rhs=self.adjoint(weighted_data),
        )

    def estimate_normal_operator_scale(
        self,
        data: Sequence[Tensor],
        *,
        initial_probe: Tensor | None = None,
        iterations: int = 8,
    ) -> float:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if len(data) != len(self.operators):
            raise ValueError("data count must match operator count")
        probe = initial_probe
        if probe is None:
            probe = self.weighted_adjoint(data)
        state = _normalized_probe(torch.as_tensor(probe))
        estimate = 0.0
        for _ in range(int(iterations)):
            image = self.normal_operator_action(state)
            estimate = _rayleigh_quotient(state, image)
            state = _normalized_probe(image)
        return estimate

    def recommended_step_size(
        self,
        data: Sequence[Tensor],
        *,
        initial_probe: Tensor | None = None,
        iterations: int = 8,
        safety: float = 0.9,
    ) -> float:
        if safety <= 0.0 or safety > 1.0:
            raise ValueError("safety must satisfy 0 < safety <= 1")
        scale = self.estimate_normal_operator_scale(
            data,
            initial_probe=initial_probe,
            iterations=iterations,
        )
        return float(safety / max(scale, 1.0e-12))

    def estimate_normal_operator_diagonal(
        self,
        data: Sequence[Tensor],
        *,
        initial_probe: Tensor | None = None,
        damping: float = 0.0,
        num_probes: int = 16,
    ) -> Tensor:
        if len(data) != len(self.operators):
            raise ValueError("data count must match operator count")
        prototype = torch.as_tensor(
            initial_probe if initial_probe is not None else self.weighted_adjoint(data)
        )
        diagonal = torch.zeros_like(prototype)
        for _ in range(int(num_probes)):
            if torch.is_complex(prototype):
                re = (
                    torch.rand(
                        prototype.shape,
                        dtype=prototype.real.dtype,
                        device=prototype.device,
                    )
                    > 0.5
                ).to(prototype.real.dtype) * 2 - 1
                z = re.to(prototype.dtype)
            else:
                z = (torch.rand_like(prototype) > 0.5).to(prototype.dtype) * 2 - 1
            ANAz = self.normal_operator_action(z)
            if damping > 0.0:
                ANAz = ANAz + float(damping) * z
            diagonal = diagonal + (z.conj() * ANAz).real.to(diagonal.dtype)
        diagonal = diagonal / float(num_probes)
        if torch.is_complex(diagonal):
            diagonal = diagonal.real.clamp(min=0.0)
        else:
            diagonal = diagonal.clamp(min=0.0)
        return diagonal

    def solve_weighted_least_squares(
        self,
        data: Sequence[Tensor],
        *,
        initial_solution: Tensor | None = None,
        iterations: int = 16,
        step_size: float | None = None,
        scale_iterations: int = 8,
        step_safety: float = 0.9,
    ) -> LinearSolveResult:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if len(data) != len(self.operators):
            raise ValueError("data count must match operator count")
        history: list[float] = []
        data_t = tuple(torch.as_tensor(item) for item in data)
        if initial_solution is None:
            prototype = self.weighted_adjoint(data_t)
            solution = torch.zeros_like(prototype)
        else:
            solution = torch.as_tensor(initial_solution).clone()
        step = (
            float(step_size)
            if step_size is not None
            else self.recommended_step_size(
                data_t,
                initial_probe=solution,
                iterations=scale_iterations,
                safety=step_safety,
            )
        )
        if step <= 0.0:
            raise ValueError("step_size must be positive")
        for _ in range(int(iterations)):
            update, residual_norm = self._residual_adjoint_and_norm(
                solution,
                data_t,
            )
            history.append(float(residual_norm.item()))
            solution = solution + step * update.to(dtype=solution.dtype)
        return LinearSolveResult(
            solution=solution,
            residual_norm_history=tuple(history),
        )

    def solve_normal_equation_cg(
        self,
        data: Sequence[Tensor],
        *,
        initial_solution: Tensor | None = None,
        iterations: int = 16,
        damping: float = 0.0,
        preconditioner: Preconditioner[Tensor] | PreconditionerFn | Tensor | None = None,
        rtol: float = 1.0e-11,
        atol: float = 0.0,
        projection: GaugeProjection[Tensor] | Callable[[Tensor], Tensor] | None = None,
        operator_is_hpd: bool = True,
    ) -> LinearSolveResult:
        if len(data) != len(self.operators):
            raise ValueError("data count must match operator count")
        data_t = tuple(torch.as_tensor(item) for item in data)
        rhs = self.weighted_adjoint(data_t)
        return self.solve_normal_system_cg(
            rhs,
            initial_solution=initial_solution,
            iterations=iterations,
            damping=damping,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            projection=projection,
            operator_is_hpd=operator_is_hpd,
        )


def compose_bound_operators(
    *parts: BoundObservationOperator | BoundObservationOperatorSet,
    exposure_chunk_size: int = 1,
) -> BoundObservationOperatorSet:
    """Concatenate bound operators and operator sets into one set.

    Data operators and prior operators are stacked so a single normal-equation
    solve sees all of them; ``exposure_chunk_size`` bounds how many are applied
    at once.
    """
    operators: list[BoundObservationOperator] = []
    for part in parts:
        if isinstance(part, BoundObservationOperator):
            operators.append(part)
        elif isinstance(part, BoundObservationOperatorSet):
            operators.extend(part.operators)
        else:
            raise TypeError(
                "compose_bound_operators accepts bound operators or bound-operator sets"
            )
    return BoundObservationOperatorSet(
        tuple(operators),
        exposure_chunk_size=exposure_chunk_size,
    )
