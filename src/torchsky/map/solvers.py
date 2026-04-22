"""Internal solver hooks for future discrete inverse mapmaking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from ..sphere.spectral import _alm_ell_array
from .healpix_image_plane import accumulate_image_on_healpix_map_adjoint
from .healpix_spectral import (
    accumulate_image_on_alm_adjoint,
    accumulate_image_on_spin_alm_adjoint,
)
from .operators.bundle import ObservationOperatorBundle
from .operators.bundle import spherical_spectral_operator_bundle
from .scenes import (
    DiscreteTargetImage,
    HarmonicAlmScene,
    HealpixMapScene,
    SceneRepresentation,
    SpinHarmonicAlmScene,
    as_scene_representation,
)
from .target import TargetRenderSpec


ApplyOperatorFn = Callable[[SceneRepresentation, TargetRenderSpec], Tensor]
AdjointOperatorFn = Callable[[Tensor], Tensor]
ApplyNoiseFn = Callable[[Tensor], Tensor]
PreconditionerFn = Callable[[Tensor], Tensor]
BindTargetFn = Callable[
    [ObservationOperatorBundle, TargetRenderSpec],
    tuple[ApplyOperatorFn | None, AdjointOperatorFn | None, ApplyNoiseFn | None],
]


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
class LinearSolveResult:
    solution: Tensor
    residual_norm_history: tuple[float, ...]


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


def _rayleigh_quotient(state: Tensor, image: Tensor) -> float:
    state_flat = state.reshape(-1).to(torch.complex128)
    image_flat = image.reshape(-1).to(torch.complex128)
    denom = torch.sum(torch.conj(state_flat) * state_flat).real
    if float(denom.item()) <= 0.0:
        return 0.0
    numer = torch.sum(torch.conj(state_flat) * image_flat).real
    value = float((numer / denom).item())
    return max(value, 0.0)


def _inner_product(lhs: Tensor, rhs: Tensor) -> Tensor:
    lhs_flat = torch.as_tensor(lhs).reshape(-1).to(torch.complex128)
    rhs_flat = torch.as_tensor(rhs).reshape(-1).to(torch.complex128)
    return torch.sum(torch.conj(lhs_flat) * rhs_flat)


def _scene_state_tensor(scene: SceneRepresentation) -> Tensor:
    if isinstance(scene, DiscreteTargetImage):
        return torch.as_tensor(scene.data)
    if isinstance(scene, HealpixMapScene):
        return torch.as_tensor(scene.data)
    if isinstance(scene, HarmonicAlmScene):
        return torch.as_tensor(scene.data)
    if isinstance(scene, SpinHarmonicAlmScene):
        return torch.stack(
            [torch.as_tensor(scene.alm_e), torch.as_tensor(scene.alm_b)],
            dim=0,
        )
    raise TypeError(
        "state-space identity operator requires a tensor-backed scene representation"
    )


def _flatten_state(tensor: Tensor) -> Tensor:
    return torch.as_tensor(tensor).reshape(-1)


def _reshape_state(flat: Tensor, reference: Tensor) -> Tensor:
    return torch.as_tensor(flat).reshape_as(reference)


def _canonical_subspace_indices(
    reference_state: Tensor,
    selector: int | slice | Sequence[int] | Tensor,
) -> tuple[Tensor, bool]:
    state = torch.as_tensor(reference_state)
    if state.ndim < 1:
        raise ValueError("reference_state must have at least one dimension")
    if isinstance(selector, int):
        idx = torch.tensor([int(selector)], dtype=torch.int64)
        squeeze = True
    elif isinstance(selector, slice):
        start, stop, step = selector.indices(int(state.shape[0]))
        idx = torch.arange(start, stop, step, dtype=torch.int64)
        squeeze = False
    else:
        idx = torch.as_tensor(selector, dtype=torch.int64).reshape(-1)
        squeeze = False
    if idx.numel() == 0:
        raise ValueError("subspace selector must select at least one index")
    if int(idx.min()) < 0 or int(idx.max()) >= int(state.shape[0]):
        raise ValueError("subspace selector is out of bounds for reference_state")
    return idx, squeeze


def make_diagonal_preconditioner(diagonal: Tensor) -> PreconditionerFn:
    diagonal_t = torch.as_tensor(diagonal)
    reciprocal = torch.where(
        torch.abs(diagonal_t) > 0,
        1.0 / diagonal_t,
        torch.zeros_like(diagonal_t),
    )

    def _apply(vector: Tensor) -> Tensor:
        return reciprocal.to(dtype=vector.dtype, device=vector.device) * vector

    return _apply


def _apply_preconditioner(
    preconditioner: PreconditionerFn | Tensor | None,
    vector: Tensor,
) -> Tensor:
    if preconditioner is None:
        return vector
    if callable(preconditioner):
        return torch.as_tensor(preconditioner(vector))
    return (
        torch.as_tensor(preconditioner).to(dtype=vector.dtype, device=vector.device)
        * vector
    )


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
        max_entries: int = 4096,
        damping: float = 0.0,
    ) -> Tensor:
        prototype = torch.as_tensor(
            initial_probe
            if initial_probe is not None
            else self.adjoint(self.apply_noise_model(torch.as_tensor(data)))
        )
        if int(prototype.numel()) > int(max_entries):
            raise ValueError(
                "exact normal-operator diagonal probe exceeds max_entries; pass a smaller state or raise max_entries"
            )
        diagonal = torch.empty_like(prototype)
        flat_diagonal = diagonal.reshape(-1)
        flat = prototype.reshape(-1)
        for idx in range(int(flat.numel())):
            basis = torch.zeros_like(flat)
            basis[idx] = 1
            basis_state = basis.reshape_as(prototype)
            image = self.adjoint(self.apply_noise_model(self.apply(basis_state)))
            if damping > 0.0:
                image = image + float(damping) * basis_state
            flat_diagonal[idx] = image.reshape(-1)[idx].to(dtype=flat_diagonal.dtype)
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
        preconditioner: PreconditionerFn | Tensor | None = None,
    ) -> LinearSolveResult:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if damping < 0.0:
            raise ValueError("damping must be non-negative")
        data_t = torch.as_tensor(data)
        rhs = self.adjoint(self.apply_noise_model(data_t))
        if initial_solution is None:
            solution = torch.zeros_like(rhs)
        else:
            solution = torch.as_tensor(initial_solution).clone()

        def _normal_action(state: Tensor) -> Tensor:
            image = self.adjoint(self.apply_noise_model(self.apply(state)))
            if damping > 0.0:
                image = image + float(damping) * state
            return image

        residual = rhs - _normal_action(solution)
        preconditioned_residual = _apply_preconditioner(preconditioner, residual).to(
            dtype=residual.dtype
        )
        direction = preconditioned_residual.clone()
        rz_old = _inner_product(residual, preconditioned_residual).real
        history: list[float] = []
        for _ in range(int(iterations)):
            image_residual = data_t.to(dtype=self.apply(solution).dtype) - self.apply(
                solution
            )
            history.append(float(torch.linalg.vector_norm(image_residual).item()))
            ap = _normal_action(direction)
            denom = _inner_product(direction, ap).real
            if float(torch.abs(denom).item()) <= 0.0:
                break
            alpha = rz_old / denom
            solution = solution + alpha.to(dtype=solution.dtype) * direction
            residual = residual - alpha.to(dtype=residual.dtype) * ap
            preconditioned_residual = _apply_preconditioner(
                preconditioner, residual
            ).to(dtype=residual.dtype)
            rz_new = _inner_product(residual, preconditioned_residual).real
            if float(torch.sqrt(torch.clamp_min(rz_new, 0.0)).item()) == 0.0:
                break
            beta = rz_new / rz_old
            direction = (
                preconditioned_residual + beta.to(dtype=direction.dtype) * direction
            )
            rz_old = rz_new
        return LinearSolveResult(
            solution=solution,
            residual_norm_history=tuple(history),
        )


@dataclass
class BoundObservationOperatorSet:
    """Shared-scene operator assembled from multiple target-bound observations."""

    operators: tuple[BoundObservationOperator, ...]

    def __post_init__(self) -> None:
        if len(self.operators) == 0:
            raise ValueError(
                "BoundObservationOperatorSet requires at least one operator"
            )

    def apply(
        self,
        scene: SceneRepresentation | nn.Module | Tensor,
    ) -> tuple[Tensor, ...]:
        return tuple(operator.apply(scene) for operator in self.operators)

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
            weighted_data = self.apply_noise_model(
                tuple(torch.as_tensor(item) for item in data)
            )
            probe = self.adjoint(weighted_data)
        state = _normalized_probe(torch.as_tensor(probe))
        estimate = 0.0
        for _ in range(int(iterations)):
            weighted_predictions = self.apply_noise_model(self.apply(state))
            image = self.adjoint(weighted_predictions)
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
        max_entries: int = 4096,
        damping: float = 0.0,
    ) -> Tensor:
        if len(data) != len(self.operators):
            raise ValueError("data count must match operator count")
        prototype = torch.as_tensor(
            initial_probe
            if initial_probe is not None
            else self.adjoint(
                self.apply_noise_model(tuple(torch.as_tensor(item) for item in data))
            )
        )
        if int(prototype.numel()) > int(max_entries):
            raise ValueError(
                "exact normal-operator diagonal probe exceeds max_entries; pass a smaller state or raise max_entries"
            )
        diagonal = torch.empty_like(prototype)
        flat_diagonal = diagonal.reshape(-1)
        flat = prototype.reshape(-1)
        for idx in range(int(flat.numel())):
            basis = torch.zeros_like(flat)
            basis[idx] = 1
            basis_state = basis.reshape_as(prototype)
            image = self.adjoint(self.apply_noise_model(self.apply(basis_state)))
            if damping > 0.0:
                image = image + float(damping) * basis_state
            flat_diagonal[idx] = image.reshape(-1)[idx].to(dtype=flat_diagonal.dtype)
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
            predictions = self.apply(solution)
            residuals = tuple(
                datum.to(dtype=prediction.dtype) - prediction
                for datum, prediction in zip(data_t, predictions, strict=True)
            )
            residual_norm = torch.sqrt(
                torch.stack(
                    [torch.sum(torch.abs(residual).square()) for residual in residuals]
                ).sum()
            )
            history.append(float(residual_norm.item()))
            weighted_residuals = self.apply_noise_model(residuals)
            update = self.adjoint(weighted_residuals).to(dtype=solution.dtype)
            solution = solution + step * update
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
        preconditioner: PreconditionerFn | Tensor | None = None,
    ) -> LinearSolveResult:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if damping < 0.0:
            raise ValueError("damping must be non-negative")
        if len(data) != len(self.operators):
            raise ValueError("data count must match operator count")
        data_t = tuple(torch.as_tensor(item) for item in data)
        rhs = self.adjoint(self.apply_noise_model(data_t))
        if initial_solution is None:
            solution = torch.zeros_like(rhs)
        else:
            solution = torch.as_tensor(initial_solution).clone()

        def _normal_action(state: Tensor) -> Tensor:
            image = self.adjoint(self.apply_noise_model(self.apply(state)))
            if damping > 0.0:
                image = image + float(damping) * state
            return image

        residual = rhs - _normal_action(solution)
        preconditioned_residual = _apply_preconditioner(preconditioner, residual).to(
            dtype=residual.dtype
        )
        direction = preconditioned_residual.clone()
        rz_old = _inner_product(residual, preconditioned_residual).real
        history: list[float] = []
        for _ in range(int(iterations)):
            predictions = self.apply(solution)
            residuals = tuple(
                datum.to(dtype=prediction.dtype) - prediction
                for datum, prediction in zip(data_t, predictions, strict=True)
            )
            residual_norm = torch.sqrt(
                torch.stack(
                    [torch.sum(torch.abs(resid).square()) for resid in residuals]
                ).sum()
            )
            history.append(float(residual_norm.item()))
            ap = _normal_action(direction)
            denom = _inner_product(direction, ap).real
            if float(torch.abs(denom).item()) <= 0.0:
                break
            alpha = rz_old / denom
            solution = solution + alpha.to(dtype=solution.dtype) * direction
            residual = residual - alpha.to(dtype=residual.dtype) * ap
            preconditioned_residual = _apply_preconditioner(
                preconditioner, residual
            ).to(dtype=residual.dtype)
            rz_new = _inner_product(residual, preconditioned_residual).real
            if float(torch.sqrt(torch.clamp_min(rz_new, 0.0)).item()) == 0.0:
                break
            beta = rz_new / rz_old
            direction = (
                preconditioned_residual + beta.to(dtype=direction.dtype) * direction
            )
            rz_old = rz_new
        return LinearSolveResult(
            solution=solution,
            residual_norm_history=tuple(history),
        )


def compose_bound_operators(
    *parts: BoundObservationOperator | BoundObservationOperatorSet,
) -> BoundObservationOperatorSet:
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
    return BoundObservationOperatorSet(tuple(operators))


def build_state_identity_operator(
    reference_state: Tensor,
    *,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    target = TargetRenderSpec(
        header={
            "NAXIS": 2,
            "NAXIS1": int(state.numel()),
            "NAXIS2": 1,
        },
        shape=(1, int(state.numel())),
    )

    def _apply_operator(
        scene: SceneRepresentation, target_spec: TargetRenderSpec
    ) -> Tensor:
        del target_spec
        tensor = _scene_state_tensor(scene)
        if tuple(int(v) for v in tensor.shape) != tuple(int(v) for v in state.shape):
            raise ValueError(
                "state-space identity operator received a scene with incompatible shape"
            )
        return tensor

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        mapped = torch.as_tensor(tensor)
        if tuple(int(v) for v in mapped.shape) != tuple(int(v) for v in state.shape):
            raise ValueError(
                "state-space identity adjoint received data with incompatible shape"
            )
        return mapped

    return BoundObservationOperator(
        operators=ObservationOperatorBundle(),
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=apply_noise,
    )


def build_state_subspace_operator(
    reference_state: Tensor,
    base_operator: BoundObservationOperator | BoundObservationOperatorSet,
    selector: int | slice | Sequence[int] | Tensor,
) -> BoundObservationOperator | BoundObservationOperatorSet:
    state = torch.as_tensor(reference_state)
    indices, squeeze = _canonical_subspace_indices(state, selector)

    def _extract(scene: SceneRepresentation | nn.Module | Tensor) -> Tensor:
        tensor = _scene_state_tensor(as_scene_representation(scene))
        if tuple(int(v) for v in tensor.shape) != tuple(int(v) for v in state.shape):
            raise ValueError(
                "subspace operator received a scene with incompatible shape"
            )
        subset = tensor.index_select(0, indices.to(device=tensor.device))
        if squeeze:
            return subset[0]
        return subset

    def _embed(adjoint_tensor: Tensor) -> Tensor:
        out = torch.zeros_like(state)
        if squeeze:
            adj = torch.as_tensor(adjoint_tensor)
            if adj.ndim != state.ndim - 1 or tuple(int(v) for v in adj.shape) != tuple(
                int(v) for v in state.shape[1:]
            ):
                raise ValueError("subspace adjoint returned an incompatible tensor")
            out[indices[0]] = adj.to(dtype=state.dtype, device=state.device)
            return out
        adj = torch.as_tensor(adjoint_tensor)
        expected = (int(indices.numel()), *tuple(int(v) for v in state.shape[1:]))
        if tuple(int(v) for v in adj.shape) != expected:
            raise ValueError("subspace adjoint returned an incompatible tensor")
        out.index_copy_(
            0,
            indices.to(device=out.device),
            adj.to(dtype=state.dtype, device=out.device),
        )
        return out

    def _wrap_bound(operator: BoundObservationOperator) -> BoundObservationOperator:
        def _apply_operator(
            scene: SceneRepresentation, target_spec: TargetRenderSpec
        ) -> Tensor:
            del target_spec
            return operator.apply(_extract(scene))

        def _apply_adjoint(tensor: Tensor) -> Tensor:
            return _embed(operator.adjoint(tensor))

        return BoundObservationOperator(
            operators=operator.operators,
            target=operator.target,
            apply_operator=_apply_operator,
            apply_adjoint=_apply_adjoint,
            apply_noise=operator.apply_noise,
        )

    if isinstance(base_operator, BoundObservationOperator):
        return _wrap_bound(base_operator)
    if isinstance(base_operator, BoundObservationOperatorSet):
        return BoundObservationOperatorSet(
            tuple(_wrap_bound(op) for op in base_operator.operators)
        )
    raise TypeError(
        "base_operator must be a BoundObservationOperator or BoundObservationOperatorSet"
    )


def build_state_component_mixture_operator(
    reference_state: Tensor,
    base_operator: BoundObservationOperator | BoundObservationOperatorSet,
    components: Sequence[int] | Tensor,
    *,
    weights: Sequence[complex | float] | Tensor | None = None,
) -> BoundObservationOperator | BoundObservationOperatorSet:
    state = torch.as_tensor(reference_state)
    if state.ndim < 2:
        raise ValueError(
            "component mixture operator requires a reference_state with a leading component axis"
        )
    component_idx = torch.as_tensor(components, dtype=torch.int64).reshape(-1)
    if int(component_idx.numel()) == 0:
        raise ValueError("components must select at least one latent component")
    if int(component_idx.min()) < 0 or int(component_idx.max()) >= int(state.shape[0]):
        raise ValueError("components are out of bounds for the reference_state")
    weights_t = (
        torch.ones(
            int(component_idx.numel()),
            dtype=state.dtype if state.is_complex() else torch.float32,
            device=state.device,
        )
        if weights is None
        else torch.as_tensor(
            weights,
            dtype=state.dtype if state.is_complex() else torch.float32,
            device=state.device,
        ).reshape(-1)
    )
    if int(weights_t.numel()) != int(component_idx.numel()):
        raise ValueError("weights length must match the number of selected components")

    def _mix(scene: SceneRepresentation | nn.Module | Tensor) -> Tensor:
        tensor = _scene_state_tensor(as_scene_representation(scene))
        if tuple(int(v) for v in tensor.shape) != tuple(int(v) for v in state.shape):
            raise ValueError(
                "component mixture operator received a scene with incompatible shape"
            )
        subset = tensor.index_select(0, component_idx.to(device=tensor.device))
        coeffs = weights_t.to(dtype=subset.dtype, device=subset.device).view(
            int(component_idx.numel()),
            *([1] * (subset.ndim - 1)),
        )
        return torch.sum(coeffs * subset, dim=0)

    def _embed(adjoint_tensor: Tensor) -> Tensor:
        adj = torch.as_tensor(adjoint_tensor)
        expected = tuple(int(v) for v in state.shape[1:])
        if tuple(int(v) for v in adj.shape) != expected:
            raise ValueError(
                "component mixture adjoint returned an incompatible tensor"
            )
        out = torch.zeros_like(state)
        coeffs = torch.conj(weights_t.to(dtype=adj.dtype, device=adj.device))
        for idx, weight in zip(component_idx.tolist(), coeffs, strict=True):
            out[idx] = out[idx] + (weight * adj).to(dtype=out.dtype, device=out.device)
        return out

    def _wrap_bound(operator: BoundObservationOperator) -> BoundObservationOperator:
        def _apply_operator(
            scene: SceneRepresentation, target_spec: TargetRenderSpec
        ) -> Tensor:
            del target_spec
            return operator.apply(_mix(scene))

        def _apply_adjoint(tensor: Tensor) -> Tensor:
            return _embed(operator.adjoint(tensor))

        return BoundObservationOperator(
            operators=operator.operators,
            target=operator.target,
            apply_operator=_apply_operator,
            apply_adjoint=_apply_adjoint,
            apply_noise=operator.apply_noise,
        )

    if isinstance(base_operator, BoundObservationOperator):
        return _wrap_bound(base_operator)
    if isinstance(base_operator, BoundObservationOperatorSet):
        return BoundObservationOperatorSet(
            tuple(_wrap_bound(op) for op in base_operator.operators)
        )
    raise TypeError(
        "base_operator must be a BoundObservationOperator or BoundObservationOperatorSet"
    )


class SurveyInstrumentComponentMixer(nn.Module):
    """Mechanistic learned context mixer over latent shared/private components.

    Each latent component is assigned a role:
    - ``shared``: seen by every context through one learned global weight
    - ``survey``: context-specific survey weight
    - ``instrument``: context-specific instrument weight
    - ``interaction``: context-specific survey-instrument interaction weight

    This keeps the decomposition explicit while still allowing the relative
    contribution of each component family to be learned from data.
    """

    _VALID_ROLES = ("shared", "survey", "instrument", "interaction")

    def __init__(
        self,
        *,
        component_roles: Sequence[str],
        surveys: Sequence[str] = (),
        instruments: Sequence[str] = (),
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        roles = tuple(str(role) for role in component_roles)
        unknown = tuple(role for role in roles if role not in self._VALID_ROLES)
        if unknown:
            raise ValueError(f"unsupported component roles: {unknown!r}")
        self.component_roles = roles
        self.surveys = tuple(str(name) for name in surveys)
        self.instruments = tuple(str(name) for name in instruments)
        ncomp = len(roles)
        self.shared_weights = nn.Parameter(
            torch.zeros(ncomp, dtype=dtype, device=device)
        )
        self.survey_weights = nn.Parameter(
            torch.zeros(len(self.surveys) + 1, ncomp, dtype=dtype, device=device)
        )
        self.instrument_weights = nn.Parameter(
            torch.zeros(len(self.instruments) + 1, ncomp, dtype=dtype, device=device)
        )
        self.interaction_weights = nn.Parameter(
            torch.zeros(
                len(self.surveys) + 1,
                len(self.instruments) + 1,
                ncomp,
                dtype=dtype,
                device=device,
            )
        )
        role_tensor = {
            role: torch.tensor(
                [1.0 if item == role else 0.0 for item in roles],
                dtype=dtype,
                device=device,
            )
            for role in self._VALID_ROLES
        }
        for role, tensor in role_tensor.items():
            self.register_buffer(f"{role}_mask", tensor)

    def _survey_index(self, survey: str | None) -> int:
        if survey is None:
            return 0
        try:
            return self.surveys.index(str(survey)) + 1
        except ValueError:
            return 0

    def _instrument_index(self, instrument: str | None) -> int:
        if instrument is None:
            return 0
        try:
            return self.instruments.index(str(instrument)) + 1
        except ValueError:
            return 0

    def forward(
        self,
        *,
        survey: str | None = None,
        instrument: str | None = None,
    ) -> Tensor:
        sidx = self._survey_index(survey)
        iidx = self._instrument_index(instrument)
        return (
            self.shared_mask * self.shared_weights
            + self.survey_mask * self.survey_weights[sidx]
            + self.instrument_mask * self.instrument_weights[iidx]
            + self.interaction_mask * self.interaction_weights[sidx, iidx]
        )

    def scale_component_columns_(
        self,
        scales: Tensor | Sequence[float],
    ) -> None:
        """Rescale role-matched mixer columns in-place for exact gauge preservation."""
        scales_t = torch.as_tensor(
            scales,
            dtype=self.shared_weights.dtype,
            device=self.shared_weights.device,
        ).reshape(-1)
        if int(scales_t.numel()) != len(self.component_roles):
            raise ValueError("scales length must match the number of components")
        if not torch.all(torch.isfinite(scales_t)):
            raise ValueError("scales must be finite")
        if torch.any(scales_t == 0.0):
            raise ValueError("scales must be non-zero")
        with torch.no_grad():
            for idx, role in enumerate(self.component_roles):
                scale = scales_t[idx]
                if role == "shared":
                    self.shared_weights[idx].mul_(scale)
                elif role == "survey":
                    self.survey_weights[:, idx].mul_(scale)
                elif role == "instrument":
                    self.instrument_weights[:, idx].mul_(scale)
                elif role == "interaction":
                    self.interaction_weights[:, :, idx].mul_(scale)
                else:  # pragma: no cover - guarded by __init__
                    raise ValueError(f"unsupported component role {role!r}")


@dataclass(frozen=True)
class ComponentDecompositionDiagnostics:
    """Diagnostics for explicit shared/private latent decompositions."""

    shared_to_private_leakage: float
    private_to_shared_leakage: float
    shared_relative_error: float
    private_relative_error: float
    calibration_transfer_stability: float | None = None


class HarmonicComponentDecomposition(nn.Module):
    """Learnable harmonic component state with explicit shared/private roles."""

    def __init__(
        self,
        *,
        lmax: int,
        component_roles: Sequence[str],
        surveys: Sequence[str] = (),
        instruments: Sequence[str] = (),
        init_scale: float = 1.0e-2,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        ll = int(lmax)
        if ll < 0:
            raise ValueError("lmax must be non-negative")
        self.lmax = ll
        self.nalm = (ll + 1) * (ll + 2) // 2
        self.component_roles = tuple(str(role) for role in component_roles)
        ncomp = len(self.component_roles)
        self.coeff_real = nn.Parameter(
            float(init_scale)
            * torch.randn(ncomp, self.nalm, dtype=dtype, device=device)
        )
        self.coeff_imag = nn.Parameter(
            float(init_scale)
            * torch.randn(ncomp, self.nalm, dtype=dtype, device=device)
        )
        self.mixer = SurveyInstrumentComponentMixer(
            component_roles=self.component_roles,
            surveys=surveys,
            instruments=instruments,
            dtype=dtype,
            device=device,
        )

    def state(self) -> Tensor:
        return torch.complex(self.coeff_real, self.coeff_imag)

    def forward(self) -> Tensor:
        return self.state()

    def conditioned_operator(
        self,
        base_operator: BoundObservationOperator | BoundObservationOperatorSet,
        *,
        survey: str | None = None,
        instrument: str | None = None,
    ) -> BoundObservationOperator | BoundObservationOperatorSet:
        return build_conditioned_component_mixture_operator(
            self.state(),
            base_operator,
            self.mixer,
            survey=survey,
            instrument=instrument,
        )

    def component_mask(self, role: str) -> Tensor:
        if role not in SurveyInstrumentComponentMixer._VALID_ROLES:
            raise ValueError(f"unsupported component role {role!r}")
        return getattr(self.mixer, f"{role}_mask").to(
            device=self.coeff_real.device,
            dtype=self.coeff_real.dtype,
        )

    def component_state(
        self,
        roles: str | Sequence[str],
        *,
        state: Tensor | None = None,
        aggregate: bool = True,
    ) -> Tensor:
        state_t = self.state() if state is None else torch.as_tensor(state)
        if state_t.ndim < 2 or int(state_t.shape[0]) != len(self.component_roles):
            raise ValueError(
                "state must have a leading component axis matching component_roles"
            )
        role_names = (
            (roles,) if isinstance(roles, str) else tuple(str(role) for role in roles)
        )
        unknown = tuple(
            role
            for role in role_names
            if role not in SurveyInstrumentComponentMixer._VALID_ROLES
        )
        if unknown:
            raise ValueError(f"unsupported component roles: {unknown!r}")
        selected = [
            idx
            for idx, role in enumerate(self.component_roles)
            if role in set(role_names)
        ]
        if not selected:
            empty = state_t[:0]
            if aggregate:
                return torch.zeros_like(state_t[0])
            return empty
        subset = state_t.index_select(
            0,
            torch.tensor(selected, dtype=torch.int64, device=state_t.device),
        )
        if aggregate:
            return subset.sum(dim=0)
        return subset

    def component_norms(self) -> Tensor:
        state = self.state()
        return torch.linalg.vector_norm(state, dim=1).to(dtype=self.coeff_real.dtype)

    def fix_component_scale_gauge_(
        self,
        *,
        target_norm: float = 1.0,
        min_norm: float = 1.0e-12,
    ) -> Tensor:
        """Canonicalize multiplicative component gauge without changing predictions.

        Each component column is rescaled to ``target_norm`` in harmonic state space,
        and the matching mixer weights are scaled by the inverse factor so every
        conditioned operator remains exactly unchanged.
        """
        if target_norm <= 0.0:
            raise ValueError("target_norm must be positive")
        if min_norm <= 0.0:
            raise ValueError("min_norm must be positive")
        norms = self.component_norms()
        scale = torch.where(
            norms > float(min_norm),
            norms / float(target_norm),
            torch.ones_like(norms),
        )
        with torch.no_grad():
            self.coeff_real.div_(scale[:, None])
            self.coeff_imag.div_(scale[:, None])
        self.mixer.scale_component_columns_(scale)
        return self.component_norms()

    def decomposition_diagnostics(
        self,
        truth_state: Tensor,
        *,
        state: Tensor | None = None,
        calibration_anchor: Tensor | None = None,
        calibration_selector: int | slice | Sequence[int] | Tensor | None = None,
    ) -> ComponentDecompositionDiagnostics:
        estimate = self.state() if state is None else torch.as_tensor(state)
        truth = torch.as_tensor(truth_state)
        if tuple(int(v) for v in estimate.shape) != tuple(int(v) for v in truth.shape):
            raise ValueError("state and truth_state must have matching shape")
        if estimate.ndim < 2 or int(estimate.shape[0]) != len(self.component_roles):
            raise ValueError(
                "state and truth_state must have a leading component axis matching component_roles"
            )
        shared_truth = self.component_state("shared", state=truth, aggregate=True)
        private_truth = self.component_state(
            ("survey", "instrument", "interaction"),
            state=truth,
            aggregate=True,
        )
        private_truth_basis = self.component_state(
            ("survey", "instrument", "interaction"),
            state=truth,
            aggregate=False,
        )
        shared_estimate = self.component_state("shared", state=estimate, aggregate=True)
        private_estimate = self.component_state(
            ("survey", "instrument", "interaction"),
            state=estimate,
            aggregate=True,
        )

        def _relative_norm(error: Tensor, reference: Tensor) -> float:
            error_norm = torch.linalg.vector_norm(torch.as_tensor(error).reshape(-1))
            ref_norm = torch.linalg.vector_norm(torch.as_tensor(reference).reshape(-1))
            if float(ref_norm.item()) == 0.0:
                return (
                    0.0 if float(error_norm.item()) == 0.0 else float(error_norm.item())
                )
            return float((error_norm / ref_norm).item())

        def _subspace_leakage(error: Tensor, basis: Tensor, reference: Tensor) -> float:
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
                return (
                    0.0 if float(proj_norm.item()) == 0.0 else float(proj_norm.item())
                )
            return float((proj_norm / ref_norm).item())

        shared_error = shared_estimate - shared_truth
        private_error = private_estimate - private_truth
        calibration_transfer_stability = None
        if calibration_anchor is not None:
            anchor = torch.as_tensor(calibration_anchor)
            if tuple(int(v) for v in anchor.shape) != tuple(
                int(v) for v in shared_truth.shape
            ):
                raise ValueError(
                    "calibration_anchor must match the aggregated shared-state shape"
                )
            shared_for_anchor = shared_estimate
            anchor_ref = anchor
            if calibration_selector is not None:
                selector_idx, _ = _canonical_subspace_indices(
                    shared_truth, calibration_selector
                )
                shared_for_anchor = shared_for_anchor.index_select(0, selector_idx)
                anchor_ref = anchor_ref.index_select(
                    0, selector_idx.to(device=anchor_ref.device)
                )
            calibration_transfer_stability = _relative_norm(
                shared_for_anchor - anchor_ref,
                anchor_ref,
            )
        return ComponentDecompositionDiagnostics(
            shared_to_private_leakage=_subspace_leakage(
                shared_error,
                private_truth_basis,
                private_truth,
            ),
            private_to_shared_leakage=_subspace_leakage(
                private_error,
                shared_truth.unsqueeze(0),
                shared_truth,
            ),
            shared_relative_error=_relative_norm(shared_error, shared_truth),
            private_relative_error=_relative_norm(private_error, private_truth),
            calibration_transfer_stability=calibration_transfer_stability,
        )

    def regularization_loss(
        self,
        *,
        private_harmonic_diagonal: Tensor | None = None,
        private_weight: float = 1.0,
        mixer_weight: float = 1.0,
    ) -> Tensor:
        state = self.state()
        loss = torch.zeros(
            (), dtype=self.coeff_real.dtype, device=self.coeff_real.device
        )
        private_mask = (
            self.component_mask("survey")
            + self.component_mask("instrument")
            + self.component_mask("interaction")
        ).clamp(max=1.0)
        if private_harmonic_diagonal is not None:
            diag = torch.as_tensor(
                private_harmonic_diagonal,
                dtype=self.coeff_real.dtype,
                device=self.coeff_real.device,
            ).reshape(1, self.nalm)
            weighted = diag * torch.abs(state)
            loss = loss + float(private_weight) * torch.mean(
                private_mask.view(-1, 1) * torch.square(weighted)
            )
        loss = loss + float(mixer_weight) * (
            torch.mean(torch.square(self.mixer.survey_weights))
            + torch.mean(torch.square(self.mixer.instrument_weights))
            + torch.mean(torch.square(self.mixer.interaction_weights))
        )
        return loss


def build_conditioned_component_mixture_operator(
    reference_state: Tensor,
    base_operator: BoundObservationOperator | BoundObservationOperatorSet,
    mixer: SurveyInstrumentComponentMixer,
    *,
    survey: str | None = None,
    instrument: str | None = None,
) -> BoundObservationOperator | BoundObservationOperatorSet:
    state = torch.as_tensor(reference_state)
    if state.ndim < 2:
        raise ValueError(
            "conditioned component mixture requires a reference_state with a leading component axis"
        )
    ncomp = int(state.shape[0])
    if len(mixer.component_roles) != ncomp:
        raise ValueError(
            "mixer component_roles length must match reference_state leading dimension"
        )

    def _weights_for(dtype: torch.dtype, device: torch.device) -> Tensor:
        weights = mixer(survey=survey, instrument=instrument)
        return weights.to(dtype=dtype, device=device)

    def _mix(scene: SceneRepresentation | nn.Module | Tensor) -> Tensor:
        tensor = _scene_state_tensor(as_scene_representation(scene))
        if tuple(int(v) for v in tensor.shape) != tuple(int(v) for v in state.shape):
            raise ValueError(
                "conditioned component mixture received a scene with incompatible shape"
            )
        coeffs = _weights_for(tensor.dtype, tensor.device).view(
            ncomp,
            *([1] * (tensor.ndim - 1)),
        )
        return torch.sum(coeffs * tensor, dim=0)

    def _embed(adjoint_tensor: Tensor) -> Tensor:
        adj = torch.as_tensor(adjoint_tensor)
        expected = tuple(int(v) for v in state.shape[1:])
        if tuple(int(v) for v in adj.shape) != expected:
            raise ValueError(
                "conditioned component mixture adjoint returned an incompatible tensor"
            )
        coeffs = torch.conj(_weights_for(adj.dtype, adj.device)).view(
            ncomp,
            *([1] * adj.ndim),
        )
        return (coeffs * adj.unsqueeze(0)).to(dtype=state.dtype, device=state.device)

    def _wrap_bound(operator: BoundObservationOperator) -> BoundObservationOperator:
        def _apply_operator(
            scene: SceneRepresentation, target_spec: TargetRenderSpec
        ) -> Tensor:
            del target_spec
            return operator.apply(_mix(scene))

        def _apply_adjoint(tensor: Tensor) -> Tensor:
            return _embed(operator.adjoint(tensor))

        return BoundObservationOperator(
            operators=operator.operators,
            target=operator.target,
            apply_operator=_apply_operator,
            apply_adjoint=_apply_adjoint,
            apply_noise=operator.apply_noise,
        )

    if isinstance(base_operator, BoundObservationOperator):
        return _wrap_bound(base_operator)
    if isinstance(base_operator, BoundObservationOperatorSet):
        return BoundObservationOperatorSet(
            tuple(_wrap_bound(op) for op in base_operator.operators)
        )
    raise TypeError(
        "base_operator must be a BoundObservationOperator or BoundObservationOperatorSet"
    )


def build_state_diagonal_operator(
    reference_state: Tensor,
    diagonal: Tensor,
    *,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    diag = torch.as_tensor(diagonal)
    if tuple(int(v) for v in diag.shape) != tuple(int(v) for v in state.shape):
        raise ValueError("diagonal shape must match reference_state shape")
    target = TargetRenderSpec(
        header={
            "NAXIS": 2,
            "NAXIS1": int(state.numel()),
            "NAXIS2": 1,
        },
        shape=(1, int(state.numel())),
    )

    def _apply_operator(
        scene: SceneRepresentation, target_spec: TargetRenderSpec
    ) -> Tensor:
        del target_spec
        tensor = _scene_state_tensor(scene)
        if tuple(int(v) for v in tensor.shape) != tuple(int(v) for v in state.shape):
            raise ValueError(
                "state-space diagonal operator received a scene with incompatible shape"
            )
        return diag.to(dtype=tensor.dtype, device=tensor.device) * tensor

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        mapped = torch.as_tensor(tensor)
        if tuple(int(v) for v in mapped.shape) != tuple(int(v) for v in state.shape):
            raise ValueError(
                "state-space diagonal adjoint received data with incompatible shape"
            )
        return torch.conj(diag).to(dtype=mapped.dtype, device=mapped.device) * mapped

    return BoundObservationOperator(
        operators=ObservationOperatorBundle(),
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=apply_noise,
    )


def build_state_matrix_operator(
    reference_state: Tensor,
    matrix: Tensor,
    *,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    matrix_t = torch.as_tensor(matrix)
    nstate = int(state.numel())
    if (
        matrix_t.ndim != 2
        or int(matrix_t.shape[0]) != nstate
        or int(matrix_t.shape[1]) != nstate
    ):
        raise ValueError("matrix operator requires a square (nstate, nstate) matrix")
    target = TargetRenderSpec(
        header={
            "NAXIS": 2,
            "NAXIS1": nstate,
            "NAXIS2": 1,
        },
        shape=(1, nstate),
    )

    def _apply_operator(
        scene: SceneRepresentation, target_spec: TargetRenderSpec
    ) -> Tensor:
        del target_spec
        tensor = _scene_state_tensor(scene)
        if tuple(int(v) for v in tensor.shape) != tuple(int(v) for v in state.shape):
            raise ValueError(
                "state-space matrix operator received a scene with incompatible shape"
            )
        flat = _flatten_state(tensor).to(dtype=matrix_t.dtype, device=matrix_t.device)
        mapped = matrix_t @ flat
        return _reshape_state(
            mapped.to(dtype=tensor.dtype, device=tensor.device), state
        )

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        mapped = torch.as_tensor(tensor)
        if tuple(int(v) for v in mapped.shape) != tuple(int(v) for v in state.shape):
            raise ValueError(
                "state-space matrix adjoint received data with incompatible shape"
            )
        flat = _flatten_state(mapped).to(dtype=matrix_t.dtype, device=matrix_t.device)
        adj = torch.conj(matrix_t.transpose(-2, -1)) @ flat
        return _reshape_state(adj.to(dtype=mapped.dtype, device=mapped.device), state)

    return BoundObservationOperator(
        operators=ObservationOperatorBundle(),
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=apply_noise,
    )


def build_state_low_rank_operator(
    reference_state: Tensor,
    basis: Tensor,
    *,
    weights: Tensor | None = None,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    basis_t = torch.as_tensor(basis)
    nstate = int(state.numel())
    if basis_t.ndim != 2 or int(basis_t.shape[0]) != nstate:
        raise ValueError("low-rank basis must have shape (nstate, rank)")
    rank = int(basis_t.shape[1])
    if weights is None:
        weights_t = torch.ones(rank, dtype=basis_t.dtype, device=basis_t.device)
    else:
        weights_t = torch.as_tensor(weights, dtype=basis_t.dtype, device=basis_t.device)
        if weights_t.ndim != 1 or int(weights_t.shape[0]) != rank:
            raise ValueError("weights must have shape (rank,)")
    target = TargetRenderSpec(
        header={
            "NAXIS": 2,
            "NAXIS1": rank,
            "NAXIS2": 1,
        },
        shape=(1, rank),
    )

    def _apply_operator(
        scene: SceneRepresentation, target_spec: TargetRenderSpec
    ) -> Tensor:
        del target_spec
        tensor = _scene_state_tensor(scene)
        if tuple(int(v) for v in tensor.shape) != tuple(int(v) for v in state.shape):
            raise ValueError(
                "state-space low-rank operator received a scene with incompatible shape"
            )
        flat = _flatten_state(tensor).to(dtype=basis_t.dtype, device=basis_t.device)
        coeffs = torch.conj(basis_t).transpose(-2, -1) @ flat
        return (weights_t * coeffs).to(dtype=tensor.dtype, device=tensor.device)

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        mapped = torch.as_tensor(tensor)
        if mapped.ndim != 1 or int(mapped.shape[0]) != rank:
            raise ValueError(
                "state-space low-rank adjoint received data with incompatible shape"
            )
        coeffs = mapped.to(dtype=basis_t.dtype, device=basis_t.device)
        flat = basis_t @ (torch.conj(weights_t) * coeffs)
        return _reshape_state(flat.to(dtype=state.dtype, device=state.device), state)

    return BoundObservationOperator(
        operators=ObservationOperatorBundle(),
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=apply_noise,
    )


def build_state_kronecker_operator(
    reference_state: Tensor,
    left_factor: Tensor,
    right_factor: Tensor,
    *,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    if state.ndim != 2:
        raise ValueError("Kronecker state operator requires a 2D reference_state")
    left_t = torch.as_tensor(left_factor)
    right_t = torch.as_tensor(right_factor)
    rows, cols = int(state.shape[0]), int(state.shape[1])
    if left_t.ndim != 2 or int(left_t.shape[1]) != rows:
        raise ValueError("left_factor must have shape (out_rows, state_rows)")
    if right_t.ndim != 2 or int(right_t.shape[1]) != cols:
        raise ValueError("right_factor must have shape (out_cols, state_cols)")
    out_rows, out_cols = int(left_t.shape[0]), int(right_t.shape[0])
    target = TargetRenderSpec(
        header={
            "NAXIS": 2,
            "NAXIS1": out_cols,
            "NAXIS2": out_rows,
        },
        shape=(out_rows, out_cols),
    )

    def _apply_operator(
        scene: SceneRepresentation, target_spec: TargetRenderSpec
    ) -> Tensor:
        del target_spec
        tensor = _scene_state_tensor(scene)
        if tuple(int(v) for v in tensor.shape) != tuple(int(v) for v in state.shape):
            raise ValueError(
                "Kronecker state operator received a scene with incompatible shape"
            )
        tensor_t = tensor.to(dtype=left_t.dtype, device=left_t.device)
        mapped = left_t @ tensor_t @ torch.conj(right_t).transpose(-2, -1)
        return mapped.to(dtype=tensor.dtype, device=tensor.device)

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        mapped = torch.as_tensor(tensor)
        if mapped.ndim != 2 or tuple(int(v) for v in mapped.shape) != (
            out_rows,
            out_cols,
        ):
            raise ValueError("Kronecker adjoint received data with incompatible shape")
        mapped_t = mapped.to(dtype=left_t.dtype, device=left_t.device)
        adj = torch.conj(left_t).transpose(-2, -1) @ mapped_t @ right_t
        return adj.to(dtype=state.dtype, device=state.device)

    return BoundObservationOperator(
        operators=ObservationOperatorBundle(),
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=apply_noise,
    )


def harmonic_degree_diagonal(
    *,
    lmax: int,
    mmax: int | None = None,
    power: float = 1.0,
    ell_offset: float = 0.0,
    amplitude: float = 1.0,
    include_monopole: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if ll < 0:
        raise ValueError("lmax must be non-negative")
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    ell = _alm_ell_array(ll, mm).to(device=device)
    diagonal = float(amplitude) * torch.pow(
        ell.to(dtype=torch.float64) + float(ell_offset),
        float(power),
    )
    if include_monopole:
        diagonal = torch.where(ell == 0, torch.ones_like(diagonal), diagonal)
    else:
        diagonal = torch.where(ell == 0, torch.zeros_like(diagonal), diagonal)
    return diagonal.to(dtype=dtype)


def build_harmonic_degree_prior_operator(
    reference_state: Tensor,
    *,
    lmax: int,
    mmax: int | None = None,
    power: float = 1.0,
    ell_offset: float = 0.0,
    amplitude: float = 1.0,
    include_monopole: bool = True,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    if state.ndim == 1:
        diagonal = harmonic_degree_diagonal(
            lmax=lmax,
            mmax=mmax,
            power=power,
            ell_offset=ell_offset,
            amplitude=amplitude,
            include_monopole=include_monopole,
            dtype=torch.float32,
            device=state.device,
        ).to(dtype=state.real.dtype if state.is_complex() else state.dtype)
    elif state.ndim == 2 and int(state.shape[0]) == 2:
        base = harmonic_degree_diagonal(
            lmax=lmax,
            mmax=mmax,
            power=power,
            ell_offset=ell_offset,
            amplitude=amplitude,
            include_monopole=include_monopole,
            dtype=torch.float32,
            device=state.device,
        ).to(dtype=state.real.dtype if state.is_complex() else state.dtype)
        diagonal = torch.stack([base, base], dim=0)
    else:
        raise ValueError(
            "reference_state for harmonic prior must be shape (nalm,) or (2, nalm)"
        )
    return build_state_diagonal_operator(
        state,
        diagonal,
        apply_noise=apply_noise,
    )


def harmonic_dense_covariance(
    *,
    lmax: int,
    mmax: int | None = None,
    ell_correlation_length: float = 1.0,
    diagonal_jitter: float = 1.0e-3,
    amplitude: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> Tensor:
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if ll < 0:
        raise ValueError("lmax must be non-negative")
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    if ell_correlation_length <= 0.0:
        raise ValueError("ell_correlation_length must be positive")
    ell = _alm_ell_array(ll, mm).to(dtype=torch.float64, device=device)
    delta = ell[:, None] - ell[None, :]
    cov = float(amplitude) * torch.exp(
        -0.5 * torch.square(delta / float(ell_correlation_length))
    )
    cov = cov + float(diagonal_jitter) * torch.eye(
        int(cov.shape[0]),
        dtype=cov.dtype,
        device=cov.device,
    )
    return cov.to(dtype=dtype)


def harmonic_cl_diagonal(
    *,
    lmax: int,
    cls: Tensor,
    mmax: int | None = None,
    beam_transfer: Tensor | None = None,
    min_variance: float = 1.0e-8,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if ll < 0:
        raise ValueError("lmax must be non-negative")
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    cl_t = torch.as_tensor(cls, dtype=torch.float64, device=device)
    if cl_t.ndim != 1 or int(cl_t.numel()) < ll + 1:
        raise ValueError("cls must be a 1D spectrum with length >= lmax+1")
    variance_l = cl_t[: ll + 1]
    if beam_transfer is not None:
        beam_t = torch.as_tensor(beam_transfer, dtype=torch.float64, device=device)
        if beam_t.ndim != 1 or int(beam_t.numel()) < ll + 1:
            raise ValueError(
                "beam_transfer must be a 1D spectrum with length >= lmax+1"
            )
        variance_l = variance_l * torch.square(beam_t[: ll + 1])
    variance_l = torch.clamp(variance_l, min=float(min_variance))
    ell = _alm_ell_array(ll, mm).to(device=variance_l.device)
    diagonal = torch.rsqrt(variance_l.index_select(0, ell))
    return diagonal.to(dtype=dtype)


def build_harmonic_dense_prior_operator(
    reference_state: Tensor,
    covariance: Tensor,
    *,
    apply_noise: ApplyNoiseFn | None = None,
    jitter: float = 1.0e-8,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    covariance_t = torch.as_tensor(covariance)
    if state.ndim == 1:
        nstate = int(state.numel())
        if covariance_t.shape != (nstate, nstate):
            raise ValueError("covariance shape must match flattened harmonic state")
        eye = torch.eye(nstate, dtype=covariance_t.dtype, device=covariance_t.device)
        precision = torch.linalg.inv(covariance_t + float(jitter) * eye)
        factor = torch.linalg.cholesky(precision)
        return build_state_matrix_operator(
            state,
            factor,
            apply_noise=apply_noise,
        )
    if state.ndim == 2 and int(state.shape[0]) == 2:
        nalm = int(state.shape[1])
        if covariance_t.shape == (nalm, nalm):
            covariance_t = torch.block_diag(covariance_t, covariance_t)
        elif covariance_t.shape != (2 * nalm, 2 * nalm):
            raise ValueError(
                "spin-harmonic covariance must have shape (nalm, nalm) or (2*nalm, 2*nalm)"
            )
        eye = torch.eye(
            int(covariance_t.shape[0]),
            dtype=covariance_t.dtype,
            device=covariance_t.device,
        )
        precision = torch.linalg.inv(covariance_t + float(jitter) * eye)
        factor = torch.linalg.cholesky(precision)
        return build_state_matrix_operator(
            state,
            factor,
            apply_noise=apply_noise,
        )
    raise ValueError(
        "reference_state for dense harmonic prior must be shape (nalm,) or (2, nalm)"
    )


def build_harmonic_cl_prior_operator(
    reference_state: Tensor,
    *,
    lmax: int,
    cls: Tensor | tuple[Tensor, Tensor] | list[Tensor],
    mmax: int | None = None,
    beam_transfer: Tensor | tuple[Tensor, Tensor] | list[Tensor] | None = None,
    min_variance: float = 1.0e-8,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    if state.ndim == 1:
        diagonal = harmonic_cl_diagonal(
            lmax=lmax,
            mmax=mmax,
            cls=torch.as_tensor(cls),
            beam_transfer=None
            if beam_transfer is None
            else torch.as_tensor(beam_transfer),
            min_variance=min_variance,
            dtype=state.real.dtype if state.is_complex() else state.dtype,
            device=state.device,
        )
        return build_state_diagonal_operator(
            state,
            diagonal,
            apply_noise=apply_noise,
        )
    if state.ndim == 2 and int(state.shape[0]) == 2:
        if isinstance(cls, (tuple, list)):
            if len(cls) != 2:
                raise ValueError(
                    "spin harmonic cls must be a single spectrum or a length-2 sequence"
                )
            cls_e, cls_b = cls
        else:
            cls_e = cls_b = cls
        if beam_transfer is None:
            beam_e = beam_b = None
        elif isinstance(beam_transfer, (tuple, list)):
            if len(beam_transfer) != 2:
                raise ValueError(
                    "spin harmonic beam_transfer must be a single spectrum or a length-2 sequence"
                )
            beam_e, beam_b = beam_transfer
        else:
            beam_e = beam_b = beam_transfer
        diag_e = harmonic_cl_diagonal(
            lmax=lmax,
            mmax=mmax,
            cls=torch.as_tensor(cls_e),
            beam_transfer=None if beam_e is None else torch.as_tensor(beam_e),
            min_variance=min_variance,
            dtype=state.real.dtype if state.is_complex() else state.dtype,
            device=state.device,
        )
        diag_b = harmonic_cl_diagonal(
            lmax=lmax,
            mmax=mmax,
            cls=torch.as_tensor(cls_b),
            beam_transfer=None if beam_b is None else torch.as_tensor(beam_b),
            min_variance=min_variance,
            dtype=state.real.dtype if state.is_complex() else state.dtype,
            device=state.device,
        )
        diagonal = torch.stack([diag_e, diag_b], dim=0)
        return build_state_diagonal_operator(
            state,
            diagonal,
            apply_noise=apply_noise,
        )
    raise ValueError(
        "reference_state for harmonic Cl prior must be shape (nalm,) or (2, nalm)"
    )


def build_spin_harmonic_kronecker_prior_operator(
    reference_state: Tensor,
    *,
    harmonic_factor: Tensor,
    eb_coupling: Tensor | None = None,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    if state.ndim != 2 or int(state.shape[0]) != 2:
        raise ValueError(
            "spin-harmonic Kronecker prior requires reference_state shape (2, nalm)"
        )
    harmonic_t = torch.as_tensor(harmonic_factor)
    if harmonic_t.ndim == 1:
        harmonic_t = torch.diag(harmonic_t)
    if harmonic_t.ndim != 2 or int(harmonic_t.shape[1]) != int(state.shape[1]):
        raise ValueError("harmonic_factor must have shape (out_nalm, nalm) or (nalm,)")
    if eb_coupling is None:
        eb_t = torch.eye(2, dtype=harmonic_t.dtype, device=harmonic_t.device)
    else:
        eb_t = torch.as_tensor(
            eb_coupling, dtype=harmonic_t.dtype, device=harmonic_t.device
        )
        if eb_t.ndim != 2 or tuple(int(v) for v in eb_t.shape) != (2, 2):
            raise ValueError("eb_coupling must have shape (2, 2)")
    return build_state_kronecker_operator(
        state,
        eb_t,
        harmonic_t,
        apply_noise=apply_noise,
    )


def teb_cls_precision_factors(
    *,
    lmax: int,
    cls_tt: Tensor,
    cls_ee: Tensor,
    cls_bb: Tensor,
    cls_te: Tensor | None = None,
    cls_tb: Tensor | None = None,
    cls_eb: Tensor | None = None,
    mmax: int | None = None,
    beam_transfer_t: Tensor | None = None,
    beam_transfer_pol: Tensor | None = None,
    min_variance: float = 1.0e-8,
    jitter: float = 1.0e-8,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | None = None,
) -> Tensor:
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if ll < 0:
        raise ValueError("lmax must be non-negative")
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    cl_tt = torch.as_tensor(cls_tt, dtype=torch.float64, device=device)
    cl_ee = torch.as_tensor(cls_ee, dtype=torch.float64, device=device)
    cl_bb = torch.as_tensor(cls_bb, dtype=torch.float64, device=device)
    for name, arr in (("cls_tt", cl_tt), ("cls_ee", cl_ee), ("cls_bb", cl_bb)):
        if arr.ndim != 1 or int(arr.numel()) < ll + 1:
            raise ValueError(f"{name} must be a 1D spectrum with length >= lmax+1")
    cl_te = (
        torch.zeros(ll + 1, dtype=torch.float64, device=device)
        if cls_te is None
        else torch.as_tensor(cls_te, dtype=torch.float64, device=device)
    )
    cl_tb = (
        torch.zeros(ll + 1, dtype=torch.float64, device=device)
        if cls_tb is None
        else torch.as_tensor(cls_tb, dtype=torch.float64, device=device)
    )
    cl_eb = (
        torch.zeros(ll + 1, dtype=torch.float64, device=device)
        if cls_eb is None
        else torch.as_tensor(cls_eb, dtype=torch.float64, device=device)
    )
    for name, arr in (("cls_te", cl_te), ("cls_tb", cl_tb), ("cls_eb", cl_eb)):
        if arr.ndim != 1 or int(arr.numel()) < ll + 1:
            raise ValueError(f"{name} must be a 1D spectrum with length >= lmax+1")
    bt = (
        torch.ones(ll + 1, dtype=torch.float64, device=device)
        if beam_transfer_t is None
        else torch.as_tensor(beam_transfer_t, dtype=torch.float64, device=device)
    )
    bp = (
        torch.ones(ll + 1, dtype=torch.float64, device=device)
        if beam_transfer_pol is None
        else torch.as_tensor(beam_transfer_pol, dtype=torch.float64, device=device)
    )
    if bt.ndim != 1 or int(bt.numel()) < ll + 1:
        raise ValueError("beam_transfer_t must be a 1D spectrum with length >= lmax+1")
    if bp.ndim != 1 or int(bp.numel()) < ll + 1:
        raise ValueError(
            "beam_transfer_pol must be a 1D spectrum with length >= lmax+1"
        )

    factors_l: list[Tensor] = []
    eye3 = torch.eye(3, dtype=torch.float64, device=device)
    for ell in range(ll + 1):
        cov = torch.tensor(
            [
                [
                    cl_tt[ell] * bt[ell] * bt[ell],
                    cl_te[ell] * bt[ell] * bp[ell],
                    cl_tb[ell] * bt[ell] * bp[ell],
                ],
                [
                    cl_te[ell] * bt[ell] * bp[ell],
                    cl_ee[ell] * bp[ell] * bp[ell],
                    cl_eb[ell] * bp[ell] * bp[ell],
                ],
                [
                    cl_tb[ell] * bt[ell] * bp[ell],
                    cl_eb[ell] * bp[ell] * bp[ell],
                    cl_bb[ell] * bp[ell] * bp[ell],
                ],
            ],
            dtype=torch.float64,
            device=device,
        )
        cov = cov + float(min_variance) * eye3
        precision = torch.linalg.inv(cov + float(jitter) * eye3)
        factors_l.append(torch.linalg.cholesky(precision))
    factors_l_t = torch.stack(factors_l, dim=0)  # (lmax+1, 3, 3)
    ell = _alm_ell_array(ll, mm).to(device=device)
    return factors_l_t.index_select(0, ell).to(dtype=dtype)


def build_teb_cls_prior_operator(
    reference_state: Tensor,
    *,
    lmax: int,
    cls_tt: Tensor,
    cls_ee: Tensor,
    cls_bb: Tensor,
    cls_te: Tensor | None = None,
    cls_tb: Tensor | None = None,
    cls_eb: Tensor | None = None,
    mmax: int | None = None,
    beam_transfer_t: Tensor | None = None,
    beam_transfer_pol: Tensor | None = None,
    min_variance: float = 1.0e-8,
    jitter: float = 1.0e-8,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    if state.ndim != 2 or int(state.shape[0]) != 3:
        raise ValueError("TEB Cl prior requires reference_state shape (3, nalm)")
    factors = teb_cls_precision_factors(
        lmax=lmax,
        mmax=mmax,
        cls_tt=cls_tt,
        cls_ee=cls_ee,
        cls_bb=cls_bb,
        cls_te=cls_te,
        cls_tb=cls_tb,
        cls_eb=cls_eb,
        beam_transfer_t=beam_transfer_t,
        beam_transfer_pol=beam_transfer_pol,
        min_variance=min_variance,
        jitter=jitter,
        dtype=state.dtype if state.is_complex() else torch.complex64,
        device=state.device,
    )
    target = TargetRenderSpec(
        header={
            "NAXIS": 2,
            "NAXIS1": int(state.shape[1]),
            "NAXIS2": 3,
        },
        shape=(3, int(state.shape[1])),
    )

    def _apply_operator(
        scene: SceneRepresentation, target_spec: TargetRenderSpec
    ) -> Tensor:
        del target_spec
        tensor = _scene_state_tensor(scene)
        if tuple(int(v) for v in tensor.shape) != tuple(int(v) for v in state.shape):
            raise ValueError("TEB Cl prior received a scene with incompatible shape")
        tensor_t = tensor.to(dtype=factors.dtype, device=factors.device)
        mapped = torch.einsum("nij,jn->in", factors, tensor_t)
        return mapped.to(dtype=tensor.dtype, device=tensor.device)

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        mapped = torch.as_tensor(tensor)
        if tuple(int(v) for v in mapped.shape) != tuple(int(v) for v in state.shape):
            raise ValueError(
                "TEB Cl prior adjoint received data with incompatible shape"
            )
        mapped_t = mapped.to(dtype=factors.dtype, device=factors.device)
        adj = torch.einsum("nji,in->jn", torch.conj(factors), mapped_t)
        return adj.to(dtype=state.dtype, device=state.device)

    return BoundObservationOperator(
        operators=ObservationOperatorBundle(),
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=apply_noise,
    )


@dataclass
class SolverHooks:
    """Internal assembly surface for future GLS / MAP solvers."""

    operators: ObservationOperatorBundle
    apply_operator: ApplyOperatorFn | None = None
    apply_adjoint: AdjointOperatorFn | None = None
    apply_noise: ApplyNoiseFn | None = None
    bind_target_fn: BindTargetFn | None = None

    def render_scene(
        self, scene: SceneRepresentation, target: TargetRenderSpec
    ) -> Tensor:
        return self.bind_target(target).apply(scene)

    def apply_noise_model(self, tensor: Tensor) -> Tensor:
        fn = self.apply_noise or identity_noise
        return fn(tensor)

    def apply_adjoint_model(self, tensor: Tensor) -> Tensor:
        if self.apply_adjoint is None:
            raise NotImplementedError("adjoint operator hook is not configured")
        return self.apply_adjoint(tensor)

    def normal_equation_probe(
        self,
        scene: SceneRepresentation,
        target: TargetRenderSpec,
        data: Tensor,
    ) -> NormalEquationProbe:
        """Assemble A^T N^-1 A and A^T N^-1 y probes without solving."""
        return self.bind_target(target).normal_equation_probe(scene, data)

    def bind_target(self, target: TargetRenderSpec) -> BoundObservationOperator:
        apply_operator = self.apply_operator
        apply_adjoint = self.apply_adjoint
        apply_noise = self.apply_noise
        if self.bind_target_fn is not None:
            apply_operator, apply_adjoint, apply_noise = self.bind_target_fn(
                self.operators,
                target,
            )
        return BoundObservationOperator(
            operators=self.operators,
            target=target,
            apply_operator=apply_operator,
            apply_adjoint=apply_adjoint,
            apply_noise=apply_noise,
        )

    def bind_targets(
        self,
        targets: Sequence[TargetRenderSpec],
    ) -> BoundObservationOperatorSet:
        return BoundObservationOperatorSet(
            tuple(self.bind_target(target) for target in targets)
        )


def build_solver_hooks(
    operators: ObservationOperatorBundle,
    *,
    apply_operator: ApplyOperatorFn | None = None,
    apply_adjoint: AdjointOperatorFn | None = None,
    apply_noise: ApplyNoiseFn | None = None,
    bind_target_fn: BindTargetFn | None = None,
) -> SolverHooks:
    return SolverHooks(
        operators=operators,
        apply_operator=apply_operator,
        apply_adjoint=apply_adjoint,
        apply_noise=apply_noise,
        bind_target_fn=bind_target_fn,
    )


def build_spherical_healpix_solver_hooks(
    *,
    nside: int,
    nest: bool = False,
    mode: str = "pixel_bilinear",
    lmax: int | None = None,
    target_nside: int | None = None,
    fwhm_rad: float | None = None,
    beam_window: Tensor | None = None,
    iter: int = 0,
    apply_noise: ApplyNoiseFn | None = None,
) -> SolverHooks:
    """Build native sphere↔image solver hooks for HEALPix coefficient maps.

    `pixel_bilinear` is the first exact inverse-capable mode because its forward
    operator is the native HEALPix bilinear bridge and its adjoint is the
    corresponding scatter-add transpose.
    """

    if mode not in {"pixel_bilinear", "harmonic_direct", "healpix_bridge"}:
        raise ValueError(
            "mode must be 'pixel_bilinear', 'harmonic_direct', or 'healpix_bridge'"
        )
    operators = spherical_spectral_operator_bundle(mode=mode)

    def _bind_target(
        bound_operators: ObservationOperatorBundle,
        target: TargetRenderSpec,
    ) -> tuple[ApplyOperatorFn | None, AdjointOperatorFn | None, ApplyNoiseFn | None]:
        def _coerce_healpix_scene(
            scene: SceneRepresentation | nn.Module | Tensor,
        ) -> HealpixMapScene:
            scene_representation = as_scene_representation(scene)
            if isinstance(scene_representation, HealpixMapScene):
                return scene_representation
            if isinstance(scene_representation, DiscreteTargetImage):
                return HealpixMapScene(
                    scene_representation.data,
                    nside=int(nside),
                    nest=bool(nest),
                    lmax=lmax,
                    target_nside=target_nside,
                    fwhm_rad=fwhm_rad,
                    beam_window=beam_window,
                    iter=int(iter),
                    mode=mode,
                )
            if isinstance(scene, Tensor):
                return HealpixMapScene(
                    scene,
                    nside=int(nside),
                    nest=bool(nest),
                    lmax=lmax,
                    target_nside=target_nside,
                    fwhm_rad=fwhm_rad,
                    beam_window=beam_window,
                    iter=int(iter),
                    mode=mode,
                )
            raise TypeError(
                "spherical HEALPix solver hooks require a HealpixMapScene or tensor coefficients"
            )

        def _apply_operator(
            scene: SceneRepresentation, target_spec: TargetRenderSpec
        ) -> Tensor:
            healpix_scene = _coerce_healpix_scene(scene)
            return healpix_scene.render_to_target(
                target_spec,
                operators=bound_operators,
            )

        if mode == "pixel_bilinear":

            def _apply_adjoint(tensor: Tensor) -> Tensor:
                image = torch.as_tensor(tensor)
                if image.ndim == 2:
                    return accumulate_image_on_healpix_map_adjoint(
                        image,
                        nside=int(nside),
                        nest=bool(nest),
                        source_header=target.header,
                    )
                if image.ndim == 3:
                    return torch.stack(
                        [
                            accumulate_image_on_healpix_map_adjoint(
                                image[idx],
                                nside=int(nside),
                                nest=bool(nest),
                                source_header=target.header,
                            )
                            for idx in range(int(image.shape[0]))
                        ],
                        dim=0,
                    )
                raise ValueError("adjoint image must be 2D or band-first 3D")
        else:

            def _apply_adjoint(tensor: Tensor) -> Tensor:
                raise NotImplementedError(
                    "exact spherical adjoint is currently implemented for mode='pixel_bilinear' only"
                )

        return _apply_operator, _apply_adjoint, apply_noise

    return build_solver_hooks(
        operators,
        apply_noise=apply_noise,
        bind_target_fn=_bind_target,
    )


def build_spherical_harmonic_solver_hooks(
    *,
    lmax: int,
    mmax: int | None = None,
    apply_noise: ApplyNoiseFn | None = None,
) -> SolverHooks:
    """Build native image↔alm solver hooks for direct harmonic projection."""
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    operators = spherical_spectral_operator_bundle(mode="harmonic_direct")

    def _bind_target(
        bound_operators: ObservationOperatorBundle,
        target: TargetRenderSpec,
    ) -> tuple[ApplyOperatorFn | None, AdjointOperatorFn | None, ApplyNoiseFn | None]:
        def _coerce_alm_scene(
            scene: SceneRepresentation | nn.Module | Tensor,
        ) -> HarmonicAlmScene:
            scene_representation = as_scene_representation(scene)
            if isinstance(scene_representation, HarmonicAlmScene):
                return scene_representation
            if isinstance(scene_representation, DiscreteTargetImage):
                return HarmonicAlmScene(
                    scene_representation.data,
                    lmax=ll,
                    mmax=mm,
                )
            if isinstance(scene, Tensor):
                return HarmonicAlmScene(
                    scene,
                    lmax=ll,
                    mmax=mm,
                )
            raise TypeError(
                "spherical harmonic solver hooks require a HarmonicAlmScene or tensor alm coefficients"
            )

        def _apply_operator(
            scene: SceneRepresentation, target_spec: TargetRenderSpec
        ) -> Tensor:
            alm_scene = _coerce_alm_scene(scene)
            return alm_scene.render_to_target(
                target_spec,
                operators=bound_operators,
            )

        def _apply_adjoint(tensor: Tensor) -> Tensor:
            image = torch.as_tensor(tensor)
            if image.ndim == 2:
                return accumulate_image_on_alm_adjoint(
                    image,
                    source_header=target.header,
                    lmax=ll,
                    mmax=mm,
                )
            if image.ndim == 3:
                return torch.stack(
                    [
                        accumulate_image_on_alm_adjoint(
                            image[idx],
                            source_header=target.header,
                            lmax=ll,
                            mmax=mm,
                        )
                        for idx in range(int(image.shape[0]))
                    ],
                    dim=0,
                )
            raise ValueError("adjoint image must be 2D or band-first 3D")

        return _apply_operator, _apply_adjoint, apply_noise

    return build_solver_hooks(
        operators,
        apply_noise=apply_noise,
        bind_target_fn=_bind_target,
    )


def build_spherical_spin_harmonic_solver_hooks(
    *,
    spin: int,
    lmax: int,
    mmax: int | None = None,
    apply_noise: ApplyNoiseFn | None = None,
) -> SolverHooks:
    """Build native image↔(E,B) alm solver hooks for direct spin harmonic projection."""
    raw_spin = float(torch.as_tensor(spin).item())
    spin_i = int(raw_spin)
    if float(spin_i) != raw_spin or spin_i < 0:
        raise ValueError("spin must be a non-negative integer")
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    operators = spherical_spectral_operator_bundle(mode="harmonic_direct")

    def _bind_target(
        bound_operators: ObservationOperatorBundle,
        target: TargetRenderSpec,
    ) -> tuple[ApplyOperatorFn | None, AdjointOperatorFn | None, ApplyNoiseFn | None]:
        def _coerce_spin_scene(
            scene: SceneRepresentation | nn.Module | Tensor,
        ) -> SpinHarmonicAlmScene:
            scene_representation = as_scene_representation(scene)
            if isinstance(scene_representation, SpinHarmonicAlmScene):
                return scene_representation
            if isinstance(scene_representation, DiscreteTargetImage):
                data = torch.as_tensor(scene_representation.data)
                if data.ndim != 2 or int(data.shape[0]) != 2:
                    raise TypeError(
                        "spin harmonic solver hooks require a SpinHarmonicAlmScene or a (2, nalm) tensor"
                    )
                return SpinHarmonicAlmScene(
                    data[0],
                    data[1],
                    spin=spin_i,
                    lmax=ll,
                    mmax=mm,
                )
            if isinstance(scene, Tensor):
                data = torch.as_tensor(scene)
                if data.ndim != 2 or int(data.shape[0]) != 2:
                    raise TypeError(
                        "spin harmonic solver hooks require a SpinHarmonicAlmScene or a (2, nalm) tensor"
                    )
                return SpinHarmonicAlmScene(
                    data[0],
                    data[1],
                    spin=spin_i,
                    lmax=ll,
                    mmax=mm,
                )
            raise TypeError(
                "spin harmonic solver hooks require a SpinHarmonicAlmScene or a (2, nalm) tensor"
            )

        def _apply_operator(
            scene: SceneRepresentation, target_spec: TargetRenderSpec
        ) -> Tensor:
            spin_scene = _coerce_spin_scene(scene)
            return spin_scene.render_to_target(
                target_spec,
                operators=bound_operators,
            )

        def _apply_adjoint(tensor: Tensor) -> Tensor:
            maps = torch.as_tensor(tensor)
            if maps.ndim != 3 or int(maps.shape[0]) != 2:
                raise ValueError("adjoint maps must have shape (2, H, W)")
            alm_e, alm_b = accumulate_image_on_spin_alm_adjoint(
                maps,
                spin=spin_i,
                source_header=target.header,
                lmax=ll,
                mmax=mm,
            )
            return torch.stack([alm_e, alm_b], dim=0)

        return _apply_operator, _apply_adjoint, apply_noise

    return build_solver_hooks(
        operators,
        apply_noise=apply_noise,
        bind_target_fn=_bind_target,
    )
