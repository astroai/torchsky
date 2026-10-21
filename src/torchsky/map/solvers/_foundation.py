"""Reusable matrix-free linear-solver foundations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
import math
from typing import Generic, Protocol, TypeVar, cast, runtime_checkable

import torch
from torch import Tensor


type StateTree = (
    Tensor
    | tuple[StateTree, ...]
    | list[StateTree]
    | dict[str, StateTree]
)

StateT = TypeVar("StateT", bound=StateTree)


class TerminationReason(str, Enum):
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    BREAKDOWN = "breakdown"
    NONFINITE = "nonfinite"


class BreakdownReason(str, Enum):
    NON_POSITIVE_CURVATURE = "non_positive_curvature"
    NON_POSITIVE_PRECONDITIONER = "non_positive_preconditioner"
    NONFINITE_OPERATOR = "nonfinite_operator"
    NONFINITE_PRECONDITIONER = "nonfinite_preconditioner"
    NONFINITE_ITERATE = "nonfinite_iterate"
    BIDIAGONALIZATION = "bidiagonalization"


@dataclass(frozen=True)
class SolverBreakdown:
    reason: BreakdownReason
    iteration: int
    message: str
    scalar: float | None = None


@dataclass(frozen=True)
class SolverDiagnostics:
    termination: TerminationReason
    converged: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    relative_residual_norm: float
    stopping_threshold: float
    operator_applications: int
    adjoint_applications: int = 0
    preconditioner_applications: int = 0
    preconditioner_identity: str = "identity"
    projected: bool = False
    breakdown: SolverBreakdown | None = None
    normal_residual_norm: float | None = None


@dataclass(frozen=True)
class SolverResult(Generic[StateT]):
    solution: StateT
    residual_norm_history: tuple[float, ...] = ()
    diagnostics: SolverDiagnostics | None = None


LinearSolveResult = SolverResult[Tensor]


@runtime_checkable
class GaugeProjection(Protocol[StateT]):
    identity: str

    def project(self, state: StateT) -> StateT: ...


@runtime_checkable
class Preconditioner(Protocol[StateT]):
    identity: str

    def apply(self, residual: StateT) -> StateT: ...


@dataclass(frozen=True)
class IdentityGaugeProjection(Generic[StateT]):
    identity: str = "identity"

    def project(self, state: StateT) -> StateT:
        return state


@dataclass(frozen=True)
class IdentityPreconditioner(Generic[StateT]):
    identity: str = "identity"

    def apply(self, residual: StateT) -> StateT:
        return residual

    def __call__(self, residual: StateT) -> StateT:
        return self.apply(residual)


class TensorNullspaceProjection:
    """Orthogonally remove a declared tensor nullspace from solver states."""

    identity = "orthogonal_nullspace"

    def __init__(self, basis: Tensor, *, state_shape: tuple[int, ...] | None = None):
        basis_t = torch.as_tensor(basis)
        if not (torch.is_floating_point(basis_t) or torch.is_complex(basis_t)):
            raise TypeError("nullspace basis must have a floating-point or complex dtype")
        if basis_t.ndim < 1:
            raise ValueError("nullspace basis must have at least one dimension")
        if not bool(torch.isfinite(basis_t).all()):
            raise ValueError("nullspace basis must be finite")
        if state_shape is None:
            if basis_t.ndim != 2:
                raise ValueError("state_shape is required unless basis has shape (k, n)")
            state_shape = (int(basis_t.shape[1]),)
        if int(basis_t.shape[0]) == 0:
            raise ValueError("nullspace basis must contain at least one vector")
        if int(basis_t[0].numel()) != math.prod(state_shape):
            raise ValueError("nullspace basis vectors must match state_shape")
        rows = basis_t.reshape(int(basis_t.shape[0]), -1)
        gram = rows.conj() @ rows.mT
        if int(torch.linalg.matrix_rank(gram).item()) != int(rows.shape[0]):
            raise ValueError("nullspace basis vectors must be linearly independent")
        self._basis = rows
        self._gram = gram
        self._state_shape = state_shape

    def project(self, state: Tensor) -> Tensor:
        state_t = torch.as_tensor(state)
        if tuple(state_t.shape) != self._state_shape:
            raise ValueError("projected tensor shape must match the declared state shape")
        basis = self._basis.to(dtype=state_t.dtype, device=state_t.device)
        gram = self._gram.to(dtype=state_t.dtype, device=state_t.device)
        flat = state_t.reshape(-1)
        coefficients = torch.linalg.solve(gram, basis.conj() @ flat)
        return (flat - basis.mT @ coefficients).reshape_as(state_t)


class DiagonalPreconditioner:
    identity = "diagonal"

    def __init__(self, diagonal: Tensor):
        diagonal_t = torch.as_tensor(diagonal)
        if not (torch.is_floating_point(diagonal_t) or torch.is_complex(diagonal_t)):
            raise TypeError("preconditioner diagonal must have a floating-point or complex dtype")
        if not bool(torch.isfinite(diagonal_t).all()):
            raise ValueError("preconditioner diagonal must be finite")
        self.diagonal = diagonal_t
        self.reciprocal = torch.where(
            torch.abs(diagonal_t) > 0,
            torch.reciprocal(diagonal_t),
            torch.zeros_like(diagonal_t),
        )

    def apply(self, residual: Tensor) -> Tensor:
        residual_t = torch.as_tensor(residual)
        if tuple(residual_t.shape) != tuple(self.diagonal.shape):
            raise ValueError("preconditioner diagonal shape must match the residual")
        return self.reciprocal.to(
            dtype=residual_t.dtype, device=residual_t.device
        ) * residual_t

    def __call__(self, residual: Tensor) -> Tensor:
        return self.apply(residual)


def _tree_leaves(state: StateTree) -> tuple[Tensor, ...]:
    if isinstance(state, Tensor):
        return (state,)
    if isinstance(state, tuple | list):
        return tuple(leaf for item in state for leaf in _tree_leaves(item))
    if isinstance(state, Mapping):
        return tuple(leaf for item in state.values() for leaf in _tree_leaves(item))
    raise TypeError("solver states must be tensors or nested tuple/list/dict trees")


def _validate_tree(state: StateT, *, name: str, finite: bool = True) -> StateT:
    leaves = _tree_leaves(state)
    if not leaves:
        raise ValueError(f"{name} must contain at least one tensor")
    for leaf in leaves:
        if not (torch.is_floating_point(leaf) or torch.is_complex(leaf)):
            raise TypeError(f"{name} leaves must have floating-point or complex dtypes")
        if finite and not bool(torch.isfinite(leaf).all()):
            raise ValueError(f"{name} must be finite")
    if any(leaf.device != leaves[0].device for leaf in leaves[1:]):
        raise ValueError(f"{name} leaves must use one device")
    return state


def _tree_map(function: Callable[[Tensor], Tensor], state: StateT) -> StateT:
    if isinstance(state, Tensor):
        return cast(StateT, function(state))
    if isinstance(state, tuple):
        return cast(StateT, tuple(_tree_map(function, item) for item in state))
    if isinstance(state, list):
        return cast(StateT, [_tree_map(function, item) for item in state])
    if isinstance(state, dict):
        return cast(
            StateT,
            {key: _tree_map(function, item) for key, item in state.items()},
        )
    raise TypeError("solver states must be tensors or nested tuple/list/dict trees")


def _tree_zip_map(
    function: Callable[[Tensor, Tensor], Tensor], lhs: StateT, rhs: StateTree
) -> StateT:
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        if lhs.shape != rhs.shape:
            raise ValueError("solver state tensor shapes must match")
        if lhs.device != rhs.device:
            raise ValueError("solver state tensor devices must match")
        return cast(StateT, function(lhs, rhs))
    if isinstance(lhs, tuple) and isinstance(rhs, tuple) and len(lhs) == len(rhs):
        return cast(
            StateT,
            tuple(
                _tree_zip_map(function, left, right)
                for left, right in zip(lhs, rhs, strict=True)
            ),
        )
    if isinstance(lhs, list) and isinstance(rhs, list) and len(lhs) == len(rhs):
        return cast(
            StateT,
            [
                _tree_zip_map(function, left, right)
                for left, right in zip(lhs, rhs, strict=True)
            ],
        )
    if isinstance(lhs, dict) and isinstance(rhs, dict) and lhs.keys() == rhs.keys():
        return cast(
            StateT,
            {key: _tree_zip_map(function, lhs[key], rhs[key]) for key in lhs},
        )
    raise ValueError("solver state tree structures must match")


def tree_zeros_like(state: StateT) -> StateT:
    return _tree_map(torch.zeros_like, state)


def tree_clone(state: StateT) -> StateT:
    return _tree_map(torch.clone, state)


def tree_add(lhs: StateT, rhs: StateTree) -> StateT:
    return _tree_zip_map(torch.add, lhs, rhs)


def tree_sub(lhs: StateT, rhs: StateTree) -> StateT:
    return _tree_zip_map(torch.sub, lhs, rhs)


def tree_scale(state: StateT, scalar: Tensor | float) -> StateT:
    return _tree_map(lambda leaf: leaf * scalar, state)


def tree_axpy(state: StateT, scalar: Tensor | float, direction: StateTree) -> StateT:
    return _tree_zip_map(lambda left, right: left + scalar * right, state, direction)


def tree_inner(lhs: StateTree, rhs: StateTree) -> Tensor:
    products = _tree_leaves(
        _tree_zip_map(
            lambda left, right: torch.sum(torch.conj(left.reshape(-1)) * right.reshape(-1)),
            lhs,
            rhs,
        )
    )
    total = products[0]
    for product in products[1:]:
        total = total + product.to(dtype=total.dtype, device=total.device)
    return total


def tree_norm(state: StateTree) -> Tensor:
    value = tree_inner(state, state).real
    return torch.sqrt(torch.clamp_min(value, 0.0))


def tree_isfinite(state: StateTree) -> bool:
    return all(bool(torch.isfinite(leaf).all()) for leaf in _tree_leaves(state))


def _validate_output(reference: StateT, output: StateTree, *, name: str) -> StateT:
    def _validate(reference_leaf: Tensor, output_leaf: Tensor) -> Tensor:
        if output_leaf.dtype != reference_leaf.dtype:
            raise ValueError(f"{name} output dtypes must match the input")
        if output_leaf.device != reference_leaf.device:
            raise ValueError(f"{name} output devices must match the input")
        return output_leaf

    return _tree_zip_map(_validate, reference, output)


def _project(
    projection: GaugeProjection[StateT] | Callable[[StateT], StateT] | None,
    state: StateT,
) -> StateT:
    if projection is None:
        return state
    projected = (
        projection.project(state)
        if isinstance(projection, GaugeProjection)
        else projection(state)
    )
    return _validate_output(state, projected, name="projection")


def _apply_preconditioner(
    preconditioner: Preconditioner[StateT] | Callable[[StateT], StateT] | StateTree | None,
    residual: StateT,
) -> StateT:
    if preconditioner is None:
        return residual
    if isinstance(preconditioner, Preconditioner):
        result = preconditioner.apply(residual)
    elif callable(preconditioner):
        result = preconditioner(residual)
    else:
        result = _tree_zip_map(torch.mul, residual, preconditioner)
    return _validate_output(residual, result, name="preconditioner")


def _preconditioner_identity(preconditioner: object | None) -> str:
    if preconditioner is None:
        return "identity"
    identity = getattr(preconditioner, "identity", None)
    if isinstance(identity, str) and identity:
        return identity
    if callable(preconditioner):
        name = getattr(preconditioner, "__name__", None)
        return name if isinstance(name, str) else type(preconditioner).__name__
    return "multiplicative_tree"


def _scalar(value: Tensor) -> float:
    return float(value.detach().item())


def _relative(residual_norm: float, rhs_norm: float) -> float:
    if rhs_norm == 0.0:
        return residual_norm
    return residual_norm / rhs_norm


def _diagnostics(
    *,
    termination: TerminationReason,
    iterations: int,
    initial_residual_norm: float,
    residual_norm: float,
    threshold: float,
    rhs_norm: float,
    operator_applications: int,
    adjoint_applications: int,
    preconditioner_applications: int,
    preconditioner_identity: str,
    projected: bool,
    breakdown: SolverBreakdown | None = None,
    normal_residual_norm: float | None = None,
) -> SolverDiagnostics:
    return SolverDiagnostics(
        termination=termination,
        converged=termination is TerminationReason.CONVERGED,
        iterations=iterations,
        initial_residual_norm=initial_residual_norm,
        residual_norm=residual_norm,
        relative_residual_norm=_relative(residual_norm, rhs_norm),
        stopping_threshold=threshold,
        operator_applications=operator_applications,
        adjoint_applications=adjoint_applications,
        preconditioner_applications=preconditioner_applications,
        preconditioner_identity=preconditioner_identity,
        projected=projected,
        breakdown=breakdown,
        normal_residual_norm=normal_residual_norm,
    )


def conjugate_gradient(
    operator: Callable[[StateT], StateT],
    rhs: StateT,
    *,
    operator_is_hpd: bool,
    initial_solution: StateT | None = None,
    max_iterations: int = 100,
    rtol: float = 1.0e-5,
    atol: float = 0.0,
    preconditioner: Preconditioner[StateT] | Callable[[StateT], StateT] | StateTree | None = None,
    projection: GaugeProjection[StateT] | Callable[[StateT], StateT] | None = None,
    record_residual_history: bool = True,
) -> SolverResult[StateT]:
    """Solve a declared HPD system using projected preconditioned CG."""

    if operator_is_hpd is not True:
        raise ValueError("conjugate_gradient requires operator_is_hpd=True")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not isinstance(value, float | int) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    rhs_t = _validate_tree(rhs, name="rhs")
    solution = tree_zeros_like(rhs_t) if initial_solution is None else tree_clone(initial_solution)
    _tree_zip_map(lambda left, right: right.to(dtype=left.dtype), rhs_t, solution)
    solution = _project(projection, solution)
    rhs_projected = _project(projection, rhs_t)
    operator_applications = 0
    preconditioner_applications = 0

    def apply_operator(state: StateT) -> StateT:
        nonlocal operator_applications
        operator_applications += 1
        image = operator(_project(projection, state))
        _validate_output(rhs_t, image, name="operator")
        return _project(projection, image)

    image = apply_operator(solution)
    rhs_norm = _scalar(tree_norm(rhs_projected))
    threshold = max(float(atol), float(rtol) * rhs_norm)
    if not tree_isfinite(image):
        breakdown = SolverBreakdown(
            BreakdownReason.NONFINITE_OPERATOR,
            0,
            "operator produced non-finite values",
        )
        diagnostics = _diagnostics(
            termination=TerminationReason.NONFINITE,
            iterations=0,
            initial_residual_norm=math.inf,
            residual_norm=math.inf,
            threshold=threshold,
            rhs_norm=rhs_norm,
            operator_applications=operator_applications,
            adjoint_applications=0,
            preconditioner_applications=0,
            preconditioner_identity=_preconditioner_identity(preconditioner),
            projected=projection is not None,
            breakdown=breakdown,
        )
        return SolverResult(solution, (), diagnostics)
    residual = tree_sub(rhs_projected, image)
    residual_norm = _scalar(tree_norm(residual))
    initial_residual_norm = residual_norm
    history = [residual_norm] if record_residual_history else []
    if residual_norm <= threshold:
        diagnostics = _diagnostics(
            termination=TerminationReason.CONVERGED,
            iterations=0,
            initial_residual_norm=initial_residual_norm,
            residual_norm=residual_norm,
            threshold=threshold,
            rhs_norm=rhs_norm,
            operator_applications=operator_applications,
            adjoint_applications=0,
            preconditioner_applications=0,
            preconditioner_identity=_preconditioner_identity(preconditioner),
            projected=projection is not None,
        )
        return SolverResult(solution, tuple(history), diagnostics)
    preconditioned = _apply_preconditioner(preconditioner, residual)
    preconditioner_applications += 1
    if not tree_isfinite(preconditioned):
        breakdown = SolverBreakdown(
            BreakdownReason.NONFINITE_PRECONDITIONER,
            0,
            "preconditioner produced non-finite values",
        )
        diagnostics = _diagnostics(
            termination=TerminationReason.NONFINITE,
            iterations=0,
            initial_residual_norm=initial_residual_norm,
            residual_norm=residual_norm,
            threshold=threshold,
            rhs_norm=rhs_norm,
            operator_applications=operator_applications,
            adjoint_applications=0,
            preconditioner_applications=preconditioner_applications,
            preconditioner_identity=_preconditioner_identity(preconditioner),
            projected=projection is not None,
            breakdown=breakdown,
        )
        return SolverResult(solution, tuple(history), diagnostics)
    preconditioned = _project(projection, preconditioned)
    rz = tree_inner(residual, preconditioned)
    if not bool(torch.isfinite(rz)) or _scalar(rz.real) <= 0.0:
        breakdown = SolverBreakdown(
            BreakdownReason.NON_POSITIVE_PRECONDITIONER,
            0,
            "preconditioner is not positive definite on the residual",
            _scalar(rz.real) if bool(torch.isfinite(rz)) else None,
        )
        diagnostics = _diagnostics(
            termination=TerminationReason.BREAKDOWN,
            iterations=0,
            initial_residual_norm=initial_residual_norm,
            residual_norm=residual_norm,
            threshold=threshold,
            rhs_norm=rhs_norm,
            operator_applications=operator_applications,
            adjoint_applications=0,
            preconditioner_applications=preconditioner_applications,
            preconditioner_identity=_preconditioner_identity(preconditioner),
            projected=projection is not None,
            breakdown=breakdown,
        )
        return SolverResult(solution, tuple(history), diagnostics)
    direction = tree_clone(preconditioned)
    completed = 0
    breakdown: SolverBreakdown | None = None
    termination = TerminationReason.MAX_ITERATIONS
    for iteration in range(1, max_iterations + 1):
        image_direction = apply_operator(direction)
        if not tree_isfinite(image_direction):
            breakdown = SolverBreakdown(
                BreakdownReason.NONFINITE_OPERATOR,
                iteration,
                "operator produced non-finite values",
            )
            termination = TerminationReason.NONFINITE
            break
        curvature = tree_inner(direction, image_direction)
        curvature_real = _scalar(curvature.real) if bool(torch.isfinite(curvature)) else math.nan
        if not bool(torch.isfinite(curvature)) or curvature_real <= 0.0:
            breakdown = SolverBreakdown(
                BreakdownReason.NON_POSITIVE_CURVATURE,
                iteration,
                "declared HPD operator has non-positive search curvature",
                curvature_real if math.isfinite(curvature_real) else None,
            )
            termination = TerminationReason.BREAKDOWN
            break
        alpha = rz.real / curvature.real
        solution = _project(projection, tree_axpy(solution, alpha, direction))
        if not tree_isfinite(solution):
            breakdown = SolverBreakdown(
                BreakdownReason.NONFINITE_ITERATE,
                iteration,
                "CG iterate became non-finite",
            )
            termination = TerminationReason.NONFINITE
            break
        residual = tree_sub(rhs_projected, apply_operator(solution))
        residual_norm = _scalar(tree_norm(residual))
        completed = iteration
        if record_residual_history:
            history.append(residual_norm)
        if not math.isfinite(residual_norm):
            breakdown = SolverBreakdown(
                BreakdownReason.NONFINITE_ITERATE,
                iteration,
                "actual residual became non-finite",
            )
            termination = TerminationReason.NONFINITE
            break
        if residual_norm <= threshold:
            termination = TerminationReason.CONVERGED
            break
        preconditioned = _apply_preconditioner(preconditioner, residual)
        preconditioner_applications += 1
        if not tree_isfinite(preconditioned):
            breakdown = SolverBreakdown(
                BreakdownReason.NONFINITE_PRECONDITIONER,
                iteration,
                "preconditioner produced non-finite values",
            )
            termination = TerminationReason.NONFINITE
            break
        preconditioned = _project(projection, preconditioned)
        rz_new = tree_inner(residual, preconditioned)
        rz_new_real = _scalar(rz_new.real) if bool(torch.isfinite(rz_new)) else math.nan
        if not bool(torch.isfinite(rz_new)) or rz_new_real <= 0.0:
            breakdown = SolverBreakdown(
                BreakdownReason.NON_POSITIVE_PRECONDITIONER,
                iteration,
                "preconditioner is not positive definite on the residual",
                rz_new_real if math.isfinite(rz_new_real) else None,
            )
            termination = TerminationReason.BREAKDOWN
            break
        beta = rz_new.real / rz.real
        direction = _project(
            projection, tree_add(preconditioned, tree_scale(direction, beta))
        )
        rz = rz_new
    diagnostics = _diagnostics(
        termination=termination,
        iterations=completed,
        initial_residual_norm=initial_residual_norm,
        residual_norm=residual_norm,
        threshold=threshold,
        rhs_norm=rhs_norm,
        operator_applications=operator_applications,
        adjoint_applications=0,
        preconditioner_applications=preconditioner_applications,
        preconditioner_identity=_preconditioner_identity(preconditioner),
        projected=projection is not None,
        breakdown=breakdown,
    )
    return SolverResult(solution, tuple(history), diagnostics)


def lsqr(
    operator: Callable[[StateT], StateTree],
    adjoint: Callable[[StateTree], StateT],
    rhs: StateTree,
    *,
    initial_solution: StateT | None = None,
    max_iterations: int = 100,
    rtol: float = 1.0e-5,
    atol: float = 0.0,
    damping: float = 0.0,
    projection: GaugeProjection[StateT] | Callable[[StateT], StateT] | None = None,
    record_residual_history: bool = True,
) -> SolverResult[StateT]:
    """Solve a rectangular least-squares problem with Golub-Kahan LSQR."""

    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    for name, value in (("rtol", rtol), ("atol", atol), ("damping", damping)):
        if not isinstance(value, float | int) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    rhs_t = _validate_tree(rhs, name="rhs")
    operator_applications = 0
    adjoint_applications = 0

    def apply_adjoint(data: StateTree) -> StateT:
        nonlocal adjoint_applications
        adjoint_applications += 1
        result = adjoint(data)
        return _project(projection, result)

    if initial_solution is None:
        prototype = apply_adjoint(rhs_t)
        _validate_tree(prototype, name="adjoint(rhs)")
        solution = tree_zeros_like(prototype)
    else:
        solution = _project(
            projection, _validate_tree(tree_clone(initial_solution), name="initial_solution")
        )

    def apply_operator(state: StateT) -> StateTree:
        nonlocal operator_applications
        operator_applications += 1
        return operator(_project(projection, state))

    sqrt_damping = math.sqrt(float(damping))

    def augmented_operator(state: StateT) -> StateTree:
        image = apply_operator(state)
        _validate_output(rhs_t, image, name="operator")
        if damping == 0.0:
            return image
        return (image, tree_scale(state, sqrt_damping))

    def augmented_adjoint(data: StateTree) -> StateT:
        if damping == 0.0:
            return apply_adjoint(data)
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError("internal augmented LSQR range must contain two blocks")
        return tree_add(apply_adjoint(data[0]), tree_scale(data[1], sqrt_damping))

    augmented_rhs: StateTree = (
        rhs_t if damping == 0.0 else (rhs_t, tree_zeros_like(solution))
    )
    augmented_residual = tree_sub(augmented_rhs, augmented_operator(solution))
    if not tree_isfinite(augmented_residual):
        breakdown = SolverBreakdown(
            BreakdownReason.NONFINITE_OPERATOR,
            0,
            "operator produced non-finite values",
        )
        diagnostics = _diagnostics(
            termination=TerminationReason.NONFINITE,
            iterations=0,
            initial_residual_norm=math.inf,
            residual_norm=math.inf,
            threshold=math.inf,
            rhs_norm=_scalar(tree_norm(rhs_t)),
            operator_applications=operator_applications,
            adjoint_applications=adjoint_applications,
            preconditioner_applications=0,
            preconditioner_identity="identity",
            projected=projection is not None,
            breakdown=breakdown,
            normal_residual_norm=math.inf,
        )
        return SolverResult(solution, (), diagnostics)
    actual_residual = tree_sub(rhs_t, apply_operator(solution))
    residual_norm = _scalar(tree_norm(actual_residual))
    initial_residual_norm = residual_norm
    rhs_norm = _scalar(tree_norm(rhs_t))
    residual_threshold = max(float(atol), float(rtol) * rhs_norm)
    normal_residual = augmented_adjoint(augmented_residual)
    normal_residual_norm = _scalar(tree_norm(normal_residual))
    initial_normal_norm = normal_residual_norm
    normal_threshold = max(float(atol), float(rtol) * initial_normal_norm)
    history = [residual_norm] if record_residual_history else []
    if residual_norm <= residual_threshold or normal_residual_norm <= normal_threshold:
        diagnostics = _diagnostics(
            termination=TerminationReason.CONVERGED,
            iterations=0,
            initial_residual_norm=initial_residual_norm,
            residual_norm=residual_norm,
            threshold=residual_threshold,
            rhs_norm=rhs_norm,
            operator_applications=operator_applications,
            adjoint_applications=adjoint_applications,
            preconditioner_applications=0,
            preconditioner_identity="identity",
            projected=projection is not None,
            normal_residual_norm=normal_residual_norm,
        )
        return SolverResult(solution, tuple(history), diagnostics)
    beta = tree_norm(augmented_residual)
    if _scalar(beta) == 0.0:
        breakdown = SolverBreakdown(
            BreakdownReason.BIDIAGONALIZATION,
            0,
            "LSQR initial residual has zero norm without satisfying convergence",
        )
        diagnostics = _diagnostics(
            termination=TerminationReason.BREAKDOWN,
            iterations=0,
            initial_residual_norm=initial_residual_norm,
            residual_norm=residual_norm,
            threshold=residual_threshold,
            rhs_norm=rhs_norm,
            operator_applications=operator_applications,
            adjoint_applications=adjoint_applications,
            preconditioner_applications=0,
            preconditioner_identity="identity",
            projected=projection is not None,
            breakdown=breakdown,
            normal_residual_norm=normal_residual_norm,
        )
        return SolverResult(solution, tuple(history), diagnostics)
    u = tree_scale(augmented_residual, torch.reciprocal(beta))
    v_unscaled = augmented_adjoint(u)
    alpha = tree_norm(v_unscaled)
    if _scalar(alpha) == 0.0:
        breakdown = SolverBreakdown(
            BreakdownReason.BIDIAGONALIZATION,
            0,
            "LSQR adjoint bidiagonalization vector has zero norm",
        )
        diagnostics = _diagnostics(
            termination=TerminationReason.BREAKDOWN,
            iterations=0,
            initial_residual_norm=initial_residual_norm,
            residual_norm=residual_norm,
            threshold=residual_threshold,
            rhs_norm=rhs_norm,
            operator_applications=operator_applications,
            adjoint_applications=adjoint_applications,
            preconditioner_applications=0,
            preconditioner_identity="identity",
            projected=projection is not None,
            breakdown=breakdown,
            normal_residual_norm=normal_residual_norm,
        )
        return SolverResult(solution, tuple(history), diagnostics)
    v = tree_scale(v_unscaled, torch.reciprocal(alpha))
    w = tree_clone(v)
    phi_bar = beta
    rho_bar = alpha
    completed = 0
    breakdown: SolverBreakdown | None = None
    termination = TerminationReason.MAX_ITERATIONS
    for iteration in range(1, max_iterations + 1):
        u_unscaled = tree_sub(augmented_operator(v), tree_scale(u, alpha))
        beta = tree_norm(u_unscaled)
        if not bool(torch.isfinite(beta)):
            breakdown = SolverBreakdown(
                BreakdownReason.NONFINITE_OPERATOR,
                iteration,
                "LSQR forward bidiagonalization became non-finite",
            )
            termination = TerminationReason.NONFINITE
            break
        u = tree_zeros_like(u_unscaled) if _scalar(beta) == 0.0 else tree_scale(u_unscaled, torch.reciprocal(beta))
        v_unscaled = tree_sub(augmented_adjoint(u), tree_scale(v, beta))
        alpha = tree_norm(v_unscaled)
        if not bool(torch.isfinite(alpha)):
            breakdown = SolverBreakdown(
                BreakdownReason.NONFINITE_OPERATOR,
                iteration,
                "LSQR adjoint bidiagonalization became non-finite",
            )
            termination = TerminationReason.NONFINITE
            break
        v = tree_zeros_like(v_unscaled) if _scalar(alpha) == 0.0 else tree_scale(v_unscaled, torch.reciprocal(alpha))
        rho = torch.sqrt(rho_bar.square() + beta.square())
        if _scalar(rho) == 0.0 or not bool(torch.isfinite(rho)):
            breakdown = SolverBreakdown(
                BreakdownReason.BIDIAGONALIZATION,
                iteration,
                "LSQR plane rotation has zero or non-finite norm",
            )
            termination = TerminationReason.BREAKDOWN
            break
        cosine = rho_bar / rho
        sine = beta / rho
        theta = sine * alpha
        rho_bar = -cosine * alpha
        phi = cosine * phi_bar
        phi_bar = sine * phi_bar
        solution = _project(projection, tree_axpy(solution, phi / rho, w))
        w = tree_sub(v, tree_scale(w, theta / rho))
        if not tree_isfinite(solution):
            breakdown = SolverBreakdown(
                BreakdownReason.NONFINITE_ITERATE,
                iteration,
                "LSQR iterate became non-finite",
            )
            termination = TerminationReason.NONFINITE
            break
        actual_residual = tree_sub(rhs_t, apply_operator(solution))
        residual_norm = _scalar(tree_norm(actual_residual))
        augmented_residual = tree_sub(augmented_rhs, augmented_operator(solution))
        normal_residual = augmented_adjoint(augmented_residual)
        normal_residual_norm = _scalar(tree_norm(normal_residual))
        completed = iteration
        if record_residual_history:
            history.append(residual_norm)
        if not math.isfinite(residual_norm) or not math.isfinite(normal_residual_norm):
            breakdown = SolverBreakdown(
                BreakdownReason.NONFINITE_ITERATE,
                iteration,
                "LSQR actual residual became non-finite",
            )
            termination = TerminationReason.NONFINITE
            break
        if residual_norm <= residual_threshold or normal_residual_norm <= normal_threshold:
            termination = TerminationReason.CONVERGED
            break
        if _scalar(alpha) == 0.0 and _scalar(beta) == 0.0:
            breakdown = SolverBreakdown(
                BreakdownReason.BIDIAGONALIZATION,
                iteration,
                "LSQR bidiagonalization terminated before the stopping criterion",
            )
            termination = TerminationReason.BREAKDOWN
            break
    diagnostics = _diagnostics(
        termination=termination,
        iterations=completed,
        initial_residual_norm=initial_residual_norm,
        residual_norm=residual_norm,
        threshold=residual_threshold,
        rhs_norm=rhs_norm,
        operator_applications=operator_applications,
        adjoint_applications=adjoint_applications,
        preconditioner_applications=0,
        preconditioner_identity="identity",
        projected=projection is not None,
        breakdown=breakdown,
        normal_residual_norm=normal_residual_norm,
    )
    return SolverResult(solution, tuple(history), diagnostics)


__all__ = [
    "BreakdownReason",
    "DiagonalPreconditioner",
    "GaugeProjection",
    "IdentityGaugeProjection",
    "IdentityPreconditioner",
    "LinearSolveResult",
    "Preconditioner",
    "SolverBreakdown",
    "SolverDiagnostics",
    "SolverResult",
    "StateTree",
    "TensorNullspaceProjection",
    "TerminationReason",
    "conjugate_gradient",
    "lsqr",
    "tree_add",
    "tree_axpy",
    "tree_clone",
    "tree_inner",
    "tree_isfinite",
    "tree_norm",
    "tree_scale",
    "tree_sub",
    "tree_zeros_like",
]
