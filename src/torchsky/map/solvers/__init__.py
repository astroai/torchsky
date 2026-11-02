"""Solver hooks for field fitting and discrete inverse mapmaking.

The implementation is grouped into ``_core`` (operators, probes,
preconditioners), ``_state`` (state-layout operators and component
decomposition), ``_priors`` (harmonic/TEB prior operators), and ``_hooks``
(survey hooks). The historical :mod:`torchsky.map.solvers` names are re-exported
here.
"""

# ruff: noqa: F401
# Imported names outside ``__all__`` stay importable from this module; only
# ``__all__`` is the documented experimental surface.

from __future__ import annotations

from ._types import (
    AdjointOperatorFn,
    ApplyNoiseFn,
    ApplyOperatorFn,
    BindTargetFn,
    NormalOperatorFn,
    PreconditionerFn,
)
from ._foundation import (
    BreakdownReason,
    DiagonalPreconditioner,
    GaugeProjection,
    IdentityGaugeProjection,
    IdentityPreconditioner,
    Preconditioner,
    SolverBreakdown,
    SolverDiagnostics,
    SolverResult,
    StateTree,
    TensorNullspaceProjection,
    TerminationReason,
    conjugate_gradient,
    lsqr,
    tree_add,
    tree_axpy,
    tree_clone,
    tree_inner,
    tree_isfinite,
    tree_norm,
    tree_scale,
    tree_sub,
    tree_zeros_like,
)
from ._implicit import ImplicitDifferentiationPolicy, SolverDifferentiationMode
from ._core import (
    _relative_norm,
    _subspace_leakage,
    default_apply_operator,
    identity_noise,
    NormalEquationProbe,
    LinearSolveResult,
    MultiObservationNormalEquationProbe,
    _normalized_probe,
    _solver_complex_dtype,
    _rayleigh_quotient,
    _inner_product,
    make_diagonal_preconditioner,
    _apply_preconditioner,
    _solve_normal_system_cg,
    _estimate_inverse_diagonal_hutchinson,
    BoundObservationOperator,
    BoundObservationOperatorSet,
    compose_bound_operators,
)

from ._state import (
    _scene_state_tensor,
    _flatten_state,
    _reshape_state,
    _canonical_subspace_indices,
    build_state_identity_operator,
    build_state_subspace_operator,
    build_state_component_mixture_operator,
    SurveyInstrumentComponentMixer,
    ComponentDecompositionDiagnostics,
    HarmonicComponentDecomposition,
    build_conditioned_component_mixture_operator,
    build_state_diagonal_operator,
    build_state_matrix_operator,
    build_state_low_rank_operator,
    build_state_kronecker_operator,
)

from ._priors import (
    harmonic_degree_diagonal,
    build_harmonic_degree_prior_operator,
    harmonic_dense_covariance,
    harmonic_cl_diagonal,
    build_harmonic_dense_prior_operator,
    build_harmonic_cl_prior_operator,
    build_spin_harmonic_kronecker_prior_operator,
    teb_cls_precision_factors,
    build_teb_cls_prior_operator,
)

from ._hooks import (
    SolverHooks,
    build_solver_hooks,
    build_spherical_healpix_solver_hooks,
    build_spherical_harmonic_solver_hooks,
    build_spherical_spin_harmonic_solver_hooks,
)

__all__ = [
    "BoundObservationOperator",
    "BoundObservationOperatorSet",
    "SolverHooks",
    "SolverResult",
    "SolverDiagnostics",
    "build_harmonic_cl_prior_operator",
    "build_harmonic_degree_prior_operator",
    "build_harmonic_dense_prior_operator",
    "build_spherical_harmonic_solver_hooks",
    "build_spherical_spin_harmonic_solver_hooks",
    "build_state_identity_operator",
    "build_state_subspace_operator",
    "build_teb_cls_prior_operator",
    "compose_bound_operators",
    "conjugate_gradient",
    "harmonic_dense_covariance",
    "lsqr",
    "make_diagonal_preconditioner",
]
