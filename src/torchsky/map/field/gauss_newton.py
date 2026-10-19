"""Private experimental Gauss--Newton refinement of neural scene parameters."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from ..solvers import _solve_normal_system_cg, make_diagonal_preconditioner


def gauss_newton_pcg_step(
    residual_functions: Sequence[Callable[[Tensor], Tensor]],
    parameters: Tensor,
    *,
    cg_iterations: int = 16,
    damping: float = 1.0e-3,
    preconditioner_probes: int = 2,
    seed: int = 101,
    backtracking: int = 8,
) -> tuple[Tensor, dict[str, Any]]:
    """Reduce sum-of-squared whitened residuals with a damped matrix-free step.

    Damping is relative to the median positive Hutchinson Fisher diagonal.
    Residual blocks must be deterministic; calibration and noise stay fixed.
    CPU/CUDA use the existing CG solver's FP64 inner products.
    """
    if parameters.ndim != 1 or not parameters.is_floating_point():
        raise ValueError("parameters must be a flat floating-point tensor")
    if parameters.numel() == 0 or not bool(torch.isfinite(parameters).all()):
        raise ValueError("parameters must be nonempty and finite")
    if parameters.device.type not in {"cpu", "cuda"}:
        raise ValueError("neural PCG currently supports CPU and CUDA")
    if not residual_functions or any(not callable(fn) for fn in residual_functions):
        raise ValueError("at least one residual function is required")
    for name, value in (
        ("cg_iterations", cg_iterations),
        ("preconditioner_probes", preconditioner_probes),
        ("backtracking", backtracking),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and non-negative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    state = parameters.detach().clone()
    rhs = torch.zeros_like(state)
    diagonal = torch.zeros_like(state)
    initial_loss = 0.0
    generator = torch.Generator(device=state.device).manual_seed(seed)
    # ponytail: retain one residual-block graph (one exposure in the benchmark).
    # Upgrade to PSF-halo tile blocks when full-exposure activations exceed memory.
    for residual_function in residual_functions:
        residual, pullback = torch.func.vjp(residual_function, state)[:2]
        if (
            residual.ndim != 1
            or residual.numel() == 0
            or not residual.is_floating_point()
            or residual.device != state.device
        ):
            raise ValueError(
                "residuals must be nonempty flat real tensors on the parameter device"
            )
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("initial residuals must be finite")
        initial_loss += float(residual.detach().double().square().sum())
        rhs -= pullback(residual)[0].detach()
        for _ in range(preconditioner_probes):
            signs = (
                torch.randint(
                    0, 2, residual.shape, device=state.device, generator=generator
                ).to(dtype=residual.dtype)
                * 2
                - 1
            )
            diagonal += pullback(signs)[0].detach().square() / preconditioner_probes
        del pullback, residual
    if not bool(torch.isfinite(rhs).all() & torch.isfinite(diagonal).all()):
        raise FloatingPointError("neural linearization produced nonfinite values")
    positive_diagonal = diagonal[diagonal > 0]
    reference = float(positive_diagonal.median()) if positive_diagonal.numel() else 0.0
    effective_damping = float(damping) * reference
    if not math.isfinite(initial_loss) or not math.isfinite(effective_damping):
        raise FloatingPointError("neural objective or damping is nonfinite")

    def normal_action(direction: Tensor) -> Tensor:
        action = torch.zeros_like(state)
        for residual_function in residual_functions:
            _, tangent = torch.func.jvp(residual_function, (state,), (direction,))[:2]
            _, pullback = torch.func.vjp(residual_function, state)[:2]
            action += pullback(tangent)[0].detach()
            del pullback, tangent, _
        if not bool(torch.isfinite(action).all()):
            raise FloatingPointError("neural normal action produced nonfinite values")
        return action

    solved = _solve_normal_system_cg(
        normal_action,
        rhs,
        iterations=cg_iterations,
        damping=effective_damping,
        preconditioner=make_diagonal_preconditioner(diagonal + effective_damping),
    )
    step = solved.solution.detach()
    if not bool(torch.isfinite(step).all()):
        raise FloatingPointError("neural CG produced a nonfinite step")
    cg_residual = normal_action(step) + effective_damping * step - rhs
    updated = state
    final_loss = initial_loss
    accepted_step_scale = 0.0
    with torch.no_grad():
        for attempt in range(backtracking):
            scale = 0.5**attempt
            candidate = state + scale * step
            candidate_loss = 0.0
            for residual_function in residual_functions:
                residual = residual_function(candidate)
                candidate_loss += float(residual.double().square().sum())
            if math.isfinite(candidate_loss) and candidate_loss < initial_loss:
                updated = candidate
                final_loss = candidate_loss
                accepted_step_scale = scale
                break
    return updated, {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "objective": "sum_squared_whitened_residuals",
        "effective_damping": effective_damping,
        "damping_reference": "median_positive_hutchinson_fisher_diagonal",
        "accepted_step_scale": accepted_step_scale,
        "cg_residual_norm": float(torch.linalg.vector_norm(cg_residual.double())),
        "cg_residual_norm_history": list(solved.residual_norm_history),
    }
