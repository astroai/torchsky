#!/usr/bin/env python
"""Native shared-vs-survey-private harmonic decomposition example.

This models one latent spherical harmonic state with three components:

- a shared astrophysical field seen by all surveys
- a survey-A residual component
- a survey-B residual component

Each survey sees a mechanistic sum of the shared field and its own private
component. The split solve is then compared against a naive shared-only model
to show how survey-specific structure can leak into the inferred sky if the
decomposition is not made explicit.
"""

from __future__ import annotations

import math

import torch

from torchsky.map import build_spherical_harmonic_solver_hooks
from torchsky.map.solvers import (
    build_harmonic_degree_prior_operator,
    build_state_component_mixture_operator,
    build_state_subspace_operator,
    compose_bound_operators,
    make_diagonal_preconditioner,
)
from torchsky.map.target import TargetRenderSpec


def _tan_header(
    *,
    crval1: float,
    crval2: float,
) -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 72,
        "NAXIS2": 72,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 36.5,
        "CRPIX2": 36.5,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CDELT1": -0.14,
        "CDELT2": 0.14,
    }


def main() -> None:
    torch.manual_seed(53)
    lmax = 3
    nalm = (lmax + 1) * (lmax + 2) // 2

    shared_truth = torch.zeros(nalm, dtype=torch.complex64)
    shared_truth[0] = complex(math.sqrt(4.0 * math.pi) * 0.28, 0.0)
    shared_truth[1] = complex(0.07, 0.0)
    shared_truth[2] = complex(-0.05, 0.02)
    shared_truth[4] = complex(0.03, -0.01)

    survey_a_truth = torch.zeros(nalm, dtype=torch.complex64)
    survey_a_truth[3] = complex(0.05, -0.01)
    survey_a_truth[6] = complex(-0.02, 0.01)

    survey_b_truth = torch.zeros(nalm, dtype=torch.complex64)
    survey_b_truth[1] = complex(-0.04, 0.01)
    survey_b_truth[7] = complex(0.05, -0.02)

    joint_truth = torch.stack([shared_truth, survey_a_truth, survey_b_truth], dim=0)

    hooks = build_spherical_harmonic_solver_hooks(lmax=lmax)
    survey_a_targets = (
        TargetRenderSpec.from_header(_tan_header(crval1=210.0, crval2=54.0), band_label="survey_a"),
        TargetRenderSpec.from_header(_tan_header(crval1=213.0, crval2=52.8), band_label="survey_a"),
    )
    survey_b_targets = (
        TargetRenderSpec.from_header(_tan_header(crval1=208.0, crval2=55.2), band_label="survey_b"),
        TargetRenderSpec.from_header(_tan_header(crval1=205.8, crval2=53.7), band_label="survey_b"),
    )

    survey_a_block = build_state_component_mixture_operator(
        joint_truth,
        hooks.bind_targets(survey_a_targets),
        components=(0, 1),
    )
    survey_b_block = build_state_component_mixture_operator(
        joint_truth,
        hooks.bind_targets(survey_b_targets),
        components=(0, 2),
    )
    survey_a_prior = build_state_subspace_operator(
        joint_truth,
        build_harmonic_degree_prior_operator(
            survey_a_truth,
            lmax=lmax,
            power=2.0,
            include_monopole=False,
            apply_noise=lambda tensor: tensor * 0.08,
        ),
        selector=1,
    )
    survey_b_prior = build_state_subspace_operator(
        joint_truth,
        build_harmonic_degree_prior_operator(
            survey_b_truth,
            lmax=lmax,
            power=2.0,
            include_monopole=False,
            apply_noise=lambda tensor: tensor * 0.08,
        ),
        selector=2,
    )
    split_operator_set = compose_bound_operators(
        survey_a_block,
        survey_b_block,
        survey_a_prior,
        survey_b_prior,
    )

    clean_a = survey_a_block.apply(joint_truth)
    clean_b = survey_b_block.apply(joint_truth)
    observed_a = tuple(cutout + 0.006 * torch.randn_like(cutout) for cutout in clean_a)
    observed_b = tuple(cutout + 0.006 * torch.randn_like(cutout) for cutout in clean_b)
    split_data = (*observed_a, *observed_b, torch.zeros_like(survey_a_truth), torch.zeros_like(survey_b_truth))
    split_diagonal = split_operator_set.estimate_normal_operator_diagonal(
        split_data,
        initial_probe=torch.zeros_like(joint_truth),
        max_entries=64,
        damping=1.0e-4,
    )
    split_fit = split_operator_set.solve_normal_equation_cg(
        split_data,
        initial_solution=torch.zeros_like(joint_truth),
        iterations=10,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(split_diagonal),
    )

    shared_only_block = hooks.bind_targets((*survey_a_targets, *survey_b_targets))
    shared_only_data = (*observed_a, *observed_b)
    shared_only_diagonal = shared_only_block.estimate_normal_operator_diagonal(
        shared_only_data,
        initial_probe=torch.zeros_like(shared_truth),
        max_entries=32,
        damping=1.0e-4,
    )
    shared_only_fit = shared_only_block.solve_normal_equation_cg(
        shared_only_data,
        initial_solution=torch.zeros_like(shared_truth),
        iterations=10,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(shared_only_diagonal),
    )

    shared_error_split = float(torch.linalg.vector_norm(split_fit.solution[0] - shared_truth).item())
    shared_error_only = float(torch.linalg.vector_norm(shared_only_fit.solution - shared_truth).item())
    private_norms = [
        float(torch.linalg.vector_norm(split_fit.solution[idx]).item())
        for idx in (1, 2)
    ]

    print("Native shared-vs-survey-private harmonic decomposition example")
    print(f"lmax={lmax} nalm={nalm} survey_a_obs={len(observed_a)} survey_b_obs={len(observed_b)}")
    print(f"split residual norm history (first 5): {[round(v, 5) for v in split_fit.residual_norm_history[:5]]}")
    print(f"split residual norm history (last 5): {[round(v, 5) for v in split_fit.residual_norm_history[-5:]]}")
    print(f"shared-component error with split model: {shared_error_split:.6f}")
    print(f"shared-component error with naive shared-only model: {shared_error_only:.6f}")
    print(f"survey-private component norms: {[round(v, 6) for v in private_norms]}")
    print("useful when surveys like SPHEREx, WISE, or GALEX share sky structure but carry survey-specific residual fields")


if __name__ == "__main__":
    main()
