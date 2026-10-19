#!/usr/bin/env python
"""Native mixed-constraint spherical inverse example.

This combines four observation families against one shared alm state:

- image-plane cutouts
- coefficient-space anchor / prior constraints
- harmonic degree-weighted regularization
- dense harmonic covariance regularization

It is useful when image data should fit a smooth spherical field while still
remaining close to an external low-order model or calibration transfer.
"""

from __future__ import annotations

import math

import torch

from torchsky.map.solvers import build_spherical_harmonic_solver_hooks
from torchsky.map.solvers import (
    build_harmonic_dense_prior_operator,
    build_harmonic_degree_prior_operator,
    build_state_identity_operator,
    compose_bound_operators,
    harmonic_dense_covariance,
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
        "CDELT1": -0.15,
        "CDELT2": 0.15,
    }


def main() -> None:
    torch.manual_seed(31)
    lmax = 3
    nalm = (lmax + 1) * (lmax + 2) // 2

    truth = torch.zeros(nalm, dtype=torch.complex64)
    truth[0] = complex(math.sqrt(4.0 * math.pi) * 0.30, 0.0)
    truth[1] = complex(0.10, 0.0)
    truth[2] = complex(-0.06, 0.02)
    truth[4] = complex(0.03, -0.01)
    truth[7] = complex(-0.02, 0.01)

    anchor = truth.clone()
    anchor[2] += complex(0.02, -0.01)
    anchor[7] += complex(-0.01, 0.01)

    targets = (
        TargetRenderSpec.from_header(
            _tan_header(crval1=210.0, crval2=54.0), band_label="bg"
        ),
        TargetRenderSpec.from_header(
            _tan_header(crval1=214.0, crval2=52.5), band_label="bg"
        ),
    )
    hooks = build_spherical_harmonic_solver_hooks(lmax=lmax)
    image_block = hooks.bind_targets(targets)
    anchor_block = build_state_identity_operator(
        truth,
        apply_noise=lambda tensor: tensor * 0.2,
    )
    smoothness_block = build_harmonic_degree_prior_operator(
        truth,
        lmax=lmax,
        power=2.0,
        include_monopole=False,
        apply_noise=lambda tensor: tensor * 0.05,
    )
    dense_covariance = harmonic_dense_covariance(
        lmax=lmax,
        ell_correlation_length=1.2,
        diagonal_jitter=1.0e-3,
        amplitude=1.0,
        dtype=torch.complex128,
    )
    dense_prior_block = build_harmonic_dense_prior_operator(
        truth,
        dense_covariance,
        apply_noise=lambda tensor: tensor * 0.05,
    )
    operator_set = compose_bound_operators(
        image_block,
        anchor_block,
        smoothness_block,
        dense_prior_block,
    )

    clean_cutouts = image_block.apply(truth)
    observed_cutouts = tuple(
        cutout + 0.01 * torch.randn_like(cutout) for cutout in clean_cutouts
    )
    data = (*observed_cutouts, anchor, torch.zeros_like(truth), torch.zeros_like(truth))

    diagonal = operator_set.estimate_normal_operator_diagonal(
        data,
        initial_probe=torch.zeros_like(truth),
        max_entries=32,
        damping=1.0e-4,
    )
    fit = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=8,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(diagonal),
    )

    reconstructed_cutouts = image_block.apply(fit.solution)
    cutout_rmses = [
        math.sqrt(float(torch.mean(torch.square(pred - obs)).item()))
        for pred, obs in zip(reconstructed_cutouts, observed_cutouts, strict=True)
    ]
    anchor_distance = float(torch.linalg.vector_norm(fit.solution - anchor).item())
    truth_distance = float(torch.linalg.vector_norm(fit.solution - truth).item())
    smoothness_norm = float(
        torch.linalg.vector_norm(smoothness_block.apply(fit.solution)).item()
    )
    dense_prior_norm = float(
        torch.linalg.vector_norm(dense_prior_block.apply(fit.solution)).item()
    )

    print("Native mixed-constraint harmonic inverse example")
    print(f"lmax={lmax} nalm={nalm} image_observations={len(targets)}")
    print(
        f"residual norm history (first 5): {[round(v, 5) for v in fit.residual_norm_history[:5]]}"
    )
    print(
        f"residual norm history (last 5): {[round(v, 5) for v in fit.residual_norm_history[-5:]]}"
    )
    print(f"per-cutout image RMSE: {[round(v, 6) for v in cutout_rmses]}")
    print(f"distance to coefficient anchor: {anchor_distance:.6f}")
    print(f"distance to truth coefficients: {truth_distance:.6f}")
    print(f"harmonic smoothness penalty norm: {smoothness_norm:.6f}")
    print(f"dense covariance prior norm: {dense_prior_norm:.6f}")
    print(
        "useful when image cutouts should stay close to an external low-order spherical model"
    )


if __name__ == "__main__":
    main()
