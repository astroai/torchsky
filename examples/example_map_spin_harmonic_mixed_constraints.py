#!/usr/bin/env python
"""Native mixed-constraint spin-harmonic inverse example.

This combines polarized Q/U cutouts with a separable Kronecker prior against
one shared spherical E/B state.

It is useful when local polarization data should stay compatible with a smooth
large-scale spherical model while expressing different prior strength and mild
coupling between E and B modes.
"""

from __future__ import annotations

import math

import torch

from torchsky.map.solvers import build_spherical_spin_harmonic_solver_hooks
from torchsky.map.solvers import (
    build_spin_harmonic_kronecker_prior_operator,
    compose_bound_operators,
    harmonic_degree_diagonal,
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
        "CDELT1": -0.10,
        "CDELT2": 0.10,
    }


def main() -> None:
    torch.manual_seed(41)
    lmax = 3
    nalm = (lmax + 1) * (lmax + 2) // 2

    alm_e = torch.zeros(nalm, dtype=torch.complex64)
    alm_b = torch.zeros(nalm, dtype=torch.complex64)
    alm_e[2] = complex(0.08, 0.03)
    alm_e[5] = complex(-0.04, 0.01)
    alm_b[4] = complex(0.05, -0.02)
    alm_b[7] = complex(-0.03, 0.01)
    truth = torch.stack([alm_e, alm_b], dim=0)

    targets = (
        TargetRenderSpec.from_header(
            _tan_header(crval1=120.0, crval2=35.0), band_label="pol"
        ),
        TargetRenderSpec.from_header(
            _tan_header(crval1=121.4, crval2=34.2), band_label="pol"
        ),
    )
    hooks = build_spherical_spin_harmonic_solver_hooks(
        spin=2,
        lmax=lmax,
    )
    image_block = hooks.bind_targets(targets)
    harmonic_factor = harmonic_degree_diagonal(
        lmax=lmax,
        power=2.0,
        include_monopole=False,
        dtype=torch.complex64,
    )
    eb_coupling = torch.tensor(
        [[1.0 + 0.0j, 0.12 + 0.0j], [0.0 + 0.0j, 0.7 + 0.0j]],
        dtype=torch.complex64,
    )
    smoothness_block = build_spin_harmonic_kronecker_prior_operator(
        truth,
        harmonic_factor=harmonic_factor,
        eb_coupling=eb_coupling,
        apply_noise=lambda tensor: tensor * 0.08,
    )
    operator_set = compose_bound_operators(image_block, smoothness_block)

    clean_maps = image_block.apply(truth)
    observed_maps = tuple(maps + 0.005 * torch.randn_like(maps) for maps in clean_maps)
    data = (*observed_maps, torch.zeros_like(truth))

    diagonal = operator_set.estimate_normal_operator_diagonal(
        data,
        initial_probe=torch.zeros_like(truth),
        max_entries=32,
        damping=1.0e-4,
    )
    fit = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=10,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(diagonal),
    )
    reconstructed_maps = image_block.apply(fit.solution)
    rmses = [
        math.sqrt(float(torch.mean(torch.square(pred - obs)).item()))
        for pred, obs in zip(reconstructed_maps, observed_maps, strict=True)
    ]
    smoothness_norm = float(
        torch.linalg.vector_norm(smoothness_block.apply(fit.solution)).item()
    )

    print("Native mixed-constraint spin harmonic inverse example")
    print(f"spin=2 lmax={lmax} nalm={nalm} image_observations={len(targets)}")
    print(
        f"residual norm history (first 5): {[round(v, 5) for v in fit.residual_norm_history[:5]]}"
    )
    print(
        f"residual norm history (last 5): {[round(v, 5) for v in fit.residual_norm_history[-5:]]}"
    )
    print(f"per-cutout Q/U RMSE: {[round(v, 6) for v in rmses]}")
    print(f"Kronecker E/B-harmonic prior norm: {smoothness_norm:.6f}")
    print(
        "useful when local polarized cutouts should remain compatible with a smooth spherical E/B model"
    )


if __name__ == "__main__":
    main()
