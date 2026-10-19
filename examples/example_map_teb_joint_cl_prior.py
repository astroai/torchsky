#!/usr/bin/env python
"""Native joint T/E/B harmonic inverse with a cross-spectrum prior.

This is useful when temperature and polarization cutouts should constrain one
shared spherical harmonic state under an astronomer-style prior specified by
TT/EE/BB and cross-spectra such as TE.
"""

from __future__ import annotations

import math

import torch

from torchsky.map.solvers import build_spherical_harmonic_solver_hooks, build_spherical_spin_harmonic_solver_hooks
from torchsky.map.solvers import (
    build_state_subspace_operator,
    build_teb_cls_prior_operator,
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
        "CDELT1": -0.12,
        "CDELT2": 0.12,
    }


def main() -> None:
    torch.manual_seed(47)
    lmax = 3
    nalm = (lmax + 1) * (lmax + 2) // 2

    teb_truth = torch.zeros((3, nalm), dtype=torch.complex64)
    teb_truth[0, 0] = complex(math.sqrt(4.0 * math.pi) * 0.35, 0.0)
    teb_truth[0, 1] = complex(0.08, 0.0)
    teb_truth[0, 2] = complex(-0.04, 0.02)
    teb_truth[1, 2] = complex(0.06, 0.02)
    teb_truth[1, 5] = complex(-0.03, 0.01)
    teb_truth[2, 4] = complex(0.04, -0.01)
    teb_truth[2, 7] = complex(-0.02, 0.01)

    temp_hooks = build_spherical_harmonic_solver_hooks(lmax=lmax)
    pol_hooks = build_spherical_spin_harmonic_solver_hooks(spin=2, lmax=lmax)

    temp_block = build_state_subspace_operator(
        teb_truth,
        temp_hooks.bind_targets(
            (
                TargetRenderSpec.from_header(
                    _tan_header(crval1=210.0, crval2=54.0), band_label="temp"
                ),
                TargetRenderSpec.from_header(
                    _tan_header(crval1=214.0, crval2=52.0), band_label="temp"
                ),
            )
        ),
        selector=0,
    )
    pol_block = build_state_subspace_operator(
        teb_truth,
        pol_hooks.bind_targets(
            (
                TargetRenderSpec.from_header(
                    _tan_header(crval1=210.0, crval2=54.0), band_label="pol"
                ),
                TargetRenderSpec.from_header(
                    _tan_header(crval1=208.5, crval2=55.0), band_label="pol"
                ),
            )
        ),
        selector=[1, 2],
    )
    teb_prior = build_teb_cls_prior_operator(
        teb_truth,
        lmax=lmax,
        cls_tt=torch.tensor([10.0, 2.0, 0.8, 0.3], dtype=torch.float64),
        cls_ee=torch.tensor([5.0, 1.2, 0.5, 0.2], dtype=torch.float64),
        cls_bb=torch.tensor([5.0, 1.2, 0.5, 0.2], dtype=torch.float64),
        cls_te=torch.tensor([0.0, 0.4, 0.15, 0.05], dtype=torch.float64),
        apply_noise=lambda tensor: tensor * 0.08,
    )
    operator_set = compose_bound_operators(temp_block, pol_block, teb_prior)

    clean_temp = temp_block.apply(teb_truth)
    clean_pol = pol_block.apply(teb_truth)
    observed_temp = tuple(
        cutout + 0.01 * torch.randn_like(cutout) for cutout in clean_temp
    )
    observed_pol = tuple(maps + 0.005 * torch.randn_like(maps) for maps in clean_pol)
    data = (*observed_temp, *observed_pol, torch.zeros_like(teb_truth))

    diagonal = operator_set.estimate_normal_operator_diagonal(
        data,
        initial_probe=torch.zeros_like(teb_truth),
        damping=1.0e-4,
    )
    fit = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(teb_truth),
        iterations=10,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(diagonal),
    )

    reconstructed_temp = temp_block.apply(fit.solution)
    reconstructed_pol = pol_block.apply(fit.solution)
    temp_rmses = [
        math.sqrt(float(torch.mean(torch.square(pred - obs)).item()))
        for pred, obs in zip(reconstructed_temp, observed_temp, strict=True)
    ]
    pol_rmses = [
        math.sqrt(float(torch.mean(torch.square(pred - obs)).item()))
        for pred, obs in zip(reconstructed_pol, observed_pol, strict=True)
    ]
    prior_norm = float(torch.linalg.vector_norm(teb_prior.apply(fit.solution)).item())

    print("Native joint T/E/B C_ell-prior inverse example")
    print(
        f"lmax={lmax} nalm={nalm} temp_obs={len(observed_temp)} pol_obs={len(observed_pol)}"
    )
    print(
        f"residual norm history (first 5): {[round(v, 5) for v in fit.residual_norm_history[:5]]}"
    )
    print(
        f"residual norm history (last 5): {[round(v, 5) for v in fit.residual_norm_history[-5:]]}"
    )
    print(f"temperature cutout RMSE: {[round(v, 6) for v in temp_rmses]}")
    print(f"polarization cutout RMSE: {[round(v, 6) for v in pol_rmses]}")
    print(f"T/E/B cross-spectrum prior norm: {prior_norm:.6f}")
    print(
        "useful when one latent spherical T/E/B model should explain both scalar and polarized sky cutouts"
    )


if __name__ == "__main__":
    main()
