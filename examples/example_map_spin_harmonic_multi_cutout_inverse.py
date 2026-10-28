#!/usr/bin/env python
"""Native joint multi-cutout image↔(E,B) inverse example.

This is useful when several Q/U cutouts should share one smooth spherical
polarization model:

- coarse polarized foreground fitting across visits
- shared E/B regularization across pointings
- initializing a polarized latent sphere before richer local refinement
"""

from __future__ import annotations

import math

import torch

from torchsky.map.solvers import build_spherical_spin_harmonic_solver_hooks
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
    torch.manual_seed(29)
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
            _tan_header(crval1=121.5, crval2=34.0), band_label="pol"
        ),
        TargetRenderSpec.from_header(
            _tan_header(crval1=118.2, crval2=36.1), band_label="pol"
        ),
    )
    hooks = build_spherical_spin_harmonic_solver_hooks(
        spin=2,
        lmax=lmax,
    )
    operator_set = hooks.bind_targets(targets)

    clean_maps = operator_set.apply(truth)
    observed_maps = tuple(maps + 0.005 * torch.randn_like(maps) for maps in clean_maps)

    fit = operator_set.solve_normal_equation_cg(
        observed_maps,
        initial_solution=torch.zeros_like(truth),
        iterations=18,
        damping=1.0e-4,
    )
    reconstructed_maps = operator_set.apply(fit.solution)
    rmses = [
        math.sqrt(float(torch.mean(torch.square(pred - obs)).item()))
        for pred, obs in zip(reconstructed_maps, observed_maps, strict=True)
    ]

    print("Native multi-cutout spin harmonic inverse example")
    print(f"spin=2 lmax={lmax} nalm={nalm} observations={len(targets)}")
    print(
        f"residual norm history (first 5): {[round(v, 5) for v in fit.residual_norm_history[:5]]}"
    )
    print(
        f"residual norm history (last 5): {[round(v, 5) for v in fit.residual_norm_history[-5:]]}"
    )
    print(f"per-cutout Q/U RMSE: {[round(v, 6) for v in rmses]}")
    print(
        "useful when several polarized cutouts should share one smooth spherical E/B model"
    )


if __name__ == "__main__":
    main()
