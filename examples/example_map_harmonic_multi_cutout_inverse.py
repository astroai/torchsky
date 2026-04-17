#!/usr/bin/env python
"""Native joint multi-cutout image↔alm inverse example.

This is useful when multiple image cutouts constrain one shared smooth
all-sky component:

- diffuse foreground/background fitting across several visits
- low-order calibration/background transfer between pointings
- initializing latent spherical fields before a richer field-model fit
"""

from __future__ import annotations

import math

import torch

from torchsky.map import build_spherical_harmonic_solver_hooks
from torchsky.map.solvers import make_diagonal_preconditioner
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
    torch.manual_seed(17)
    lmax = 3
    nalm = (lmax + 1) * (lmax + 2) // 2

    truth = torch.zeros(nalm, dtype=torch.complex64)
    truth[0] = complex(math.sqrt(4.0 * math.pi) * 0.35, 0.0)
    truth[1] = complex(0.08, 0.0)
    truth[2] = complex(-0.05, 0.03)
    truth[4] = complex(0.04, -0.01)
    truth[7] = complex(-0.02, 0.01)

    targets = (
        TargetRenderSpec.from_header(_tan_header(crval1=210.0, crval2=54.0), band_label="bg"),
        TargetRenderSpec.from_header(_tan_header(crval1=214.0, crval2=52.5), band_label="bg"),
        TargetRenderSpec.from_header(_tan_header(crval1=205.5, crval2=55.5), band_label="bg"),
    )
    hooks = build_spherical_harmonic_solver_hooks(lmax=lmax)
    operator_set = hooks.bind_targets(targets)

    clean_cutouts = operator_set.apply(truth)
    observed_cutouts = tuple(
        cutout + 0.01 * torch.randn_like(cutout)
        for cutout in clean_cutouts
    )

    diagonal = operator_set.estimate_normal_operator_diagonal(
        observed_cutouts,
        initial_probe=torch.zeros_like(truth),
        max_entries=32,
        damping=1.0e-4,
    )
    fit = operator_set.solve_normal_equation_cg(
        observed_cutouts,
        initial_solution=torch.zeros_like(truth),
        iterations=12,
        damping=1.0e-4,
        preconditioner=make_diagonal_preconditioner(diagonal),
    )
    reconstructed_cutouts = operator_set.apply(fit.solution)
    rmses = [
        math.sqrt(float(torch.mean(torch.square(pred - obs)).item()))
        for pred, obs in zip(reconstructed_cutouts, observed_cutouts, strict=True)
    ]

    print("Native multi-cutout harmonic inverse example")
    print(f"lmax={lmax} nalm={nalm} observations={len(targets)}")
    print(f"residual norm history (first 5): {[round(v, 5) for v in fit.residual_norm_history[:5]]}")
    print(f"residual norm history (last 5): {[round(v, 5) for v in fit.residual_norm_history[-5:]]}")
    print(f"per-cutout image RMSE: {[round(v, 6) for v in rmses]}")
    print("uses exact-diagonal PCG on the shared normal equations")
    print("useful when several pointings must share one smooth latent spherical component")


if __name__ == "__main__":
    main()
