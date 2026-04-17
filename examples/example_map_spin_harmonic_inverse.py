#!/usr/bin/env python
"""Native image↔spin-alm inverse example for polarized sky cutouts.

This is useful when Q/U images should be fit with a smooth spherical
polarization model directly in coefficient space:

- coarse polarized foreground modeling
- low-order E/B transfer across cutouts
- regularized polarization backgrounds that should remain spherical
"""

from __future__ import annotations

import math

import torch

from torchsky.map import (
    MapBandpassSpec,
    MapProductSpec,
    build_spherical_spin_harmonic_solver_hooks,
)


def _tan_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 72,
        "NAXIS2": 72,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 36.5,
        "CRPIX2": 36.5,
        "CRVAL1": 120.0,
        "CRVAL2": 35.0,
        "CDELT1": -0.10,
        "CDELT2": 0.10,
    }


def main() -> None:
    torch.manual_seed(19)
    lmax = 3
    nalm = (lmax + 1) * (lmax + 2) // 2

    alm_e = torch.zeros(nalm, dtype=torch.complex64)
    alm_b = torch.zeros(nalm, dtype=torch.complex64)
    alm_e[2] = complex(0.08, 0.03)
    alm_e[5] = complex(-0.04, 0.01)
    alm_b[4] = complex(0.05, -0.02)
    alm_b[7] = complex(-0.03, 0.01)
    truth = torch.stack([alm_e, alm_b], dim=0)

    product = MapProductSpec(
        target_header=_tan_header(),
        bands=(MapBandpassSpec(label="pol", transmission_center=30000.0),),
    )
    hooks = build_spherical_spin_harmonic_solver_hooks(
        spin=2,
        lmax=lmax,
    )
    operator = hooks.bind_target(product.target_render_spec())

    clean_maps = operator.apply(truth)
    observed_maps = clean_maps + 0.005 * torch.randn_like(clean_maps)

    fit = operator.solve_weighted_least_squares(
        observed_maps,
        initial_solution=torch.zeros_like(truth),
        iterations=40,
        step_size=1.0e-4,
    )
    reconstructed_maps = operator.apply(fit.solution)
    final_rmse = math.sqrt(
        float(torch.mean(torch.square(reconstructed_maps - observed_maps)).item())
    )

    print("Native spin harmonic inverse example")
    print(f"spin=2 lmax={lmax} nalm={nalm} target={tuple(observed_maps.shape[-2:])}")
    print(f"residual norm history (first 5): {[round(v, 5) for v in fit.residual_norm_history[:5]]}")
    print(f"residual norm history (last 5): {[round(v, 5) for v in fit.residual_norm_history[-5:]]}")
    print(f"final Q/U image RMSE: {final_rmse:.6f}")
    print("note: a single cutout constrains the rendered polarization field more strongly than every E/B coefficient")


if __name__ == "__main__":
    main()
