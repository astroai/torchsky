#!/usr/bin/env python
"""Native image↔alm inverse example for low-order sky/background fitting.

This is useful when a local image cutout is well described by a small number of
global spherical modes:

- diffuse foreground/background fitting
- low-order calibration transfer from sphere to cutout
- imposing smooth all-sky priors while optimizing in image space
"""

from __future__ import annotations

import math

import torch

from torchsky.map import (
    MapBandpassSpec,
    MapProductSpec,
    build_spherical_harmonic_solver_hooks,
)


def _tan_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 80,
        "NAXIS2": 80,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 40.5,
        "CRPIX2": 40.5,
        "CRVAL1": 210.0,
        "CRVAL2": 54.0,
        "CDELT1": -0.12,
        "CDELT2": 0.12,
    }


def main() -> None:
    torch.manual_seed(11)
    lmax = 3
    nalm = (lmax + 1) * (lmax + 2) // 2

    truth = torch.zeros(nalm, dtype=torch.complex64)
    truth[0] = complex(math.sqrt(4.0 * math.pi) * 0.4, 0.0)
    truth[1] = complex(0.12, 0.0)
    truth[2] = complex(-0.08, 0.03)
    truth[4] = complex(0.05, -0.02)
    truth[7] = complex(-0.03, 0.01)

    product = MapProductSpec(
        target_header=_tan_header(),
        bands=(MapBandpassSpec(label="background", transmission_center=7000.0),),
    )
    hooks = build_spherical_harmonic_solver_hooks(lmax=lmax)
    operator = hooks.bind_target(product.target_render_spec())

    clean_cutout = operator.apply(truth)
    observed_cutout = clean_cutout + 0.01 * torch.randn_like(clean_cutout)

    fit = operator.solve_weighted_least_squares(
        observed_cutout,
        initial_solution=torch.zeros_like(truth),
        iterations=40,
        step_size=1.0e-4,
    )
    reconstructed_cutout = operator.apply(fit.solution)

    final_rmse = math.sqrt(
        float(torch.mean(torch.square(reconstructed_cutout - observed_cutout)).item())
    )

    print("Native harmonic inverse example")
    print(f"lmax={lmax} nalm={nalm} target={tuple(observed_cutout.shape)}")
    print(
        f"residual norm history (first 5): {[round(v, 5) for v in fit.residual_norm_history[:5]]}"
    )
    print(
        f"residual norm history (last 5): {[round(v, 5) for v in fit.residual_norm_history[-5:]]}"
    )
    print(f"final image RMSE: {final_rmse:.6f}")
    print(
        "note: one cutout constrains the rendered image well, not every alm coefficient uniquely"
    )


if __name__ == "__main__":
    main()
