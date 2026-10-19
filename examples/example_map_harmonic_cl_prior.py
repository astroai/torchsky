#!/usr/bin/env python
"""Native harmonic inverse with a power-spectrum prior.

This is useful when a spherical background or diffuse component is expected to
follow a known or assumed angular power spectrum `C_ell`, optionally after a
beam or transfer function.
"""

from __future__ import annotations

import math

import torch

from torchsky.map.solvers import build_spherical_harmonic_solver_hooks
from torchsky.map.solvers import (
    build_harmonic_cl_prior_operator,
    compose_bound_operators,
    make_diagonal_preconditioner,
)
from torchsky.map.target import TargetRenderSpec
from torchsky.sphere import gaussian_beam


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
    torch.manual_seed(43)
    lmax = 3
    nalm = (lmax + 1) * (lmax + 2) // 2

    truth = torch.zeros(nalm, dtype=torch.complex64)
    truth[0] = complex(math.sqrt(4.0 * math.pi) * 0.25, 0.0)
    truth[1] = complex(0.12, 0.0)
    truth[2] = complex(-0.05, 0.02)
    truth[4] = complex(0.03, -0.01)

    targets = (
        TargetRenderSpec.from_header(
            _tan_header(crval1=210.0, crval2=54.0), band_label="bg"
        ),
        TargetRenderSpec.from_header(
            _tan_header(crval1=214.0, crval2=52.0), band_label="bg"
        ),
    )
    hooks = build_spherical_harmonic_solver_hooks(lmax=lmax)
    image_block = hooks.bind_targets(targets)

    cl_prior = torch.tensor([10.0, 1.5, 0.4, 0.15], dtype=torch.float64)
    beam = gaussian_beam(math.radians(1.0), lmax)
    cl_block = build_harmonic_cl_prior_operator(
        truth,
        lmax=lmax,
        cls=cl_prior,
        beam_transfer=beam,
        apply_noise=lambda tensor: tensor * 0.1,
    )
    operator_set = compose_bound_operators(image_block, cl_block)

    clean_cutouts = image_block.apply(truth)
    observed_cutouts = tuple(
        cutout + 0.01 * torch.randn_like(cutout) for cutout in clean_cutouts
    )
    data = (*observed_cutouts, torch.zeros_like(truth))

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
    rmses = [
        math.sqrt(float(torch.mean(torch.square(pred - obs)).item()))
        for pred, obs in zip(reconstructed_cutouts, observed_cutouts, strict=True)
    ]
    cl_penalty = float(torch.linalg.vector_norm(cl_block.apply(fit.solution)).item())

    print("Native harmonic C_ell-prior inverse example")
    print(f"lmax={lmax} nalm={nalm} image_observations={len(targets)}")
    print(
        f"residual norm history (first 5): {[round(v, 5) for v in fit.residual_norm_history[:5]]}"
    )
    print(
        f"residual norm history (last 5): {[round(v, 5) for v in fit.residual_norm_history[-5:]]}"
    )
    print(f"per-cutout image RMSE: {[round(v, 6) for v in rmses]}")
    print(f"C_ell prior norm: {cl_penalty:.6f}")
    print(
        "useful when diffuse sky structure is better specified by an angular power spectrum than by hand-tuned penalties"
    )


if __name__ == "__main__":
    main()
