#!/usr/bin/env python
"""Native mixed-constraint HEALPix inverse example.

This combines image cutouts with two astronomically useful spherical priors
against one shared HEALPix latent state:

- a coarse all-sky map anchor
- low-rank large-scale template suppression

It is useful when local image-plane data should refine, but not completely
ignore, an external spherical background model while suppressing simple
monopole/dipole-like nuisance structure.
"""

from __future__ import annotations

import math

import torch

from torchsky.map import build_spherical_healpix_solver_hooks
from torchsky.map.solvers import (
    build_state_identity_operator,
    build_state_low_rank_operator,
    compose_bound_operators,
)
from torchsky.map.target import TargetRenderSpec
from torchsky.sphere import healpix as hp


def _tan_header(
    *,
    crval1: float,
    crval2: float,
) -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 64,
        "NAXIS2": 64,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 32.5,
        "CRPIX2": 32.5,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CDELT1": -0.08,
        "CDELT2": 0.08,
    }


def _synthetic_foreground_map(nside: int) -> torch.Tensor:
    npix = hp.nside2npix(nside)
    ipix = torch.arange(npix, dtype=torch.int64)
    lon_deg, lat_deg = hp.pix2ang(nside, ipix, nest=False, lonlat=True)
    lon_rad = torch.deg2rad(lon_deg.to(torch.float32))
    lat_rad = torch.deg2rad(lat_deg.to(torch.float32))

    diffuse = 0.18 + 0.10 * torch.cos(2.0 * lat_rad) + 0.07 * torch.sin(lon_rad)
    blob_a = torch.exp(-(((lon_deg - 150.0) / 7.0) ** 2) - ((lat_deg - 2.0) / 4.5) ** 2)
    blob_b = 0.5 * torch.exp(
        -(((lon_deg - 145.5) / 5.5) ** 2) - ((lat_deg + 1.0) / 3.2) ** 2
    )
    return diffuse + blob_a.to(torch.float32) + blob_b.to(torch.float32)


def main() -> None:
    torch.manual_seed(37)
    nside = 32
    truth = _synthetic_foreground_map(nside)
    anchor = truth + 0.04 * torch.randn_like(truth)

    targets = (
        TargetRenderSpec.from_header(
            _tan_header(crval1=150.0, crval2=2.0), band_label="fg"
        ),
        TargetRenderSpec.from_header(
            _tan_header(crval1=146.5, crval2=0.8), band_label="fg"
        ),
    )
    hooks = build_spherical_healpix_solver_hooks(
        nside=nside,
        mode="pixel_bilinear",
    )
    image_block = hooks.bind_targets(targets)
    anchor_block = build_state_identity_operator(
        truth,
        apply_noise=lambda tensor: tensor * 0.1,
    )
    ipix = torch.arange(truth.numel(), dtype=torch.int64)
    lon_deg, lat_deg = hp.pix2ang(nside, ipix, nest=False, lonlat=True)
    template_basis = torch.stack(
        [
            torch.ones_like(truth),
            torch.cos(torch.deg2rad(lon_deg.to(torch.float32))),
            torch.sin(torch.deg2rad(lon_deg.to(torch.float32))),
            torch.sin(torch.deg2rad(lat_deg.to(torch.float32))),
        ],
        dim=1,
    )
    low_rank_block = build_state_low_rank_operator(
        truth,
        template_basis,
        weights=torch.full((4,), 0.05, dtype=torch.float32),
    )
    operator_set = compose_bound_operators(image_block, anchor_block, low_rank_block)

    clean_cutouts = image_block.apply(truth)
    observed_cutouts = tuple(
        cutout + 0.03 * torch.randn_like(cutout) for cutout in clean_cutouts
    )
    data = (*observed_cutouts, anchor, torch.zeros(4, dtype=torch.float32))

    fit = operator_set.solve_normal_equation_cg(
        data,
        initial_solution=torch.zeros_like(truth),
        iterations=12,
        damping=1.0e-4,
    )
    reconstructed_cutouts = image_block.apply(fit.solution)
    cutout_rmses = [
        math.sqrt(float(torch.mean(torch.square(pred - obs)).item()))
        for pred, obs in zip(reconstructed_cutouts, observed_cutouts, strict=True)
    ]
    anchor_distance = float(torch.linalg.vector_norm(fit.solution - anchor).item())
    truth_distance = float(torch.linalg.vector_norm(fit.solution - truth).item())
    template_norm = float(
        torch.linalg.vector_norm(low_rank_block.apply(fit.solution)).item()
    )

    print("Native mixed-constraint HEALPix inverse example")
    print(f"nside={nside} npix={truth.numel()} image_observations={len(targets)}")
    print(
        f"residual norm history (first 5): {[round(v, 5) for v in fit.residual_norm_history[:5]]}"
    )
    print(
        f"residual norm history (last 5): {[round(v, 5) for v in fit.residual_norm_history[-5:]]}"
    )
    print(f"per-cutout image RMSE: {[round(v, 6) for v in cutout_rmses]}")
    print(f"distance to HEALPix anchor: {anchor_distance:.6f}")
    print(f"distance to truth map: {truth_distance:.6f}")
    print(f"large-scale template penalty norm: {template_norm:.6f}")
    print(
        "useful when local cutouts should refine a coarse all-sky background without discarding it"
    )


if __name__ == "__main__":
    main()
