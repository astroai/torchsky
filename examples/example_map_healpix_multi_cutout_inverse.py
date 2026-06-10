#!/usr/bin/env python
"""Native joint multi-cutout sphere↔image inverse example with HEALPix maps.

This is useful when several image cutouts should constrain one shared
low-resolution spherical sky/background map:

- diffuse foreground fitting across many visits
- transferring a spherical prior into overlapping local cutouts
- building a native coarse sky model from multiple image pointings
"""

from __future__ import annotations

import math

import torch

from torchsky.map import build_spherical_healpix_solver_hooks
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

    diffuse = 0.20 + 0.12 * torch.cos(2.0 * lat_rad) + 0.08 * torch.sin(lon_rad)
    blob_a = torch.exp(-(((lon_deg - 150.0) / 7.0) ** 2) - ((lat_deg - 2.0) / 4.5) ** 2)
    blob_b = 0.6 * torch.exp(
        -(((lon_deg - 145.0) / 5.5) ** 2) - ((lat_deg + 1.0) / 3.8) ** 2
    )
    blob_c = 0.5 * torch.exp(
        -(((lon_deg - 154.5) / 4.2) ** 2) - ((lat_deg - 4.0) / 2.8) ** 2
    )
    return (
        diffuse
        + blob_a.to(torch.float32)
        + blob_b.to(torch.float32)
        + blob_c.to(torch.float32)
    )


def main() -> None:
    torch.manual_seed(23)
    nside = 32
    truth = _synthetic_foreground_map(nside)

    targets = (
        TargetRenderSpec.from_header(
            _tan_header(crval1=150.0, crval2=2.0), band_label="fg"
        ),
        TargetRenderSpec.from_header(
            _tan_header(crval1=146.0, crval2=0.5), band_label="fg"
        ),
        TargetRenderSpec.from_header(
            _tan_header(crval1=154.0, crval2=3.5), band_label="fg"
        ),
    )
    hooks = build_spherical_healpix_solver_hooks(
        nside=nside,
        mode="pixel_bilinear",
    )
    operator_set = hooks.bind_targets(targets)

    clean_cutouts = operator_set.apply(truth)
    observed_cutouts = tuple(
        cutout + 0.03 * torch.randn_like(cutout) for cutout in clean_cutouts
    )

    fit = operator_set.solve_normal_equation_cg(
        observed_cutouts,
        initial_solution=torch.zeros_like(truth),
        iterations=18,
        damping=1.0e-4,
    )
    reconstructed_cutouts = operator_set.apply(fit.solution)
    rmses = [
        math.sqrt(float(torch.mean(torch.square(pred - obs)).item()))
        for pred, obs in zip(reconstructed_cutouts, observed_cutouts, strict=True)
    ]
    support = operator_set.adjoint(
        tuple(torch.ones_like(obs) for obs in observed_cutouts)
    )
    supported_pixels = int((support > 0.0).sum().item())

    print("Native multi-cutout HEALPix inverse example")
    print(f"nside={nside} npix={truth.numel()} observations={len(targets)}")
    print(f"supported HEALPix pixels from all cutouts: {supported_pixels}")
    print(
        f"residual norm history (first 5): {[round(v, 5) for v in fit.residual_norm_history[:5]]}"
    )
    print(
        f"residual norm history (last 5): {[round(v, 5) for v in fit.residual_norm_history[-5:]]}"
    )
    print(f"per-cutout image RMSE: {[round(v, 6) for v in rmses]}")
    print(
        "useful when overlapping pointings should constrain one shared spherical foreground map"
    )


if __name__ == "__main__":
    main()
