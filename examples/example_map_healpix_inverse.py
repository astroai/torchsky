#!/usr/bin/env python
"""Native sphere↔image inverse example with HEALPix coefficients.

This example shows a useful pattern for real survey work:

1. start from an all-sky HEALPix prior or latent coefficient map
2. project it onto a local TAN cutout on the fly
3. accumulate cutout residuals back onto the sphere with the exact bilinear adjoint
4. iterate directly on HEALPix coefficients without leaving torchsky

This is useful for:
- fitting all-sky diffuse backgrounds from many overlapping cutouts
- carrying spherical priors into local image-plane optimization
- building coarse sky models that stay native on the sphere while observations stay in image space
"""

from __future__ import annotations

import math

import torch

from torchsky.map import BandpassSpec, ProductSpec
from torchsky.map.solvers import build_spherical_healpix_solver_hooks
from torchsky.sphere import healpix as hp


def _tan_header() -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 64,
        "NAXIS2": 64,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 32.5,
        "CRPIX2": 32.5,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -0.08,
        "CDELT2": 0.08,
    }


def _synthetic_foreground_map(nside: int) -> torch.Tensor:
    npix = hp.nside2npix(nside)
    ipix = torch.arange(npix, dtype=torch.int64)
    lon_deg, lat_deg = hp.pix2ang(nside, ipix, nest=False, lonlat=True)
    lon_rad = torch.deg2rad(lon_deg.to(torch.float32))
    lat_rad = torch.deg2rad(lat_deg.to(torch.float32))

    diffuse = 0.25 + 0.15 * torch.cos(2.0 * lat_rad) + 0.10 * torch.sin(lon_rad)
    blob_a = torch.exp(-(((lon_deg - 150.0) / 7.0) ** 2) - ((lat_deg - 2.0) / 4.5) ** 2)
    blob_b = 0.7 * torch.exp(
        -(((lon_deg - 144.0) / 5.0) ** 2) - ((lat_deg + 1.5) / 3.5) ** 2
    )
    return diffuse + blob_a.to(torch.float32) + blob_b.to(torch.float32)


def main() -> None:
    torch.manual_seed(7)
    nside = 32
    truth = _synthetic_foreground_map(nside)

    product = ProductSpec(
        target_header=_tan_header(),
        bands=(BandpassSpec(label="foreground", transmission_center=8500.0),),
    )
    hooks = build_spherical_healpix_solver_hooks(
        nside=nside,
        mode="pixel_bilinear",
    )
    operator = hooks.bind_target(product.target_render_spec())

    clean_cutout = operator.apply(truth)
    observed_cutout = clean_cutout + 0.03 * torch.randn_like(clean_cutout)

    initial_solution = torch.zeros_like(truth)
    fit = operator.solve_weighted_least_squares(
        observed_cutout,
        initial_solution=initial_solution,
        iterations=20,
        step_size=1.0e-3,
    )
    reconstructed_cutout = operator.apply(fit.solution)

    initial_rmse = math.sqrt(float(torch.mean(torch.square(observed_cutout)).item()))
    final_rmse = math.sqrt(
        float(torch.mean(torch.square(reconstructed_cutout - observed_cutout)).item())
    )
    support = operator.adjoint(torch.ones_like(observed_cutout))
    supported_pixels = int((support > 0.0).sum().item())

    print("Native spherical inverse example")
    print(f"nside={nside} npix={truth.numel()} target={tuple(observed_cutout.shape)}")
    print(f"supported HEALPix pixels from this cutout: {supported_pixels}")
    print(
        f"residual norm history (first 5): {[round(v, 5) for v in fit.residual_norm_history[:5]]}"
    )
    print(
        f"residual norm history (last 5): {[round(v, 5) for v in fit.residual_norm_history[-5:]]}"
    )
    print(f"initial image RMSE vs zero map: {initial_rmse:.5f}")
    print(f"final image RMSE after HEALPix solve: {final_rmse:.5f}")


if __name__ == "__main__":
    main()
