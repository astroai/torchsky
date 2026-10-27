"""Coadd bounded synthetic detector observations with exact square drizzle.

This deterministic, network-free flagship renders two compact detector images
onto one declared target WCS, uses the native ``drizzle_square`` kernel with
flux semantics, and validates total flux, centroid, and non-empty bounded
support without importing an external comparator.
"""

from __future__ import annotations

import json

import torch

from torchsky.map import Observation, coadd


def _header(crpix1: float, crpix2: float) -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 32,
        "NAXIS2": 32,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": crpix1,
        "CRPIX2": crpix2,
        "CRVAL1": 42.0,
        "CRVAL2": -12.0,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }


def _detector_source(x0: float, y0: float) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(32, dtype=torch.float32),
        torch.arange(32, dtype=torch.float32),
        indexing="ij",
    )
    radius_squared = (x - x0) ** 2 + (y - y0) ** 2
    image = torch.where(
        radius_squared <= 16.0,
        torch.exp(-0.5 * radius_squared / 1.4**2),
        torch.zeros_like(radius_squared),
    )
    return image / image.sum()


def _centroid(image: torch.Tensor) -> tuple[float, float]:
    y, x = torch.meshgrid(
        torch.arange(image.shape[0], dtype=image.dtype),
        torch.arange(image.shape[1], dtype=image.dtype),
        indexing="ij",
    )
    total = image.sum()
    return float((image * x).sum().item() / total.item()), float(
        (image * y).sum().item() / total.item()
    )


def main() -> None:
    target = _header(16.5, 16.5)
    shifts = ((-0.35, 0.2), (0.4, -0.25))
    observations = []
    for dx, dy in shifts:
        detector = _detector_source(15.5 + dx, 15.5 + dy)
        observations.append(
            Observation(
                data=detector,
                wcs=_header(16.5 + dx, 16.5 + dy),
                weight_map=torch.ones_like(detector),
                weight_is_inverse_variance=True,
            )
        )

    result = coadd(
        observations,
        target,
        kernel="drizzle_square",
        pixfrac=1.0,
        semantics="flux",
        method="weighted_mean",
        device="cpu",
    )
    support = result.weight > 0.0
    measured_flux = float(result.data[support].sum().item())
    centroid_x, centroid_y = _centroid(torch.where(support, result.data, 0.0))
    support_pixels = int(support.sum().item())
    assert abs(measured_flux - 1.0) < 2.0e-3
    assert abs(centroid_x - 15.5) < 2.0e-2
    assert abs(centroid_y - 15.5) < 2.0e-2
    assert 0 < support_pixels <= result.data.numel()

    print(
        json.dumps(
            {
                "centroid_pixel": [centroid_x, centroid_y],
                "example": "map_drizzle_flagship",
                "flux": measured_flux,
                "kernel": "drizzle_square",
                "pixfrac": 1.0,
                "semantics": "flux",
                "status": "pass",
                "support_pixels": support_pixels,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
