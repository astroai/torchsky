"""Shared helpers for the map_image_plane_support test modules."""

from __future__ import annotations
from pathlib import Path
import torch
import torchsky


def _simple_header() -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 4.0,
        "CRPIX2": 4.0,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }


def _astropy_sip_fixture() -> Path:
    import astropy

    return (
        Path(astropy.__file__).resolve().parent
        / "nddata"
        / "tests"
        / "data"
        / "sip-wcs.fits"
    )


def _sky_from_header(
    header: dict,
    *,
    amplitude: float = 1.0,
    center_ra_deg: float | None = None,
    center_dec_deg: float | None = None,
) -> torch.Tensor:
    wcs = torchsky.WCS(header)
    height = int(header["NAXIS2"])
    width = int(header["NAXIS1"])
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    ra_deg, dec_deg = wcs.pixel_to_world(x, y)
    if center_ra_deg is None:
        center_ra_deg = float(header["CRVAL1"])
    if center_dec_deg is None:
        center_dec_deg = float(header["CRVAL2"])
    dra = ((ra_deg - center_ra_deg + 180.0) % 360.0) - 180.0
    ddec = dec_deg - center_dec_deg
    sky = (
        3.0
        + amplitude * torch.exp(-((dra / 0.035) ** 2 + (ddec / 0.028) ** 2))
        + 0.15 * dra
        - 0.08 * ddec
        + 0.04 * dra * ddec
    )
    return sky.to(dtype=torch.float32)
