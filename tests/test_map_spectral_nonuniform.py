"""Tests for the native nonuniform spectral remapper."""

from __future__ import annotations

import torch

import torchsky
from torchsky.map.spectral_nonuniform import resample_nonuniform_spectral_2d


def _rotated_cd(
    scale_deg: float, rotation_deg: float
) -> tuple[float, float, float, float]:
    theta = torch.deg2rad(torch.tensor(float(rotation_deg), dtype=torch.float64)).item()
    cos_t = float(torch.cos(torch.tensor(theta)).item())
    sin_t = float(torch.sin(torch.tensor(theta)).item())
    return (
        -scale_deg * cos_t,
        scale_deg * sin_t,
        scale_deg * sin_t,
        scale_deg * cos_t,
    )


def _tpv_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    rotation_deg: float = 0.0,
    scale_deg: float = 0.01,
) -> dict:
    header = {}
    header["NAXIS"] = 2
    header["NAXIS1"] = size
    header["NAXIS2"] = size
    header["CTYPE1"] = "RA---TPV"
    header["CTYPE2"] = "DEC--TPV"
    header["CRPIX1"] = size / 2.0
    header["CRPIX2"] = size / 2.0
    header["CRVAL1"] = crval1
    header["CRVAL2"] = crval2
    cd11, cd12, cd21, cd22 = _rotated_cd(scale_deg, rotation_deg)
    header["CD1_1"] = cd11
    header["CD1_2"] = cd12
    header["CD2_1"] = cd21
    header["CD2_2"] = cd22
    header["PV1_0"] = 0.0
    header["PV1_1"] = 1.0
    header["PV1_2"] = -0.02
    header["PV1_4"] = 2.0e-4
    header["PV1_5"] = -3.0e-4
    header["PV1_6"] = 1.5e-4
    header["PV1_7"] = 2.0e-6
    header["PV1_39"] = 2.0e-11
    header["PV2_0"] = 0.0
    header["PV2_1"] = 0.02
    header["PV2_2"] = 1.0
    header["PV2_4"] = -1.0e-4
    header["PV2_5"] = 2.5e-4
    header["PV2_6"] = -2.0e-4
    header["PV2_7"] = -1.5e-6
    header["PV2_39"] = -1.0e-11
    return header


def test_nonuniform_spectral_identity_on_integer_grid_preserves_image() -> None:
    image = torch.arange(25, dtype=torch.float32).reshape(5, 5)
    yy, xx = torch.meshgrid(
        torch.arange(5, dtype=torch.float32),
        torch.arange(5, dtype=torch.float32),
        indexing="ij",
    )
    out = resample_nonuniform_spectral_2d(image, xx, yy, boundary_mode="zero")
    torch.testing.assert_close(out, image, atol=1e-5, rtol=1e-5)


def test_nonuniform_spectral_zero_boundary_masks_samples_outside_source() -> None:
    image = torch.zeros((6, 6), dtype=torch.float32)
    image[0, 0] = 1.0
    yy, xx = torch.meshgrid(
        torch.arange(6, dtype=torch.float32),
        torch.arange(6, dtype=torch.float32),
        indexing="ij",
    )
    out = resample_nonuniform_spectral_2d(
        image,
        xx + 1.0,
        yy,
        boundary_mode="zero",
    )
    assert float(out.abs().max()) < 1e-5


def test_nonuniform_spectral_periodic_mode_wraps_coordinates() -> None:
    image = torch.zeros((6, 6), dtype=torch.float32)
    image[0, 0] = 1.0
    yy, xx = torch.meshgrid(
        torch.arange(6, dtype=torch.float32),
        torch.arange(6, dtype=torch.float32),
        indexing="ij",
    )
    out = resample_nonuniform_spectral_2d(
        image,
        xx + 1.0,
        yy,
        boundary_mode="periodic",
    )
    assert float(out[:, -1].abs().max()) > 0.5


def test_nonuniform_spectral_tpv_identity_coordinate_grid_reconstructs_image() -> None:
    size = 28
    header = dict(_tpv_header(size, crval1=180.0, crval2=2.0))
    wcs = torchsky.WCS(header)
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    image = 0.8 * torch.exp(
        -((xx - 9.0) ** 2 + (yy - 13.5) ** 2) / (2.0 * 3.0**2)
    ) + 0.4 * torch.exp(-((xx - 18.0) ** 2 + (yy - 17.0) ** 2) / (2.0 * 2.0**2))
    ra, dec = wcs.pixel_to_world(
        xx.reshape(-1).to(torch.float64), yy.reshape(-1).to(torch.float64)
    )
    x_src, y_src = wcs.world_to_pixel(ra, dec)
    out = resample_nonuniform_spectral_2d(
        image,
        x_src.reshape_as(xx).to(torch.float32),
        y_src.reshape_as(yy).to(torch.float32),
        boundary_mode="zero",
    )
    torch.testing.assert_close(out, image, atol=0.03, rtol=0.03)
