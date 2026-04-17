"""Intermediate-size AIT and ZPN parity vs Astropy (benchmark deficit focus)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs.core import WCS as TorchWCS


def _ra_wrap_delta(ra_a: np.ndarray, ra_b: np.ndarray) -> np.ndarray:
    return ((ra_a - ra_b + 180.0) % 360.0) - 180.0


def test_ait_intermediate_grid_matches_astropy() -> None:
    """AIT at 512x384 with a denser interior grid than small parity tests."""
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 512
    header["NAXIS2"] = 384
    header["CRPIX1"] = 256.5
    header["CRPIX2"] = 192.5
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CD1_1"] = -1.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0
    header["CTYPE1"] = "RA---AIT"
    header["CTYPE2"] = "DEC--AIT"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"

    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))

    x = np.linspace(80.0, 432.0, 24, dtype=np.float64)
    y = np.linspace(60.0, 324.0, 18, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    x_flat, y_flat = xx.ravel(), yy.ravel()

    ra_ast, dec_ast = awcs.all_pix2world(x_flat, y_flat, 0)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x_flat), torch.from_numpy(y_flat), origin=0
    )
    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.sum(valid) > 100

    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]
    np.testing.assert_allclose(dra, 0.0, atol=1e-8)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-8)


def test_zpn_intermediate_grid_matches_astropy() -> None:
    """ZPN linear-ish header at 512x384; interior grid away from image borders."""
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 512
    header["NAXIS2"] = 384
    header["CRPIX1"] = 256.5
    header["CRPIX2"] = 192.5
    header["CRVAL1"] = 180.0
    header["CRVAL2"] = 0.0
    header["CD1_1"] = -2.8e-4
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 2.8e-4
    header["CTYPE1"] = "RA---ZPN"
    header["CTYPE2"] = "DEC--ZPN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["PV2_1"] = 1.0

    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))

    x = np.linspace(96.0, 416.0, 20, dtype=np.float64)
    y = np.linspace(72.0, 312.0, 16, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    x_flat, y_flat = xx.ravel(), yy.ravel()

    ra_ast, dec_ast = awcs.all_pix2world(x_flat, y_flat, 0)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x_flat), torch.from_numpy(y_flat), origin=0
    )
    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.sum(valid) > 100

    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]
    np.testing.assert_allclose(dra, 0.0, atol=1e-7)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-7)


@pytest.mark.parametrize("origin", [0, 1])
def test_ait_intermediate_inverse_matches_astropy(origin: int) -> None:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 512
    header["NAXIS2"] = 384
    header["CRPIX1"] = 256.5
    header["CRPIX2"] = 192.5
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CD1_1"] = -1.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0
    header["CTYPE1"] = "RA---AIT"
    header["CTYPE2"] = "DEC--AIT"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"

    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    x = np.linspace(100.0, 412.0, 12, dtype=np.float64)
    y = np.linspace(80.0, 304.0, 10, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    x_flat, y_flat = xx.ravel(), yy.ravel()
    ra_ast, dec_ast = awcs.all_pix2world(x_flat, y_flat, origin)
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    ra_v, dec_v = ra_ast[valid], dec_ast[valid]
    x_ast, y_ast = awcs.all_world2pix(ra_v, dec_v, origin)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_v), torch.from_numpy(dec_v), origin=origin
    )
    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=1e-5)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=1e-5)
