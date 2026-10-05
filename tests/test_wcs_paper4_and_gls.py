from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsky.wcs import WCS, DistortionLookupTable


def test_gls_global_sinusoidal_forward_inverse_and_sfl_equivalence() -> None:
    hdr_gls = {
        "NAXIS": 2,
        "NAXIS1": 64,
        "NAXIS2": 64,
        "CTYPE1": "RA---GLS",
        "CTYPE2": "DEC--GLS",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRVAL1": 120.0,
        "CRVAL2": 25.0,
        "CDELT1": -0.05,
        "CDELT2": 0.05,
    }
    # WCSLIB celfix translates GLS with (CRVAL1=alpha0, CRVAL2=delta0) into
    # equatorial SFL with CRVAL1=alpha0, CRVAL2=0 and CRPIX2 shifted by -delta0 / CDELT2
    # (or equivalently SFL with PV1_0=1.0, PV1_2=delta0 and unshifted CRPIX2).
    hdr_sfl = {
        "NAXIS": 2,
        "NAXIS1": 64,
        "NAXIS2": 64,
        "CTYPE1": "RA---SFL",
        "CTYPE2": "DEC--SFL",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0 - 25.0 / 0.05,
        "CRVAL1": 120.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.05,
        "CDELT2": 0.05,
    }
    wcs_gls = WCS(hdr_gls)
    wcs_sfl = WCS(hdr_sfl)

    y, x = torch.meshgrid(
        torch.linspace(4.0, 56.0, 9, dtype=torch.float64),
        torch.linspace(4.0, 56.0, 9, dtype=torch.float64),
        indexing="ij",
    )
    ra_gls, dec_gls = wcs_gls.pixel_to_world(x, y)
    ra_sfl, dec_sfl = wcs_sfl.pixel_to_world(x, y)
    torch.testing.assert_close(ra_gls, ra_sfl, atol=1e-9, rtol=0.0)
    torch.testing.assert_close(dec_gls, dec_sfl, atol=1e-9, rtol=0.0)

    x_back, y_back = wcs_gls.world_to_pixel(ra_gls, dec_gls)
    torch.testing.assert_close(x_back, x, atol=1e-9, rtol=0.0)
    torch.testing.assert_close(y_back, y, atol=1e-9, rtol=0.0)


def test_paper4_distortion_lookup_table_roundtrip_and_astropy_parity() -> None:
    astropy_wcs = pytest.importorskip("astropy.wcs")

    # Build smooth 2D lookup tables for DET2IM and CPDIS
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 9, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, 11, dtype=torch.float32),
        indexing="ij",
    )
    det_dx = 0.15 * grid_x - 0.05 * grid_y
    det_dy = 0.10 * grid_y + 0.04 * grid_x
    cp_dx = 0.25 * (grid_x**2 - grid_y**2)
    cp_dy = 0.20 * (grid_x * grid_y)

    # Covers a 64x64 detector with 8-pixel spacing:
    # 11 columns: CRPIX1=1, CRVAL1=1, CDELT1=8 -> spans 1..81 in 1-based FITS coords
    # 9 rows: CRPIX2=1, CRVAL2=1, CDELT2=8 -> spans 1..65 in 1-based FITS coords
    det2im1 = DistortionLookupTable(
        data=det_dx, crpix=(1.0, 1.0), crval=(1.0, 1.0), cdelt=(8.0, 8.0)
    )
    det2im2 = DistortionLookupTable(
        data=det_dy, crpix=(1.0, 1.0), crval=(1.0, 1.0), cdelt=(8.0, 8.0)
    )
    # CPDIS is indexed in intermediate pixel coordinates (relative to main CRPIX when CRVAL=(0,0))
    cpdis1 = DistortionLookupTable(
        data=cp_dx, crpix=(6.0, 5.0), crval=(0.0, 0.0), cdelt=(8.0, 8.0)
    )
    cpdis2 = DistortionLookupTable(
        data=cp_dy, crpix=(6.0, 5.0), crval=(0.0, 0.0), cdelt=(8.0, 8.0)
    )

    hdr = {
        "NAXIS": 2,
        "NAXIS1": 64,
        "NAXIS2": 64,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRVAL1": 50.0,
        "CRVAL2": -20.0,
        "CD1_1": -1.0e-4,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0e-4,
    }
    ts_wcs = WCS(
        hdr,
        det2im1=det2im1,
        det2im2=det2im2,
        cpdis1=cpdis1,
        cpdis2=cpdis2,
    )

    # Compare against Astropy WCS with DistortionLookupTable
    aw = astropy_wcs.WCS(hdr)
    aw.det2im1 = astropy_wcs.DistortionLookupTable(
        det_dx.numpy(), (1.0, 1.0), (1.0, 1.0), (8.0, 8.0)
    )
    aw.det2im2 = astropy_wcs.DistortionLookupTable(
        det_dy.numpy(), (1.0, 1.0), (1.0, 1.0), (8.0, 8.0)
    )
    aw.cpdis1 = astropy_wcs.DistortionLookupTable(
        cp_dx.numpy(), (6.0, 5.0), (0.0, 0.0), (8.0, 8.0)
    )
    aw.cpdis2 = astropy_wcs.DistortionLookupTable(
        cp_dy.numpy(), (6.0, 5.0), (0.0, 0.0), (8.0, 8.0)
    )

    py, px = torch.meshgrid(
        torch.linspace(8.0, 48.0, 6, dtype=torch.float64),
        torch.linspace(8.0, 48.0, 6, dtype=torch.float64),
        indexing="ij",
    )
    ra_ts, dec_ts = ts_wcs.pixel_to_world(px, py)
    ra_ap, dec_ap = aw.all_pix2world(px.numpy(), py.numpy(), 0)

    np.testing.assert_allclose(ra_ts.numpy(), ra_ap, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(dec_ts.numpy(), dec_ap, atol=1e-8, rtol=0.0)

    # Inverse world_to_pixel fixed-point iteration
    px_back, py_back = ts_wcs.world_to_pixel(ra_ts, dec_ts)
    torch.testing.assert_close(px_back, px, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(py_back, py, atol=1e-6, rtol=0.0)
