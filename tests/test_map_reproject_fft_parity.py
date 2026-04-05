"""Parity checks against the ``reproject`` package on small synthetic images.

``reproject`` (as of 0.19) exposes interpolation, spherical polygon overlap, adaptive
resampling, and HEALPix helpers; there is no separate stable public "FFT reprojection"
API to target. This module locks **bilinear** agreement between ``reproject.reproject_interp`` and
``torchsky.map.coadd_image_plane``, plus a looser **bicubic** spot-check (kernels differ
from Astropy’s polynomial in detail).

Optional HEALPix coverage: a **large enough** tangent-plane cutout plus sufficiently
fine ``nside`` yields a non-degenerate footprint and a bounded bilinear round-trip
error (see ``test_reproject_healpix_roundtrip_large_field``). The tiny-cutout test
below may still skip when coverage is sparse.

Requires: ``torchsky.cpp`` plus ``reproject`` (and its ``astropy`` dependency), e.g. from
``torchsky[bench]`` — this module is skipped when ``reproject`` is not installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torchsky.cpp")
pytest.importorskip("reproject")
from astropy.io import fits  # noqa: E402

from torchsky.map import (  # noqa: E402
    ImageObservation,
    coadd_image_plane,
    plan_image_plane_target,
)


def _tan_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    cdelt: float = 0.02,
) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CDELT1": -cdelt,
        "CDELT2": cdelt,
    }


def test_torchsky_bilinear_matches_reproject_interp_on_shifted_grid() -> None:
    from reproject import reproject_interp

    size = 32
    src_hdr = _tan_header(size, crval1=180.0, crval2=0.0)
    tgt_hdr = _tan_header(size, crval1=180.05, crval2=0.02)
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(
        -((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2.0 * 4.0**2)
    ).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "src.fits"
        fits.writeto(path, blob, fits.Header(src_hdr))
        obs = [ImageObservation(path=str(path))]
        plan = plan_image_plane_target(obs, header_overrides=tgt_hdr)
        ts = coadd_image_plane(
            obs,
            target_header=plan.header,
            resampling_kernel="bilinear",
        )
        ts_np = ts.data.detach().cpu().numpy()

    rp, footprint = reproject_interp((blob, fits.Header(src_hdr)), fits.Header(tgt_hdr))
    mask = footprint > 0.5
    assert bool(mask.any())
    delta = np.abs(ts_np[mask] - rp[mask])
    assert float(delta.max()) < 0.01


def test_torchsky_bicubic_reasonably_close_to_reproject_interp_on_shifted_grid() -> None:
    from reproject import reproject_interp

    size = 32
    src_hdr = _tan_header(size, crval1=180.0, crval2=0.0)
    tgt_hdr = _tan_header(size, crval1=180.06, crval2=0.025)
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(
        -((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2.0 * 5.0**2)
    ).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "src.fits"
        fits.writeto(path, blob, fits.Header(src_hdr))
        obs = [ImageObservation(path=str(path))]
        plan = plan_image_plane_target(obs, header_overrides=tgt_hdr)
        ts = coadd_image_plane(
            obs,
            target_header=plan.header,
            resampling_kernel="bicubic",
        )
        ts_np = ts.data.detach().cpu().numpy()

    rp, footprint = reproject_interp(
        (blob, fits.Header(src_hdr)),
        fits.Header(tgt_hdr),
        order="bicubic",
    )
    mask = footprint > 0.5
    assert bool(mask.any())
    delta = np.abs(ts_np[mask] - rp[mask])
    assert float(delta.max()) < 0.06


def test_reproject_healpix_roundtrip_skipped_when_no_coverage() -> None:
    """Document HEALPix path; skip if the tiny footprint does not map to the target."""
    from reproject import reproject_from_healpix, reproject_to_healpix

    size = 32
    src_hdr = _tan_header(size, crval1=180.0, crval2=0.0)
    tgt_hdr = _tan_header(size, crval1=180.05, crval2=0.02)
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(
        -((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2.0 * 4.0**2)
    ).astype(np.float32)

    hp, _ = reproject_to_healpix(
        (blob, fits.Header(src_hdr)),
        "icrs",
        nside=256,
        order="bilinear",
    )
    _arr, fp = reproject_from_healpix((hp, "icrs"), fits.Header(tgt_hdr), nested=False)
    if float(fp.max()) < 0.5:
        pytest.skip("HEALPix round-trip produced empty footprint for this cutout")


def test_reproject_healpix_roundtrip_large_field() -> None:
    """Bilinear TAN → HEALPix → TAN restores the map within HEALPix discretization error."""
    from reproject import reproject_from_healpix, reproject_to_healpix

    size = 128
    cdelt = 0.01
    src_hdr = _tan_header(size, crval1=45.0, crval2=10.0, cdelt=cdelt)
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(
        -((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2.0 * 20.0**2)
    ).astype(np.float32)

    hp, _ = reproject_to_healpix(
        (blob, fits.Header(src_hdr)),
        "icrs",
        nside=512,
        order="bilinear",
    )
    arr, fp = reproject_from_healpix((hp, "icrs"), fits.Header(src_hdr), nested=False)
    assert float(fp.max()) > 0.5
    mask = fp > 0.5
    delta = np.abs(arr[mask].astype(np.float64) - blob[mask].astype(np.float64))
    assert float(np.nanmax(delta)) < 0.12


def test_reproject_adaptive_runs_and_is_finite_on_shifted_tan_grid() -> None:
    """``reproject_adaptive`` (distinct from bilinear interp) stays on the smoke matrix."""
    from astropy.wcs import WCS as AstropyWCS

    from reproject import reproject_adaptive

    size = 28
    src_hdr = _tan_header(size, crval1=180.0, crval2=0.0)
    tgt_hdr = _tan_header(size, crval1=180.04, crval2=0.03)
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(
        -((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2.0 * 5.0**2)
    ).astype(np.float32)
    out_wcs = AstropyWCS(fits.Header(tgt_hdr))
    arr, fp = reproject_adaptive(
        (blob, fits.Header(src_hdr)),
        out_wcs,
        shape_out=(size, size),
    )
    assert arr.shape == (size, size)
    assert np.isfinite(arr[fp > 0.1]).all()
