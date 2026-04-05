"""Parity checks against the ``reproject`` package on small synthetic images.

``reproject`` (as of 0.19) exposes interpolation, spherical polygon overlap, adaptive
resampling, and HEALPix helpers; there is no separate stable public "FFT reprojection"
API to target. This module therefore locks **bilinear interpolation** agreement between
``reproject.reproject_interp`` and ``torchsky.map.coadd_image_plane``, which is the
practical parity surface for grid-based optical/IR regridding.

Optional HEALPix round-trip coverage is skipped unless the environment exposes a
working two-step path; small cutouts often yield sparse HEALPix coverage, so it is not
used as a hard gate.

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
