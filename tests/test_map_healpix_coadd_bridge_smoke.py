"""HEALPix ↔ TAN bridge: torchsky coadd on the TAN grid after ``reproject`` round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS
import torch

pytest.importorskip("torchsky.cpp")
pytest.importorskip("reproject")

from torchsky.map import ImageObservation, coadd_image_plane, plan_image_plane_target  # noqa: E402
import torchfits


def test_torchsky_coadd_after_healpix_roundtrip_matches_baseline(
    tmp_path: Path,
) -> None:
    from reproject import reproject_from_healpix, reproject_to_healpix

    size = 96
    hdr = {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": 55.0,
        "CRVAL2": 12.0,
        "CDELT1": -0.008,
        "CDELT2": 0.008,
    }
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(
        -((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2.0 * 14.0**2)
    ).astype(np.float32)
    hp_map, _ = reproject_to_healpix(
        (blob, WCS(hdr)),
        "icrs",
        nside=256,
        order="bilinear",
    )
    arr, fp = reproject_from_healpix((hp_map, "icrs"), WCS(hdr), nested=False, shape_out=(hdr['NAXIS2'], hdr['NAXIS1']))
    assert float(fp.max()) > 0.5
    path = tmp_path / "from_hp.fits"
    torchfits.write(
        str(path), torch.from_numpy(arr.astype(np.float32)), header=hdr, overwrite=True
    )
    obs = [ImageObservation(str(path))]
    plan = plan_image_plane_target(obs)
    res = coadd_image_plane(
        obs,
        target_header=plan.header,
        resampling_kernel="bilinear",
    )
    mask = fp > 0.5
    delta = np.abs(res.data.detach().cpu().numpy()[mask] - arr.astype(np.float64)[mask])
    assert float(np.nanmax(delta)) < 0.02
