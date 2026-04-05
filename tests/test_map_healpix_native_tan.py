"""HEALPix → TAN without ``reproject``: :func:`sample_healpix_map_on_tan_header` parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from astropy.io import fits

pytest.importorskip("torchsky.cpp")
pytest.importorskip("reproject")

from reproject import reproject_from_healpix, reproject_to_healpix  # noqa: E402

import torchsky  # noqa: E402
from torchsky.map import (  # noqa: E402
    healpix_nside_from_map_len,
    sample_healpix_map_on_tan_header,
)


def test_healpix_nside_from_map_len_roundtrip() -> None:
    assert healpix_nside_from_map_len(12 * 64 * 64) == 64


def test_sample_healpix_map_on_tan_header_near_reproject_from_healpix() -> None:
    size = 64
    hdr = {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": 142.0,
        "CRVAL2": -35.0,
        "CDELT1": -0.02,
        "CDELT2": 0.02,
    }
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(
        -((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (2.0 * 10.0**2)
    ).astype(np.float32)
    nside = 128
    hp_map, _ = reproject_to_healpix(
        (blob, fits.Header(hdr)),
        "icrs",
        nside=nside,
        order="bilinear",
    )
    rp, fp = reproject_from_healpix((hp_map, "icrs"), fits.Header(hdr), nested=False)
    ts = sample_healpix_map_on_tan_header(
        torch.from_numpy(hp_map.astype(np.float64)),
        nside=nside,
        nest=False,
        target_header=hdr,
    )
    mask = fp > 0.4
    d = np.abs(ts.detach().cpu().numpy()[mask] - rp.astype(np.float64)[mask])
    assert float(np.nanmax(d)) < 0.03
