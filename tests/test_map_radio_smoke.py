"""Smoke tests for radio-style FITS headers (TP / image-plane mosaic path).

Uses **synthetic** FITS written at runtime via ``torchfits.write`` (no astropy/reproject).
Exercises ``primary_beam_path`` weighting and ``ImageObservation.hdu=None`` auto payload
behavior where applicable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torchsky._cpp")

from torchsky.map.image_plane_coadd import coadd_image_plane
from torchsky.map.planning import plan_image_plane_target
from torchsky.map.observation import ImageObservation
import torchfits


def _radio_tp_header_dict(*, size: int = 24, crval2: float = -10.0) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": 45.0,
        "CRVAL2": crval2,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
        "BUNIT": "JY/BEAM",
        "BMAJ": 0.02,
        "BMIN": 0.018,
        "BPA": 15.0,
    }


def test_radio_stokes_cube_header_write_read_roundtrip(tmp_path: Path) -> None:
    """Small I-only cube with RESTFRQ / SPECSYS (mm-style metadata) round-trips via FITS."""
    size = 12
    h = _radio_tp_header_dict(size=size)
    h["NAXIS"] = 3
    h["NAXIS3"] = 1
    h["CTYPE3"] = "STOKES"
    h["CRPIX3"] = 1.0
    h["CRVAL3"] = 1.0
    h["CDELT3"] = 1.0
    h["RESTFRQ"] = 1.420405752e9
    h["SPECSYS"] = "LSRK"
    cube = torch.ones(1, size, size, dtype=torch.float32)
    path = tmp_path / "stokes_i.fits"
    torchfits.write(str(path), cube, header=h, overwrite=True)
    back, hdr = torchfits.read(str(path), return_header=True)
    assert tuple(int(back.shape[i]) for i in range(back.ndim)) == (1, size, size)
    assert int(hdr.get("NAXIS", 0)) == 3
    assert float(hdr["RESTFRQ"]) == pytest.approx(1.420405752e9)


def test_coadd_synthetic_radio_pair_smoke(tmp_path: Path) -> None:
    size = 24
    yy, xx = np.mgrid[0:size, 0:size]
    g = np.exp(-((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / 8.0).astype(np.float32)

    h1 = _radio_tp_header_dict(size=size, crval2=-10.0)
    h2 = _radio_tp_header_dict(size=size, crval2=-10.01)
    p1 = tmp_path / "tp_a.fits"
    p2 = tmp_path / "tp_b.fits"
    torchfits.write(str(p1), torch.from_numpy(g), header=h1, overwrite=True)
    torchfits.write(str(p2), torch.from_numpy(g * 0.9), header=h2, overwrite=True)

    obs = [
        ImageObservation(path=str(p1), hdu=None, weight=1.0),
        ImageObservation(path=str(p2), hdu=None, weight=1.0),
    ]
    plan = plan_image_plane_target(obs)
    res = coadd_image_plane(
        obs,
        target_header=plan.header,
        resampling_kernel="bilinear",
        value_semantics="surface_brightness",
        stacking_method="weighted_mean",
    )
    assert res.data.shape == (size, size)
    assert torch.isfinite(res.data).all()
    assert torch.isfinite(res.weight).all()
    wsum = float(res.weight.sum().item())
    assert wsum > 0.0


def test_primary_beam_multiplies_weights(tmp_path: Path) -> None:
    size = 16
    flat = np.ones((size, size), dtype=np.float32)
    h = _radio_tp_header_dict(size=size)
    img_path = tmp_path / "img.fits"
    pb_path = tmp_path / "pb.fits"
    yy, xx = np.mgrid[0:size, 0:size]
    pb = np.exp(-((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / 18.0).astype(
        np.float32
    )
    torchfits.write(str(img_path), torch.from_numpy(flat), header=h, overwrite=True)
    torchfits.write(str(pb_path), torch.from_numpy(pb), header=h, overwrite=True)

    obs = [
        ImageObservation(
            path=str(img_path),
            hdu=None,
            weight=1.0,
            primary_beam_path=str(pb_path),
            primary_beam_hdu=None,
        )
    ]
    plan = plan_image_plane_target(obs)
    res = coadd_image_plane(
        obs,
        target_header=plan.header,
        resampling_kernel="bilinear",
    )
    assert res.weight.shape == (size, size)
    cy, cx = size // 2, size // 2
    assert float(res.weight[cy, cx].item()) > float(res.weight[0, 0].item())
