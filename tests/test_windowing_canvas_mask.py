"""Tests for torchsky.map.windowing.canvas_footprint_mask."""

from __future__ import annotations

import numpy as np
import torch
from astropy.wcs import WCS

from torchsky.map.windowing import canvas_footprint_mask


def _tan_header(h: int = 512, w: int = 512) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": w,
        "NAXIS2": h,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (w + 1) / 2,
        "CRPIX2": (h + 1) / 2,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CD1_1": -3e-5,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 3e-5,
    }


def _world_at_pixels(header: dict, pixels: np.ndarray):
    world = WCS(header).all_pix2world(pixels[:, 0], pixels[:, 1], 0)
    return (
        torch.tensor(np.asarray(world[0], dtype=np.float64)),
        torch.tensor(np.asarray(world[1], dtype=np.float64)),
    )


def test_inside_outside_border_and_nonfinite() -> None:
    header = _tan_header()
    pixels = np.array(
        [
            [100.0, 100.0],  # interior
            [600.0, 256.0],  # beyond right edge
            [256.0, -20.0],  # below bottom edge
            [0.0, 0.0],      # border pixel center -> inside
            [511.0, 511.0],  # far corner pixel center -> inside
        ],
        dtype=np.float64,
    )
    ra, dec = _world_at_pixels(header, pixels)
    ra[3] = float("nan")

    mask = canvas_footprint_mask(ra, dec, template_header=header)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, False, False, False, True]


def test_chunking_matches_single_pass() -> None:
    header = _tan_header(64, 64)
    generator = np.random.default_rng(11)
    pixels = generator.uniform(-40, 104, size=(97, 2))
    ra, dec = _world_at_pixels(header, pixels)
    ra[:5] = float("nan")
    dec[-3:] = np.inf

    full = canvas_footprint_mask(ra, dec, template_header=header)
    chunked = canvas_footprint_mask(ra, dec, template_header=header, chunk=7)
    assert torch.equal(full, chunked)


def test_shape_and_device_preserved() -> None:
    header = _tan_header(16, 16)
    ra = torch.full((3, 4), 150.0, dtype=torch.float64)
    dec = torch.full((3, 4), 2.0, dtype=torch.float64)
    mask = canvas_footprint_mask(ra, dec, template_header=header)
    assert mask.shape == (3, 4)
    assert mask.device == ra.device


def test_runtime_does_not_import_astropy(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def _no_astropy(name, *args, **kwargs):
        if name.startswith("astropy"):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_astropy)
    mask = canvas_footprint_mask(
        torch.tensor([150.0]), torch.tensor([2.0]), template_header=_tan_header(8, 8)
    )
    assert mask.tolist() == [True]
