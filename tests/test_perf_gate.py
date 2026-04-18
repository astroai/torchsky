"""Loose wall-time gates for a few hot paths (CI-friendly; override via env)."""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchsky.cpp")

import torchsky  # noqa: E402
from torchsky.wcs import healpix  # noqa: E402

_FIXTURE = Path(__file__).resolve().parent / "data" / "radio_mm_stub.fits"


def _median_seconds(fn, *, warmup: int = 1, iters: int = 3) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(statistics.median(samples))


def test_gate_healpix_ang2pix_bulk() -> None:
    """~100k lon/lat → nest indices should stay cheap on CI runners."""
    nside = 1024
    n = 100_000
    lon = torch.linspace(0.0, 359.0, steps=n, dtype=torch.float64)
    lat = torch.linspace(-60.0, 60.0, steps=n, dtype=torch.float64)

    def run() -> None:
        healpix.ang2pix(nside, lon, lat, nest=True, lonlat=True)

    max_s = float(os.environ.get("TORCHSKY_PERF_GATE_ANG2PIX_S", "2.5"))
    assert _median_seconds(run) < max_s


def test_gate_fits_image_read_stub() -> None:
    assert _FIXTURE.is_file(), f"missing {_FIXTURE}"

    def run() -> None:
        torchsky.read(str(_FIXTURE), return_header=True)

    max_s = float(os.environ.get("TORCHSKY_PERF_GATE_READ_FITS_S", "3.0"))
    assert _median_seconds(run) < max_s


def test_gate_healpix_interp_weights_bulk() -> None:
    nside = 512
    n = 50_000
    lon = torch.linspace(10.0, 20.0, steps=n, dtype=torch.float64)
    lat = torch.linspace(-5.0, 5.0, steps=n, dtype=torch.float64)

    def run() -> None:
        healpix.get_interp_weights(nside, lon, lat, nest=True, lonlat=True)

    max_s = float(os.environ.get("TORCHSKY_PERF_GATE_INTERP_WEIGHTS_S", "3.0"))
    assert _median_seconds(run) < max_s
