"""Loose wall-time gates for a few hot paths (CI-friendly; override via env)."""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

pytestmark = pytest.mark.performance

pytest.importorskip("torchsky._cpp")

import torchsky  # noqa: E402
from torchsky.sphere import healpix  # noqa: E402


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


def test_gate_healpix_interp_weights_bulk() -> None:
    nside = 512
    n = 50_000
    lon = torch.linspace(10.0, 20.0, steps=n, dtype=torch.float64)
    lat = torch.linspace(-5.0, 5.0, steps=n, dtype=torch.float64)

    def run() -> None:
        healpix.get_interp_weights(nside, lon, lat, nest=True, lonlat=True)

    max_s = float(os.environ.get("TORCHSKY_PERF_GATE_INTERP_WEIGHTS_S", "3.0"))
    assert _median_seconds(run) < max_s


def _tan_wcs() -> torchsky.WCS:
    return torchsky.WCS(
        {
            "NAXIS": 2,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 2048.0,
            "CRPIX2": 2048.0,
            "CRVAL1": 180.0,
            "CRVAL2": 0.5,
            "CDELT1": -0.0002777778,
            "CDELT2": 0.0002777778,
        }
    )


def test_gate_wcs_tan_pixel_to_world_bulk() -> None:
    """Common TAN vector transforms should stay cheap for map/catalog workflows."""
    wcs = _tan_wcs()
    n = int(os.environ.get("TORCHSKY_PERF_GATE_WCS_N", "200000"))
    x = torch.linspace(1500.0, 2500.0, steps=n, dtype=torch.float64)
    y = torch.linspace(1500.0, 2500.0, steps=n, dtype=torch.float64)
    pixels = torch.stack([x, y], dim=-1)

    def run() -> None:
        wcs.pixel_to_world(pixels)

    max_s = float(os.environ.get("TORCHSKY_PERF_GATE_WCS_P2W_S", "2.0"))
    assert _median_seconds(run) < max_s


def test_gate_wcs_tan_world_to_pixel_bulk() -> None:
    """Common TAN inverse transforms should stay cheap for reprojection workflows."""
    wcs = _tan_wcs()
    n = int(os.environ.get("TORCHSKY_PERF_GATE_WCS_N", "200000"))
    x = torch.linspace(1500.0, 2500.0, steps=n, dtype=torch.float64)
    y = torch.linspace(1500.0, 2500.0, steps=n, dtype=torch.float64)
    world = wcs.pixel_to_world(torch.stack([x, y], dim=-1))

    def run() -> None:
        wcs.world_to_pixel(world)

    max_s = float(os.environ.get("TORCHSKY_PERF_GATE_WCS_W2P_S", "2.0"))
    assert _median_seconds(run) < max_s


def test_gate_crossmatch_sky_scales_with_candidates_not_pairs() -> None:
    """Uniform all-sky crossmatch prunes to ~O(N) candidates and stays fast."""
    from torchsky.catalogs import core as catalogs_core
    from torchsky.catalogs import crossmatch_sky

    n = 200_000
    g = torch.Generator().manual_seed(5)
    ra_l = torch.rand(n, generator=g, dtype=torch.float64) * 360.0
    dec_l = torch.rad2deg(
        torch.asin(2.0 * torch.rand(n, generator=g, dtype=torch.float64) - 1.0)
    )
    ra_r = (ra_l + torch.randn(n, generator=g, dtype=torch.float64) * 1e-4) % 360.0
    dec_r = (dec_l + torch.randn(n, generator=g, dtype=torch.float64) * 1e-4).clamp(
        -90.0, 90.0
    )
    radius = torch.full((n,), 1.0, dtype=torch.float64)

    _, stats = catalogs_core._crossmatch_sky_strips(
        ra_l,
        dec_l,
        ra_r,
        dec_r,
        radius_arcsec=radius,
        left_chunk_size=65536,
        right_chunk_size=65536,
        find="all",
        return_stats=True,
    )
    # The candidate area per left row is ~6 r^2, so a few candidates per row at
    # most; an O(N*M) or per-pixel ring path would be orders of magnitude above.
    assert stats.pairs_evaluated < 20 * n

    def run() -> None:
        crossmatch_sky(ra_l, dec_l, ra_r, dec_r, radius_arcsec=1.0, find="best")

    max_s = float(os.environ.get("TORCHSKY_PERF_GATE_CROSSMATCH_S", "5.0"))
    assert _median_seconds(run, iters=1) < max_s


def test_gate_moc_range_algebra_scales_with_ranges() -> None:
    """MOC union/intersection/difference are sort+sweep merges, not per-cell loops.

    Cell count grows ~14x between the two sizes; a Python loop over cells took
    ~10-50 ms at the small size alone (see the CHANGELOG).
    """
    from torchsky.sphere import MOC

    def pair(radius: float, order: int):
        return (
            MOC.from_circle(30.0, 20.0, radius, order),
            MOC.from_circle(30.0 + 0.8 * radius, 20.0 + 0.5 * radius, radius, order),
        )

    a, b = pair(10.0, 10)  # ~1.4k cells
    big_a, big_b = pair(2.0, 16)  # ~18k cells
    assert big_a.uniq.numel() > 10 * a.uniq.numel()

    def run_small() -> None:
        a.union(b), a.intersection(b), a.difference(b)

    def run_big() -> None:
        big_a.union(big_b), big_a.intersection(big_b), big_a.difference(big_b)

    assert _median_seconds(run_small) < float(
        os.environ.get("TORCHSKY_PERF_GATE_MOC_OPS_S", "0.25")
    )
    assert _median_seconds(run_big) < float(
        os.environ.get("TORCHSKY_PERF_GATE_MOC_OPS_BIG_S", "1.0")
    )


def test_gate_moc_from_circle_hierarchical() -> None:
    from torchsky.sphere import MOC

    def run() -> None:
        MOC.from_circle(30.0, 20.0, 40.0, 10)

    max_s = float(os.environ.get("TORCHSKY_PERF_GATE_MOC_CIRCLE_S", "1.0"))
    assert _median_seconds(run) < max_s


def test_gate_sht_scalable_at_two_and_three_nside() -> None:
    """map2alm/alm2map at lmax=2*nside and 3*nside-1 stay on the ring path."""
    from torchsky.sphere import alm2map, map2alm

    nside = 128
    g = torch.Generator().manual_seed(0)
    m = torch.randn(12 * nside * nside, generator=g, dtype=torch.float64)
    for lmax in (2 * nside, 3 * nside - 1):
        alm: torch.Tensor | None = None

        def run() -> None:
            nonlocal alm
            alm = map2alm(m, nside=nside, lmax=lmax)
            alm2map(alm, nside=nside, lmax=lmax)

        assert _median_seconds(run, iters=1) < float(
            os.environ.get("TORCHSKY_PERF_GATE_SHT_S", "5.0")
        )
        assert alm is not None and bool(torch.isfinite(alm).all())
