"""Torch fallback of HEALPix disc/polygon queries: native parity and scaling.

Without the compiled extension (or on MPS), ``query_disc``, ``query_polygon`` and
``SphericalPolygon.pixelize`` run a torch quadtree descent. It must select exactly
the pixels the native path selects, and its cost must follow the region boundary,
not the number of sky pixels.
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from torchsky import _ops
from torchsky.sphere import SphericalPolygon, healpix as hp

_HAS_NATIVE = _ops.has_fixed_dispatcher()
needs_native = pytest.mark.skipif(
    not _HAS_NATIVE, reason="native extension unavailable"
)


def _fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_ops, "has_fixed_dispatcher", lambda: False)


def _both(monkeypatch: pytest.MonkeyPatch, fn):
    native = fn()
    with monkeypatch.context() as m:
        _fallback(m)
        fallback = fn()
    return torch.sort(native).values, torch.sort(fallback).values


def _centres(g: torch.Generator, n: int) -> list[tuple[float, float]]:
    """Random centres plus poles and RA-wrap cases."""
    fixed = [(0.0, 90.0), (123.0, -90.0), (359.999, 0.0), (0.0, -7.25), (180.0, 89.9)]
    rnd = [
        (
            float(torch.rand(1, generator=g)) * 360.0,
            math.degrees(math.asin(2.0 * float(torch.rand(1, generator=g)) - 1.0)),
        )
        for _ in range(n)
    ]
    return fixed + rnd


def _polygon(lon0: float, lat0: float, size_deg: float, g: torch.Generator):
    """Random convex-ish or concave polygon around (lon0, lat0) in a tangent plane."""
    nvert = int(torch.randint(3, 8, (1,), generator=g))
    ang = torch.sort(torch.rand(nvert, generator=g, dtype=torch.float64)).values
    ang = ang * 2.0 * math.pi
    rad = size_deg * (0.4 + 0.6 * torch.rand(nvert, generator=g, dtype=torch.float64))
    x = torch.deg2rad(rad) * torch.cos(ang)
    y = torch.deg2rad(rad) * torch.sin(ang)
    # Inverse gnomonic projection about (lon0, lat0).
    lam0, phi0 = math.radians(lon0), math.radians(lat0)
    rho = torch.sqrt(x * x + y * y)
    c = torch.atan(rho)
    sin_c, cos_c = torch.sin(c), torch.cos(c)
    lat = torch.asin(
        cos_c * math.sin(phi0)
        + torch.where(rho > 0, y * sin_c * math.cos(phi0) / rho, 0)
    )
    lon = lam0 + torch.atan2(
        x * sin_c, rho * math.cos(phi0) * cos_c - y * math.sin(phi0) * sin_c
    )
    return torch.rad2deg(lon) % 360.0, torch.rad2deg(lat)


# (nside, region size in degrees): output stays small at the finest nside.
_LADDER = [(1, 40.0), (8, 20.0), (256, 3.0), (4096, 0.2), (65536, 0.01)]


@needs_native
@pytest.mark.parametrize("nside,size", _LADDER)
@pytest.mark.parametrize("nest", [False, True])
@pytest.mark.parametrize("inclusive", [False, True])
def test_disc_fallback_matches_native(monkeypatch, nside, size, nest, inclusive):
    g = torch.Generator().manual_seed(nside + 2 * nest + inclusive)
    for lon, lat in _centres(g, 12):
        radius = math.radians(size * float(torch.rand(1, generator=g)))
        x, y, z = hp.lonlat_to_xyz(
            torch.tensor(lon, dtype=torch.float64),
            torch.tensor(lat, dtype=torch.float64),
        )
        vec = torch.stack((x, y, z))
        native, fallback = _both(
            monkeypatch,
            lambda: hp.query_disc(nside, vec, radius, nest=nest, inclusive=inclusive),
        )
        assert torch.equal(native, fallback), (nside, lon, lat, radius)


@needs_native
@pytest.mark.parametrize("fact", [2, 8])
def test_disc_fallback_matches_native_inclusive_fact(monkeypatch, fact):
    g = torch.Generator().manual_seed(fact)
    for lon, lat in _centres(g, 6):
        vec = torch.stack(
            hp.lonlat_to_xyz(
                torch.tensor(lon, dtype=torch.float64),
                torch.tensor(lat, dtype=torch.float64),
            )
        )
        for nest in (False, True):
            native, fallback = _both(
                monkeypatch,
                lambda: hp.query_disc(
                    64, vec, math.radians(2.0), inclusive=True, fact=fact, nest=nest
                ),
            )
            assert torch.equal(native, fallback)


@needs_native
@pytest.mark.parametrize("nside,size", _LADDER)
@pytest.mark.parametrize("nest", [False, True])
@pytest.mark.parametrize("inclusive", [False, True])
def test_polygon_fallback_matches_native(monkeypatch, nside, size, nest, inclusive):
    g = torch.Generator().manual_seed(1000 + nside + 2 * nest + inclusive)
    for lon0, lat0 in _centres(g, 10):
        lon, lat = _polygon(lon0, lat0, size, g)
        try:
            native, fallback = _both(
                monkeypatch,
                lambda: hp.query_polygon(
                    nside, lon, lat, nest=nest, inclusive=inclusive
                ),
            )
        except ValueError:  # degenerate random polygon
            continue
        assert torch.equal(native, fallback), (nside, lon0, lat0)


@needs_native
@pytest.mark.parametrize("nside,size", _LADDER)
def test_pixelize_fallback_matches_native(monkeypatch, nside, size):
    g = torch.Generator().manual_seed(2000 + nside)
    for lon0, lat0 in _centres(g, 6):
        lon, lat = _polygon(lon0, lat0, size, g)
        poly = SphericalPolygon(lon_deg=lon, lat_deg=lat)
        for nest in (False, True):
            try:
                native, fallback = _both(
                    monkeypatch, lambda: poly.pixelize(nside, nest=nest).pixels
                )
            except ValueError:
                continue
            assert torch.equal(native, fallback), (nside, lon0, lat0)


# The 1.6-arcmin polygon from docs/sphere/index.md (straddles RA = 0).
_SMALL = (
    [358.83839378, 358.83840792, 358.81182784, 358.81181209],
    [-7.26829861, -7.24193051, -7.24191619, -7.26828424],
)


def test_pixelize_fallback_scales_with_boundary(monkeypatch):
    """nside 2^16 (5e10 sky pixels) must not cost O(npix) in the torch path."""
    _fallback(monkeypatch)
    poly = SphericalPolygon(
        lon_deg=torch.tensor(_SMALL[0], dtype=torch.float64),
        lat_deg=torch.tensor(_SMALL[1], dtype=torch.float64),
    )
    poly.pixelize(nside=2**10)  # warm caches
    t0 = time.perf_counter()
    ring = poly.pixelize(nside=2**16)
    nest = poly.pixelize(nside=2**16, nest=True)
    disc = hp.query_circle(2**16, 358.825, -7.255, 0.02)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, elapsed
    area_arcmin2 = float(poly.area()) * (180.0 * 60.0 / math.pi) ** 2
    pix_arcmin2 = hp.nside2pixarea(2**16, degrees=True) * 3600.0
    assert abs(ring.pixels.numel() * pix_arcmin2 / area_arcmin2 - 1.0) < 0.05
    assert torch.equal(torch.sort(hp.ring2nest(2**16, ring.pixels)).values, nest.pixels)
    assert disc.numel() > 0
