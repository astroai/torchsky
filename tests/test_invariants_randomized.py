"""Randomized invariants for cross-cutting primitives.

These are not comparisons against a reference implementation (the harness owns
that); they are properties that must hold for any input: round trips, index
ranges, symmetry, conservation, and endpoint behaviour. Each test uses a fixed
seed so a failure is reproducible.
"""

from __future__ import annotations

import math

import pytest
import torch

import torchsky.sphere as sphere
from torchsky.wcs import WCS


def _tan_header(ra: float, dec: float, *, crpix: tuple[float, float] = (16.5, 16.5)) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 32,
        "NAXIS2": 32,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": crpix[0],
        "CRPIX2": crpix[1],
        "CRVAL1": ra,
        "CRVAL2": dec,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }


@pytest.mark.parametrize("seed", range(4))
def test_tan_roundtrip_is_subpixel(seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    ra = torch.rand(1, generator=g).item() * 360.0
    dec = (torch.rand(1, generator=g).item() * 120.0) - 60.0
    wcs = WCS(_tan_header(ra, dec))
    x = torch.rand(256, generator=g, dtype=torch.float64) * 31.0 + 0.5
    y = torch.rand(256, generator=g, dtype=torch.float64) * 31.0 + 0.5
    lon, lat = wcs.pixel_to_world(x, y)
    x_back, y_back = wcs.world_to_pixel(lon, lat)
    error = torch.hypot(x_back - x, y_back - y)
    assert float(error.max()) < 1e-6


@pytest.mark.parametrize("nside", [1, 2, 8, 32])
@pytest.mark.parametrize("nest", [False, True])
def test_healpix_index_roundtrip(nside: int, nest: bool) -> None:
    g = torch.Generator().manual_seed(nside)
    lon = torch.rand(512, generator=g) * 360.0 - 180.0
    lat = torch.asin(torch.rand(512, generator=g) * 2.0 - 1.0).rad2deg()
    pix = sphere.ang2pix(nside, lon, lat, nest=nest, lonlat=True)
    assert int(pix.min()) >= 0
    assert int(pix.max()) < sphere.nside2npix(nside)

    lon_back, lat_back = sphere.pix2ang(nside, pix, nest=nest, lonlat=True)
    angular = sphere.great_circle_distance(lon, lat, lon_back, lat_back)
    resolution = sphere.nside2resol(nside, arcmin=True) / 60.0
    assert float(angular.max()) <= resolution
    again = sphere.ang2pix(nside, lon_back, lat_back, nest=nest, lonlat=True)
    assert torch.equal(pix, again)


@pytest.mark.parametrize("nside", [4, 16])
def test_query_disc_membership_and_uniqueness(nside: int) -> None:
    g = torch.Generator().manual_seed(7 * nside)
    lon0 = float(torch.rand(1, generator=g).item() * 360.0 - 180.0)
    lat0 = float((torch.rand(1, generator=g).item() * 120.0) - 60.0)
    resolution = math.degrees(math.sqrt(float(sphere.nside2pixarea(nside))))
    radius = resolution * (1.5 + 2.0 * float(torch.rand(1, generator=g).item()))
    pixels = sphere.query_circle(nside, lon0, lat0, radius)
    npix = sphere.nside2npix(nside)
    assert pixels.numel() > 0
    assert int(pixels.min()) >= 0 and int(pixels.max()) < npix
    assert pixels.unique().numel() == pixels.numel()

    lon, lat = sphere.pix2ang(nside, pixels, lonlat=True)
    angular = sphere.great_circle_distance(lon0, lat0, lon, lat)
    assert float(angular.max()) <= radius + 1e-9


def test_great_circle_distance_matches_planar_limit() -> None:
    g = torch.Generator().manual_seed(11)
    lat = ((torch.rand(64, generator=g, dtype=torch.float64) * 120.0) - 60.0)
    dlon = (torch.rand(64, generator=g, dtype=torch.float64) * 2.0 - 1.0) * 1e-3
    dlat = (torch.rand(64, generator=g, dtype=torch.float64) * 2.0 - 1.0) * 1e-3
    distance = sphere.great_circle_distance(0.0 * lat, lat, dlon, lat + dlat)
    planar = torch.hypot(dlon * torch.cos(lat.deg2rad()), dlat)
    assert torch.allclose(distance, planar, rtol=1e-6, atol=1e-8)


def test_great_circle_distance_is_antipodal_on_the_sphere() -> None:
    distance = sphere.great_circle_distance(10.0, 20.0, -170.0, -20.0)
    assert float(distance) == pytest.approx(180.0, abs=1e-9)


def test_slerp_endpoints_and_constant_speed() -> None:
    g = torch.Generator().manual_seed(3)
    lon_a = torch.rand(4, generator=g, dtype=torch.float64) * 360.0 - 180.0
    lat_a = (torch.rand(4, generator=g, dtype=torch.float64) * 120.0) - 60.0
    lon_b = lon_a + (torch.rand(4, generator=g, dtype=torch.float64) * 20.0 - 10.0)
    lat_b = torch.clamp(lat_a + (torch.rand(4, generator=g, dtype=torch.float64) * 20.0 - 10.0), -89.0, 89.0)

    lon_a0, lat_a0 = sphere.slerp_lonlat(lon_a, lat_a, lon_b, lat_b, 0.0)
    lon_b1, lat_b1 = sphere.slerp_lonlat(lon_a, lat_a, lon_b, lat_b, 1.0)
    assert torch.allclose(sphere.wrap_longitude(lon_a0), sphere.wrap_longitude(lon_a))
    assert torch.allclose(lat_a0, lat_a)
    assert torch.allclose(
        sphere.wrap_longitude(lon_b1), sphere.wrap_longitude(lon_b), atol=1e-9
    )
    assert torch.allclose(lat_b1, lat_b, atol=1e-9)

    full = sphere.great_circle_distance(lon_a, lat_a, lon_b, lat_b)
    steps = [0.25, 0.5, 0.75]
    fractions = []
    for step in steps:
        lon_t, lat_t = sphere.slerp_lonlat(lon_a, lat_a, lon_b, lat_b, step)
        d = sphere.great_circle_distance(lon_a, lat_a, lon_t, lat_t)
        fractions.append((d / full).mean())
    fractions = torch.stack(fractions)
    assert torch.allclose(fractions, torch.tensor(steps, dtype=torch.float64), atol=2e-3)


def test_observing_geometry_zenith_distance_is_consistent() -> None:
    g = torch.Generator().manual_seed(5)
    lon = torch.rand(64, generator=g, dtype=torch.float64) * 360.0 - 180.0
    lat = torch.rad2deg(torch.asin(torch.rand(64, generator=g, dtype=torch.float64)))
    xyz = sphere.lonlat_to_unit_xyz(lon, lat)
    back_lon, back_lat = sphere.unit_xyz_to_lonlat(xyz)
    assert torch.allclose(sphere.wrap_longitude(back_lon), sphere.wrap_longitude(lon), atol=1e-10)
    assert torch.allclose(back_lat, lat, atol=1e-10)
    assert torch.allclose(torch.linalg.vector_norm(xyz, dim=-1), torch.ones(64, dtype=torch.float64), atol=1e-12)


def _small_tan_header(crval: tuple[float, float], *, size: int = 16) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (size + 1) / 2.0,
        "CRPIX2": (size + 1) / 2.0,
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }


def _interior(mask: torch.Tensor, margin: int = 2) -> torch.Tensor:
    result = mask.clone()
    result[:margin] = False
    result[-margin:] = False
    result[:, :margin] = False
    result[:, -margin:] = False
    return result


@pytest.mark.parametrize("n_obs", [1, 2])
def test_identity_coadd_preserves_constant_and_sums_weight(n_obs: int) -> None:
    from torchsky.map import Observation, coadd, plan

    header = _small_tan_header((120.0, -20.0))
    data = torch.ones(16, 16, dtype=torch.float64)
    observations = [
        Observation(data=data, header=header, weight=1.0) for _ in range(n_obs)
    ]
    target = plan(observations)
    result = coadd(observations, target)

    covered = _interior(result.weight > 0)
    assert bool(covered.any())
    assert torch.allclose(result.weight[covered], torch.full_like(result.weight[covered], float(n_obs)), atol=1e-9)
    assert torch.allclose(result.data[covered], torch.ones_like(result.data[covered]), atol=1e-9)


def test_coadd_is_deterministic() -> None:
    from torchsky.map import Observation, coadd, plan

    header = _small_tan_header((12.0, 45.0))
    data = torch.linspace(0.0, 1.0, 256, dtype=torch.float64).reshape(16, 16)
    observations = [
        Observation(data=data, header=header, weight=1.0),
        Observation(data=data, header=_small_tan_header((12.002, 45.0)), weight=1.0),
    ]
    target = plan(observations)
    first = coadd(observations, target)
    second = coadd(observations, target)
    assert torch.equal(first.data, second.data)
    assert torch.equal(first.weight, second.weight)


@pytest.mark.parametrize("crval", [(0.0, 0.0), (200.0, 60.0), (310.0, -45.0)])
def test_constant_sky_is_projection_invariant(crval: tuple[float, float]) -> None:
    from torchsky.map import Observation, coadd, plan

    header = _small_tan_header(crval)
    data = torch.full((16, 16), 3.5, dtype=torch.float64)
    observation = Observation(data=data, header=header, weight=1.0)
    target = plan([observation])
    result = coadd([observation], target)
    covered = _interior(result.weight > 0)
    assert bool(covered.any()), f"no coverage at {crval}"
    assert torch.allclose(result.data[covered], torch.full_like(result.data[covered], 3.5), atol=1e-6)


@pytest.mark.parametrize(
    ("ctype", "crval", "cdelt"),
    [
        ("WAVE", 5000.0, 1.5),
        ("WAVE-LOG", 5000.0, 0.001),
        ("FREQ", 1.4e9, 1.0e6),
        ("VELO", 0.0, 10.0),
    ],
)
def test_single_axis_spectral_wcs_roundtrip(ctype: str, crval: float, cdelt: float) -> None:
    header = {
        "NAXIS": 1,
        "NAXIS1": 32,
        "CTYPE1": ctype,
        "CRPIX1": 16.5,
        "CRVAL1": crval,
        "CDELT1": cdelt,
    }
    wcs = WCS(header)
    pixels = torch.linspace(0.5, 31.5, 8, dtype=torch.float64)
    world = wcs.pixel_to_world(pixels)
    # A single-axis WCS returns the world array directly; multi-axis WCS
    # objects return one tensor per axis.
    back = wcs.world_to_pixel(world)
    assert float((back - pixels).abs().max()) < 1e-9


def test_celestial_spectral_cube_wcs_roundtrip() -> None:
    header = {
        "NAXIS": 3,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "NAXIS3": 4,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "WAVE",
        "CRPIX1": 4.5,
        "CRPIX2": 4.5,
        "CRPIX3": 2.5,
        "CRVAL1": 30.0,
        "CRVAL2": 10.0,
        "CRVAL3": 5000.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
        "CDELT3": 10.0,
    }
    wcs = WCS(header)
    pixels = torch.meshgrid(
        torch.arange(8.0, dtype=torch.float64),
        torch.arange(8.0, dtype=torch.float64),
        torch.arange(4.0, dtype=torch.float64),
        indexing="ij",
    )
    world = wcs.pixel_to_world(*pixels)
    assert isinstance(world, tuple) and len(world) == 3
    back = wcs.world_to_pixel(*world)
    assert len(back) == 3
    for original, recovered in zip(pixels, back):
        assert float((original - recovered).abs().max()) < 1e-8
