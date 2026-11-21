"""Precision regressions for HEALPix point operations near the poles and at order 29.

Expected pixels were produced by healpy 1.20.1 (cross-checked with cdshealpix 0.9.0)
and are hard-coded so the tests need no oracle at run time. Every test runs on the
native CPU kernels (when built) and on the pure-torch path.
"""

import math

import pytest
import torch

import torchsky._ops as _ops
from torchsky.sphere import healpix as hp

NSIDE29 = 1 << 29
NPIX29 = 12 * NSIDE29 * NSIDE29


@pytest.fixture(params=["native", "torch"])
def path(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "native":
        if not _ops.has_fixed_dispatcher():
            pytest.skip("native extension unavailable")
    else:
        monkeypatch.setattr(_ops, "has_fixed_dispatcher", lambda: False)
    return request.param


def _polar_colatitude(nside: int, ring: int) -> float:
    # 1 - cos(theta) = ring^2 * 4 / npix for polar-cap rings: theta = 2 asin(sqrt(.../2)).
    tmp = ring * ring * 4.0 / (12.0 * nside * nside)
    return 2.0 * math.asin(math.sqrt(0.5 * tmp))


def test_pix2ang_polar_rings_at_order_29(path: str) -> None:
    ring_pix = torch.tensor([0, 5, 2 * 10 * 9 + 3, NPIX29 - 1], dtype=torch.int64)
    expected = torch.tensor(
        [
            _polar_colatitude(NSIDE29, 1),
            _polar_colatitude(NSIDE29, 2),
            _polar_colatitude(NSIDE29, 10),
            math.pi - _polar_colatitude(NSIDE29, 1),
        ],
        dtype=torch.float64,
    )
    theta, _ = hp.pix2ang(NSIDE29, ring_pix, nest=False)
    # healpy 1.20.1: 1.5208433958286904e-09, 3.041686791657381e-09, 1.52084339582869e-08
    torch.testing.assert_close(theta, expected, rtol=1.0e-9, atol=1.0e-15)

    nest_pix = torch.tensor(
        [NSIDE29 * NSIDE29 - 1, 4 * NSIDE29 * NSIDE29 - 1, NPIX29 - NSIDE29 * NSIDE29],
        dtype=torch.int64,
    )
    theta_n, _ = hp.pix2ang(NSIDE29, nest_pix, nest=True)
    expected_n = torch.tensor(
        [
            _polar_colatitude(NSIDE29, 1),
            _polar_colatitude(NSIDE29, 1),
            math.pi - _polar_colatitude(NSIDE29, 1),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(theta_n, expected_n, rtol=1.0e-9, atol=1.0e-15)


def test_ang2pix_near_poles_at_order_29_matches_healpy(path: str) -> None:
    lon = torch.tensor(
        [0.0, 287.5598214190986, 15.004008974059948, 132.53432008811856],
        dtype=torch.float64,
    )
    lat = torch.tensor(
        [89.999999, 89.99999906424611, 89.99999988036483, -89.99999902711349],
        dtype=torch.float64,
    )
    ring = hp.ang2pix(NSIDE29, lon, lat, lonlat=True)
    nest = hp.ang2pix(NSIDE29, lon, lat, nest=True, lonlat=True)
    assert ring.tolist() == [264, 255, 4, 3458764513820540680]
    assert nest.tolist() == [
        288230376151711674,
        1152921504606846903,
        288230376151711742,
        2594073385365405747,
    ]


def test_theta_phi_and_vectors_near_poles_at_order_29_match_healpy(path: str) -> None:
    theta = torch.tensor([1.0e-8, 3.0e-8, math.pi - 2.0e-8], dtype=torch.float64)
    phi = torch.tensor([0.3, 1.0, 2.0], dtype=torch.float64)
    ring_expected = [85, 772, 3458764513820540580]
    nest_expected = [288230376151711724, 288230376151711562, 2594073385365405831]
    assert hp.ang2pix(NSIDE29, theta, phi).tolist() == ring_expected
    assert hp.ang2pix(NSIDE29, theta, phi, nest=True).tolist() == nest_expected

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    assert hp.vec2pix(NSIDE29, x, y, z).tolist() == ring_expected
    assert hp.vec2pix(NSIDE29, x, y, z, nest=True).tolist() == nest_expected
    theta_back, _ = hp.vec2ang(torch.stack((x, y, z), dim=-1))
    # vec2ang goes through latitude in degrees, quantised to ~2.5e-16 rad at 90 deg.
    torch.testing.assert_close(theta_back, theta, rtol=1.0e-12, atol=5.0e-16)


def test_polar_pixel_centres_roundtrip_at_order_29(path: str) -> None:
    rings = torch.arange(1, 64, dtype=torch.int64)
    first = 2 * rings * (rings - 1)
    ring_pix = torch.cat((first, first + rings, NPIX29 - 1 - first))
    lon, lat = hp.pix2ang(NSIDE29, ring_pix, lonlat=True)
    assert torch.equal(hp.ang2pix(NSIDE29, lon, lat, lonlat=True), ring_pix)
    nest_pix = hp.ring2nest(NSIDE29, ring_pix)
    lon_n, lat_n = hp.pix2ang(NSIDE29, nest_pix, nest=True, lonlat=True)
    assert torch.equal(
        hp.ang2pix(NSIDE29, lon_n, lat_n, nest=True, lonlat=True), nest_pix
    )
    x, y, z = hp.pix2vec(NSIDE29, ring_pix)
    assert torch.equal(hp.vec2pix(NSIDE29, x, y, z), ring_pix)


def test_python_scalars_are_not_rounded_to_float32(path: str) -> None:
    lon, lat = 123.456789, -45.678901
    t64 = hp.ang2pix(
        NSIDE29,
        torch.tensor(lon, dtype=torch.float64),
        torch.tensor(lat, dtype=torch.float64),
        nest=True,
        lonlat=True,
    )
    assert int(hp.ang2pix(NSIDE29, lon, lat, nest=True, lonlat=True)) == int(t64)
    assert int(t64) == 2759817459223466280  # healpy 1.20.1
    assert hp.ang2pix(NSIDE29, [lon], [lat], lonlat=True).tolist() == [
        2966643707646160727
    ]
    x, y, z = hp.lonlat_to_xyz(lon, lat)
    assert x.dtype == torch.float64
    back_lon, back_lat = hp.xyz_to_lonlat(x, y, z)
    assert abs(float(back_lon) - lon) < 1.0e-12
    assert abs(float(back_lat) - lat) < 1.0e-12


def test_ring_nest_torch_path_needs_no_per_ring_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_ops, "has_fixed_dispatcher", lambda: False)
    pix = torch.tensor([0, 7, NPIX29 // 2, NPIX29 - 1, NPIX29 - 9], dtype=torch.int64)
    nest = hp.ring2nest(NSIDE29, pix)
    assert torch.equal(hp.nest2ring(NSIDE29, nest), pix)
    from torchsky.sphere import _healpix_scheme

    # The cap-ring lookup used to cache an O(nside) table (4 GiB at order 29).
    assert not any(
        key[2] == NSIDE29 for key in getattr(_healpix_scheme, "_CAP_START_CACHE", {})
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("order", [20, 26, 29])
def test_mps_ring_nest_and_centres_are_exact_at_high_order(order: int) -> None:
    nside = 1 << order
    npix = 12 * nside * nside
    gen = torch.Generator().manual_seed(order)
    pix = torch.cat(
        (
            torch.arange(0, 512),
            torch.randint(0, npix, (4096,), generator=gen),
            npix - 1 - torch.arange(512),
        )
    )
    ref_r2n = hp.ring2nest(nside, pix)
    ref_n2r = hp.nest2ring(nside, pix)
    assert torch.equal(hp.ring2nest(nside, pix.to("mps")).cpu(), ref_r2n)
    assert torch.equal(hp.nest2ring(nside, pix.to("mps")).cpu(), ref_n2r)
    _, lat = hp.pix2ang(nside, pix, lonlat=True)
    _, lat_m = hp.pix2ang(nside, pix.to("mps"), lonlat=True)
    # Two float32 ulps of |lat| in [64, 90] deg (7.6e-6 each); 90 - acos(z) in
    # the belt and asin(1 - tmp) in the caps used to reach ~2.7e-5.
    assert float((lat_m.cpu().double() - lat).abs().max()) < 1.5e-5


def test_core_lonlat_helpers_keep_python_floats_in_float64() -> None:
    from torchsky.sphere import core

    lon, lat = 123.456789, -45.678901
    ref = core.lonlat_to_unit_xyz(
        torch.tensor(lon, dtype=torch.float64), torch.tensor(lat, dtype=torch.float64)
    )
    got = core.lonlat_to_unit_xyz(lon, lat)
    assert got.dtype == torch.float64
    assert torch.equal(got, ref)
    # 1 mas: float32 rounding of the inputs alone was ~7 mas at lon ~ 123 deg.
    d = core.great_circle_distance(lon, lat, lon + 1.0 / 3.6e6, lat)
    expected = (1.0 / 3.6e6) * math.cos(math.radians(lat))
    assert abs(float(d) - expected) < 1.0e-12


def test_xyz_to_lonlat_is_exact_and_differentiable_at_the_poles() -> None:
    x = torch.tensor([0.0, 1.0e-9, 0.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    z = torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64, requires_grad=True)
    lon, lat = hp.xyz_to_lonlat(x, y, z)
    # asin(z / |v|) returned exactly 90 deg for the 1e-9 rad offset.
    assert lat.tolist()[0] == 90.0 and lat.tolist()[2] == -90.0
    assert abs((90.0 - float(lat[1].detach())) - math.degrees(1.0e-9)) < 1.0e-14
    (lon.sum() + lat.sum()).backward()
    for g in (x.grad, y.grad, z.grad):
        assert g is not None and torch.isfinite(g).all()
