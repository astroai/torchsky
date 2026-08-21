import math

import pytest
import torch

from torchsky.catalogs import cap_covering_pixels, crossmatch_sky, nside_for_cap
from torchsky.catalogs.index import HealpixIndex, candidate_coarse_pixels_for_left_pixel
from torchsky.sphere import chord_radius, great_circle_distance
from torchsky.sphere import healpix as _healpix


def test_chord_radius_right_angle() -> None:
    chord = chord_radius(0.5 * math.pi)
    torch.testing.assert_close(chord, torch.tensor(math.sqrt(2.0), dtype=chord.dtype))


def test_nside_for_cap_is_covering_derived() -> None:
    wide = nside_for_cap(math.radians(30.0))
    tight = nside_for_cap(math.radians(1.0 / 3600.0))
    assert wide == 1 or _healpix.max_pixrad(wide, degrees=False) <= math.radians(30.0)
    assert tight >= wide
    assert tight <= 65536


def test_cap_covering_matches_pixel_centers() -> None:
    nside = 32
    lon = torch.tensor([10.0], dtype=torch.float64)
    lat = torch.tensor([20.0], dtype=torch.float64)
    radius = math.radians(3.0)
    got = cap_covering_pixels(nside, lon, lat, radius, nest=True)
    pix = torch.arange(_healpix.nside2npix(nside), dtype=torch.int64)
    clon, clat = _healpix.pix2ang(nside, pix, nest=True, lonlat=True)
    sep = great_circle_distance(lon, lat, clon, clat, degrees=False)
    expect = pix[sep <= radius]
    assert torch.equal(got.sort().values, expect.sort().values)


def test_cap_covering_sound_for_random_pairs() -> None:
    rng = torch.Generator().manual_seed(20260821)
    n_left = 48
    n_right = 96
    left_ra = torch.rand(n_left, generator=rng, dtype=torch.float64) * 360.0
    u = torch.rand(n_left, generator=rng, dtype=torch.float64)
    left_dec = torch.rad2deg(torch.asin(2.0 * u - 1.0))
    right_ra = torch.rand(n_right, generator=rng, dtype=torch.float64) * 360.0
    v = torch.rand(n_right, generator=rng, dtype=torch.float64)
    right_dec = torch.rad2deg(torch.asin(2.0 * v - 1.0))
    radius_arcsec = 3600.0
    brute = great_circle_distance(
        left_ra.unsqueeze(1),
        left_dec.unsqueeze(1),
        right_ra.unsqueeze(0),
        right_dec.unsqueeze(0),
        degrees=True,
    )
    brute_pairs = torch.nonzero(brute * 3600.0 <= radius_arcsec, as_tuple=False)
    result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=radius_arcsec,
        find="all",
    )
    got = torch.stack((result.left_index, result.right_index), dim=1)
    brute_set = {(int(a), int(b)) for a, b in brute_pairs.tolist()}
    got_set = {(int(a), int(b)) for a, b in got.tolist()}
    assert brute_set == got_set


def test_one_arcsec_all_pairs_match_brute() -> None:
    rng = torch.Generator().manual_seed(7)
    n = 64
    left_ra = 150.0 + torch.rand(n, generator=rng, dtype=torch.float64) * 0.02
    left_dec = 2.0 + torch.rand(n, generator=rng, dtype=torch.float64) * 0.02
    right_ra = 150.0 + torch.rand(n, generator=rng, dtype=torch.float64) * 0.02
    right_dec = 2.0 + torch.rand(n, generator=rng, dtype=torch.float64) * 0.02
    right_ra[: n // 2] = left_ra[: n // 2]
    right_dec[: n // 2] = left_dec[: n // 2]
    radius_arcsec = 5.0
    brute = great_circle_distance(
        left_ra.unsqueeze(1),
        left_dec.unsqueeze(1),
        right_ra.unsqueeze(0),
        right_dec.unsqueeze(0),
        degrees=True,
    )
    brute_pairs = torch.nonzero(brute * 3600.0 <= radius_arcsec, as_tuple=False)
    result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=radius_arcsec,
        find="all",
    )
    got = {
        (int(a), int(b))
        for a, b in zip(
            result.left_index.tolist(), result.right_index.tolist(), strict=True
        )
    }
    brute_set = {(int(a), int(b)) for a, b in brute_pairs.tolist()}
    assert brute_set == got


def test_per_row_radius_does_not_drop_wide_matches() -> None:
    left_ra = torch.tensor([0.0, 0.0], dtype=torch.float64)
    left_dec = torch.tensor([0.0, 0.0], dtype=torch.float64)
    right_ra = torch.tensor([0.001, 1.0], dtype=torch.float64)
    right_dec = torch.tensor([0.0, 0.0], dtype=torch.float64)
    radii = torch.tensor([5.0, 7200.0], dtype=torch.float64)
    result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=radii,
        find="all",
    )
    pairs = set(
        zip(result.left_index.tolist(), result.right_index.tolist(), strict=True)
    )
    assert (0, 0) in pairs
    assert (1, 0) in pairs
    assert (1, 1) in pairs
    assert (0, 1) not in pairs


def test_ra_wrap_and_pole_caps() -> None:
    result_wrap = crossmatch_sky(
        torch.tensor([359.999], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([0.001], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        radius_arcsec=15.0,
        find="all",
    )
    assert result_wrap.left_index.tolist() == [0]
    result_pole = crossmatch_sky(
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([89.99], dtype=torch.float64),
        torch.tensor([180.0], dtype=torch.float64),
        torch.tensor([89.99], dtype=torch.float64),
        radius_arcsec=120.0,
        find="all",
    )
    assert result_pole.left_index.numel() == 1


def test_candidate_pixels_match_cap_covering() -> None:
    nside = 16
    pix = 100
    radius = math.radians(8.0)
    device = torch.device("cpu")
    got = candidate_coarse_pixels_for_left_pixel(
        nside, pix, radius_rad=radius, device=device
    )
    lon, lat = _healpix.pix2ang(
        nside, torch.tensor([pix], dtype=torch.int64), nest=True, lonlat=True
    )
    expect = cap_covering_pixels(nside, lon, lat, radius, nest=True)
    assert torch.equal(got.sort().values, expect.sort().values)


def test_cap_covering_matches_healpy_query_disc() -> None:
    healpy = pytest.importorskip("healpy")
    nside = 32
    vec = healpy.ang2vec(math.radians(90.0 - 20.0), math.radians(10.0))
    radius = math.radians(4.0)
    hp_pix = healpy.query_disc(nside, vec, radius, inclusive=False, nest=True)
    lon = torch.tensor([10.0], dtype=torch.float64)
    lat = torch.tensor([20.0], dtype=torch.float64)
    got = cap_covering_pixels(nside, lon, lat, radius, nest=True)
    assert set(got.tolist()) == {int(p) for p in hp_pix}


def test_cap_covering_matches_hpgeom_query_circle() -> None:
    hpgeom = pytest.importorskip("hpgeom")
    nside = 32
    radius = 4.0
    lon = 10.0
    lat = 20.0
    hp_pix = hpgeom.query_circle(
        nside, lon, lat, radius, inclusive=False, nest=True, degrees=True
    )
    got = cap_covering_pixels(
        nside,
        torch.tensor([lon], dtype=torch.float64),
        torch.tensor([lat], dtype=torch.float64),
        math.radians(radius),
        nest=True,
    )
    assert set(got.tolist()) == {int(p) for p in hp_pix}


def test_index_expand_keeps_query_alignment() -> None:
    ra = torch.tensor([10.0, 10.0, 40.0], dtype=torch.float64)
    dec = torch.tensor([20.0, 20.0, -10.0], dtype=torch.float64)
    index = HealpixIndex.build(ra, dec, nside=32, nest=True)
    pix = _healpix.ang2pix(32, ra, dec, nest=True, lonlat=True).to(torch.int64)
    query, rows = index.expand(pix)
    got = {(int(q), int(r)) for q, r in zip(query.tolist(), rows.tolist(), strict=True)}
    assert (0, 0) in got
    assert (0, 1) in got
    assert (1, 0) in got
    assert (1, 1) in got
    assert (2, 2) in got
    assert (2, 0) not in got
