import pytest
import torch
from torch.utils.data import DataLoader

import torchsky.catalogs.core as catalogs_core
from torchsky.catalogs import (
    Catalog,
    CatalogStreamDataset,
    CrossmatchResult,
    HealpixIndex,
    crossmatch_sky,
)
from torchsky.coords import EarthOrientation, Site, transform
from torchsky.sphere import MOC, great_circle_distance


def test_crossmatch_sky_returns_nearest_matches_within_radius() -> None:
    left_ra = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    left_dec = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    right_ra = torch.tensor([10.0001, 25.0, 29.9998], dtype=torch.float64)
    right_dec = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)

    result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=2.0,
    )

    assert result.left_index.tolist() == [0, 2]
    assert result.right_index.tolist() == [0, 2]
    assert torch.all(result.separation_arcsec < 2.0)


def test_catalog_crossmatch_explicitly_aligns_epochs() -> None:
    left = Catalog(
        lon=torch.tensor([10.0], dtype=torch.float64),
        lat=torch.tensor([0.0], dtype=torch.float64),
        epoch=2000.0,
        pm_lon_coslat_mas_yr=torch.tensor([1000.0], dtype=torch.float64),
        pm_lat_mas_yr=torch.tensor([0.0], dtype=torch.float64),
        columns={"id": torch.tensor([2**60 + 1])},
    )
    right = left.propagate_angular(2020.0).to_frame("GALACTIC")
    assert left.crossmatch(right, radius_arcsec=0.1).left_index.numel() == 0
    result = left.crossmatch(right, radius_arcsec=0.1, target_epoch=2020.0)
    assert result.left_index.tolist() == result.right_index.tolist() == [0]
    assert result.separation_arcsec.item() < 1e-6
    assert left.columns["id"][result.left_index].tolist() == [2**60 + 1]
    assert float(left.epoch) == 2000.0


def test_catalog_crossmatch_physical_epoch_alignment_uses_complete_motion() -> None:
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    original = SkyCoord(
        ra=269.4520769586186 * u.deg,
        dec=4.693364966576667 * u.deg,
        distance=(1000.0 / 547.45) * u.pc,
        pm_ra_cosdec=-802.803 * u.mas / u.yr,
        pm_dec=10362.542 * u.mas / u.yr,
        radial_velocity=-110.6 * u.km / u.s,
        obstime=Time(2000.0, format="jyear"),
    )
    moved = original.apply_space_motion(new_obstime=Time(2020.0, format="jyear"))
    left = Catalog(
        lon=torch.tensor([original.ra.deg]),
        lat=torch.tensor([original.dec.deg]),
        epoch=2000.0,
        pm_lon_coslat_mas_yr=torch.tensor([-802.803], dtype=torch.float64),
        pm_lat_mas_yr=torch.tensor([10362.542], dtype=torch.float64),
        parallax_mas=torch.tensor([547.45], dtype=torch.float64),
        radial_velocity_km_s=torch.tensor([-110.6], dtype=torch.float64),
    )
    right = Catalog(
        lon=torch.tensor([moved.ra.deg]),
        lat=torch.tensor([moved.dec.deg]),
        epoch=2020.0,
    )
    result = left.crossmatch(
        right, radius_arcsec=0.001, target_epoch=2020.0, motion="physical"
    )
    assert result.left_index.tolist() == result.right_index.tolist() == [0]
    assert result.separation_arcsec.item() < 0.001


def test_catalog_crossmatch_epoch_alignment_refuses_missing_motion_and_stale_index() -> (
    None
):
    cat = Catalog(lon=torch.tensor([10.0]), lat=torch.tensor([0.0]), epoch=2000.0)
    with pytest.raises(ValueError, match="pm_lon"):
        cat.crossmatch(cat, radius_arcsec=1.0, target_epoch=2020.0)
    with pytest.raises(ValueError, match="right_index"):
        cat.crossmatch(
            cat, radius_arcsec=1.0, target_epoch=2020.0, right_index=cat.index()
        )
    with pytest.raises(ValueError, match="scalar"):
        cat.crossmatch(
            cat, radius_arcsec=1.0, target_epoch=torch.tensor([2000.0, 2020.0])
        )
    with pytest.raises(ValueError, match="motion"):
        cat.crossmatch(cat, radius_arcsec=1.0, target_epoch=2000.0, motion="invented")


def test_catalog_preserves_fractional_epoch_precision() -> None:
    cat = Catalog(
        lon=torch.tensor([10.0]), lat=torch.tensor([0.0]), epoch=2000.123456789
    )
    assert float(cat.epoch) == pytest.approx(2000.123456789, abs=1e-10, rel=0.0)


def test_crossmatch_rejects_index_from_different_right_coordinates() -> None:
    from torchsky.catalogs import HealpixIndex

    wrong_index = HealpixIndex.build(torch.tensor([200.0]), torch.tensor([0.0]))
    with pytest.raises(ValueError, match="right_index"):
        crossmatch_sky(
            torch.tensor([10.0]),
            torch.tensor([0.0]),
            torch.tensor([10.0]),
            torch.tensor([0.0]),
            radius_arcsec=1.0,
            right_index=wrong_index,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_crossmatch_sky_cuda_matches_cpu() -> None:
    left_ra = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    left_dec = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    right_ra = torch.tensor([10.0001, 25.0, 29.9998], dtype=torch.float64)
    right_dec = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    kwargs = dict(radius_arcsec=2.0, left_chunk_size=2, right_chunk_size=2)
    cpu = crossmatch_sky(left_ra, left_dec, right_ra, right_dec, **kwargs)
    gpu = crossmatch_sky(
        left_ra.cuda(), left_dec.cuda(), right_ra.cuda(), right_dec.cuda(), **kwargs
    )
    assert gpu.left_index.device.type == "cuda"
    assert gpu.left_index.cpu().tolist() == cpu.left_index.tolist()
    assert gpu.right_index.cpu().tolist() == cpu.right_index.tolist()
    torch.testing.assert_close(
        gpu.separation_arcsec.cpu(), cpu.separation_arcsec, atol=1e-9, rtol=0.0
    )


def test_crossmatch_sky_best_breaks_equal_separation_by_right_index() -> None:
    result = crossmatch_sky(
        torch.tensor([10.0], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([10.0, 10.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        radius_arcsec=1.0,
        right_chunk_size=1,
    )

    assert result.left_index.tolist() == [0]
    assert result.right_index.tolist() == [0]


def test_crossmatch_sky_handles_empty_right_catalog() -> None:
    result = crossmatch_sky(
        torch.tensor([10.0], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
        torch.empty(0, dtype=torch.float64),
        radius_arcsec=1.0,
    )
    assert result.left_index.numel() == 0
    assert result.right_index.numel() == 0


def test_crossmatch_sky_find_all_returns_stably_ordered_pairs() -> None:
    result = crossmatch_sky(
        torch.tensor([10.0, 20.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        torch.tensor([10.0002, 10.0001, 20.0001, 80.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        radius_arcsec=2.0,
        left_chunk_size=1,
        right_chunk_size=1,
        find="all",
    )

    assert result.left_index.tolist() == [0, 0, 1]
    assert result.right_index.tolist() == [1, 0, 2]
    assert result.separation_arcsec[0] <= result.separation_arcsec[1]


def test_crossmatch_sky_rejects_unknown_find_mode() -> None:
    with pytest.raises(ValueError, match="find"):
        crossmatch_sky(
            torch.tensor([10.0], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([10.0], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            radius_arcsec=1.0,
            find="nearest",
        )


def test_crossmatch_sky_find_all_matches_brute_force_across_chunks() -> None:
    rng = torch.Generator().manual_seed(20260716)
    left_ra = torch.rand(12, generator=rng, dtype=torch.float64) * 360.0
    left_dec = torch.rad2deg(
        torch.asin((torch.rand(12, generator=rng, dtype=torch.float64) * 2.0) - 1.0)
    )
    right_ra = torch.rand(24, generator=rng, dtype=torch.float64) * 360.0
    right_dec = torch.rad2deg(
        torch.asin((torch.rand(24, generator=rng, dtype=torch.float64) * 2.0) - 1.0)
    )
    right_ra[:8] = torch.remainder(
        left_ra[:4].repeat_interleave(2)
        + torch.tensor(
            [0.05, -0.1, 0.15, -0.2, 0.1, -0.15, 0.2, -0.05],
            dtype=torch.float64,
        )
        / 3600.0,
        360.0,
    )
    right_dec[:8] = left_dec[:4].repeat_interleave(2)

    result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=1.0,
        left_chunk_size=3,
        right_chunk_size=4,
        find="all",
    )
    separation_arcsec = (
        catalogs_core.great_circle_distance(
            left_ra.unsqueeze(1),
            left_dec.unsqueeze(1),
            right_ra.unsqueeze(0),
            right_dec.unsqueeze(0),
            degrees=True,
        )
        * 3600.0
    )
    expected_left, expected_right = torch.nonzero(
        separation_arcsec <= 1.0, as_tuple=True
    )
    expected = sorted(
        zip(
            expected_left.tolist(),
            expected_right.tolist(),
            separation_arcsec[expected_left, expected_right].tolist(),
            strict=True,
        ),
        key=lambda row: (row[0], row[2], row[1]),
    )

    assert result.left_index.tolist() == [row[0] for row in expected]
    assert result.right_index.tolist() == [row[1] for row in expected]
    torch.testing.assert_close(
        result.separation_arcsec,
        torch.tensor([row[2] for row in expected], dtype=torch.float64),
        atol=1e-9,
        rtol=0.0,
    )


def test_crossmatch_sky_handles_ra_wraparound() -> None:
    result = crossmatch_sky(
        torch.tensor([359.9998], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([0.0001, 12.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        radius_arcsec=2.0,
    )
    assert result.left_index.tolist() == [0]
    assert result.right_index.tolist() == [0]
    assert result.separation_arcsec.item() < 2.0


def test_crossmatch_sky_dense_dec_band_matches_declination_baseline() -> None:
    left_ra = torch.tensor([20.0, 20.01, 20.02, 20.03], dtype=torch.float64)
    left_dec = torch.tensor([0.0, 0.002, -0.002, 0.001], dtype=torch.float64)

    band_ra = torch.linspace(0.0, 359.82, 2000, dtype=torch.float64)
    band_dec = torch.zeros_like(band_ra)
    matched_ra = left_ra + torch.tensor([0.00005, -0.00004, 0.00003, -0.00002])
    matched_dec = left_dec + torch.tensor([0.0, 0.0, 0.0, 0.00001])
    right_ra = torch.cat((matched_ra, band_ra))
    right_dec = torch.cat((matched_dec, band_dec))

    coarse_result, coarse_stats = catalogs_core._crossmatch_sky_coarse_indexed(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=3.0,
        left_chunk_size=2,
        right_chunk_size=128,
        return_stats=True,
    )
    decl_result, decl_stats = catalogs_core._crossmatch_sky_declination_pruned(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=3.0,
        left_chunk_size=2,
        right_chunk_size=128,
        return_stats=True,
    )

    assert coarse_result.left_index.tolist() == decl_result.left_index.tolist()
    assert coarse_result.right_index.tolist() == decl_result.right_index.tolist()
    torch.testing.assert_close(
        coarse_result.separation_arcsec,
        decl_result.separation_arcsec,
        atol=1e-9,
        rtol=0.0,
    )
    assert coarse_stats.candidate_rows_considered < decl_stats.candidate_rows_considered
    assert coarse_stats.pairs_evaluated < decl_stats.pairs_evaluated


def test_crossmatch_sky_scalar_radius_matches_declination_indexed() -> None:
    left_ra = torch.tensor([10.0, 10.01, 350.0, 180.0], dtype=torch.float64)
    left_dec = torch.tensor([0.0, 0.002, -5.0, 45.0], dtype=torch.float64)
    right_ra = torch.tensor(
        [10.0001, 10.02, 349.999, 180.001, 90.0], dtype=torch.float64
    )
    right_dec = torch.tensor(
        [0.0001, 0.0015, -5.001, 45.0005, 0.0], dtype=torch.float64
    )
    radius_arcsec = 5.0
    public = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=radius_arcsec,
        left_chunk_size=2,
        right_chunk_size=3,
        find="all",
    )
    baseline = catalogs_core._crossmatch_sky_declination_indexed(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=torch.full(
            (left_ra.numel(),), radius_arcsec, dtype=torch.float64
        ),
        left_chunk_size=2,
        right_chunk_size=3,
        find="all",
    )
    assert public.left_index.tolist() == baseline.left_index.tolist()
    assert public.right_index.tolist() == baseline.right_index.tolist()
    torch.testing.assert_close(
        public.separation_arcsec,
        baseline.separation_arcsec,
        atol=1e-9,
        rtol=0.0,
    )


def test_crossmatch_sky_handles_pole_adjacent_matches() -> None:
    left_ra = torch.tensor([5.0, 220.0], dtype=torch.float64)
    left_dec = torch.tensor([89.9990, 89.9985], dtype=torch.float64)
    right_ra = torch.tensor([185.0, 40.0, 10.0], dtype=torch.float64)
    right_dec = torch.tensor([89.9991, 88.5, 89.9987], dtype=torch.float64)

    result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=12.0,
        left_chunk_size=1,
        right_chunk_size=2,
    )
    baseline = catalogs_core._crossmatch_sky_declination_pruned(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=12.0,
        left_chunk_size=1,
        right_chunk_size=2,
    )

    assert result.left_index.tolist() == [0, 1]
    assert result.left_index.tolist() == baseline.left_index.tolist()
    assert result.right_index.tolist() == baseline.right_index.tolist()
    torch.testing.assert_close(
        result.separation_arcsec,
        baseline.separation_arcsec,
        atol=1e-9,
        rtol=0.0,
    )


def test_crossmatch_sky_matches_declination_baseline_across_chunk_boundaries() -> None:
    left_ra = torch.tensor(
        [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
        dtype=torch.float64,
    )
    left_dec = torch.tensor(
        [-20.0, -10.0, 0.0, 10.0, 20.0, -15.0, 5.0, 25.0],
        dtype=torch.float64,
    )
    right_ra = left_ra + torch.tensor(
        [0.0001, -0.0001, 0.0002, -0.0002, 0.0001, 0.0002, -0.0001, 0.0001],
        dtype=torch.float64,
    )
    right_dec = left_dec + torch.tensor(
        [0.0, 0.0001, -0.0001, 0.0, 0.0001, -0.0001, 0.0001, 0.0],
        dtype=torch.float64,
    )
    right_ra = torch.cat(
        (right_ra, torch.tensor([12.0, 144.0, 288.0], dtype=torch.float64))
    )
    right_dec = torch.cat(
        (right_dec, torch.tensor([50.0, -55.0, 70.0], dtype=torch.float64))
    )

    result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=2.0,
        left_chunk_size=3,
        right_chunk_size=2,
    )
    baseline = catalogs_core._crossmatch_sky_declination_pruned(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=2.0,
        left_chunk_size=3,
        right_chunk_size=2,
    )

    assert result.left_index.tolist() == baseline.left_index.tolist()
    assert result.right_index.tolist() == baseline.right_index.tolist()
    torch.testing.assert_close(
        result.separation_arcsec,
        baseline.separation_arcsec,
        atol=1e-9,
        rtol=0.0,
    )


def test_crossmatch_sky_randomized_parity_against_declination_baseline() -> None:
    rng = torch.Generator().manual_seed(1234)
    n_left = 48
    n_right = 80

    left_ra = torch.rand(n_left, generator=rng, dtype=torch.float64) * 360.0
    left_dec = torch.rad2deg(
        torch.asin((torch.rand(n_left, generator=rng, dtype=torch.float64) * 2.0) - 1.0)
    )
    right_ra = torch.rand(n_right, generator=rng, dtype=torch.float64) * 360.0
    right_dec = torch.rad2deg(
        torch.asin(
            (torch.rand(n_right, generator=rng, dtype=torch.float64) * 2.0) - 1.0
        )
    )

    right_ra[:12] = (
        left_ra[:12]
        + (torch.rand(12, generator=rng, dtype=torch.float64) - 0.5) * 0.0002
    )
    right_dec[:12] = (
        left_dec[:12]
        + (torch.rand(12, generator=rng, dtype=torch.float64) - 0.5) * 0.0002
    )

    result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=2.0,
        left_chunk_size=11,
        right_chunk_size=13,
    )
    baseline = catalogs_core._crossmatch_sky_declination_pruned(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=2.0,
        left_chunk_size=11,
        right_chunk_size=13,
    )

    assert result.left_index.tolist() == baseline.left_index.tolist()
    assert result.right_index.tolist() == baseline.right_index.tolist()
    torch.testing.assert_close(
        result.separation_arcsec,
        baseline.separation_arcsec,
        atol=1e-9,
        rtol=0.0,
    )


def test_catalog_stream_dataset_splits_work_across_workers() -> None:
    ds = CatalogStreamDataset(range(8))
    loader = DataLoader(ds, num_workers=2, batch_size=None)
    got = sorted(int(x) for x in loader)
    assert got == list(range(8))


def test_catalog_stream_dataset_supports_partition_factory() -> None:
    ds = CatalogStreamDataset(lambda: iter(range(4)))
    assert list(ds) == [0, 1, 2, 3]
    assert list(ds) == [0, 1, 2, 3]


def test_finalize_crossmatch_empty_lists() -> None:
    device = torch.device("cpu")
    stats = catalogs_core._CrossmatchStats(left_points=0, right_points=0)
    result = catalogs_core._finalize_crossmatch([], [], [], device=device, stats=stats)
    assert result.left_index.numel() == 0
    assert result.right_index.numel() == 0
    assert result.separation_arcsec.numel() == 0


def test_finalize_crossmatch_combines_and_sorts_lists() -> None:
    device = torch.device("cpu")
    stats = catalogs_core._CrossmatchStats(left_points=5, right_points=5)

    left_hits = [
        torch.tensor([3, 1], dtype=torch.int64),
        torch.tensor([0, 2], dtype=torch.int64),
    ]
    right_hits = [
        torch.tensor([13, 11], dtype=torch.int64),
        torch.tensor([10, 12], dtype=torch.int64),
    ]
    sep_hits = [
        torch.tensor([0.3, 0.1], dtype=torch.float64),
        torch.tensor([0.0, 0.2], dtype=torch.float64),
    ]

    result = catalogs_core._finalize_crossmatch(
        left_hits, right_hits, sep_hits, device=device, stats=stats
    )

    # Left indices should be sorted: 0, 1, 2, 3
    assert result.left_index.tolist() == [0, 1, 2, 3]
    # Right indices should be sorted corresponding to left indices: 10, 11, 12, 13
    assert result.right_index.tolist() == [10, 11, 12, 13]
    # Separations should be sorted corresponding to left indices: 0.0, 0.1, 0.2, 0.3
    torch.testing.assert_close(
        result.separation_arcsec,
        torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64),
    )
    assert stats.matched_rows == 4


def test_crossmatch_result_is_public() -> None:
    result = crossmatch_sky(
        torch.tensor([10.0], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([10.0], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        radius_arcsec=1.0,
    )
    assert isinstance(result, CrossmatchResult)


def test_healpix_index_query_disc_matches_brute_small_n() -> None:

    lon = torch.tensor([10.0, 10.2, 80.0, 10.05], dtype=torch.float64)
    lat = torch.tensor([0.0, 0.1, 20.0, -0.02], dtype=torch.float64)
    index = HealpixIndex.build(lon, lat, nside=32)
    got = torch.sort(index.query_disc(10.0, 0.0, 0.5)).values
    sep = great_circle_distance(10.0, 0.0, lon, lat, degrees=True)
    expected = torch.nonzero(sep <= 0.5, as_tuple=False).reshape(-1)
    assert got.tolist() == expected.tolist()


def test_crossmatch_with_prebuilt_index_matches_crossmatch_sky() -> None:

    left_ra = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    left_dec = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    right_ra = torch.tensor([10.0001, 25.0, 29.9998], dtype=torch.float64)
    right_dec = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    expected = crossmatch_sky(left_ra, left_dec, right_ra, right_dec, radius_arcsec=2.0)
    index = HealpixIndex.build(right_ra, right_dec, nside=64)
    got = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=2.0,
        right_index=index,
    )
    assert got.left_index.tolist() == expected.left_index.tolist()
    assert got.right_index.tolist() == expected.right_index.tolist()
    torch.testing.assert_close(got.separation_arcsec, expected.separation_arcsec)


def test_catalog_roundtrip_cone_inside_take_and_crossmatch() -> None:

    lon = torch.tensor([10.0, 120.0, 200.0], dtype=torch.float64)
    lat = torch.tensor([20.0, -10.0, 5.0], dtype=torch.float64)
    ids = torch.tensor([7, 8, 9], dtype=torch.int64)
    cat = Catalog(lon=lon, lat=lat, frame="icrs", columns={"id": ids})
    assert len(cat) == 3
    state = cat.to_state()
    back = Catalog.from_state(state, columns=cat.columns)
    torch.testing.assert_close(back.lon, cat.lon)
    assert back.columns["id"].tolist() == [7, 8, 9]

    galactic = cat.to_frame("galactic")
    lon_g, lat_g = transform(lon, lat, from_frame="ICRS", to_frame="GALACTIC")
    torch.testing.assert_close(galactic.lon, lon_g)
    torch.testing.assert_close(galactic.lat, lat_g)

    cone_idx = cat.cone(10.0, 20.0, 1.0)
    assert cone_idx.tolist() == [0]

    moc = MOC.from_lonlat(lon[:1], lat[:1], max_order=6)
    mask = cat.inside(moc)
    assert mask.tolist() == [True, False, False]
    assert moc.contains_rows(cat).tolist() == mask.tolist()
    taken = cat.take(mask)
    assert taken.columns["id"].tolist() == [7]
    taken2 = cat.take(torch.tensor([0, 2], dtype=torch.int64))
    assert taken2.columns["id"].tolist() == [7, 9]

    other = Catalog(
        lon=torch.tensor([10.0001, 50.0], dtype=torch.float64),
        lat=torch.tensor([20.0, 0.0], dtype=torch.float64),
    )
    matches = cat.crossmatch(other, radius_arcsec=2.0)
    assert isinstance(matches, CrossmatchResult)
    assert matches.left_index.tolist() == [0]
    assert matches.right_index.tolist() == [0]

    site = Site(lon_deg=-155.47, lat_deg=19.83)
    eop = EarthOrientation(dut1_s=0.184)
    altaz = cat.observe("ALTAZ", obstime_mjd_utc=60000.0, site=site, eop=eop)
    lon_a, lat_a = transform(
        lon,
        lat,
        from_frame="ICRS",
        to_frame="ALTAZ",
        obstime_mjd_utc=60000.0,
        site=site,
        eop=eop,
    )
    torch.testing.assert_close(altaz.lon, lon_a)
    torch.testing.assert_close(altaz.lat, lat_a)
    assert altaz.frame == "ALTAZ"


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
def test_catalog_index_and_moc_contains_stay_on_device(device_type: str) -> None:

    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device_type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        pytest.skip("MPS not available")
    device = torch.device(device_type)
    dtype = torch.float32 if device_type == "mps" else torch.float64
    lon = torch.tensor([10.0, 40.0, 80.0], dtype=dtype, device=device)
    lat = torch.tensor([0.0, 5.0, -5.0], dtype=dtype, device=device)
    cat = Catalog(lon=lon, lat=lat)
    assert cat.device.type == device_type
    moved = cat.to(device)
    assert moved.lon.dtype == dtype
    idx = HealpixIndex.build(moved.lon, moved.lat, nside=16)
    hit = idx.query_disc(10.0, 0.0, 2.0)
    assert hit.device.type == device_type
    cone = moved.cone(10.0, 0.0, 2.0)
    assert cone.device.type == device_type
    moc = MOC.from_lonlat(lon[:1].cpu(), lat[:1].cpu(), max_order=4)
    mask = moc.contains(moved.lon, moved.lat)
    assert mask.device.type == device_type
    inside = moved.inside(moc)
    assert inside.device.type == device_type


def _brute_crossmatch(left, right, radius, find):
    sep = (
        great_circle_distance(
            left[0][:, None], left[1][:, None], right[0][None], right[1][None]
        )
        * 3600.0
    )
    li, ri = torch.nonzero(sep <= radius.reshape(-1, 1), as_tuple=True)
    s = sep[li, ri]
    order = sorted(
        range(li.numel()), key=lambda k: (li[k].item(), s[k].item(), ri[k].item())
    )
    idx = torch.tensor(order, dtype=torch.long)
    li, ri, s = li[idx], ri[idx], s[idx]
    if find == "best":
        keep = torch.ones_like(li, dtype=torch.bool)
        keep[1:] = li[1:] != li[:-1]
        li, ri, s = li[keep], ri[keep], s[keep]
    return li, ri, s


@pytest.mark.parametrize("find", ["best", "all"])
@pytest.mark.parametrize("region", ["all_sky", "north_pole", "ra_wrap"])
def test_crossmatch_strip_candidates_match_brute_force(
    region: str, find: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tiny candidate batches force left rows to straddle batches.
    monkeypatch.setattr(catalogs_core, "_MIN_CANDIDATE_PAIRS", 7)
    g = torch.Generator().manual_seed(17)

    def draw(n: int):
        ra = torch.rand(n, generator=g, dtype=torch.float64) * 360.0
        if region == "all_sky":
            dec = torch.rad2deg(
                torch.asin(2.0 * torch.rand(n, generator=g, dtype=torch.float64) - 1.0)
            )
        elif region == "north_pole":
            dec = 87.0 + 3.0 * torch.rand(n, generator=g, dtype=torch.float64)
        else:
            ra = (ra * 0.01 - 0.5) % 360.0
            dec = 10.0 * torch.rand(n, generator=g, dtype=torch.float64) - 5.0
        return ra, dec

    n = 250
    left, right = draw(n), draw(n)
    # Exact poles and RA outside [0, 360) are valid input.
    left = (
        torch.cat([left[0], torch.tensor([0.0, 725.0, -10.0], dtype=torch.float64)]),
        torch.cat([left[1], torch.tensor([90.0, -90.0, 0.0], dtype=torch.float64)]),
    )
    radius = 3600.0 * (0.2 + 2.0 * torch.rand(n + 3, generator=g, dtype=torch.float64))
    got = crossmatch_sky(
        left[0],
        left[1],
        right[0],
        right[1],
        radius_arcsec=radius,
        find=find,
        left_chunk_size=64,
    )
    li, ri, s = _brute_crossmatch(left, right, radius, find)
    assert li.numel() > 0
    assert torch.equal(got.left_index, li)
    assert torch.equal(got.right_index, ri)
    assert torch.allclose(got.separation_arcsec, s)
