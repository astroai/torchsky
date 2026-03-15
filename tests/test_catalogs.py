import torch
from torch.utils.data import DataLoader

import torchsky.catalogs.core as catalogs_core
from torchsky.catalogs import CatalogStreamDataset, crossmatch_sky


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
