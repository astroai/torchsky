from dataclasses import FrozenInstanceError

import pytest
import torch

from torchsky.sphere import BMOC, MOC, PreparedRandomSampler, SparseHealpixMap
from torchsky.sphere import healpix
from torchsky.sphere.moc import nest2uniq


def _sampled_nested_pixels(
    nside: int, lon: torch.Tensor, lat: torch.Tensor
) -> torch.Tensor:
    return healpix.ang2pix(nside, lon, lat, nest=True, lonlat=True)


def test_moc_prepared_matches_one_shot_seed_and_contains_samples() -> None:
    uniq = torch.cat([nest2uniq(1, torch.tensor([0])), nest2uniq(2, torch.tensor([8]))])
    moc = MOC(uniq)
    prepared = moc.prepare_random_sampler(device="cpu")

    assert isinstance(prepared, PreparedRandomSampler)
    assert prepared.device == torch.device("cpu")
    direct = moc.sample_randoms(1024, generator=torch.Generator().manual_seed(17))
    repeated = prepared.sample_randoms(
        1024, generator=torch.Generator().manual_seed(17)
    )

    torch.testing.assert_close(repeated[0], direct[0], rtol=0, atol=0)
    torch.testing.assert_close(repeated[1], direct[1], rtol=0, atol=0)
    assert bool(moc.contains(*repeated).all())


def test_moc_prepared_samples_cells_in_proportion_to_solid_angle() -> None:
    moc = MOC(
        torch.cat([nest2uniq(1, torch.tensor([0])), nest2uniq(2, torch.tensor([8]))])
    )
    lon, lat = moc.prepare_random_sampler().sample_randoms(
        20_000, generator=torch.Generator().manual_seed(5)
    )
    pixels = _sampled_nested_pixels(2, lon, lat)
    coarse_fraction = torch.isin(pixels, torch.arange(4)).to(torch.float64).mean()

    assert float(coarse_fraction) == pytest.approx(0.8, abs=0.02)


def test_bmoc_prepared_boundary_filter_and_snapshot_semantics() -> None:
    bmoc = BMOC(
        nest2uniq(2, torch.tensor([0, 4])), torch.tensor([1, 0], dtype=torch.uint8)
    )
    interior = bmoc.prepare_random_sampler(include_boundary=False)
    all_cells = bmoc.prepare_random_sampler(include_boundary=True)
    bmoc.uniq.fill_(nest2uniq(2, torch.tensor(20)).item())
    bmoc.flag.zero_()

    lon_i, lat_i = interior.sample_randoms(
        512, generator=torch.Generator().manual_seed(2)
    )
    lon_a, lat_a = all_cells.sample_randoms(
        4000, generator=torch.Generator().manual_seed(3)
    )
    pix_i = _sampled_nested_pixels(2, lon_i, lat_i)
    pix_a = _sampled_nested_pixels(2, lon_a, lat_a)

    assert torch.equal(torch.unique(pix_i), torch.tensor([0]))
    assert set(torch.unique(pix_a).tolist()) == {0, 4}


@pytest.mark.parametrize("nest", [True, False])
def test_sparse_prepared_samples_only_covered_pixels_for_both_orderings(
    nest: bool,
) -> None:
    nside = 4
    nested_pixels = torch.tensor([0, 7, 31], dtype=torch.int64)
    pixels = nested_pixels if nest else healpix.nest2ring(nside, nested_pixels)
    sparse = SparseHealpixMap.from_pixels(
        pixels, torch.ones(3), nside_sparse=nside, nest=nest
    )

    lon, lat = sparse.prepare_random_sampler().sample_randoms(
        2000, generator=torch.Generator().manual_seed(19)
    )
    sampled = _sampled_nested_pixels(nside, lon, lat)

    assert bool(torch.isin(sampled, nested_pixels).all())
    assert set(torch.unique(sampled).tolist()) == set(nested_pixels.tolist())


def test_sparse_prepared_weighted_sampling_and_weight_snapshot() -> None:
    sparse = SparseHealpixMap.from_columns(
        torch.tensor([2, 9]),
        {"depth": torch.tensor([1.0, 3.0])},
        nside_sparse=4,
    )
    prepared = sparse.prepare_random_sampler(weights="depth")
    sparse.values[0].zero_()

    lon, lat = prepared.sample_randoms(
        20_000, generator=torch.Generator().manual_seed(23)
    )
    sampled = _sampled_nested_pixels(4, lon, lat)
    high_weight_fraction = (sampled == 9).to(torch.float64).mean()

    assert float(high_weight_fraction) == pytest.approx(0.75, abs=0.02)


def test_prepared_repeated_draws_advance_and_replay_generator_state() -> None:
    prepared = MOC(nest2uniq(2, torch.tensor([1, 5]))).prepare_random_sampler()
    generator = torch.Generator().manual_seed(29)
    first = prepared.sample_randoms(64, generator=generator)
    second = prepared.sample_randoms(64, generator=generator)
    replay_generator = torch.Generator().manual_seed(29)
    replay_first = prepared.sample_randoms(64, generator=replay_generator)
    replay_second = prepared.sample_randoms(64, generator=replay_generator)

    torch.testing.assert_close(first[0], replay_first[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], replay_first[1], rtol=0, atol=0)
    torch.testing.assert_close(second[0], replay_second[0], rtol=0, atol=0)
    torch.testing.assert_close(second[1], replay_second[1], rtol=0, atol=0)
    assert not torch.equal(first[0], second[0])


def test_prepared_sampler_is_frozen() -> None:
    prepared = MOC(nest2uniq(1, torch.tensor([0]))).prepare_random_sampler()

    with pytest.raises(FrozenInstanceError):
        setattr(prepared, "_nsides", 2)


@pytest.mark.parametrize(
    "footprint",
    [
        MOC.empty(),
        BMOC.empty(),
        SparseHealpixMap.from_pixels(
            torch.empty(0, dtype=torch.int64),
            torch.empty(0),
            nside_sparse=2,
        ),
    ],
)
def test_empty_footprint_prepare_errors_but_zero_draw_is_preserved(
    footprint: MOC | BMOC | SparseHealpixMap,
) -> None:
    with pytest.raises(ValueError, match="empty"):
        footprint.prepare_random_sampler()

    lon, lat = footprint.sample_randoms(0)
    assert lon.shape == lat.shape == (0,)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (torch.tensor([0.0, float("nan")]), "positive sum"),
        (torch.tensor([1.0]), "one entry per valid pixel"),
        ("missing", "unknown weights specification"),
    ],
)
def test_sparse_prepared_rejects_invalid_weights(
    weights: torch.Tensor | str, message: str
) -> None:
    sparse = SparseHealpixMap.from_pixels(
        torch.tensor([0, 1]), torch.ones(2), nside_sparse=2
    )

    with pytest.raises(ValueError, match=message):
        sparse.prepare_random_sampler(weights=weights)
    with pytest.raises(ValueError, match=message):
        sparse.sample_randoms(1, weights=weights)


def test_prepared_sampler_rejects_negative_draw_count() -> None:
    prepared = MOC(nest2uniq(1, torch.tensor([0]))).prepare_random_sampler()

    with pytest.raises(ValueError, match="non-negative"):
        prepared.sample_randoms(-1)
