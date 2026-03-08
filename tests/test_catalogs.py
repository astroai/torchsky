import torch
from torch.utils.data import DataLoader

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


def test_catalog_stream_dataset_splits_work_across_workers() -> None:
    ds = CatalogStreamDataset(range(8))
    loader = DataLoader(ds, num_workers=2, batch_size=None)
    got = sorted(int(x) for x in loader)
    assert got == list(range(8))


def test_catalog_stream_dataset_supports_partition_factory() -> None:
    ds = CatalogStreamDataset(lambda: iter(range(4)))
    assert list(ds) == [0, 1, 2, 3]
    assert list(ds) == [0, 1, 2, 3]
