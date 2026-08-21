import torch

from torchsky.catalogs import Catalog


def test_catalog_is_not_a_dataframe() -> None:
    assert not hasattr(Catalog, "from_polars")
    assert not hasattr(Catalog, "to_polars")
    assert not hasattr(Catalog, "iter_partitions")
    cat = Catalog(
        lon=torch.tensor([10.0], dtype=torch.float64),
        lat=torch.tensor([0.0], dtype=torch.float64),
    )
    assert len(cat) == 1
