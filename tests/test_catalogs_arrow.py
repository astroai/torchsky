"""Arrow batches into catalog tensors."""

from __future__ import annotations

import pyarrow as pa
import torch

from torchsky.catalogs import Catalog, CatalogStreamDataset


def _random_ra_dec(n: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    ra = torch.rand(n, generator=rng) * 360.0
    dec = torch.asin(torch.rand(n, generator=rng) * 2.0 - 1.0) * 180.0 / torch.pi
    return ra.to(torch.float64), dec.to(torch.float64)


def test_catalog_stream_dataset_from_arrow_yields_tensor_pairs(tmp_path) -> None:
    ra = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    dec = torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float64)
    table = pa.table({"ra": ra.numpy(), "dec": dec.numpy()})

    ds = CatalogStreamDataset.from_arrow(table, batch_size=2)
    batches = list(iter(ds))
    assert len(batches) == 2
    ra0, dec0 = batches[0]
    ra1, dec1 = batches[1]
    assert torch.equal(ra0, ra[:2])
    assert torch.equal(dec0, dec[:2])
    assert torch.equal(ra1, ra[2:])
    assert torch.equal(dec1, dec[2:])


def test_catalog_stream_dataset_from_arrow_file(tmp_path) -> None:
    ra = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    dec = torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float64)
    table = pa.table({"ra": ra.numpy(), "dec": dec.numpy()})
    path = str(tmp_path / "test.arrow")
    with pa.ipc.new_file(pa.OSFile(path, "wb"), table.schema) as writer:
        writer.write_table(table)

    ds = CatalogStreamDataset.from_arrow(path, batch_size=2)
    batches = list(iter(ds))
    assert len(batches) == 2
    ra0, dec0 = batches[0]
    assert torch.equal(ra0, ra[:2])
    assert torch.equal(dec0, dec[:2])


def test_catalog_from_arrow_roundtrip() -> None:
    ra, dec = _random_ra_dec(32, seed=3)
    mag = torch.linspace(12.0, 18.0, 32)
    table = pa.table({"ra": ra.numpy(), "dec": dec.numpy(), "mag": mag.numpy()})
    cat = Catalog.from_arrow(table, columns=("mag",))
    torch.testing.assert_close(cat.lon, ra)
    torch.testing.assert_close(cat.lat, dec)
    torch.testing.assert_close(cat.columns["mag"], mag)
