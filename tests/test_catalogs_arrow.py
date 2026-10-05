"""Arrow-streaming crossmatch tests."""

from __future__ import annotations

import pyarrow as pa
import pytest
import torch

from torchsky.catalogs import (
    Catalog,
    CatalogStreamDataset,
    crossmatch_sky,
    crossmatch_sky_from_arrow,
    crossmatch_sky_streamed,
)


def test_arrow_crossmatch_preserves_input_row_indices_after_tiling() -> None:
    table = pa.table({"ra": [220.0, 10.0], "dec": [0.0, 0.0]})
    result = crossmatch_sky_from_arrow(table, table, radius_arcsec=1.0)
    assert result.left_index.tolist() == [0, 1]
    assert result.right_index.tolist() == [0, 1]


@pytest.mark.parametrize("source_kind", ["table", "reader", "file"])
@pytest.mark.parametrize("find", ["best", "all"])
def test_arrow_crossmatch_keeps_rows_and_native_ids_across_batches(
    source_kind, find, tmp_path
) -> None:
    left = pa.table({"ra": [220.0, 10.0, 220.0, 359.9999, 60.0], "dec": [0.0] * 5})
    right = pa.table(
        {
            "ra": [60.0, 220.0, 10.0, 0.0001, 220.0],
            "dec": [0.0] * 5,
            "source_id": pa.array([2**60 + i for i in range(5)], type=pa.uint64()),
        }
    )
    sources = []
    for name, table in [("left", left), ("right", right)]:
        if source_kind == "reader":
            sources.append(
                pa.RecordBatchReader.from_batches(table.schema, table.to_batches(2))
            )
        elif source_kind == "file":
            path = tmp_path / f"{name}.arrow"
            with pa.ipc.new_file(str(path), table.schema) as writer:
                writer.write_table(table, max_chunksize=2)
            sources.append(path)
        else:
            sources.append(table)
    result = crossmatch_sky_from_arrow(
        *sources, radius_arcsec=1.0, batch_size=2, find=find
    )
    if find == "best":
        expected_left, expected_right = [0, 1, 2, 3, 4], [1, 2, 1, 3, 0]
    else:
        expected_left, expected_right = [0, 0, 1, 2, 2, 3, 4], [1, 4, 2, 1, 4, 3, 0]
    assert result.left_index.tolist() == expected_left
    assert result.right_index.tolist() == expected_right
    assert right["source_id"].take(result.right_index.numpy()).to_pylist() == [
        2**60 + index for index in expected_right
    ]
    assert result.separation_arcsec.max().item() < 1.0


@pytest.mark.parametrize("empty_side", ["left", "right", "both"])
def test_arrow_crossmatch_empty_catalogue_returns_no_pairs(empty_side) -> None:
    empty = pa.table(
        {"ra": pa.array([], type=pa.float64()), "dec": pa.array([], type=pa.float64())}
    )
    nonempty = pa.table({"ra": [10.0], "dec": [0.0]})
    result = crossmatch_sky_from_arrow(
        empty if empty_side != "right" else nonempty,
        empty if empty_side != "left" else nonempty,
        radius_arcsec=1.0,
    )
    assert result.left_index.numel() == result.right_index.numel() == 0


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), float("inf")])
def test_arrow_crossmatch_rejects_invalid_radius(radius) -> None:
    table = pa.table({"ra": [10.0], "dec": [0.0]})
    with pytest.raises(ValueError):
        crossmatch_sky_from_arrow(table, table, radius_arcsec=radius)


def _random_ra_dec(n: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    ra = torch.rand(n, generator=rng) * 360.0
    dec = torch.asin(torch.rand(n, generator=rng) * 2.0 - 1.0) * 180.0 / torch.pi
    return ra.to(torch.float64), dec.to(torch.float64)


def test_crossmatch_sky_streamed_matches_single_batch() -> None:
    left_ra, left_dec = _random_ra_dec(100, seed=10)
    right_ra, right_dec = _random_ra_dec(200, seed=20)
    expected = crossmatch_sky(left_ra, left_dec, right_ra, right_dec, radius_arcsec=5.0)
    got = crossmatch_sky_streamed(
        [(left_ra, left_dec)],
        right_ra,
        right_dec,
        radius_arcsec=5.0,
    )
    assert torch.equal(expected.left_index, got.left_index)
    assert torch.equal(expected.right_index, got.right_index)
    torch.testing.assert_close(expected.separation_arcsec, got.separation_arcsec)


def test_crossmatch_sky_streamed_matches_multi_batch() -> None:
    left_ra, left_dec = _random_ra_dec(100, seed=10)
    right_ra, right_dec = _random_ra_dec(200, seed=20)
    expected = crossmatch_sky(left_ra, left_dec, right_ra, right_dec, radius_arcsec=5.0)
    batches = [
        (left_ra[:40], left_dec[:40]),
        (left_ra[40:70], left_dec[40:70]),
        (left_ra[70:], left_dec[70:]),
    ]
    got = crossmatch_sky_streamed(
        batches,
        right_ra,
        right_dec,
        radius_arcsec=5.0,
    )
    assert torch.equal(expected.left_index, got.left_index)
    assert torch.equal(expected.right_index, got.right_index)
    torch.testing.assert_close(expected.separation_arcsec, got.separation_arcsec)


def test_crossmatch_sky_streamed_empty_batches() -> None:
    right_ra, right_dec = _random_ra_dec(50, seed=30)
    got = crossmatch_sky_streamed(
        [],
        right_ra,
        right_dec,
        radius_arcsec=5.0,
    )
    assert got.left_index.numel() == 0
    assert got.right_index.numel() == 0


def test_crossmatch_sky_from_arrow_matches_tensor() -> None:
    left_ra, left_dec = _random_ra_dec(80, seed=42)
    right_ra, right_dec = _random_ra_dec(120, seed=84)
    expected = crossmatch_sky(left_ra, left_dec, right_ra, right_dec, radius_arcsec=3.0)

    left_table = pa.table({"ra": left_ra.numpy(), "dec": left_dec.numpy()})
    right_table = pa.table({"ra": right_ra.numpy(), "dec": right_dec.numpy()})

    got = crossmatch_sky_from_arrow(
        left_table,
        right_table,
        radius_arcsec=3.0,
    )
    assert torch.equal(expected.left_index, got.left_index)
    assert torch.equal(expected.right_index, got.right_index)
    torch.testing.assert_close(expected.separation_arcsec, got.separation_arcsec)


def test_crossmatch_sky_from_arrow_with_custom_columns() -> None:
    left_ra, left_dec = _random_ra_dec(20, seed=7)
    right_ra, right_dec = _random_ra_dec(30, seed=14)
    expected = crossmatch_sky(
        left_ra, left_dec, right_ra, right_dec, radius_arcsec=10.0
    )

    left_table = pa.table({"RA": left_ra.numpy(), "DEC": left_dec.numpy()})
    right_table = pa.table({"RA": right_ra.numpy(), "DEC": right_dec.numpy()})

    got = crossmatch_sky_from_arrow(
        left_table,
        right_table,
        radius_arcsec=10.0,
        left_columns=("RA", "DEC"),
        right_columns=("RA", "DEC"),
    )
    assert torch.equal(expected.left_index, got.left_index)
    assert torch.equal(expected.right_index, got.right_index)


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


def test_crossmatch_sky_from_arrow_with_file(tmp_path) -> None:
    left_ra, left_dec = _random_ra_dec(30, seed=1)
    right_ra, right_dec = _random_ra_dec(40, seed=2)
    expected = crossmatch_sky(left_ra, left_dec, right_ra, right_dec, radius_arcsec=5.0)

    left_path = str(tmp_path / "left.arrow")
    right_path = str(tmp_path / "right.arrow")
    for path, r, d in [
        (left_path, left_ra, left_dec),
        (right_path, right_ra, right_dec),
    ]:
        table = pa.table({"ra": r.numpy(), "dec": d.numpy()})
        with pa.ipc.new_file(pa.OSFile(path, "wb"), table.schema) as writer:
            writer.write_table(table)

    got = crossmatch_sky_from_arrow(left_path, right_path, radius_arcsec=5.0)
    assert torch.equal(expected.left_index, got.left_index)
    assert torch.equal(expected.right_index, got.right_index)
    torch.testing.assert_close(expected.separation_arcsec, got.separation_arcsec)


def test_crossmatch_sky_from_arrow_large_right(tmp_path) -> None:
    """Verify the Arrow streaming path with a right catalog that doesn't fit
    in GPU memory (500K rows, CPU-only test)."""
    left_ra, left_dec = _random_ra_dec(500, seed=42)
    right_ra, right_dec = _random_ra_dec(500_000, seed=84)
    expected = crossmatch_sky(left_ra, left_dec, right_ra, right_dec, radius_arcsec=2.0)

    right_path = str(tmp_path / "right_large.arrow")
    table = pa.table({"ra": right_ra.numpy(), "dec": right_dec.numpy()})
    with pa.ipc.new_file(pa.OSFile(right_path, "wb"), table.schema) as writer:
        writer.write_table(table)

    got = crossmatch_sky_from_arrow(
        pa.table({"ra": left_ra.numpy(), "dec": left_dec.numpy()}),
        right_path,
        radius_arcsec=2.0,
    )
    assert torch.equal(expected.left_index, got.left_index)
    assert torch.equal(expected.right_index, got.right_index)
    torch.testing.assert_close(expected.separation_arcsec, got.separation_arcsec)


def test_catalog_from_arrow_roundtrip() -> None:
    ra, dec = _random_ra_dec(32, seed=3)
    mag = torch.linspace(12.0, 18.0, 32)
    table = pa.table({"ra": ra.numpy(), "dec": dec.numpy(), "mag": mag.numpy()})
    cat = Catalog.from_arrow(table, columns=("mag",))
    torch.testing.assert_close(cat.lon, ra)
    torch.testing.assert_close(cat.lat, dec)
    torch.testing.assert_close(cat.columns["mag"], mag)
