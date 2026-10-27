import torch
import pytest

from torchsky.sphere import MOC, SparseHealpixMap
from torchsky.sphere import healpix as hp
from torchsky.sphere.moc import nest2uniq


def test_high_nside_healpix_centers_neighbors_and_interp_weights() -> None:
    nside = 1024
    pixels_nest = torch.tensor(
        [0, 1, 2, 12345, 999999, hp.nside2npix(nside) - 1],
        dtype=torch.int64,
    )
    lon, lat = hp.pix2ang(nside, pixels_nest, nest=True, lonlat=True)

    torch.testing.assert_close(
        hp.ang2pix(nside, lon, lat, nest=True, lonlat=True),
        pixels_nest,
    )
    pixels_ring = hp.nest2ring(nside, pixels_nest)
    torch.testing.assert_close(
        hp.ang2pix(nside, lon, lat, nest=False, lonlat=True),
        pixels_ring,
    )

    neighbors = hp.neighbors(nside, pixels_nest, nest=True)
    assert neighbors.shape == (pixels_nest.numel(), 8)
    assert torch.all(
        (neighbors == -1) | ((neighbors >= 0) & (neighbors < hp.nside2npix(nside)))
    )

    pix4, weights4 = hp.get_interp_weights(nside, lon, lat, nest=True, lonlat=True)
    assert pix4.shape == (4, pixels_nest.numel())
    assert weights4.shape == (4, pixels_nest.numel())
    torch.testing.assert_close(
        weights4.sum(dim=0),
        torch.ones(pixels_nest.numel(), dtype=weights4.dtype),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert torch.all((pix4 >= 0) & (pix4 < hp.nside2npix(nside)))


def test_high_nside_sparse_lookup_moc_contains_and_catalog_filter() -> None:
    nside = 1024
    pixels = torch.tensor([0, 17, 123456, 999999, 7_000_000], dtype=torch.int64)
    values = torch.arange(pixels.numel(), dtype=torch.float64) + 10.0
    sparse = SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=nside,
        nest=True,
        nside_coverage=64,
        sentinel=-1.0,
    )

    query = torch.tensor([0, 1, 17, 7_000_000], dtype=torch.int64)
    torch.testing.assert_close(
        sparse.get_values(query),
        torch.tensor([10.0, -1.0, 11.0, 14.0], dtype=torch.float64),
    )
    assert sparse.coverage_mask.shape == (hp.nside2npix(64),)
    assert int(sparse.coverage_mask.sum().item()) > 0

    moc = MOC(nest2uniq(nside, pixels))
    expected = torch.tensor([True, False, True, True])
    torch.testing.assert_close(moc.contains_pixels(nside, query, nest=True), expected)

    lon, lat = hp.pix2ang(nside, query, nest=True, lonlat=True)
    filtered_lon, filtered_lat = moc.filter_catalog(lon, lat)
    torch.testing.assert_close(filtered_lon, lon[expected])
    torch.testing.assert_close(filtered_lat, lat[expected])


@pytest.mark.parametrize("order", [28, 29])
def test_order_28_29_sparse_ring_nested_roundtrips(order: int) -> None:
    nside = 1 << order
    npface = nside * nside
    npix = 12 * npface
    nested = torch.tensor(
        [0, 1, npface - 1, npface, 5 * npface + 1234567, npix - 1],
        dtype=torch.int64,
    )

    ring = hp.nest2ring(nside, nested)

    assert nested.numel() == 6
    assert ring.numel() == 6
    torch.testing.assert_close(hp.ring2nest(nside, ring), nested)
    assert torch.all((ring >= 0) & (ring < npix))

    ring_input = torch.tensor(
        [0, 1, 2 * nside * (nside - 1), npix // 2, npix - 2, npix - 1],
        dtype=torch.int64,
    )
    torch.testing.assert_close(
        hp.nest2ring(nside, hp.ring2nest(nside, ring_input)), ring_input
    )


def test_order_29_sparse_centers_and_neighbors_do_not_require_full_map() -> None:
    nside = 1 << 29
    npix = 12 * nside * nside
    assert hp.nside2npix(nside) == 3_458_764_513_820_540_928
    nested = torch.tensor(
        [0, 1, nside * nside - 1, nside * nside, npix - 1], dtype=torch.int64
    )

    lon, lat = hp.pix2ang(nside, nested, nest=True, lonlat=True)
    back = hp.ang2pix(nside, lon, lat, nest=True, lonlat=True)
    neighbors = hp.neighbors(nside, nested, nest=True)

    torch.testing.assert_close(back, nested)
    assert lon.shape == lat.shape == nested.shape == (5,)
    assert neighbors.shape == (5, 8)
    assert torch.all((neighbors == -1) | ((neighbors >= 0) & (neighbors < npix)))


def test_order_30_is_rejected_without_allocating() -> None:
    nside = 1 << 30
    sparse = torch.tensor([0], dtype=torch.int64)

    for operation in (
        lambda: hp.nside2npix(nside),
        lambda: hp.ring2nest(nside, sparse),
        lambda: hp.nest2ring(nside, sparse),
        lambda: hp.neighbors(nside, sparse, nest=True),
    ):
        with pytest.raises(ValueError, match=r"2\*\*29"):
            operation()
