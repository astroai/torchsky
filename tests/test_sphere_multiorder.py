import math

import pytest
import torch

from torchsky.catalogs import HealpixIndex
from torchsky.sphere import MOC, MultiOrderHealpixMap, SparseHealpixMap
from torchsky.sphere import healpix
from torchsky.sphere.moc import _order_from_uniq, nest2uniq, uniq2nest


def _brute_overlap_fraction(moc: MOC, uniq: torch.Tensor) -> torch.Tensor:
    query_orders = _order_from_uniq(uniq)
    max_order = max(moc.max_order, int(query_orders.max().item()))
    nside = 1 << max_order
    _, pixels = uniq2nest(uniq)
    result = []
    for pixel, order in zip(pixels.tolist(), query_orders.tolist(), strict=True):
        factor = 1 << (2 * (max_order - order))
        children = torch.arange(pixel * factor, (pixel + 1) * factor)
        result.append(moc.contains_pixels(nside, children).to(torch.float64).mean())
    return torch.stack(result)


def test_moc_uniq_overlap_fraction_matches_low_order_brute_force() -> None:
    moc = MOC(
        torch.cat(
            (
                nest2uniq(2, torch.tensor([0, 5, 6])),
                nest2uniq(4, torch.tensor([31, 32])),
            )
        )
    )
    query = torch.cat(
        (
            nest2uniq(1, torch.tensor([0, 1, 2])),
            nest2uniq(2, torch.tensor([0, 4, 5, 7, 8])),
            nest2uniq(4, torch.tensor([31, 33])),
        )
    )
    expected = _brute_overlap_fraction(moc, query)
    torch.testing.assert_close(moc.overlap_fraction(query), expected)
    torch.testing.assert_close(moc.intersects_uniq(query), expected > 0.0)
    torch.testing.assert_close(moc.contains_uniq(query), expected == 1.0)


def test_moc_footprint_overlap_delegates_to_moc_intersection() -> None:
    left = MOC(nest2uniq(2, torch.tensor([0, 1])))
    overlapping = MOC(nest2uniq(8, torch.tensor([3, 32])))
    disjoint = MOC(nest2uniq(2, torch.tensor([8, 9])))
    assert left.overlaps(overlapping)
    assert not left.overlaps(disjoint)


def test_multiorder_map_canonicalizes_uniq_and_rejects_overlap() -> None:
    uniq = nest2uniq(2, torch.tensor([8, 4, 7]))
    values = torch.tensor([80.0, 40.0, 70.0], requires_grad=True)
    multi = MultiOrderHealpixMap(uniq, values)
    torch.testing.assert_close(multi.uniq, torch.sort(uniq).values)
    torch.testing.assert_close(multi.values, torch.tensor([40.0, 70.0, 80.0]))

    parent = nest2uniq(1, torch.tensor([0]))
    child = nest2uniq(2, torch.tensor([1]))
    with pytest.raises(ValueError, match="non-overlapping"):
        MultiOrderHealpixMap(
            torch.cat((parent, child)), torch.tensor([1.0, 2.0])
        )


def test_multiorder_density_integral_matches_low_order_oracle() -> None:
    uniq = nest2uniq(torch.tensor([1, 2]), torch.tensor([0, 4]))
    values = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    multi = MultiOrderHealpixMap(uniq, values, values_are_densities=True)
    footprint = MOC(nest2uniq(2, torch.tensor([0, 4])))
    expected = 2.0 * (math.pi / 3.0) / 4.0 + 3.0 * (math.pi / 12.0)
    result = multi.integral(footprint)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
    result.backward()
    torch.testing.assert_close(
        values.grad,
        torch.tensor([math.pi / 12.0, math.pi / 12.0], dtype=torch.float64),
    )


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
def test_multiorder_lookup_preserves_device_and_value_autograd(
    device_type: str,
) -> None:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device_type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        pytest.skip("MPS not available")
    device = torch.device(device_type)
    dtype = torch.float32 if device_type == "mps" else torch.float64
    pixels = torch.tensor([4, 7], dtype=torch.int64, device=device)
    uniq = nest2uniq(2, pixels)
    values = torch.tensor([3.0, 5.0], dtype=dtype, device=device, requires_grad=True)
    multi = MultiOrderHealpixMap(uniq, values)
    lon, lat = healpix.pix2ang(2, pixels, nest=True, lonlat=True)
    result, valid = multi.get_values_pos(lon, lat, return_valid=True)
    assert result.device.type == device_type
    assert valid.device.type == device_type
    torch.testing.assert_close(result, values)
    assert valid.tolist() == [True, True]
    footprint = MOC(uniq)
    (result.sum() + multi.integral(footprint)).backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, 2.0))


def test_multiorder_integer_integral_keeps_fractional_overlap() -> None:
    uniq = nest2uniq(1, torch.tensor([0]))
    multi = MultiOrderHealpixMap(uniq, torch.tensor([3], dtype=torch.int64))
    footprint = MOC(nest2uniq(2, torch.tensor([0])))
    torch.testing.assert_close(
        multi.integral(footprint), torch.tensor(0.75, dtype=torch.float64)
    )


def test_multiorder_columns_and_fixed_sparse_roundtrip() -> None:
    uniq = nest2uniq(8, torch.tensor([5, 9, 20]))
    values = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    multi = MultiOrderHealpixMap.from_columns(
        {"UNIQ": uniq, "PROBDENSITY": values},
        value_column="PROBDENSITY",
        values_are_densities=True,
    )
    columns = multi.to_columns(value_column="PROBDENSITY")
    torch.testing.assert_close(columns["UNIQ"], uniq)
    torch.testing.assert_close(columns["PROBDENSITY"], values)

    sparse = multi.to_sparse_map(nside_coverage=2)
    back = MultiOrderHealpixMap.from_sparse_map(sparse)
    torch.testing.assert_close(back.uniq, uniq)
    torch.testing.assert_close(back.values, values)


def test_multiorder_high_order_queries_do_not_expand_dense_pixels(monkeypatch) -> None:
    uniq = nest2uniq(1 << 29, torch.tensor([123456789], dtype=torch.int64))
    values = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    multi = MultiOrderHealpixMap(uniq, values, values_are_densities=True)
    footprint = MOC(uniq)

    def fail_dense(*args, **kwargs):
        raise AssertionError("dense expansion is forbidden")

    monkeypatch.setattr(MOC, "_flat_pixels", fail_dense)
    torch.testing.assert_close(
        footprint.overlap_fraction(uniq), torch.ones(1, dtype=torch.float64)
    )
    result = multi.integral(footprint)
    assert result.item() > 0.0
    result.backward()
    assert values.grad is not None


def test_multiorder_overlap_scales_with_stored_cells_not_sky_size() -> None:
    order = 20
    count = 8192
    pixels = torch.arange(count, dtype=torch.int64)
    uniq = nest2uniq(1 << order, pixels)
    values = torch.ones(count, dtype=torch.float64)
    multi = MultiOrderHealpixMap(uniq, values)
    footprint = MOC(nest2uniq(1, torch.tensor([0], dtype=torch.int64)))
    overlap = multi.overlap_fraction(footprint)
    assert overlap.shape == (count,)
    assert bool(torch.all(overlap == 1.0))
    torch.testing.assert_close(
        multi.integral(footprint), torch.tensor(float(count), dtype=torch.float64)
    )


def test_healpix_index_moc_lookup_matches_point_oracle() -> None:
    lon = torch.tensor([10.0, 10.2, 40.0, 80.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 0.1, -5.0, 20.0], dtype=torch.float64)
    index = HealpixIndex.build(lon, lat, nside=8, nest=False)
    fine_pixels = healpix.ang2pix(
        32, lon[[0, 2]], lat[[0, 2]], nest=True, lonlat=True
    )
    footprint = MOC(nest2uniq(32, fine_pixels))
    expected = torch.nonzero(footprint.contains(lon, lat), as_tuple=False).reshape(-1)
    torch.testing.assert_close(index.lookup_moc(footprint), expected)
    assert index.occupied_moc().contains_rows(index).all()


def test_mocpy_probability_and_json_interchange_optional() -> None:
    mocpy = pytest.importorskip("mocpy")
    table_module = pytest.importorskip("astropy.table")
    uniq = nest2uniq(torch.tensor([1, 2]), torch.tensor([0, 4]))
    values = torch.tensor([2.0, 3.0], dtype=torch.float64)
    multi = MultiOrderHealpixMap(uniq, values, values_are_densities=True)
    footprint = MOC(nest2uniq(2, torch.tensor([0, 4])))
    peer = footprint.to_mocpy()
    table = table_module.Table(
        [uniq.numpy().astype("uint64"), values.numpy()],
        names=("UNIQ", "PROBDENSITY"),
    )
    expected = peer.probability_in_multiordermap(table)
    torch.testing.assert_close(
        multi.integral(footprint), torch.tensor(expected, dtype=torch.float64)
    )
    assert MOC.from_mocpy(peer) == footprint
    assert isinstance(peer, mocpy.MOC)


def test_mhealpy_interchange_optional() -> None:
    pytest.importorskip("mhealpy")
    uniq = nest2uniq(torch.tensor([1, 2]), torch.tensor([1, 8]))
    values = torch.tensor([0.25, 0.75], dtype=torch.float64)
    multi = MultiOrderHealpixMap(uniq, values, values_are_densities=True)
    back = MultiOrderHealpixMap.from_mhealpy(multi.to_mhealpy())
    torch.testing.assert_close(back.uniq, multi.uniq)
    torch.testing.assert_close(back.values, multi.values)
    assert back.values_are_densities


def test_healsparse_interchange_optional() -> None:
    pytest.importorskip("healsparse")
    sparse = SparseHealpixMap.from_pixels(
        torch.tensor([5, 9, 20]),
        torch.tensor([1.0, 2.0, 3.0]),
        nside_sparse=8,
        nside_coverage=2,
        nest=True,
        sentinel=-1.0,
    )
    back = SparseHealpixMap.from_healsparse(sparse.to_healsparse())
    torch.testing.assert_close(back.pixels, sparse.pixels)
    torch.testing.assert_close(back.values, sparse.values)


def test_cdshealpix_interchange_optional() -> None:
    pytest.importorskip("cdshealpix")
    uniq = nest2uniq(8, torch.tensor([5, 9, 20]))
    values = torch.tensor([1.0, 2.0, 3.0])
    multi = MultiOrderHealpixMap(uniq, values)
    back = MultiOrderHealpixMap.from_cdshealpix(multi.to_cdshealpix())
    torch.testing.assert_close(back.uniq, multi.uniq)
    torch.testing.assert_close(back.values, multi.values)
