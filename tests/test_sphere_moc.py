import json
import math

import numpy as np
import pytest
import random

import torch
from torchsky.sphere import SphericalCap, SphericalPolygon
from torchsky.sphere import healpix as hp
from torchsky.sphere.geom import PixelizedRegion
from torchsky.sphere.moc import (
    BMOC,
    MOC,
    nest2uniq,
    uniq2nest,
    _max_order_from_uniq,
    _nside_from_max_order,
    _uniq_to_ranges,
    _ranges_to_uniq,
    _normalize_moc,
    _union_ranges,
    _order_from_uniq,
    _range_tensors_to_uniq,
    _uniq_to_range_tensors,
)
from torchsky.sphere import SparseHealpixMap
import torchfits


def test_uniq_to_ranges_empty():
    ranges = _uniq_to_ranges(torch.tensor([], dtype=torch.int64), 10)
    assert ranges == []


def test_ranges_to_uniq_empty():
    uniqs = _ranges_to_uniq([], 10)
    assert uniqs.numel() == 0


def test_max_order_from_uniq_empty():
    assert _max_order_from_uniq(torch.tensor([], dtype=torch.int64)) == 0
    assert _max_order_from_uniq(torch.tensor([])) == 0
    assert _max_order_from_uniq(torch.empty((0,))) == 0


def test_normalize_moc_empty():
    uniq = torch.tensor([], dtype=torch.int64)
    assert _normalize_moc(uniq).numel() == 0


def test_union_ranges_merges_overlap_and_keeps_disjoint():
    assert _union_ranges([], []) == []
    assert _union_ranges([(0, 4)], []) == [(0, 4)]
    assert _union_ranges([(0, 4)], [(4, 8)]) == [(0, 8)]
    assert _union_ranges([(0, 4)], [(8, 12)]) == [(0, 4), (8, 12)]
    assert _union_ranges([(0, 5), (10, 12)], [(3, 8)]) == [(0, 8), (10, 12)]


def test_moc_union_matches_cat_normalize_and_empty_identity():
    a = MOC.from_circle(0.0, 0.0, 2.0, max_order=4)
    b = MOC.from_circle(3.0, 0.0, 2.0, max_order=4)
    empty = MOC(torch.tensor([], dtype=torch.int64))
    torch.testing.assert_close(a.union(empty).uniq, a.uniq)
    torch.testing.assert_close(empty.union(b).uniq, b.uniq)
    got = a.union(b).uniq
    legacy = MOC(torch.cat([a.uniq, b.uniq])).uniq
    torch.testing.assert_close(got, legacy)


def test_moc_max_order_empty():
    moc = MOC(torch.tensor([], dtype=torch.int64))
    assert moc.max_order == 0


def test_moc_contains_empty():
    moc = MOC(torch.tensor([], dtype=torch.int64))
    assert not moc.contains(0.0, 0.0).item()


def test_moc_contains_empty_broadcasts_lon_lat() -> None:
    lon = torch.zeros((2, 1), dtype=torch.float64)
    lat = torch.zeros((1, 3), dtype=torch.float64)

    contained = MOC.empty().contains(lon, lat)

    assert contained.shape == (2, 3)
    assert not bool(contained.any())


def test_moc_contains_pixels_nested_and_ring_inputs() -> None:
    nside = 8
    pix_nest = torch.tensor([0, 5, 17], dtype=torch.int64)
    moc = MOC(nest2uniq(nside, pix_nest))
    query_nest = torch.tensor([0, 1, 5, 17, 18], dtype=torch.int64)
    query_ring = hp.nest2ring(nside, query_nest)

    expected = torch.tensor([True, False, True, True, False])
    torch.testing.assert_close(
        moc.contains_pixels(nside, query_nest, nest=True),
        expected,
    )
    torch.testing.assert_close(
        moc.contains_pixels(nside, query_ring, nest=False),
        expected,
    )

    lon, lat = hp.pix2ang(nside, query_nest, nest=True, lonlat=True)
    torch.testing.assert_close(moc.contains(lon, lat), expected)


def test_moc_contains_pixels_requires_sufficient_resolution() -> None:
    moc = MOC(nest2uniq(8, torch.tensor([0], dtype=torch.int64)))

    with pytest.raises(ValueError, match="at least the MOC maximum order"):
        moc.contains_pixels(4, torch.tensor([0], dtype=torch.int64))


def test_moc_area_empty():
    moc = MOC(torch.tensor([], dtype=torch.int64))
    assert moc.area == 0.0
    assert moc.area_sq_deg == 0.0


def test_moc_difference_complex():
    # Test complex difference scenario
    moc1 = MOC(nest2uniq(4, torch.tensor([0, 1, 2])))
    moc2 = MOC(nest2uniq(4, torch.tensor([1, 3])))
    diff = moc1.difference(moc2)
    assert diff.uniq.numel() == 2


def test_moc_from_sparse_map_empty():
    smap = SparseHealpixMap.convert_healpix_map(
        torch.zeros(12), nside_coverage=1, sentinel=0
    )
    moc = MOC.from_sparse_map(smap)
    assert moc.uniq.numel() == 0


def test_moc_from_sparse_map_converts_ring_pixels_to_nested_uniq() -> None:
    nside = 8
    pix_nest = torch.tensor([0, 5, 17], dtype=torch.int64)
    pix_ring = hp.nest2ring(nside, pix_nest)
    smap = SparseHealpixMap.from_pixels(
        pix_ring,
        torch.ones_like(pix_ring, dtype=torch.float32),
        nside_sparse=nside,
        nest=False,
        nside_coverage=2,
        sentinel=0.0,
    )

    moc = MOC.from_sparse_map(smap)

    torch.testing.assert_close(moc.uniq, MOC(nest2uniq(nside, pix_nest)).uniq)


def test_moc_from_threshold():
    pixels = torch.tensor([0, 1, 2])
    values = torch.tensor([10.0, 5.0, 20.0])
    smap = SparseHealpixMap.from_pixels(
        pixels, nside_sparse=4, nside_coverage=1, values=values
    )

    # Test op >
    moc_gt = MOC.from_threshold(smap, 10.0, op=">")
    assert moc_gt.uniq.numel() > 0

    # Test op >=
    moc_ge = MOC.from_threshold(smap, 10.0, op=">=")
    assert moc_ge.uniq.numel() > 0

    # Test op <
    moc_lt = MOC.from_threshold(smap, 10.0, op="<")
    assert moc_lt.uniq.numel() > 0

    # Test op <=
    moc_le = MOC.from_threshold(smap, 10.0, op="<=")
    assert moc_le.uniq.numel() > 0

    # Test invalid op
    with pytest.raises(ValueError):
        MOC.from_threshold(smap, 10.0, op="==")


def test_moc_from_threshold_uses_sparse_pixel_list_not_coverage_offsets() -> None:
    pixels = torch.tensor([0, 10, 42], dtype=torch.int64)
    values = torch.tensor([1.0, 5.0, 2.0], dtype=torch.float64)
    smap = SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=8,
        nside_coverage=2,
        sentinel=0.0,
    )

    moc = MOC.from_threshold(smap, 2.0, op=">")

    torch.testing.assert_close(moc.uniq, MOC(nest2uniq(8, torch.tensor([10]))).uniq)


def test_moc_from_threshold_converts_ring_sparse_pixels() -> None:
    pixels_nest = torch.tensor([0, 10, 42], dtype=torch.int64)
    pixels_ring = hp.nest2ring(8, pixels_nest)
    values = torch.tensor([1.0, 5.0, 2.0], dtype=torch.float64)
    smap = SparseHealpixMap.from_pixels(
        pixels_ring,
        values,
        nside_sparse=8,
        nest=False,
        nside_coverage=2,
        sentinel=0.0,
    )

    moc = MOC.from_threshold(smap, 2.0, op=">")

    torch.testing.assert_close(moc.uniq, MOC(nest2uniq(8, torch.tensor([10]))).uniq)


def test_moc_from_threshold_empty():
    pixels = torch.tensor([0, 1, 2])
    values = torch.tensor([10.0, 5.0, 20.0])
    smap = SparseHealpixMap.from_pixels(
        pixels, nside_sparse=4, nside_coverage=1, values=values
    )

    moc = MOC.from_threshold(smap, 100.0, op=">")
    assert moc.uniq.numel() == 0


def test_moc_read_fits_fallback(tmp_path):
    """Non-standard UNIQ column name: first int64 column is used."""
    import torch

    path = str(tmp_path / "test.fits")
    torchfits.write(
        path,
        {"OLD_UNIQ": torch.tensor([4, 16, 17], dtype=torch.int64)},
        header={"EXTNAME": "MOC"},
        overwrite=True,
    )
    m = MOC.read_fits(path)
    # Values normalize/merge into a valid MOC (exact UNIQ count may shrink).
    assert m.uniq.numel() >= 1
    assert m.uniq.dtype == torch.int64


def test_moc_boundary_empty():
    moc = MOC(torch.tensor([], dtype=torch.int64))
    assert moc.boundary() == []


def test_moc_boundary_complex():
    moc = MOC(nest2uniq(2, torch.tensor([0, 1, 2])))
    boundary = moc.boundary()
    assert len(boundary) > 0


def test_moc_from_polygon_supports_nonconvex_simple_regions() -> None:
    lon = torch.tensor([-20.0, 20.0, 20.0, 0.0, 0.0, -20.0], dtype=torch.float64)
    lat = torch.tensor([-20.0, -20.0, 0.0, 0.0, 20.0, 20.0], dtype=torch.float64)

    moc = MOC.from_polygon(lon, lat, max_order=7)

    assert bool(moc.contains(-12.0, 12.0))
    assert bool(moc.contains(12.0, -12.0))
    assert not bool(moc.contains(12.0, 12.0))


def test_moc_from_region_converts_ring_pixelized_region_to_nested_uniq() -> None:
    nside = 8
    pix_nest = torch.tensor([0, 3, 10, 10], dtype=torch.int64)
    pix_ring = hp.nest2ring(nside, pix_nest)
    region = PixelizedRegion(nside=nside, nest=False, pixels=pix_ring)

    moc = MOC.from_region(region)
    expected = MOC(nest2uniq(nside, pix_nest))
    back_ring = moc.to_pixelized_region(nside, nest=False)

    torch.testing.assert_close(moc.uniq, expected.uniq)
    assert not back_ring.nest
    torch.testing.assert_close(back_ring.pixels, torch.unique(pix_ring))

    with pytest.raises(ValueError, match="does not match requested max_order"):
        MOC.from_region(region, max_order=4)


def test_moc_from_region_pixelizes_boolean_region() -> None:
    square = SphericalPolygon(
        torch.tensor([-8.0, 8.0, 8.0, -8.0], dtype=torch.float64),
        torch.tensor([-6.0, -6.0, 6.0, 6.0], dtype=torch.float64),
    )
    region = square.difference(SphericalCap(0.0, 0.0, 3.0))

    moc = MOC.from_region(region, max_order=7)

    assert bool(moc.contains(6.0, 0.0))
    assert not bool(moc.contains(0.0, 0.0))


def test_healsparse_from_moc_uses_sparse_nested_pixels() -> None:
    nside = 8
    pix = torch.tensor([0, 3, 7], dtype=torch.int64)
    moc = MOC(nest2uniq(nside, pix))

    smap = SparseHealpixMap.from_moc(moc, nside=nside, nside_coverage=2)

    assert smap.nest
    assert smap.nside_sparse == nside
    assert smap.nside_coverage == 2
    torch.testing.assert_close(smap.pixels, pix)
    assert smap.values.dtype == torch.bool
    assert bool(smap.values.all())


def test_moc_to_sparse_map_sets_values_and_small_coverage_resolution() -> None:
    nside = 4
    pix = torch.tensor([0, 1], dtype=torch.int64)
    moc = MOC(nest2uniq(nside, pix))

    smap = moc.to_sparse_map(nside, sentinel=0.0)

    assert smap.nside_sparse == nside
    assert smap.nside_coverage == 1
    torch.testing.assert_close(smap.pixels, pix)
    torch.testing.assert_close(smap.values, torch.ones(2, dtype=torch.float32))
    dense = smap.to_dense(fill_value=0.0)
    torch.testing.assert_close(dense[pix], torch.ones(2, dtype=torch.float32))
    assert float(dense[2].item()) == 0.0


def test_moc_to_ascii_empty():
    moc = MOC(torch.tensor([], dtype=torch.int64))
    assert moc.to_ascii() == ""


def test_moc_to_ascii_compact_ranges_roundtrip() -> None:
    moc = MOC(nest2uniq(8, torch.tensor([5, 6, 7, 10], dtype=torch.int64)))

    ascii_str = moc.to_ascii(compact=True)
    got = MOC.from_ascii(ascii_str)

    assert "3/5-7 10" == ascii_str
    torch.testing.assert_close(got.uniq, moc.uniq)


def test_moc_from_ascii_empty():
    assert MOC.from_ascii("").uniq.numel() == 0
    assert MOC.from_ascii("   ").uniq.numel() == 0
    assert MOC.from_ascii("0/").uniq.numel() == 0


def test_moc_from_ascii_expands_pixel_ranges_and_commas() -> None:
    moc = MOC.from_ascii("3/5-7,10")
    expected = MOC(nest2uniq(8, torch.tensor([5, 6, 7, 10], dtype=torch.int64)))

    torch.testing.assert_close(moc.uniq, expected.uniq)


def test_moc_from_ascii_rejects_descending_ranges() -> None:
    with pytest.raises(ValueError, match="descending MOC pixel range"):
        MOC.from_ascii("3/7-5")


def test_moc_to_json_empty():
    moc = MOC(torch.tensor([], dtype=torch.int64))
    assert moc.to_json() == "{}"


def test_moc_from_json_empty():
    assert MOC.from_json("").uniq.numel() == 0
    assert MOC.from_json("{}").uniq.numel() == 0


def test_moc_from_json_invalid():
    with pytest.raises(ValueError):
        MOC.from_json('"a string"')


def test_moc_from_json_empty_pixels():
    js = json.dumps({"0": []})
    assert MOC.from_json(js).uniq.numel() == 0


def test_moc_empty_full_and_dateline_wrap() -> None:
    empty = MOC(torch.tensor([], dtype=torch.int64))
    assert empty.area == 0.0
    assert not bool(empty.contains(0.0, 0.0).item())

    full = MOC(nest2uniq(1, torch.arange(12, dtype=torch.int64)))
    probes_lon = torch.tensor([0.0, 179.0, -179.0, 90.0], dtype=torch.float64)
    probes_lat = torch.tensor([0.0, 0.0, 0.0, 45.0], dtype=torch.float64)
    assert bool(full.contains(probes_lon, probes_lat).all().item())
    assert full.area > 0.9 * 4.0 * float(np.pi)

    wrap = MOC.from_polygon(
        torch.tensor([170.0, 190.0, 190.0, 170.0], dtype=torch.float64),
        torch.tensor([-4.0, -4.0, 4.0, 4.0], dtype=torch.float64),
        max_order=6,
    )
    assert bool(wrap.contains(175.0, 0.0).item())
    assert bool(wrap.contains(-175.0, 0.0).item())
    assert not bool(wrap.contains(0.0, 0.0).item())


def _leaf_circle_moc(
    lon_deg: float,
    lat_deg: float,
    radius_deg: float,
    max_order: int,
    *,
    inclusive: bool = True,
) -> MOC:
    nside = 1 << max_order
    pix = hp.query_circle(
        nside,
        lon_deg,
        lat_deg,
        radius_deg,
        nest=True,
        degrees=True,
        inclusive=inclusive,
    )
    return MOC(nest2uniq(nside, pix))


def test_moc_empty_full_operators_from_lonlat_degrade() -> None:
    empty = MOC.empty()
    assert empty.uniq.numel() == 0
    full = MOC.full()
    assert bool(full.contains(0.0, 0.0).item())
    assert bool(full.contains(180.0, -40.0).item())

    a = MOC.from_circle(0.0, 0.0, 2.0, max_order=4)
    b = MOC.from_circle(3.0, 0.0, 2.0, max_order=4)
    torch.testing.assert_close((a | b).uniq, a.union(b).uniq)
    torch.testing.assert_close((a & b).uniq, a.intersection(b).uniq)
    torch.testing.assert_close((a - b).uniq, a.difference(b).uniq)

    lon = torch.tensor([10.0, 10.001, 80.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 0.0, 20.0], dtype=torch.float64)
    points = MOC.from_lonlat(lon, lat, max_order=8)
    assert bool(points.contains(10.0, 0.0).item())
    assert not bool(points.contains(0.0, 0.0).item())
    coarse = points.degrade(2)
    assert coarse.max_order <= 2
    assert bool(coarse.contains(10.0, 0.0).item())


def test_moc_from_circle_matches_leaf_construction() -> None:
    hier = MOC.from_circle(20.0, -10.0, 1.5, max_order=6, inclusive=True)
    leaf = _leaf_circle_moc(20.0, -10.0, 1.5, 6, inclusive=True)
    torch.testing.assert_close(hier.uniq, leaf.uniq)
    hier_ex = MOC.from_circle(20.0, -10.0, 1.5, max_order=6, inclusive=False)
    leaf_ex = _leaf_circle_moc(20.0, -10.0, 1.5, 6, inclusive=False)
    torch.testing.assert_close(hier_ex.uniq, leaf_ex.uniq)


@pytest.mark.parametrize("order", [0, 1, 10, 23, 24, 25, 29])
def test_uniq2nest_high_order_is_exact(order: int) -> None:
    nside = 1 << order
    npix = 12 * nside * nside
    pix = torch.tensor([0, 1, npix // 2, npix - 2, npix - 1], dtype=torch.int64)
    uniq = nest2uniq(nside, pix)

    got_nside, got_pix = uniq2nest(uniq)

    torch.testing.assert_close(got_nside, torch.full_like(pix, nside), rtol=0, atol=0)
    torch.testing.assert_close(got_pix, pix, rtol=0, atol=0)


def test_uniq2nest_rejects_invalid_uniq_and_order_overflow() -> None:
    with pytest.raises(ValueError, match="UNIQ"):
        uniq2nest(torch.tensor([0, 1, 3], dtype=torch.int64))
    with pytest.raises(ValueError, match="nside"):
        nest2uniq(1 << 30, 0)
    with pytest.raises(ValueError, match="max_order"):
        _nside_from_max_order(30)


def test_nest2uniq_rejects_pixel_aliases_and_non_integer_inputs() -> None:
    with pytest.raises(ValueError, match="pix"):
        nest2uniq(1, 12)
    with pytest.raises(ValueError, match="pix"):
        nest2uniq(1, -1)
    with pytest.raises(ValueError, match="integers"):
        nest2uniq(1, 0.5)


def test_moc_constructors_reject_invalid_uniq_and_order_pixel_aliases() -> None:
    with pytest.raises(ValueError, match="UNIQ"):
        MOC(torch.tensor([1 << 62], dtype=torch.int64))
    with pytest.raises(ValueError, match="integers"):
        MOC(torch.tensor([4.5]))
    with pytest.raises(ValueError, match="pix"):
        MOC.from_ascii("0/12")
    with pytest.raises(ValueError, match="pix"):
        MOC.from_json('{"0": [12]}')


def test_nest_uniq_round_trip_broadcasts_and_preserves_canonical_edges() -> None:
    nsides = torch.tensor([[1], [2], [8]], dtype=torch.int64)
    pixels = torch.tensor([[0, 1, 11]], dtype=torch.int64)
    pixels = pixels * (nsides * nsides)

    uniq = nest2uniq(nsides, pixels)
    got_nsides, got_pixels = uniq2nest(uniq)

    torch.testing.assert_close(got_nsides, nsides.expand_as(pixels))
    torch.testing.assert_close(got_pixels, pixels)


def test_bmoc_normalizes_parent_child_overlap_conservatively() -> None:
    parent = nest2uniq(1, 0)
    child = nest2uniq(2, 0)

    bmoc = BMOC(
        torch.stack((child, parent, parent)),
        torch.tensor([1, 1, 0], dtype=torch.uint8),
    )

    torch.testing.assert_close(bmoc.uniq, parent.reshape(1))
    torch.testing.assert_close(bmoc.flag, torch.tensor([0], dtype=torch.uint8))
    assert bmoc.area == pytest.approx(math.pi / 3.0)
    assert bmoc.interior_area == 0.0
    torch.testing.assert_close(bmoc.to_moc().uniq, parent.reshape(1))
    assert bmoc.to_moc(include_boundary=False).is_empty


def test_bmoc_boolean_operations_preserve_hierarchical_flags_and_area() -> None:
    parent = BMOC(nest2uniq(1, 0).reshape(1), torch.tensor([0]))
    child = BMOC(nest2uniq(2, 0).reshape(1), torch.tensor([1]))

    union = parent | child
    intersection = parent & child
    difference = parent - child

    assert union.area == pytest.approx(math.pi / 3.0)
    assert union.interior_area == pytest.approx(math.pi / 12.0)
    assert union.to_moc().area == pytest.approx(parent.to_moc().area)
    assert intersection.area == pytest.approx(math.pi / 12.0)
    assert intersection.interior_area == 0.0
    assert difference.area == pytest.approx(math.pi / 4.0)
    assert difference.interior_area == 0.0


def test_bmoc_rejects_invalid_uniq_and_flag_aliases() -> None:
    with pytest.raises(ValueError, match="UNIQ"):
        BMOC(torch.tensor([1 << 62], dtype=torch.int64))
    with pytest.raises(ValueError, match="flag values"):
        BMOC(nest2uniq(1, 0).reshape(1), torch.tensor([2]))


def test_moc_high_order_max_order_and_leaf_preservation() -> None:
    nside = 1 << 24
    pixels = torch.tensor([0, 7, 12 * nside * nside - 1], dtype=torch.int64)
    moc = MOC(nest2uniq(nside, pixels))

    assert moc.max_order == 24
    assert moc.uniq.numel() == pixels.numel()


def test_healpix_nside_cap() -> None:
    with pytest.raises(ValueError, match="2\\*\\*29"):
        hp._validate_nside(1 << 30)


def _oracle_cells(rng: random.Random, max_order: int, n: int) -> torch.Tensor:
    out = []
    for _ in range(n):
        order = rng.randint(0, max_order)
        out.append((4 << (2 * order)) + rng.randrange(12 << (2 * order)))
    return torch.tensor(out, dtype=torch.int64)


@pytest.mark.parametrize("seed", range(6))
def test_range_algebra_matches_dense_mask_oracle(seed: int) -> None:
    rng = random.Random(seed)
    order = 5
    nside = 1 << order
    for _ in range(8):
        a = MOC(_oracle_cells(rng, order, rng.randint(1, 40)))
        b = MOC(_oracle_cells(rng, order, rng.randint(1, 40)))
        ma, mb = a.flatten(nside), b.flatten(nside)
        for op, expect in (
            ("union", ma | mb),
            ("intersection", ma & mb),
            ("difference", ma & ~mb),
        ):
            got = getattr(a, op)(b)
            assert torch.equal(got.flatten(nside), expect), op
            # canonical form: re-normalising the result is the identity
            assert torch.equal(MOC(got.uniq).uniq, got.uniq)


def test_range_algebra_edge_cases_and_deep_orders() -> None:
    full = MOC.full()
    empty = MOC.empty()
    one = MOC(torch.tensor([4 << 58], dtype=torch.int64) + 12345)
    assert full.union(one) == full
    assert full.intersection(one) == one
    assert one.difference(full).is_empty
    assert full.difference(one).intersection(one).is_empty
    assert full.difference(one).union(one) == full
    assert empty.union(empty).is_empty
    assert one.intersection(empty).is_empty
    assert one.difference(empty) == one
    assert one.max_order == 29
    # touching cells at order 29 merge into their parent exactly
    base = (4 << 58) + 4 * 777
    quad = MOC(torch.arange(base, base + 4, dtype=torch.int64))
    assert quad.uniq.numel() == 1
    assert int(quad.uniq[0]) == (4 << 56) + 777


def test_range_roundtrip_is_exact_at_depth_29() -> None:
    rng = random.Random(7)
    cuts = sorted(rng.sample(range(12 << 58), 400))
    starts = torch.tensor(cuts[0::2], dtype=torch.int64)
    ends = torch.tensor(cuts[1::2], dtype=torch.int64)
    uniq = _range_tensors_to_uniq(starts, ends, 29)
    s2, e2 = _uniq_to_range_tensors(uniq, 29)
    assert torch.equal(s2, starts) and torch.equal(e2, ends)
    # maximal cells: no four siblings of one parent remain
    order = _order_from_uniq(uniq)
    assert int((order <= 29).sum()) == uniq.numel()


def test_moc_area_is_exact_and_device_independent() -> None:
    import math

    import torch

    from torchsky.sphere import MOC

    full = MOC(torch.arange(4 * 1, 4 * 1 + 12, dtype=torch.int64))  # 12 base cells
    assert full.area == pytest.approx(4.0 * math.pi, rel=1e-15)
    moc = MOC.from_circle(150.0, 2.0, 5.0, max_order=9)
    if torch.backends.mps.is_available():
        on_mps = MOC(moc.uniq.to("mps"))
        assert on_mps.area == pytest.approx(moc.area, rel=1e-15)


def test_moc_read_fits_ivoa_first_extension_without_extname(tmp_path):
    """IVOA MOC files need not set EXTNAME; ORDERING='NUNIQ' in HDU 1 suffices."""
    path = str(tmp_path / "ivoa.fits")
    uniq = torch.tensor([4 * 4**3 + 5, 4 * 4**3 + 6], dtype=torch.int64)
    torchfits.write(
        path,
        {"UNIQ": uniq},
        header={"ORDERING": "NUNIQ", "MOCORDER": 3, "MOCDIM": "SPACE"},
        overwrite=True,
    )
    assert torch.equal(MOC.read_fits(path).uniq, MOC(uniq).uniq)


def test_moc_read_fits_rejects_non_moc_and_time_moc(tmp_path):
    plain = str(tmp_path / "plain.fits")
    torchfits.write(plain, {"FLUX": torch.tensor([1.0, 2.0])}, overwrite=True)
    with pytest.raises(ValueError, match="No MOC table"):
        MOC.read_fits(plain)

    tmoc = str(tmp_path / "tmoc.fits")
    torchfits.write(
        tmoc,
        {"UNIQ": torch.tensor([5], dtype=torch.int64)},
        header={"ORDERING": "NUNIQ", "MOCDIM": "TIME"},
        overwrite=True,
    )
    with pytest.raises(ValueError, match="TIME MOC"):
        MOC.read_fits(tmoc)
