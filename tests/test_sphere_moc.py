import json

import numpy as np
import pytest
import torch
from torchsky.sphere import SphericalCap, SphericalPolygon
from torchsky.sphere import healpix as hp
from torchsky.sphere.geom import PixelizedRegion
from torchsky.sphere.moc import (
    MOC,
    nest2uniq,
    _max_order_from_uniq,
    _uniq_to_ranges,
    _ranges_to_uniq,
    _normalize_moc,
    _union_ranges,
)
from torchsky.sphere import HealSparseMap


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
    smap = HealSparseMap.convert_healpix_map(
        torch.zeros(12), nside_coverage=1, sentinel=0
    )
    moc = MOC.from_sparse_map(smap)
    assert moc.uniq.numel() == 0


def test_moc_from_sparse_map_converts_ring_pixels_to_nested_uniq() -> None:
    nside = 8
    pix_nest = torch.tensor([0, 5, 17], dtype=torch.int64)
    pix_ring = hp.nest2ring(nside, pix_nest)
    smap = HealSparseMap.from_pixels(
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
    smap = HealSparseMap.from_pixels(
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
    smap = HealSparseMap.from_pixels(
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
    smap = HealSparseMap.from_pixels(
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
    smap = HealSparseMap.from_pixels(
        pixels, nside_sparse=4, nside_coverage=1, values=values
    )

    moc = MOC.from_threshold(smap, 100.0, op=">")
    assert moc.uniq.numel() == 0


def test_moc_read_fits_fallback(tmp_path):
    """Non-standard UNIQ column name: first int64 column is used."""
    import torch
    import torchsky

    path = str(tmp_path / "test.fits")
    torchsky.write(
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

    smap = HealSparseMap.from_moc(moc, nside=nside, nside_coverage=2)

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


def test_mocpy_contains_union_area_and_polygon_parity() -> None:
    mocpy = pytest.importorskip("mocpy")
    import astropy.units as u

    lon = torch.tensor([10.0, 12.0, 11.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float64)
    ts_poly = MOC.from_polygon(lon, lat, max_order=6)

    mocpy_cls = mocpy.MOC
    if hasattr(mocpy_cls, "from_polygon"):
        ref = mocpy_cls.from_polygon(
            lon.numpy() * u.deg, lat.numpy() * u.deg, max_depth=6
        )
    else:
        pytest.skip("mocpy.MOC.from_polygon unavailable")

    qlon = torch.tensor([11.0, 50.0], dtype=torch.float64)
    qlat = torch.tensor([0.4, 40.0], dtype=torch.float64)
    got = ts_poly.contains(qlon, qlat).cpu().numpy()
    if hasattr(ref, "contains_lonlat"):
        exp = np.asarray(ref.contains_lonlat(qlon.numpy() * u.deg, qlat.numpy() * u.deg))
    elif hasattr(ref, "contains"):
        exp = np.asarray(ref.contains(qlon.numpy() * u.deg, qlat.numpy() * u.deg))
    else:
        pytest.skip("mocpy contains API unavailable")
    np.testing.assert_array_equal(got.reshape(-1), np.asarray(exp).reshape(-1))

    a = MOC.from_circle(0.0, 0.0, 2.0, max_order=5)
    b = MOC.from_circle(3.0, 0.0, 2.0, max_order=5)
    u_ts = a.union(b)
    i_ts = a.intersection(b)
    four_pi = 4.0 * float(np.pi)
    if hasattr(mocpy_cls, "from_cone"):
        ra = mocpy_cls.from_cone(
            lon=0.0 * u.deg, lat=0.0 * u.deg, radius=2.0 * u.deg, max_depth=5
        )
        rb = mocpy_cls.from_cone(
            lon=3.0 * u.deg, lat=0.0 * u.deg, radius=2.0 * u.deg, max_depth=5
        )
        uref = ra.union(rb)
        iref = ra.intersection(rb)
        if hasattr(uref, "sky_fraction"):
            assert abs(u_ts.area / four_pi - float(uref.sky_fraction)) < 5e-3
        if hasattr(iref, "sky_fraction"):
            assert abs(i_ts.area / four_pi - float(iref.sky_fraction)) < 5e-3


def test_mocpy_fits_roundtrip_optional(tmp_path) -> None:
    mocpy = pytest.importorskip("mocpy")
    moc = MOC.from_circle(20.0, -10.0, 1.5, max_order=6)
    path = tmp_path / "moc.fits"
    moc.write_fits(str(path), overwrite=True)
    if not hasattr(mocpy.MOC, "from_fits"):
        pytest.skip("mocpy.MOC.from_fits unavailable")
    loaded = mocpy.MOC.from_fits(str(path))
    assert loaded is not None
    back = MOC.read_fits(str(path))
    torch.testing.assert_close(back.uniq.sort().values, moc.uniq.sort().values)


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
