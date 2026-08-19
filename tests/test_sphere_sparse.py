import numpy as np
import pytest
import torch

from torchsky.sphere.sparse import SkyMaskPipe, SparseHealpixMap
from torchsky.sphere import healpix as hp
from torchsky.sphere import SphericalCap
from torchsky.sphere.geom import PixelizedRegion
from torchsky.sphere.moc import MOC, nest2uniq


def test_sparse_from_dense_roundtrip() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    dense = torch.full((npix,), float(hp.UNSEEN), dtype=torch.float64)
    keep = torch.arange(0, npix, 7, dtype=torch.int64)
    dense[keep] = torch.linspace(0.0, 1.0, steps=keep.numel(), dtype=torch.float64)

    sp = SparseHealpixMap.from_dense(dense, nside=nside, nest=False)
    back = sp.to_dense()
    torch.testing.assert_close(back, dense)
    assert sp.pixels.numel() == keep.numel()


def test_sparse_from_pixels_canonicalizes_unsorted_duplicates() -> None:
    pixels = torch.tensor([5, 2, 5, 1], dtype=torch.int64)
    values = torch.tensor([50.0, 20.0, 55.0, 10.0], dtype=torch.float64)

    sp = SparseHealpixMap.from_pixels(
        pixels, values, nside_sparse=8, sentinel=-1.0
    )

    torch.testing.assert_close(sp.pixels, torch.tensor([1, 2, 5], dtype=torch.int64))
    torch.testing.assert_close(
        sp.values,
        torch.tensor([10.0, 20.0, 55.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        sp.get_values(torch.tensor([5, 2, 1, 7], dtype=torch.int64)),
        torch.tensor([55.0, 20.0, 10.0, -1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        sp.get_values(torch.tensor([[5, 2], [1, 7]], dtype=torch.int64)),
        torch.tensor([55.0, 20.0, 10.0, -1.0], dtype=torch.float64),
    )


@pytest.mark.parametrize("nest", [True, False])
def test_sparse_get_values_coverage_blocks_match_searchsorted(nest: bool) -> None:
    pixels = torch.tensor([0, 1, 2, 33, 34, 4095], dtype=torch.int64)
    values = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=torch.float64)
    blocked = SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=64,
        nest=nest,
        nside_coverage=32,
        sentinel=-1.0,
    )
    plain = SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=64,
        nest=nest,
        sentinel=-1.0,
    )
    query = torch.tensor([0, 1, 10, 33, 8192, -1, 49152], dtype=torch.int64)
    torch.testing.assert_close(blocked.get_values(query), plain.get_values(query))
    assert blocked._block_index is not None


def test_sparse_get_values_coverage_blocks_multiband_match_searchsorted() -> None:
    pixels = torch.tensor([0, 1, 2, 33, 34, 4095], dtype=torch.int64)
    values = torch.tensor(
        [
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        ],
        dtype=torch.float64,
    )
    blocked = SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=64,
        nest=True,
        nside_coverage=32,
        sentinel=-1.0,
    )
    plain = SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=64,
        nest=True,
        sentinel=-1.0,
    )
    query = torch.tensor([0, 1, 10, 33, 8192, -1, 49152], dtype=torch.int64)
    torch.testing.assert_close(blocked.get_values(query), plain.get_values(query))


@pytest.mark.parametrize(
    "dtype,fill,values",
    [
        (torch.float32, -1.0, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
        (torch.int64, -1, [1, 2, 3, 4, 5, 6]),
        (torch.int32, -1, [1, 2, 3, 4, 5, 6]),
        (torch.bool, False, [True, True, True, True, True, True]),
    ],
)
def test_sparse_get_values_coverage_blocks_match_searchsorted_dtypes(
    dtype: torch.dtype, fill: object, values: list[object]
) -> None:
    pixels = torch.tensor([0, 1, 2, 33, 34, 4095], dtype=torch.int64)
    vals = torch.tensor(values, dtype=dtype)
    blocked = SparseHealpixMap.from_pixels(
        pixels,
        vals,
        nside_sparse=64,
        nest=True,
        nside_coverage=32,
        sentinel=fill,
    )
    plain = SparseHealpixMap.from_pixels(
        pixels,
        vals,
        nside_sparse=64,
        nest=True,
        sentinel=fill,
    )
    query = torch.tensor([0, 1, 10, 33, 8192, -1, 49152], dtype=torch.int64)
    got = blocked.get_values(query)
    exp = plain.get_values(query)
    assert got.dtype == dtype
    if dtype == torch.bool:
        assert torch.equal(got, exp)
    else:
        torch.testing.assert_close(got, exp, atol=0, rtol=0)


@pytest.mark.parametrize("nest", [True, False])
@pytest.mark.parametrize("pess", [False, True])
def test_sparse_ud_grade_coverage_blocks_match_pixel_list(
    nest: bool, pess: bool
) -> None:
    pixels = torch.tensor([0, 1, 2, 33, 34, 4095], dtype=torch.int64)
    values = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=torch.float64)
    blocked = SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=64,
        nest=nest,
        nside_coverage=32,
        sentinel=-1.0,
    )
    plain = SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=64,
        nest=nest,
        sentinel=-1.0,
    )
    for nside_out in (32, 16, 128):
        got = blocked.ud_grade(nside_out, pess=pess)
        exp = plain.ud_grade(nside_out, pess=pess)
        torch.testing.assert_close(got.to_dense(), exp.to_dense(), atol=1e-12, rtol=0.0)
        assert got.nside_coverage == min(32, nside_out)
        q = torch.arange(0, hp.nside2npix(nside_out), 17, dtype=torch.int64)
        torch.testing.assert_close(got.get_values(q), exp.get_values(q), atol=1e-12, rtol=0.0)


def test_sparse_ud_grade_coverage_blocks_multiband_match_pixel_list() -> None:
    pixels = torch.tensor([0, 1, 2, 33, 34, 4095], dtype=torch.int64)
    values = torch.tensor(
        [
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        ],
        dtype=torch.float64,
    )
    blocked = SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=64,
        nest=True,
        nside_coverage=32,
        sentinel=-1.0,
    )
    plain = SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=64,
        nest=True,
        sentinel=-1.0,
    )
    got = blocked.ud_grade(32, pess=False)
    exp = plain.ud_grade(32, pess=False)
    torch.testing.assert_close(got.to_dense(), exp.to_dense(), atol=1e-12, rtol=0.0)


def test_sparse_from_pixels_canonicalizes_multiband_values() -> None:
    pixels = torch.tensor([4, 1, 4, 2], dtype=torch.int64)
    values = torch.tensor(
        [[40.0, 10.0, 44.0, 20.0], [400.0, 100.0, 444.0, 200.0]],
        dtype=torch.float64,
    )

    sp = SparseHealpixMap.from_pixels(
        pixels, values, nside_sparse=8, sentinel=-1.0
    )

    torch.testing.assert_close(sp.pixels, torch.tensor([1, 2, 4], dtype=torch.int64))
    torch.testing.assert_close(
        sp.values,
        torch.tensor(
            [[10.0, 20.0, 44.0], [100.0, 200.0, 444.0]],
            dtype=torch.float64,
        ),
    )


def test_sparse_update_values_pix_keeps_latest_duplicate() -> None:
    sp = SparseHealpixMap.from_pixels(
        torch.tensor([2, 5], dtype=torch.int64),
        torch.tensor([20.0, 50.0], dtype=torch.float64),
        nside_sparse=8,
        sentinel=-1.0,
    )

    sp.update_values_pix(
        torch.tensor([5, 1, 5], dtype=torch.int64),
        torch.tensor([51.0, 10.0, 55.0], dtype=torch.float64),
    )

    torch.testing.assert_close(sp.pixels, torch.tensor([1, 2, 5], dtype=torch.int64))
    torch.testing.assert_close(
        sp.values,
        torch.tensor([10.0, 20.0, 55.0], dtype=torch.float64),
    )


def test_sparse_update_values_pix_keeps_latest_multiband_duplicate() -> None:
    sp = SparseHealpixMap.from_pixels(
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([[20.0], [200.0]], dtype=torch.float64),
        nside_sparse=8,
        sentinel=-1.0,
    )

    sp.update_values_pix(
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([[10.0, 22.0], [100.0, 222.0]], dtype=torch.float64),
    )

    torch.testing.assert_close(sp.pixels, torch.tensor([1, 2], dtype=torch.int64))
    torch.testing.assert_close(
        sp.values,
        torch.tensor([[10.0, 22.0], [100.0, 222.0]], dtype=torch.float64),
    )


def test_sparse_to_moc_and_pixelized_region_preserve_ordering() -> None:
    nside = 8
    pix_nest = torch.tensor([0, 3, 10], dtype=torch.int64)
    pix_ring = hp.nest2ring(nside, pix_nest)
    sp = SparseHealpixMap.from_pixels(
        pix_ring,
        torch.ones(pix_ring.numel(), dtype=torch.float32),
        nside_sparse=nside,
        nest=False,
        nside_coverage=2,
        sentinel=0.0,
    )

    moc = sp.to_moc()
    torch.testing.assert_close(moc.uniq, MOC(nest2uniq(nside, pix_nest)).uniq)

    ring_region = sp.to_pixelized_region()
    assert not ring_region.nest
    torch.testing.assert_close(ring_region.pixels, torch.unique(pix_ring))

    nested_region = sp.to_pixelized_region(nest=True)
    assert nested_region.nest
    torch.testing.assert_close(nested_region.pixels, torch.unique(pix_nest))

    upsampled = sp.to_pixelized_region(nside=16, nest=True)
    expected_children = (
        pix_nest.unsqueeze(1) * 4
        + torch.arange(4, dtype=torch.int64).unsqueeze(0)
    ).reshape(-1)
    torch.testing.assert_close(upsampled.pixels, torch.sort(expected_children)[0])


def test_sparse_from_region_preserves_pixelized_region_values_and_ordering() -> None:
    nside = 8
    pix_nest = torch.tensor([0, 3, 10, 10], dtype=torch.int64)
    pix_ring = hp.nest2ring(nside, pix_nest)
    region = PixelizedRegion(nside=nside, nest=False, pixels=pix_ring)

    smap = SparseHealpixMap.from_region(
        region,
        nside=nside,
        nest=False,
        nside_coverage=2,
        value=2.5,
        sentinel=0.0,
        dtype=torch.float32,
        metadata={"SURVEY": "synthetic"},
    )

    assert not smap.nest
    assert smap.nside_coverage == 2
    assert smap.metadata == {"SURVEY": "synthetic"}
    torch.testing.assert_close(smap.pixels, torch.unique(pix_ring))
    torch.testing.assert_close(
        smap.values,
        torch.full((torch.unique(pix_ring).numel(),), 2.5, dtype=torch.float32),
    )

    nested = SparseHealpixMap.from_region(
        region,
        nside=nside,
        nest=True,
        value=True,
        sentinel=False,
    )
    assert nested.nest
    assert nested.values.dtype == torch.bool
    assert bool(nested.values.all())
    torch.testing.assert_close(nested.pixels, torch.unique(pix_nest))


def test_sparse_from_region_pixelizes_exact_region_without_dense_map() -> None:
    cap = SphericalCap(0.0, 0.0, 5.0)

    smap = SparseHealpixMap.from_region(
        cap,
        nside=64,
        nside_coverage=8,
        value=1.0,
        sentinel=0.0,
        dtype=torch.float32,
    )

    assert smap.nest
    assert smap.nside_coverage == 8
    assert smap.coverage_map is not None
    assert smap.pixels.numel() > 0
    torch.testing.assert_close(smap.values, torch.ones_like(smap.values))
    moc = smap.to_moc()
    assert bool(moc.contains(0.0, 0.0))
    assert not bool(moc.contains(180.0, 0.0))


def test_sparse_nearest_interpolation_on_centers() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    dense = torch.full((npix,), float(hp.UNSEEN), dtype=torch.float64)
    keep = torch.arange(0, npix, 5, dtype=torch.int64)
    dense[keep] = keep.to(torch.float64)

    sp = SparseHealpixMap.from_dense(dense, nside=nside, nest=False)
    lon, lat = hp.pix2ang(nside, keep, nest=False, lonlat=True)
    got = sp.interpolate(lon, lat, method="nearest")
    torch.testing.assert_close(got, dense[keep], atol=0.0, rtol=0.0)


def test_sparse_bilinear_matches_dense_when_fully_covered() -> None:
    nside = 16
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(5)
    dense = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)
    sp = SparseHealpixMap.from_dense(dense, nside=nside, nest=False)

    lon = torch.from_numpy(rng.uniform(0.0, 360.0, size=64)).to(torch.float64)
    lat = torch.from_numpy(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=64)))).to(
        torch.float64
    )

    got = sp.interpolate(lon, lat, method="bilinear")
    exp = hp.get_interp_val(dense, lon, lat, nest=False, lonlat=True)
    torch.testing.assert_close(got, exp, atol=1e-12, rtol=0.0)


def test_sparse_ud_grade_workflow() -> None:
    nside = 16
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(17)
    dense = torch.full((npix,), float(hp.UNSEEN), dtype=torch.float64)
    keep = torch.from_numpy(rng.choice(npix, size=npix // 3, replace=False)).to(
        torch.int64
    )
    dense[keep] = torch.from_numpy(rng.normal(size=keep.numel())).to(torch.float64)

    sp = SparseHealpixMap.from_dense(dense, nside=nside, nest=False)
    sp2 = sp.ud_grade(8)
    assert sp2.nside == 8
    assert sp2.pixels.numel() > 0


def test_sparse_ud_grade_preserves_valid_coverage_hierarchy() -> None:
    sp = SparseHealpixMap.from_pixels(
        torch.tensor([0, 1, 2, 63], dtype=torch.int64),
        torch.ones(4, dtype=torch.float64),
        nside_sparse=16,
        nest=True,
        nside_coverage=4,
        metadata={"SURVEY": "synthetic"},
    )

    for nside_out, expected_cov in ((32, 4), (8, 4), (4, 4), (2, 2)):
        got = sp.ud_grade(nside_out)
        assert got.nside_coverage == expected_cov
        assert got.coverage_map is not None
        assert got.coverage_mask.shape == (hp.nside2npix(expected_cov),)
        assert got.metadata == {"SURVEY": "synthetic"}
        assert got.metadata is not sp.metadata


def test_sparse_multiband_support() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    dense = torch.full((3, npix), float(hp.UNSEEN), dtype=torch.float64)
    keep = torch.arange(0, npix, 9, dtype=torch.int64)
    for b in range(3):
        dense[b, keep] = (b + 1) * 10.0 + keep.to(torch.float64)

    sp = SparseHealpixMap.from_dense(dense, nside=nside, nest=False)
    assert sp.values.shape[0] == 3

    lon, lat = hp.pix2ang(nside, keep[:12], nest=False, lonlat=True)
    got = sp.interpolate(lon, lat, method="nearest")
    exp = dense[:, keep[:12]]
    torch.testing.assert_close(got, exp, atol=0.0, rtol=0.0)


def test_sparse_ud_grade_full_coverage_matches_dense_ud_grade() -> None:
    rng = np.random.default_rng(1234)
    for nest in (False, True):
        nside = 8
        npix = hp.nside2npix(nside)
        dense = torch.from_numpy(rng.normal(size=npix)).to(torch.float64)
        sp = SparseHealpixMap.from_dense(dense, nside=nside, nest=nest)
        order = "NEST" if nest else "RING"
        for nside_out in (4, 16):
            for power in (None, 1.0, -2.0):
                got = sp.ud_grade(nside_out, power=power).to_dense()
                exp = hp.ud_grade(
                    dense,
                    nside_out,
                    pess=False,
                    order_in=order,
                    order_out=order,
                    power=power,
                )
                torch.testing.assert_close(got, exp, atol=1e-12, rtol=0.0)


def test_sparse_ud_grade_partial_coverage_pess_semantics() -> None:
    # NSIDE=4 -> NSIDE=2 (parent_mult=4 in NEST hierarchy).
    pixels = torch.tensor([12, 13, 28, 29, 30, 31], dtype=torch.int64)
    values = torch.tensor([1.0, 3.0, 10.0, 14.0, 18.0, 22.0], dtype=torch.float64)
    sp = SparseHealpixMap(nside=4, nest=True, pixels=pixels, values=values)

    got = sp.ud_grade(2, pess=False)
    torch.testing.assert_close(
        got.pixels, torch.tensor([3, 7], dtype=torch.int64), atol=0, rtol=0
    )
    torch.testing.assert_close(
        got.values, torch.tensor([2.0, 16.0], dtype=torch.float64), atol=1e-12, rtol=0.0
    )

    got_pess = sp.ud_grade(2, pess=True)
    torch.testing.assert_close(
        got_pess.pixels, torch.tensor([7], dtype=torch.int64), atol=0, rtol=0
    )
    torch.testing.assert_close(
        got_pess.values, torch.tensor([16.0], dtype=torch.float64), atol=1e-12, rtol=0.0
    )

    got_power = sp.ud_grade(2, pess=False, power=1.0)
    torch.testing.assert_close(
        got_power.values,
        torch.tensor(
            [1.0, 8.0], dtype=torch.float64
        ),  # scale by (nside_out/nside_in)^power = 0.5
        atol=1e-12,
        rtol=0.0,
    )


def test_sparse_coverage_mask_and_any_coverage_mode() -> None:
    nside = 4
    npix = hp.nside2npix(nside)
    dense = torch.full((2, npix), float(hp.UNSEEN), dtype=torch.float64)
    dense[0, 3] = 1.0
    dense[1, 7] = 2.0

    sp_all = SparseHealpixMap.from_dense(
        dense, nside=nside, nest=True, coverage_mode="all"
    )
    assert sp_all.pixels.numel() == 0

    sp_any = SparseHealpixMap.from_dense(
        dense, nside=nside, nest=True, coverage_mode="any"
    )
    torch.testing.assert_close(
        sp_any.pixels, torch.tensor([3, 7], dtype=torch.int64), atol=0, rtol=0
    )
    cov = sp_any.coverage_mask
    assert cov.dtype == torch.bool
    assert bool(cov[3]) and bool(cov[7]) and (int(cov.sum().item()) == 2)


def test_sparse_direct_construction_builds_coverage_map_after_canonicalization() -> None:
    sp = SparseHealpixMap(
        nside=8,
        nest=True,
        pixels=torch.tensor([20, 5, 5], dtype=torch.int64),
        values=torch.tensor([2.0, 1.0, 3.0], dtype=torch.float64),
        nside_coverage=2,
    )

    torch.testing.assert_close(sp.pixels, torch.tensor([5, 20], dtype=torch.int64))
    torch.testing.assert_close(
        sp.values, torch.tensor([3.0, 2.0], dtype=torch.float64)
    )
    assert sp.coverage_map is not None
    torch.testing.assert_close(
        torch.nonzero(sp.coverage_map >= 0, as_tuple=False).flatten(),
        torch.tensor([0, 1], dtype=torch.int64),
    )
    assert int(sp.coverage_map[0].item()) == 0
    assert int(sp.coverage_map[1].item()) == 1


def test_sparse_coverage_map_keeps_first_index_for_duplicate_parents() -> None:
    sp = SparseHealpixMap.from_pixels(
        torch.tensor([3, 2, 21, 20], dtype=torch.int64),
        torch.tensor([30.0, 20.0, 210.0, 200.0], dtype=torch.float64),
        nside_sparse=8,
        nest=True,
        nside_coverage=2,
    )

    assert sp.coverage_map is not None
    torch.testing.assert_close(sp.pixels, torch.tensor([2, 3, 20, 21]))
    assert int(sp.coverage_map[0].item()) == 0
    assert int(sp.coverage_map[1].item()) == 2


def test_sparse_ring_coverage_mask_uses_nested_coverage_pixels() -> None:
    nside = 8
    nside_cov = 2
    ratio2 = (nside // nside_cov) ** 2
    all_nest = torch.arange(hp.nside2npix(nside), dtype=torch.int64)
    all_ring = hp.nest2ring(nside, all_nest)
    pix_nest = all_nest[(all_nest // ratio2) != (all_ring // ratio2)][::37][:8]
    pix_ring = hp.nest2ring(nside, pix_nest)
    expected_cov = torch.unique(torch.div(pix_nest, ratio2, rounding_mode="floor"))

    wrong_cov = torch.unique(torch.div(pix_ring, ratio2, rounding_mode="floor"))
    assert set(wrong_cov.tolist()) != set(expected_cov.tolist())

    smap = SparseHealpixMap.from_pixels(
        pix_ring,
        torch.arange(pix_ring.numel(), dtype=torch.float64),
        nside_sparse=nside,
        nest=False,
        nside_coverage=nside_cov,
        sentinel=-1.0,
    )
    expected_mask = torch.zeros(hp.nside2npix(nside_cov), dtype=torch.bool)
    expected_mask[expected_cov] = True

    torch.testing.assert_close(smap.coverage_mask, expected_mask)
    assert smap.coverage_map is not None
    torch.testing.assert_close(
        torch.nonzero(smap.coverage_map >= 0, as_tuple=False).flatten(),
        expected_cov,
    )


def test_sparse_rejects_coverage_nside_larger_than_sparse_nside() -> None:
    with pytest.raises(ValueError, match="nside_coverage"):
        SparseHealpixMap.from_pixels(
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([1.0], dtype=torch.float64),
            nside_sparse=4,
            nside_coverage=8,
        )


def test_sky_mask_pipe_combines_masks_without_dense_materialization() -> None:
    nside = 1024
    a = SparseHealpixMap.from_pixels(
        torch.tensor([1, 10, 100, 1000, 10_000], dtype=torch.int64),
        torch.ones(5, dtype=torch.float32),
        nside_sparse=nside,
        nest=True,
        nside_coverage=64,
        sentinel=0.0,
    )
    b = SparseHealpixMap.from_pixels(
        torch.tensor([10, 100, 999, 10_000], dtype=torch.int64),
        torch.ones(4, dtype=torch.float32),
        nside_sparse=nside,
        nest=True,
        nside_coverage=64,
        sentinel=0.0,
    )
    pipe = SkyMaskPipe(nside=nside, nest=True)
    pipe.add_stage("a", a)
    pipe.add_stage("b", b)

    both = pipe.combine("and")
    either = pipe.combine("or")

    torch.testing.assert_close(
        both.pixels,
        torch.tensor([10, 100, 10_000], dtype=torch.int64),
    )
    torch.testing.assert_close(
        either.pixels,
        torch.tensor([1, 10, 100, 999, 1000, 10_000], dtype=torch.int64),
    )
    assert both.nside_coverage == 64
    assert either.nside_coverage == 64
    torch.testing.assert_close(both.values, torch.ones(3, dtype=torch.float32))
    torch.testing.assert_close(either.values, torch.ones(6, dtype=torch.float32))


def test_sky_mask_pipe_rejects_unknown_operation() -> None:
    pipe = SkyMaskPipe(nside=8, nest=True)
    pipe.add_stage(
        "mask",
        SparseHealpixMap.from_pixels(
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([1.0], dtype=torch.float32),
            nside_sparse=8,
            nest=True,
            sentinel=0.0,
        ),
    )

    with pytest.raises(ValueError, match="operation"):
        pipe.combine("xor")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("values", "sentinel"),
    [
        (torch.tensor([True, True, True], dtype=torch.bool), False),
        (torch.tensor([10, 20, 30], dtype=torch.int64), -1.0),
        (torch.tensor([1.25, 2.5, 3.75], dtype=torch.float64), -999.0),
    ],
)
def test_sparse_fits_roundtrip_preserves_value_dtype(
    tmp_path, values: torch.Tensor, sentinel: float | bool
) -> None:
    path = tmp_path / f"sparse_{values.dtype}.fits"
    pixels = torch.tensor([0, 10, 42], dtype=torch.int64)
    smap = SparseHealpixMap.from_pixels(
        pixels,
        values,
        nside_sparse=8,
        nside_coverage=2,
        sentinel=float(sentinel),
        metadata={"BAND": "R"},
    )

    smap.write_fits(str(path), overwrite=True)
    got = SparseHealpixMap.read_fits(str(path))

    assert got.values.dtype == values.dtype
    torch.testing.assert_close(got.pixels, pixels)
    torch.testing.assert_close(got.values, values)
    assert got.metadata == {"BAND": "R"}


def test_sparse_ud_grade_dense_equivalence_with_bad_values() -> None:
    nside_in = 8
    nside_out = 4
    npix = hp.nside2npix(nside_in)
    bad = -12345.0
    rng = np.random.default_rng(91)

    dense = torch.full((2, npix), bad, dtype=torch.float64)
    keep = torch.from_numpy(rng.choice(npix, size=npix // 2, replace=False)).to(
        torch.int64
    )
    dense[:, keep] = torch.from_numpy(rng.normal(size=(2, keep.numel()))).to(
        torch.float64
    )
    dense[0, keep[:20]] = torch.nan
    dense[1, keep[20:40]] = torch.inf

    sp = SparseHealpixMap.from_dense(
        dense, nside=nside_in, nest=False, fill_value=bad, coverage_mode="any"
    )
    got = sp.ud_grade(nside_out, pess=False, power=2.0).to_dense()
    exp = hp.ud_grade(
        dense,
        nside_out,
        pess=False,
        badval=bad,
        order_in="RING",
        order_out="RING",
        power=2.0,
    )
    torch.testing.assert_close(got, exp, atol=1e-12, rtol=0.0, equal_nan=True)
