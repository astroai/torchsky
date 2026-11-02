"""Hierarchical HEALPix range queries and native MOC range algebra.

Native kernels (``torchsky::moc_*``, ``torchsky::sphere_query_disc_nest_ranges``)
must reproduce the torch implementations exactly; mocpy is the external oracle and
runs in a child process (``TORCHSKY_TEST_MOCPY_PYTHON``) like the healpy tests.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from torchsky import _ops
from torchsky.sphere import MOC, healpix as hp
from torchsky.sphere import moc as moc_mod
from torchsky.sphere.compat import query_circle

pytestmark = pytest.mark.skipif(
    not _ops.has_fixed_dispatcher(), reason="native extension unavailable"
)


def _random_unit(g: torch.Generator) -> torch.Tensor:
    v = torch.randn(3, generator=g, dtype=torch.float64)
    return v / v.norm()


@pytest.mark.parametrize("nside", [1, 2, 8, 64, 256])
def test_query_disc_nest_matches_ring_query(nside: int) -> None:
    """Hierarchical NESTED disc == ring-range disc mapped to NESTED (exact)."""
    g = torch.Generator().manual_seed(100 + nside)
    for t in range(30):
        v = _random_unit(g)
        radius = float(torch.rand(1, generator=g)) * math.pi * (1.0 if t % 3 else 0.05)
        if t % 11 == 0:
            radius = math.pi
        if t % 13 == 0:
            radius = 0.0
        ring = hp.query_disc(nside, v, radius, nest=False)
        nested = hp.query_disc(nside, v, radius, nest=True)
        expected = torch.sort(hp.ring2nest(nside, ring)).values
        assert torch.equal(nested, expected)


def test_query_disc_nest_ranges_do_not_enumerate_pixels() -> None:
    """nside 4096 at 179 deg returns 215 ranges, not 2e8 pixels."""
    v = torch.tensor([0.3, 0.4, 0.8], dtype=torch.float64)
    v = v / v.norm()
    ranges = torch.ops.torchsky.sphere_query_disc_nest_ranges(
        4096, v, math.cos(math.radians(179.0))
    )
    assert ranges.shape[1] == 2 and ranges.shape[0] < 1000
    covered = int((ranges[:, 1] - ranges[:, 0]).sum())
    cap_fraction = (1.0 - math.cos(math.radians(179.0))) / 2.0
    assert abs(covered / (12 * 4096 * 4096) - cap_fraction) < 1e-3
    assert bool((ranges[1:, 0] > ranges[:-1, 1]).all())


@pytest.mark.parametrize(
    "inclusive,fact", [(False, 4), (True, 4), (True, 2), (True, 8)]
)
def test_query_circle_pixel_ranges_match_pixels(inclusive: bool, fact: int) -> None:
    g = torch.Generator().manual_seed(7)
    for nside in (2, 16, 128):
        for _ in range(15):
            lon = float(torch.rand(1, generator=g)) * 360.0
            lat = float(torch.rand(1, generator=g)) * 180.0 - 90.0
            radius = float(torch.rand(1, generator=g)) * 40.0
            kwargs = dict(inclusive=inclusive, fact=fact, nest=True)
            ranges = query_circle(
                nside, lon, lat, radius, return_pixel_ranges=True, **kwargs
            )
            pixels = query_circle(nside, lon, lat, radius, **kwargs)
            assert torch.equal(ranges, hp.pixels_to_pixel_ranges(pixels))


@pytest.mark.parametrize("nside", [1, 4, 32, 128])
def test_query_strip_nest_matches_ring_band(nside: int) -> None:
    g = torch.Generator().manual_seed(200 + nside)
    for _ in range(15):
        t1 = float(torch.rand(1, generator=g)) * math.pi
        t2 = float(torch.rand(1, generator=g)) * math.pi
        for inclusive in (False, True):
            ring = hp.query_strip(nside, t1, t2, nest=False, inclusive=inclusive)
            nested = hp.query_strip(nside, t1, t2, nest=True, inclusive=inclusive)
            expected = torch.sort(hp.ring2nest(nside, ring)).values
            assert torch.equal(nested, expected)


@pytest.mark.parametrize("nside", [1, 4, 32, 128])
def test_query_polygon_nest_matches_ring_query(nside: int) -> None:
    g = torch.Generator().manual_seed(300 + nside)
    reference = torch.tensor([0.3, 0.5, 0.9], dtype=torch.float64)
    checked = 0
    for t in range(20):
        c = _random_unit(g)
        e1 = torch.linalg.cross(c, reference)
        e1 = e1 / e1.norm()
        e2 = torch.linalg.cross(c, e1)
        rad = float(torch.rand(1, generator=g)) * 1.2 + 0.05
        ang = torch.sort(
            torch.rand(3 + t % 4, generator=g, dtype=torch.float64) * 2.0 * math.pi
        ).values
        vertices = torch.stack(
            [
                c * math.cos(rad)
                + math.sin(rad) * (math.cos(a) * e1 + math.sin(a) * e2)
                for a in ang
            ]
        )
        for inclusive in (False, True):
            try:
                ring = hp.query_polygon(nside, vertices, inclusive=inclusive)
            except ValueError:
                continue
            nested = hp.query_polygon(nside, vertices, inclusive=inclusive, nest=True)
            expected = torch.sort(hp.ring2nest(nside, ring)).values
            assert torch.equal(nested, expected)
            checked += 1
    assert checked > 0


def _random_moc(g: torch.Generator, order: int) -> MOC:
    lon = float(torch.rand(1, generator=g)) * 360.0
    lat = float(torch.rand(1, generator=g)) * 120.0 - 60.0
    radius = 1.0 + float(torch.rand(1, generator=g)) * 25.0
    return MOC.from_circle(lon, lat, radius, order)


@pytest.mark.parametrize("op", ["union", "intersection", "difference"])
def test_native_range_algebra_matches_torch(op: str) -> None:
    g = torch.Generator().manual_seed(11)
    for _ in range(25):
        a = _random_moc(g, 6 + int(torch.randint(0, 5, (1,), generator=g)))
        b = _random_moc(g, 6 + int(torch.randint(0, 5, (1,), generator=g)))
        order = max(a.max_order, b.max_order)
        left = moc_mod._uniq_to_range_tensors_torch(a.uniq, order)
        right = moc_mod._uniq_to_range_tensors_torch(b.uniq, order)
        native = moc_mod._combine_range_tensors(left, right, op)
        reference = moc_mod._combine_range_tensors_torch(left, right, op)
        assert torch.equal(native[0], reference[0])
        assert torch.equal(native[1], reference[1])
        uniq = moc_mod._range_tensors_to_uniq(*native, order)
        assert torch.equal(
            uniq, moc_mod._range_tensors_to_uniq_torch(*reference, order)
        )


def test_native_uniq_range_roundtrip_and_complement() -> None:
    g = torch.Generator().manual_seed(5)
    for _ in range(20):
        m = _random_moc(g, 4 + int(torch.randint(0, 8, (1,), generator=g)))
        order = m.max_order
        s, e = moc_mod._uniq_to_range_tensors(m.uniq, order)
        s_ref, e_ref = moc_mod._uniq_to_range_tensors_torch(m.uniq, order)
        assert torch.equal(s, s_ref) and torch.equal(e, e_ref)
        assert torch.equal(moc_mod._range_tensors_to_uniq(s, e, order), m.uniq)
        cs, ce = moc_mod._complement_range_tensors(s, e, order)
        total = 12 << (2 * order)
        assert int((e - s).sum()) + int((ce - cs).sum()) == total
        both = moc_mod._combine_range_tensors((s, e), (cs, ce), "intersection")
        assert both[0].numel() == 0
        back = moc_mod._complement_range_tensors(cs, ce, order)
        assert torch.equal(back[0], s) and torch.equal(back[1], e)


def test_native_degrade_matches_torch_semantics() -> None:
    g = torch.Generator().manual_seed(9)
    for _ in range(10):
        m = _random_moc(g, 9)
        for order in (0, 3, 6):
            got = m.degrade(order)
            nside, pix = moc_mod.uniq2nest(m.uniq)
            cell_orders = moc_mod._order_from_uniq(m.uniq)
            delta = (cell_orders - order).clamp(min=0)
            expected = MOC(
                moc_mod.nest2uniq(
                    torch.where(delta > 0, torch.full_like(nside, 1 << order), nside),
                    pix >> (2 * delta),
                )
            )
            assert got == expected


def test_native_moc_rejects_invalid_uniq_with_torch_error() -> None:
    with pytest.raises(ValueError):
        MOC(torch.tensor([1, 2, 3], dtype=torch.int64))


_MOCPY_SCRIPT = textwrap.dedent(
    """
    import sys, numpy as np
    from mocpy import MOC
    d = np.load(sys.argv[1])
    def build(u):
        u = np.asarray(u, dtype=np.uint64)
        depth = (np.floor(np.log2(u.astype(np.float64))).astype(np.int64) - 2) // 2
        ipix = u.astype(np.int64) - 4 * (4 ** depth)
        return MOC.from_healpix_cells(ipix, depth.astype(np.uint8), max_depth=int(depth.max()))
    a, b = build(d["a"]), build(d["b"])
    np.savez(sys.argv[2], union=a.union(b).uniq_hpx, inter=a.intersection(b).uniq_hpx,
             diff=a.difference(b).uniq_hpx)
    """
)


def test_native_moc_algebra_matches_mocpy(tmp_path) -> None:
    child = os.environ.get("TORCHSKY_TEST_MOCPY_PYTHON", sys.executable)
    probe = subprocess.run(
        [child, "-c", "import mocpy"], capture_output=True, check=False
    )
    if probe.returncode != 0:
        pytest.skip("mocpy not importable in a child process")
    a = MOC.from_circle(30.0, 20.0, 10.0, 8)
    b = MOC.from_circle(36.0, 24.0, 10.0, 8)
    src = tmp_path / "in.npz"
    out = tmp_path / "out.npz"
    np.savez(src, a=a.uniq.numpy(), b=b.uniq.numpy())
    subprocess.run([child, "-c", _MOCPY_SCRIPT, str(src), str(out)], check=True)
    ref = np.load(out)
    np.testing.assert_array_equal(np.sort((a | b).uniq.numpy()), np.sort(ref["union"]))
    np.testing.assert_array_equal(np.sort((a & b).uniq.numpy()), np.sort(ref["inter"]))
    np.testing.assert_array_equal(np.sort((a - b).uniq.numpy()), np.sort(ref["diff"]))


def test_query_polygon_nest_follows_input_device(compute_device: torch.device) -> None:
    device = compute_device
    dtype = torch.float32 if device.type == "mps" else torch.float64
    verts = torch.tensor(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=dtype, device=device
    )
    ring = hp.query_polygon(8, verts, lonlat=True, nest=False)
    nested = hp.query_polygon(8, verts, lonlat=True, nest=True)
    assert nested.device.type == device.type
    assert torch.equal(torch.sort(hp.ring2nest(8, ring.cpu())).values, nested.cpu())
