"""Exact quad / pixel-box overlap used by the square drizzle and conservative kernels."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from torchsky.map.operators import geometry_drizzle as gd


# ---------------------------------------------------------------------------
# Reference: the generic Sutherland-Hodgman clip against the four box half-planes
# (the implementation this module used before the axis-aligned boundary integral).
# ---------------------------------------------------------------------------
def _reference_clip_polygon(
    xs: list[float], ys: list[float], axis: int, bound: float, keep_greater: bool
):
    inside = (lambda v: v >= bound) if keep_greater else (lambda v: v <= bound)
    out_x: list[float] = []
    out_y: list[float] = []
    n = len(xs)
    for i in range(n):
        px, py = xs[i - 1], ys[i - 1]
        cx, cy = xs[i], ys[i]
        pv, cv = (px, cx) if axis == 0 else (py, cy)
        pi, ci = inside(pv), inside(cv)
        if ci != pi:
            t = (bound - pv) / (cv - pv)
            ix, iy = px + t * (cx - px), py + t * (cy - py)
            out_x.append(ix)
            out_y.append(iy)
        if ci:
            out_x.append(cx)
            out_y.append(cy)
    return out_x, out_y


def _reference_area(qx: Tensor, qy: Tensor, cx: float, cy: float) -> float:
    xs, ys = [float(v) for v in qx], [float(v) for v in qy]
    for axis, bound, keep_greater in (
        (0, cx - 0.5, True),
        (0, cx + 0.5, False),
        (1, cy - 0.5, True),
        (1, cy + 0.5, False),
    ):
        if len(xs) < 3:
            return 0.0
        xs, ys = _reference_clip_polygon(xs, ys, axis, bound, keep_greater)
    if len(xs) < 3:
        return 0.0
    s = sum(
        xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i]
        for i in range(len(xs))
    )
    return 0.5 * abs(s)


def _random_quads(
    n: int, size: float, seed: int, *, shear: float = 0.3, origin: float = 40.0
) -> tuple[Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    f64 = {"dtype": torch.float64}
    centre = torch.rand(n, 2, generator=gen, **f64) * 12.0 + origin
    angle = torch.rand(n, generator=gen, **f64) * 6.283185307179586
    hx = 0.5 * size * (0.6 + 0.8 * torch.rand(n, generator=gen, **f64))
    hy = 0.5 * size * (0.6 + 0.8 * torch.rand(n, generator=gen, **f64))
    skew = (torch.rand(n, generator=gen, **f64) - 0.5) * shear
    ux = torch.stack([-hx, hx, hx, -hx], dim=1)
    uy = torch.stack([-hy, -hy, hy, hy], dim=1)
    ux = ux + skew[:, None] * uy
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]
    return (
        centre[:, :1] + ux * cos - uy * sin,
        centre[:, 1:] + ux * sin + uy * cos,
    )


def _candidate_pairs(
    qx: Tensor, qy: Tensor, margin: int = 1
) -> tuple[Tensor, Tensor, Tensor]:
    """Every pixel in the bounding box grown by ``margin`` pixels: includes disjoint boxes."""
    lo_x = torch.floor(qx.min(1).values - 0.5).long() - margin
    hi_x = torch.ceil(qx.max(1).values + 0.5).long() + margin
    lo_y = torch.floor(qy.min(1).values - 0.5).long() - margin
    hi_y = torch.ceil(qy.max(1).values + 0.5).long() + margin
    nx = hi_x - lo_x + 1
    counts = nx * (hi_y - lo_y + 1)
    owners = torch.repeat_interleave(torch.arange(qx.shape[0]), counts)
    local = torch.arange(owners.numel()) - (torch.cumsum(counts, 0) - counts)[owners]
    cx = (lo_x[owners] + local % nx[owners]).to(torch.float64)
    cy = (lo_y[owners] + local // nx[owners]).to(torch.float64)
    return owners, cx, cy


@pytest.mark.parametrize("size", [0.02, 0.4, 0.8, 1.0, 1.7, 4.5])
def test_box_overlap_matches_generic_clipping(size: float) -> None:
    """Random convex quads (both orientations, sheared, pixfrac-like shrinks)."""
    qx, qy = _random_quads(150, size, seed=1)
    flip = torch.arange(150) % 2 == 1  # clockwise half
    qx[flip] = qx[flip].flip(1)
    qy[flip] = qy[flip].flip(1)
    owners, cx, cy = _candidate_pairs(qx, qy)
    centers = torch.stack((cx, cy), dim=1)
    got = gd._batch_quads_pixel_areas(qx[owners], qy[owners], centers)
    expected = torch.tensor(
        [
            _reference_area(qx[o], qy[o], float(a), float(b))
            for o, a, b in zip(owners.tolist(), cx.tolist(), cy.tolist())
        ],
        dtype=torch.float64,
    )
    # The reference works in absolute coordinates (~1e-12 absolute noise at |x| ~ 50).
    assert (got - expected).abs().max() < 5.0e-11
    # Disjoint candidates are exactly zero, not rounding noise.
    assert bool((got[expected == 0.0] == 0.0).all())
    assert bool((got >= 0.0).all())


def test_box_overlap_exact_axis_aligned_values() -> None:
    qx = torch.tensor([[0.25, 1.25, 1.25, 0.25]], dtype=torch.float64)
    qy = torch.tensor([[0.5, 0.5, 1.5, 1.5]], dtype=torch.float64)
    cx = torch.tensor([0.0, 1.0, 0.0, 1.0, 2.0, -1.0], dtype=torch.float64)
    cy = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    areas = gd._batch_quads_pixel_areas(
        qx.expand(6, 4), qy.expand(6, 4), torch.stack((cx, cy), dim=1)
    )
    # Quad spans [0.25, 1.25] x [0.5, 1.5]; pixel (i, j) spans [i - 0.5, i + 0.5] x [j - 0.5, j + 0.5].
    # Pixels (0, 0), (1, 0) only touch the quad along y = 0.5: exactly zero.
    expected = torch.tensor([0.0, 0.0, 0.25, 0.75, 0.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(areas, expected, rtol=0.0, atol=1.0e-14)


def test_degenerate_quads_have_zero_overlap() -> None:
    zero = torch.zeros((1, 4), dtype=torch.float64)
    line_x = torch.tensor([[0.0, 0.5, 1.0, 1.5]], dtype=torch.float64)
    line_y = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    centers = torch.zeros((1, 2), dtype=torch.float64)
    assert gd._batch_quads_pixel_areas(zero, zero, centers).item() == 0.0
    assert (
        gd._batch_quads_pixel_areas(line_x, 0.2 * line_x + 0.1, centers).item() == 0.0
    )
    assert (
        gd._batch_quads_pixel_areas(line_x, line_y * 0.0 + 0.3, centers).item() == 0.0
    )


def test_pair_areas_conserve_each_quad_area() -> None:
    for size, seed in ((0.3, 3), (1.0, 4), (3.2, 5)):
        qx, qy = _random_quads(200, size, seed)
        areas, valid = gd._quadrilateral_areas_and_validity(qx, qy)
        assert bool(valid.all())
        owners, cx, cy = _candidate_pairs(qx, qy)
        overlaps = gd._batch_quads_pixel_areas(
            qx[owners], qy[owners], torch.stack((cx, cy), 1)
        )
        total = torch.zeros(200, dtype=torch.float64).index_add_(0, owners, overlaps)
        torch.testing.assert_close(total, areas, rtol=1.0e-12, atol=0.0)


# ---------------------------------------------------------------------------
# Accumulation into the target (torch path and native CPU kernel).
# ---------------------------------------------------------------------------
def _accumulate(
    qx: Tensor,
    qy: Tensor,
    numer: Tensor,
    weight: Tensor,
    width: int,
    height: int,
    *,
    native: bool,
    variance: bool = True,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Tensor, Tensor, Tensor | None]:
    monkeypatch.setattr(gd, "_NATIVE_OVERLAP_ENABLED", native)
    flat_n = torch.zeros(width * height, dtype=torch.float32)
    flat_w = torch.zeros_like(flat_n)
    flat_v = torch.zeros_like(flat_n) if variance else None
    gd._accumulate_quadrilateral_overlaps(
        qx,
        qy,
        numer,
        weight,
        width=width,
        height=height,
        flat_numer=flat_n,
        flat_weight=flat_w,
        flat_variance_weight=flat_v,
    )
    return flat_n, flat_w, flat_v


def _has_native() -> bool:
    return gd._native_overlap_op() is not None


def test_accumulation_conserves_flux_and_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width = height = 64
    qx, qy = _random_quads(400, 1.3, seed=7, origin=20.0)
    gen = torch.Generator().manual_seed(8)
    numer = torch.randn(400, generator=gen).float()
    weight = torch.rand(400, generator=gen).float() + 0.5
    for native in (False, True):
        if native and not _has_native():
            continue
        n, w, v = _accumulate(
            qx, qy, numer, weight, width, height, native=native, monkeypatch=monkeypatch
        )
        torch.testing.assert_close(
            n.double().sum(), numer.double().sum(), rtol=1e-5, atol=1e-4
        )
        torch.testing.assert_close(
            w.double().sum(), weight.double().sum(), rtol=1e-5, atol=1e-4
        )
        assert v is not None and bool((v >= 0).all())
        # sum_j c_j^2 <= 1 per quad, so the variance weight never exceeds the weight.
        assert float(v.sum()) <= float(w.sum()) * (1.0 + 1e-6)


@pytest.mark.skipif(not _has_native(), reason="native drizzle kernel unavailable")
@pytest.mark.parametrize("size", [0.05, 0.8, 1.0, 2.5])
def test_native_kernel_matches_torch_path(
    size: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    width, height = 37, 29  # quads straddle all four target edges
    qx, qy = _random_quads(1500, size, seed=11, origin=-2.0)
    qx = qx * 1.6
    qy = qy * 1.2
    flip = torch.arange(1500) % 3 == 0
    qx[flip] = qx[flip].flip(1)
    qy[flip] = qy[flip].flip(1)
    gen = torch.Generator().manual_seed(12)
    numer = torch.randn(1500, generator=gen).float()
    weight = torch.rand(1500, generator=gen).float() + 0.1
    ref = _accumulate(
        qx, qy, numer, weight, width, height, native=False, monkeypatch=monkeypatch
    )
    got = _accumulate(
        qx, qy, numer, weight, width, height, native=True, monkeypatch=monkeypatch
    )
    for a, b in zip(got, ref):
        assert a is not None and b is not None
        scale = float(b.abs().max())
        assert float((a - b).abs().max()) <= 2.0e-5 * max(scale, 1.0)
    # Identical coverage.
    assert bool(((got[1] > 0) == (ref[1] > 0)).all())


@pytest.mark.skipif(not _has_native(), reason="native drizzle kernel unavailable")
def test_native_kernel_is_thread_count_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width = height = 48
    qx, qy = _random_quads(3000, 1.1, seed=13, origin=5.0)
    numer = torch.randn(3000).float()
    weight = torch.rand(3000).float() + 0.1
    threads = torch.get_num_threads()
    try:
        results = []
        for count in (1, max(2, threads)):
            torch.set_num_threads(count)
            results.append(
                _accumulate(
                    qx,
                    qy,
                    numer,
                    weight,
                    width,
                    height,
                    native=True,
                    monkeypatch=monkeypatch,
                )
            )
    finally:
        torch.set_num_threads(threads)
    for a, b in zip(*results):
        assert a is not None and b is not None
        assert torch.equal(a, b)


@pytest.mark.parametrize("native", [False, True])
def test_non_finite_footprints_are_skipped_and_invalid_rejected(
    native: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if native and not _has_native():
        pytest.skip("native drizzle kernel unavailable")
    qx = torch.tensor([[0.0, 1.0, 1.0, 0.0]] * 3, dtype=torch.float64)
    qy = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 3, dtype=torch.float64)
    qx[1, 2] = float("nan")
    numer = torch.ones(3)
    weight = torch.ones(3)
    _n, w, _ = _accumulate(
        qx, qy, numer, weight, 4, 4, native=native, monkeypatch=monkeypatch
    )
    assert float(w.sum()) == pytest.approx(2.0, abs=1e-6)
    bow_x = torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float64)
    bow_y = torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="self-intersecting"):
        _accumulate(
            bow_x,
            bow_y,
            numer[:1],
            weight[:1],
            4,
            4,
            native=native,
            monkeypatch=monkeypatch,
        )


@pytest.mark.parametrize("native", [False, True])
def test_square_drizzle_project_native_and_torch_agree(
    native: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if native and not _has_native():
        pytest.skip("native drizzle kernel unavailable")
    monkeypatch.setattr(gd, "_NATIVE_OVERLAP_ENABLED", native)

    class _Rotate:
        """Rigid rotation + scale + offset between source and target pixel grids."""

        def pixel_to_world(self, x: Tensor, y: Tensor):
            return x, y

        def world_to_pixel(self, x: Tensor, y: Tensor):
            c, s = 0.8660254037844386, 0.5
            return 1.1 * (c * x - s * y) + 15.0, 1.1 * (s * x + c * y) + 3.0

    image = torch.arange(24 * 20, dtype=torch.float32).reshape(24, 20) / 100.0
    weight = torch.ones_like(image)
    target = SimpleNamespace(shape=(44, 40), wcs=_Rotate())
    numer, wgt, var = gd.exact_square_drizzle_project(
        image,
        weight,
        _Rotate(),
        target,
        value_semantics="surface_brightness",
        drizzle_pixfrac=0.7,
        pixel_area_sr_map_fn=None,
        chunk_pixels=97,
        return_variance_weight=True,
    )
    # Every input footprint lies inside the target: flux and weight are conserved.
    assert float(numer.sum()) == pytest.approx(float(image.sum()), rel=1e-5)
    assert float(wgt.sum()) == pytest.approx(float(weight.sum()), rel=1e-5)
    assert float(var.sum()) <= float(wgt.sum()) * (1.0 + 1e-6)


@pytest.mark.parametrize("native", [False, True])
def test_grazing_contact_below_vertex_resolution_is_not_overlap(
    native: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if native and not _has_native():
        pytest.skip("native drizzle kernel unavailable")
    # The right edge overshoots the pixel 0 / pixel 1 boundary (x = 0.5) by 2 float64 ulp.
    right = 0.5 + 2.0 * torch.finfo(torch.float64).eps * 0.5
    qx = torch.tensor([[-0.3, right, right, -0.3]], dtype=torch.float64)
    qy = torch.tensor([[-0.2, -0.2, 0.2, 0.2]], dtype=torch.float64)
    _, w, _ = _accumulate(
        qx,
        qy,
        torch.ones(1),
        torch.ones(1),
        3,
        1,
        native=native,
        monkeypatch=monkeypatch,
    )
    assert w[0] == pytest.approx(1.0, abs=1e-6)
    assert float(w[1]) == 0.0 and float(w[2]) == 0.0


def test_scatter_add_ordered_matches_scatter_add() -> None:
    from torchsky.map.operators.geometry_drizzle import _scatter_add_ordered

    gen = torch.Generator().manual_seed(3)
    index = torch.randint(0, 40, (50_000,), generator=gen)
    values = [torch.rand(50_000, generator=gen), torch.rand(50_000, generator=gen)]
    reference = [torch.zeros(64) for _ in values]
    ordered = [torch.zeros(64) for _ in values]
    for target, value in zip(reference, values):
        target.scatter_add_(0, index, value)
    _scatter_add_ordered(ordered, index, values, ordered=True)
    for expected, got in zip(reference, ordered):
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_scatter_add_ordered_is_bitwise_reproducible_on_cuda() -> None:
    from torchsky.map.operators.geometry_drizzle import _scatter_add_ordered

    gen = torch.Generator().manual_seed(5)
    index = torch.randint(0, 64, (1_000_000,), generator=gen).cuda()
    values = [torch.rand(1_000_000, generator=gen).cuda()]
    runs = []
    for _ in range(4):
        target = [torch.zeros(64, device="cuda")]
        _scatter_add_ordered(target, index, values)
        runs.append(target[0])
    assert all(torch.equal(runs[0], run) for run in runs[1:])
    # float atomics (scatter_add_) are the non-reproducible baseline being replaced
    reference = torch.zeros(64, device="cuda").scatter_add_(0, index, values[0])
    torch.testing.assert_close(runs[0], reference, rtol=1e-4, atol=1e-2)
