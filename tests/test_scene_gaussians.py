from __future__ import annotations

import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from torchsky.map.field import FieldObservation
from torchsky.map.scenes import SceneRepresentation
from torchsky.map.target import TargetRenderSpec
from torchsky.scene import SourceCatalogScene
from torchsky.scene.gaussians import (
    GaussianItemScene,
    _covariance_entries,
    _Level,
    _patch_half_sizes,
    _rasterize_gaussians,
    _sky_from_pixel_arcsec,
)
from torchsky.wcs import WCS

F64 = torch.float64
MODES = ("direct", "pyramid")
PIXEL_VARIANCE = 1.0 / 12.0


# ------------------------------------------------------------------ references
def _analytic(shape, x, y, major, minor, angle, *, integrate=True):
    """Gaussian density per pixel (covariance + 1/12 I if ``integrate``) at the nodes, and its peak."""
    q = PIXEL_VARIANCE if integrate else 0.0
    va, vb = major**2 + q, minor**2 + q
    c, s = math.cos(angle), math.sin(angle)
    txx, tyy, txy = c * c * va + s * s * vb, s * s * va + c * c * vb, c * s * (va - vb)
    det = txx * tyy - txy * txy
    yy, xx = torch.meshgrid(
        torch.arange(shape[0], dtype=F64),
        torch.arange(shape[1], dtype=F64),
        indexing="ij",
    )
    dx, dy = xx - x, yy - y
    quad = (tyy * dx * dx + txx * dy * dy - 2 * txy * dx * dy) / det
    peak = 1.0 / (2 * math.pi * math.sqrt(det))
    return peak * torch.exp(-0.5 * quad), peak


def _lattice_reference(shape, x, y, major, minor, angle):
    """Same density, normalised so that the sum over the infinite lattice of nodes is one
    (Poisson summation, |k| <= 4): what the exact-flux renormalisation converges to."""
    pdf, peak = _analytic(shape, x, y, major, minor, angle)
    va, vb = major**2 + PIXEL_VARIANCE, minor**2 + PIXEL_VARIANCE
    c, s = math.cos(angle), math.sin(angle)
    txx, tyy, txy = c * c * va + s * s * vb, s * s * va + c * c * vb, c * s * (va - vb)
    total = 0.0
    for kx in range(-4, 5):
        for ky in range(-4, 5):
            e = 2 * math.pi**2 * (kx * kx * txx + ky * ky * tyy + 2 * kx * ky * txy)
            if e < 700:
                total += math.exp(-e) * math.cos(2 * math.pi * (kx * x + ky * y))
    return pdf / total, peak


def _one(shape, x, y, major, minor, angle, mode, *, dtype=F64, **kwargs):
    return _rasterize_gaussians(
        torch.tensor([[x, y]], dtype=dtype),
        torch.tensor([[major, minor, angle]], dtype=dtype),
        torch.ones(1, 1, dtype=dtype),
        shape,
        mode=mode,
        **kwargs,
    )[0]


def _random_items(count, shape, *, seed, k=2, wmin=1.5, wmax=9.0, margin=0.0):
    gen = torch.Generator().manual_seed(seed)
    h, w = shape

    def u(*s):
        return torch.rand(*s, generator=gen, dtype=F64)

    pos = torch.stack(
        (margin + u(count) * (w - 2 * margin), margin + u(count) * (h - 2 * margin)), -1
    )
    major = wmin + u(count) * (wmax - wmin)
    shp = torch.stack((major, major * (0.4 + 0.5 * u(count)), u(count) * math.pi), -1)
    coef = 0.5 + u(count, k)
    return pos, shp, coef


# ------------------------------------------------------------------ values
@pytest.mark.parametrize("mode", MODES)
def test_single_item_values_match_analytic_density(mode: str) -> None:
    """Width = smaller principal width, major = ratio x width.  From 1 px up every case is
    within 1e-3 of the peak (measured: below 1e-4); at 0.5 px the exact-flux renormalisation
    changes a node-centred round item by 4 exp(-2 pi^2 (w^2 + 1/12)) = 5.5e-3, asserted <= 6e-3."""
    shape = (96, 130)
    cases = [
        (1, 1),
        (1, 5),
        (2, 1),
        (2, 2),
        (2, 5),
        (5, 2),
        (12, 1),
        (12, 5),
        (30, 2),
        (60, 1),
        (60, 5),
    ]
    offsets = [
        (65.37, 48.81, 0.0),
        (63.5, 47.5, 0.7),
        (64.0, 48.0, 1.9),
        (66.25, 49.9, 2.6),
    ]
    worst = 0.0
    for i, (width, ratio) in enumerate(cases):
        for x, y, angle in offsets[i % 2 :: 2]:
            out = _one(shape, x, y, ratio * width, width, angle, mode)
            ref, peak = _analytic(shape, x, y, ratio * width, width, angle)
            worst = max(worst, float((out - ref).abs().max() / peak))
    assert worst <= 1e-3, worst
    worst_half = 0.0
    for ratio in (1, 2, 5):
        for x, y, angle in offsets:
            out = _one(shape, x, y, ratio * 0.5, 0.5, angle, mode)
            ref, peak = _analytic(shape, x, y, ratio * 0.5, 0.5, angle)
            worst_half = max(worst_half, float((out - ref).abs().max() / peak))
    assert worst_half <= 6e-3, worst_half


@pytest.mark.parametrize("mode", MODES)
def test_subpixel_items_follow_the_lattice_normalisation(mode: str) -> None:
    """Below ~0.5 px the exact-flux renormalisation departs from the analytic density by the
    aliasing exp(-2 pi^2 var): measured against analytic <= 0.1 of the peak (measured 0.078)
    down to a 0.1 px minor width; against the lattice-normalised reference it is <= 2e-4."""
    shape = (40, 44)
    worst_analytic = worst_lattice = 0.0
    for width, ratio, x, y, angle in [
        (0.1, 1, 20.5, 20.5, 0.0),
        (0.1, 3, 20.3, 19.7, 0.9),
        (0.25, 1, 21.2, 18.4, 0.0),
        (0.25, 4, 20.0, 20.0, 2.2),
    ]:
        out = _one(shape, x, y, ratio * width, width, angle, mode)
        a, peak = _analytic(shape, x, y, ratio * width, width, angle)
        b, _ = _lattice_reference(shape, x, y, ratio * width, width, angle)
        worst_analytic = max(worst_analytic, float((out - a).abs().max() / peak))
        worst_lattice = max(worst_lattice, float((out - b).abs().max() / peak))
    assert worst_analytic <= 0.1, worst_analytic
    assert worst_lattice <= 2e-4, worst_lattice


@pytest.mark.parametrize("mode", MODES)
def test_pixel_integration_flag_removes_the_pixel_variance(mode: str) -> None:
    shape = (60, 60)
    out = _one(shape, 30.4, 29.7, 3.0, 2.0, 0.5, mode, pixel_integration=False)
    ref, peak = _analytic(shape, 30.4, 29.7, 3.0, 2.0, 0.5, integrate=False)
    assert float((out - ref).abs().max() / peak) <= 1e-3
    with_term = _one(shape, 30.4, 29.7, 3.0, 2.0, 0.5, mode)
    assert (
        float((with_term - out).abs().max() / peak) > 3e-3
    )  # the term is a real effect


@pytest.mark.parametrize("mode", MODES)
def test_pixel_term_against_the_exact_cell_integral(mode: str) -> None:
    """The 1/12 pixel^2 term replaces the box by a Gaussian.  Against the exact cell integral
    (differences of erf, axis-aligned items) the error is 3.2e-4 of the peak at 2 px, 7.5e-5
    at 3 px, 4.6e-3 at 1 px (so it is a poor model below about 1.5 px): asserted for >= 2 px."""

    def cell(n: int, centre: float, sigma: float) -> torch.Tensor:
        edge = torch.arange(n, dtype=F64)

        def cdf(u):
            return 0.5 * (1 + torch.erf(u / (sigma * math.sqrt(2))))

        return cdf(edge + 0.5 - centre) - cdf(edge - 0.5 - centre)

    shape = (60, 70)
    for sigma in (2.0, 3.0, 5.0, 10.0):
        for x, y in [(35.0, 30.0), (35.5, 30.5), (35.3, 30.2)]:
            exact = cell(70, x, sigma)[None, :] * cell(60, y, 0.7 * sigma)[:, None]
            out = _one(shape, x, y, sigma, 0.7 * sigma, 0.0, mode)
            assert float((out - exact).abs().max() / exact.max()) <= 5e-4, (sigma, x, y)


def test_covariance_and_width_forms_agree() -> None:
    pos, shp, coef = _random_items(5, (40, 50), seed=3)
    c, s = torch.cos(shp[:, 2]), torch.sin(shp[:, 2])
    a2, b2 = shp[:, 0] ** 2, shp[:, 1] ** 2
    cov = torch.stack(
        (
            torch.stack((c * c * a2 + s * s * b2, c * s * (a2 - b2)), -1),
            torch.stack((c * s * (a2 - b2), s * s * a2 + c * c * b2), -1),
        ),
        -2,
    )
    for mode in MODES:
        one = _rasterize_gaussians(pos, shp, coef, (40, 50), mode=mode)
        two = _rasterize_gaussians(pos, cov, coef, (40, 50), mode=mode)
        torch.testing.assert_close(one, two, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------ flux
@pytest.mark.parametrize("mode", MODES)
def test_flux_is_exact_for_items_inside_the_grid(mode: str) -> None:
    shape = (128, 160)
    pos, shp, coef = _random_items(
        12, shape, seed=1, k=3, wmin=0.4, wmax=5.0, margin=40.0
    )
    for n in range(pos.shape[0]):
        plane = _rasterize_gaussians(
            pos[n : n + 1], shp[n : n + 1], coef[n : n + 1], shape, mode=mode
        )
        torch.testing.assert_close(plane.sum((1, 2)), coef[n], rtol=0, atol=1e-10)
    total = _rasterize_gaussians(pos, shp, coef, shape, mode=mode)
    torch.testing.assert_close(total.sum((1, 2)), coef.sum(0), rtol=0, atol=1e-9)


@pytest.mark.parametrize("mode", MODES)
def test_flux_off_the_grid_is_dropped_not_renormalised(mode: str) -> None:
    small, pad = (40, 50), 60
    big = (small[0] + 2 * pad, small[1] + 2 * pad)
    centres = [(-0.5, 20.0), (49.0, -3.0), (-8.0, -8.0), (60.0, 20.0), (25.0, 20.0)]
    for x, y in centres:
        args = (
            torch.tensor([[x, y]], dtype=F64),
            torch.tensor([[4.0, 2.5, 0.8]], dtype=F64),
            torch.tensor([[3.0]], dtype=F64),
        )
        cut = _rasterize_gaussians(*args, small, mode=mode)
        shifted = (args[0] + pad, args[1], args[2])
        whole = _rasterize_gaussians(*shifted, big, mode=mode)
        torch.testing.assert_close(
            cut[0],
            whole[0, pad : pad + small[0], pad : pad + small[1]],
            rtol=0,
            atol=1e-12,
        )
    half = _rasterize_gaussians(
        torch.tensor([[-0.5, 20.0]], dtype=F64),
        torch.tensor([[4.0, 4.0, 0.0]], dtype=F64),
        torch.tensor([[3.0]], dtype=F64),
        small,
        mode=mode,
    )
    assert 1.2 < float(half.sum()) < 1.8  # about half of 3, certainly not 3
    gone = _rasterize_gaussians(
        torch.tensor([[400.0, 20.0]], dtype=F64),
        torch.tensor([[4.0, 4.0, 0.0]], dtype=F64),
        torch.tensor([[3.0]], dtype=F64),
        small,
        mode=mode,
    )
    assert float(gone.abs().max()) == 0.0


# ------------------------------------------------------------------ gradients
def _scalar(pos, shp, coef, weights, shape, mode, **kwargs):
    return (
        _rasterize_gaussians(pos, shp, coef, shape, mode=mode, **kwargs) * weights
    ).sum()


@pytest.mark.parametrize("mode", MODES)
def test_gradients_match_finite_differences(mode: str) -> None:
    shape = (72, 90)
    pos, shp, coef = _random_items(36, shape, seed=7, k=2, wmin=1.5, wmax=14.0)
    weights = torch.randn(
        2, *shape, generator=torch.Generator().manual_seed(8), dtype=F64
    )
    params = [t.clone().requires_grad_(True) for t in (pos, shp, coef)]
    grads = torch.autograd.grad(_scalar(*params, weights, shape, mode), params)
    gen = torch.Generator().manual_seed(9)
    step = 1e-6
    # separate directions: positions, major widths, minor widths, angles, coefficients
    masks = {
        "pos": (0, torch.ones(36, 2, dtype=F64)),
        "major": (1, torch.tensor([1.0, 0.0, 0.0], dtype=F64).expand(36, 3)),
        "minor": (1, torch.tensor([0.0, 1.0, 0.0], dtype=F64).expand(36, 3)),
        "angle": (1, torch.tensor([0.0, 0.0, 1.0], dtype=F64).expand(36, 3)),
        "coef": (2, torch.ones(36, 2, dtype=F64)),
    }
    for name, (which, mask) in masks.items():
        direction = torch.randn(params[which].shape, generator=gen, dtype=F64) * mask
        plus = [t.detach().clone() for t in (pos, shp, coef)]
        minus = [t.detach().clone() for t in (pos, shp, coef)]
        plus[which] += step * direction
        minus[which] -= step * direction
        fd = (
            _scalar(*plus, weights, shape, mode) - _scalar(*minus, weights, shape, mode)
        ) / (2 * step)
        analytic = (grads[which] * direction).sum()
        assert abs(float(fd - analytic)) <= 1e-6 * max(1.0, abs(float(fd))), (
            name,
            float(fd),
            float(analytic),
        )


def _plain_reference(
    pos, shp, coef, shape, *, pixel_integration=True, truncation_sigma=4.0
):
    """Unchunked, loop-per-item, plain autograd direct rasteriser written from the definition:
    Gaussian of covariance Sigma + q I on a window of +-half nodes around round(centre),
    normalised over the whole window, nodes off the grid dropped."""
    h, w = shape
    q = PIXEL_VARIANCE if pixel_integration else 0.0
    cov3 = _covariance_entries(shp)
    half = _patch_half_sizes(cov3, _Level(h, w, 1.0, 0.0, q), truncation_sigma)
    out = torch.zeros(coef.shape[1], h * w + 1, dtype=coef.dtype)
    for n in range(pos.shape[0]):
        r = int(half[n])
        off = torch.arange(-r, r + 1)
        jx = torch.round(pos[n, 0].detach()).long() + off
        jy = torch.round(pos[n, 1].detach()).long() + off
        dx, dy = (jx - pos[n, 0])[None, :], (jy - pos[n, 1])[:, None]
        txx, tyy, txy = cov3[n, 0] + q, cov3[n, 1] + q, cov3[n, 2]
        det = txx * tyy - txy * txy
        g = torch.exp(-0.5 * (tyy * dx * dx + txx * dy * dy - 2 * txy * dx * dy) / det)
        g = g / g.sum()
        inside = ((jy >= 0) & (jy < h))[:, None] & ((jx >= 0) & (jx < w))[None, :]
        flat = torch.where(inside, jy[:, None] * w + jx[None, :], torch.tensor(h * w))
        out = out.index_add(1, flat.reshape(-1), coef[n][:, None] * g.reshape(1, -1))
    return out[:, :-1].reshape(coef.shape[1], h, w)


def test_chunked_backward_equals_plain_autograd_reference() -> None:
    shape = (60, 76)
    pos, shp, coef = _random_items(25, shape, seed=11, k=3, wmin=0.8, wmax=7.0)
    pos[0] = torch.tensor([-2.0, 10.0], dtype=F64)  # partly off the grid
    pos[1] = torch.tensor([200.0, 10.0], dtype=F64)  # entirely off
    weights = torch.randn(
        3, *shape, generator=torch.Generator().manual_seed(2), dtype=F64
    )
    ref_in = [t.clone().requires_grad_(True) for t in (pos, shp, coef)]
    ref_out = _plain_reference(*ref_in, shape)
    ref_grads = torch.autograd.grad((ref_out * weights).sum(), ref_in)
    for chunk in (3_000, 50_000, 4_000_000):
        got_in = [t.clone().requires_grad_(True) for t in (pos, shp, coef)]
        got = _rasterize_gaussians(*got_in, shape, mode="direct", chunk_elements=chunk)
        torch.testing.assert_close(got, ref_out.detach(), rtol=1e-11, atol=1e-12)
        grads = torch.autograd.grad((got * weights).sum(), got_in)
        for a, b in zip(grads, ref_grads, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("mode", MODES)
def test_forward_and_gradients_do_not_depend_on_the_chunk_size(mode: str) -> None:
    shape = (50, 64)
    pos, shp, coef = _random_items(30, shape, seed=5, k=2, wmin=1.0, wmax=12.0)
    weights = torch.randn(
        2, *shape, generator=torch.Generator().manual_seed(6), dtype=F64
    )
    results = []
    for chunk in (2_000, 20_000, 4_000_000):
        inputs = [t.clone().requires_grad_(True) for t in (pos, shp, coef)]
        out = _rasterize_gaussians(*inputs, shape, mode=mode, chunk_elements=chunk)
        grads = torch.autograd.grad((out * weights).sum(), inputs)
        results.append((out.detach(), *grads))
    for other in results[1:]:
        for a, b in zip(results[0], other, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)


class _MaxNumel(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.largest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for leaf in tree_leaves(out):
            if isinstance(leaf, torch.Tensor):
                self.largest = max(self.largest, leaf.numel())
        return out


def test_backward_memory_is_bounded_by_the_chunk() -> None:
    """No tensor of items x patch x K elements is created in the forward or the backward pass."""
    shape, count, k = (64, 64), 60, 4
    pos, shp, coef = _random_items(
        count, shape, seed=4, k=k, wmin=5.5, wmax=6.5, margin=20.0
    )
    shp[:, 1] = shp[:, 0]
    halves = _patch_half_sizes(
        _covariance_entries(shp), _Level(*shape, 1.0, 0.0, PIXEL_VARIANCE), 4.0
    )
    everything = int(((2 * halves + 1) ** 2).sum()) * k
    chunk = 30_000
    assert (
        everything > 20 * chunk
    )  # the test is only meaningful if the chunk is much smaller
    weights = torch.randn(
        k, *shape, generator=torch.Generator().manual_seed(1), dtype=F64
    )

    def largest(chunk_elements: int) -> tuple[int, int]:
        inputs = [t.clone().requires_grad_(True) for t in (pos, shp, coef)]
        with _MaxNumel() as forward_mode:
            out = _rasterize_gaussians(
                *inputs, shape, mode="direct", chunk_elements=chunk_elements
            )
        loss = (out * weights).sum()
        with _MaxNumel() as backward_mode:
            torch.autograd.grad(loss, inputs)
        return forward_mode.largest, backward_mode.largest

    fwd, bwd = largest(chunk)
    assert fwd <= 2 * chunk and bwd <= 2 * chunk, (fwd, bwd, chunk)
    fwd_all, bwd_all = largest(10 * everything)  # control: one chunk holds everything
    # items of one patch size share a chunk, so the control holds a bucket, not everything
    assert fwd_all > 10 * chunk and bwd_all > 10 * chunk, (fwd_all, bwd_all)


# ------------------------------------------------------------------ SourceCatalogScene
def _tan_header(shape, *, scale_arcsec=0.5, rotation_deg=0.0, dec=40.0):
    h, w = shape
    s, c = math.sin(math.radians(rotation_deg)), math.cos(math.radians(rotation_deg))
    d = scale_arcsec / 3600.0
    return {
        "NAXIS": 2, "NAXIS1": w, "NAXIS2": h,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": 0.5 * (w + 1), "CRPIX2": 0.5 * (h + 1),
        "CRVAL1": 150.0, "CRVAL2": dec,
        "CD1_1": -d * c, "CD1_2": d * s, "CD2_1": d * s, "CD2_2": d * c,
    }  # fmt: skip


def _sky_items(header, pixels):
    pix = torch.tensor(pixels, dtype=F64)
    lon, lat = WCS(header).pixel_to_world(pix[:, 0], pix[:, 1])
    return lon, lat


def test_agrees_with_source_catalog_scene_on_a_rotated_header() -> None:
    """Point sampling (pixel_integration=False): differences are the two 4-width truncations
    (<= 2 x exp(-8) of the peak) and the exact-flux renormalisation (negligible above 1 px)."""
    shape = (60, 72)
    header = _tan_header(shape, scale_arcsec=0.5, rotation_deg=27.0, dec=40.0)
    lon, lat = _sky_items(header, [(30.2, 28.7), (48.0, 20.4), (14.6, 40.1)])
    gauss = torch.tensor(
        [[3.0, 1.5, 25.0], [2.0, 2.0, 0.0], [4.5, 1.2, 130.0]], dtype=F64
    )  # arcsec
    gauss_pix = gauss[:, :2] / 0.5
    coef = torch.tensor([[5.0], [2.0], [7.5]], dtype=F64)
    reference = SourceCatalogScene(
        lon, lat, morphology="gaussian", gaussian_parameters=gauss, coefficients=coef
    )
    observation = FieldObservation(image=torch.zeros(shape, dtype=F64), header=header)
    expected = reference.render_observation(observation)
    scene = GaussianItemScene(lon, lat, gauss, coef, pixel_integration=False)
    for mode in MODES:
        scene.mode = mode
        image = scene.render_coefficients(header)[0]
        assert image.shape == expected.shape == shape
        peak = float(
            (coef[:, 0] / (2 * math.pi * gauss_pix[:, 0] * gauss_pix[:, 1])).max()
        )
        assert float((image - expected).detach().abs().max()) <= 3e-4 * peak, mode


def test_pixel_integration_equals_a_widened_source_catalog_scene() -> None:
    """With the pixel term, items equal SourceCatalogScene items widened in quadrature by
    scale / sqrt(12) (valid for square pixels: the 1/12 term is isotropic)."""
    shape = (60, 72)
    scale = 0.5
    header = _tan_header(shape, scale_arcsec=scale, rotation_deg=-33.0, dec=-20.0)
    lon, lat = _sky_items(header, [(30.2, 28.7), (45.0, 22.4)])
    gauss = torch.tensor([[3.0, 1.5, 70.0], [2.5, 2.0, 10.0]], dtype=F64)
    widened = gauss.clone()
    widened[:, :2] = torch.sqrt(gauss[:, :2] ** 2 + scale**2 * PIXEL_VARIANCE)
    coef = torch.tensor([[5.0], [2.0]], dtype=F64)
    expected = SourceCatalogScene(
        lon, lat, morphology="gaussian", gaussian_parameters=widened, coefficients=coef
    ).render_observation(
        FieldObservation(image=torch.zeros(shape, dtype=F64), header=header)
    )
    for mode in MODES:
        image = GaussianItemScene(lon, lat, gauss, coef, mode=mode).render_coefficients(
            header
        )[0]
        peak = float(
            5.0 / (2 * math.pi * (widened[0, 0] / scale) * (widened[0, 1] / scale))
        )
        assert float((image - expected).detach().abs().max()) <= 3e-4 * peak, mode


# ------------------------------------------------------------------ grids and dtypes
@pytest.mark.parametrize("mode", MODES)
def test_rectangular_grids_and_axis_conventions(mode: str) -> None:
    for shape in ((37, 91), (91, 37)):
        out = _one(shape, 20.4, 30.6, 6.0, 2.0, 0.6, mode)
        ref, peak = _analytic(shape, 20.4, 30.6, 6.0, 2.0, 0.6)
        assert out.shape == shape
        assert float((out - ref).abs().max() / peak) <= 1e-3
    # swapping x and y: the image of the transposed problem is the transpose
    pos, shp, coef = _random_items(8, (40, 70), seed=21, wmin=1.0, wmax=9.0)

    def cov(s):
        return torch.stack(
            (
                torch.stack((torch.cos(s[:, 2]) ** 2 * s[:, 0] ** 2 + torch.sin(s[:, 2]) ** 2 * s[:, 1] ** 2,
                             torch.cos(s[:, 2]) * torch.sin(s[:, 2]) * (s[:, 0] ** 2 - s[:, 1] ** 2)), -1),
                torch.stack((torch.cos(s[:, 2]) * torch.sin(s[:, 2]) * (s[:, 0] ** 2 - s[:, 1] ** 2),
                             torch.sin(s[:, 2]) ** 2 * s[:, 0] ** 2 + torch.cos(s[:, 2]) ** 2 * s[:, 1] ** 2), -1),
            ), -2)  # fmt: skip

    c = cov(shp)
    swapped = c.flip(-1).flip(-2)
    a = _rasterize_gaussians(pos, c, coef, (40, 70), mode=mode)
    b = _rasterize_gaussians(pos.flip(-1), swapped, coef, (70, 40), mode=mode)
    torch.testing.assert_close(a, b.transpose(-1, -2), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_float32_is_finite_and_within_three_times_the_float64_error(mode: str) -> None:
    shape = (64, 80)
    pos, shp, coef = _random_items(20, shape, seed=13, k=2, wmin=1.0, wmax=12.0)
    weights = torch.randn(
        2, *shape, generator=torch.Generator().manual_seed(3), dtype=F64
    )
    in64 = [t.clone().requires_grad_(True) for t in (pos, shp, coef)]
    out64 = _rasterize_gaussians(*in64, shape, mode=mode)
    grads64 = torch.autograd.grad((out64 * weights).sum(), in64)
    # the exact reference: the same items in float64 evaluated one by one against the analytic density
    ref = torch.zeros_like(out64)
    peak = 0.0
    for n in range(pos.shape[0]):
        r, pk = _analytic(
            shape,
            float(pos[n, 0]),
            float(pos[n, 1]),
            float(shp[n, 0]),
            float(shp[n, 1]),
            float(shp[n, 2]),
        )
        ref += coef[n][:, None, None] * r
        peak = max(peak, float(coef[n].max()) * pk)
    in32 = [t.detach().float().requires_grad_(True) for t in (pos, shp, coef)]
    out32 = _rasterize_gaussians(*in32, shape, mode=mode)
    grads32 = torch.autograd.grad((out32 * weights.float()).sum(), in32)
    assert out32.dtype == torch.float32
    assert torch.isfinite(out32).all() and all(torch.isfinite(g).all() for g in grads32)
    err64 = float((out64 - ref).detach().abs().max()) / peak
    err32 = float((out32.detach().double() - ref).abs().max()) / peak
    assert err32 <= 3 * max(err64, 1e-6), (err32, err64)
    for g32, g64 in zip(grads32, grads64, strict=True):
        scale = float(g64.abs().max())
        assert float((g32.double() - g64).abs().max()) <= 1e-3 * scale


@pytest.mark.parametrize("mode", MODES)
def test_float32_tiny_items_stay_finite(mode: str) -> None:
    pos = torch.tensor([[10.5, 10.5], [3.2, 4.7]])
    shp = torch.tensor([[0.02, 0.01, 0.3], [0.3, 0.05, 1.0]])
    coef = torch.ones(2, 1, requires_grad=True)
    out = _rasterize_gaussians(
        pos, shp, coef, (20, 20), mode=mode, pixel_integration=False
    )
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(coef.grad).all()
    torch.testing.assert_close(out.sum(), torch.tensor(2.0), rtol=0, atol=1e-5)


def test_pyramid_edges_and_many_levels() -> None:
    """Items much wider than the grid, on the edge and just outside it, match the direct path."""
    shape = (48, 70)
    cases = [
        (35.0, 24.0, 40.0, 25.0),
        (-3.0, 10.0, 9.0, 6.0),
        (72.0, 47.0, 20.0, 3.0),
        (0.0, 0.0, 14.0, 14.0),
        (80.0, 24.0, 30.0, 8.0),
    ]
    for x, y, major, minor in cases:
        args = (
            torch.tensor([[x, y]], dtype=F64),
            torch.tensor([[major, minor, 0.4]], dtype=F64),
            torch.ones(1, 1, dtype=F64),
        )
        direct = _rasterize_gaussians(*args, shape, mode="direct")
        pyramid = _rasterize_gaussians(*args, shape, mode="pyramid")
        peak = 1.0 / (
            2 * math.pi * math.sqrt((major**2 + 1 / 12) * (minor**2 + 1 / 12))
        )
        assert float((direct - pyramid).abs().max()) <= 1e-3 * peak, (
            x,
            y,
            major,
            minor,
        )


# ------------------------------------------------------------------ errors
def test_bad_inputs_raise_clear_errors() -> None:
    pos = torch.tensor([[5.0, 5.0]], dtype=F64)
    shp = torch.tensor([[2.0, 1.0, 0.0]], dtype=F64)
    coef = torch.ones(1, 1, dtype=F64)
    ok = (pos, shp, coef, (10, 12))
    with pytest.raises(ValueError, match="shape must be two positive"):
        _rasterize_gaussians(pos, shp, coef, (0, 12))
    with pytest.raises(ValueError, match="shape must be two positive"):
        _rasterize_gaussians(pos, shp, coef, (10, 12, 3))
    with pytest.raises(ValueError, match=r"centers_pix must have shape \(C, 2\)"):
        _rasterize_gaussians(pos[:, :1], shp, coef, (10, 12))
    with pytest.raises(ValueError, match="share C items"):
        _rasterize_gaussians(pos.repeat(2, 1), shp, coef, (10, 12))
    with pytest.raises(ValueError, match="shape_pix must have shape"):
        _rasterize_gaussians(pos, shp[:, :2], coef, (10, 12))
    with pytest.raises(ValueError, match="widths must be positive"):
        _rasterize_gaussians(
            pos, torch.tensor([[2.0, 0.0, 0.0]], dtype=F64), coef, (10, 12)
        )
    with pytest.raises(ValueError, match="widths must be positive"):
        _rasterize_gaussians(
            pos, torch.tensor([[-2.0, 1.0, 0.0]], dtype=F64), coef, (10, 12)
        )
    with pytest.raises(ValueError, match="minor width cannot exceed major"):
        _rasterize_gaussians(
            pos, torch.tensor([[1.0, 2.0, 0.0]], dtype=F64), coef, (10, 12)
        )
    with pytest.raises(ValueError, match="positive definite"):
        _rasterize_gaussians(
            pos, torch.tensor([[[1.0, 2.0], [2.0, 1.0]]], dtype=F64), coef, (10, 12)
        )
    with pytest.raises(ValueError, match="symmetric"):
        _rasterize_gaussians(
            pos, torch.tensor([[[2.0, 0.5], [0.1, 2.0]]], dtype=F64), coef, (10, 12)
        )
    with pytest.raises(ValueError, match="finite"):
        _rasterize_gaussians(
            torch.tensor([[float("nan"), 1.0]], dtype=F64), shp, coef, (10, 12)
        )
    with pytest.raises(ValueError, match="mode must be one of"):
        _rasterize_gaussians(*ok, mode="fast")
    with pytest.raises(ValueError, match="chunk_elements"):
        _rasterize_gaussians(*ok, chunk_elements=0)
    with pytest.raises(ValueError, match="truncation_sigma"):
        _rasterize_gaussians(*ok, truncation_sigma=-1.0)
    with pytest.raises(ValueError, match="coefficients"):
        _rasterize_gaussians(pos, shp, torch.ones(1, 0, dtype=F64), (10, 12))
    assert _rasterize_gaussians(pos[:0], shp[:0], coef[:0], (10, 12)).shape == (
        1,
        10,
        12,
    )


# ------------------------------------------------------------------ the scene class
def _scene(count=4, k=2, *, mode="direct", seed=0, **kwargs):
    header = _tan_header((50, 64), scale_arcsec=0.4, rotation_deg=15.0, dec=55.0)
    gen = torch.Generator().manual_seed(seed)
    pix = torch.stack(
        (
            16 + 32 * torch.rand(count, generator=gen, dtype=F64),
            12 + 26 * torch.rand(count, generator=gen, dtype=F64),
        ),
        -1,
    )
    lon, lat = WCS(header).pixel_to_world(pix[:, 0], pix[:, 1])
    major = 0.5 + 1.5 * torch.rand(count, generator=gen, dtype=F64)
    gauss = torch.stack(
        (
            major,
            major * (0.4 + 0.5 * torch.rand(count, generator=gen, dtype=F64)),
            180 * torch.rand(count, generator=gen, dtype=F64),
        ),
        -1,
    )
    coef = 1.0 + torch.rand(count, k, generator=gen, dtype=F64)
    return GaussianItemScene(lon, lat, gauss, coef, mode=mode, **kwargs), header


def test_scene_is_a_scene_representation_and_renders_a_target() -> None:
    scene, header = _scene(k=1)
    assert isinstance(scene, SceneRepresentation) and isinstance(scene, torch.nn.Module)
    assert scene.component_count == 4 and scene.coefficient_count == 1
    assert scene.gaussian_parameters.shape == (4, 3)
    target = TargetRenderSpec.from_header(header)
    image = scene.render_to_target(target)
    planes = scene.render_coefficients(WCS(header), (50, 64))
    assert image.shape == (50, 64) and planes.shape == (1, 50, 64)
    torch.testing.assert_close(image, planes[0])
    torch.testing.assert_close(image.sum(), scene.coefficients.sum(), rtol=0, atol=1e-9)
    assert {len(list(scene.parameters()))} == {5}


def test_scene_band_weights_fold_the_planes() -> None:
    scene, header = _scene(k=3)
    target = TargetRenderSpec.from_header(header)
    with pytest.raises(ValueError, match="band_weights"):
        scene.render_to_target(target)
    weights = torch.tensor([0.5, -1.0, 2.0], dtype=F64)
    folded = scene.render_to_target(target, band_weights=weights)
    planes = scene.render_coefficients(header)
    torch.testing.assert_close(
        folded, torch.einsum("k,khw->hw", weights, planes), rtol=1e-10, atol=1e-11
    )
    with pytest.raises(ValueError, match="band_weights"):
        scene.render_to_target(target, band_weights=torch.ones(2, dtype=F64))
    with pytest.raises(TypeError, match="unexpected keyword"):
        scene.render_to_target(target, observation=None)


def test_scene_gradients_reach_every_parameter_and_match_finite_differences() -> None:
    for mode in MODES:
        scene, header = _scene(count=5, k=2, mode=mode, seed=3)
        weights = torch.randn(
            2, 50, 64, generator=torch.Generator().manual_seed(4), dtype=F64
        )

        def loss(scene=scene, header=header, weights=weights) -> torch.Tensor:
            return (scene.render_coefficients(header) * weights).sum()

        names = ["lon", "lat", "sigma_arcsec", "angle_deg", "coefficients"]
        params = [getattr(scene, n) for n in names]
        grads = torch.autograd.grad(loss(), params)
        assert all(float(g.abs().sum()) > 0 for g in grads), mode
        gen = torch.Generator().manual_seed(5)
        for name, param, grad in zip(names, params, grads, strict=True):
            step = 1e-7 if name in ("lon", "lat") else 1e-6
            direction = torch.randn(param.shape, generator=gen, dtype=F64)
            with torch.no_grad():
                saved = param.clone()
                param.copy_(saved + step * direction)
                plus = float(loss())
                param.copy_(saved - step * direction)
                minus = float(loss())
                param.copy_(saved)
            fd = (plus - minus) / (2 * step)
            analytic = float((grad * direction).sum())
            assert abs(fd - analytic) <= 1e-5 * max(1.0, abs(fd)), (
                mode,
                name,
                fd,
                analytic,
            )


def test_scene_width_floor_and_pixel_term() -> None:
    scene, header = _scene(count=3, k=1)
    floored, _ = _scene(count=3, k=1, min_sigma_arcsec=0.8)
    widened = GaussianItemScene(
        scene.lon.detach(),
        scene.lat.detach(),
        torch.cat(
            (
                torch.sqrt(scene.sigma_arcsec.detach() ** 2 + 0.8**2),
                scene.angle_deg.detach()[:, None],
            ),
            1,
        ),
        scene.coefficients.detach(),
    )
    torch.testing.assert_close(
        floored.render_coefficients(header),
        widened.render_coefficients(header),
        rtol=1e-12,
        atol=1e-12,
    )
    assert floored.min_sigma_arcsec == 0.8 and scene.min_sigma_arcsec is None
    # sigma = 0 is legal with a floor in the render path, but not at construction
    with pytest.raises(ValueError, match="positive"):
        GaussianItemScene(
            scene.lon.detach(),
            scene.lat.detach(),
            torch.zeros(3, 3, dtype=F64),
            scene.coefficients.detach(),
        )


def test_scene_drops_items_the_target_projection_cannot_place() -> None:
    header = _tan_header((30, 40), scale_arcsec=1.0, dec=0.0)
    lon = torch.tensor(
        [150.0, 330.0], dtype=F64
    )  # the second is on the far side of the sky
    lat = torch.tensor([0.0, 0.0], dtype=F64)
    gauss = torch.tensor([[2.0, 1.0, 0.0]] * 2, dtype=F64)
    scene = GaussianItemScene(lon, lat, gauss, torch.tensor([[4.0], [9.0]], dtype=F64))
    image = scene.render_coefficients(header)
    assert torch.isfinite(image).all()
    torch.testing.assert_close(
        image.sum(), torch.tensor(4.0, dtype=F64), rtol=0, atol=1e-9
    )
    image.sum().backward()
    assert (
        scene.coefficients.grad is not None and float(scene.coefficients.grad[1]) == 0.0
    )


# ------------------------------------------------------------------ SIP inverse edge
def _strong_sip_header(size: int) -> dict:
    # Same as tests/test_map_interpolated_mapping.py: the negative cubic makes the
    # forward map non-monotone, so the Newton world -> pixel inverse returns NaN for
    # points some 100 px beyond the detector edge.
    return {
        "NAXIS": 2, "NAXIS1": size, "NAXIS2": size,
        "CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP",
        "CRPIX1": size / 2 + 0.5, "CRPIX2": size / 2 + 0.5,
        "CRVAL1": 50.0, "CRVAL2": -20.0,
        "CD1_1": -2.0e-5, "CD1_2": 3.0e-6, "CD2_1": 3.0e-6, "CD2_2": 2.0e-5,
        "A_ORDER": 5, "B_ORDER": 5,
        "A_2_0": 3.0e-5, "A_1_1": -2.0e-5, "A_3_0": -2.0e-6, "A_1_2": 1.0e-6,
        "A_5_0": -2.0e-11,
        "B_0_2": -2.5e-5, "B_1_1": 1.5e-5, "B_0_3": -2.0e-6, "B_2_1": 1.0e-6,
        "B_0_5": 1.0e-11,
    }  # fmt: skip


def _sip_inverse_edge(
    header: dict, probe_arcsec: float, *, one_sided: bool
) -> tuple[float, float]:
    """A sky point whose SIP inverse converges while a neighbour probe_arcsec east or
    north of it (or west/south too, unless ``one_sided``) does not.

    The probes mirror the old finite-difference stamp Jacobians: one-sided +1 arcsec
    in SourceCatalogScene, central +-0.5 arcsec in GaussianItemScene.
    """
    wcs = WCS(header)

    def placed(lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        x, y = wcs.world_to_pixel(lon, lat)
        return torch.isfinite(x) & torch.isfinite(y)

    # Walk west from the reference point along its declination, in 1/4 px steps.
    lat = torch.full((1600,), -20.0, dtype=F64)
    cos_lat = math.cos(math.radians(20.0))
    lon = 50.0 - torch.arange(100.0, 500.0, 0.25, dtype=F64) * 2.0e-5 / cos_lat
    d = probe_arcsec / 3600.0
    neighbours = placed(lon + d / cos_lat, lat) & placed(lon, lat + d)
    if not one_sided:
        neighbours &= placed(lon - d / cos_lat, lat) & placed(lon, lat - d)
    hits = (placed(lon, lat) & ~neighbours).nonzero()
    assert hits.numel(), "the strong SIP header no longer has an inverse edge"
    i = int(hits[0])
    return float(lon[i]), float(lat[i])


@pytest.mark.parametrize("morphology", ["gaussian", "sersic"])
def test_source_scene_renders_finite_where_the_sip_inverse_stops_converging(
    morphology: str,
) -> None:
    """The stamp Jacobian comes from the forward map at the centre pixel, so a source
    whose centre has a pixel position renders finite even when points 1 arcsec away
    have none; before, one such source turned the whole observation into NaN."""
    header = _strong_sip_header(200)
    lon0, lat0 = _sip_inverse_edge(header, 1.0, one_sided=True)
    shape_kw = (
        {"gaussian_parameters": torch.tensor([[0.1, 0.1, 0.0], [3.0, 2.0, 0.0]])}
        if morphology == "gaussian"
        else {
            "sersic_parameters": torch.tensor(
                [[0.1, 1.0, 0.0, 1.0], [3.0, 0.7, 0.0, 1.0]]
            )
        }
    )
    lon = torch.tensor([50.0, lon0], dtype=F64, requires_grad=True)
    lat = torch.tensor([-20.0, lat0], dtype=F64, requires_grad=True)
    scene = SourceCatalogScene(
        lon,
        lat,
        morphology=["point", morphology],
        coefficients=torch.tensor([[5.0], [10.0]], dtype=F64),
        support_radius_arcsec=12.0,
        **{k: v.to(F64) for k, v in shape_kw.items()},
    )
    observation = FieldObservation(
        image=torch.zeros(200, 200, dtype=F64), header=header
    )
    stamp = scene.render_component(1, observation)
    assert stamp is not None
    assert torch.isfinite(stamp.data).all() and float(stamp.data.detach().sum()) > 0.0
    image = scene.render_observation(observation)
    assert torch.isfinite(image).all()
    # The point source keeps its flux next to the edge source.
    assert float(image.detach()[95:106, 95:106].sum()) >= 5.0 - 1e-9
    image.sum().backward()
    assert lon.grad is not None and torch.isfinite(lon.grad).all()
    assert lat.grad is not None and torch.isfinite(lat.grad).all()


def test_scene_places_items_whose_jacobian_probes_leave_the_sip_inverse() -> None:
    """An item whose centre has a pixel position is rendered even where the
    +-0.5 arcsec points around it have none; before, render_coefficients raised."""
    header = _strong_sip_header(200)
    lon0, lat0 = _sip_inverse_edge(header, 0.5, one_sided=False)
    scene = GaussianItemScene(
        torch.tensor([lon0, 50.0], dtype=F64),
        torch.tensor([lat0, -20.0], dtype=F64),
        torch.tensor([[0.3, 0.2, 10.0]] * 2, dtype=F64),
        torch.tensor([[5.0], [3.0]], dtype=F64),
    )
    _, _, index = scene._pixel_geometry(WCS(header))
    assert index.tolist() == [0, 1]
    planes = scene.render_coefficients(header)
    assert torch.isfinite(planes).all()
    torch.testing.assert_close(
        planes.sum(), torch.tensor(3.0, dtype=F64), rtol=0, atol=1e-9
    )
    planes.sum().backward()
    for param in (scene.lon, scene.lat, scene.sigma_arcsec, scene.angle_deg):
        assert param.grad is not None and torch.isfinite(param.grad).all()


def test_stamp_jacobian_is_the_exact_derivative_on_every_device(
    compute_device: torch.device,
) -> None:
    """The local pixel <-> sky Jacobian is the autograd derivative of pixel -> world,
    not a difference of world coordinates, so float32 (MPS) does not cancel: one
    longitude ulp near 50 deg is ~0.014 arcsec, a fifth of this header's pixel."""
    header = _strong_sip_header(200)
    wcs64 = WCS(header)
    # The last row is the reference pixel, where the WCS uses its grad-safe path.
    pix = torch.tensor(
        [[30.2, 40.7], [150.0, 60.4], [10.0, 180.0], [99.5, 99.5]], dtype=F64
    )
    lon, lat = wcs64.pixel_to_world(pix[:, 0], pix[:, 1])
    jac = wcs64.pixel_to_world_jacobian(pix)  # (N, world, pixel) in deg / px
    cos_lat = torch.cos(torch.deg2rad(lat))[:, None]
    exact = torch.stack((jac[:, 0] * cos_lat, jac[:, 1]), dim=-2) * 3600.0
    dtype = torch.float32 if compute_device.type == "mps" else F64
    got = _sky_from_pixel_arcsec(
        WCS(header).to(compute_device),
        pix.to(dtype).to(compute_device),
        lat.to(dtype).to(compute_device),
    )
    tol = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(
        got.cpu().to(F64), exact, rtol=0, atol=tol * float(exact.abs().max())
    )


def test_source_scene_position_gradients_include_the_jacobian() -> None:
    """Gradients flow through the Jacobian's dependence on the centre pixel
    (create_graph): they match central differences on a strongly distorted header."""
    header = _strong_sip_header(200)
    # Away from integer pixels, so the steps do not move the stamp bounds.
    lon0, lat0 = _sky_items(header, [(80.3, 90.7), (119.6, 110.4)])
    weights = torch.rand(
        200, 200, generator=torch.Generator().manual_seed(3), dtype=F64
    )
    observation = FieldObservation(
        image=torch.zeros(200, 200, dtype=F64), header=header
    )

    def loss(lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        scene = SourceCatalogScene(
            lon,
            lat,
            morphology="gaussian",
            gaussian_parameters=torch.tensor(
                [[1.0, 0.6, 25.0], [0.8, 0.5, 70.0]], dtype=F64
            ),
            coefficients=torch.tensor([[5.0], [2.0]], dtype=F64),
            support_radius_arcsec=4.0,
        )
        return (scene.render_observation(observation) * weights).sum()

    lon = lon0.clone().requires_grad_(True)
    lat = lat0.clone().requires_grad_(True)
    loss(lon, lat).backward()
    eps = 1e-3 * 0.072 / 3600.0  # a thousandth of a pixel
    numeric = []
    for which in (0, 1):
        for i in range(2):
            step = torch.zeros(2, dtype=F64)
            step[i] = eps
            if which == 0:
                diff = loss(lon0 + step, lat0) - loss(lon0 - step, lat0)
            else:
                diff = loss(lon0, lat0 + step) - loss(lon0, lat0 - step)
            numeric.append(float(diff) / (2 * eps))
    analytic = torch.cat((lon.grad, lat.grad))
    numeric_t = torch.tensor(numeric, dtype=F64)
    torch.testing.assert_close(
        analytic, numeric_t, rtol=0, atol=1e-5 * float(numeric_t.abs().max())
    )


def test_scenes_render_the_same_under_inference_mode() -> None:
    """The Jacobian's autograd runs outside inference mode, on a clone of a WCS
    built inside it, so inference-mode renders equal no_grad renders."""
    header = _strong_sip_header(200)
    lon, lat = _sky_items(header, [(80.3, 90.1), (120.0, 110.0)])
    gauss = torch.tensor([[1.0, 0.6, 25.0], [0.8, 0.5, 70.0]], dtype=F64)
    coef = torch.tensor([[5.0], [2.0]], dtype=F64)

    def render() -> tuple[torch.Tensor, torch.Tensor]:
        sources = SourceCatalogScene(
            lon,
            lat,
            morphology=["gaussian", "sersic"],
            gaussian_parameters=gauss,
            sersic_parameters=torch.tensor([[0.8, 0.7, 25.0, 1.5]] * 2, dtype=F64),
            coefficients=coef,
            support_radius_arcsec=4.0,
        ).render_observation(
            FieldObservation(image=torch.zeros(200, 200, dtype=F64), header=header)
        )
        items = GaussianItemScene(lon, lat, gauss, coef).render_coefficients(header)
        return sources, items

    with torch.no_grad():
        expected = render()
    with torch.inference_mode():
        got = render()
    for a, b in zip(got, expected):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a.clone(), b, rtol=0, atol=0)


def test_scene_float32_parameters_render_in_float32() -> None:
    scene, header = _scene(count=3, k=2)
    scene32 = GaussianItemScene(
        scene.lon.detach().double(),
        scene.lat.detach().double(),
        scene.gaussian_parameters.detach().double(),
        scene.coefficients.detach().float(),
        mode="pyramid",
    )
    out = scene32.render_coefficients(header).detach()
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    full = scene.render_coefficients(header).detach()
    torch.testing.assert_close(
        out.double(), full, rtol=0, atol=2e-4 * float(full.max())
    )


def test_scene_rejects_bad_parameters_and_grids() -> None:
    lon = torch.tensor([10.0], dtype=F64)
    lat = torch.tensor([5.0], dtype=F64)
    gauss = torch.tensor([[2.0, 1.0, 0.0]], dtype=F64)
    coef = torch.ones(1, 1, dtype=F64)
    with pytest.raises(ValueError, match="minor sigma cannot exceed"):
        GaussianItemScene(lon, lat, torch.tensor([[1.0, 2.0, 0.0]], dtype=F64), coef)
    with pytest.raises(ValueError, match="positive"):
        GaussianItemScene(lon, lat, torch.tensor([[-1.0, -2.0, 0.0]], dtype=F64), coef)
    with pytest.raises(ValueError, match="shape"):
        GaussianItemScene(lon, lat, gauss[:, :2], coef)
    with pytest.raises(ValueError, match="same number"):
        GaussianItemScene(lon, lat.repeat(2), gauss, coef)
    with pytest.raises(ValueError, match="lat"):
        GaussianItemScene(lon, torch.tensor([95.0], dtype=F64), gauss, coef)
    with pytest.raises(ValueError, match="finite"):
        GaussianItemScene(
            lon, lat, torch.tensor([[float("inf"), 1.0, 0.0]], dtype=F64), coef
        )
    with pytest.raises(TypeError, match="floating"):
        GaussianItemScene(torch.tensor([10]), lat, gauss, coef)
    with pytest.raises(ValueError, match="coefficients"):
        GaussianItemScene(lon, lat, gauss, torch.ones(2, 1, dtype=F64))
    with pytest.raises(ValueError, match="min_sigma_arcsec"):
        GaussianItemScene(lon, lat, gauss, coef, min_sigma_arcsec=-1.0)
    with pytest.raises(ValueError, match="mode"):
        GaussianItemScene(lon, lat, gauss, coef, mode="fast")
    with pytest.raises(ValueError, match="chunk_elements"):
        GaussianItemScene(lon, lat, gauss, coef, chunk_elements=-5)
    scene = GaussianItemScene(lon, lat, gauss, coef)
    with pytest.raises(ValueError, match="shape is required"):
        scene.render_coefficients(WCS(_tan_header((10, 10))))
    with pytest.raises(ValueError, match="NAXIS"):
        scene.render_coefficients({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"})
