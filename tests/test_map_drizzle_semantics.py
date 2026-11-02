"""Drizzle semantics: exact gradients, rotated box splats, cube Jacobian footprints."""

from __future__ import annotations

import math
import types

import pytest
import torch

import torchsky
from torchsky.map.image_plane_projection import (
    _forward_box_axis_aligned,
    _forward_box_jacobian,
    _forward_project_weighted_to_target,
)
from torchsky.map.observation import ImageObservation
from torchsky.map.operators import geometry_drizzle as gd
from torchsky.map.operators.geometry_conservative import conservative_polygon_project
from torchsky.map.operators.geometry_drizzle import (
    _accumulate_quadrilateral_overlaps,
    _unit_box_overlap_areas_value,
    exact_point_drizzle_project,
    exact_square_drizzle_project,
)

F64 = torch.float64


def _hdr(n, crpix, cdelt, *, rot_deg=0.0, crval=(150.0, 2.0), shape=None):
    ny, nx = shape if shape is not None else (n, n)
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    return {
        "NAXIS": 2,
        "NAXIS1": nx,
        "NAXIS2": ny,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CRPIX1": crpix,
        "CRPIX2": crpix,
        "CD1_1": -cdelt * c,
        "CD1_2": cdelt * s,
        "CD2_1": cdelt * s,
        "CD2_2": cdelt * c,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }


def _target(wcs, n):
    return types.SimpleNamespace(
        shape=(n, n), wcs=wcs, representative_pixel_area_sr=None
    )


# ---------------------------------------------------------------------------
# 1. Gradients: analytic, exact at pixel-aligned geometry
# ---------------------------------------------------------------------------
def _block_quads(scale, shift, rot, offset):
    xs, ys = torch.meshgrid(
        torch.arange(2, 6, dtype=F64), torch.arange(2, 6, dtype=F64), indexing="xy"
    )
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    cx = cy = 3.5
    ux = torch.stack((xs - 0.5, xs + 0.5, xs + 0.5, xs - 0.5), 1) - cx
    uy = torch.stack((ys - 0.5, ys - 0.5, ys + 0.5, ys + 0.5), 1) - cy
    c, s = torch.cos(rot), torch.sin(rot)
    return (
        cx + offset + shift + scale * (c * ux - s * uy),
        cy + offset + scale * (s * ux + c * uy),
    )


def _drizzle_block(scale, shift, rot, offset):
    qx, qy = _block_quads(scale, shift, rot, offset)
    n = torch.ones(16, dtype=torch.float32)
    fn, fw = torch.zeros(64), torch.zeros(64)
    _accumulate_quadrilateral_overlaps(
        qx,
        qy,
        n,
        torch.ones_like(n),
        width=8,
        height=8,
        flat_numer=fn,
        flat_weight=fw,
        flat_variance_weight=None,
    )
    return fn


@pytest.mark.parametrize("offset", [0.0, 0.37])
@pytest.mark.parametrize("name", ["scale", "shift", "rot"])
def test_square_drizzle_total_flux_gradient_is_exactly_zero(name, offset):
    """Total flux is conserved for every scale/shift/rotation: d(total)/dp == 0.

    At offset 0 and scale 1 (and shift/rot 0) the block is pixel-aligned: the
    neighbouring pixels only touch, and their boundary terms must enter the gradient.
    """
    values = {
        "scale": torch.ones((), dtype=F64),
        "shift": torch.zeros((), dtype=F64),
        "rot": torch.zeros((), dtype=F64),
    }
    param = values[name].clone().requires_grad_(True)
    values[name] = param
    total = _drizzle_block(
        values["scale"], values["shift"], values["rot"], offset
    ).sum()
    assert total.item() == pytest.approx(16.0, abs=1e-5)
    (grad,) = torch.autograd.grad(total, param)
    assert abs(grad.item()) < 1e-6


@pytest.mark.parametrize("name", ["scale", "shift", "rot"])
@pytest.mark.parametrize("aligned", [True, False])
def test_square_drizzle_pixel_gradient_matches_central_difference(name, aligned):
    """Per-pixel gradient = mean of the one-sided derivatives (central difference)."""
    offset = 0.0 if aligned else 0.21
    base = {"scale": 1.0, "shift": 0.0, "rot": 0.0}

    def run(p):
        v = {k: torch.tensor(b, dtype=F64) for k, b in base.items()}
        v[name] = p
        return _drizzle_block(v["scale"], v["shift"], v["rot"], offset)

    p0 = torch.tensor(base[name], dtype=F64, requires_grad=True)
    out = run(p0)
    weights = torch.linspace(0.5, 1.5, 64, dtype=torch.float32)
    (grad,) = torch.autograd.grad((out * weights).sum(), p0)
    eps = 1.0e-4
    plus = run(torch.tensor(base[name] + eps, dtype=F64))
    minus = run(torch.tensor(base[name] - eps, dtype=F64))
    fd = (((plus - minus) * weights).sum() / (2 * eps)).item()
    assert grad.item() == pytest.approx(fd, abs=2e-3)


def test_unit_box_overlap_analytic_gradient_matches_finite_difference():
    gen = torch.Generator().manual_seed(7)
    centre = torch.rand(64, 2, generator=gen, dtype=F64) * 1.4 - 0.2
    half = 0.2 + 0.5 * torch.rand(64, 2, generator=gen, dtype=F64)
    angle = torch.rand(64, generator=gen, dtype=F64) * 6.28
    ux = torch.stack([-half[:, 0], half[:, 0], half[:, 0], -half[:, 0]])
    uy = torch.stack([-half[:, 1], -half[:, 1], half[:, 1], half[:, 1]])
    x = centre[:, 0] + ux * angle.cos() - uy * angle.sin()
    y = centre[:, 1] + ux * angle.sin() + uy * angle.cos()
    xg, yg = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
    gx, gy = torch.autograd.grad(gd._unit_box_overlap_areas(xg, yg).sum(), (xg, yg))
    eps = 1e-6
    for k in range(4):
        for grad, axis in ((gx, 0), (gy, 1)):
            dp = torch.zeros(4, 1, dtype=F64)
            dp[k] = eps
            a = (x, y)
            plus = [a[0] + dp * (axis == 0), a[1] + dp * (axis == 1)]
            minus = [a[0] - dp * (axis == 0), a[1] - dp * (axis == 1)]
            fd = (
                _unit_box_overlap_areas_value(*plus)
                - _unit_box_overlap_areas_value(*minus)
            ) / (2 * eps)
            assert (grad[k] - fd).abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# 2. Cube footprint follows the detector -> cube Jacobian
# ---------------------------------------------------------------------------
def _interval_weights(centre, width, n):
    pix = torch.arange(n, dtype=F64)
    lo, hi = centre - width / 2, centre + width / 2
    ov = (
        torch.minimum(pix + 0.5, torch.tensor(hi, dtype=F64))
        - torch.maximum(pix - 0.5, torch.tensor(lo, dtype=F64))
    ).clamp_min(0)
    return ov / width


@pytest.mark.parametrize("pixfrac", [0.5, 1.0, 1.7])
@pytest.mark.parametrize("spectral_pixfrac", [0.6, 1.0])
@pytest.mark.parametrize(
    ("cdelt_ratio_x", "cdelt_ratio_y", "dz_per_pixel"),
    [(1.0, 1.0, 20.0), (0.5, 0.5, 20.0), (2.0, 3.0, 55.0), (0.7, 1.9, 130.0)],
)
def test_cube_drizzle_footprint_follows_detector_cube_jacobian(
    pixfrac, spectral_pixfrac, cdelt_ratio_x, cdelt_ratio_y, dz_per_pixel
):
    from torchsky.map.cube import _coadd_ifu_3d_cube

    nd, nt, nw = 7, 40, 40
    det_cdelt = 1e-4
    hdr_det = _hdr(nd, 4.0, det_cdelt)
    # Target pixel = det pixel / ratio along x and y (separable: x_t depends on x_d only).
    hdr_tgt = _hdr(nt, 20.0, det_cdelt)
    hdr_tgt["CD1_1"] = -det_cdelt / cdelt_ratio_x
    hdr_tgt["CD2_2"] = det_cdelt / cdelt_ratio_y
    xx = torch.arange(nd, dtype=F64).expand(nd, nd)
    lam0, dlam_axis = 5000.0, 20.0
    wavelength_map = (lam0 + dz_per_pixel * xx).to(torch.float32)
    img = torch.zeros(nd, nd)
    img[3, 3] = 1.0
    obs = ImageObservation(
        data=img.clone(),
        header=hdr_det,
        wavelength_map=wavelength_map,
        weight_map=torch.where(img > 0, 1.0, 0.0),
    )
    axis = lam0 + dlam_axis * (torch.arange(nw, dtype=F64) - 10.0)
    result = _coadd_ifu_3d_cube(
        [obs],
        hdr_tgt,
        spectral_axis=axis,
        spectral_bin_edges=None,
        spectral_unit="Angstrom",
        spectral_type="WAVE",
        spectral_kernel="drizzle",
        pixfrac=pixfrac,
        spectral_pixfrac=spectral_pixfrac,
        emsm_scale=1.0,
        device="cpu",
    )
    w = result.weight.double()
    assert w.sum().item() == pytest.approx(1.0, abs=1e-5)

    src, tgt = torchsky.WCS(hdr_det), torchsky.WCS(hdr_tgt)
    ex, ey = tgt.world_to_pixel(
        *src.pixel_to_world(
            torch.tensor([3.0], dtype=F64), torch.tensor([3.0], dtype=F64)
        )
    )
    ez = (lam0 + dz_per_pixel * 3.0 - float(axis[0])) / dlam_axis
    wx = pixfrac * cdelt_ratio_x
    wy = pixfrac * cdelt_ratio_y
    wz = spectral_pixfrac * dz_per_pixel / dlam_axis
    expected = (
        _interval_weights(ez, wz, nw)[:, None, None]
        * _interval_weights(ey.item(), wy, nt)[None, :, None]
        * _interval_weights(ex.item(), wx, nt)[None, None, :]
    )
    # Cell-by-cell agreement with the analytic separable box overlap fixes the
    # footprint widths and the centroid (binning bias included).
    assert (w - expected).abs().max().item() < 2e-5


# ---------------------------------------------------------------------------
# 3. Forward box splat is exact for rotated / sheared WCS pairs
# ---------------------------------------------------------------------------
def _random_image(n, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, n, generator=gen) + 0.5, torch.rand(n, n, generator=gen) + 0.5


@pytest.mark.parametrize("kernel", ["drizzle_turbo", "splat_box"])
@pytest.mark.parametrize("rot_deg", [0.0, 12.0, 45.0, 90.0, 137.0])
@pytest.mark.parametrize("pixfrac", [0.4, 1.0, 1.8])
@pytest.mark.parametrize("ratio", [0.6, 1.0, 1.9])
def test_forward_box_splat_matches_exact_square_overlap_under_rotation(
    kernel, rot_deg, pixfrac, ratio
):
    n_src, n_tgt = 24, 40
    sw = torchsky.WCS(_hdr(n_src, 12.5, 1e-4))
    tw = torchsky.WCS(_hdr(n_tgt, 20.5 + 0.31, 1e-4 * ratio, rot_deg=rot_deg))
    image, weight = _random_image(n_src, 3)
    numer, wmap, var = _forward_project_weighted_to_target(
        image,
        weight,
        sw,
        _target(tw, n_tgt),
        resampling_kernel=kernel,
        value_semantics="surface_brightness",
        drizzle_pixfrac=pixfrac,
        return_variance_weight=True,
    )
    r_numer, r_wmap, r_var = exact_square_drizzle_project(
        image,
        weight,
        sw,
        _target(tw, n_tgt),
        value_semantics="surface_brightness",
        drizzle_pixfrac=pixfrac,
        pixel_area_sr_map_fn=None,
        chunk_pixels=1 << 16,
        return_variance_weight=True,
    )
    for got, ref in ((numer, r_numer), (wmap, r_wmap), (var, r_var)):
        assert (got - ref).abs().max().item() < 3e-5 * max(1.0, ref.abs().max().item())


def test_forward_box_axis_aligned_condition_is_the_mathematical_one():
    sw = torchsky.WCS(_hdr(8, 4.5, 1e-4))
    x = torch.tensor([3.0, 4.0], dtype=F64)
    for rot, expect in ((0.0, True), (90.0, True), (1e-3, False), (30.0, False)):
        tw = torchsky.WCS(_hdr(8, 4.5, 1e-4, rot_deg=rot))
        cx, cy = tw.world_to_pixel(*sw.pixel_to_world(x, x))
        jac = _forward_box_jacobian(sw, tw, x, x, cx, cy)
        assert bool(_forward_box_axis_aligned(jac).all()) is expect, rot


def test_forward_box_splat_rotated_conserves_flux_in_interior():
    n_src, n_tgt = 9, 40
    sw = torchsky.WCS(_hdr(n_src, 5.0, 1e-4))
    tw = torchsky.WCS(_hdr(n_tgt, 20.0, 1e-4 * 0.8, rot_deg=33.0))
    img = torch.zeros(n_src, n_src)
    img[4, 5] = 2.0
    numer, wmap = _forward_project_weighted_to_target(
        img,
        torch.where(img > 0, 1.0, 0.0),
        sw,
        _target(tw, n_tgt),
        resampling_kernel="drizzle_turbo",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
    )
    assert numer.double().sum().item() == pytest.approx(2.0, abs=1e-5)
    assert wmap.double().sum().item() == pytest.approx(1.0, abs=1e-5)


def test_forward_box_splat_rotated_variance_weight_distinct_from_weight():
    """Representative-area surface-brightness: source variance (w/A_s^2) != weight (w/A_s)."""
    from torchsky.map.operators.pixel_area import representative_pixel_area_sr

    n_src, n_tgt = 12, 30
    sw = torchsky.WCS(_hdr(n_src, 6.5, 1e-4))
    tw = torchsky.WCS(_hdr(n_tgt, 15.5, 1.5e-4, rot_deg=25.0))
    image, weight = _random_image(n_src, 5)
    a_t = (1.5e-4 * math.pi / 180) ** 2
    a_s = representative_pixel_area_sr(sw, (n_src, n_src), device=image.device)
    tgt = types.SimpleNamespace(
        shape=(n_tgt, n_tgt), wcs=tw, representative_pixel_area_sr=a_t
    )
    numer, wmap, var = _forward_project_weighted_to_target(
        image,
        weight,
        sw,
        tgt,
        resampling_kernel="drizzle_turbo",
        value_semantics="surface_brightness",
        drizzle_pixfrac=1.0,
        return_variance_weight=True,
    )
    _, ref_w, ref_var = _sq(
        image, weight, sw, tw, n_tgt, 1.0, return_variance_weight=True
    )
    ratio = a_t / a_s
    assert (wmap - ref_w * ratio).abs().max().item() < 1e-4 * ref_w.max().item() * ratio
    assert (
        var - ref_var * ratio**2
    ).abs().max().item() < 1e-4 * ref_var.max().item() * ratio**2


# ---------------------------------------------------------------------------
# 4. Point drizzle edge semantics (torchsky keeps drops in the outer half-pixel rim)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("x_centre", "kept"),
    [(-0.45, True), (-0.5, True), (-0.6, False), (7.45, True), (7.6, False)],
)
def test_point_drizzle_keeps_drops_in_outer_half_pixel_rim(x_centre, kept):
    """STScI drizzle 3.0 drops centres outside [0, n-1]; torchsky keeps [-0.5, n-0.5]."""
    n = 8
    src = torchsky.WCS(_hdr(n, 4.5, 1e-4))
    # Source pixel 4 (0-based) lands at target x == x_centre via a CRPIX shift.
    shift = x_centre - 4.0
    tgt = types.SimpleNamespace(
        shape=(n, n), wcs=torchsky.WCS({**_hdr(n, 4.5, 1e-4), "CRPIX1": 4.5 + shift})
    )
    img = torch.zeros(n, n)
    img[3, 4] = 1.0
    numer, weight = exact_point_drizzle_project(
        img,
        torch.where(img > 0, 1.0, 0.0),
        src,
        tgt,
        value_semantics="surface_brightness",
        pixel_area_sr_map_fn=None,
        chunk_pixels=1 << 16,
    )
    assert (weight.sum().item() == 1.0) is kept


# ---------------------------------------------------------------------------
# 5. Variants: flux conservation, variance (MC), weights and masks
# ---------------------------------------------------------------------------
def _sq(image, weight, sw, tw, n_tgt, pixfrac, **kw):
    return exact_square_drizzle_project(
        image,
        weight,
        sw,
        _target(tw, n_tgt),
        value_semantics=kw.pop("value_semantics", "surface_brightness"),
        drizzle_pixfrac=pixfrac,
        pixel_area_sr_map_fn=None,
        chunk_pixels=kw.pop("chunk_pixels", 1 << 16),
        **kw,
    )


@pytest.mark.parametrize("variant", ["square", "conservative", "turbo"])
@pytest.mark.parametrize("pixfrac", [0.3, 1.0, 2.0])
@pytest.mark.parametrize("rot_deg", [0.0, 29.0])
@pytest.mark.parametrize("ratio", [0.55, 1.0, 1.7])
def test_drizzle_variants_conserve_flux_and_propagate_variance(
    variant, pixfrac, rot_deg, ratio
):
    """Interior source: sum(numer)==sum(image*w); variance weight = sum w * c^2 (MC)."""
    n_src, n_tgt = 10, 60
    sw = torchsky.WCS(_hdr(n_src, 5.5, 1e-4))
    tw = torchsky.WCS(_hdr(n_tgt, 30.5 + 0.17, 1e-4 * ratio, rot_deg=rot_deg))
    image, weight = _random_image(n_src, 11)
    kw = dict(return_variance_weight=True)
    if variant == "square":
        numer, wmap, var = _sq(image, weight, sw, tw, n_tgt, pixfrac, **kw)
    elif variant == "conservative":
        numer, wmap, var = conservative_polygon_project(
            image,
            weight,
            sw,
            _target(tw, n_tgt),
            value_semantics="surface_brightness",
            drizzle_pixfrac=pixfrac,
            pixel_area_sr_map_fn=None,
            chunk_pixels=1 << 16,
            **kw,
        )
    else:
        numer, wmap, var = _forward_project_weighted_to_target(
            image,
            weight,
            sw,
            _target(tw, n_tgt),
            resampling_kernel="drizzle_turbo",
            value_semantics="surface_brightness",
            drizzle_pixfrac=pixfrac,
            **kw,
        )
    assert numer.double().sum().item() == pytest.approx(
        float((image * weight).double().sum()), rel=2e-5
    )
    assert wmap.double().sum().item() == pytest.approx(
        float(weight.double().sum()), rel=2e-5
    )
    # Variance weight: sum_i w_i c_i^2 never exceeds the weight and is positive.
    assert 0.0 < var.sum().item() <= wmap.sum().item() * (1 + 1e-4)


@pytest.mark.parametrize("variant", ["square", "turbo"])
@pytest.mark.parametrize("rot_deg", [0.0, 31.0])
def test_drizzle_variance_weight_matches_monte_carlo(variant, rot_deg):
    """Var[numer/weight] = variance_weight / weight**2 for inverse-variance weights."""
    n_src, n_tgt, trials = 14, 26, 6000
    sw = torchsky.WCS(_hdr(n_src, 7.5, 1e-4))
    tw = torchsky.WCS(_hdr(n_tgt, 13.5 + 0.2, 1.3e-4, rot_deg=rot_deg))
    gen = torch.Generator().manual_seed(0)
    weight = torch.rand(n_src, n_src, generator=gen) * 3.0 + 0.5
    sigma = weight.rsqrt()

    def project(image, return_variance):
        if variant == "square":
            return _sq(
                image,
                weight,
                sw,
                tw,
                n_tgt,
                1.2,
                return_variance_weight=return_variance,
            )
        return _forward_project_weighted_to_target(
            image,
            weight,
            sw,
            _target(tw, n_tgt),
            resampling_kernel="drizzle_turbo",
            value_semantics="surface_brightness",
            drizzle_pixfrac=1.2,
            return_variance_weight=return_variance,
        )

    _, wmap, var = project(torch.zeros(n_src, n_src), True)
    outs = []
    for _ in range(trials):
        noise = torch.randn(n_src, n_src, generator=gen) * sigma
        numer, _ = project(noise, False)
        outs.append(numer / wmap.clamp_min(1e-12))
    sample = torch.stack(outs).double()
    mc_var = sample.var(dim=0)
    expected = (var / wmap.clamp_min(1e-12) ** 2).double()
    core = (wmap > 0.5 * wmap.max()) & (wmap > 0)
    assert core.sum() > 20
    assert (mc_var[core] / expected[core]).mean().item() == pytest.approx(1.0, abs=0.03)
    assert ((mc_var[core] / expected[core]) - 1).abs().max().item() < 0.2


@pytest.mark.parametrize("kernel", ["square", "turbo", "conservative"])
def test_drizzle_zero_weight_and_nonfinite_pixels_leave_no_trace(kernel):
    n_src, n_tgt = 12, 30
    sw = torchsky.WCS(_hdr(n_src, 6.5, 1e-4))
    tw = torchsky.WCS(_hdr(n_tgt, 15.5, 1.1e-4, rot_deg=17.0))
    image, weight = _random_image(n_src, 9)
    bad = torch.zeros(n_src, n_src, dtype=torch.bool)
    bad[2, 3] = bad[7, 7] = bad[5, 1] = True
    dirty_image, dirty_weight = image.clone(), weight.clone()
    dirty_weight[bad] = 0.0
    dirty_image[2, 3] = float("nan")
    dirty_image[7, 7] = 1e30
    dirty_weight[5, 1] = float("nan")
    clean_weight = weight.clone()
    clean_weight[bad] = 0.0
    clean_image = torch.where(bad, torch.zeros(()), image)

    def run(img, w):
        if kernel == "square":
            return _sq(img, w, sw, tw, n_tgt, 1.0)
        if kernel == "conservative":
            return conservative_polygon_project(
                img,
                w,
                sw,
                _target(tw, n_tgt),
                value_semantics="surface_brightness",
                drizzle_pixfrac=1.0,
                pixel_area_sr_map_fn=None,
                chunk_pixels=1 << 16,
            )
        return _forward_project_weighted_to_target(
            img,
            w,
            sw,
            _target(tw, n_tgt),
            resampling_kernel="drizzle_turbo",
            value_semantics="surface_brightness",
            drizzle_pixfrac=1.0,
        )

    got = run(dirty_image, dirty_weight)
    ref = run(clean_image, clean_weight)
    for g, r in zip(got, ref):
        assert torch.isfinite(g).all()
        assert torch.equal(g, r)


@pytest.mark.parametrize("pixfrac", [1e-3, 0.05, 2.0])
def test_square_drizzle_tiny_and_max_pixfrac_conserve_flux(pixfrac):
    n_src, n_tgt = 8, 30
    sw = torchsky.WCS(_hdr(n_src, 4.5, 1e-4))
    tw = torchsky.WCS(_hdr(n_tgt, 15.5, 0.8e-4, rot_deg=20.0))
    image, weight = _random_image(n_src, 2)
    numer, wmap = _sq(image, weight, sw, tw, n_tgt, pixfrac)
    assert numer.double().sum().item() == pytest.approx(
        float((image * weight).double().sum()), rel=3e-5
    )


@pytest.mark.parametrize("variant", ["square", "turbo"])
def test_drizzle_is_independent_of_chunking(variant, monkeypatch):
    from torchsky.map import image_plane_projection as ipp

    n_src, n_tgt = 20, 36
    sw = torchsky.WCS(_hdr(n_src, 10.5, 1e-4))
    tw = torchsky.WCS(_hdr(n_tgt, 18.5, 0.9e-4, rot_deg=23.0))
    image, weight = _random_image(n_src, 4)

    def run():
        if variant == "square":
            return _sq(image, weight, sw, tw, n_tgt, 1.0, chunk_pixels=7)
        return _forward_project_weighted_to_target(
            image,
            weight,
            sw,
            _target(tw, n_tgt),
            resampling_kernel="drizzle_turbo",
            value_semantics="surface_brightness",
            drizzle_pixfrac=1.0,
        )

    whole = _sq(image, weight, sw, tw, n_tgt, 1.0) if variant == "square" else run()
    if variant == "turbo":
        monkeypatch.setattr(ipp, "_DRIZZLE_FORWARD_CHUNK_PIXELS", 7)
    chunked = run()
    for a, b in zip(whole, chunked):
        assert (a - b).abs().max().item() < 1e-5 * max(1.0, a.abs().max().item())


def test_square_drizzle_native_matches_torch_path(monkeypatch):
    from torchsky._native import cpp

    if cpp is None or not hasattr(torch.ops.torchsky, "drizzle_square_accumulate"):
        pytest.skip("native drizzle operator not built")
    n_src, n_tgt = 30, 44
    sw = torchsky.WCS(_hdr(n_src, 15.5, 1e-4))
    tw = torchsky.WCS(_hdr(n_tgt, 22.5, 1.2e-4, rot_deg=37.0))
    image, weight = _random_image(n_src, 6)
    native = _sq(image, weight, sw, tw, n_tgt, 0.8, return_variance_weight=True)
    monkeypatch.setattr(gd, "_NATIVE_OVERLAP_ENABLED", False)
    torch_path = _sq(image, weight, sw, tw, n_tgt, 0.8, return_variance_weight=True)
    for a, b in zip(native, torch_path):
        assert (a - b).abs().max().item() < 1e-5 * max(1.0, a.abs().max().item())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cube_drizzle_jacobian_footprint_mps_matches_cpu():
    from torchsky.map.cube import _coadd_ifu_3d_cube

    hdr_det, hdr_tgt = _hdr(7, 4.0, 1e-4), _hdr(30, 15.0, 1e-4 / 1.5)
    xx = torch.arange(7, dtype=torch.float32).expand(7, 7)
    img = torch.zeros(7, 7)
    img[3, 3] = 1.0
    axis = 5000.0 + 20.0 * (torch.arange(30, dtype=F64) - 10.0)

    def run(device):
        obs = ImageObservation(
            data=img.clone(),
            header=hdr_det,
            wavelength_map=5000.0 + 30.0 * xx,
            weight_map=torch.where(img > 0, 1.0, 0.0),
        )
        return _coadd_ifu_3d_cube(
            [obs],
            hdr_tgt,
            spectral_axis=axis,
            spectral_bin_edges=None,
            spectral_unit="Angstrom",
            spectral_type="WAVE",
            spectral_kernel="drizzle",
            pixfrac=1.0,
            spectral_pixfrac=1.0,
            emsm_scale=1.0,
            device=device,
        ).weight.cpu()

    mps, cpu = run("mps"), run("cpu")
    # MPS world coordinates are float32 (~0.1 pixel at this 0.4 arcsec scale), so compare
    # the footprint, not the cell values: flux is conserved and the spectral extent agrees.
    assert mps.double().sum().item() == pytest.approx(1.0, abs=1e-4)
    assert (mps.sum((1, 2)) - cpu.sum((1, 2))).abs().max().item() < 0.15
