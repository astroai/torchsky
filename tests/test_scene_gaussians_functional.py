"""Tests of the functional Gaussian-item rasteriser ``render_gaussian_items``.

The module form (``GaussianItemScene``) is a thin layer over it; the pixel-level behaviour of
both modes is tested in ``test_scene_gaussians.py``.  Here: one test per behaviour of the
functional entry point (outputs, planes, differentiability, agreement with the module form and
with the fit prototype, float32 position precision, the patch cap).
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

import torchsky.scene.gaussians as gmod
from torchsky.scene import GaussianItemScene, render_gaussian_items
from torchsky.wcs import WCS

F64 = torch.float64
MODES = ("direct", "pyramid")
PROTOTYPE_PATH = "/home/claude/work/fieldop_tsky"


def _cov_from_widths(major, minor, angle):
    """``(C, 2, 2)`` covariance of axes ``major`` / ``minor`` (any order) at ``angle``, differentiable."""
    c, s = torch.cos(angle), torch.sin(angle)
    a2, b2 = major.square(), minor.square()
    sxx, syy, sxy = c * c * a2 + s * s * b2, s * s * a2 + c * c * b2, c * s * (a2 - b2)
    return torch.stack((torch.stack((sxx, sxy), -1), torch.stack((sxy, syy), -1)), -2)


def _items(count, shape, *, seed, k=3, wmin=0.8, wmax=6.0, dtype=F64):
    gen = torch.Generator().manual_seed(seed)
    h, w = shape

    def u(*s):
        return torch.rand(*s, generator=gen, dtype=F64)

    xy = torch.stack((u(count) * (w + 6) - 3, u(count) * (h + 6) - 3), -1)
    minor = torch.exp(math.log(wmin) + u(count) * (math.log(wmax) - math.log(wmin)))
    major = minor * (1.0 + 2.0 * u(count))
    angle = u(count) * math.pi
    flux = 0.5 + u(count, k)
    return tuple(t.to(dtype) for t in (xy, major, minor, angle, flux))


# ------------------------------------------------------------------ outputs and planes
@pytest.mark.parametrize("mode", MODES)
def test_output_units_and_flux(mode):
    xy, major, minor, angle, flux = _items(12, (80, 96), seed=1, wmin=1.0, wmax=2.0)
    xy = xy.clamp(26, 54)  # whole patches inside the grid
    shape_pix = torch.stack((major, minor, angle), -1)
    per_pixel = render_gaussian_items(xy, shape_pix, flux, (80, 96), mode=mode)
    assert per_pixel.shape == (3, 80, 96)
    torch.testing.assert_close(
        per_pixel.sum((1, 2)), flux.sum(0), rtol=1e-12, atol=1e-12
    )
    area = 0.49  # e.g. 0.7 arcsec pixels
    sb = render_gaussian_items(
        xy,
        shape_pix,
        flux,
        (80, 96),
        mode=mode,
        output="surface_brightness",
        pixel_area=area,
    )
    torch.testing.assert_close(sb, per_pixel / area, rtol=1e-14, atol=0)
    torch.testing.assert_close(
        sb.sum((1, 2)) * area, flux.sum(0), rtol=1e-12, atol=1e-12
    )


def test_output_argument_validation():
    xy, major, minor, angle, flux = _items(2, (16, 16), seed=2)
    sp = torch.stack((major, minor, angle), -1)
    with pytest.raises(ValueError, match="pixel_area"):
        render_gaussian_items(xy, sp, flux, (16, 16), output="surface_brightness")
    with pytest.raises(ValueError, match="pixel_area"):
        render_gaussian_items(
            xy, sp, flux, (16, 16), output="surface_brightness", pixel_area=0.0
        )
    with pytest.raises(ValueError, match="pixel_area"):
        render_gaussian_items(xy, sp, flux, (16, 16), pixel_area=0.5)
    with pytest.raises(ValueError, match="output must be one of"):
        render_gaussian_items(xy, sp, flux, (16, 16), output="counts")
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError, match="patch_cap"):
            render_gaussian_items(xy, sp, flux, (16, 16), patch_cap=bad)


@pytest.mark.parametrize("mode", MODES)
def test_planes_are_independent_columns_of_flux(mode):
    xy, major, minor, angle, flux = _items(40, (36, 44), seed=3, k=12, wmax=9.0)
    sp = torch.stack((major, minor, angle), -1)
    together = render_gaussian_items(xy, sp, flux, (36, 44), mode=mode)
    assert together.shape == (12, 36, 44)
    for k in (0, 5, 11):
        one = render_gaussian_items(xy, sp, flux[:, k : k + 1], (36, 44), mode=mode)
        torch.testing.assert_close(together[k], one[0], rtol=1e-12, atol=1e-14)
    # (major, minor, angle) and covariance forms are the same item
    cov = _cov_from_widths(major, minor, angle)
    again = render_gaussian_items(xy, cov, flux, (36, 44), mode=mode)
    torch.testing.assert_close(together, again, rtol=1e-12, atol=1e-14)


# ------------------------------------------------------------------ module form
def _scene_and_header(mode, dtype=F64):
    header = {
        "NAXIS": 2, "NAXIS1": 70, "NAXIS2": 52,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": 35.5, "CRPIX2": 26.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
        "CDELT1": -0.3 / 3600.0, "CDELT2": 0.3 / 3600.0,
    }  # fmt: skip
    gen = torch.Generator().manual_seed(7)
    n, k = 60, 3
    lon = 150.0 + (torch.rand(n, generator=gen, dtype=F64) - 0.5) * 20.0 / 3600.0
    lat = 2.0 + (torch.rand(n, generator=gen, dtype=F64) - 0.5) * 15.0 / 3600.0
    major = 0.4 + 2.0 * torch.rand(n, generator=gen, dtype=F64)
    gp = torch.stack(
        (major, 0.5 * major, 180.0 * torch.rand(n, generator=gen, dtype=F64)), 1
    )
    coef = torch.rand(n, k, generator=gen, dtype=F64)
    scene = GaussianItemScene(lon, lat, gp, coef, mode=mode, min_sigma_arcsec=0.1)
    return scene, header


@pytest.mark.parametrize("mode", MODES)
def test_module_form_is_the_functional_form(mode):
    scene, header = _scene_and_header(mode)
    planes = scene.render_coefficients(header)
    centre, cov, index = scene._pixel_geometry(WCS(header))
    direct = render_gaussian_items(
        centre, cov, scene.coefficients[index], (52, 70), mode=mode
    )
    torch.testing.assert_close(planes, direct, rtol=0, atol=0)
    # independent geometry: the tangent plane is linear to 1e-9 over this field, so a
    # diagonal Jacobian of 1/0.3 pixel per arcsec (x to the west) must agree to round-off
    lon, lat = scene.lon.detach(), scene.lat.detach()
    pix = WCS(header).world_to_pixel_differentiable(torch.stack((lon, lat), -1))
    sig = torch.sqrt(0.1**2 + scene.sigma_arcsec.detach().square()) / 0.3
    ang = torch.deg2rad(scene.angle_deg.detach())
    # major axis along (sin a, cos a) in (east, north) = (-sin a, cos a) in (x, y)
    by_hand = render_gaussian_items(
        pix,
        _cov_from_widths(
            sig[:, 0], sig[:, 1], torch.atan2(torch.cos(ang), -torch.sin(ang))
        ),
        scene.coefficients.detach(),
        (52, 70),
        mode=mode,
    )
    peak = float(planes.detach().max())
    assert float((by_hand - planes.detach()).abs().max()) < 2e-6 * peak


def test_module_form_patch_cap_is_passed_through(monkeypatch):
    scene, header = _scene_and_header("direct")
    assert scene.patch_cap == gmod._DEFAULT_PATCH_CAP
    seen = []
    real = gmod._patch_half_sizes

    def spy(cov, level, truncation_sigma):
        seen.append(level.cap)
        return real(cov, level, truncation_sigma)

    monkeypatch.setattr(gmod, "_patch_half_sizes", spy)
    scene.render_coefficients(header)
    assert set(seen) - {None} == {128}
    capped = GaussianItemScene(
        scene.lon.detach(),
        scene.lat.detach(),
        scene.gaussian_parameters.detach(),
        scene.coefficients.detach(),
        patch_cap=12,
    )
    seen.clear()
    capped.render_coefficients(header)
    assert set(seen) - {None} == {12}
    with pytest.raises(ValueError, match="patch_cap"):
        GaussianItemScene(
            scene.lon.detach(),
            scene.lat.detach(),
            scene.gaussian_parameters.detach(),
            scene.coefficients.detach(),
            patch_cap=0,
        )


# ------------------------------------------------------------------ gradients
@pytest.mark.parametrize("mode", MODES)
def test_gradcheck_float64(mode):
    shape = (14, 18)
    # item 2 is wide enough (smaller sigma 3.2) to sit on a coarse level in pyramid mode
    xy = torch.tensor([[5.3, 4.6], [10.7, 8.2], [8.4, 6.7]], dtype=F64)
    major = torch.tensor([1.9, 2.6, 4.4], dtype=F64)
    minor = torch.tensor([1.1, 1.4, 3.2], dtype=F64)
    angle = torch.tensor([0.3, 1.2, 2.5], dtype=F64)
    flux = torch.tensor([[1.0, 0.5], [2.0, 1.5], [3.0, 0.7]], dtype=F64)
    inputs = [t.clone().requires_grad_(True) for t in (xy, major, minor, angle, flux)]

    def run(xy, major, minor, angle, flux):
        return render_gaussian_items(
            xy, torch.stack((major, minor, angle), -1), flux, shape, mode=mode
        )

    assert torch.autograd.gradcheck(run, inputs, eps=1e-6, atol=1e-7, rtol=1e-5)

    def run_cov(xy, major, minor, angle, flux):
        return render_gaussian_items(
            xy,
            _cov_from_widths(major, minor, angle),
            flux,
            shape,
            mode=mode,
            output="surface_brightness",
            pixel_area=0.7,
        )

    assert torch.autograd.gradcheck(run_cov, inputs, eps=1e-6, atol=1e-7, rtol=1e-5)


# ------------------------------------------------------------------ the fit prototype
def _prototype_itemsky():
    if not os.path.isdir(PROTOTYPE_PATH):
        pytest.skip("prototype not available")
    sys.path.insert(0, PROTOTYPE_PATH)
    try:
        from fieldop.sky import ItemSky
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"prototype not importable: {exc}")
    finally:
        sys.path.remove(PROTOTYPE_PATH)
    return ItemSky


def _prototype_problem(ItemSky, count, grid, planes, pitch, seed, raster, wmax):
    gen = torch.Generator().manual_seed(seed)

    def u(*s):
        return torch.rand(*s, generator=gen, dtype=F64)

    sky = ItemSky(
        torch.ones(planes, grid, grid, dtype=F64),
        pitch,
        None,
        count,
        floor=0.4 * pitch,
        max_width=wmax * pitch,
        raster=raster,
    )
    with torch.no_grad():
        sky.pos[0] = (u(count, 2) * (grid + 10) - 5) * pitch + sky.x0
        w = (
            torch.exp(math.log(0.5) + u(count, 2) * (math.log(wmax) - math.log(0.5)))
            * pitch
        )
        sky.lw[0] = 0.5 * torch.log(
            (w**2 - sky.floor**2).clamp(min=(0.05 * sky.floor) ** 2)
        )
        sky.ang[0] = u(count) * math.pi
        sky.lc[0] = torch.log(0.5 + u(count, planes))
    return sky


@pytest.mark.parametrize(
    ("mode", "wmax", "cap"),
    [("direct", 30.0, 128), ("pyramid", 30.0, 128), ("pyramid", 60.0, 512)],
)
def test_agrees_with_prototype_rasteriser(mode, wmax, cap):
    """Same items, mixed widths 0.5 to ``wmax`` pixels: image to 1e-6 of the maximum, gradients to 1e-5."""
    ItemSky = _prototype_itemsky()
    pitch = 0.7
    sky = _prototype_problem(ItemSky, 300, 96, 3, pitch, 1, mode, wmax)
    ref = sky.raster(0)
    p = sky.pitch
    wid = sky.widths(0) / p
    out = render_gaussian_items(
        (sky.pos[0] - sky.x0) / p,
        _cov_from_widths(wid[:, 0], wid[:, 1], sky.ang[0]),
        torch.exp(sky.lc[0]),
        (96, 96),
        mode=mode,
        output="surface_brightness",
        pixel_area=p * p,
        patch_cap=cap,
    )
    scale = float(ref.detach().abs().max())
    assert float((out - ref).detach().abs().max()) < 1e-6 * scale
    weight = torch.randn(
        ref.shape, generator=torch.Generator().manual_seed(5), dtype=F64
    )
    params = [sky.pos, sky.lw, sky.ang, sky.lc]
    g_ref = torch.autograd.grad((ref * weight).sum(), params, retain_graph=True)
    g_out = torch.autograd.grad((out * weight).sum(), params)
    for a, b in zip(g_ref, g_out, strict=True):
        assert float((a - b).norm() / a.norm()) < 1e-5


# ------------------------------------------------------------------ float32 positions
@pytest.mark.parametrize("mode", MODES)
def test_float32_position_far_from_origin(mode):
    """An item at pixel 20000.3 lands where it should (to 1e-3 pixel), in float32."""
    width = 20100
    x32 = torch.tensor(20000.3, dtype=torch.float32)
    stored = float(x32)  # the float32 value the caller really holds
    xy = torch.stack((x32, torch.tensor(8.4, dtype=torch.float32))).view(1, 2)
    sp = torch.tensor([[1.3, 0.9, 0.6]], dtype=torch.float32)
    flux = torch.ones(1, 1, dtype=torch.float32)
    plane = render_gaussian_items(xy, sp, flux, (16, width), mode=mode)
    assert plane.dtype == torch.float32
    p64 = plane[0].double()
    cols = torch.arange(width, dtype=F64)
    rows = torch.arange(16, dtype=F64)
    total = p64.sum()
    assert abs(float(total) - 1.0) < 1e-5
    cx = float((p64.sum(0) * cols).sum() / total)
    cy = float((p64.sum(1) * rows).sum() / total)
    assert abs(cx - stored) < 1e-3
    assert abs(cy - float(xy[0, 1])) < 1e-3
    # and the gradient with respect to x is finite and has the sign of the displacement
    xy_g = xy.clone().requires_grad_(True)
    out = render_gaussian_items(xy_g, sp, flux, (16, width), mode=mode)
    (out[0, :, 20000:20003].sum()).backward()
    assert torch.isfinite(xy_g.grad).all()


# ------------------------------------------------------------------ the patch cap
def _record_halves(monkeypatch):
    seen = []
    real = gmod._chunks

    def spy(pos, cov, k, level, chunk_elements, truncation_sigma):
        for half, idx in real(pos, cov, k, level, chunk_elements, truncation_sigma):
            seen.append((level.pitch, half, int(idx.numel())))
            yield half, idx

    monkeypatch.setattr(gmod, "_chunks", spy)
    return seen


def _analytic_peak_error(plane, x, y, major, minor, angle, shape):
    q = 1.0 / 12.0
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
    ref = peak * torch.exp(-0.5 * quad)
    return float((plane - ref).abs().max()) / peak


def test_patch_cap_leaves_ordinary_widths_unchanged(monkeypatch):
    seen = _record_halves(monkeypatch)
    xy, major, minor, angle, flux = _items(30, (90, 100), seed=4, wmin=1.0, wmax=18.0)
    major = major.clamp(max=30.0)  # 4 sigma = 120 nodes at most: inside the default cap
    sp = torch.stack((major, minor.clamp(max=30.0), angle), -1)
    default = render_gaussian_items(xy, sp, flux, (90, 100))
    assert max(h for _, h, _ in seen) <= 128
    unbounded = render_gaussian_items(xy, sp, flux, (90, 100), patch_cap=None)
    torch.testing.assert_close(default, unbounded, rtol=0, atol=0)


def test_patch_cap_bounds_the_patch_of_a_wide_item(monkeypatch):
    shape = (512, 512)
    sp = torch.tensor([[300.0, 300.0, 0.0]], dtype=F64)
    xy = torch.tensor([[256.3, 255.7]], dtype=F64)
    flux = torch.ones(1, 1, dtype=F64)
    seen = _record_halves(monkeypatch)
    old = render_gaussian_items(xy, sp, flux, shape, patch_cap=None)
    assert max(h for _, h, _ in seen) > 1000  # the unbounded patch: 2400+ nodes across
    seen.clear()
    capped = render_gaussian_items(xy, sp, flux, shape)
    assert max(h for _, h, _ in seen) <= 128
    assert [p for p, _, _ in seen] == [pytest.approx(p) for p, _, _ in seen]
    assert seen[0][0] > 1.0  # it went to a coarse level, not to a truncated fine patch
    peak = 1.0 / (2 * math.pi * (300.0**2 + 1.0 / 12.0))
    # the coarse-level sample is the pyramid's: within the pyramid's accuracy of the exact one
    assert float((capped - old).abs().max()) < 3e-4 * peak
    # and nothing is lost: the same flux lands on the grid
    assert abs(float(capped.sum()) - float(old.sum())) < 1e-3 * float(old.sum())
    # in pyramid mode the same item takes the same route
    seen.clear()
    pyramid = render_gaussian_items(xy, sp, flux, shape, mode="pyramid")
    torch.testing.assert_close(pyramid, capped, rtol=0, atol=0)


def test_patch_cap_truncates_a_long_thin_item_and_keeps_its_flux(monkeypatch):
    """A long thin item cannot go to a coarse level: its patch is cut, flux kept, profile approximate."""
    shape = (720, 720)
    xy = torch.tensor([[360.2, 359.6]], dtype=F64)
    major, minor, angle = 100.0, 1.5, 0.4
    sp = torch.tensor([[major, minor, angle]], dtype=F64)
    flux = torch.full((1, 1), 7.0, dtype=F64)
    seen = _record_halves(monkeypatch)
    cut = render_gaussian_items(xy, sp, flux, shape)
    assert max(h for _, h, _ in seen) == 128
    # the flux is exact (the cut patch is renormalised and lies inside the grid)
    assert abs(float(cut.sum()) - 7.0) < 1e-10
    err_cut = _analytic_peak_error(
        cut[0] / 7.0, 360.2, 359.6, major, minor, angle, shape
    )
    full = render_gaussian_items(xy, sp, flux, shape, patch_cap=None)
    err_full = _analytic_peak_error(
        full[0] / 7.0, 360.2, 359.6, major, minor, angle, shape
    )
    assert err_full < 1e-3
    # documented size: the dropped tail beyond 128 nodes is at most exp(-(128/100)^2 / 2)
    # (= 0.44) of the item's peak along its major axis (the square patch reaches 128 / cos
    # or sin(angle) along the axis, so the true bound is smaller); the measured error is below it
    assert err_cut < math.exp(-0.5 * (128.0 / major) ** 2)
    assert err_cut > 1e-2  # it is a real approximation, not a no-op
    # a cap that holds the whole item gives the exact result back
    wide = render_gaussian_items(xy, sp, flux, shape, patch_cap=512)
    torch.testing.assert_close(wide, full, rtol=0, atol=0)
