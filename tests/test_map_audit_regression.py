"""Regression tests for the map audit: coarse mapping, box splat, cubes, areas, gradients."""

from __future__ import annotations

import math
import types

import pytest
import torch

import torchsky
from torchsky.map import resample
from torchsky.map.image_plane_projection import _forward_project_weighted_to_target
from torchsky.map.observation import ImageObservation


def _hdr(n, crpix, cdelt, ctype=("RA---TAN", "DEC--TAN"), crval=(150.0, 2.0)):
    return {
        "NAXIS": 2,
        "NAXIS1": n,
        "NAXIS2": n,
        "CTYPE1": ctype[0],
        "CTYPE2": ctype[1],
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CRPIX1": crpix,
        "CRPIX2": crpix,
        "CDELT1": -cdelt,
        "CDELT2": cdelt,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }


def test_resample_linear_ramp_exact_on_large_non_tan_target():
    """Large SIN target: bilinear of a linear ramp is exact, last row/column included."""
    src_h = _hdr(1400, 700.0, 2.0e-4)
    tgt_h = _hdr(1000, 500.0, 2.5e-4, ctype=("RA---SIN", "DEC--SIN"))
    xx = torch.arange(1400, dtype=torch.float32).expand(1400, 1400)
    res = resample(
        ImageObservation(data=xx.clone(), header=src_h),
        tgt_h,
        kernel="bilinear",
        semantics="surface_brightness",
    )
    src, tgt = torchsky.WCS(src_h).to("cpu"), torchsky.WCS(tgt_h).to("cpu")
    ty, tx = torch.meshgrid(
        torch.arange(1000, dtype=torch.float64),
        torch.arange(1000, dtype=torch.float64),
        indexing="ij",
    )
    xe, _ = src.world_to_pixel(*tgt.pixel_to_world(tx, ty))
    data = res.data.cpu()
    d = data.double() - xe
    ok = torch.isfinite(d) & (data != 0)
    assert ok.sum() > 900_000
    assert d[ok].abs().max().item() < 1.0e-3
    assert d[500, -1].abs().item() < 1.0e-3
    assert d[-1, 500].abs().item() < 1.0e-3


@pytest.mark.parametrize("kernel", ["drizzle_turbo", "splat_box"])
@pytest.mark.parametrize("pixfrac", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    ("n_src", "src_cdelt", "n_tgt", "tgt_cdelt"),
    [(9, 1e-4, 9, 1e-4), (9, 1e-4, 30, 0.5e-4), (40, 1e-4, 120, 2e-4)],
)
def test_forward_box_splat_conserves_flux_and_centroid(
    kernel, pixfrac, n_src, src_cdelt, n_tgt, tgt_cdelt
):
    sw = torchsky.WCS(_hdr(n_src, (n_src + 1) / 2, src_cdelt))
    tw = torchsky.WCS(_hdr(n_tgt, (n_tgt + 1) / 2 + 0.3, tgt_cdelt))
    tgt = types.SimpleNamespace(
        shape=(n_tgt, n_tgt), wcs=tw, representative_pixel_area_sr=None
    )
    img = torch.zeros(n_src, n_src)
    sx, sy = n_src // 2 + 1, n_src // 2 - 1
    img[sy, sx] = 1.0
    numer, _ = _forward_project_weighted_to_target(
        img,
        torch.where(img > 0, 1.0, 0.0),
        sw,
        tgt,
        resampling_kernel=kernel,
        value_semantics="surface_brightness",
        drizzle_pixfrac=pixfrac,
    )
    d = numer.double()
    assert d.sum().item() == pytest.approx(1.0, abs=1e-5)
    ex, ey = tw.world_to_pixel(
        *sw.pixel_to_world(
            torch.tensor([float(sx)], dtype=torch.float64),
            torch.tensor([float(sy)], dtype=torch.float64),
        )
    )
    # Independent oracle: exact 1-D interval overlaps (axis-aligned WCS pair).
    width = pixfrac * tgt_cdelt**-1 * src_cdelt
    pix = torch.arange(n_tgt, dtype=torch.float64)

    def axis(c):
        lo, hi = c - width / 2, c + width / 2
        ov = (
            torch.minimum(pix + 0.5, torch.tensor(hi))
            - torch.maximum(pix - 0.5, torch.tensor(lo))
        ).clamp_min(0)
        return ov / width

    expected = axis(ey.item()).unsqueeze(1) * axis(ex.item()).unsqueeze(0)
    assert (d - expected).abs().max().item() < 2e-5


@pytest.mark.parametrize("pixfrac", [0.5, 1.0, 2.0])
def test_cube_drizzle_pixfrac_exact_box_overlap(pixfrac):
    from torchsky.map.cube import _coadd_ifu_3d_cube

    h = _hdr(9, 5.0, 1e-4)
    img = torch.zeros(9, 9)
    img[4, 4] = 1.0
    obs = ImageObservation(
        data=img.clone(),
        header=h,
        wavelength_map=torch.full((9, 9), 5000.0),
        weight_map=torch.where(img > 0, 1.0, 0.0),
    )
    axis = torch.tensor([4000.0, 5000.0, 6000.0], dtype=torch.float64)
    r = _coadd_ifu_3d_cube(
        [obs],
        h,
        spectral_axis=axis,
        spectral_bin_edges=None,
        spectral_unit="Angstrom",
        spectral_type="WAVE",
        spectral_kernel="drizzle",
        pixfrac=pixfrac,
        spectral_pixfrac=1.0,
        emsm_scale=1.0,
        device="cpu",
    )
    w = r.weight.double()
    assert w.sum().item() == pytest.approx(1.0, abs=1e-6)
    plane = w[1]
    pix = torch.arange(9, dtype=torch.float64)
    ov = (
        torch.minimum(pix + 0.5, torch.tensor(4 + pixfrac / 2))
        - torch.maximum(pix - 0.5, torch.tensor(4 - pixfrac / 2))
    ).clamp_min(0) / pixfrac
    assert (plane - ov.unsqueeze(1) * ov.unsqueeze(0)).abs().max().item() < 1e-6


def test_coadd_cube_accepts_cube_target_for_ifu_frames():
    """The IFU recipe in docs/maps/cubes.md: Observation(ifu_wcs=...) + CubeTarget."""
    from torchsky.map import CubeTarget, Observation, ResampleConfig, coadd_cube
    from torchsky.wcs import WCS
    from torchsky.wcs.gwcs_sampling import IFUSliceWCS

    header = _hdr(16, 8.5, 0.001, crval=(150.0, 2.0))
    n_wave, n_slice = 12, 16
    rows = torch.arange(n_wave, dtype=torch.float64)
    cols = torch.arange(n_slice, dtype=torch.float64)
    ifu_wcs = IFUSliceWCS(
        ra_map=(150.0 + (cols - 7.5) * 0.001)
        .unsqueeze(0)
        .expand(n_wave, n_slice)
        .clone(),
        dec_map=torch.full((n_wave, n_slice), 2.0, dtype=torch.float64),
        wavelength_map=(5000.0 + 2.0 * rows)
        .unsqueeze(1)
        .expand(n_wave, n_slice)
        .clone(),
        slice_id_map=cols.to(torch.int64).unsqueeze(0).expand(n_wave, n_slice).clone(),
    )
    frame = Observation(
        data=torch.ones((n_wave, n_slice), dtype=torch.float32),
        ifu_wcs=ifu_wcs,
        weight=torch.ones((n_wave, n_slice), dtype=torch.float32),
        weight_is_inverse_variance=True,
    )
    target = CubeTarget(
        spatial_wcs=WCS(header), spectral_bin_edges=[4980.0, 5000.0, 5020.0, 5040.0]
    )
    cube = coadd_cube(
        [frame],
        target,
        resample=ResampleConfig(spectral_kernel="drizzle", spectral_pixfrac=0.8),
    )
    assert tuple(cube.data.shape) == (3, 16, 16)
    assert float(cube.weight.sum()) > 0.0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_pixel_area_sin_wcs_on_mps_matches_cpu_float64():
    from torchsky.map.operators.pixel_area import pixel_area_sr_map

    pix = 1.0
    h = _hdr(64, 32, pix / 3600, ctype=("RA---SIN", "DEC--SIN"))
    cpu = pixel_area_sr_map(torchsky.WCS(h), (64, 64), device=torch.device("cpu"))
    mps = pixel_area_sr_map(
        torchsky.WCS(h).to("mps"), (64, 64), device=torch.device("cpu")
    )
    expected = (pix / 3600 * math.pi / 180) ** 2
    assert torch.equal(cpu, mps)
    assert ((mps - expected).abs().max() / expected).item() < 1e-6


def _run_quad_accumulate(shift, offset):
    from torchsky.map.operators.geometry_drizzle import (
        _accumulate_quadrilateral_overlaps,
    )

    w = h = 6
    xs, ys = torch.meshgrid(
        torch.arange(1, 5, dtype=torch.float64),
        torch.arange(1, 5, dtype=torch.float64),
        indexing="xy",
    )
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    qx = torch.stack((xs - 0.5, xs + 0.5, xs + 0.5, xs - 0.5), 1) + shift + offset
    qy = torch.stack((ys - 0.5, ys - 0.5, ys + 0.5, ys + 0.5), 1) + offset
    n = torch.ones(xs.shape[0], dtype=torch.float32)
    fn, fw = torch.zeros(w * h), torch.zeros(w * h)
    _accumulate_quadrilateral_overlaps(
        qx,
        qy,
        n,
        torch.ones_like(n),
        width=w,
        height=h,
        flat_numer=fn,
        flat_weight=fw,
        flat_variance_weight=None,
    )
    return fn


@pytest.mark.parametrize("offset", [0.0, 0.123])
def test_square_drizzle_total_flux_has_zero_shift_gradient_on_aligned_grids(offset):
    shift = torch.zeros((), dtype=torch.float64, requires_grad=True)
    total = _run_quad_accumulate(shift, offset).sum()
    assert total.item() == pytest.approx(16.0, abs=1e-5)
    (grad,) = torch.autograd.grad(total, shift)
    assert abs(grad.item()) < 1e-6


def test_square_drizzle_pixel_gradient_matches_finite_difference_off_grid():
    k, eps = 1 * 6 + 1, 1e-2
    shift = torch.zeros((), dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(_run_quad_accumulate(shift, 0.123)[k], shift)
    fd = (
        _run_quad_accumulate(torch.tensor(eps, dtype=torch.float64), 0.123)[k]
        - _run_quad_accumulate(torch.tensor(-eps, dtype=torch.float64), 0.123)[k]
    ) / (2 * eps)
    assert grad.item() == pytest.approx(fd.item(), abs=1e-3)


@pytest.mark.parametrize("dx,dy", [(-0.3, 0.0), (0.0, -0.3), (0.3, 0.0), (0.0, 0.3)])
def test_point_drizzle_edge_acceptance_is_symmetric_in_x_and_y(dx, dy):
    from torchsky.map.operators.geometry_drizzle import exact_point_drizzle_project

    src = torchsky.WCS(_hdr(8, 4.0, 1e-4))
    tgt = types.SimpleNamespace(shape=(8, 8), wcs=torchsky.WCS(_hdr(8, 4.0 + dx, 1e-4)))
    tgt.wcs = torchsky.WCS({**_hdr(8, 4.0 + dx, 1e-4), "CRPIX2": 4.0 + dy})
    _, weight = exact_point_drizzle_project(
        torch.ones(8, 8),
        torch.ones(8, 8),
        src,
        tgt,
        value_semantics="surface_brightness",
        pixel_area_sr_map_fn=None,
        chunk_pixels=1 << 16,
    )
    assert weight.sum().item() == 64.0


@pytest.mark.parametrize("x0", [2.0, 2.3])
def test_splat_bilinear_position_gradient_at_integer_coordinates(x0):
    from torchsky.map.splat import splat_bilinear

    flux = torch.ones(1, 1, 1, dtype=torch.float64)
    xy = torch.tensor([[[[x0, 2.0]]]], dtype=torch.float64, requires_grad=True)
    out = splat_bilinear(flux, xy, (5, 5))
    xs = torch.arange(5, dtype=torch.float64)
    (grad,) = torch.autograd.grad((out[0] * xs[None, :]).sum(), xy)
    assert grad[0, 0, 0, 0].item() == pytest.approx(1.0, abs=1e-12)


def test_native_drizzle_accumulate_rejects_bad_shapes_and_layouts():
    from torchsky._native import cpp

    if cpp is None or not hasattr(torch.ops.torchsky, "drizzle_square_accumulate"):
        pytest.skip("native drizzle operator not built")
    op = torch.ops.torchsky.drizzle_square_accumulate
    qx = torch.tensor([[0.0, 1.0, 1.0, 0.0]] * 4, dtype=torch.float64)
    qy = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 4, dtype=torch.float64)
    ones = torch.ones(4)
    out = lambda n: (torch.zeros(n), torch.zeros(n), torch.zeros(0))  # noqa: E731
    on, ow, ov = out(9)
    op(qx, qy, ones, ones, 3, 3, on, ow, ov, False)  # valid call
    with pytest.raises(RuntimeError, match="one entry per quad"):
        op(qx, qy, torch.ones(1), ones, 3, 3, *out(9), False)
    with pytest.raises(RuntimeError, match="width \\* height"):
        op(qx, qy, ones, ones, 3, 3, *out(4), False)
    with pytest.raises(RuntimeError, match="contiguous"):
        op(qx.t().contiguous().t(), qy, ones, ones, 3, 3, *out(9), False)
    with pytest.raises(RuntimeError, match="float64"):
        op(qx.float(), qy, ones, ones, 3, 3, *out(9), False)


def test_lanczos_sinc_gradients_are_finite_at_integer_positions():
    from torchsky.map.operators.resampling import (
        custom_separable_sample,
        lanczos_weight,
    )

    x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64, requires_grad=True)
    lanczos_weight(x, a=3).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0].item() == 0.0
    torch.manual_seed(0)
    img = torch.randn(16, 16, dtype=torch.float64)
    xs = torch.full((2, 2), 7.0, dtype=torch.float64, requires_grad=True)
    ys = torch.full((2, 2), 6.25, dtype=torch.float64)
    out = custom_separable_sample(img, x_src=xs, y_src=ys, kernel="lanczos3")
    (gx,) = torch.autograd.grad(out.sum(), xs)
    assert torch.isfinite(gx).all()


def test_field_fit_diagnostics_propagates_render_errors():
    import torch.nn as nn

    from torchsky.map.field.training import compute_field_fit_diagnostics

    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, *args, **kwargs):
            raise RuntimeError("model is broken")

    class Obs:
        image = torch.zeros(4, 4)
        weight = None
        mask = None
        psf_spec = None

    with pytest.raises(Exception):
        compute_field_fit_diagnostics(
            Broken(),
            [Obs()],
            [None],
            [None],
            [None],
            num_spectral_samples=1,
            fit_duration_sec=1.0,
            steps=1,
            total_pixels=16,
            max_mem=0,
        )


def test_gaussian_nll_learns_variance_below_old_clamp():
    from torchsky.map.field.losses import _field_residual_loss, soft_bound_logvar

    torch.manual_seed(0)
    residual = 1.0e-3 * torch.randn(4000, dtype=torch.float64)
    raw = torch.tensor(-9.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = _field_residual_loss(
            residual,
            loss_kind="gaussian_nll",
            charbonnier_eps=1e-3,
            huber_delta=1.0,
            logvar=raw.expand_as(residual),
        ).mean()
        loss.backward()
        assert raw.grad is not None and raw.grad.abs() > 0
        opt.step()
    learned = torch.exp(soft_bound_logvar(raw.detach())).item()
    assert learned == pytest.approx(residual.square().mean().item(), rel=0.05)


def test_linear_band_grid_cg_exposes_documented_thresholds():
    import inspect

    from torchsky.map.field.inverse import (
        solve_joint_multiband_grid_cg,
        solve_linear_band_grid_cg,
    )

    for fn in (solve_linear_band_grid_cg, solve_joint_multiband_grid_cg):
        params = inspect.signature(fn).parameters
        assert params["well_covered_fraction"].default == 0.5
        assert params["source_dead_zone_sigma"].default == 2.0
    with pytest.raises(ValueError, match="source_dead_zone_sigma"):
        solve_linear_band_grid_cg(None, None, source_dead_zone_sigma=-1.0)  # type: ignore[arg-type]


def test_learned_psf_variance_matches_explicit_matrix_for_asymmetric_kernels(
    monkeypatch,
):
    import torchsky.map.operators.psf as psf

    torch.manual_seed(0)
    h, w = 10, 12
    kernels = torch.rand(4, 5, 5, dtype=torch.float64)
    kernels = kernels / kernels.sum(dim=(-1, -2), keepdim=True)
    monkeypatch.setattr(
        psf,
        "_cell_kernels_from_spec",
        lambda spec, device, dtype: kernels.to(device=device, dtype=dtype),
    )
    spec = types.SimpleNamespace(mode="learned_lowrank", frozen=True)
    matrix = torch.stack(
        [
            psf.apply_learned_psf_spec(e.reshape(h, w), spec).reshape(-1)
            for e in torch.eye(h * w, dtype=torch.float64)
        ],
        dim=1,
    )
    variance = torch.rand(h, w, dtype=torch.float64)
    exact = (matrix.square() @ variance.reshape(-1)).reshape(h, w)
    torch.testing.assert_close(
        psf.apply_learned_psf_variance_spec(variance, spec), exact
    )


@pytest.mark.parametrize("exponent", [0.5, 1.0, 2.0])
def test_one_over_f_psd_gradients_are_finite_at_zero_knee(exponent):
    from torchsky.map.operators.noise import one_over_f_row_psd

    e = torch.tensor(exponent, dtype=torch.float64, requires_grad=True)
    k = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    one_over_f_row_psd(16, knee_frequency=k, exponent=e).sum().backward()
    assert torch.isfinite(e.grad) and torch.isfinite(k.grad)


def test_one_over_f_row_noise_default_knee_is_learnable():
    from torchsky.map.operators.noise import OneOverFRowNoise

    model = OneOverFRowNoise((1,))
    assert model.knee_frequency.item() > 1.0e-3
    torch.manual_seed(0)
    rows = torch.cumsum(torch.randn(64, 256, dtype=torch.float32), dim=-1)
    spec = torch.fft.rfft(rows, dim=-1, norm="ortho")
    psd = model.psd(256)
    nll = (spec.abs().square() / psd + psd.log()).sum()
    nll.backward()
    grad = model.raw_knee_frequency.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().max() > 1.0e-3


def test_gaussian_kernel_rho_has_gradient_beyond_old_clamp():
    from torchsky.map.operators.learned_kernel import gaussian_kernels_from_params

    base = torch.tensor([1.5, 1.5, 0.995], dtype=torch.float64, requires_grad=True)
    kernels = gaussian_kernels_from_params(base, size=9)
    (kernels * torch.randn_like(kernels)).sum().backward()
    assert base.grad is not None and base.grad[2].abs() > 0
    inside = torch.tensor([1.5, 1.5, 0.3], dtype=torch.float64)
    # identity region is unchanged
    k_in = gaussian_kernels_from_params(inside, size=9)
    assert torch.isfinite(k_in).all()


def test_coadd_healpix_converts_galactic_wcs_to_icrs_pixels():
    from torchsky.map import Observation, coadd_healpix
    from torchsky.sphere import healpix as hp
    from torchsky.wcs import WCS
    from torchsky.wcs.frames import galactic_to_icrs

    header = {
        "NAXIS": 2,
        "NAXIS1": 16,
        "NAXIS2": 16,
        "CTYPE1": "GLON-TAN",
        "CTYPE2": "GLAT-TAN",
        "CRPIX1": 8.5,
        "CRPIX2": 8.5,
        "CRVAL1": 120.0,
        "CRVAL2": 30.0,
        "CDELT1": -0.2,
        "CDELT2": 0.2,
    }
    obs = Observation(
        data=torch.ones(16, 16),
        header=header,
        wcs=WCS(header),
        weight_map=torch.ones(16, 16),
    )
    result = coadd_healpix([obs], nside=64, nest=True)
    covered = (
        torch.nonzero(result.overlap_count > 0).reshape(-1)
        if hasattr(result, "overlap_count")
        else None
    )
    if covered is None:
        covered = torch.nonzero(result.weights > 0).reshape(-1)
    assert covered.numel() > 0
    ra, dec = hp.pix2ang(64, covered, nest=True, lonlat=True)
    ra0, dec0 = galactic_to_icrs(torch.tensor(120.0), torch.tensor(30.0))
    # Every touched cell lies within the image half-diagonal (2.3 deg) plus the
    # largest nside-64 cell radius (0.9 deg) of the ICRS image centre.
    sep = hp.angular_distance_deg(
        ra.to(torch.float64), dec.to(torch.float64), ra0.double(), dec0.double()
    )
    assert float(sep.max()) < 2.3 + 0.9


def test_ifu_angular_distance_gradient_finite_for_coincident_points():
    from torchsky.map.ifu import _angular_distance_arcsec

    ra = torch.tensor([10.0, 10.0], dtype=torch.float64, requires_grad=True)
    dec = torch.tensor([5.0, 5.001], dtype=torch.float64, requires_grad=True)
    ref_ra = torch.tensor(10.0, dtype=torch.float64)
    ref_dec = torch.tensor(5.0, dtype=torch.float64)
    dist = _angular_distance_arcsec(ra, dec, ref_ra, ref_dec)
    dist.sum().backward()
    assert torch.isfinite(ra.grad).all() and torch.isfinite(dec.grad).all()
    assert dist[1].item() == pytest.approx(3.6, rel=1e-6)


def test_ifu_cube_requires_explicit_spectral_axis():
    from torchsky.map.cube import _coadd_ifu_3d_cube

    with pytest.raises(ValueError, match="spectral_axis"):
        _coadd_ifu_3d_cube(
            [],
            _hdr(4, 2.5, 0.001),
            spectral_axis=None,
            spectral_bin_edges=None,
            spectral_unit="Angstrom",
            spectral_type="wavelength",
            spectral_kernel="linear",
            pixfrac=1.0,
            spectral_pixfrac=1.0,
            emsm_scale=1.0,
            device=None,
        )


def test_soft_bounds_have_unit_gradient_at_zero_and_nonzero_far_out():
    from torchsky.map.field.losses import soft_bound_logvar

    x = torch.tensor([0.0, 3.0, 15.0, -40.0], dtype=torch.float64, requires_grad=True)
    y = soft_bound_logvar(x)
    y.sum().backward()
    assert x.grad[0] == 1.0 and x.grad[1] == 1.0
    assert (x.grad[2:] > 0).all()
    assert y.abs().max() < 20.0


def test_astrometric_shifts_call_current_dcr_and_ephemeris_api():
    """DCR and moving-target shifts must call the live ``wcs.iau2006`` signatures."""
    from torchsky.map.image_plane_coadd import _apply_observation_astrometric_shifts

    n = 8
    hdr = _hdr(n, crpix=4.5, cdelt=1.0e-4)
    base = ImageObservation(data=torch.ones(n, n, dtype=torch.float64), header=hdr)
    plain = _apply_observation_astrometric_shifts([base], "cpu")[0]
    assert plain.header["CRVAL1"] == pytest.approx(150.0)

    dcr = ImageObservation(
        data=torch.ones(n, n, dtype=torch.float64),
        header=dict(hdr),
        wavelength=4000.0,
        dcr_zenith_deg=45.0,
        dcr_parallactic_deg=30.0,
    )
    shifted = _apply_observation_astrometric_shifts([dcr], "cpu")[0]
    moved = (shifted.header["CRVAL1"] - 150.0, shifted.header["CRVAL2"] - 2.0)
    assert moved != (0.0, 0.0)

    moving = ImageObservation(
        data=torch.ones(n, n, dtype=torch.float64),
        header=dict(hdr),
        tracking_rate=(36.0, 0.0),
        mjd_obs=60000.5,
        reference_mjd=60000.0,
    )
    shifted = _apply_observation_astrometric_shifts([moving], "cpu")[0]
    # Comoving frame subtracts 36 arcsec/hr * 12 hr = 0.12 deg of motion in RA*cos(dec).
    expected = 150.0 - 0.12 / math.cos(math.radians(2.0))
    assert shifted.header["CRVAL1"] == pytest.approx(expected, abs=1e-9)
