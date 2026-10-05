"""Regression tests for historically critical correctness issues (see AUDIT_REPORT.md).

These guard against ``MOC.flatten`` placing its dense mask on a different
device than the UNIQ pixels.
"""

from __future__ import annotations


import pytest
import torch

from torchsky.sphere.moc import MOC, nest2uniq


def test_moc_flatten_dense_mask_matches_uniq_device_cpu() -> None:
    u = nest2uniq(4, torch.tensor([0, 1], dtype=torch.int64))
    moc = MOC(u)
    mask = moc.flatten(4)
    assert mask.dtype == torch.bool
    assert mask.device == moc.uniq.device
    assert int(mask.sum().item()) >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_moc_flatten_dense_mask_matches_uniq_device_cuda() -> None:
    dev = torch.device("cuda")
    u = nest2uniq(4, torch.tensor([0, 1], dtype=torch.int64, device=dev))
    moc = MOC(u)
    mask = moc.flatten(4)
    assert mask.device.type == "cuda"
    assert mask.device == moc.uniq.device


def test_hips_tile_world_to_sample_grid_origin_zero() -> None:
    from torchsky.viz.renderers.hips_tile import _world_to_sample_grid
    from torchsky.wcs import WCS

    header = {
        "NAXIS": 2,
        "NAXIS1": 11,
        "NAXIS2": 11,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 6.0,
        "CRPIX2": 6.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }
    wcs = WCS(header)
    ra = torch.tensor([[180.0]], dtype=torch.float64)
    dec = torch.tensor([[0.0]], dtype=torch.float64)
    gx, gy, valid = _world_to_sample_grid(ra, dec, wcs, image_h=11, image_w=11)
    assert bool(valid.all())
    torch.testing.assert_close(
        gx, torch.zeros_like(gx), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        gy, torch.zeros_like(gy), atol=1e-6, rtol=1e-6
    )


def test_zscale_limits_matches_transform_zscale() -> None:
    from torchsky.transforms import ZScale
    from torchsky.viz.renderers.image import zscale_limits

    img = torch.linspace(10.0, 110.0, 256, dtype=torch.float32).reshape(16, 16)
    z1, z2 = zscale_limits(img)
    scaled = ZScale()(img)
    expected = ((img - z1) / (z2 - z1)).clamp(0.0, 1.0)
    torch.testing.assert_close(scaled, expected, atol=1e-5, rtol=1e-5)


def test_extinction_and_foreground_batch_broadcasting() -> None:
    from torchsky.extinction import a_lambda, f99_a_lambda
    from torchsky.foreground import dib_optical_depth

    wave = torch.tensor([4000.0, 5500.0, 7000.0], dtype=torch.float64)
    a_v = torch.tensor([[0.31], [0.62]], dtype=torch.float64)
    r_v = torch.tensor(3.1, dtype=torch.float64)
    a_lam = f99_a_lambda(wave, a_v, r_v)
    assert a_lam.shape == (2, 3)

    # B == L == 3 edge case in _align_scale + f99_a_lambda
    ebv_3 = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
    ext_3x3 = a_lambda(wave, ebv=ebv_3, law="f99")
    assert ext_3x3.shape == (3, 3)
    torch.testing.assert_close(ext_3x3[0], torch.zeros(3, dtype=torch.float64))
    assert bool((ext_3x3[2] > ext_3x3[1]).all())

    # dib_optical_depth with 1D batch ebv and empty bands
    tau = dib_optical_depth(wave, ebv=ebv_3)
    assert tau.shape == (3, 3)
    tau_empty = dib_optical_depth(
        wave, ebv=ebv_3, bands=(), tau_per_ebv=(), sigma_angstrom=()
    )
    assert tau_empty.shape == (3, 3)
    assert float(tau_empty.abs().max()) == 0.0


def test_synthetic_photometry_2d_close_wavelengths_and_coincident_knot_grad() -> None:
    from torchsky.spectrophotometry import Passband, synthetic_photometry

    pb_wave = torch.tensor([4000.0, 5000.0, 6000.0], dtype=torch.float64)
    pb_trans = torch.tensor([0.1, 1.0, 0.1], dtype=torch.float64)
    pb = Passband(wavelength=pb_wave, throughput=pb_trans)

    # 2D wavelengths differing by < 1e-5 relative tolerance must still keep per-row grad
    w2d = torch.tensor(
        [[4000.0, 5000.0, 6000.0], [4000.001, 5000.001, 6000.001]],
        dtype=torch.float64,
        requires_grad=True,
    )
    f_lam = torch.stack([w2d[0] * 1e-17, w2d[1] * 2e-17], dim=0)
    ab = synthetic_photometry(w2d, f_lam, pb, output="ab_mag")
    ab.value.sum().backward()
    assert w2d.grad is not None
    assert bool((w2d.grad[1].abs() > 0).any())

    # Interior spectrum knot coinciding with passband knot must keep wave grad
    w1d = torch.tensor(
        [3500.0, 5000.0, 6500.0], dtype=torch.float64, requires_grad=True
    )
    f1d = torch.tensor([1e-17, 2e-17, 3e-17], dtype=torch.float64)
    flux = synthetic_photometry(w1d, f1d, pb, output="mean_f_lambda")
    flux.value.backward()
    assert w1d.grad is not None
    assert float(w1d.grad[1].abs()) > 0.0


def test_marginalize_association_logits_zero_weight_grad_and_float32_spectral_encoding() -> None:
    from torchsky.encoders import log_spectral_encoding
    from torchsky.encoders.photometry import marginalize_association_logits

    logits = torch.tensor([[1.0, 2.0], [3.0, 0.5]], requires_grad=True)
    w = torch.tensor([1.0, 0.0], requires_grad=True)
    out = marginalize_association_logits(logits, w)
    out[0].backward()
    assert w.grad is not None
    assert float(w.grad[1].abs()) > 0.0

    lam32 = torch.tensor([5000.0, 6000.0], dtype=torch.float32)
    enc = log_spectral_encoding(wavelength=lam32, num_frequencies=4)
    assert enc.dtype == torch.float32


def test_cosmology_comoving_distance_no_cross_batch_grad_leakage() -> None:
    from torchsky.cosmology import FLRW

    cosmo = FLRW(H0=70.0, Om0=0.3)
    z = torch.tensor([0.2, 1.5], dtype=torch.float64, requires_grad=True)
    dc = cosmo.comoving_distance(z)
    dc[0].backward()
    assert z.grad is not None
    assert float(z.grad[1].abs()) == 0.0


def test_datacube_extract_slice_narrow_width_and_to_bin_edges() -> None:
    from torchsky.spectral import DataCube, SpectralAxis

    axis = SpectralAxis(
        values=torch.tensor([4000.0, 5000.0, 6000.0], dtype=torch.float32),
        unit="Angstrom",
        type="WAVE",
        bin_edges=torch.tensor([3500.0, 4500.0, 5500.0, 6500.0], dtype=torch.float32),
    )
    cube = DataCube(
        data=torch.arange(12, dtype=torch.float32).reshape(3, 2, 2),
        spectral_axis=axis,
    )
    sl = cube.extract_slice(4900.0, width=10.0)
    assert bool(torch.isfinite(sl).all())
    torch.testing.assert_close(sl, cube.data[1])

    cube_cpu = cube.to(torch.device("cpu"))
    assert cube_cpu.spectral_axis.bin_edges is not None
    torch.testing.assert_close(cube_cpu.spectral_axis.bin_edges, axis.bin_edges)


def test_sphere_box_wrap_moc_downshift_and_sparse_interpolate() -> None:
    from torchsky.sphere.geom import SphericalBox
    from torchsky.sphere.healpix import query_box
    from torchsky.sphere.moc import nest2uniq
    from torchsky.sphere.sparse import SparseHealpixMap

    box = SphericalBox(lon_min=350.0, lon_max=10.0, lat_min=-5.0, lat_max=5.0)
    assert bool(box.contains(torch.tensor([355.0, 5.0]), torch.tensor([0.0, 0.0])).all())
    assert not bool(box.contains(torch.tensor([180.0]), torch.tensor([0.0])).any())
    pix = box.query_pixels(nside=8)
    assert pix.numel() > 0

    full_pix = query_box(4, 0.0, -90.0, 360.0, 90.0)
    assert full_pix.numel() == 12 * 16

    # _uniq_to_ranges when max_order < order of UNIQ cells (order 2 -> max_order 1)
    from torchsky.sphere.moc import _uniq_to_ranges

    u = nest2uniq(4, torch.tensor([0, 3, 7], dtype=torch.int64))
    ranges = _uniq_to_ranges(u, max_order=1)
    assert ranges == [(0, 2)]

    # SparseHealpixMap.interpolate must ignore fill_value (UNSEEN) cells
    dense = torch.full((12 * 16,), 10.0, dtype=torch.float64)
    smap = SparseHealpixMap.from_dense(dense, nside=4, nest=True)
    smap.values[0] = smap.fill_value
    val = smap.interpolate(torch.tensor([45.0]), torch.tensor([45.0]))
    assert bool(torch.isfinite(val).all())
    assert float(val[0]) > 0.0


def test_wcs_zea_domain_zpx_inverse_and_batched_iau2006() -> None:
    from torchsky.wcs import WCS
    from torchsky.wcs.iau2006 import cirs_matrix

    zea = WCS(
        {
            "NAXIS": 2,
            "CTYPE1": "RA---ZEA",
            "CTYPE2": "DEC--ZEA",
            "CRPIX1": 0.0,
            "CRPIX2": 0.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 1.0,
            "CDELT2": 1.0,
        }
    )
    ra, dec = zea.pixel_to_world(
        torch.tensor([0.0, 200.0], dtype=torch.float64),
        torch.tensor([0.0, 200.0], dtype=torch.float64),
        origin=0,
    )
    assert bool(torch.isfinite(ra[0])) and bool(torch.isnan(ra[1]))

    zpx = WCS(
        {
            "NAXIS": 2,
            "CTYPE1": "RA---ZPX",
            "CTYPE2": "DEC--ZPX",
            "CRPIX1": 50.0,
            "CRPIX2": 50.0,
            "CRVAL1": 120.0,
            "CRVAL2": 30.0,
            "CDELT1": -0.001,
            "CDELT2": 0.001,
            "PV2_1": 1.0,
            "PV2_3": 0.1,
        }
    )
    px = torch.tensor([45.0, 55.0], dtype=torch.float64)
    py = torch.tensor([48.0, 52.0], dtype=torch.float64)
    w_ra, w_dec = zpx.pixel_to_world(px, py)
    rx, ry = zpx.world_to_pixel(w_ra, w_dec)
    torch.testing.assert_close(rx, px, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ry, py, atol=1e-6, rtol=1e-6)

    mat = cirs_matrix(torch.tensor([59000.0, 59001.0], dtype=torch.float64))
    assert mat.shape == (2, 3, 3)


def test_map_field_and_sky_regressions() -> None:
    from torchsky.map.field.core import (
        BandpassConditionedField,
        OperatorConstrainedFeatureField,
        SkyLambdaFeatureField,
    )
    from torchsky.map.field.types import FieldDomain
    from torchsky.map.operators import LowRankChromaticPSF
    from torchsky.map.sky.estimator import MeshSkyEstimator
    from torchsky.map.sky.polynomial import PolynomialSkyModel
    from torchsky.map.sky.stats import huber_location, tukey_location

    domain = FieldDomain(
        center_ra_deg=180.0,
        center_dec_deg=0.0,
        spatial_scale_deg=0.01,
        wavelength_center=5000.0,
        wavelength_scale=1000.0,
    )
    ra = torch.tensor([180.0, 180.001], dtype=torch.float32)
    dec = torch.tensor([0.0, 0.001], dtype=torch.float32)
    w_vec = torch.tensor([4500.0, 5500.0], dtype=torch.float32)

    for model in (
        BandpassConditionedField(domain, hidden_dim=16, bandpass_hidden_dim=8),
        SkyLambdaFeatureField(
            domain,
            base_resolution=4,
            num_scales=2,
            features_per_plane=4,
            scale_hidden_dim=8,
            decoder_hidden_dim=8,
            refine_hidden_dim=8,
            bandpass_hidden_dim=8,
            aggregate_hidden_moments=True,
            predict_variance=True,
        ),
        OperatorConstrainedFeatureField(
            domain,
            base_resolution=4,
            num_scales=2,
            features_per_plane=4,
            scale_hidden_dim=8,
            decoder_hidden_dim=8,
            refine_hidden_dim=8,
            bandpass_hidden_dim=8,
        ),
    ):
        pred_batch = model(ra, dec, w_vec)
        pred_0 = model(ra[:1], dec[:1], float(w_vec[0].item()))
        pred_1 = model(ra[1:], dec[1:], float(w_vec[1].item()))
        torch.testing.assert_close(
            pred_batch, torch.cat([pred_0, pred_1]), atol=1e-5, rtol=1e-5
        )

    # LowRankChromaticPSF.filter_image preserves gradients to model parameters
    psf = LowRankChromaticPSF(
        grid_side=2, spectral_degree=1, base_sigma_pix=1.2, extent_pix=16.0, kernel_size=5
    )
    img = torch.ones((8, 8), dtype=torch.float32, requires_grad=True)
    yy, xx = torch.meshgrid(
        torch.arange(8, dtype=torch.float32),
        torch.arange(8, dtype=torch.float32),
        indexing="ij",
    )
    filtered = psf.filter_image(img, spatial_x=xx, spatial_y=yy, wavelength=2.0)
    filtered.square().sum().backward()
    assert psf.raw_log_sigma.grad is not None

    # huber_location / tukey_location along dim=0
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [100.0, 40.0]])
    assert huber_location(x, dim=0).shape == (2,)
    assert tukey_location(x, dim=0).shape == (2,)

    # MeshSkyEstimator float64 Gaussian filter and masked Tukey statistic
    est = MeshSkyEstimator(
        block_size=8, statistic="tukey", mesh_filter="gaussian", mesh_filter_size=3
    )
    sky_img = torch.full((16, 16), 5.0, dtype=torch.float64)
    mask = torch.zeros((16, 16), dtype=torch.bool)
    mask[:3, :3] = True
    sky_out = est(sky_img, mask=mask)
    torch.testing.assert_close(sky_out, sky_img, atol=1e-5, rtol=1e-5)

    # PolynomialSkyModel Huber IRLS backward with exact zero residuals
    poly = PolynomialSkyModel(order=1, statistic="huber")
    clean_plane = torch.full((8, 8), 3.0, dtype=torch.float32, requires_grad=True)
    fit_plane = poly(clean_plane)
    fit_plane.sum().backward()
    assert clean_plane.grad is not None
    assert bool(torch.isfinite(clean_plane.grad).all())

