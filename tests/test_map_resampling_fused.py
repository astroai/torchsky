"""Fused, chunked separable-kernel sampling: parity with the dense formulation."""

from __future__ import annotations

import pytest
import torch
import torchfits

import torchsky.map.image_plane_projection as projection_mod
import torchsky.map.operators.resampling as rs
from torchsky.map.image_plane_coadd import coadd_image_plane
from torchsky.map.observation import ImageObservation

pytest.importorskip("torchsky._cpp")

KERNELS = ["lanczos2", "lanczos3", "lanczos4", "mitchell_netravali", "catmull_rom"]
HEIGHT, WIDTH = 17, 23


def _dense_reference(image, variance, x, y, kernel):
    """Dense (N, T, T) formulation: explicit tap weights and independent variance."""
    dtype = image.dtype
    support = rs.kernel_support(kernel)
    offsets = torch.arange(-support + 1, support + 1)
    h, w = image.shape[-2:]
    x0 = torch.floor(x).to(torch.int64)
    y0 = torch.floor(y).to(torch.int64)
    xs = x0.unsqueeze(-1) + offsets
    ys = y0.unsqueeze(-1) + offsets
    wx = rs.kernel_weight(kernel, x.unsqueeze(-1) - xs.to(dtype))
    wy = rs.kernel_weight(kernel, y.unsqueeze(-1) - ys.to(dtype))
    if kernel.startswith("lanczos"):
        wx = rs._normalize_lanczos_coefficients(wx)
        wy = rs._normalize_lanczos_coefficients(wy)
    wx = wx * ((xs >= 0) & (xs < w))
    wy = wy * ((ys >= 0) & (ys < h))
    xs = xs.clamp(0, w - 1)
    ys = ys.clamp(0, h - 1)
    coefficient = wy.unsqueeze(-1) * wx.unsqueeze(-2)
    pixels = image[..., ys.unsqueeze(-1), xs.unsqueeze(-2)]
    sampled = (pixels * coefficient).sum(dim=(-1, -2))
    var = (variance[ys.unsqueeze(-1), xs.unsqueeze(-2)] * coefficient**2).sum(
        dim=(-1, -2)
    )
    return sampled, var


def _coordinates(count, generator):
    x = torch.rand(count, generator=generator) * (WIDTH + 6) - 3.0
    y = torch.rand(count, generator=generator) * (HEIGHT + 6) - 3.0
    edges_x = torch.tensor(
        [0.0, WIDTH - 1.0, -0.5, WIDTH - 0.5, -2.7, WIDTH + 4.1, 0.0, 3.0, 1e-4]
    )
    edges_y = torch.tensor(
        [0.0, HEIGHT - 1.0, HEIGHT - 0.5, -0.5, 5.0, -3.3, HEIGHT + 9.0, 3.0, 2.0]
    )
    return torch.cat([x, edges_x]), torch.cat([y, edges_y])


def _assert_close_to_std(actual, expected, tol=3e-6):
    scale = float(expected.std())
    assert float((actual - expected).abs().max()) <= tol * scale


@pytest.fixture
def data():
    g = torch.Generator().manual_seed(7)
    image = torch.randn(3, HEIGHT, WIDTH, generator=g) * 4.0 + 1.0
    variance = torch.rand(HEIGHT, WIDTH, generator=g) + 0.1
    x, y = _coordinates(3000, g)
    return image, variance, x, y


@pytest.mark.parametrize("kernel", KERNELS)
def test_fused_matches_dense_reference_with_edges_and_out_of_bounds(kernel, data):
    image, variance, x, y = data
    sampled, var = rs.sample_separable_with_variance(
        image,
        x_src=x.reshape(1, -1),
        y_src=y.reshape(1, -1),
        kernel=kernel,
        source_variance=variance,
    )
    expected, expected_var = _dense_reference(image, variance, x, y, kernel)
    assert sampled.shape == (3, 1, x.numel())
    assert var is not None and var.shape == (1, x.numel())
    _assert_close_to_std(sampled[:, 0], expected)
    _assert_close_to_std(var[0], expected_var)
    # fully out-of-bounds samples contribute exactly nothing
    far = x < -rs.kernel_support(kernel) - 1.0
    far = far | (x > WIDTH + rs.kernel_support(kernel))
    assert far.any()
    assert torch.all(var[0][far] == 0)


@pytest.mark.parametrize("kernel", KERNELS)
def test_old_entry_points_are_unchanged_wrappers(kernel, data):
    image, variance, x, y = data
    x2, y2 = x.reshape(1, -1), y.reshape(1, -1)
    old = rs.custom_separable_sample(image, x_src=x2, y_src=y2, kernel=kernel)
    old_var = rs.sample_image_plane_diagonal_variance(
        variance, x_src=x2, y_src=y2, kernel=kernel
    )
    sampled, var = rs.sample_separable_with_variance(
        image, x_src=x2, y_src=y2, kernel=kernel, source_variance=variance
    )
    assert old.shape == (3, 1, x.numel())
    torch.testing.assert_close(old, sampled[:, :], atol=0.0, rtol=0.0)
    torch.testing.assert_close(old_var, var, atol=0.0, rtol=0.0)
    expected, expected_var = _dense_reference(image, variance, x, y, kernel)
    _assert_close_to_std(old[:, 0], expected)
    _assert_close_to_std(old_var[0], expected_var)


@pytest.mark.parametrize("kernel", ["lanczos3", "catmull_rom"])
@pytest.mark.parametrize("chunk", [1, 5, 64, 1000, 10**9])
def test_chunk_size_invariance(kernel, chunk, data):
    image, variance, x, y = data
    x2, y2 = x.reshape(-1, 1), y.reshape(-1, 1)
    reference = rs.sample_separable_with_variance(
        image, x_src=x2, y_src=y2, kernel=kernel, source_variance=variance
    )
    chunked = rs.sample_separable_with_variance(
        image,
        x_src=x2,
        y_src=y2,
        kernel=kernel,
        source_variance=variance,
        chunk_size=chunk,
    )
    # chunking only changes reduction vectorisation: float32 rounding level
    _assert_close_to_std(chunked[0], reference[0], tol=5e-6)
    assert reference[1] is not None and chunked[1] is not None
    _assert_close_to_std(chunked[1], reference[1], tol=5e-6)


def test_chunking_bounds_the_working_set(data, monkeypatch):
    image, variance, x, y = data
    sizes = []
    original = rs._axis_interpolation_coefficients

    def spy(coordinate, **kwargs):
        sizes.append(coordinate.numel())
        return original(coordinate, **kwargs)

    monkeypatch.setattr(rs, "_axis_interpolation_coefficients", spy)
    rs.sample_separable_with_variance(
        image,
        x_src=x.reshape(1, -1),
        y_src=y.reshape(1, -1),
        kernel="lanczos3",
        source_variance=variance,
        chunk_size=256,
    )
    assert max(sizes) == 256
    assert len(sizes) == 2 * -(-x.numel() // 256)
    # default chunk shrinks as the per-sample footprint (taps, columns) grows
    small = rs._separable_chunk_size(2, 1, 4)
    large = rs._separable_chunk_size(10, 6, 4)
    assert rs._SEPARABLE_MIN_CHUNK <= large <= small <= rs._SEPARABLE_MAX_CHUNK


def test_batched_coordinates_and_batched_images(data):
    image, variance, x, y = data
    g = torch.Generator().manual_seed(3)
    images = torch.randn(2, 3, HEIGHT, WIDTH, generator=g)
    xb = torch.stack([x[:40], x[40:80]]).reshape(2, 5, 8)
    yb = torch.stack([y[:40], y[40:80]]).reshape(2, 5, 8)
    sampled, var = rs.sample_separable_with_variance(
        images,
        x_src=xb,
        y_src=yb,
        kernel="lanczos3",
        source_variance=variance,
        chunk_size=7,
    )
    assert sampled.shape == (2, 3, 5, 8)
    assert var is not None and var.shape == (2, 5, 8)
    for b in range(2):
        expected, expected_var = _dense_reference(
            images[b], variance, xb[b], yb[b], "lanczos3"
        )
        _assert_close_to_std(sampled[b], expected)
        _assert_close_to_std(var[b], expected_var)
    # single [C,H,W] image shared by a coordinate batch
    shared, _ = rs.sample_separable_with_variance(
        image, x_src=xb, y_src=yb, kernel="lanczos3"
    )
    old = rs.custom_separable_sample(image, x_src=xb, y_src=yb, kernel="lanczos3")
    torch.testing.assert_close(shared, old, atol=0.0, rtol=0.0)
    assert shared.shape == (2, 3, 5, 8)


def test_rejects_non_finite_coordinates_and_negative_variance(data):
    image, variance, x, y = data
    bad = x.clone()
    bad[3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        rs.sample_separable_with_variance(
            image, x_src=bad.reshape(1, -1), y_src=y.reshape(1, -1), kernel="lanczos3"
        )
    with pytest.raises(ValueError, match="finite"):
        rs.custom_separable_sample(
            image, x_src=bad.reshape(1, -1), y_src=y.reshape(1, -1), kernel="lanczos3"
        )
    with pytest.raises(ValueError, match="non-negative"):
        rs.sample_separable_with_variance(
            image,
            x_src=x.reshape(1, -1),
            y_src=y.reshape(1, -1),
            kernel="lanczos3",
            source_variance=-variance,
        )
    # the variance-only wrapper keeps its tolerance for non-finite coordinates
    out = rs.sample_image_plane_diagonal_variance(
        variance, x_src=bad.reshape(1, -1), y_src=y.reshape(1, -1), kernel="lanczos3"
    )
    assert out[0, 3] == 0


@pytest.mark.parametrize("kernel", ["nearest", "bilinear", "bicubic", "lanczos3"])
def test_variance_wrapper_valid_mask_zeroes_samples(kernel, data):
    _, variance, x, y = data
    x2, y2 = x.reshape(1, -1), y.reshape(1, -1)
    valid = torch.rand(x2.shape, generator=torch.Generator().manual_seed(1)) > 0.5
    full = rs.sample_image_plane_diagonal_variance(
        variance, x_src=x2, y_src=y2, kernel=kernel
    )
    masked = rs.sample_image_plane_diagonal_variance(
        variance, x_src=x2, y_src=y2, kernel=kernel, sample_valid=valid
    )
    torch.testing.assert_close(masked, torch.where(valid, full, torch.zeros_like(full)))


def test_gradient_flows_to_image_and_coordinates(data):
    image, variance, x, y = data
    image = image.clone().requires_grad_(True)
    xs = x[:50].clone().requires_grad_(True)
    ys = y[:50].clone().requires_grad_(True)
    sampled, _ = rs.sample_separable_with_variance(
        image,
        x_src=xs.reshape(1, -1),
        y_src=ys.reshape(1, -1),
        kernel="lanczos3",
        chunk_size=16,
    )
    sampled.square().sum().backward()
    image_ref = image.detach().clone().requires_grad_(True)
    xr = x[:50].clone().requires_grad_(True)
    yr = y[:50].clone().requires_grad_(True)
    expected, _ = _dense_reference(image_ref, variance, xr, yr, "lanczos3")
    expected.square().sum().backward()
    assert image.grad is not None and image_ref.grad is not None
    assert xs.grad is not None and xr.grad is not None
    scale = float(image_ref.grad.abs().max())
    assert float((image.grad - image_ref.grad).abs().max()) <= 1e-4 * scale
    scale = float(xr.grad.abs().max())
    assert float((xs.grad - xr.grad).abs().max()) <= 1e-3 * scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu(data):
    image, variance, x, y = data
    cpu = rs.sample_separable_with_variance(
        image,
        x_src=x.reshape(1, -1),
        y_src=y.reshape(1, -1),
        kernel="lanczos3",
        source_variance=variance,
    )
    gpu = rs.sample_separable_with_variance(
        image.cuda(),
        x_src=x.reshape(1, -1).cuda(),
        y_src=y.reshape(1, -1).cuda(),
        kernel="lanczos3",
        source_variance=variance.cuda(),
        chunk_size=100,
    )
    _assert_close_to_std(gpu[0].cpu(), cpu[0])
    assert gpu[1] is not None and cpu[1] is not None
    _assert_close_to_std(gpu[1].cpu(), cpu[1])


@pytest.mark.parametrize("kernel", ["lanczos3", "catmull_rom"])
def test_coadd_variance_matches_unfused_two_pass_path(tmp_path, monkeypatch, kernel):
    header = {
        "NAXIS": 2,
        "NAXIS1": 24,
        "NAXIS2": 20,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 12.0,
        "CRPIX2": 10.0,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }
    g = torch.Generator().manual_seed(11)
    path = tmp_path / "exposure.fits"
    torchfits.write(
        str(path), torch.randn(20, 24, generator=g) + 5.0, header=header, overwrite=True
    )
    target = {
        **header,
        "NAXIS1": 30,
        "NAXIS2": 26,
        "CRPIX1": 16.3,
        "CRPIX2": 12.8,
        "CDELT1": -0.0087,
        "CDELT2": 0.0091,
    }

    def run():
        return coadd_image_plane(
            [ImageObservation(str(path), weight=2.0, weight_is_inverse_variance=True)],
            target_header=target,
            resampling_kernel=kernel,
        )

    fused = run()
    monkeypatch.setattr(projection_mod, "supports_fused_variance", lambda _k: False)
    unfused = run()
    assert fused.inverse_variance is not None and unfused.inverse_variance is not None
    assert fused.correlation_factor is not None
    assert unfused.correlation_factor is not None
    torch.testing.assert_close(fused.data, unfused.data, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(fused.weight, unfused.weight, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        fused.inverse_variance, unfused.inverse_variance, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        fused.correlation_factor, unfused.correlation_factor, atol=2e-5, rtol=2e-5
    )
