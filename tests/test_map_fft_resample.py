"""Unit tests for ``torchsky.map.fft_resample`` (FFT shift building block)."""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.map.fft_resample import (
    resample_diagonal_affine_fft2d,
    resample_lastdim_periodic_affine_fft,
    resample_translation_fft2d,
)

pytest.importorskip("torchsky._cpp")


def test_fft_diagonal_affine_matches_translation_fft_for_pure_shift() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    d = resample_diagonal_affine_fft2d(
        image,
        scale_x_pix=1.0,
        shift_x_pix=-1.0,
        scale_y_pix=1.0,
        shift_y_pix=0.0,
    )
    t = resample_translation_fft2d(image, shift_y_pix=0.0, shift_x_pix=1.0)
    torch.testing.assert_close(d, t, atol=1e-4, rtol=1e-4)


def test_fft_shift_matches_integer_roll() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    shifted = resample_translation_fft2d(image, shift_y_pix=1.0, shift_x_pix=0.0)
    expected = torch.roll(image, shifts=1, dims=0)
    torch.testing.assert_close(shifted, expected, atol=1e-5, rtol=1e-5)


def test_fft_shift_smooth_interior_near_bilinear_reference() -> None:
    """Interior agreement vs ``grid_sample`` for a band-limited blob (not edges)."""
    size = 64
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / (2.0 * 6.0**2))

    sy, sx = 0.25, -0.125
    fft_out = resample_translation_fft2d(blob, shift_y_pix=sy, shift_x_pix=sx)

    # Reference: bilinear ``grid_sample`` with zero padding (differs at borders).
    theta = torch.empty(1, size, size, 2, dtype=torch.float32)
    y = (yy - sy) / (size - 1) * 2.0 - 1.0
    x = (xx - sx) / (size - 1) * 2.0 - 1.0
    theta[0, :, :, 0] = x
    theta[0, :, :, 1] = y
    ref = torch.nn.functional.grid_sample(
        blob.view(1, 1, size, size),
        theta,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze()

    inner = slice(12, 52)
    delta = (fft_out[inner, inner] - ref[inner, inner]).abs()
    assert float(delta.max()) < 0.02


@pytest.mark.parametrize("upsample", [64, 4096])
@pytest.mark.parametrize("n", [5, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fft_affine_preserves_dc_for_large_upsampling_factors(
    upsample: int, n: int, dtype: torch.dtype
) -> None:
    x = torch.full((2, 3, n), 7.25, dtype=dtype)
    out = resample_lastdim_periodic_affine_fft(
        x, scale=-1.375, shift=0.3125, upsample=upsample
    )
    torch.testing.assert_close(out, x, atol=4 * torch.finfo(dtype).eps, rtol=0.0)


@pytest.mark.parametrize("n", [7, 8])
@pytest.mark.parametrize("scale", [0.625, -1.5, 2.25])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fft_affine_matches_periodic_sinusoid(
    n: int, scale: float, dtype: torch.dtype
) -> None:
    frequency = 2
    phase = 0.37
    k = torch.arange(n, dtype=dtype)
    x = torch.cos(2.0 * torch.pi * frequency * k / n + phase)
    out = resample_lastdim_periodic_affine_fft(
        x, scale=scale, shift=-0.1875, upsample=64
    )
    source = torch.remainder(k * scale - 0.1875, n)
    expected = torch.cos(2.0 * torch.pi * frequency * source / n + phase)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(out, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("n", [4, 10])
def test_fft_affine_splits_even_length_nyquist(n: int) -> None:
    k = torch.arange(n, dtype=torch.float64)
    x = torch.cos(torch.pi * k)
    scale = -0.75
    shift = 0.125
    out = resample_lastdim_periodic_affine_fft(
        x, scale=scale, shift=shift, upsample=64
    )
    expected = torch.cos(torch.pi * (k * scale + shift))
    torch.testing.assert_close(out, expected, atol=2e-14, rtol=2e-14)


def test_fft_translation_splits_nyquist_on_each_axis() -> None:
    y = torch.arange(6, dtype=torch.float64)
    x = torch.arange(8, dtype=torch.float64)
    image = torch.cos(torch.pi * y[:, None]) * torch.cos(torch.pi * x[None, :])
    shift_y = 0.25
    shift_x = -0.375
    out = resample_translation_fft2d(
        image, shift_y_pix=shift_y, shift_x_pix=shift_x
    )
    expected = image * math.cos(math.pi * shift_y) * math.cos(math.pi * shift_x)
    torch.testing.assert_close(out, expected, atol=2e-14, rtol=2e-14)


def test_fft_affine_impulse_has_unit_dc_normalization() -> None:
    impulse = torch.zeros(9, dtype=torch.float64)
    impulse[3] = 1.0
    out = resample_lastdim_periodic_affine_fft(
        impulse, scale=-1.0, shift=0.375, upsample=64
    )
    torch.testing.assert_close(out.sum(), torch.tensor(1.0, dtype=out.dtype))


def test_fft_diagonal_affine_matches_separable_analytic_field() -> None:
    h, w = 7, 8
    y = torch.arange(h, dtype=torch.float64)
    x = torch.arange(w, dtype=torch.float64)
    image = torch.cos(2.0 * torch.pi * y[:, None] / h + 0.2) * torch.sin(
        2.0 * torch.pi * 2.0 * x[None, :] / w - 0.3
    )
    sy, sx = -1.25, 0.625
    by, bx = 0.125, -0.25
    out = resample_diagonal_affine_fft2d(
        image,
        scale_x_pix=sx,
        shift_x_pix=bx,
        scale_y_pix=sy,
        shift_y_pix=by,
        upsample=64,
    )
    expected = torch.cos(2.0 * torch.pi * (sy * y[:, None] + by) / h + 0.2) * (
        torch.sin(2.0 * torch.pi * 2.0 * (sx * x[None, :] + bx) / w - 0.3)
    )
    torch.testing.assert_close(out, expected, atol=2e-12, rtol=2e-12)


def test_fft_affine_applies_same_field_semantics_to_signal_and_weight() -> None:
    weight = torch.full((5, 6), 4.0, dtype=torch.float64)
    weighted_signal = weight * 3.5
    kwargs = {
        "scale_x_pix": -1.25,
        "shift_x_pix": 0.375,
        "scale_y_pix": 0.75,
        "shift_y_pix": -0.125,
    }
    sampled_weight = resample_diagonal_affine_fft2d(weight, **kwargs)
    sampled_signal = resample_diagonal_affine_fft2d(weighted_signal, **kwargs)
    torch.testing.assert_close(sampled_weight, weight)
    torch.testing.assert_close(sampled_signal / sampled_weight, torch.full_like(weight, 3.5))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fft_affine_preserves_dtype_and_device(dtype: torch.dtype) -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available() and dtype == torch.float32:
        devices.append(torch.device("mps"))
    for device in devices:
        x = torch.arange(15, dtype=dtype, device=device).reshape(3, 5)
        out = resample_lastdim_periodic_affine_fft(
            x, scale=-0.75, shift=0.125, upsample=8
        )
        assert out.dtype == dtype
        assert out.device.type == device.type


def test_fft_affine_gradcheck() -> None:
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: resample_lastdim_periodic_affine_fft(
            value, scale=-1.25, shift=0.375, upsample=8
        ),
        (x,),
    )


def test_fft_diagonal_affine_gradcheck() -> None:
    image = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: resample_diagonal_affine_fft2d(
            value,
            scale_x_pix=-0.75,
            shift_x_pix=0.125,
            scale_y_pix=1.25,
            shift_y_pix=-0.375,
            upsample=8,
        ),
        (image,),
    )
