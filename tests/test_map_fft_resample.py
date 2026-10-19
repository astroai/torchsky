"""Unit tests for ``torchsky.map.fft_resample`` (FFT shift building block)."""

from __future__ import annotations

import pytest
import torch

from torchsky.map.fft_resample import (
    resample_diagonal_affine_fft2d,
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
