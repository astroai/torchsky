"""Tests for the native global spectral affine remapper."""

from __future__ import annotations

import torch

from torchsky.map.spectral_affine import resample_linear_affine_spectral_2d


def test_spectral_linear_affine_identity_preserves_image() -> None:
    image = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    out = resample_linear_affine_spectral_2d(
        image,
        a00=1.0,
        a01=0.0,
        a10=0.0,
        a11=1.0,
        bx=0.0,
        by=0.0,
        upsample=4,
        boundary_mode="zero",
    )
    torch.testing.assert_close(out, image, atol=1e-5, rtol=1e-5)


def test_spectral_linear_affine_zero_boundary_does_not_wrap_edges() -> None:
    image = torch.zeros((8, 8), dtype=torch.float32)
    image[0, 0] = 1.0
    out_zero = resample_linear_affine_spectral_2d(
        image,
        a00=1.0,
        a01=0.0,
        a10=0.0,
        a11=1.0,
        bx=1.0,
        by=0.0,
        upsample=8,
        boundary_mode="zero",
    )
    out_periodic = resample_linear_affine_spectral_2d(
        image,
        a00=1.0,
        a01=0.0,
        a10=0.0,
        a11=1.0,
        bx=1.0,
        by=0.0,
        upsample=8,
        boundary_mode="periodic",
    )
    assert float(out_zero.abs().max()) < 1e-5
    assert float(out_periodic[:, -1].abs().max()) > 0.5


def test_spectral_linear_affine_periodic_shift_tracks_fft_translation_for_blob() -> None:
    size = 48
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - 24.0) ** 2 + (yy - 24.0) ** 2) / (2.0 * 5.0**2))
    out = resample_linear_affine_spectral_2d(
        blob,
        a00=1.0,
        a01=0.0,
        a10=0.0,
        a11=1.0,
        bx=0.3,
        by=-0.2,
        upsample=8,
        boundary_mode="periodic",
    )

    ref_x = xx + 0.3
    ref_y = yy - 0.2
    x0 = torch.floor(ref_x).to(torch.long) % size
    y0 = torch.floor(ref_y).to(torch.long) % size
    x1 = (x0 + 1) % size
    y1 = (y0 + 1) % size
    wx = ref_x - torch.floor(ref_x)
    wy = ref_y - torch.floor(ref_y)
    ref = (1.0 - wy) * ((1.0 - wx) * blob[y0, x0] + wx * blob[y0, x1]) + wy * (
        (1.0 - wx) * blob[y1, x0] + wx * blob[y1, x1]
    )

    inner = slice(8, size - 8)
    torch.testing.assert_close(
        out[inner, inner],
        ref[inner, inner],
        atol=0.03,
        rtol=0.03,
    )
