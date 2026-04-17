"""Tests for the fast gridded planar nonuniform spectral remapper."""

from __future__ import annotations

import torch

from torchsky.map.spectral_nufft import resample_nonuniform_spectral_fast_2d
from torchsky.map.spectral_nonuniform import resample_nonuniform_spectral_2d


def test_fast_nonuniform_spectral_matches_direct_on_integer_grid() -> None:
    image = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    yy, xx = torch.meshgrid(
        torch.arange(6, dtype=torch.float32),
        torch.arange(6, dtype=torch.float32),
        indexing="ij",
    )
    direct = resample_nonuniform_spectral_2d(image, xx, yy, boundary_mode="zero")
    fast = resample_nonuniform_spectral_fast_2d(image, xx, yy, boundary_mode="zero")
    torch.testing.assert_close(fast, direct, atol=1e-5, rtol=1e-5)


def test_fast_nonuniform_spectral_tracks_direct_on_random_subpixel_grid() -> None:
    size = 40
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    image = torch.exp(-((xx - 21.0) ** 2 + (yy - 18.0) ** 2) / (2.0 * 5.0**2))
    x_src = xx + 0.27 * torch.sin(0.12 * yy) - 0.15
    y_src = yy + 0.23 * torch.cos(0.09 * xx) + 0.11
    direct = resample_nonuniform_spectral_2d(
        image,
        x_src,
        y_src,
        boundary_mode="zero",
        chunk_size=1024,
    )
    fast = resample_nonuniform_spectral_fast_2d(
        image,
        x_src,
        y_src,
        boundary_mode="zero",
        upsample=8,
    )
    inner = (slice(8, size - 8), slice(8, size - 8))
    torch.testing.assert_close(
        fast[inner],
        direct[inner],
        atol=0.04,
        rtol=0.04,
    )


def test_fast_nonuniform_spectral_zero_boundary_masks_outside_source() -> None:
    image = torch.zeros((8, 8), dtype=torch.float32)
    image[0, 0] = 1.0
    yy, xx = torch.meshgrid(
        torch.arange(8, dtype=torch.float32),
        torch.arange(8, dtype=torch.float32),
        indexing="ij",
    )
    out = resample_nonuniform_spectral_fast_2d(
        image,
        xx + 1.0,
        yy,
        boundary_mode="zero",
    )
    assert float(out.abs().max()) < 1e-5
