"""Bilinear splat conserves in-bounds flux."""

from __future__ import annotations

import pytest
import torch

from torchsky.map.splat import splat_bilinear


def test_splat_conserves_in_bounds_flux() -> None:
    flux = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    xy = torch.tensor(
        [[[[1.25, 1.25], [2.25, 1.25]], [[1.25, 2.25], [2.25, 2.25]]]],
        dtype=torch.float32,
    )
    native = splat_bilinear(flux, xy, (4, 4))
    assert native.shape == (1, 4, 4)
    assert native.sum().item() == pytest.approx(float(flux.sum()), rel=1e-5)


def test_splat_is_adjoint_of_bilinear_gather() -> None:
    torch.manual_seed(0)
    flux = torch.rand(1, 3, 3)
    xy = torch.tensor(
        [[[[1.2, 1.3], [2.1, 1.4], [0.6, 2.2]],
          [[1.7, 0.8], [2.4, 2.1], [1.1, 1.9]],
          [[0.4, 0.5], [2.6, 1.2], [1.8, 2.7]]]],
        dtype=torch.float32,
    )
    native = splat_bilinear(flux, xy, (4, 4))
    # Dual: bilinear sample of a test image at the same coordinates.
    image = torch.rand(1, 4, 4)
    x = xy[..., 0]
    y = xy[..., 1]
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    sampled = torch.zeros_like(flux)
    for dx in (0.0, 1.0):
        for dy in (0.0, 1.0):
            xi = (x0 + dx).long()
            yi = (y0 + dy).long()
            w = (1.0 - torch.abs(x - (x0 + dx))) * (1.0 - torch.abs(y - (y0 + dy)))
            valid = (xi >= 0) & (xi < 4) & (yi >= 0) & (yi < 4) & (w > 0)
            w = torch.where(valid, w, torch.zeros_like(w))
            sampled = sampled + w * image[0, yi.clamp(0, 3), xi.clamp(0, 3)]
    lhs = (native * image).sum()
    rhs = (flux * sampled).sum()
    assert lhs.item() == pytest.approx(rhs.item(), rel=1e-5, abs=1e-6)


def test_out_of_bounds_is_dropped() -> None:
    flux = torch.ones(1, 1, 1)
    xy = torch.tensor([[[[-2.0, -2.0]]]])
    native = splat_bilinear(flux, xy, (3, 3))
    assert float(native.sum()) == 0.0
