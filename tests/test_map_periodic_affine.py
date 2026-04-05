"""Tests for toroidal linear affine resampling (``map/periodic_affine.py``)."""

from __future__ import annotations

import pytest
import torch

from torchsky.map.periodic_affine import resample_linear_affine_periodic_bilinear_2d

pytest.importorskip("torchsky.cpp")


def test_periodic_linear_affine_identity_map() -> None:
    im = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    out = resample_linear_affine_periodic_bilinear_2d(
        im,
        a00=1.0,
        a01=0.0,
        a10=0.0,
        a11=1.0,
        bx=0.0,
        by=0.0,
    )
    torch.testing.assert_close(out, im, atol=0.0, rtol=0.0)


def test_periodic_linear_affine_rejects_singular_matrix() -> None:
    im = torch.ones(4, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="degenerate"):
        resample_linear_affine_periodic_bilinear_2d(
            im,
            a00=1.0,
            a01=2.0,
            a10=2.0,
            a11=4.0,
            bx=0.0,
            by=0.0,
        )
