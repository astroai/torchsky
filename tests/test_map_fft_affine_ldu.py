"""Unit tests for ``torchsky.map.fft_affine_ldu``."""

from __future__ import annotations

import pytest
import torch

from torchsky.map.fft_affine_ldu import (
    _invert_2x2,
    _per_col_vertical_shift_fft,
    resample_fft_ldu_affine_2d,
)

pytest.importorskip("torchsky._cpp")


def test_invert_2x2_identity() -> None:
    inv = _invert_2x2(1.0, 0.0, 0.0, 1.0)
    assert inv == (1.0, 0.0, 0.0, 1.0)


def test_invert_2x2_general() -> None:
    # A = [[4, 7], [2, 6]]
    # det(A) = 24 - 14 = 10
    # A^-1 = [[0.6, -0.7], [-0.2, 0.4]]
    inv00, inv01, inv10, inv11 = _invert_2x2(4.0, 7.0, 2.0, 6.0)
    assert pytest.approx(inv00) == 0.6
    assert pytest.approx(inv01) == -0.7
    assert pytest.approx(inv10) == -0.2
    assert pytest.approx(inv11) == 0.4


def test_invert_2x2_singular() -> None:
    with pytest.raises(ValueError, match="singular 2×2 map"):
        _invert_2x2(1.0, 2.0, 2.0, 4.0)


def test_per_col_vertical_shift_fft_shape_errors() -> None:
    image = torch.zeros(4, 4, 4, dtype=torch.float32)
    shift = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="expected \\[H, W\\]"):
        _per_col_vertical_shift_fft(image, shift)

    image2 = torch.zeros(4, 4, dtype=torch.float32)
    shift2 = torch.zeros(5, dtype=torch.float64)
    with pytest.raises(ValueError, match="shift_y_per_col must be \\[W\\]"):
        _per_col_vertical_shift_fft(image2, shift2)


def test_per_col_vertical_shift_fft_zero_shift() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    shift = torch.zeros(4, dtype=torch.float64)
    out = _per_col_vertical_shift_fft(image, shift)
    torch.testing.assert_close(out, image, atol=1e-5, rtol=1e-5)


def test_per_col_vertical_shift_fft_integer_shift() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    # shift column 0 by 0, column 1 by 1, column 2 by 2, column 3 by 3
    shift = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    out = _per_col_vertical_shift_fft(image, shift)

    expected = image.clone()
    expected[:, 1] = torch.roll(image[:, 1], shifts=1)
    expected[:, 2] = torch.roll(image[:, 2], shifts=2)
    expected[:, 3] = torch.roll(image[:, 3], shifts=3)

    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_resample_fft_ldu_affine_2d_shape_errors() -> None:
    image = torch.zeros(4, 4, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="expected \\[H, W\\]"):
        resample_fft_ldu_affine_2d(
            image, a00=1.0, a01=0.0, a10=0.0, a11=1.0, bx=0.0, by=0.0
        )

    image2 = torch.zeros(1, 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="fft_ldu_affine expects H,W >= 2"):
        resample_fft_ldu_affine_2d(
            image2, a00=1.0, a01=0.0, a10=0.0, a11=1.0, bx=0.0, by=0.0
        )


def test_resample_fft_ldu_affine_2d_degenerate_errors() -> None:
    image = torch.ones(4, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="LDU pivot too small"):
        resample_fft_ldu_affine_2d(
            image, a00=0.0, a01=1.0, a10=1.0, a11=0.0, bx=0.0, by=0.0
        )

    with pytest.raises(ValueError, match="degenerate LDU factor"):
        resample_fft_ldu_affine_2d(
            image, a00=1.0, a01=1.0, a10=1.0, a11=1.0, bx=0.0, by=0.0
        )


def test_resample_fft_ldu_affine_2d_identity() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    out = resample_fft_ldu_affine_2d(
        image, a00=1.0, a01=0.0, a10=0.0, a11=1.0, bx=0.0, by=0.0
    )
    torch.testing.assert_close(out, image, atol=1e-4, rtol=1e-4)


def test_resample_fft_ldu_affine_2d_translation() -> None:
    image = torch.arange(25, dtype=torch.float32).reshape(5, 5)
    out = resample_fft_ldu_affine_2d(
        image, a00=1.0, a01=0.0, a10=0.0, a11=1.0, bx=1.0, by=0.0
    )
    expected = torch.roll(image, shifts=-1, dims=1)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
