"""Fast gridded spectral remapper for arbitrary source-coordinate grids.

This is the first native torchsky acceleration layer for nonuniform spectral
reprojection on the plane. It approximates arbitrary-grid Fourier evaluation by
building a globally upsampled image through FFT zero-padding and then sampling
that dense grid at arbitrary source coordinates.

It is not yet a full gridded Kaiser-Bessel NUFFT, but it provides the right
operator surface for fast planar nonuniform spectral remapping.
"""

from __future__ import annotations

import math

from torch import Tensor

from .spectral_affine import _bilinear_sample_grid, _fft_upsample_2d


SpectralBoundaryMode = str


def resample_nonuniform_spectral_fast_2d(
    image: Tensor,
    x_src: Tensor,
    y_src: Tensor,
    *,
    upsample: int = 8,
    boundary_mode: SpectralBoundaryMode = "zero",
    canvas_pad_factor: float = 2.0,
) -> Tensor:
    """Approximate arbitrary-grid spectral resampling by FFT upsampling + sampling."""
    if image.ndim != 2:
        raise ValueError("expected a 2D [H, W] tensor")
    if x_src.shape != y_src.shape:
        raise ValueError("x_src and y_src must have the same shape")
    if boundary_mode not in {"periodic", "zero"}:
        raise ValueError("boundary_mode must be 'periodic' or 'zero'")
    if int(upsample) < 1:
        raise ValueError("upsample must be >= 1")

    h_src, w_src = int(image.shape[0]), int(image.shape[1])
    if boundary_mode == "periodic":
        image_up, y0, x0 = _fft_upsample_2d(image, upsample=upsample)
    else:
        pad = max(float(canvas_pad_factor), 1.0)
        canvas_h = max(int(math.ceil(h_src * pad)), h_src)
        canvas_w = max(int(math.ceil(w_src * pad)), w_src)
        image_up, y0, x0 = _fft_upsample_2d(
            image,
            upsample=upsample,
            canvas_h=canvas_h,
            canvas_w=canvas_w,
        )
    sx = (x_src.to(dtype=image_up.dtype) + float(x0)) * float(upsample)
    sy = (y_src.to(dtype=image_up.dtype) + float(y0)) * float(upsample)
    return _bilinear_sample_grid(
        image_up,
        sx,
        sy,
        boundary_mode=boundary_mode,
    ).to(dtype=image.dtype)
