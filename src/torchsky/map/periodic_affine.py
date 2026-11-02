"""Periodic (toroidal) sampling for same-shape linear pixel maps.

Used when source and target share array dimensions but the linearized WCS maps
target pixel indices ``(iy, jx)`` to source indices via a full 2×2 affine map
plus translation, with **independent** wrapping along each axis (product torus).
This is **bilinear** on the wrapped grid, not a band-limited FFT resampler.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..device import require_float64


def resample_linear_affine_periodic_bilinear_2d(
    image: Tensor,
    *,
    a00: float,
    a01: float,
    a10: float,
    a11: float,
    bx: float,
    by: float,
) -> Tensor:
    """Resample ``image[H, W]`` with periodic boundaries (bilinear on the torus).

    For each output pixel at integer target row ``iy`` and column ``jx``,
    samples the source at continuous indices::

        sx = a00 * jx + a01 * iy + bx
        sy = a10 * jx + a11 * iy + by

    then wraps ``sx`` into ``[0, W)`` and ``sy`` into ``[0, H)`` via
    ``torch.remainder`` before bilinear interpolation.

    Args:
        image: Real ``[H, W]`` tensor.
        a00, a01, a10, a11: Affine part mapping ``(jx, iy) → (sx, sy)`` as above.
        bx, by: Translation in pixel units.
    """
    require_float64(image.device, what="periodic affine resampling")
    if image.ndim != 2:
        raise ValueError("expected a 2D [H, W] tensor")
    h, w = int(image.shape[0]), int(image.shape[1])
    if h < 1 or w < 1:
        raise ValueError("image must be non-empty")
    params = (a00, a01, a10, a11, bx, by)
    if not all(math.isfinite(p) for p in params):
        raise ValueError("affine parameters must be finite")
    det = float(a00) * float(a11) - float(a01) * float(a10)
    if abs(det) < 1e-12:
        raise ValueError("degenerate affine (determinant too small)")

    device = image.device
    dtype = image.dtype
    iy, jx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float64),
        torch.arange(w, device=device, dtype=torch.float64),
        indexing="ij",
    )
    sx = jx * float(a00) + iy * float(a01) + float(bx)
    sy = jx * float(a10) + iy * float(a11) + float(by)
    sx = torch.remainder(sx, float(w))
    sy = torch.remainder(sy, float(h))
    x0 = torch.floor(sx).long() % w
    x1 = (x0 + 1) % w
    y0 = torch.floor(sy).long() % h
    y1 = (y0 + 1) % h
    wx = (sx - torch.floor(sx)).to(dtype=dtype)
    wy = (sy - torch.floor(sy)).to(dtype=dtype)
    im = image
    v00 = im[y0, x0]
    v01 = im[y0, x1]
    v10 = im[y1, x0]
    v11 = im[y1, x1]
    return (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * (
        (1.0 - wx) * v10 + wx * v11
    )
