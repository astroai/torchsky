"""Global spectral affine remapper for same-shape 2D images.

This provides a torch-native spectral remapper that first constructs a globally
upsampled image via centered FFT zero-padding, then samples an arbitrary 2×2
affine source map on that dense grid. Unlike the older restricted FFT helpers,
this supports full affine mixing in one operator surface and makes boundary
semantics explicit.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


SpectralBoundaryMode = str


def _center_embed_2d(
    image: Tensor,
    *,
    out_h: int,
    out_w: int,
) -> tuple[Tensor, int, int]:
    if image.ndim != 2:
        raise ValueError("expected a 2D [H, W] tensor")
    h, w = int(image.shape[0]), int(image.shape[1])
    if out_h < h or out_w < w:
        raise ValueError("output canvas must be at least as large as the input image")
    y0 = (out_h - h) // 2
    x0 = (out_w - w) // 2
    out = image.new_zeros((out_h, out_w))
    out[y0 : y0 + h, x0 : x0 + w] = image
    return out, y0, x0


def _fft_upsample_2d(
    image: Tensor,
    *,
    upsample: int,
    canvas_h: int | None = None,
    canvas_w: int | None = None,
) -> tuple[Tensor, int, int]:
    if image.ndim != 2:
        raise ValueError("expected a 2D [H, W] tensor")
    u = int(upsample)
    if u < 1:
        raise ValueError("upsample must be >= 1")
    h, w = int(image.shape[0]), int(image.shape[1])
    base_h = h if canvas_h is None else int(canvas_h)
    base_w = w if canvas_w is None else int(canvas_w)
    base, y0, x0 = _center_embed_2d(image, out_h=base_h, out_w=base_w)
    up_h = base_h * u
    up_w = base_w * u
    dtype = base.dtype
    work = base.to(dtype=torch.float32 if dtype == torch.float16 else dtype)
    spectrum = torch.fft.fftshift(torch.fft.fft2(work), dim=(-2, -1))
    pad_left = (up_w - base_w) // 2
    pad_right = up_w - base_w - pad_left
    pad_top = (up_h - base_h) // 2
    pad_bottom = up_h - base_h - pad_top
    spectrum_up = F.pad(
        spectrum,
        (pad_left, pad_right, pad_top, pad_bottom),
    )
    image_up = torch.fft.ifft2(torch.fft.ifftshift(spectrum_up, dim=(-2, -1))).real
    image_up = image_up * (float(up_h * up_w) / float(base_h * base_w))
    return image_up.to(dtype), y0, x0


def _bilinear_sample_grid(
    image: Tensor,
    sx: Tensor,
    sy: Tensor,
    *,
    boundary_mode: SpectralBoundaryMode,
) -> Tensor:
    if image.ndim != 2:
        raise ValueError("expected a 2D [H, W] tensor")
    h, w = int(image.shape[0]), int(image.shape[1])
    if boundary_mode not in {"periodic", "zero"}:
        raise ValueError("boundary_mode must be 'periodic' or 'zero'")
    if boundary_mode == "periodic":
        sx_eff = torch.remainder(sx, float(w))
        sy_eff = torch.remainder(sy, float(h))
        valid = torch.ones_like(sx_eff, dtype=torch.bool)
    else:
        sx_eff = sx
        sy_eff = sy
        valid = (
            (sx_eff >= 0.0)
            & (sx_eff <= float(w - 1))
            & (sy_eff >= 0.0)
            & (sy_eff <= float(h - 1))
        )
    x0 = torch.floor(sx_eff).to(torch.long)
    y0 = torch.floor(sy_eff).to(torch.long)
    if boundary_mode == "periodic":
        x0 = torch.remainder(x0, w)
        y0 = torch.remainder(y0, h)
        x1 = torch.remainder(x0 + 1, w)
        y1 = torch.remainder(y0 + 1, h)
    else:
        x0 = torch.clamp(x0, 0, w - 1)
        y0 = torch.clamp(y0, 0, h - 1)
        x1 = torch.clamp(x0 + 1, 0, w - 1)
        y1 = torch.clamp(y0 + 1, 0, h - 1)
    wx = (sx_eff - torch.floor(sx_eff)).to(dtype=image.dtype)
    wy = (sy_eff - torch.floor(sy_eff)).to(dtype=image.dtype)
    v00 = image[y0, x0]
    v01 = image[y0, x1]
    v10 = image[y1, x0]
    v11 = image[y1, x1]
    out = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * (
        (1.0 - wx) * v10 + wx * v11
    )
    return torch.where(valid, out, torch.zeros_like(out))


def resample_linear_affine_spectral_2d(
    image: Tensor,
    *,
    a00: float,
    a01: float,
    a10: float,
    a11: float,
    bx: float,
    by: float,
    upsample: int = 8,
    boundary_mode: SpectralBoundaryMode = "zero",
    canvas_pad_factor: float = 2.0,
) -> Tensor:
    """Resample ``out(iy,jx) = image(A @ [jx,iy] + b)`` by global FFT upsampling.

    Args:
        image: Source ``[H, W]`` tensor.
        a00, a01, a10, a11, bx, by: Source-pixel affine map.
        upsample: FFT zero-padding factor before bilinear sampling.
        boundary_mode: ``"periodic"`` or ``"zero"``.
        canvas_pad_factor: Only used for ``boundary_mode="zero"``. The source is
            embedded in a larger zero canvas before spectral upsampling so edge
            behavior does not wrap.
    """
    if image.ndim != 2:
        raise ValueError("expected a 2D [H, W] tensor")
    params = (a00, a01, a10, a11, bx, by)
    if not all(math.isfinite(p) for p in params):
        raise ValueError("affine parameters must be finite")
    h, w = int(image.shape[0]), int(image.shape[1])
    det = float(a00) * float(a11) - float(a01) * float(a10)
    if abs(det) < 1e-12:
        raise ValueError("degenerate affine (determinant too small)")
    if boundary_mode == "periodic":
        image_up, y0, x0 = _fft_upsample_2d(image, upsample=upsample)
    else:
        pad = max(float(canvas_pad_factor), 1.0)
        canvas_h = max(int(math.ceil(h * pad)), h)
        canvas_w = max(int(math.ceil(w * pad)), w)
        image_up, y0, x0 = _fft_upsample_2d(
            image,
            upsample=upsample,
            canvas_h=canvas_h,
            canvas_w=canvas_w,
        )
    iy, jx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=torch.float64),
        torch.arange(w, device=image.device, dtype=torch.float64),
        indexing="ij",
    )
    sx = jx * float(a00) + iy * float(a01) + float(bx)
    sy = jx * float(a10) + iy * float(a11) + float(by)
    sx = (sx + float(x0)) * float(upsample)
    sy = (sy + float(y0)) * float(upsample)
    return _bilinear_sample_grid(
        image_up,
        sx,
        sy,
        boundary_mode=boundary_mode,
    )
