"""Native nonuniform spectral remapper for arbitrary source-coordinate grids.

This is a torch-native reference-quality spectral remapper that evaluates the
inverse Fourier series at arbitrary real-valued source coordinates. It is not a
gridded fast NUFFT yet, but it closes the native runtime gap for distorted WCS
cases where interpolation is not the intended operator family.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .spectral_affine import _center_embed_2d


SpectralBoundaryMode = str


def resample_nonuniform_spectral_2d(
    image: Tensor,
    x_src: Tensor,
    y_src: Tensor,
    *,
    boundary_mode: SpectralBoundaryMode = "zero",
    canvas_pad_factor: float = 2.0,
    chunk_size: int = 2048,
) -> Tensor:
    """Evaluate a 2D inverse Fourier series on arbitrary source coordinates.

    Args:
        image: Source ``[H, W]`` tensor.
        x_src: Source x-coordinate samples in source-pixel units.
        y_src: Source y-coordinate samples in source-pixel units.
        boundary_mode: ``"periodic"`` or ``"zero"``. Zero mode embeds the
            source in a larger zero canvas, evaluates the spectral series there,
            and masks coordinates outside the original source footprint.
        canvas_pad_factor: Zero-boundary canvas enlargement factor.
        chunk_size: Number of output samples per evaluation chunk.
    """
    if image.ndim != 2:
        raise ValueError("expected a 2D [H, W] tensor")
    if x_src.shape != y_src.shape:
        raise ValueError("x_src and y_src must have the same shape")
    if boundary_mode not in {"periodic", "zero"}:
        raise ValueError("boundary_mode must be 'periodic' or 'zero'")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be >= 1")

    h_src, w_src = int(image.shape[0]), int(image.shape[1])
    if boundary_mode == "periodic":
        work = image
        y0 = 0
        x0 = 0
    else:
        pad = max(float(canvas_pad_factor), 1.0)
        canvas_h = max(int(math.ceil(h_src * pad)), h_src)
        canvas_w = max(int(math.ceil(w_src * pad)), w_src)
        work, y0, x0 = _center_embed_2d(image, out_h=canvas_h, out_w=canvas_w)

    h_work, w_work = int(work.shape[0]), int(work.shape[1])
    flat_x = x_src.reshape(-1).to(dtype=torch.float64) + float(x0)
    flat_y = y_src.reshape(-1).to(dtype=torch.float64) + float(y0)

    work_dtype = work.dtype
    if work_dtype == torch.float16:
        work_real = work.to(dtype=torch.float32)
    else:
        work_real = work
    spectrum = torch.fft.fft2(work_real)
    complex_dtype = spectrum.dtype
    freq_x = torch.fft.fftfreq(w_work, d=1.0, device=image.device).to(torch.float64)
    freq_y = torch.fft.fftfreq(h_work, d=1.0, device=image.device).to(torch.float64)

    out = torch.empty_like(flat_x, dtype=work_real.dtype)
    normalizer = float(h_work * w_work)
    for start in range(0, int(flat_x.numel()), int(chunk_size)):
        stop = min(start + int(chunk_size), int(flat_x.numel()))
        chunk_x = flat_x[start:stop]
        chunk_y = flat_y[start:stop]
        phase_x = torch.exp(
            2j * math.pi * chunk_x[:, None] * freq_x[None, :]
        ).to(dtype=complex_dtype)
        phase_y = torch.exp(
            2j * math.pi * chunk_y[:, None] * freq_y[None, :]
        ).to(dtype=complex_dtype)
        temp = phase_x @ spectrum.transpose(0, 1)
        values = (temp * phase_y).sum(dim=-1) / normalizer
        out[start:stop] = values.real.to(dtype=work_real.dtype)

    result = out.reshape_as(x_src).to(dtype=image.dtype)
    if boundary_mode == "zero":
        in_bounds = (
            (x_src >= 0.0)
            & (x_src <= float(w_src - 1))
            & (y_src >= 0.0)
            & (y_src <= float(h_src - 1))
        )
        result = torch.where(in_bounds, result, torch.zeros_like(result))
    return result
