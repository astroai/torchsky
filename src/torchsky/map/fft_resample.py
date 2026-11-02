"""FFT-based resampling helpers (tangent-plane building blocks).

``resample_translation_fft2d`` is used by the ``fft_shift`` resampling kernel in
``coadd_image_plane`` when source/target WCS differ by a pure pixel translation
on a common grid. ``resample_diagonal_affine_fft2d`` supports a **separable**
diagonal affine map in pixel space (scale + shift per axis) with the same
**periodic** boundary model.

The separable affine path upsamples each axis via zero-padded real FFTs, then
linearly interpolates on a periodic 1D grid. It preserves field amplitudes and
does not apply a Jacobian factor: signal and weight fields use the same linear
operator. Propagating independent-sample variance instead requires squared
interpolation coefficients and is not represented by these helpers.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def resample_translation_fft2d(
    image: Tensor,
    *,
    shift_y_pix: float,
    shift_x_pix: float,
) -> Tensor:
    """Apply a subpixel shift with periodic boundaries (2D FFT phase multiply).

    Args:
        image: Real-valued ``[H, W]`` tensor.
        shift_y_pix: Forward shift in **pixel** units along axis 0 (rows).
        shift_x_pix: Forward shift in **pixel** units along axis 1 (columns).

    Returns:
        Real ``[H, W]`` tensor of the same dtype and device as ``image``.
    """
    if image.ndim != 2:
        raise ValueError("resample_translation_fft2d expects a 2D image [H, W]")
    h, w = int(image.shape[0]), int(image.shape[1])
    if h < 1 or w < 1:
        raise ValueError("image must be non-empty")
    if image.dtype not in {torch.float32, torch.float64}:
        raise TypeError("image must have dtype float32 or float64")
    if not math.isfinite(float(shift_y_pix)) or not math.isfinite(float(shift_x_pix)):
        raise ValueError("shifts must be finite")

    shifted_x = _shift_lastdim_periodic_fft(image, float(shift_x_pix))
    shifted_y = _shift_lastdim_periodic_fft(
        shifted_x.transpose(0, 1), float(shift_y_pix)
    )
    return shifted_y.transpose(0, 1)


def _shift_lastdim_periodic_fft(x: Tensor, shift: float) -> Tensor:
    n = int(x.shape[-1])
    spectrum = torch.fft.rfft(x, dim=-1)
    freq = torch.arange(spectrum.shape[-1], device=x.device, dtype=x.dtype) / float(n)
    phase = torch.exp((-2j * math.pi * shift) * freq).to(dtype=spectrum.dtype)
    if n % 2 == 0:
        phase[-1] = math.cos(math.pi * shift)
    return torch.fft.irfft(spectrum * phase, n=n, dim=-1)


def _expand_index_along_last_dim(idx_1d: Tensor, lead_shape: tuple[int, ...]) -> Tensor:
    """Broadcast ``idx_1d`` of shape ``[N]`` to ``[*lead_shape, N]`` for ``gather``."""
    if idx_1d.ndim != 1:
        raise ValueError("idx_1d must be 1D")
    view = (1,) * len(lead_shape) + (idx_1d.shape[0],)
    return idx_1d.view(*view).expand(*lead_shape, idx_1d.shape[0])


def resample_lastdim_periodic_affine_fft(
    x: Tensor,
    *,
    scale: float,
    shift: float,
    upsample: int = 64,
) -> Tensor:
    """Periodic band-limited resample along the **last** dimension (real signal).

    For last dimension length ``N``, output has the same shape as ``x``. Output
    sample ``k`` evaluates the trigonometric interpolant of the period-``N``
    embedding of ``x`` at continuous source coordinate
    ``remainder(k * scale + shift, N)``.

    Args:
        x: Real tensor with last dimension length ``N >= 1``.
        scale: Must be finite and ``|scale| >= 1e-9``.
        shift: Additive pixel offset after scaling (same units as the grid index).
        upsample: FFT upsampling factor along this axis (larger is more accurate).
    """
    if x.ndim < 1:
        raise ValueError("expected tensor with at least one dimension")
    n = int(x.shape[-1])
    if n < 1:
        raise ValueError("last dimension must be non-empty")
    if x.dtype not in {torch.float32, torch.float64}:
        raise TypeError("x must have dtype float32 or float64")
    sc = float(scale)
    sh = float(shift)
    if not math.isfinite(sc) or abs(sc) < 1e-9:
        raise ValueError("scale must be finite and |scale| >= 1e-9")
    if not math.isfinite(sh):
        raise ValueError("shift must be finite")
    u = int(upsample)
    if u < 4:
        raise ValueError("upsample must be >= 4")
    nu = n * u
    dtype = x.dtype
    device = x.device
    spec = torch.fft.rfft(x, n=n, dim=-1)
    nfreq = nu // 2 + 1
    if spec.shape[-1] < nfreq:
        spec_p = F.pad(spec, (0, nfreq - spec.shape[-1]))
    else:
        spec_p = spec[..., :nfreq]
    if n > 1 and n % 2 == 0:
        spec_p[..., n // 2] *= 0.5
    x_up = torch.fft.irfft(spec_p, n=nu, dim=-1) * float(u)
    k = torch.arange(n, device=device, dtype=dtype)
    t = torch.remainder(k * sc + sh, float(n))
    float_idx = t * float(u)
    idx0 = torch.floor(float_idx).to(torch.int64)
    w = (float_idx - idx0.to(dtype)).clamp(0.0, 1.0)
    idx0m = torch.remainder(idx0, nu).to(torch.long)
    idx1m = torch.remainder(idx0m + 1, nu).to(torch.long)
    lead = tuple(x.shape[:-1])
    idx0e = _expand_index_along_last_dim(idx0m, lead)
    idx1e = _expand_index_along_last_dim(idx1m, lead)
    v0 = torch.gather(x_up, -1, idx0e)
    v1 = torch.gather(x_up, -1, idx1e)
    we = _expand_index_along_last_dim(w, lead)
    out = (1.0 - we) * v0 + we * v1
    return out


def resample_diagonal_affine_fft2d(
    image: Tensor,
    *,
    scale_x_pix: float,
    shift_x_pix: float,
    scale_y_pix: float,
    shift_y_pix: float,
    upsample: int = 64,
) -> Tensor:
    """Separable periodic affine resampling on a 2D grid (columns then rows).

    With target pixel indices ``(iy, jx)`` (row, column), this samples the
    source image at periodic coordinates ``(scale_y * iy + shift_y,
    scale_x * jx + shift_x)`` by applying
    :func:`resample_lastdim_periodic_affine_fft` along axis 1 then axis 0.

    When both scales are unity, this matches :func:`resample_translation_fft2d`
    up to floating-point noise (uses the translation fast path in callers).
    """
    if image.ndim != 2:
        raise ValueError("resample_diagonal_affine_fft2d expects a 2D [H, W] tensor")
    sx = float(scale_x_pix)
    sy = float(scale_y_pix)
    if sx == 1.0 and sy == 1.0:
        return resample_translation_fft2d(
            image,
            shift_y_pix=-float(shift_y_pix),
            shift_x_pix=-float(shift_x_pix),
        )
    tmp = resample_lastdim_periodic_affine_fft(
        image,
        scale=sx,
        shift=shift_x_pix,
        upsample=upsample,
    )
    tmp_t = tmp.transpose(0, 1)
    out_t = resample_lastdim_periodic_affine_fft(
        tmp_t,
        scale=sy,
        shift=shift_y_pix,
        upsample=upsample,
    )
    return out_t.transpose(0, 1)
