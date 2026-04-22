"""Mixed FFT + periodic-bilinear resample for ``2×2`` pixel affines (LDU pivot).

For ``A = L @ D @ U`` (unit lower ``L``, diagonal ``D``, unit upper ``U``),

``f(L D U p) = (M ∘ R_L)(f)(p)`` with ``(R_L f)(q)=f(L q)`` and ``(M g)(p)=g(D U p)``.

The **lower shear** ``R_L`` uses **column-wise 1D FFTs**. The map ``p ↦ D U p`` is an
upper-triangular linear map; it is applied with **periodic bilinear** sampling
(:func:`torchsky.map.periodic_affine.resample_linear_affine_periodic_bilinear_2d`) so
the implementation stays **torch-native** without a full 2D NUFFT. Translation ``b`` is
applied with :func:`torchsky.map.fft_resample.resample_translation_fft2d`.

Requires ``|a00|`` large enough for a stable ``LU`` pivot; otherwise use
``periodic_linear_affine``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .fft_resample import resample_translation_fft2d
from .periodic_affine import resample_linear_affine_periodic_bilinear_2d


def _per_col_vertical_shift_fft(image: Tensor, shift_y_per_col: Tensor) -> Tensor:
    """``out[jx, iy] = in[jx, iy + shift_y_per_col[jx]]`` (periodic along rows)."""
    if image.ndim != 2:
        raise ValueError("expected [H, W]")
    h, w = int(image.shape[0]), int(image.shape[1])
    if shift_y_per_col.shape != (w,):
        raise ValueError("shift_y_per_col must be [W]")
    dtype = image.dtype
    device = image.device
    comp = torch.complex64 if dtype == torch.float32 else torch.complex128
    x = torch.fft.fft(image.to(comp), n=h, dim=0)
    k = torch.fft.fftfreq(h, d=1.0, device=device, dtype=torch.float64).view(-1, 1)
    sh = shift_y_per_col.to(device=device, dtype=torch.float64).view(1, -1)
    ph = torch.exp(-2j * math.pi * k * sh).to(comp)
    return torch.fft.ifft(x * ph, n=h, dim=0).real.to(dtype)


def _invert_2x2(
    a00: float, a01: float, a10: float, a11: float
) -> tuple[float, float, float, float]:
    det = a00 * a11 - a01 * a10
    if abs(det) < 1e-18:
        raise ValueError("singular 2×2 map")
    return (
        a11 / det,
        -a01 / det,
        -a10 / det,
        a00 / det,
    )


def resample_fft_ldu_affine_2d(
    image: Tensor,
    *,
    a00: float,
    a01: float,
    a10: float,
    a11: float,
    bx: float,
    by: float,
) -> Tensor:
    """``out(p) ≈ in(A @ p + b)`` for ``A = L D U`` (standard ``LU`` pivot on ``a00``)."""
    if image.ndim != 2:
        raise ValueError("expected [H, W]")
    h, w = int(image.shape[0]), int(image.shape[1])
    if h < 2 or w < 2:
        raise ValueError("fft_ldu_affine expects H,W >= 2")

    pivot_tol = 1e-5 * max(float(w), float(h), 1.0)
    if abs(a00) < pivot_tol:
        raise ValueError(
            "fft_linear_affine LDU pivot too small for stable shear chain; "
            "use periodic_linear_affine"
        )

    gamma = a10 / a00
    beta = a01 / a00
    d0 = a00
    d2 = a11 - a10 * a01 / a00
    if not math.isfinite(d2) or abs(d2) < 1e-18:
        raise ValueError("degenerate LDU factor; use periodic_linear_affine")

    device = image.device
    shift_col = torch.arange(w, device=device, dtype=torch.float64) * float(gamma)
    t1 = _per_col_vertical_shift_fft(image, shift_col)
    # M: q ↦ D U q with U = [[1, β],[0,1]], D = diag(d0, d2)  →  upper-triangular Q.
    t2 = resample_linear_affine_periodic_bilinear_2d(
        t1,
        a00=float(d0),
        a01=float(d0) * float(beta),
        a10=0.0,
        a11=float(d2),
        bx=0.0,
        by=0.0,
    )

    inv00, inv01, inv10, inv11 = _invert_2x2(a00, a01, a10, a11)
    dx = inv00 * bx + inv01 * by
    dy = inv10 * bx + inv11 * by
    return resample_translation_fft2d(
        t2,
        shift_y_pix=float(-dy),
        shift_x_pix=float(-dx),
    )
