"""Unit-integral Gaussian convolution on a uniformly spaced ln λ grid.

Used for LOSVD broadening and published resolving-power LSFs. The kernel is
normalized to sum to one. ``sigma → 0`` is the identity on the sampled grid
(an honest sub-pixel statement, not a continuous Dirac identity).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .redshift import C_KM_S

# Gaussian FWHM / σ. Formulation uses 2.355.
_FWHM_OVER_SIGMA = 2.355


def sigma_R_kms(resolving_power: Tensor | float) -> Tensor:
    """Velocity width of a Gaussian LSF for published resolving power ``R``.

    ``σ = c / (2.355 R)`` so the FWHM is ``c / R``.
    """
    r = torch.as_tensor(resolving_power)
    return C_KM_S / (_FWHM_OVER_SIGMA * r.clamp_min(1.0e-12))


def convolve_log_lambda(
    flux: Tensor,
    dln_lambda: float | Tensor | None = None,
    sigma_kms: Tensor | float | None = None,
    *,
    sigma_px: Tensor | float | None = None,
    kernel_radius: int = 8,
    mean_kms: Tensor | float = 0.0,
    zero_mean: bool = False,
) -> Tensor:
    """Convolve ``flux`` with a unit-sum Gaussian in ln λ.

    Args:
        flux: ``(..., n)`` samples on a uniform ``ln λ`` grid.
        dln_lambda: constant grid spacing ``Δ ln λ`` (required when ``sigma_kms``
            or nonzero ``mean_kms`` is used).
        sigma_kms: Gaussian width in km/s, broadcastable to ``flux[..., 1]`` or
            per-pixel ``flux[..., n]``. Mutually exclusive with ``sigma_px``.
        sigma_px: Gaussian width in grid pixels, broadcastable to ``flux[..., 1]``
            or per-pixel ``flux[..., n]``. Mutually exclusive with ``sigma_kms``.
        kernel_radius: half-width of the discrete kernel in pixels.
        mean_kms: optional kernel mean in km/s (LOSVD streaming), broadcastable
            to ``flux[..., 1]`` or ``flux[..., n]``. The adjoint of a mean-``μ``
            kernel is the mean-``-μ`` kernel.
        zero_mean: if True, rebuild the kernel so the discrete first moment
            is zero (truncated-Gaussian centering). Incompatible with a
            nonzero ``mean_kms``.
    """
    if kernel_radius < 0:
        raise ValueError("kernel_radius must be non-negative")
    if (sigma_kms is None) == (sigma_px is None):
        raise ValueError("Specify exactly one of sigma_kms or sigma_px")
    flux_t = torch.as_tensor(flux)
    if flux_t.ndim < 1:
        raise ValueError("flux must have a spectral axis")

    leading = flux_t.shape[:-1]
    n_pix = int(flux_t.shape[-1])
    batch = int(flux_t.numel() // n_pix)
    flux_flat = flux_t.reshape(batch, n_pix)

    def _align(value: Tensor | float) -> Tensor:
        """Broadcast a width/mean parameter to ``(batch, 1)`` or ``(batch, n)``."""
        tensor = torch.as_tensor(value, device=flux_t.device, dtype=flux_t.dtype)
        if tensor.ndim == 0:
            return tensor.reshape(1, 1).expand(batch, 1)
        # A parameter shaped like the batch axes is one value per spectrum.
        if len(leading) >= 1 and tensor.ndim == len(leading):
            try:
                return torch.broadcast_to(tensor, leading).reshape(batch, 1)
            except RuntimeError:
                pass
        if tensor.shape[-1] == n_pix:
            try:
                sized = torch.broadcast_to(tensor, leading + (n_pix,))
            except RuntimeError as exc:
                raise ValueError(
                    "per-pixel parameters must broadcast to the flux batch shape"
                ) from exc
            return sized.reshape(batch, n_pix)
        if tensor.shape[-1] == 1:
            try:
                sized = torch.broadcast_to(tensor, leading + (1,))
            except RuntimeError as exc:
                raise ValueError(
                    "per-spectrum parameters must broadcast to the flux batch shape"
                ) from exc
            return sized.reshape(batch, 1)
        raise ValueError(
            "sigma and mean parameters must be scalars, (..., 1), or (..., n_pix)"
        )

    mean = _align(mean_kms)
    if zero_mean and bool((mean.abs() > 0).any()):
        raise ValueError("zero_mean=True requires mean_kms=0")

    dln: Tensor | None = None
    if dln_lambda is not None:
        dln = torch.as_tensor(dln_lambda, device=flux_t.device, dtype=flux_t.dtype)
        if not bool((dln > 0).all()):
            raise ValueError("dln_lambda must be positive")
    need_dln = sigma_kms is not None or bool((mean.abs() > 0).any())
    if need_dln:
        if dln is None:
            raise ValueError(
                "dln_lambda is required when sigma_kms or nonzero mean_kms is used"
            )
        mean_px = mean / (C_KM_S * dln)
    else:
        mean_px = torch.zeros_like(mean)

    if sigma_px is not None:
        sigma_px_t = _align(sigma_px).clamp_min(0.0)
    else:
        assert sigma_kms is not None and dln is not None
        sigma_px_t = _align(sigma_kms).clamp_min(0.0) / (C_KM_S * dln)

    radius = int(kernel_radius)
    offsets = torch.arange(
        -radius, radius + 1, device=flux_t.device, dtype=flux_t.dtype
    )
    per_pixel = (
        sigma_px_t.shape[1] == n_pix or mean_px.shape[1] == n_pix
    ) and n_pix > 1
    if per_pixel:
        sigma_px_grid = sigma_px_t.expand(batch, n_pix).unsqueeze(-1)
        mean_px_grid = mean_px.expand(batch, n_pix).unsqueeze(-1)
        kernel = _unit_gaussian_kernel(
            offsets, sigma_px_grid, mean_px_grid, zero_mean=zero_mean
        )
        padded = F.pad(flux_flat, (radius, radius))
        windows = padded.unfold(-1, 2 * radius + 1, 1)
        out = (windows * kernel).sum(dim=-1).reshape_as(flux_t)
        return out

    kernel = _unit_gaussian_kernel(offsets, sigma_px_t, mean_px, zero_mean=zero_mean)
    padded = F.pad(flux_flat.unsqueeze(0), (radius, radius))
    weight = kernel.unsqueeze(1)
    out = F.conv1d(padded, weight, groups=batch).reshape_as(flux_t)
    return out


def _unit_gaussian_kernel(
    offsets: Tensor,
    sigma_px: Tensor,
    mean_px: Tensor,
    *,
    zero_mean: bool,
) -> Tensor:
    """``(..., 2R+1)`` kernels. Tiny ``σ`` is a discrete delta at ``round(μ)``."""
    tiny = sigma_px <= 1.0e-6
    safe_sigma = sigma_px.clamp_min(1.0e-6)
    z = (offsets - mean_px) / safe_sigma
    kernel = torch.exp(-0.5 * z * z)
    kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    if zero_mean:
        moment = (offsets * kernel).sum(dim=-1, keepdim=True)
        z = (offsets - (mean_px - moment)) / safe_sigma
        kernel = torch.exp(-0.5 * z * z)
        kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    nearest = torch.argmin((offsets - mean_px).abs(), dim=-1, keepdim=True)
    delta = torch.zeros_like(kernel)
    delta.scatter_(-1, nearest, 1.0)
    return torch.where(tiny, delta, kernel)


__all__ = [
    "C_KM_S",
    "convolve_log_lambda",
    "sigma_R_kms",
]
