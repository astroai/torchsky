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
    dln_lambda: float | Tensor,
    sigma_kms: Tensor | float,
    *,
    kernel_radius: int = 8,
    mean_kms: Tensor | float = 0.0,
    zero_mean: bool = False,
) -> Tensor:
    """Convolve ``flux`` with a unit-sum Gaussian in ln λ.

    Args:
        flux: ``(..., n)`` samples on a uniform ``ln λ`` grid.
        dln_lambda: constant grid spacing ``Δ ln λ``.
        sigma_kms: Gaussian width in km/s, broadcastable to ``flux[..., 1]``.
        kernel_radius: half-width of the discrete kernel in pixels.
        mean_kms: optional kernel mean in km/s (LOSVD streaming). The adjoint
            of a mean-``μ`` kernel is the mean-``-μ`` kernel.
        zero_mean: if True, rebuild the kernel so the discrete first moment
            is zero (truncated-Gaussian centering). Incompatible with a
            nonzero ``mean_kms``.
    """
    if kernel_radius < 0:
        raise ValueError("kernel_radius must be non-negative")
    flux_t = torch.as_tensor(flux)
    if flux_t.ndim < 1:
        raise ValueError("flux must have a spectral axis")
    dln = torch.as_tensor(dln_lambda, device=flux_t.device, dtype=flux_t.dtype)
    if bool((dln <= 0).any() if dln.numel() > 1 else float(dln) <= 0.0):
        raise ValueError("dln_lambda must be positive")

    leading = flux_t.shape[:-1]
    n_pix = int(flux_t.shape[-1])
    batch = int(flux_t.numel() // n_pix)
    flux_flat = flux_t.reshape(batch, n_pix)

    def _broadcast_velocity(value: Tensor | float) -> Tensor:
        tensor = torch.as_tensor(value, device=flux_t.device, dtype=flux_t.dtype)
        if tensor.ndim == 0:
            return tensor.expand(batch, 1)
        tensor = tensor.to(device=flux_t.device, dtype=flux_t.dtype)
        while tensor.ndim < len(leading):
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == len(leading):
            tensor = tensor.unsqueeze(-1)
        if tensor.shape[: len(leading)] != leading:
            tensor = tensor.expand(*leading, *tensor.shape[len(leading) :])
        return tensor.reshape(batch, -1)[:, :1]

    sigma = _broadcast_velocity(sigma_kms).clamp_min(0.0)
    mean = _broadcast_velocity(mean_kms)
    if zero_mean and bool((mean.abs() > 0).any()):
        raise ValueError("zero_mean=True requires mean_kms=0")

    sigma_px = sigma / (C_KM_S * dln)
    mean_px = mean / (C_KM_S * dln)
    radius = int(kernel_radius)
    offsets = torch.arange(
        -radius, radius + 1, device=flux_t.device, dtype=flux_t.dtype
    )
    kernel = _unit_gaussian_kernel(offsets, sigma_px, mean_px, zero_mean=zero_mean)
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
    """``(batch, 2R+1)`` kernels. Tiny ``σ`` is a discrete delta at ``round(μ)``."""
    tiny = sigma_px < 1.0e-6
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
