"""Shared PSF helpers for image-plane and field rendering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import PSFOp


@dataclass
class ExplicitPSFOp(PSFOp):
    def __init__(self, *, name: str = "explicit_psf") -> None:
        super().__init__(name=name, learned=False)


@dataclass
class LearnedPSFOp(PSFOp):
    model: nn.Module | None = None

    def __init__(
        self, model: nn.Module | None = None, *, name: str = "learned_psf"
    ) -> None:
        super().__init__(name=name, learned=True)
        self.model = model


def gaussian_support_radius(sigma_pix: float) -> int:
    """Return the exact finite radius used by :func:`gaussian_blur`."""
    return 0 if sigma_pix <= 0.0 else max(1, int(round(3.0 * float(sigma_pix))))


def psf_support_radius(psf_spec: object | None) -> tuple[int, int]:
    """Return the vertical and horizontal pixel halo required by a PSF."""
    if psf_spec is None:
        return 0, 0
    mode = getattr(psf_spec, "mode", "none")
    if mode == "none":
        return 0, 0
    if mode == "gaussian_sigma_pix":
        radius = gaussian_support_radius(float(getattr(psf_spec, "sigma_pix", 0.0)))
        return radius, radius
    if mode == "kernel2d":
        kernel = getattr(psf_spec, "kernel2d", None)
        if kernel is None or kernel.ndim != 2:
            raise ValueError("kernel2d PSF mode requires a 2D kernel")
        return int(kernel.shape[0]) // 2, int(kernel.shape[1]) // 2
    raise ValueError(f"unsupported PSF mode {mode!r}")


def gaussian_kernel1d(sigma_pix: float, *, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    if sigma_pix <= 0.0:
        return torch.ones((1,), dtype=dtype, device=device)
    radius = gaussian_support_radius(sigma_pix)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * torch.square(x / float(sigma_pix)))
    return kernel / torch.clamp(kernel.sum(), min=1e-8)


def gaussian_blur(image: Tensor, sigma_pix: float) -> Tensor:
    if sigma_pix <= 0.0:
        return image
    kernel = gaussian_kernel1d(sigma_pix, device=image.device, dtype=image.dtype)
    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)
    out = image.unsqueeze(0).unsqueeze(0)
    out = F.conv2d(out, kernel_x, padding=(0, kernel.numel() // 2))
    out = F.conv2d(out, kernel_y, padding=(kernel.numel() // 2, 0))
    return out.squeeze(0).squeeze(0)


def gaussian_blur_variance(variance: Tensor, sigma_pix: float) -> Tensor:
    """Propagate independent-pixel variance through a Gaussian PSF."""
    if sigma_pix <= 0.0:
        return variance
    kernel = torch.square(
        gaussian_kernel1d(sigma_pix, device=variance.device, dtype=variance.dtype)
    )
    out = variance.unsqueeze(0).unsqueeze(0)
    out = F.conv2d(
        out,
        kernel.view(1, 1, 1, -1),
        padding=(0, kernel.numel() // 2),
    )
    out = F.conv2d(
        out,
        kernel.view(1, 1, -1, 1),
        padding=(kernel.numel() // 2, 0),
    )
    return out.squeeze(0).squeeze(0)


def apply_kernel2d(image: Tensor, kernel2d: Tensor) -> Tensor:
    if kernel2d.ndim != 2:
        raise ValueError("kernel2d PSF must be 2D")
    kernel = kernel2d.to(device=image.device, dtype=image.dtype)
    kernel = kernel / torch.clamp(kernel.sum(), min=1e-8)
    ky, kx = int(kernel.shape[0]), int(kernel.shape[1])
    out = F.conv2d(
        image.unsqueeze(0).unsqueeze(0),
        kernel.view(1, 1, ky, kx),
        padding=(ky // 2, kx // 2),
    )
    return out.squeeze(0).squeeze(0)


def apply_kernel2d_variance(variance: Tensor, kernel2d: Tensor) -> Tensor:
    """Propagate independent-pixel variance through a normalized 2-D PSF."""
    if kernel2d.ndim != 2:
        raise ValueError("kernel2d PSF must be 2D")
    kernel = kernel2d.to(device=variance.device, dtype=variance.dtype)
    kernel = kernel / torch.clamp(kernel.sum(), min=1e-8)
    kernel = torch.square(kernel)
    ky, kx = int(kernel.shape[0]), int(kernel.shape[1])
    out = F.conv2d(
        variance.unsqueeze(0).unsqueeze(0),
        kernel.view(1, 1, ky, kx),
        padding=(ky // 2, kx // 2),
    )
    return out.squeeze(0).squeeze(0)


def apply_psf_spec(image: Tensor, psf_spec: object | None) -> Tensor:
    if psf_spec is None:
        return image
    mode = getattr(psf_spec, "mode", "none")
    if mode == "none":
        return image
    if mode == "gaussian_sigma_pix":
        return gaussian_blur(image, float(getattr(psf_spec, "sigma_pix", 0.0)))
    if mode == "kernel2d":
        kernel = getattr(psf_spec, "kernel2d", None)
        if kernel is None:
            raise ValueError("kernel2d PSF mode requires kernel2d")
        return apply_kernel2d(image, kernel)
    raise ValueError(f"unsupported PSF mode {mode!r}")


def apply_psf_variance_spec(variance: Tensor, psf_spec: object | None) -> Tensor:
    """Propagate independent-pixel variance through a configured PSF."""
    if psf_spec is None:
        return variance
    mode = getattr(psf_spec, "mode", "none")
    if mode == "none":
        return variance
    if mode == "gaussian_sigma_pix":
        return gaussian_blur_variance(
            variance, float(getattr(psf_spec, "sigma_pix", 0.0))
        )
    if mode == "kernel2d":
        kernel = getattr(psf_spec, "kernel2d", None)
        if kernel is None:
            raise ValueError("kernel2d PSF mode requires kernel2d")
        return apply_kernel2d_variance(variance, kernel)
    raise ValueError(f"unsupported PSF mode {mode!r}")
