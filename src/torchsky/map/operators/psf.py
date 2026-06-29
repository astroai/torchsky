"""Shared PSF helpers for image-plane and field rendering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import PSFOp


@dataclass
class ExplicitPSFOp(PSFOp):
    def __init__(self, *, name: str = "explicit_psf") -> None:
        super().__init__(name=name, learned=False)


@dataclass
class LearnedPSFOp(PSFOp):
    def __init__(self, *, name: str = "learned_psf") -> None:
        super().__init__(name=name, learned=True)


def gaussian_kernel1d(sigma_pix: float, *, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    if sigma_pix <= 0.0:
        return torch.ones((1,), dtype=dtype, device=device)
    radius = max(1, int(round(3.0 * float(sigma_pix))))
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
