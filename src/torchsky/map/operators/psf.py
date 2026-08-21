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
    if mode == "learned_lowrank":
        # Learned chromatic kernels vary per cell; reserve twice the base
        # Gaussian support so the widest plausible learned kernel fits.
        radius = 2 * gaussian_support_radius(
            float(getattr(psf_spec, "sigma_pix", 0.0))
        )
        return radius, radius
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


def effective_gaussian_sigma_pix(psf_spec: object | None) -> float:
    """Symmetric sigma standing in for the PSF (base width of a learned model).

    ponytail: callers that require a symmetric PSF (foam initialization,
    exact-observation refinement) collapse a learned chromatic PSF to its
    base-cell Gaussian; upgrade path is per-cell symmetric rendering.
    """
    if psf_spec is None:
        return 0.0
    mode = getattr(psf_spec, "mode", "none")
    if mode in ("learned_lowrank", "gaussian_sigma_pix"):
        return float(getattr(psf_spec, "sigma_pix", 0.0))
    return 0.0


def _cell_kernels_from_spec(
    psf_spec: object, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Unit-sum, zero-centroid kernel for every control cell -> [C, ky, kx]."""
    from .learned_kernel import gaussian_kernels_from_params

    model = getattr(psf_spec, "model", None)
    if model is None:
        raise ValueError("learned_lowrank PSF mode requires psf_spec.model")
    u_pix = getattr(psf_spec, "cell_u_pix", None)
    lam_norm = float(getattr(psf_spec, "cell_lam_norm", 0.5))
    n_cells = int(getattr(model, "n_cells", 1))
    if u_pix is None:
        u_pix = torch.zeros((n_cells, 2), device=device, dtype=dtype)
    else:
        u_pix = u_pix.to(device=device, dtype=dtype)
        if u_pix.shape != (n_cells, 2):
            raise ValueError(
                f"cell_u_pix must have shape ({n_cells}, 2); got {tuple(u_pix.shape)}"
            )
    params = model(u_pix, lam_norm).to(device=device, dtype=dtype)
    sigma_max = torch.max(params[:, 0], params[:, 1]).max().item()
    radius = max(1, int(torch.ceil(torch.tensor(3.0 * sigma_max)).item()))
    size = min(2 * radius + 1, 101)
    return gaussian_kernels_from_params(params, size=size)


def _cell_geometry(
    n_cells: int, height: int, width: int
) -> tuple[int, int, int, int, int]:
    grid_side = max(1, int(n_cells**0.5))
    while grid_side * grid_side < n_cells:
        grid_side += 1
    cell_h = -(-height // grid_side)
    cell_w = -(-width // grid_side)
    return grid_side, cell_h, cell_w, height, width


def _tent_weight(
    length: int,
    core_start: int,
    core_len: int,
    ramp: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Tent profile over ``length`` samples: 1 on the core, linear ramp to 0."""
    axis = torch.arange(length, device=device, dtype=dtype)
    core_end = core_start + core_len
    up = (axis - (core_start - ramp)).clamp_min(0.0) / max(ramp, 1)
    down = ((core_end + ramp) - axis).clamp_min(0.0) / max(ramp, 1)
    return torch.minimum(torch.minimum(up, down), torch.ones((), device=device, dtype=dtype))


def _learned_cells(
    psf_spec: object, height: int, width: int, dtype: torch.dtype, device: torch.device
) -> list[tuple[int, int, Tensor, Tensor, Tensor, Tensor, int, int]]:
    """Per-cell kernels, write windows, and separable tent weights."""
    kernels = _cell_kernels_from_spec(psf_spec, device=device, dtype=dtype)
    n_cells, ky, kx = (int(v) for v in kernels.shape)
    grid_side, cell_h, cell_w, _, _ = _cell_geometry(n_cells, height, width)
    pad_y, pad_x = ky // 2, kx // 2
    cells = []
    idx = 0
    for iy in range(grid_side):
        y0 = iy * cell_h
        y1 = min(height, y0 + cell_h)
        if y0 >= y1:
            continue
        for ix in range(grid_side):
            x0 = ix * cell_w
            x1 = min(width, x0 + cell_w)
            if x0 >= x1 or idx >= n_cells:
                continue
            wy0, wy1 = max(0, y0 - pad_y), min(height, y1 + pad_y)
            wx0, wx1 = max(0, x0 - pad_x), min(width, x1 + pad_x)
            wy = _tent_weight(
                wy1 - wy0, y0 - wy0, y1 - y0, pad_y, device=device, dtype=dtype
            )
            wx = _tent_weight(
                wx1 - wx0, x0 - wx0, x1 - x0, pad_x, device=device, dtype=dtype
            )
            cells.append(
                (
                    wy0,
                    wy1,
                    wx0,
                    wx1,
                    wy.view(-1, 1) * wx.view(1, -1),
                    kernels[idx],
                    pad_y,
                    pad_x,
                )
            )
            idx += 1
    return cells


def apply_learned_psf_spec(image: Tensor, psf_spec: object) -> Tensor:
    """Apply a learned low-rank chromatic PSF with blended control cells.

    Each control kernel is applied to the globally zero-padded image (same
    conv), and the per-cell results are blended by tent weights that form a
    normalized partition of unity over the control grid -- no hard cell
    boundaries. Because every cell reads one shared global convolution
    geometry, :func:`apply_learned_psf_spec_adjoint` is exact by
    construction.
    """
    flat = image.reshape(-1, int(image.shape[-2]), int(image.shape[-1]))
    height, width = int(flat.shape[-2]), int(flat.shape[-1])
    cells = _learned_cells(psf_spec, height, width, image.dtype, image.device)
    pad_y, pad_x = cells[0][6], cells[0][7]
    padded = F.pad(flat, (pad_x, pad_x, pad_y, pad_y))
    out = torch.zeros_like(flat)
    weight_sum = torch.zeros_like(flat)
    for wy0, wy1, wx0, wx1, weight, kernel, pad_y, pad_x in cells:
        conv = F.conv2d(padded, kernel.view(1, 1, *kernel.shape))[
            :, wy0:wy1, wx0:wx1
        ]
        out[:, wy0:wy1, wx0:wx1] += conv * weight
        weight_sum[:, wy0:wy1, wx0:wx1] += weight
    return (out / weight_sum.clamp_min(torch.finfo(out.dtype).tiny)).reshape(
        image.shape
    )


def apply_learned_psf_spec_adjoint(image: Tensor, psf_spec: object) -> Tensor:
    """Exact adjoint of :func:`apply_learned_psf_spec` via reverse-mode AD.

    The blended multi-kernel operator (per-cell kernels, tent weights,
    per-pixel normalization) has too many moving parts for a hand-written
    transpose to stay trustworthy; reverse-mode AD through the forward
    itself is exact by construction. Costs one forward graph per call --
    negligible for the initialization solvers that consume it.
    """
    x = torch.zeros_like(image, requires_grad=True)
    with torch.enable_grad():
        out = apply_psf_spec(x, psf_spec)
    (grad,) = torch.autograd.grad(out, x, grad_outputs=image)
    return grad


def apply_learned_psf_variance_spec(variance: Tensor, psf_spec: object) -> Tensor:
    """Propagate diagonal pixel variance through the blended cell operator.

    With ``A = sum_c (w_c / S) B_c`` and independent pixels, per-pixel
    variance combines as ``Var(Ay) = sum_c (w_c / S)^2 (K_c^2 conv V)``.
    """
    flat = variance.reshape(-1, int(variance.shape[-2]), int(variance.shape[-1]))
    height, width = int(flat.shape[-2]), int(flat.shape[-1])
    cells = _learned_cells(psf_spec, height, width, variance.dtype, variance.device)
    pad_y, pad_x = cells[0][6], cells[0][7]
    padded = F.pad(flat, (pad_x, pad_x, pad_y, pad_y))
    out = torch.zeros_like(flat)
    weight_sq_sum = torch.zeros_like(flat)
    for wy0, wy1, wx0, wx1, weight, kernel, pad_y, pad_x in cells:
        conv_var = F.conv2d(
            padded, (kernel.square()).view(1, 1, *kernel.shape)
        )[
            :, wy0:wy1, wx0:wx1
        ]
        out[:, wy0:wy1, wx0:wx1] += conv_var * weight.square()
        weight_sq_sum[:, wy0:wy1, wx0:wx1] += weight.square()
    return (
        (out / weight_sq_sum.clamp_min(torch.finfo(out.dtype).tiny))
        .reshape(variance.shape)
    )


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
    if mode == "learned_lowrank":
        return apply_learned_psf_spec(image, psf_spec)
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
    if mode == "learned_lowrank":
        return apply_learned_psf_variance_spec(variance, psf_spec)
    raise ValueError(f"unsupported PSF mode {mode!r}")
