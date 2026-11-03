"""Shared PSF helpers for image-plane and field rendering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import PSFOp
from ...device import _to


@dataclass
class ExplicitPSFOp(PSFOp):
    def __init__(
        self,
        *,
        name: str = "explicit_psf",
        pixel_response_included: bool | None = None,
        pixel_response_provenance: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            learned=False,
            frozen=True,
            pixel_response_included=pixel_response_included,
            pixel_response_provenance=pixel_response_provenance,
        )


@dataclass
class LearnedPSFOp(PSFOp):
    model: nn.Module | None = None

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        name: str = "learned_psf",
        frozen: bool = False,
        pixel_response_included: bool | None = None,
        pixel_response_provenance: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            learned=True,
            frozen=frozen,
            pixel_response_included=pixel_response_included,
            pixel_response_provenance=pixel_response_provenance,
        )
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
        if any(int(size) % 2 == 0 for size in kernel.shape):
            raise ValueError("kernel2d PSF dimensions must be odd")
        return int(kernel.shape[0]) // 2, int(kernel.shape[1]) // 2
    if mode == "learned_lowrank":
        # Learned chromatic kernels vary per cell; reserve twice the base
        # Gaussian support so the widest plausible learned kernel fits.
        radius = 2 * gaussian_support_radius(float(getattr(psf_spec, "sigma_pix", 0.0)))
        return radius, radius
    raise ValueError(f"unsupported PSF mode {mode!r}")


def gaussian_kernel1d(
    sigma_pix: float, *, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tensor:
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


def _normalized_kernel2d(reference: Tensor, kernel2d: Tensor) -> Tensor:
    if kernel2d.ndim != 2:
        raise ValueError("kernel2d PSF must be 2D")
    if kernel2d.numel() == 0:
        raise ValueError("kernel2d PSF must be non-empty")
    if any(int(size) % 2 == 0 for size in kernel2d.shape):
        raise ValueError("kernel2d PSF dimensions must be odd")
    kernel = _to(kernel2d, device=reference.device, dtype=reference.dtype)
    if not bool(torch.isfinite(kernel).all()):
        raise ValueError("kernel2d PSF must contain only finite values")
    kernel_sum = kernel.sum()
    tolerance = torch.finfo(kernel.dtype).eps ** 0.5 * kernel.abs().sum().clamp_min(1.0)
    if bool(kernel_sum.abs() <= tolerance):
        raise ValueError("kernel2d PSF sum must not be zero or near zero")
    return kernel / kernel_sum


def apply_kernel2d(image: Tensor, kernel2d: Tensor) -> Tensor:
    """Convolve with an odd image-shaped response, zero outside the image."""
    kernel = _normalized_kernel2d(image, kernel2d)
    ky, kx = int(kernel.shape[0]), int(kernel.shape[1])
    out = F.conv2d(
        image.unsqueeze(0).unsqueeze(0),
        kernel.flip(-2, -1).view(1, 1, ky, kx),
        padding=(ky // 2, kx // 2),
    )
    return out.squeeze(0).squeeze(0)


def apply_kernel2d_variance(variance: Tensor, kernel2d: Tensor) -> Tensor:
    """Propagate independent-pixel variance through :func:`apply_kernel2d`."""
    kernel = torch.square(_normalized_kernel2d(variance, kernel2d))
    ky, kx = int(kernel.shape[0]), int(kernel.shape[1])
    out = F.conv2d(
        variance.unsqueeze(0).unsqueeze(0),
        kernel.flip(-2, -1).view(1, 1, ky, kx),
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
        node_uvw = getattr(model, "node_uvw", None)
        extent_pix = getattr(model, "extent_pix", None)
        if isinstance(node_uvw, Tensor) and isinstance(extent_pix, Tensor):
            u_pix = _to(node_uvw * extent_pix, device=device, dtype=dtype)
        else:
            u_pix = torch.zeros((n_cells, 2), device=device, dtype=dtype)
    else:
        u_pix = _to(u_pix, device=device, dtype=dtype)
        if u_pix.shape != (n_cells, 2):
            raise ValueError(
                f"cell_u_pix must have shape ({n_cells}, 2); got {tuple(u_pix.shape)}"
            )
    params = _to(model(u_pix, lam_norm), device=device, dtype=dtype)
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
    return torch.minimum(
        torch.minimum(up, down), torch.ones((), device=device, dtype=dtype)
    )


def _build_blended_cells(
    kernels: Tensor, height: int, width: int, dtype: torch.dtype, device: torch.device
) -> list[tuple[int, int, int, int, Tensor, Tensor, int, int]]:
    """Per-cell write windows, separable tent weights, and kernels."""
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


def _learned_cells(
    psf_spec: object, height: int, width: int, dtype: torch.dtype, device: torch.device
) -> list[tuple[int, int, int, int, Tensor, Tensor, int, int]]:
    """Per-cell kernels, write windows, and separable tent weights."""
    kernels = _cell_kernels_from_spec(psf_spec, device=device, dtype=dtype)
    return _build_blended_cells(kernels, height, width, dtype, device)


def _require_frozen_learned_psf(psf_spec: object) -> None:
    if not bool(getattr(psf_spec, "frozen", True)):
        raise ValueError(
            "learned PSF scene rendering requires frozen=True; calibrate the PSF "
            "through its model-specific calibration path"
        )


def apply_learned_psf_spec(image: Tensor, psf_spec: object) -> Tensor:
    """Apply a learned low-rank chromatic PSF with blended control cells.

    Each control kernel is applied to the globally zero-padded image (same
    conv), and the per-cell results are blended by tent weights forming a
    normalized partition of unity -- no hard cell boundaries.

    Kernels depend only on the calibration-anchored (frozen) PSF parameters
    and the cell geometry -- never on the scene -- so they are built under
    ``no_grad`` and detached. Scene gradients still flow through every
    convolution to ``image``; only the frozen kernel construction stays out
    of the graph, capping render memory at one conv workspace per cell.
    """
    _require_frozen_learned_psf(psf_spec)
    flat = image.reshape(-1, int(image.shape[-2]), int(image.shape[-1]))
    height, width = int(flat.shape[-2]), int(flat.shape[-1])
    with torch.no_grad():
        cells = _learned_cells(psf_spec, height, width, image.dtype, image.device)
        cells = [
            (c[0], c[1], c[2], c[3], c[4], c[5].detach(), c[6], c[7]) for c in cells
        ]
    return apply_blended_cell_kernels(image, flat, cells)


def apply_blended_cell_kernels(
    image: Tensor,
    flat: Tensor,
    cells: list[tuple],
) -> Tensor:
    """Tent-weighted blend of per-cell unit-sum kernels over ``flat``.

    Shared by :func:`apply_learned_psf_spec` and
    ``LowRankChromaticPSF.filter_image`` so both render paths stay identical.
    """
    pad_y, pad_x = cells[0][6], cells[0][7]
    padded = F.pad(flat, (pad_x, pad_x, pad_y, pad_y))
    out = torch.zeros_like(flat)
    weight_sum = torch.zeros_like(flat)
    for wy0, wy1, wx0, wx1, weight, kernel, pad_y, pad_x in cells:
        conv = F.conv2d(padded.unsqueeze(1), kernel.view(1, 1, *kernel.shape)).squeeze(
            1
        )[:, wy0:wy1, wx0:wx1]
        out[:, wy0:wy1, wx0:wx1] += conv * weight
        weight_sum[:, wy0:wy1, wx0:wx1] += weight
    return (out / weight_sum.clamp_min(torch.finfo(out.dtype).tiny)).reshape(
        image.shape
    )


def apply_learned_psf_spec_adjoint(image: Tensor, psf_spec: object) -> Tensor:
    """Exact adjoint of :func:`apply_learned_psf_spec`.

    Forward contribution of one cell (global same-conv semantics):
    ``acc[p] = w(p) sum_d K[d] xp[p - pad + d]`` with ``xp[q] = x[q - pad]``.
    Hence ``grad_x[t] = sum_j g[j] K[pad + t - j]`` for
    ``g = y * w / S`` -- i.e. ``F.conv2d(pad(g, pad), K.flip(-1, -2))``
    with no further crop or shift. Runs fully under ``no_grad``: the adjoint
    is a pure linear map on the cotangent.
    """
    _require_frozen_learned_psf(psf_spec)
    flat = image.reshape(-1, int(image.shape[-2]), int(image.shape[-1]))
    height, width = int(flat.shape[-2]), int(flat.shape[-1])
    with torch.no_grad():
        cells = _learned_cells(psf_spec, height, width, image.dtype, image.device)
        pad_y, pad_x = cells[0][6], cells[0][7]

        weight_sum = torch.zeros_like(flat)
        for wy0, wy1, wx0, wx1, weight, _kernel, _py, _px in cells:
            weight_sum[:, wy0:wy1, wx0:wx1] += weight
        scaled = flat / weight_sum.clamp_min(torch.finfo(flat.dtype).tiny)

        grad = torch.zeros_like(flat)
        for wy0, wy1, wx0, wx1, weight, kernel, pad_y, pad_x in cells:
            g = torch.zeros_like(scaled)
            g[:, wy0:wy1, wx0:wx1] = scaled[:, wy0:wy1, wx0:wx1] * weight
            grad += F.conv2d(
                F.pad(g, (pad_x, pad_x, pad_y, pad_y)).unsqueeze(1),
                kernel.flip(-1, -2).view(1, 1, *kernel.shape),
            ).squeeze(1)
    return grad.reshape(image.shape)


def apply_learned_psf_variance_spec(variance: Tensor, psf_spec: object) -> Tensor:
    """Exact diagonal-noise propagation through the blended cell operator.

    With ``A[p,j] = sum_c w_c(p) K_c[j-p+pad] / S(p)``, independent pixel
    noise gives ``Var(Ay)[p] = (1/S(p)^2) sum_j (sum_c w_c(p) K_c[j-p+pad])^2
    V[j]``. Expanding the square separates into ordered cell pairs whose
    cross terms are correlations of ``V`` with the pointwise kernel products
    ``K_c * K_c'`` -- all of them, including the c = c' diagonal.
    """
    _require_frozen_learned_psf(psf_spec)
    flat = variance.reshape(-1, int(variance.shape[-2]), int(variance.shape[-1]))
    height, width = int(flat.shape[-2]), int(flat.shape[-1])
    with torch.no_grad():
        cells = _learned_cells(psf_spec, height, width, variance.dtype, variance.device)
        pad_y, pad_x = cells[0][6], cells[0][7]
        padded = F.pad(flat, (pad_x, pad_x, pad_y, pad_y)).unsqueeze(1)

        S = torch.zeros_like(flat)
        for wy0, wy1, wx0, wx1, weight, _kernel, _py, _px in cells:
            S[:, wy0:wy1, wx0:wx1] += weight

        out = torch.zeros_like(flat)
        for c1 in cells:
            for c2 in cells:
                kk = c1[5] * c2[5]
                coef = torch.zeros_like(flat)
                ry0, ry1 = max(c1[0], c2[0]), min(c1[1], c2[1])
                rx0, rx1 = max(c1[2], c2[2]), min(c1[3], c2[3])
                if ry0 >= ry1 or rx0 >= rx1:
                    continue
                coef[:, ry0:ry1, rx0:rx1] = (
                    c1[4][ry0 - c1[0] : ry1 - c1[0], rx0 - c1[2] : rx1 - c1[2]]
                    * c2[4][ry0 - c2[0] : ry1 - c2[0], rx0 - c2[2] : rx1 - c2[2]]
                )
                corr = F.conv2d(padded, kk.view(1, 1, *kk.shape)).squeeze(1)[
                    :, ry0:ry1, rx0:rx1
                ]
                out[:, ry0:ry1, rx0:rx1] += corr * coef[:, ry0:ry1, rx0:rx1]
        denom = S.square().clamp_min(torch.finfo(out.dtype).tiny)
        return (out / denom).reshape(variance.shape)


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
