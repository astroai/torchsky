"""Image-plane resampling kernels and WCS pixel-map inference helpers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ...wcs import WCS
from .config import MapResamplingKernel, MapValueSemantics


def sinc(x: Tensor) -> Tensor:
    pix = torch.pi * x
    return torch.where(
        torch.abs(x) < 1e-6,
        torch.ones_like(x),
        torch.sin(pix) / pix,
    )


def lanczos_weight(x: Tensor, *, a: int) -> Tensor:
    x = torch.abs(x)
    weight = sinc(x) * sinc(x / float(a))
    return torch.where(x < float(a), weight, torch.zeros_like(weight))


def keys_cubic_weight(x: Tensor, *, b: float, c: float) -> Tensor:
    ax = torch.abs(x)
    ax2 = ax * ax
    ax3 = ax2 * ax
    w0 = (
        ((12.0 - 9.0 * b - 6.0 * c) * ax3)
        + ((-18.0 + 12.0 * b + 6.0 * c) * ax2)
        + (6.0 - 2.0 * b)
    ) / 6.0
    w1 = (
        ((-b - 6.0 * c) * ax3)
        + ((6.0 * b + 30.0 * c) * ax2)
        + ((-12.0 * b - 48.0 * c) * ax)
        + (8.0 * b + 24.0 * c)
    ) / 6.0
    return torch.where(
        ax < 1.0,
        w0,
        torch.where(ax < 2.0, w1, torch.zeros_like(ax)),
    )


def kernel_support(kernel: MapResamplingKernel) -> int:
    if kernel == "lanczos3":
        return 3
    if kernel == "lanczos5":
        return 5
    if kernel in {"mitchell_netravali", "catmull_rom"}:
        return 2
    raise ValueError(f"custom support is undefined for kernel {kernel!r}")


def kernel_weight(kernel: MapResamplingKernel, x: Tensor) -> Tensor:
    if kernel == "lanczos3":
        return lanczos_weight(x, a=3)
    if kernel == "lanczos5":
        return lanczos_weight(x, a=5)
    if kernel == "mitchell_netravali":
        return keys_cubic_weight(x, b=1.0 / 3.0, c=1.0 / 3.0)
    if kernel == "catmull_rom":
        return keys_cubic_weight(x, b=0.0, c=0.5)
    raise ValueError(f"custom weight is undefined for kernel {kernel!r}")


def custom_separable_sample(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    if image.ndim not in {2, 3}:
        raise ValueError(
            "custom separable sampling expects a 2D image or [C, H, W] tensor"
        )
    work = image if image.ndim == 3 else image.unsqueeze(0)
    support = kernel_support(kernel)
    x_src = x_src.to(dtype=work.dtype)
    y_src = y_src.to(dtype=work.dtype)
    x0 = torch.floor(x_src).to(dtype=torch.int64)
    y0 = torch.floor(y_src).to(dtype=torch.int64)
    offsets = torch.arange(
        -support + 1,
        support + 1,
        device=image.device,
        dtype=torch.int64,
    )
    xs = x0.unsqueeze(-1) + offsets.view(1, 1, -1)
    ys = y0.unsqueeze(-1) + offsets.view(1, 1, -1)
    dx = x_src.unsqueeze(-1) - xs.to(dtype=work.dtype)
    dy = y_src.unsqueeze(-1) - ys.to(dtype=work.dtype)
    wx = kernel_weight(kernel, dx)
    wy = kernel_weight(kernel, dy)

    _, h_src, w_src = work.shape
    x_valid = (xs >= 0) & (xs < w_src)
    y_valid = (ys >= 0) & (ys < h_src)
    xs = xs.clamp(0, w_src - 1)
    ys = ys.clamp(0, h_src - 1)
    sample_idx = ys.unsqueeze(-1) * w_src + xs.unsqueeze(-2)
    sampled = (
        work.reshape(work.shape[0], -1)
        .index_select(1, sample_idx.reshape(-1))
        .reshape(
            work.shape[0],
            *x_src.shape,
            sample_idx.shape[-2],
            sample_idx.shape[-1],
        )
    )
    weights = (
        wy.unsqueeze(-1)
        * wx.unsqueeze(-2)
        * y_valid.unsqueeze(-1).to(dtype=work.dtype)
        * x_valid.unsqueeze(-2).to(dtype=work.dtype)
    )
    out = torch.sum(sampled * weights.unsqueeze(0), dim=(-1, -2))
    return out if image.ndim == 3 else out.squeeze(0)


def sample_image_plane(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    grid: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    if kernel in {"lanczos3", "lanczos5", "mitchell_netravali", "catmull_rom"}:
        return custom_separable_sample(
            image,
            x_src=x_src,
            y_src=y_src,
            kernel=kernel,
        )
    return (
        F.grid_sample(
            image.unsqueeze(0).unsqueeze(0),
            grid.unsqueeze(0),
            mode=kernel,
            padding_mode="zeros",
            align_corners=True,
        )
        .squeeze(0)
        .squeeze(0)
    )


def sample_image_plane_channels(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    grid: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    if image.ndim != 3:
        raise ValueError("channel sampling expects [C, H, W] input")
    if kernel in {"lanczos3", "lanczos5", "mitchell_netravali", "catmull_rom"}:
        return custom_separable_sample(
            image,
            x_src=x_src,
            y_src=y_src,
            kernel=kernel,
        )
    return F.grid_sample(
        image.unsqueeze(0),
        grid.unsqueeze(0),
        mode=kernel,
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)


def sample_image_plane_batched(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    grid: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    if x_src.ndim == 2:
        return sample_image_plane(
            image,
            x_src=x_src,
            y_src=y_src,
            grid=grid,
            kernel=kernel,
        )
    if kernel in {"lanczos3", "lanczos5", "mitchell_netravali", "catmull_rom"}:
        samples = [
            custom_separable_sample(
                image,
                x_src=x_src[idx],
                y_src=y_src[idx],
                kernel=kernel,
            )
            for idx in range(int(x_src.shape[0]))
        ]
        return torch.stack(samples, dim=0)
    batch = int(grid.shape[0])
    image_b = (
        image.unsqueeze(0).unsqueeze(0).expand(batch, 1, image.shape[0], image.shape[1])
    )
    return F.grid_sample(
        image_b,
        grid.to(dtype=torch.float32),
        mode=kernel,
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(1)


def sample_image_plane_batched_channels(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    grid: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    if image.ndim != 3:
        raise ValueError("batched channel sampling expects [C, H, W] input")
    if x_src.ndim == 2:
        return sample_image_plane_channels(
            image,
            x_src=x_src,
            y_src=y_src,
            grid=grid,
            kernel=kernel,
        )
    if kernel in {"lanczos3", "lanczos5", "mitchell_netravali", "catmull_rom"}:
        samples = [
            custom_separable_sample(
                image,
                x_src=x_src[idx],
                y_src=y_src[idx],
                kernel=kernel,
            )
            for idx in range(int(x_src.shape[0]))
        ]
        return torch.stack(samples, dim=0)
    batch = int(grid.shape[0])
    image_b = image.unsqueeze(0).expand(batch, *image.shape)
    return F.grid_sample(
        image_b,
        grid.to(dtype=torch.float32),
        mode=kernel,
        padding_mode="zeros",
        align_corners=True,
    )


def infer_fft_translation_shift_yx(
    source_wcs: WCS,
    target_wcs: WCS,
    shape_hw: tuple[int, int],
) -> tuple[float, float]:
    """Infer ``(shift_y, shift_x)`` for translation FFT resampling."""
    if getattr(source_wcs, "sip", None) is not None or bool(
        getattr(source_wcs, "_is_tpv", False)
    ):
        raise ValueError(
            "fft_shift requires a linear astrometric frame (no SIP or TPV on the source WCS)"
        )
    h, w = int(shape_hw[0]), int(shape_hw[1])
    dev = torch.device("cpu")
    jx = torch.tensor(
        [0.0, float(w - 1), float(w - 1) * 0.5, 0.0, float(w - 1)],
        device=dev,
        dtype=torch.float64,
    )
    iy = torch.tensor(
        [0.0, 0.0, float(h - 1) * 0.5, float(h - 1), float(h - 1)],
        device=dev,
        dtype=torch.float64,
    )
    ra, dec = target_wcs.pixel_to_world(jx, iy)
    xs, ys = source_wcs.world_to_pixel(ra, dec)
    dx = xs - jx
    dy = ys - iy
    mx = float(dx.mean().item())
    my = float(dy.mean().item())
    if not torch.allclose(dx, torch.full_like(dx, mx), rtol=1e-4, atol=1e-6):
        raise ValueError(
            "fft_shift requires a constant pixel translation between target and source WCS "
            "(column offset not uniform on probe points)"
        )
    if not torch.allclose(dy, torch.full_like(dy, my), rtol=1e-4, atol=1e-6):
        raise ValueError(
            "fft_shift requires a constant pixel translation between target and source WCS "
            "(row offset not uniform on probe points)"
        )
    return (-my, -mx)


def infer_diagonal_scale_shift_pixel_map(
    source_wcs: WCS,
    target_wcs: WCS,
    shape_hw: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Infer ``(scale_x, shift_x, scale_y, shift_y)`` for a diagonal affine map."""
    if getattr(source_wcs, "sip", None) is not None or bool(
        getattr(source_wcs, "_is_tpv", False)
    ):
        raise ValueError(
            "fft_diagonal_affine requires a linear astrometric frame (no SIP or TPV on source)"
        )
    h, w = int(shape_hw[0]), int(shape_hw[1])
    dev = torch.device("cpu")
    j_line = torch.linspace(0.0, float(w - 1), 5, device=dev, dtype=torch.float64)
    i_line = torch.linspace(0.0, float(h - 1), 5, device=dev, dtype=torch.float64)
    ii, jj = torch.meshgrid(i_line, j_line, indexing="ij")
    iy = ii.reshape(-1)
    jx = jj.reshape(-1)
    ra, dec = target_wcs.pixel_to_world(jx, iy)
    xs, ys = source_wcs.world_to_pixel(ra, dec)
    ones = torch.ones_like(jx)
    ax = torch.stack([jx, iy, ones], dim=1)
    sol_x = torch.linalg.lstsq(ax, xs.unsqueeze(-1), rcond=None).solution.squeeze(-1)
    sol_y = torch.linalg.lstsq(
        torch.stack([iy, jx, ones], dim=1), ys.unsqueeze(-1), rcond=None
    ).solution.squeeze(-1)
    sx = float(sol_x[0].item())
    cx = float(sol_x[1].item())
    bx = float(sol_x[2].item())
    sy = float(sol_y[0].item())
    dy = float(sol_y[1].item())
    by = float(sol_y[2].item())
    tol_c = max(1e-3, 1e-4 * max(abs(sx), abs(sy), 1.0))
    if abs(cx) > tol_c or abs(dy) > tol_c:
        raise ValueError(
            "fft_diagonal_affine requires a separable diagonal pixel map (cross-terms too large)"
        )
    xs_fit = sx * jx + cx * iy + bx
    ys_fit = sy * iy + dy * jx + by
    max_x = float(torch.max(torch.abs(xs_fit - xs)).item())
    max_y = float(torch.max(torch.abs(ys_fit - ys)).item())
    if max_x > 5e-3 or max_y > 5e-3:
        raise ValueError(
            "fft_diagonal_affine WCS probes do not match a diagonal affine map"
        )
    if not all(math.isfinite(v) for v in (sx, sy, bx, by)):
        raise ValueError("fft_diagonal_affine inferred non-finite parameters")
    if abs(sx) < 1e-8 or abs(sy) < 1e-8:
        raise ValueError("fft_diagonal_affine inferred degenerate scales")
    return sx, bx, sy, by


def infer_linear_affine_pixel_map(
    source_wcs: WCS,
    target_wcs: WCS,
    shape_hw: tuple[int, int],
) -> tuple[float, float, float, float, float, float]:
    """Infer a full 2x2 affine source-pixel map from target pixel coordinates."""
    if getattr(source_wcs, "sip", None) is not None or bool(
        getattr(source_wcs, "_is_tpv", False)
    ):
        raise ValueError(
            "periodic_linear_affine requires a linear astrometric frame (no SIP or TPV on source)"
        )
    h, w = int(shape_hw[0]), int(shape_hw[1])
    dev = torch.device("cpu")
    j_line = torch.linspace(0.0, float(w - 1), 5, device=dev, dtype=torch.float64)
    i_line = torch.linspace(0.0, float(h - 1), 5, device=dev, dtype=torch.float64)
    ii, jj = torch.meshgrid(i_line, j_line, indexing="ij")
    iy = ii.reshape(-1)
    jx = jj.reshape(-1)
    ra, dec = target_wcs.pixel_to_world(jx, iy)
    xs, ys = source_wcs.world_to_pixel(ra, dec)
    ones = torch.ones_like(jx)
    a_mat = torch.stack([jx, iy, ones], dim=1)
    sol_x = torch.linalg.lstsq(a_mat, xs.unsqueeze(-1), rcond=None).solution.squeeze(-1)
    sol_y = torch.linalg.lstsq(a_mat, ys.unsqueeze(-1), rcond=None).solution.squeeze(-1)
    a00 = float(sol_x[0].item())
    a01 = float(sol_x[1].item())
    bx = float(sol_x[2].item())
    a10 = float(sol_y[0].item())
    a11 = float(sol_y[1].item())
    by = float(sol_y[2].item())
    det = a00 * a11 - a01 * a10
    if not math.isfinite(det) or abs(det) < 1e-12:
        raise ValueError("periodic_linear_affine inferred a degenerate linear map")
    xs_fit = a00 * jx + a01 * iy + bx
    ys_fit = a10 * jx + a11 * iy + by
    max_x = float(torch.max(torch.abs(xs_fit - xs)).item())
    max_y = float(torch.max(torch.abs(ys_fit - ys)).item())
    if max_x > 5e-3 or max_y > 5e-3:
        raise ValueError(
            "periodic_linear_affine WCS probes do not match a linear affine pixel map"
        )
    return a00, a01, a10, a11, bx, by


def finalize_periodic_resampled_weighted(
    sampled_numer: Tensor,
    sampled_weight: Tensor,
    sampled_support: Tensor | None,
    *,
    value_semantics: MapValueSemantics,
    target_representative_area_sr: float | None,
) -> tuple[Tensor, Tensor]:
    if sampled_support is not None:
        in_bounds = sampled_support > 0.25
    else:
        in_bounds = torch.ones_like(sampled_numer, dtype=torch.bool)
    sampled_numer = torch.where(
        in_bounds,
        sampled_numer,
        torch.zeros_like(sampled_numer),
    )
    sampled_weight = torch.where(
        in_bounds,
        sampled_weight,
        torch.zeros_like(sampled_weight),
    )
    sampled_weight = torch.where(
        sampled_weight > 0.0,
        sampled_weight,
        torch.zeros_like(sampled_weight),
    )
    sampled_numer = torch.where(
        sampled_weight > 0.0,
        sampled_numer,
        torch.zeros_like(sampled_numer),
    )
    if (
        value_semantics == "surface_brightness"
        and target_representative_area_sr is not None
    ):
        sampled_numer = sampled_numer * float(target_representative_area_sr)
        sampled_weight = sampled_weight * float(target_representative_area_sr)
    return sampled_numer, sampled_weight
