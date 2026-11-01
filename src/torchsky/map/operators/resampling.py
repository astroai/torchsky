"""Image-plane resampling kernels and WCS pixel-map inference helpers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ...wcs import WCS
from .config import MapResamplingKernel, MapValueSemantics
from .execution import wcs_has_distortion


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


_CUSTOM_LANCZOS_KERNELS = frozenset({"lanczos2", "lanczos3", "lanczos4", "lanczos5"})
_CUSTOM_SEPARABLE_KERNELS = frozenset(
    {
        "lanczos2",
        "lanczos3",
        "lanczos4",
        "lanczos5",
        "mitchell_netravali",
        "catmull_rom",
    }
)


def kernel_support(kernel: MapResamplingKernel) -> int:
    if kernel == "lanczos2":
        return 2
    if kernel == "lanczos3":
        return 3
    if kernel == "lanczos4":
        return 4
    if kernel == "lanczos5":
        return 5
    if kernel in {"mitchell_netravali", "catmull_rom"}:
        return 2
    raise ValueError(f"custom support is undefined for kernel {kernel!r}")


def kernel_weight(kernel: MapResamplingKernel, x: Tensor) -> Tensor:
    if kernel == "lanczos2":
        return lanczos_weight(x, a=2)
    if kernel == "lanczos3":
        return lanczos_weight(x, a=3)
    if kernel == "lanczos4":
        return lanczos_weight(x, a=4)
    if kernel == "lanczos5":
        return lanczos_weight(x, a=5)
    if kernel == "mitchell_netravali":
        return keys_cubic_weight(x, b=1.0 / 3.0, c=1.0 / 3.0)
    if kernel == "catmull_rom":
        return keys_cubic_weight(x, b=0.0, c=0.5)
    raise ValueError(f"custom weight is undefined for kernel {kernel!r}")


def _normalize_lanczos_coefficients(coefficients: Tensor, dim: int = -1) -> Tensor:
    total = coefficients.sum(dim=dim, keepdim=True)
    safe_total = torch.where(
        torch.abs(total) > torch.finfo(coefficients.dtype).eps,
        total,
        torch.ones_like(total),
    )
    return coefficients / safe_total


def custom_separable_sample(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    if image.ndim not in {2, 3, 4}:
        raise ValueError(
            "custom separable sampling expects [H,W], [C,H,W], or [B,C,H,W]"
        )
    if not image.is_floating_point():
        raise TypeError("custom separable sampling requires a floating image")
    if x_src.shape != y_src.shape or x_src.ndim not in {2, 3}:
        raise ValueError("x_src and y_src must be matching [Hq,Wq] or [B,Hq,Wq]")
    if x_src.device != image.device or y_src.device != image.device:
        raise ValueError("image, x_src, and y_src must be on the same device")
    if not x_src.is_floating_point() or not y_src.is_floating_point():
        raise TypeError("x_src and y_src must be floating tensors")
    if not bool(torch.isfinite(x_src).all()) or not bool(torch.isfinite(y_src).all()):
        raise ValueError("x_src and y_src must contain only finite coordinates")

    original_ndim = image.ndim
    coordinate_batch = x_src.ndim == 3
    if image.ndim == 2:
        work = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        work = image.unsqueeze(0)
    else:
        work = image
    x_batch = x_src.unsqueeze(0) if x_src.ndim == 2 else x_src
    y_batch = y_src.unsqueeze(0) if y_src.ndim == 2 else y_src
    if work.shape[0] == 1 and x_batch.shape[0] > 1:
        work = work.expand(x_batch.shape[0], *work.shape[1:])
    elif x_batch.shape[0] == 1 and work.shape[0] > 1:
        x_batch = x_batch.expand(work.shape[0], *x_batch.shape[1:])
        y_batch = y_batch.expand(work.shape[0], *y_batch.shape[1:])
    elif work.shape[0] != x_batch.shape[0]:
        raise ValueError("image and coordinate batch dimensions must match")

    spatial = tuple(x_batch.shape[1:])
    sampled, _ = _separable_sample_chunked(
        work,
        None,
        x_batch.reshape(x_batch.shape[0], -1),
        y_batch.reshape(y_batch.shape[0], -1),
        kernel=kernel,
        require_finite=False,
        chunk_size=None,
    )
    out = sampled.reshape(work.shape[0], work.shape[1], *spatial)
    if original_ndim < 4 and not coordinate_batch:
        out = out.squeeze(0)
        return out.squeeze(0) if original_ndim == 2 else out
    return out.squeeze(1) if original_ndim == 2 else out


def supports_fused_variance(kernel: MapResamplingKernel) -> bool:
    """Whether :func:`sample_separable_with_variance` implements ``kernel``."""
    return kernel in _CUSTOM_SEPARABLE_KERNELS


def sample_separable_with_variance(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    kernel: MapResamplingKernel,
    source_variance: Tensor | None = None,
    chunk_size: int | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Sample channels and (optionally) diagonal variance in one fused pass.

    ``image`` is ``[C,H,W]`` with ``[Hq,Wq]`` coordinates, or ``[B,C,H,W]`` /
    ``[C,H,W]`` with ``[B,Hq,Wq]`` coordinates, as in
    :func:`custom_separable_sample`; the sampled channels are identical to it.
    ``source_variance`` is a ``[H,W]`` map of independent source variances; the
    returned variance is ``sum(coefficient**2 * source_variance)`` over the tap
    footprint with the same shape as ``x_src`` (no validity mask is applied;
    callers zero invalid samples afterwards, which equals zeroing their
    coefficients). The axis coefficients and tap indices are computed once and
    shared, and the sample dimension is processed in chunks of bounded working
    set (``chunk_size`` samples; derived from the tap/channel count by default).
    """
    if kernel not in _CUSTOM_SEPARABLE_KERNELS:
        raise ValueError(f"fused separable sampling does not support {kernel!r}")
    if image.ndim not in {3, 4} or not image.is_floating_point():
        raise ValueError("fused sampling expects floating [C,H,W] or [B,C,H,W] input")
    if x_src.shape != y_src.shape or x_src.ndim not in {2, 3}:
        raise ValueError("x_src and y_src must be matching [Hq,Wq] or [B,Hq,Wq]")
    if x_src.device != image.device or y_src.device != image.device:
        raise ValueError("image, x_src, and y_src must be on the same device")
    if not x_src.is_floating_point() or not y_src.is_floating_point():
        raise TypeError("x_src and y_src must be floating tensors")
    work = image.unsqueeze(0) if image.ndim == 3 else image
    x_batch = x_src.unsqueeze(0) if x_src.ndim == 2 else x_src
    y_batch = y_src.unsqueeze(0) if y_src.ndim == 2 else y_src
    if work.shape[0] == 1 and x_batch.shape[0] > 1:
        work = work.expand(x_batch.shape[0], *work.shape[1:])
    elif x_batch.shape[0] == 1 and work.shape[0] > 1:
        x_batch = x_batch.expand(work.shape[0], *x_batch.shape[1:])
        y_batch = y_batch.expand(work.shape[0], *y_batch.shape[1:])
    elif work.shape[0] != x_batch.shape[0]:
        raise ValueError("image and coordinate batch dimensions must match")
    if source_variance is not None and (
        source_variance.shape != work.shape[-2:]
        or source_variance.device != image.device
        or not source_variance.is_floating_point()
    ):
        raise ValueError(
            "source_variance must be a floating [H,W] tensor on the device"
        )
    if source_variance is not None and (
        bool((source_variance < 0.0).any())
        or not bool(torch.isfinite(source_variance).all())
    ):
        raise ValueError("source_variance must be finite and non-negative")
    spatial = tuple(x_batch.shape[1:])
    sampled, variance = _separable_sample_chunked(
        work,
        source_variance,
        x_batch.reshape(x_batch.shape[0], -1),
        y_batch.reshape(y_batch.shape[0], -1),
        kernel=kernel,
        require_finite=True,
        chunk_size=chunk_size,
    )
    sampled = sampled.reshape(work.shape[0], work.shape[1], *spatial)
    if variance is not None:
        variance = variance.reshape(work.shape[0], *spatial)
    if x_src.ndim == 2:
        sampled = sampled.squeeze(0)
        variance = variance.squeeze(0) if variance is not None else None
    return sampled, variance


def sample_image_plane(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    grid: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    if image.ndim != 2 or not image.is_floating_point():
        raise ValueError("image-plane sampling expects a floating 2D image")
    if kernel in _CUSTOM_SEPARABLE_KERNELS:
        return custom_separable_sample(
            image,
            x_src=x_src,
            y_src=y_src,
            kernel=kernel,
        )
    return (
        F.grid_sample(
            image.unsqueeze(0).unsqueeze(0),
            grid.to(device=image.device, dtype=image.dtype).unsqueeze(0),
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
    if image.ndim != 3 or not image.is_floating_point():
        raise ValueError("channel sampling expects floating [C, H, W] input")
    if kernel in _CUSTOM_SEPARABLE_KERNELS:
        return custom_separable_sample(
            image,
            x_src=x_src,
            y_src=y_src,
            kernel=kernel,
        )
    return F.grid_sample(
        image.unsqueeze(0),
        grid.to(device=image.device, dtype=image.dtype).unsqueeze(0),
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
    if image.ndim != 2:
        raise ValueError("batched image-plane sampling expects [H,W] input")
    sampled = sample_image_plane_batched_channels(
        image.unsqueeze(0),
        x_src=x_src,
        y_src=y_src,
        grid=grid,
        kernel=kernel,
    )
    return sampled.squeeze(0) if x_src.ndim == 2 else sampled.squeeze(1)


def sample_image_plane_batched_channels(
    image: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    grid: Tensor,
    kernel: MapResamplingKernel,
) -> Tensor:
    """Sample batched image channels at source-plane coordinates."""
    if image.ndim not in {3, 4} or not image.is_floating_point():
        raise ValueError(
            "batched channel sampling expects floating [C,H,W] or [B,C,H,W] input"
        )
    if x_src.shape != y_src.shape or x_src.ndim not in {2, 3}:
        raise ValueError("x_src and y_src must be matching [Hq,Wq] or [B,Hq,Wq]")
    if x_src.device != image.device or y_src.device != image.device:
        raise ValueError("image, x_src, and y_src must be on the same device")
    grid_expected = (*x_src.shape, 2)
    if tuple(grid.shape) != grid_expected:
        raise ValueError(
            f"grid must have shape {grid_expected}, got {tuple(grid.shape)}"
        )
    if grid.device != image.device:
        raise ValueError("image and grid must be on the same device")
    if x_src.ndim == 2 and image.ndim == 3:
        return sample_image_plane_channels(
            image,
            x_src=x_src,
            y_src=y_src,
            grid=grid,
            kernel=kernel,
        )
    if kernel in _CUSTOM_SEPARABLE_KERNELS:
        return custom_separable_sample(
            image,
            x_src=x_src,
            y_src=y_src,
            kernel=kernel,
        )

    image_b = image.unsqueeze(0) if image.ndim == 3 else image
    grid_b = grid.unsqueeze(0) if grid.ndim == 3 else grid
    if image_b.shape[0] == 1 and grid_b.shape[0] > 1:
        image_b = image_b.expand(grid_b.shape[0], *image_b.shape[1:])
    elif grid_b.shape[0] == 1 and image_b.shape[0] > 1:
        grid_b = grid_b.expand(image_b.shape[0], *grid_b.shape[1:])
    elif image_b.shape[0] != grid_b.shape[0]:
        raise ValueError("image and grid batch dimensions must match")
    sampled = F.grid_sample(
        image_b,
        grid_b.to(dtype=image.dtype),
        mode=kernel,
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(0) if image.ndim == 3 and x_src.ndim == 2 else sampled


def _axis_interpolation_coefficients(
    coordinate: Tensor,
    *,
    size: int,
    kernel: MapResamplingKernel,
) -> tuple[Tensor, Tensor]:
    """Per-axis tap indices and coefficients, tap-major: both are ``[T, *coord]``.

    Out-of-range taps and non-finite coordinates get zero coefficients; indices
    are clamped into ``[0, size)`` so they can be gathered safely. Lanczos
    coefficients are normalised over the full ``T``-tap window before masking.
    """
    finite = torch.isfinite(coordinate)
    safe_coordinate = torch.where(finite, coordinate, torch.zeros_like(coordinate))
    tap_shape = (-1,) + (1,) * coordinate.ndim
    if kernel == "nearest":
        indices = torch.round(safe_coordinate).to(dtype=torch.int64).unsqueeze(0)
        coefficients = torch.ones_like(safe_coordinate).unsqueeze(0)
    else:
        if kernel == "bilinear":
            first, count = 0, 2
        elif kernel == "bicubic":
            first, count = -1, 4
        elif kernel in _CUSTOM_SEPARABLE_KERNELS:
            first, count = 1 - kernel_support(kernel), 2 * kernel_support(kernel)
        else:
            raise ValueError(
                f"diagonal variance coefficients are unavailable for kernel {kernel!r}"
            )
        offsets = torch.arange(
            first, first + count, device=coordinate.device, dtype=torch.int64
        ).view(tap_shape)
        indices = torch.floor(safe_coordinate).to(dtype=torch.int64).unsqueeze(0)
        indices = indices + offsets
        delta = safe_coordinate.unsqueeze(0) - indices.to(dtype=coordinate.dtype)
        if kernel == "bilinear":
            coefficients = 1.0 - torch.abs(delta)
        elif kernel == "bicubic":
            coefficients = keys_cubic_weight(delta, b=0.0, c=0.75)
        else:
            coefficients = kernel_weight(kernel, delta)
            if kernel in _CUSTOM_LANCZOS_KERNELS:
                coefficients = _normalize_lanczos_coefficients(coefficients, dim=0)
    valid = finite.unsqueeze(0) & (indices >= 0) & (indices < int(size))
    return indices.clamp(0, int(size) - 1), coefficients * valid.to(
        dtype=coefficients.dtype
    )


# Working-set budget (bytes) for one chunk of the separable sampler. Peak memory
# is O(chunk * taps * channels) instead of O(N * taps^2 * channels). Throughput
# is flat for chunks of 2^14..2^18 samples on CPU (1M-sample chunks are ~3x
# slower from cache/memory pressure); the default sits in that plateau.
_SEPARABLE_WORKING_SET_BYTES = 48 * 1024 * 1024
_SEPARABLE_MIN_CHUNK = 1 << 12
_SEPARABLE_MAX_CHUNK = 1 << 20


def _axis_tap_count(kernel: MapResamplingKernel) -> int:
    if kernel == "nearest":
        return 1
    if kernel == "bilinear":
        return 2
    if kernel == "bicubic":
        return 4
    return 2 * kernel_support(kernel)


def _separable_chunk_size(taps: int, columns: int, itemsize: int) -> int:
    per_sample = (
        itemsize
        * (2 * taps * columns + 8 * taps)  # gathered rows + products, coefficients
        + 8 * 3 * taps  # int64 tap indices (x, row base, flat)
    )
    chunk = _SEPARABLE_WORKING_SET_BYTES // max(per_sample, 1)
    return int(max(_SEPARABLE_MIN_CHUNK, min(_SEPARABLE_MAX_CHUNK, chunk)))


def _column_weights(coefficients: Tensor, channels: int, with_variance: bool) -> Tensor:
    """Tap weights per table column: ``w`` for channels, ``w**2`` for variance."""
    weights = coefficients.unsqueeze(-1)
    if not with_variance:
        return weights
    return torch.cat(
        [weights.expand(*weights.shape[:-1], channels), weights * weights], dim=-1
    )


def _separable_sample_chunked(
    planes: Tensor | None,
    variance: Tensor | None,
    x: Tensor,
    y: Tensor,
    *,
    kernel: MapResamplingKernel,
    require_finite: bool,
    chunk_size: int | None,
) -> tuple[Tensor | None, Tensor | None]:
    """Separable-kernel sampling of channels and diagonal variance, chunked.

    ``planes`` is ``[B,C,H,W]`` (``B`` may be an expanded view), ``variance`` an
    ``[H,W]`` independent-variance map, ``x``/``y`` are ``[B,N]`` pixel
    coordinates. Per chunk the axis tap indices and coefficients are computed
    once (``T`` values per axis, not ``T*T``); source channels and variance are
    gathered together as one channels-last row per tap and combined in two
    separable stages (columns within a row, then rows), with squared axis
    coefficients applied to the variance column. The result of every sample is
    the same sum over the ``T x T`` footprint as the dense formulation, with the
    normalisation and out-of-bounds masking of ``_axis_interpolation_coefficients``.
    Returns ``(sampled [B,C,N] | None, variance [B,N] | None)``.
    """
    if planes is None and variance is None:
        raise ValueError("nothing to sample")
    if planes is not None:
        batch, channels, height, width = (int(v) for v in planes.shape)
        dtype, device = planes.dtype, planes.device
    else:
        assert variance is not None
        batch, channels = int(x.shape[0]), 0
        height, width = (int(v) for v in variance.shape)
        dtype, device = variance.dtype, variance.device
    count = int(x.shape[1])
    columns = channels + (1 if variance is not None else 0)
    taps = _axis_tap_count(kernel)
    chunk = (
        _separable_chunk_size(
            taps, columns, torch.empty((), dtype=dtype).element_size()
        )
        if chunk_size is None
        else int(chunk_size)
    )
    if chunk < 1:
        raise ValueError("chunk_size must be positive")
    sampled_out = (
        torch.empty((batch, channels, count), dtype=dtype, device=device)
        if planes is not None
        else None
    )
    variance_out = (
        torch.empty((batch, count), dtype=dtype, device=device)
        if variance is not None
        else None
    )
    finite_ok = torch.ones((), dtype=torch.bool, device=device)
    variance_column = (
        None if variance is None else variance.to(dtype=dtype).reshape(1, -1)
    )
    table = None
    for b in range(batch):
        if planes is not None and (table is None or planes.stride(0) != 0):
            table = planes[b].reshape(channels, -1)
            if variance_column is not None:
                table = torch.cat([table, variance_column], dim=0)
            table = table.t().contiguous()
        elif planes is None and table is None:
            table = variance_column.t().contiguous()  # type: ignore[union-attr]
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            xc = x[b, start:stop].to(dtype=dtype)
            yc = y[b, start:stop].to(dtype=dtype)
            if require_finite:
                finite_ok = (
                    finite_ok & torch.isfinite(xc).all() & torch.isfinite(yc).all()
                )
            x_idx, x_coef = _axis_interpolation_coefficients(
                xc, size=width, kernel=kernel
            )
            y_idx, y_coef = _axis_interpolation_coefficients(
                yc, size=height, kernel=kernel
            )
            x_w = _column_weights(x_coef, channels, variance is not None)
            y_w = _column_weights(y_coef, channels, variance is not None)
            acc = None
            for ty in range(y_idx.shape[0]):
                flat = x_idx + y_idx[ty] * width
                row = None
                for tx in range(x_idx.shape[0]):
                    gathered = table.index_select(0, flat[tx])
                    if row is None:
                        row = gathered * x_w[tx]
                    else:
                        row.addcmul_(gathered, x_w[tx])
                if acc is None:
                    acc = row * y_w[ty]
                else:
                    acc.addcmul_(row, y_w[ty])
            assert acc is not None
            if sampled_out is not None:
                sampled_out[b, :, start:stop] = acc[:, :channels].t()
            if variance_out is not None:
                variance_out[b, start:stop] = acc[:, channels]
    if require_finite and not bool(finite_ok):
        raise ValueError("x_src and y_src must contain only finite coordinates")
    return sampled_out, variance_out


def sample_image_plane_diagonal_variance(
    source_variance: Tensor,
    *,
    x_src: Tensor,
    y_src: Tensor,
    kernel: MapResamplingKernel,
    average_coordinate_batch: bool = False,
    sample_valid: Tensor | None = None,
) -> Tensor:
    """Apply squared interpolation coefficients to independent source variance.

    With ``average_coordinate_batch=True``, the leading coordinate dimension is
    averaged as one quadrature operator before coefficients are squared. This
    preserves covariance when multiple samples reuse the same source pixel.
    """
    if source_variance.ndim != 2 or not source_variance.is_floating_point():
        raise ValueError("source_variance must be a floating [H,W] tensor")
    if x_src.shape != y_src.shape or x_src.ndim not in {2, 3}:
        raise ValueError("x_src and y_src must be matching [Hq,Wq] or [B,Hq,Wq]")
    if x_src.device != source_variance.device or y_src.device != source_variance.device:
        raise ValueError("source_variance, x_src, and y_src must share a device")
    if bool((source_variance < 0.0).any()) or not bool(
        torch.isfinite(source_variance).all()
    ):
        raise ValueError("source_variance must be finite and non-negative")
    if average_coordinate_batch and x_src.ndim != 3:
        raise ValueError("average_coordinate_batch requires [B,Hq,Wq] coordinates")
    if sample_valid is not None:
        if sample_valid.shape != x_src.shape or sample_valid.device != x_src.device:
            raise ValueError("sample_valid must match coordinate shape and device")
        sample_valid = sample_valid.to(dtype=torch.bool)

    height, width = int(source_variance.shape[0]), int(source_variance.shape[1])
    if average_coordinate_batch:
        batch, out_height, out_width = (
            int(x_src.shape[0]),
            int(x_src.shape[1]),
            int(x_src.shape[2]),
        )
        pixel_count = out_height * out_width
        if kernel == "nearest":
            axis_terms = 1
        elif kernel == "bilinear":
            axis_terms = 2
        elif kernel == "bicubic":
            axis_terms = 4
        else:
            axis_terms = 2 * kernel_support(kernel)
        terms_per_pixel = batch * axis_terms * axis_terms
        # ponytail: bounded per-pixel sorting is exact and simple; replace the
        # Python term scan with a segmented kernel only if quadrature profiles it.
        chunk_pixels = max(
            1,
            min(pixel_count, (64 * 1024 * 1024) // max(48 * terms_per_pixel, 1)),
        )
        output = torch.empty(
            pixel_count,
            dtype=source_variance.dtype,
            device=source_variance.device,
        )
        flat_x = x_src.reshape(batch, pixel_count)
        flat_y = y_src.reshape(batch, pixel_count)
        flat_valid = (
            None if sample_valid is None else sample_valid.reshape(batch, pixel_count)
        )
        source_flat = source_variance.reshape(-1)
        for start in range(0, pixel_count, chunk_pixels):
            stop = min(start + chunk_pixels, pixel_count)
            x_indices, x_coefficients = _axis_interpolation_coefficients(
                flat_x[:, start:stop].to(dtype=source_variance.dtype),
                size=width,
                kernel=kernel,
            )
            x_indices = x_indices.movedim(0, -1)
            x_coefficients = x_coefficients.movedim(0, -1)
            y_indices, y_coefficients = _axis_interpolation_coefficients(
                flat_y[:, start:stop].to(dtype=source_variance.dtype),
                size=height,
                kernel=kernel,
            )
            y_indices = y_indices.movedim(0, -1)
            y_coefficients = y_coefficients.movedim(0, -1)
            indices = (
                y_indices.unsqueeze(-1) * width + x_indices.unsqueeze(-2)
            ).permute(1, 0, 2, 3)
            coefficients = y_coefficients.unsqueeze(-1) * x_coefficients.unsqueeze(-2)
            if flat_valid is not None:
                coefficients = coefficients * flat_valid[:, start:stop].unsqueeze(
                    -1
                ).unsqueeze(-1).to(dtype=coefficients.dtype)
            coefficients = coefficients.permute(1, 0, 2, 3) / float(batch)
            indices = indices.reshape(stop - start, terms_per_pixel)
            coefficients = coefficients.reshape(stop - start, terms_per_pixel)
            sorted_indices, order = torch.sort(indices, dim=1)
            sorted_coefficients = torch.gather(coefficients, 1, order)
            current_index = sorted_indices[:, 0]
            current_coefficient = sorted_coefficients[:, 0]
            variance = torch.zeros(
                stop - start,
                dtype=source_variance.dtype,
                device=source_variance.device,
            )
            for term in range(1, terms_per_pixel):
                next_index = sorted_indices[:, term]
                next_coefficient = sorted_coefficients[:, term]
                same = next_index == current_index
                variance = variance + torch.where(
                    same,
                    torch.zeros_like(variance),
                    source_flat[current_index] * torch.square(current_coefficient),
                )
                current_index = torch.where(same, current_index, next_index)
                current_coefficient = torch.where(
                    same,
                    current_coefficient + next_coefficient,
                    next_coefficient,
                )
            output[start:stop] = variance + source_flat[current_index] * torch.square(
                current_coefficient
            )
        return output.reshape(out_height, out_width)

    _, variance = _separable_sample_chunked(
        None,
        source_variance,
        x_src.reshape(1, -1),
        y_src.reshape(1, -1),
        kernel=kernel,
        require_finite=False,
        chunk_size=None,
    )
    assert variance is not None
    variance = variance.reshape(x_src.shape)
    if sample_valid is not None:
        variance = torch.where(sample_valid, variance, torch.zeros_like(variance))
    return variance


def _tan_homography_factors(wcs: WCS) -> tuple[Tensor, Tensor] | None:
    if not bool(getattr(wcs, "_is_tan", False)) or wcs_has_distortion(wcs):
        return None
    cd = wcs.cd.detach().to(device="cpu", dtype=torch.float64)
    det_cd = float(torch.linalg.det(cd).item())
    if not math.isfinite(det_cd) or abs(det_cd) < 1e-20:
        return None
    cd_inv = torch.linalg.inv(cd)
    crpix = wcs.crpix.detach().to(device="cpu", dtype=torch.float64)
    p0_x = float(crpix[0].item()) - 1.0
    p0_y = float(crpix[1].item()) - 1.0
    px = torch.tensor(
        [
            p0_x,
            p0_x + float(cd_inv[0, 0].item()),
            p0_x + float(cd_inv[0, 1].item()),
        ],
        dtype=torch.float64,
        device=wcs.device,
    )
    py = torch.tensor(
        [
            p0_y,
            p0_y + float(cd_inv[1, 0].item()),
            p0_y + float(cd_inv[1, 1].item()),
        ],
        dtype=torch.float64,
        device=wcs.device,
    )
    ra_deg, dec_deg = wcs.pixel_to_world(px, py)
    ra = torch.deg2rad(ra_deg.detach().to(device="cpu", dtype=torch.float64))
    dec = torch.deg2rad(dec_deg.detach().to(device="cpu", dtype=torch.float64))
    if not bool(torch.isfinite(ra).all() and torch.isfinite(dec).all()):
        return None
    cos_d = torch.cos(dec)
    vecs = torch.stack(
        (cos_d * torch.cos(ra), cos_d * torch.sin(ra), torch.sin(dec)),
        dim=0,
    )  # (3, 3) where columns are points 0, 1, 2
    s = math.pi / 180.0
    scale = math.sqrt(1.0 + s * s)
    r0 = vecs[:, 0]
    q1 = (scale * vecs[:, 1] - r0) / s
    q2 = (scale * vecs[:, 2] - r0) / s
    q_mat = torch.stack((q1, q2, r0), dim=1)  # (3, 3)

    s_cd = s * cd
    shift = torch.tensor(
        [1.0 - float(crpix[0].item()), 1.0 - float(crpix[1].item())],
        dtype=torch.float64,
    )
    trans = s_cd @ shift
    a_mat = torch.eye(3, dtype=torch.float64)
    a_mat[:2, :2] = s_cd
    a_mat[:2, 2] = trans
    return q_mat, a_mat


def infer_tan_projective_homography(
    source_wcs: WCS,
    target_wcs: WCS,
    shape_hw: tuple[int, int],
) -> Tensor | None:
    """Return exact 3x3 projective homography H from target TAN pixels to source TAN pixels."""
    if getattr(source_wcs, "celestial_frame", "ICRS") != getattr(
        target_wcs, "celestial_frame", "ICRS"
    ) or getattr(source_wcs, "celestial_equinox", None) != getattr(
        target_wcs, "celestial_equinox", None
    ):
        return None
    src_factors = _tan_homography_factors(source_wcs)
    tgt_factors = _tan_homography_factors(target_wcs)
    if src_factors is None or tgt_factors is None:
        return None
    q_src, a_src = src_factors
    q_tgt, a_tgt = tgt_factors
    if (
        float(source_wcs.alpha0) == float(target_wcs.alpha0)
        and float(source_wcs.delta0) == float(target_wcs.delta0)
        and float(getattr(source_wcs, "phi_p", 180.0))
        == float(getattr(target_wcs, "phi_p", 180.0))
    ):
        q_rot = torch.eye(3, dtype=torch.float64)
    else:
        q_rot = q_src.T @ q_tgt
    h_mat = torch.linalg.solve(a_src, q_rot @ a_tgt)
    if not bool(torch.isfinite(h_mat).all()):
        return None
    if abs(float(h_mat[2, 2].item())) > 1e-12:
        h_mat = h_mat / h_mat[2, 2]
    rounded = torch.round(h_mat)
    h_mat = torch.where(torch.abs(h_mat - rounded) < 1e-13, rounded, h_mat)

    h, w = int(shape_hw[0]), int(shape_hw[1])
    jx_cpu = torch.tensor(
        [0.0, float(w - 1), float(w - 1) * 0.5, 0.0, float(w - 1)],
        dtype=torch.float64,
    )
    iy_cpu = torch.tensor(
        [0.0, 0.0, float(h - 1) * 0.5, float(h - 1), float(h - 1)],
        dtype=torch.float64,
    )
    denom = h_mat[2, 0] * jx_cpu + h_mat[2, 1] * iy_cpu + h_mat[2, 2]
    if bool((denom <= 1e-3).any()):
        return None
    xh = (h_mat[0, 0] * jx_cpu + h_mat[0, 1] * iy_cpu + h_mat[0, 2]) / denom
    yh = (h_mat[1, 0] * jx_cpu + h_mat[1, 1] * iy_cpu + h_mat[1, 2]) / denom

    ra, dec = target_wcs.pixel_to_world(
        jx_cpu.to(device=target_wcs.device),
        iy_cpu.to(device=target_wcs.device),
    )
    xs, ys = source_wcs.world_to_pixel(
        ra.to(device=source_wcs.device),
        dec.to(device=source_wcs.device),
    )
    xs_cpu = xs.detach().to(device="cpu", dtype=torch.float64)
    ys_cpu = ys.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(xs_cpu).all() and torch.isfinite(ys_cpu).all()):
        return None
    tol = (
        2e-3
        if (source_wcs.device.type == "mps" or target_wcs.device.type == "mps")
        else 1e-5
    )
    if float(torch.max(torch.abs(xh - xs_cpu)).item()) > tol:
        return None
    if float(torch.max(torch.abs(yh - ys_cpu)).item()) > tol:
        return None
    return h_mat


def evaluate_tan_projective_homography(
    homography: Tensor,
    height: int,
    width: int,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Evaluate 3x3 projective homography on [0..height-1] x [0..width-1] using 1D outer sums."""
    compute_dtype = torch.float32 if device.type == "mps" else torch.float64
    h_mat = homography.to(device=device, dtype=compute_dtype)
    x_row = torch.arange(width, dtype=compute_dtype, device=device).unsqueeze(0)
    y_col = torch.arange(height, dtype=compute_dtype, device=device).unsqueeze(1)
    denom = (h_mat[2, 0] * x_row) + (h_mat[2, 1] * y_col + h_mat[2, 2])
    inv_denom = torch.reciprocal(denom)
    x_src = ((h_mat[0, 0] * x_row) + (h_mat[0, 1] * y_col + h_mat[0, 2])) * inv_denom
    y_src = ((h_mat[1, 0] * x_row) + (h_mat[1, 1] * y_col + h_mat[1, 2])) * inv_denom
    return x_src, y_src


def infer_fft_translation_shift_yx(
    source_wcs: WCS,
    target_wcs: WCS,
    shape_hw: tuple[int, int],
) -> tuple[float, float]:
    """Infer ``(shift_y, shift_x)`` for translation FFT resampling."""
    if wcs_has_distortion(source_wcs):
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
    if wcs_has_distortion(source_wcs):
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
    if wcs_has_distortion(source_wcs):
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
