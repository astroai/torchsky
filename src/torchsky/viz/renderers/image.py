"""
Image stretch algorithms and PNG encoding for torchsky.viz.

All operations are tensor-native. Pillow is used at the final step
only for efficient PNG byte encoding.
"""

from __future__ import annotations

import io
import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from ..palettes import COLORMAP_NAMES as COLORMAP_NAMES
from ..palettes import get_lut as get_lut

# ---------------------------------------------------------------------------
# ZScale — full IRAF iterative algorithm
# ---------------------------------------------------------------------------


def zscale_limits(
    arr: Tensor,
    contrast: float = 0.25,
    n_samples: int = 600,
    max_reject: float = 0.5,
    min_npixels: int = 5,
    krej: float = 2.5,
    max_iterations: int = 5,
) -> tuple[float, float]:
    """
    Full IRAF/DS9 ZScale algorithm.

    Samples the image, fits a line to the sorted intensities in the
    central part of the distribution, and returns (z1, z2) display limits.
    """
    from ...transforms import _zscale_bounds_tensor

    bounds = _zscale_bounds_tensor(
        arr,
        contrast=contrast,
        n_samples=n_samples,
        max_reject=max_reject,
        min_npixels=min_npixels,
        krej=krej,
        max_iterations=max_iterations,
    )
    if bounds is None:
        return 0.0, 1.0
    z1, z2 = bounds
    return float(z1.item()), float(z2.item())



# ---------------------------------------------------------------------------
# Stretch functions
# ---------------------------------------------------------------------------

StretchName = Literal["linear", "log", "sqrt", "asinh", "histeq", "power"]


def apply_stretch(
    arr: Tensor,
    vmin: float,
    vmax: float,
    stretch: StretchName = "linear",
    asinh_a: float = 0.1,
    power_gamma: float = 0.5,
) -> Tensor:
    """
    Clip to [vmin, vmax], normalize to [0, 1], then apply stretch function.

    Returns float32 tensor in [0, 1].
    """
    if stretch not in {"linear", "log", "sqrt", "asinh", "histeq", "power"}:
        raise ValueError(f"Unknown stretch {stretch!r}")
    arr = arr.float()
    span = vmax - vmin
    if abs(span) < 1e-30:
        return torch.zeros_like(arr)

    # Clip + normalize
    normed = (arr.clamp(vmin, vmax) - vmin) / span

    if stretch == "linear":
        return normed

    if stretch == "log":
        # log(1 + a*x) / log(1 + a) where a >> 1 gives log compression
        a = 1000.0
        return torch.log1p(a * normed) / math.log(1 + a)

    if stretch == "sqrt":
        return torch.sqrt(normed)

    if stretch == "asinh":
        # Lupton (2004) asinh stretch
        return torch.asinh(normed / asinh_a) / math.asinh(1.0 / asinh_a)

    if stretch == "power":
        return torch.pow(normed.clamp_min(0), power_gamma)

    if stretch == "histeq":
        return _histeq_stretch(normed)

    raise ValueError(f"Unknown stretch {stretch!r}")


def _histeq_stretch(normed: Tensor) -> Tensor:
    """CDF-based histogram equalization on the unit-normalized image."""
    flat = normed.flatten()
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return normed
    n_bins = 256
    counts = torch.histc(finite, bins=n_bins, min=0.0, max=1.0)
    cdf = counts.cumsum(0)
    cdf = cdf / cdf[-1].clamp_min(1e-10)
    # Map each pixel through the CDF
    indices = (normed * (n_bins - 1)).long().clamp(0, n_bins - 1)
    return cdf[indices]


# ---------------------------------------------------------------------------
# PNG encoding
# ---------------------------------------------------------------------------


def to_png(
    arr: Tensor,
    cmap: str = "gray",
    alpha_mask: Tensor | None = None,
    jpeg: bool = False,
    quality: int = 85,
) -> bytes:
    """
    Convert a float32 [0,1] tensor (H×W) to PNG (or JPEG) bytes.

    Args:
        arr:         H×W float32 tensor, values in [0, 1].
        cmap:        Colormap name from COLORMAP_NAMES.
        alpha_mask:  Optional H×W bool tensor; transparent where False.
        jpeg:        If True, return JPEG bytes instead of PNG (faster, lossy).
        quality:     JPEG quality (1-95) when jpeg=True.
    """
    from PIL import Image

    lut = get_lut(cmap)  # 256×3 uint8
    arr_cpu = arr.detach().cpu().float()

    # Replace non-finite with 0 (will be masked transparent)
    nan_mask = ~torch.isfinite(arr_cpu)
    arr_cpu = arr_cpu.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    indices = (arr_cpu * 255).long().numpy()  # H×W
    rgb = lut[indices]  # H×W×3

    has_nan = bool(nan_mask.any())
    if alpha_mask is not None or has_nan:
        rgba = np.zeros((*indices.shape, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        alpha = np.ones(indices.shape, dtype=np.uint8) * 255
        if has_nan:
            alpha[nan_mask.numpy()] = 0
        if alpha_mask is not None:
            alpha[~alpha_mask.cpu().numpy()] = 0
        rgba[:, :, 3] = alpha
        img = Image.fromarray(rgba, mode="RGBA")
    else:
        img = Image.fromarray(rgb, mode="RGB")

    buf = io.BytesIO()
    if jpeg:
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def to_png_symmetric(
    arr: Tensor,
    vmax: float | None = None,
    cmap: str = "rdbu_r",
) -> bytes:
    """
    Render a residual/pull map with symmetric colorbar centered at zero.

    If vmax is None, uses the 99th percentile of |arr|.
    """
    valid = arr.flatten()
    valid = valid[torch.isfinite(valid)]
    if vmax is None:
        if valid.numel() == 0:
            vmax = 1.0
        else:
            vmax = float(torch.quantile(valid.abs().float(), 0.99).item())
    if vmax < 1e-30:
        vmax = 1.0
    stretched = apply_stretch(arr, -vmax, vmax, "linear")
    return to_png(stretched, cmap=cmap)


# ---------------------------------------------------------------------------
# Dynamic range analysis
# ---------------------------------------------------------------------------


def detect_dynamic_range(arr: Tensor) -> dict:
    """
    Analyze image dynamic range and return display recommendations.

    Returns dict with:
        dr_ratio            : 99th / 1st percentile of positive pixels (robust DR)
        neg_fraction        : fraction of pixels < 0
        cr_fraction         : fraction of pixels > 5σ above median
        recommended_stretch : 'linear' | 'log' | 'asinh'
        warnings            : list of warning strings
        stats               : min/max/median/std/p1/p99 of finite pixels
    """
    flat = arr.flatten().float()
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return {
            "dr_ratio": 1.0,
            "has_negatives": False,
            "cr_fraction": 0.0,
            "recommended_stretch": "linear",
            "warnings": ["empty"],
        }

    neg_frac = float((finite < 0).float().mean().item())
    pos = finite[finite > 0]
    if pos.numel() > 1:
        p1 = float(torch.quantile(pos, 0.01).item())
        p99 = float(torch.quantile(pos, 0.99).item())
        dr = p99 / max(p1, 1e-30)
    else:
        dr = 1.0

    median = float(torch.median(finite).item())
    std = float(finite.std().item())
    cr_frac = float((finite > median + 5 * std).float().mean().item())

    warnings = []
    if dr > 1000:
        warnings.append("high_dynamic_range")
    if neg_frac > 0.01:
        warnings.append("negatives")
    if cr_frac > 0.001:
        warnings.append("cr_hits")

    if dr > 10000:
        rec = "log"
    elif dr > 100:
        rec = "asinh"
    else:
        rec = "linear"

    return {
        "dr_ratio": round(dr, 1),
        "neg_fraction": round(neg_frac, 4),
        "cr_fraction": round(cr_frac, 6),
        "recommended_stretch": rec,
        "warnings": warnings,
        "stats": {
            "min": float(finite.min().item()),
            "max": float(finite.max().item()),
            "median": round(median, 6),
            "std": round(std, 6),
            "p1": round(float(torch.quantile(finite, 0.01).item()), 6),
            "p99": round(float(torch.quantile(finite, 0.99).item()), 6),
        },
    }


def compute_histogram(arr: Tensor, n_bins: int = 256) -> dict:
    """
    Compute histogram + statistics for image display.

    Returns dict with bin centres, counts, ZScale limits, and dynamic range info.
    """
    flat = arr.flatten().float()
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return {"bins": [], "counts": [], "stats": {}, "dr": {}}

    fmin = float(finite.min().item())
    fmax = float(finite.max().item())
    if fmin == fmax:
        fmax = fmin + 1.0

    counts = torch.histc(finite, bins=n_bins, min=fmin, max=fmax)
    edges = torch.linspace(fmin, fmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    z1, z2 = zscale_limits(finite)
    dr = detect_dynamic_range(arr)

    return {
        "bins": centers.tolist(),
        "counts": counts.tolist(),
        "vmin_zscale": z1,
        "vmax_zscale": z2,
        "dr": dr,
    }
