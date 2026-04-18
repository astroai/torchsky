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

# ---------------------------------------------------------------------------
# Colormap LUTs — embedded 256×3 uint8 numpy arrays.
# ---------------------------------------------------------------------------

def _make_lut(*keypoints: tuple[float, tuple[int, int, int]]) -> np.ndarray:
    """Linearly interpolate between (position, RGB) keypoints → 256×3 uint8."""
    lut = np.zeros((256, 3), dtype=np.float32)
    positions = [p for p, _ in keypoints]
    colors = [np.array(c, dtype=np.float32) for _, c in keypoints]
    for i in range(256):
        t = i / 255.0
        # find segment
        idx = 0
        for j in range(len(positions) - 1):
            if t >= positions[j]:
                idx = j
        if idx >= len(positions) - 1:
            lut[i] = colors[-1]
        else:
            t0, t1 = positions[idx], positions[idx + 1]
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            lut[i] = (1 - alpha) * colors[idx] + alpha * colors[idx + 1]
    return np.clip(lut, 0, 255).astype(np.uint8)


_COLORMAPS: dict[str, np.ndarray] = {
    "gray": _make_lut(
        (0.0, (0, 0, 0)),
        (1.0, (255, 255, 255)),
    ),
    "viridis": _make_lut(
        (0.000, (68,   1,  84)),
        (0.125, (72,  40, 120)),
        (0.250, (62,  83, 140)),
        (0.375, (49, 120, 150)),
        (0.500, (38, 153, 137)),
        (0.625, (53, 183, 121)),
        (0.750, (110, 206,  88)),
        (0.875, (181, 222,  43)),
        (1.000, (253, 231,  37)),
    ),
    "hot": _make_lut(
        (0.000, (  0,   0,   0)),
        (0.330, (255,   0,   0)),
        (0.660, (255, 255,   0)),
        (1.000, (255, 255, 255)),
    ),
    "thermal": _make_lut(
        (0.000, (  0,   0, 128)),
        (0.250, (  0, 128, 255)),
        (0.500, (  0, 255,   0)),
        (0.750, (255, 165,   0)),
        (1.000, (255,   0,   0)),
    ),
    "rdbu_r": _make_lut(
        (0.000, (103,   0,  31)),
        (0.200, (214,  96,  77)),
        (0.350, (244, 165, 130)),
        (0.500, (247, 247, 247)),
        (0.650, (146, 197, 222)),
        (0.800, ( 67, 147, 195)),
        (1.000, (  5,  48,  97)),
    ),
    "seismic": _make_lut(
        (0.000, (  0,   0,  76)),
        (0.250, (  0, 102, 255)),
        (0.450, (138, 190, 255)),
        (0.500, (255, 255, 255)),
        (0.550, (255, 138, 138)),
        (0.750, (255,  26,  26)),
        (1.000, ( 76,   0,   0)),
    ),
}

COLORMAP_NAMES = list(_COLORMAPS.keys())


def get_lut(name: str) -> np.ndarray:
    """Return 256×3 uint8 colormap LUT by name."""
    key = name.lower().replace("-", "_")
    if key not in _COLORMAPS:
        raise ValueError(f"Unknown colormap {name!r}. Choose from {COLORMAP_NAMES}")
    return _COLORMAPS[key]


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
    flat = arr.flatten()
    # Remove NaNs and Infs
    valid = flat[torch.isfinite(flat)]
    if valid.numel() == 0:
        return 0.0, 1.0

    # Sub-sample for speed
    if valid.numel() > n_samples:
        step = max(1, valid.numel() // n_samples)
        sample = valid[::step]
    else:
        sample = valid

    sample, _ = torch.sort(sample)
    n = sample.numel()
    if n < min_npixels:
        return float(sample[0].item()), float(sample[-1].item())

    # Iterative sigma-clipped linear fit
    good = torch.ones(n, dtype=torch.bool, device=sample.device)
    x = torch.arange(n, dtype=torch.float64, device=sample.device)
    y = sample.to(torch.float64)

    for _ in range(max_iterations):
        xg = x[good]
        yg = y[good]
        ng = int(xg.numel())
        if ng < min_npixels:
            break
        xm = xg.mean()
        ym = yg.mean()
        slope = ((xg - xm) * (yg - ym)).sum() / ((xg - xm) ** 2).sum().clamp_min(1e-30)
        intercept = ym - slope * xm
        residuals = y - (slope * x + intercept)
        sigma = float(residuals[good].std().item())
        if sigma < 1e-10:
            break
        new_good = torch.abs(residuals) < krej * sigma
        if int(new_good.sum().item()) < min_npixels:
            break
        if float(new_good.sum().item()) / n < 1.0 - max_reject:
            break
        good = new_good

    xg = x[good]
    yg = y[good]
    if xg.numel() < 2:
        return float(sample[n // 4].item()), float(sample[3 * n // 4].item())

    xm = xg.mean()
    ym = yg.mean()
    slope = float(
        (((xg - xm) * (yg - ym)).sum() / ((xg - xm) ** 2).sum().clamp_min(1e-30)).item()
    )
    intercept = float(ym.item()) - slope * float(xm.item())

    center = n / 2.0
    z1 = max(float(sample[0].item()), intercept + slope * (0.0 - center) / contrast)
    z2 = min(float(sample[-1].item()), intercept + slope * (n - center) / contrast)

    if z2 <= z1:
        med = float(torch.median(sample).item())
        std = float(sample.float().std().item())
        z1, z2 = med - 3 * std, med + 3 * std

    return float(z1), float(z2)


def percentile_limits(arr: Tensor, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    """Robust display limits via percentile clipping."""
    valid = arr.flatten()
    valid = valid[torch.isfinite(valid)]
    if valid.numel() == 0:
        return 0.0, 1.0
    qs = torch.quantile(valid.float(), torch.tensor([lo / 100, hi / 100], device=valid.device))
    return float(qs[0].item()), float(qs[1].item())


# ---------------------------------------------------------------------------
# Stretch functions
# ---------------------------------------------------------------------------

# 'zscale' is an alias for 'linear' — ZScale only computes vmin/vmax;
# the actual pixel mapping is always linear after normalization.
StretchName = Literal["linear", "log", "sqrt", "asinh", "histeq", "power", "zscale"]


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
    arr = arr.float()
    span = vmax - vmin
    if abs(span) < 1e-30:
        return torch.zeros_like(arr)

    # Clip + normalize
    normed = (arr.clamp(vmin, vmax) - vmin) / span

    if stretch in ("linear", "zscale"):   # zscale = linear after external limit computation
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

    if alpha_mask is not None or nan_mask.any():
        rgba = np.zeros((*indices.shape, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        alpha = np.ones(indices.shape, dtype=np.uint8) * 255
        if nan_mask.any():
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
        dr_ratio            : max / 99th-pctile (robust DR)
        has_negatives       : fraction of pixels < 0
        cr_fraction         : fraction of pixels > 5σ above median
        recommended_stretch : 'linear' | 'log' | 'asinh'
        warnings            : list of warning strings
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

    Returns dict with bins (edges), counts, stats, and dynamic range info.
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
