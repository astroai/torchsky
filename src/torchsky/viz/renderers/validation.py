"""
Validation diagnostics renderers.

Computes truth/model/residual/pull PNGs and statistical summaries
for the model validation panel.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .image import apply_stretch, to_png, to_png_symmetric, zscale_limits


def residual_panel_pngs(
    truth: Tensor,
    model: Tensor,
    sigma: Tensor | None = None,
    stretch: str = "zscale",
    cmap: str = "gray",
    pull_sigma_range: float = 5.0,
) -> dict[str, bytes]:
    """
    Render the 4-panel diagnostic grid.

    Returns dict with keys: 'truth', 'model', 'residual', 'pull'
    Each value is PNG bytes.

    The pull map uses RdBu_r colormap symmetric at ±pull_sigma_range.
    Pixels outside ±3σ appear as near-saturated red/blue,
    making model outliers immediately visible.
    """
    truth_f = truth.float()
    model_f = model.float()
    residual = model_f - truth_f

    # Display limits for truth and model (shared scale for visual comparison)
    sample = truth_f.flatten()
    sample = sample[torch.isfinite(sample)]
    if sample.numel() > 0:
        vmin, vmax = zscale_limits(sample)
    else:
        vmin, vmax = 0.0, 1.0

    truth_png = to_png(apply_stretch(truth_f, vmin, vmax, stretch), cmap=cmap)
    model_png = to_png(apply_stretch(model_f, vmin, vmax, stretch), cmap=cmap)

    # Residual: same stretch scale for context
    res_stretch = apply_stretch(residual, -(vmax - vmin), vmax - vmin, "linear")
    residual_png = to_png(res_stretch, cmap="gray")

    # Pull map: (model - truth) / sigma, symmetric RdBu_r
    if sigma is not None:
        sigma_f = sigma.float().clamp_min(1e-30)
        pull = residual / sigma_f
    else:
        # Use robust sigma estimate of the residual
        flat = residual.flatten()
        flat = flat[torch.isfinite(flat)]
        if flat.numel() > 0:
            std_est = float(flat.std().item())
        else:
            std_est = 1.0
        pull = residual / max(std_est, 1e-30)

    pull_png = to_png_symmetric(pull, vmax=pull_sigma_range, cmap="rdbu_r")

    return {
        "truth": truth_png,
        "model": model_png,
        "residual": residual_png,
        "pull": pull_png,
    }


def pull_histogram(
    truth: Tensor,
    model: Tensor,
    sigma: Tensor | None = None,
    n_bins: int = 80,
) -> dict:
    """
    Compute pull distribution histogram + Gaussian fit.

    Returns dict with bins, counts, and fit parameters, ready for Observable Plot.
    """
    residual = (model - truth).float()

    if sigma is not None:
        pull = residual / sigma.float().clamp_min(1e-30)
    else:
        flat = residual.flatten()
        flat = flat[torch.isfinite(flat)]
        std_est = float(flat.std().item()) if flat.numel() > 1 else 1.0
        pull = residual / max(std_est, 1e-30)

    flat = pull.flatten()
    flat = flat[torch.isfinite(flat)]

    if flat.numel() == 0:
        return {"bins": [], "counts": [], "gaussian": None, "stats": {}}

    # Cap at ±10σ for display
    flat = flat.clamp(-10.0, 10.0)
    mu = float(flat.mean().item())
    std = float(flat.std().item())
    outlier_frac = float((flat.abs() > 3.0).float().mean().item())

    counts = torch.histc(flat, bins=n_bins, min=-6.0, max=6.0)
    edges = torch.linspace(-6.0, 6.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Expected Gaussian counts for comparison
    bin_width = 12.0 / n_bins
    n_total = flat.numel()
    gaussian_counts = [
        n_total
        * bin_width
        / (std * math.sqrt(2 * math.pi))
        * math.exp(-0.5 * ((float(c.item()) - mu) / max(std, 1e-6)) ** 2)
        for c in centers
    ]

    return {
        "bins": centers.tolist(),
        "counts": counts.tolist(),
        "gaussian_counts": gaussian_counts,
        "fit": {
            "mu": round(mu, 4),
            "sigma": round(std, 4),
        },
        "stats": {
            "outlier_fraction_3sigma": round(outlier_frac, 4),
            "n_pixels": int(flat.numel()),
            "rms_pull": round(float((flat**2).mean().sqrt().item()), 4),
        },
    }


def chi2_map_png(
    truth: Tensor,
    model: Tensor,
    sigma: Tensor,
) -> bytes:
    """
    Render (truth−model)²/σ² as a chi-squared map.

    Expected value is 1 per pixel if the model is correct.
    Spatial structure in this map reveals systematic model failures.
    """
    residual = (model - truth).float()
    sigma_f = sigma.float().clamp_min(1e-30)
    chi2 = (residual / sigma_f) ** 2

    # Log-stretch chi2 to reveal structure across dynamic range
    vmax = float(
        torch.quantile(chi2.flatten()[torch.isfinite(chi2.flatten())], 0.99).item()
    )
    vmax = max(vmax, 1.0)
    stretched = apply_stretch(chi2, 0.0, vmax, "log")
    return to_png(stretched, cmap="viridis")


def chi2_stats(truth: Tensor, model: Tensor, sigma: Tensor) -> dict:
    """
    Return chi-squared statistics for the validation summary.
    """
    residual = (model - truth).float()
    sigma_f = sigma.float().clamp_min(1e-30)
    chi2 = (residual / sigma_f) ** 2
    flat = chi2.flatten()
    finite = flat[torch.isfinite(flat)]

    if finite.numel() == 0:
        return {"mean_chi2": None, "median_chi2": None, "reduced_chi2": None}

    mean_chi2 = float(finite.mean().item())
    median_chi2 = float(torch.median(finite).item())
    # Reduced chi2 (DOF ≈ n_pixels - 1 for illustration)
    reduced = mean_chi2  # already per-pixel

    return {
        "mean_chi2": round(mean_chi2, 4),
        "median_chi2": round(median_chi2, 4),
        "reduced_chi2": round(reduced, 4),
        "fraction_gt1": round(float((finite > 1.0).float().mean().item()), 4),
        "fraction_gt9": round(float((finite > 9.0).float().mean().item()), 6),
    }
