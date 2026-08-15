"""Predeclared scientific graduation statistics for field reconstructions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class PairedImprovementInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0


def paired_bootstrap_improvement(
    candidate: Tensor,
    baseline: Tensor,
    *,
    larger_is_better: bool = True,
    statistic: Literal["mean", "median"] = "median",
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> PairedImprovementInterval:
    """Paired bootstrap interval for candidate-minus-baseline improvement."""
    candidate = torch.as_tensor(candidate, dtype=torch.float64).reshape(-1)
    baseline = torch.as_tensor(baseline, dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(candidate) & torch.isfinite(baseline)
    difference = candidate[finite] - baseline[finite]
    if not larger_is_better:
        difference = -difference
    if difference.numel() < 2:
        raise ValueError("paired bootstrap requires at least two finite pairs")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    if not 0.0 < float(confidence) < 1.0 or int(samples) < 100:
        raise ValueError("confidence must lie in (0, 1) and samples must be >= 100")
    reduce = torch.mean if statistic == "mean" else torch.median
    estimate = reduce(difference)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        difference.numel(),
        (int(samples), difference.numel()),
        generator=generator,
    )
    draws = difference[indices]
    bootstrap = draws.mean(dim=-1) if statistic == "mean" else draws.median(dim=-1).values
    alpha = 0.5 * (1.0 - float(confidence))
    quantiles = torch.quantile(
        bootstrap, torch.tensor([alpha, 1.0 - alpha], dtype=bootstrap.dtype)
    )
    return PairedImprovementInterval(
        estimate=float(estimate),
        lower=float(quantiles[0]),
        upper=float(quantiles[1]),
        confidence=float(confidence),
    )


def completeness_limit_magnitude(
    magnitude: Tensor,
    completeness: Tensor,
    *,
    target: float,
) -> float:
    """Linearly interpolate the faint-side magnitude at target completeness.

    Returns NaN when the measured curve does not bracket ``target``; it never
    extrapolates a depth claim beyond the injection grid.
    """
    magnitude = torch.as_tensor(magnitude, dtype=torch.float64).reshape(-1)
    completeness = torch.as_tensor(completeness, dtype=torch.float64).reshape(-1)
    if magnitude.numel() != completeness.numel() or magnitude.numel() < 2:
        raise ValueError("magnitude and completeness need matching length >= 2")
    if not 0.0 < float(target) < 1.0:
        raise ValueError("target must lie in (0, 1)")
    finite = torch.isfinite(magnitude) & torch.isfinite(completeness)
    magnitude = magnitude[finite]
    completeness = completeness[finite]
    order = torch.argsort(magnitude)
    magnitude = magnitude[order]
    completeness = completeness[order]
    if magnitude.numel() < 2 or bool((magnitude[1:] <= magnitude[:-1]).any()):
        raise ValueError("finite magnitudes must be distinct")
    # Enforce the physically relevant non-increasing envelope toward faintness.
    completeness = torch.cummin(completeness, dim=0).values
    crossings = torch.nonzero(
        (completeness[:-1] >= target) & (completeness[1:] <= target),
        as_tuple=False,
    )
    if crossings.numel() == 0:
        return math.nan
    index = int(crossings[-1, 0])
    c0, c1 = completeness[index], completeness[index + 1]
    if float(c0) == float(c1):
        return float(magnitude[index + 1])
    fraction = (float(target) - c0) / (c1 - c0)
    return float(magnitude[index] + fraction * (magnitude[index + 1] - magnitude[index]))


__all__ = [
    "PairedImprovementInterval",
    "completeness_limit_magnitude",
    "paired_bootstrap_improvement",
]
