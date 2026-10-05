from __future__ import annotations

import torch
from torch import Tensor


def huber_location(
    x: Tensor,
    *,
    dim: int = -1,
    delta: float = 1.345,
    keepdim: bool = False,
) -> Tensor:
    """One-step Huber robust location estimate along *dim*.

    Uses the median as initial location, MAD as scale, then one
    reweighted step.  Vectorised over all batch dimensions.
    """
    with torch.no_grad():
        mid = x.shape[dim] // 2
        med = torch.select(torch.sort(x, dim=dim).values, dim, mid).unsqueeze(dim)
        r = x - med
        mad = (
            torch.select(torch.sort(r.abs(), dim=dim).values, dim, mid)
            .unsqueeze(dim)
            .clamp(min=1e-10)
        )
    scale = mad * 1.4826
    thresh = delta * scale
    w = torch.where(
        r.abs() <= thresh,
        torch.ones_like(r),
        thresh / r.abs().clamp_min(1e-10),
    )
    num = (w * x).sum(dim=dim, keepdim=keepdim)
    den = w.sum(dim=dim, keepdim=keepdim).clamp(min=1e-10)
    return num / den


def tukey_location(
    x: Tensor,
    *,
    dim: int = -1,
    delta: float = 4.685,
    keepdim: bool = False,
) -> Tensor:
    """One-step Tukey bisquare robust location estimate along *dim*."""
    with torch.no_grad():
        mid = x.shape[dim] // 2
        med = torch.select(torch.sort(x, dim=dim).values, dim, mid).unsqueeze(dim)
        r = x - med
        mad = (
            torch.select(torch.sort(r.abs(), dim=dim).values, dim, mid)
            .unsqueeze(dim)
            .clamp(min=1e-10)
        )
    scale = mad * 1.4826
    u = r / (delta * scale)
    u_sq = u**2
    w = torch.where(u_sq < 1.0, (1.0 - u_sq) ** 2, torch.zeros_like(u))
    num = (w * x).sum(dim=dim, keepdim=keepdim)
    den = w.sum(dim=dim, keepdim=keepdim).clamp(min=1e-10)
    return num / den
