"""Flux-conserving bilinear splat (adjoint of bilinear ``grid_sample``)."""

from __future__ import annotations

import torch
from torch import Tensor


def splat_bilinear(
    flux: Tensor, xy: Tensor, out_shape: tuple[int, int]
) -> Tensor:
    """Deposit canonical-pixel flux onto a native grid with bilinear weights.

    Args:
        flux: ``[B, Hc, Wc]`` flux per source pixel.
        xy: ``[B, Hc, Wc, 2]`` native ``(x, y)`` of source-pixel centres
            (0-indexed, pixel-centre convention).
        out_shape: ``(Hn, Wn)`` of the destination grid.

    Returns:
        ``[B, Hn, Wn]`` destination flux. Total flux is conserved for
        in-bounds deposits; out-of-bounds weight is dropped.
    """
    if flux.ndim != 3:
        raise ValueError(f"flux must be [B,Hc,Wc], got {tuple(flux.shape)}")
    if xy.shape[:3] != flux.shape or xy.shape[-1] != 2:
        raise ValueError(
            f"xy must be [B,Hc,Wc,2] matching flux, got {tuple(xy.shape)}"
        )
    hn, wn = int(out_shape[0]), int(out_shape[1])
    batch = flux.shape[0]
    device = flux.device
    dtype = flux.dtype

    x = xy[..., 0].to(dtype)
    y = xy[..., 1].to(dtype)
    x0 = torch.floor(x)
    y0 = torch.floor(y)

    out = torch.zeros(batch, hn * wn, device=device, dtype=dtype)
    for dx in (0.0, 1.0):
        for dy in (0.0, 1.0):
            xi = (x0 + dx).long()
            yi = (y0 + dy).long()
            weight = (1.0 - torch.abs(x - (x0 + dx))) * (1.0 - torch.abs(y - (y0 + dy)))
            valid = (xi >= 0) & (xi < wn) & (yi >= 0) & (yi < hn) & (weight > 0)
            weight = torch.where(valid, weight, torch.zeros_like(weight))
            flat = yi.clamp(0, hn - 1) * wn + xi.clamp(0, wn - 1)
            out.scatter_add_(
                1, flat.reshape(batch, -1), (weight * flux).reshape(batch, -1)
            )
    return out.reshape(batch, hn, wn)


splat_to_native = splat_bilinear
