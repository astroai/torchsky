"""Flux-conserving bilinear splat (adjoint of bilinear ``grid_sample``)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SplatDiagnostics:
    """Per-batch accounting for bilinear deposits at finite grid edges."""

    retained_weight: Tensor
    input_flux: Tensor
    deposited_flux: Tensor
    lost_flux: Tensor


def splat_bilinear(
    flux: Tensor,
    xy: Tensor,
    out_shape: tuple[int, int],
    *,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, SplatDiagnostics]:
    """Deposit canonical-pixel flux onto a native grid with bilinear weights.

    Args:
        flux: ``[B, Hc, Wc]`` flux per source pixel.
        xy: ``[B, Hc, Wc, 2]`` native ``(x, y)`` of source-pixel centres
            (0-indexed, pixel-centre convention).
        out_shape: ``(Hn, Wn)`` of the destination grid.

    Returns:
        ``[B, Hn, Wn]`` destination flux. Total flux is conserved for
        in-bounds deposits; out-of-bounds weight is dropped. With
        ``return_diagnostics=True``, also returns explicit retained weights and
        input/deposited/lost flux totals for every batch item.
    """
    if flux.ndim != 3:
        raise ValueError(f"flux must be [B,Hc,Wc], got {tuple(flux.shape)}")
    if xy.shape[:3] != flux.shape or xy.shape[-1] != 2:
        raise ValueError(f"xy must be [B,Hc,Wc,2] matching flux, got {tuple(xy.shape)}")
    if not flux.is_floating_point() or not xy.is_floating_point():
        raise TypeError("flux and xy must be floating tensors")
    if flux.device != xy.device:
        raise ValueError("flux and xy must be on the same device")
    if not bool(torch.isfinite(flux).all()) or not bool(torch.isfinite(xy).all()):
        raise ValueError("flux and xy must contain only finite values")
    hn, wn = int(out_shape[0]), int(out_shape[1])
    if hn <= 0 or wn <= 0:
        raise ValueError("out_shape must contain two positive sizes")
    batch = flux.shape[0]
    device = flux.device
    dtype = flux.dtype

    x = xy[..., 0].to(dtype)
    y = xy[..., 1].to(dtype)
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    # Fractional offsets keep d(weight)/d(position) = +-1 at integer positions
    # (``abs`` has a zero sub-gradient there); the right-hand derivative is used.
    fx = x - x0
    fy = y - y0

    out = torch.zeros(batch, hn * wn, device=device, dtype=dtype)
    retained_weight = torch.zeros_like(flux)
    for dx in (0.0, 1.0):
        for dy in (0.0, 1.0):
            xi = (x0 + dx).long()
            yi = (y0 + dy).long()
            wx = fx if dx else 1.0 - fx
            wy = fy if dy else 1.0 - fy
            weight = wx * wy
            valid = (xi >= 0) & (xi < wn) & (yi >= 0) & (yi < hn)
            weight = torch.where(valid, weight, torch.zeros_like(weight))
            retained_weight = retained_weight + weight
            flat = yi.clamp(0, hn - 1) * wn + xi.clamp(0, wn - 1)
            out.scatter_add_(
                1, flat.reshape(batch, -1), (weight * flux).reshape(batch, -1)
            )
    result = out.reshape(batch, hn, wn)
    if not return_diagnostics:
        return result
    input_flux = flux.sum(dim=(-2, -1))
    deposited_flux = result.sum(dim=(-2, -1))
    diagnostics = SplatDiagnostics(
        retained_weight=retained_weight,
        input_flux=input_flux,
        deposited_flux=deposited_flux,
        lost_flux=input_flux - deposited_flux,
    )
    return result, diagnostics


splat_to_native = splat_bilinear
