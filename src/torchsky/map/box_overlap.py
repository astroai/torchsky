"""Exact 1-D box/pixel overlap weights shared by the forward box splatters."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from ..device import _as_tensor


_POINT_WIDTH = 1.0e-6


def box_overlap_taps(center: Tensor, width: Tensor | float) -> tuple[Tensor, Tensor]:
    """Overlap fractions of boxes ``[center - width/2, center + width/2]`` with pixels.

    Pixel ``i`` covers ``[i - 0.5, i + 0.5]``.  Returns ``(first, weights)`` where
    ``first`` is the int64 index of the first overlapped pixel and ``weights`` has
    shape ``(n_taps, *center.shape)``; tap ``k`` is pixel ``first + k`` and holds
    ``overlap_length / width`` so the taps of an unclipped box sum to one.
    Footprints narrower than 1e-6 pixel are point masses.
    ``n_taps = ceil(max(width)) + 1`` is the most pixels a box can touch.
    """
    width_t = _as_tensor(width, dtype=center.dtype, device=center.device)
    width_t = width_t.expand_as(center)
    # A footprint narrower than 1e-6 pixel is a point mass: all weight on the pixel that
    # contains the centre (the limit of the box overlap fractions, free of the float32
    # cancellation a 1e-6-wide interval would suffer).
    point = width_t < _POINT_WIDTH
    width_t = width_t.clamp_min(_POINT_WIDTH)
    n_taps = int(math.ceil(float(width_t.max()))) + 1 if center.numel() else 1
    lo = center - 0.5 * width_t + 0.5
    hi = center + 0.5 * width_t + 0.5
    first = torch.floor(lo)
    edges = first.unsqueeze(0) + torch.arange(
        n_taps + 1, dtype=center.dtype, device=center.device
    ).reshape((-1,) + (1,) * center.ndim)
    overlap = (torch.minimum(edges[1:], hi) - torch.maximum(edges[:-1], lo)).clamp_min(
        0.0
    )
    weights = overlap / width_t
    if bool(point.any()):
        first_tap = torch.zeros_like(weights)
        first_tap[0] = 1.0
        weights = torch.where(point.unsqueeze(0), first_tap, weights)
    return first.to(torch.int64), weights
