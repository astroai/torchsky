"""Shared scalar helpers for WCS projection rotation code."""

from __future__ import annotations

import torch
from torch import Tensor


_HAS_TORCH_SINCOS = hasattr(torch, "sincos")


def _sincos(x: Tensor) -> tuple[Tensor, Tensor]:
    if _HAS_TORCH_SINCOS:
        return torch.sincos(x)  # type: ignore[attr-defined]
    return torch.sin(x), torch.cos(x)


def _wrap_lon360(angle: Tensor) -> Tensor:
    """Wrap longitudes to [0, 360)."""
    return torch.remainder(angle, 360.0)


def _wrap_lon180(angle: Tensor) -> Tensor:
    """Wrap longitudes to [-180, 180)."""
    return torch.remainder(angle + 180.0, 360.0) - 180.0

