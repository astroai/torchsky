"""Shared working-dtype policy for tensor-native transforms.

MPS has no float64 support in the kernels torchsky uses, so device transforms
work in float32 there and float64 everywhere else. Keep this policy in one
place instead of redefining it in every domain module.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def work_dtype(device: torch.device) -> torch.dtype:
    """Return the working floating dtype for ``device``."""
    return torch.float32 if device.type == "mps" else torch.float64


def work_complex_dtype(device: torch.device) -> torch.dtype:
    """Return the working complex dtype for ``device``."""
    return torch.complex64 if device.type == "mps" else torch.complex128


def as_work_tensor(value: Any) -> Tensor:
    """Convert ``value`` to a tensor in the device's working dtype."""
    if isinstance(value, Tensor):
        return value.to(dtype=work_dtype(value.device))
    return torch.as_tensor(value, dtype=torch.float64)
