"""Default compute device selection for tensor pipelines.

Environment:

* ``TORCHSKY_DEVICE`` — if set (e.g. ``cuda``, ``cuda:1``, ``cpu``), overrides the
  automatic default for APIs that use :func:`default_compute_device`.
"""

from __future__ import annotations

import os

import torch
from torch import nn


def default_compute_device() -> torch.device:
    """Prefer CUDA when available; otherwise CPU.

    If ``TORCHSKY_DEVICE`` is set to a non-empty string, it is parsed as a
    :class:`torch.device` and returned.
    """
    raw = os.environ.get("TORCHSKY_DEVICE", "").strip()
    if raw:
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def module_compute_device(module: nn.Module) -> torch.device:
    """Device of the first parameter or buffer, else :func:`default_compute_device`."""
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return default_compute_device()


def resolve_fit_device(
    model: nn.Module,
    device: torch.device | str | None,
) -> torch.device:
    """Pick where to run training for ``fit_field_model``-style loops.

    * Explicit ``device`` always wins.
    * Otherwise, if any parameter or buffer is already off-CPU (e.g. ``cuda:1``),
      that device is kept.
    * Otherwise :func:`default_compute_device` (CUDA when available, else CPU).
    """
    if device is not None:
        return torch.device(device)
    for tensor in list(model.parameters()) + list(model.buffers()):
        t = tensor.device.type
        if t not in ("cpu", "meta"):
            return tensor.device
    return default_compute_device()
