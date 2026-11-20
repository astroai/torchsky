"""Native extension loader surface."""

from __future__ import annotations

import torch as _torch

from . import _cpp_ext as _ext
from . import _native_ops as _native_ops  # noqa: F401 - registers torch.ops.torchsky

del _torch


def selected_cpu_isa() -> str:
    return str(_ext.selected_cpu_isa())


__all__ = ["selected_cpu_isa"]
