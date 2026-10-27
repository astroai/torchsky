"""Native extension loader surface."""

from __future__ import annotations

from . import _cpp_ext as _ext


def selected_cpu_isa() -> str:
    return str(_ext.selected_cpu_isa())


__all__ = ["selected_cpu_isa"]
