"""Thin deprecated aliases for public names renamed after a release.

A renamed name that shipped in a release keeps one alias for one minor
release. The alias resolves to the new object and emits ``DeprecationWarning``
on each access; it is never listed in ``__all__`` or the API manifest.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any
import warnings

REMOVED_IN = "0.7"


def deprecated_attribute(
    module: str, name: str, target: tuple[str, str], *, stacklevel: int = 3
) -> Any:
    """Warn that ``module.name`` is deprecated and return its replacement."""
    target_module, target_name = target
    warnings.warn(
        f"{module}.{name} is deprecated and will be removed in torchsky "
        f"{REMOVED_IN}; use {target_module}.{target_name}",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return getattr(import_module(target_module), target_name)


def deprecated_module_getattr(
    module: str, aliases: Mapping[str, tuple[str, str]]
) -> Callable[[str], Any]:
    """Return a module ``__getattr__`` that serves only deprecated aliases."""

    def __getattr__(name: str) -> Any:
        target = aliases.get(name)
        if target is None:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        return deprecated_attribute(module, name, target)

    return __getattr__
