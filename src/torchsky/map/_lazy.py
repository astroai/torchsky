"""Lazy cross-module lookups for the map package.

Several map modules need image-plane primitives but are imported by
``image_plane`` itself; a function-local import avoids an import cycle.
"""

from __future__ import annotations

from types import ModuleType


def image_plane() -> ModuleType:
    """Return the imported :mod:`torchsky.map.image_plane` module."""
    from . import image_plane as module

    return module
