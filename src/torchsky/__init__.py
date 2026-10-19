"""Lean public API for torchsky.

The package root intentionally stays light: importing :mod:`torchsky` must not
load tensor runtimes, NumPy, compiled extensions, map models, or optional
astronomy ecosystem packages. Domain APIs live under their own namespaces and
are imported on first access.

FITS images, HDUs, headers, tables, caches, and datasets are owned by
``torchfits`` and are intentionally not re-exported here.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.4.0"

_NAMESPACES: dict[str, str] = {
    "wcs": "torchsky.wcs",
    "sphere": "torchsky.sphere",
    "spectral": "torchsky.spectral",
    "spectrophotometry": "torchsky.spectrophotometry",
    "redshift": "torchsky.redshift",
    "log_lambda": "torchsky.log_lambda",
    "extinction": "torchsky.extinction",
    "cosmology": "torchsky.cosmology",
    "map": "torchsky.map",
    "scene": "torchsky.scene",
    "field": "torchsky.field",
    "transforms": "torchsky.transforms",
    "encoders": "torchsky.encoders",
    "catalogs": "torchsky.catalogs",
    "coords": "torchsky.coords",
    "quality": "torchsky.quality",
}

__all__ = ("WCS", *_NAMESPACES)


def __getattr__(name: str) -> Any:
    if name in _NAMESPACES:
        module = import_module(_NAMESPACES[name])
        globals()[name] = module
        return module
    if name == "WCS":
        from .wcs import WCS

        globals()[name] = WCS
        return WCS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from . import (
        catalogs as catalogs,
        coords as coords,
        cosmology as cosmology,
        encoders as encoders,
        extinction as extinction,
        field as field,
        log_lambda as log_lambda,
        map as map,
        quality as quality,
        redshift as redshift,
        scene as scene,
        spectral as spectral,
        spectrophotometry as spectrophotometry,
        sphere as sphere,
        transforms as transforms,
        wcs as wcs,
    )
    from .wcs import WCS as WCS
