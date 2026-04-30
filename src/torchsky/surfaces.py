"""Code-level API stability tiers for torchsky public surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class SurfaceTier(str, Enum):
    """Supported stability tiers for exported torchsky APIs."""

    CORE = "core"
    COMPAT = "compat"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class APISurface:
    """A declared API surface and its intended stability contract."""

    name: str
    tier: SurfaceTier
    reason: str


_SURFACES: dict[str, APISurface] = {
    "torchsky.read": APISurface(
        "torchsky.read",
        SurfaceTier.CORE,
        "primary FITS image/table read entry point",
    ),
    "torchsky.write": APISurface(
        "torchsky.write",
        SurfaceTier.CORE,
        "primary FITS image/table write entry point",
    ),
    "torchsky.open": APISurface(
        "torchsky.open",
        SurfaceTier.CORE,
        "multi-HDU handle API",
    ),
    "torchsky.table": APISurface(
        "torchsky.table",
        SurfaceTier.CORE,
        "Arrow-oriented FITS table API",
    ),
    "torchsky.wcs.WCS": APISurface(
        "torchsky.wcs.WCS",
        SurfaceTier.CORE,
        "torch-native WCS transform object",
    ),
    "torchsky.sphere": APISurface(
        "torchsky.sphere",
        SurfaceTier.CORE,
        "native spherical geometry, HEALPix, sparse, and spectral primitives",
    ),
    "torchsky.sphere.compat": APISurface(
        "torchsky.sphere.compat",
        SurfaceTier.COMPAT,
        "healpy/hpgeom-shaped compatibility facade over native primitives",
    ),
    "torchsky.sphere.healpy": APISurface(
        "torchsky.sphere.healpy",
        SurfaceTier.COMPAT,
        "healpy-shaped import path over native primitives",
    ),
    "torchsky.map.coadd_image_plane": APISurface(
        "torchsky.map.coadd_image_plane",
        SurfaceTier.CORE,
        "native image-plane coaddition API",
    ),
    "torchsky.map.stack_map_product": APISurface(
        "torchsky.map.stack_map_product",
        SurfaceTier.CORE,
        "declared map-product execution API",
    ),
    "torchsky.map.field": APISurface(
        "torchsky.map.field",
        SurfaceTier.EXPERIMENTAL,
        "continuous-field/INR fitting and training surface",
    ),
    "torchsky.map.fit_field_model": APISurface(
        "torchsky.map.fit_field_model",
        SurfaceTier.EXPERIMENTAL,
        "INR optimizer entry point with evolving science contracts",
    ),
    "torchsky.viz": APISurface(
        "torchsky.viz",
        SurfaceTier.EXPERIMENTAL,
        "browser visualization server and renderers",
    ),
}

API_SURFACES = MappingProxyType(_SURFACES)


def get_api_surface(name: str) -> APISurface:
    """Return stability metadata for a declared public API surface."""
    try:
        return API_SURFACES[name]
    except KeyError as exc:
        raise KeyError(f"unknown torchsky API surface {name!r}") from exc


def api_surfaces_by_tier(tier: SurfaceTier | str) -> tuple[APISurface, ...]:
    """Return declared surfaces for one stability tier."""
    tier_value = SurfaceTier(tier)
    return tuple(surface for surface in API_SURFACES.values() if surface.tier is tier_value)


__all__ = [
    "APISurface",
    "API_SURFACES",
    "SurfaceTier",
    "api_surfaces_by_tier",
    "get_api_surface",
]
