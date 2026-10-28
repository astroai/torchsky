"""Catalog streaming, indexing, and crossmatching helpers."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from .index import HealpixIndex
from .index import cap_covering_pixels
from .index import nside_for_cap
from .catalog import Catalog
from .core import (
    CatalogStreamDataset,
    CrossmatchResult,
    crossmatch_sky,
    crossmatch_sky_from_arrow,
    crossmatch_sky_streamed,
)

_PHOTOMETRY_EXPORTS = {
    "BackgroundModel",
    "BlendFitDiagnostics",
    "BlockSparseCovariance",
    "ForwardPhotometryResult",
    "PhotometrySolver",
    "SourceGrouping",
    "build_source_groups",
    "fit_forced_photometry",
}

_DEBLENDING_EXPORTS = {
    "FactorizationBlendDiagnostics",
    "FailurePolicy",
    "FitAdequacyDiagnostics",
    "KnownComponentFitError",
    "MorphologySpectrumDeblendResult",
    "ParametricBlendDiagnostics",
    "ParametricBounds",
    "ParametricDeblendResult",
    "fit_morphology_spectrum_deblending",
    "fit_parametric_deblending",
}


def __getattr__(name: str) -> Any:
    if name in _PHOTOMETRY_EXPORTS:
        module = ".photometry"
    elif name in _DEBLENDING_EXPORTS:
        module = ".deblending"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def crossmatch(
    left: Any,
    right: Any,
    radius_arcsec_or_right_lon: Any = None,
    right_lat: Any = None,
    radius_arcsec: float | Tensor | None = None,
    *,
    find: str = "best",
    streamed: bool = False,
    **kwargs: Any,
) -> CrossmatchResult:
    """Unified sky crossmatch entry point.

    Supports:
    - ``crossmatch(cat_left, cat_right, radius_arcsec, ...)`` for :class:`Catalog` objects
    - ``crossmatch(left_lon, left_lat, right_lon, right_lat, radius_arcsec=..., ...)`` for coordinate tensors
    - ``crossmatch((left_lon, left_lat), (right_lon, right_lat), radius_arcsec, ...)``
    - ``crossmatch(left_arrow, right_arrow, radius_arcsec, streamed=False, ...)`` for Arrow tables/paths
    """
    if isinstance(left, Catalog) and isinstance(right, Catalog):
        rad = radius_arcsec if radius_arcsec is not None else radius_arcsec_or_right_lon
        if rad is None:
            raise ValueError("radius_arcsec is required")
        return left.crossmatch(right, rad, find=find, **kwargs)

    if right_lat is not None:
        rad = radius_arcsec
        if rad is None:
            raise ValueError(
                "radius_arcsec is required when 4 coordinate arrays are passed"
            )
        return crossmatch_sky(
            left,
            right,
            radius_arcsec_or_right_lon,
            right_lat,
            radius_arcsec=rad,
            find=find,
            **kwargs,
        )

    if (
        isinstance(left, tuple)
        and len(left) == 2
        and isinstance(right, tuple)
        and len(right) == 2
    ):
        rad = radius_arcsec if radius_arcsec is not None else radius_arcsec_or_right_lon
        if rad is None:
            raise ValueError("radius_arcsec is required")
        return crossmatch_sky(
            left[0],
            left[1],
            right[0],
            right[1],
            radius_arcsec=rad,
            find=find,
            **kwargs,
        )

    rad = radius_arcsec if radius_arcsec is not None else radius_arcsec_or_right_lon
    if rad is None:
        raise ValueError("radius_arcsec is required")
    if streamed:
        return crossmatch_sky_streamed(
            left, right, radius_arcsec=float(rad), find=find, **kwargs
        )
    return crossmatch_sky_from_arrow(
        left, right, radius_arcsec=float(rad), find=find, **kwargs
    )


__all__ = [
    "Catalog",
    "CatalogStreamDataset",
    "BackgroundModel",
    "BlendFitDiagnostics",
    "BlockSparseCovariance",
    "CrossmatchResult",
    "ForwardPhotometryResult",
    "FactorizationBlendDiagnostics",
    "FailurePolicy",
    "FitAdequacyDiagnostics",
    "HealpixIndex",
    "KnownComponentFitError",
    "MorphologySpectrumDeblendResult",
    "ParametricBlendDiagnostics",
    "ParametricBounds",
    "ParametricDeblendResult",
    "PhotometrySolver",
    "SourceGrouping",
    "build_source_groups",
    "cap_covering_pixels",
    "crossmatch",
    "crossmatch_sky",
    "crossmatch_sky_from_arrow",
    "crossmatch_sky_streamed",
    "fit_forced_photometry",
    "fit_morphology_spectrum_deblending",
    "fit_parametric_deblending",
    "nside_for_cap",
]
