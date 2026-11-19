"""Catalog streaming, indexing, and crossmatching helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .index import HealpixIndex
from .index import cap_covering_pixels
from .index import nside_for_cap
from .astrometry import Astrometry, MotionPrior, propagate_astrometry
from .association import (
    Association,
    HistogramRatio,
    NWayAssociation,
    assign_one_to_one,
    associate,
    associate_many,
)
from .catalog import Catalog
from .partition import PartitionPlan, healpix29, margin_arcsec
from .multi import (
    ManyMatchResult,
    SelfMatchResult,
    crossmatch_exact,
    crossmatch_many,
    crossmatch_self,
    join_rows,
)
from .core import (
    CatalogStreamDataset,
    CrossmatchResult,
    crossmatch_sky,
)
from .sky_index import (
    NearestResult,
    SkyIndex,
    count_within,
    match_nearest,
    match_within,
)

if TYPE_CHECKING:
    from typing import Any

    from torch import Tensor

del TYPE_CHECKING

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
    **kwargs: Any,
) -> CrossmatchResult:
    """Unified sky crossmatch entry point.

    Supports:
    - ``crossmatch(cat_left, cat_right, radius_arcsec, ...)`` for :class:`Catalog` objects
    - ``crossmatch(left_lon, left_lat, right_lon, right_lat, radius_arcsec=..., ...)`` for coordinate tensors
    - ``crossmatch((left_lon, left_lat), (right_lon, right_lat), radius_arcsec, ...)``

    Tables and files are read by the caller (torchfits, pyarrow, polars) and
    passed in as tensors.
    """
    if isinstance(left, Catalog) and isinstance(right, Catalog):
        rad = radius_arcsec if radius_arcsec is not None else radius_arcsec_or_right_lon
        if rad is None and not (
            kwargs.get("left_err_arcsec") is not None
            or kwargs.get("right_err_arcsec") is not None
        ):
            raise ValueError("radius_arcsec is required")
        return left.crossmatch(right, rad, find=find, **kwargs)

    if right_lat is not None:
        rad = radius_arcsec
        if rad is None:
            raise ValueError(
                "radius_arcsec is required when 4 coordinate arrays are passed"
            )
        # Not the Catalog pair handled above: these are coordinate arrays.
        left_lon: Any = left
        right_lon: Any = right
        return crossmatch_sky(
            left_lon,
            right_lon,
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

    raise TypeError(
        "crossmatch takes two Catalogs, two (lon, lat) tuples, or four coordinate arrays"
    )


__all__ = [
    "Association",
    "Astrometry",
    "HistogramRatio",
    "assign_one_to_one",
    "associate",
    "associate_many",
    "NWayAssociation",
    "MotionPrior",
    "propagate_astrometry",
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
    "ManyMatchResult",
    "PartitionPlan",
    "NearestResult",
    "MorphologySpectrumDeblendResult",
    "ParametricBlendDiagnostics",
    "ParametricBounds",
    "ParametricDeblendResult",
    "PhotometrySolver",
    "SelfMatchResult",
    "SkyIndex",
    "SourceGrouping",
    "build_source_groups",
    "cap_covering_pixels",
    "count_within",
    "crossmatch",
    "crossmatch_exact",
    "crossmatch_many",
    "crossmatch_self",
    "crossmatch_sky",
    "fit_forced_photometry",
    "fit_morphology_spectrum_deblending",
    "fit_parametric_deblending",
    "healpix29",
    "join_rows",
    "margin_arcsec",
    "match_nearest",
    "match_within",
    "nside_for_cap",
]
