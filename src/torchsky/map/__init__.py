"""Lean mapmaking facade.

Implementation modules remain available under ``torchsky.map.*``. The flat
``torchsky.map`` namespace is intentionally limited to promoted coadd/product
entry points and result/specification objects.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from . import plan as _plan_module  # noqa: F401 — pre-populate sys.modules["torchsky.map.plan"]


def plan(*args: Any, **kwargs: Any) -> Any:
    """Plan a 2D target WCS grid covering a sequence of observations.

    Args:
        *args: Observation sequence accepted by :func:`torchsky.map.planning.plan`.
        **kwargs: Projection, scale, rotation, footprint, and device controls.

    Returns:
        Target plan containing the output WCS, shape, and footprint metadata.

    Examples:
        >>> from torchsky.map import plan
        >>> callable(plan)
        True
    """
    from .planning import plan as _plan_fn

    return _plan_fn(*args, **kwargs)


_EXPORTS: dict[str, tuple[str, str]] = {
    # Public facade: one entry per name in ``__all__``.
    "Observation": ("torchsky.map.observation", "Observation"),
    "IFUObservation": ("torchsky.map.observation", "IFUObservation"),
    "ImageObservation": ("torchsky.map.observation", "ImageObservation"),
    "Target": ("torchsky.map.observation", "Target"),
    "CubeTarget": ("torchsky.map.observation", "CubeTarget"),
    "ResampleConfig": ("torchsky.map.observation", "ResampleConfig"),
    "SkyConfig": ("torchsky.map.observation", "SkyConfig"),
    "OutlierConfig": ("torchsky.map.observation", "OutlierConfig"),
    "DQPolicy": ("torchsky.map.quality", "DQPolicy"),
    "DQRejectionLog": ("torchsky.map.quality", "DQRejectionLog"),
    "QualityProduct": ("torchsky.map.quality", "QualityProduct"),
    "StackConfig": ("torchsky.map.observation", "StackConfig"),
    "MapResult": ("torchsky.map.observation", "MapResult"),
    "CubeResult": ("torchsky.map.observation", "CubeResult"),
    "MultibandResult": ("torchsky.map.observation", "MultibandResult"),
    "HealpixResult": ("torchsky.map.observation", "HealpixResult"),
    "ApertureUncertainty": ("torchsky.map.observation", "ApertureUncertainty"),
    "AdaptiveResampleConfig": (
        "torchsky.map.operators.adaptive",
        "AdaptiveResampleConfig",
    ),
    "ProductSpec": ("torchsky.map.product", "ProductSpec"),
    "BandpassSpec": ("torchsky.map.product", "BandpassSpec"),
    "ProductResult": ("torchsky.map.product", "ProductResult"),
    "UncertaintyProduct": ("torchsky.map.product", "UncertaintyProduct"),
    "TargetRenderSpec": ("torchsky.map.target", "TargetRenderSpec"),
    "plan": ("torchsky.map.planning", "plan"),
    "plan_cube": ("torchsky.map.planning", "plan_cube"),
    "plan_product": ("torchsky.map.product", "plan_map_product"),
    "resample": ("torchsky.map.image_plane_coadd", "resample"),
    "resample_healpix": ("torchsky.map.healpix_coadd", "resample_healpix"),
    "blot": ("torchsky.map.image_plane_coadd", "blot"),
    "detect_outliers": ("torchsky.map.sky_matching", "detect_outliers"),
    "match_sky": ("torchsky.map.sky_matching", "match_sky"),
    "coadd": ("torchsky.map.image_plane_coadd", "coadd"),
    "coadd_multiband": ("torchsky.map.image_plane_coadd", "coadd_multiband"),
    "coadd_image_plane": ("torchsky.map.image_plane_coadd", "coadd_image_plane"),
    "coadd_cube": ("torchsky.map.cube", "coadd_cube"),
    "plan_ifu_cube_response": ("torchsky.map.ifu", "plan_ifu_cube_response"),
    "build_ifu_cube": ("torchsky.map.ifu", "build_ifu_cube"),
    "coadd_healpix": ("torchsky.map.healpix_coadd", "coadd_healpix"),
    "stack_product": ("torchsky.map.product", "stack_product"),
    "aperture_uncertainty": (
        "torchsky.map.image_plane_coadd",
        "aperture_uncertainty",
    ),
    "adaptive_reprojection_response": (
        "torchsky.map.operators.adaptive",
        "adaptive_reprojection_response",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    # Keep compatibility with internal module access (e.g. torchsky.map.field)
    if name in {
        "field",
        "product",
        "target",
        "solvers",
        "operators",
        "exposure_pose",
    }:
        module = import_module(f"torchsky.map.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_EXPORTS))


if TYPE_CHECKING:
    from .cube import (
        coadd_cube as coadd_cube,
    )
    from .healpix_coadd import (
        coadd_healpix as coadd_healpix,
        resample_healpix as resample_healpix,
    )
    from .ifu import (
        build_ifu_cube as build_ifu_cube,
        plan_ifu_cube_response as plan_ifu_cube_response,
    )
    from .image_plane_coadd import (
        aperture_uncertainty as aperture_uncertainty,
        blot as blot,
        coadd as coadd,
        coadd_image_plane as coadd_image_plane,
        coadd_multiband as coadd_multiband,
        resample as resample,
    )
    from .observation import (
        ApertureUncertainty as ApertureUncertainty,
        CubeResult as CubeResult,
        CubeTarget as CubeTarget,
        HealpixResult as HealpixResult,
        IFUObservation as IFUObservation,
        ImageObservation as ImageObservation,
        MapResult as MapResult,
        MultibandResult as MultibandResult,
        Observation as Observation,
        OutlierConfig as OutlierConfig,
        ResampleConfig as ResampleConfig,
        SkyConfig as SkyConfig,
        StackConfig as StackConfig,
        Target as Target,
    )
    from .operators.adaptive import (
        AdaptiveResampleConfig as AdaptiveResampleConfig,
        adaptive_reprojection_response as adaptive_reprojection_response,
    )
    from .quality import (
        DQPolicy as DQPolicy,
        DQRejectionLog as DQRejectionLog,
        QualityProduct as QualityProduct,
    )
    from .product import (
        BandpassSpec as BandpassSpec,
        ProductResult as ProductResult,
        ProductSpec as ProductSpec,
        UncertaintyProduct as UncertaintyProduct,
        plan_map_product as plan_product,
        stack_product as stack_product,
    )
    from .planning import plan as plan, plan_cube as plan_cube
    from .sky_matching import (
        detect_outliers as detect_outliers,
        match_sky as match_sky,
    )
    from .target import (
        TargetRenderSpec as TargetRenderSpec,
    )

__all__ = [
    "Observation",
    "IFUObservation",
    "ImageObservation",
    "Target",
    "CubeTarget",
    "ResampleConfig",
    "SkyConfig",
    "OutlierConfig",
    "DQPolicy",
    "DQRejectionLog",
    "QualityProduct",
    "StackConfig",
    "MapResult",
    "CubeResult",
    "MultibandResult",
    "HealpixResult",
    "ApertureUncertainty",
    "AdaptiveResampleConfig",
    "ProductSpec",
    "BandpassSpec",
    "ProductResult",
    "UncertaintyProduct",
    "TargetRenderSpec",
    "plan",
    "plan_cube",
    "plan_product",
    "resample",
    "resample_healpix",
    "blot",
    "detect_outliers",
    "match_sky",
    "coadd",
    "coadd_multiband",
    "coadd_image_plane",
    "coadd_cube",
    "plan_ifu_cube_response",
    "build_ifu_cube",
    "coadd_healpix",
    "stack_product",
    "aperture_uncertainty",
    "adaptive_reprojection_response",
]
