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
    """Plan a 2D target WCS grid covering a sequence of observations."""
    from .image_plane import plan as _plan_fn

    return _plan_fn(*args, **kwargs)


_EXPORTS: dict[str, tuple[str, str]] = {
    # Public facade: one entry per name in ``__all__``.
    "Observation": ("torchsky.map.observation", "Observation"),
    "Target": ("torchsky.map.observation", "Target"),
    "CubeTarget": ("torchsky.map.observation", "CubeTarget"),
    "ResampleConfig": ("torchsky.map.observation", "ResampleConfig"),
    "SkyConfig": ("torchsky.map.observation", "SkyConfig"),
    "OutlierConfig": ("torchsky.map.observation", "OutlierConfig"),
    "StackConfig": ("torchsky.map.observation", "StackConfig"),
    "MapResult": ("torchsky.map.observation", "MapResult"),
    "CubeResult": ("torchsky.map.observation", "CubeResult"),
    "MultibandResult": ("torchsky.map.observation", "MultibandResult"),
    "HealpixResult": ("torchsky.map.observation", "HealpixResult"),
    "ApertureUncertainty": ("torchsky.map.observation", "ApertureUncertainty"),
    "ProductSpec": ("torchsky.map.product", "ProductSpec"),
    "BandpassSpec": ("torchsky.map.product", "BandpassSpec"),
    "ProductResult": ("torchsky.map.product", "ProductResult"),
    "UncertaintyProduct": ("torchsky.map.product", "UncertaintyProduct"),
    "TargetRenderSpec": ("torchsky.map.target", "TargetRenderSpec"),
    "plan": ("torchsky.map.image_plane", "plan"),
    "plan_cube": ("torchsky.map.image_plane", "plan_cube"),
    "plan_product": ("torchsky.map.product", "plan_map_product"),
    "resample": ("torchsky.map.image_plane", "resample"),
    "resample_healpix": ("torchsky.map.healpix_coadd", "resample_healpix"),
    "blot": ("torchsky.map.image_plane", "blot"),
    "detect_outliers": ("torchsky.map.image_plane", "detect_outliers"),
    "match_sky": ("torchsky.map.image_plane", "match_sky"),
    "coadd": ("torchsky.map.image_plane", "coadd"),
    "coadd_multiband": ("torchsky.map.image_plane", "coadd_multiband"),
    "coadd_cube": ("torchsky.map.cube", "coadd_cube"),
    "coadd_healpix": ("torchsky.map.healpix_coadd", "coadd_healpix"),
    "stack_product": ("torchsky.map.product", "stack_product"),
    "aperture_uncertainty": ("torchsky.map.image_plane", "aperture_uncertainty"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name)
        if module_name == "torchsky.map.plan":
            # Python binds the submodule `plan` into package globals() on import;
            # restore the public `plan` function from `torchsky.map.image_plane`.
            globals()["plan"] = getattr(
                import_module("torchsky.map.image_plane"), "plan"
            )
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    # Keep compatibility with internal module access (e.g. torchsky.map.field)
    if name in {
        "field",
        "image_plane",
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
    from .image_plane import (
        aperture_uncertainty as aperture_uncertainty,
        blot as blot,
        coadd as coadd,
        coadd_multiband as coadd_multiband,
        detect_outliers as detect_outliers,
        match_sky as match_sky,
        plan as plan,
        plan_cube as plan_cube,
        resample as resample,
    )
    from .observation import (
        ApertureUncertainty as ApertureUncertainty,
        CubeResult as CubeResult,
        CubeTarget as CubeTarget,
        HealpixResult as HealpixResult,
        MapResult as MapResult,
        MultibandResult as MultibandResult,
        Observation as Observation,
        OutlierConfig as OutlierConfig,
        ResampleConfig as ResampleConfig,
        SkyConfig as SkyConfig,
        StackConfig as StackConfig,
        Target as Target,
    )
    from .product import (
        BandpassSpec as BandpassSpec,
        ProductResult as ProductResult,
        ProductSpec as ProductSpec,
        UncertaintyProduct as UncertaintyProduct,
        plan_map_product as plan_product,
        stack_product as stack_product,
    )
    from .target import (
        TargetRenderSpec as TargetRenderSpec,
    )

__all__ = [
    "Observation",
    "Target",
    "CubeTarget",
    "ResampleConfig",
    "SkyConfig",
    "OutlierConfig",
    "StackConfig",
    "MapResult",
    "CubeResult",
    "MultibandResult",
    "HealpixResult",
    "ApertureUncertainty",
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
    "coadd_cube",
    "coadd_healpix",
    "stack_product",
    "aperture_uncertainty",
]

