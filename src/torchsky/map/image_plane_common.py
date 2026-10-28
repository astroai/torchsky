"""Image-plane-first mapmaking primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torchfits import read

from ..device import default_compute_device
from ..sphere.moc import MOC
from ..wcs import WCS
from .io_utils import (
    _observation_header_for_area_scaling,
)
from .io_utils import (
    load_observation_image_and_header as _io_load_observation_image_and_header,
)

# Header probe for area scaling: recoverable failures when reading metadata only.
from .observation import (
    CubeTarget,
    ImageObservation,
    PlannedImagePlaneCubeTarget,
    PlannedImagePlaneTarget,
    Target,
)
from .operators.config import (
    MapResamplingKernel,
    MapStackingMethod,
    MapValueSemantics,
    validate_map_resampling_kernel,
    validate_map_stacking_method,
    validate_map_value_semantics,
)
from .windowing import (
    _LARGE_TAN_WINDOWING_EDGE_SAMPLES,  # noqa: F401
    _LARGE_TAN_WINDOWING_MARGIN_PIXELS,  # noqa: F401
    _LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS,  # noqa: F401
    )



from .calibration import (  # noqa: F401 - re-exported helpers
    _accumulate_variance_weight,
    _apply_observation_calibration,
    _context_provenance,
    _effective_observation_photometric_scale,
    _linear_inverse_variance,
    _load_observation_weight_map,
    _local_noise_correlation_factor,
    _observation_operator_provenance,
    _set_context_bit,
    _should_apply_surface_brightness_area_scaling,
)

def load_observation_image_and_header(
    obs: object,
    device: str | torch.device,
) -> tuple[Tensor, dict]:
    return _io_load_observation_image_and_header(obs, device, read_fn=read)


def _resolve_map_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return default_compute_device()


@dataclass
class _ProjectedImagePlaneObservation:
    numer: Tensor
    weight: Tensor
    variance_weight: Tensor | None
    overlap: Tensor
    moc: MOC | None
    target_window: tuple[int, int, int, int] | None = None


def _is_splat_kernel(kernel: MapResamplingKernel) -> bool:
    return kernel in {
        "splat_point",
        "splat_box",
        "drizzle_turbo",
        "drizzle_gaussian",
        "drizzle_lanczos2",
        "drizzle_lanczos3",
    }


def _convert_world_between_wcs(
    lon: Tensor,
    lat: Tensor,
    from_wcs: WCS,
    to_wcs: WCS,
) -> tuple[Tensor, Tensor]:
    from_frame = getattr(from_wcs, "celestial_frame", "ICRS")
    to_frame = getattr(to_wcs, "celestial_frame", "ICRS")
    from_eq = getattr(from_wcs, "celestial_equinox", None)
    to_eq = getattr(to_wcs, "celestial_equinox", None)
    if from_frame == to_frame and (
        from_eq == to_eq or from_frame in {"ICRS", "GALACTIC", "ECLIPTIC"}
    ):
        return lon, lat
    orig_shape = lon.shape
    lon_c, lat_c = from_wcs.convert_celestial(
        lon.reshape(-1),
        lat.reshape(-1),
        to_frame,
        to_equinox=to_eq,
    )
    return lon_c.reshape(orig_shape), lat_c.reshape(orig_shape)


def _splat_for_interpolation(kernel: MapResamplingKernel) -> MapResamplingKernel | None:
    """Return equivalent splat kernel for interpolation kernels, or None."""
    if kernel == "nearest":
        return "splat_point"
    if kernel == "bilinear":
        return "splat_box"
    return None


def _validated_map_resampling_kernel(
    resampling_kernel: str,
) -> MapResamplingKernel:
    return validate_map_resampling_kernel(resampling_kernel)


def _validated_map_value_semantics(
    value_semantics: str,
) -> MapValueSemantics:
    return validate_map_value_semantics(value_semantics)


def _validated_drizzle_pixfrac(drizzle_pixfrac: float) -> float:
    pixfrac = float(drizzle_pixfrac)
    if not math.isfinite(pixfrac) or pixfrac <= 0.0:
        raise ValueError("drizzle_pixfrac must be a finite positive float")
    return pixfrac


def _validated_map_stacking_method(
    stacking_method: str,
) -> MapStackingMethod:
    return validate_map_stacking_method(stacking_method)


def _true_image_shape(header: dict) -> tuple[int, int]:
    """Return (height, width) of the image described by *header*.

    Compressed FITS stores the true dimensions in ``ZNAXIS1``/``ZNAXIS2``;
    uncompressed images use the standard ``NAXIS1``/``NAXIS2`` cards.
    """
    if header.get("ZCMPTYPE") is not None:
        nx = int(header.get("ZNAXIS1", header.get("NAXIS1", 0)))
        ny = int(header.get("ZNAXIS2", header.get("NAXIS2", 0)))
    else:
        nx = int(header.get("NAXIS1", 0))
        ny = int(header.get("NAXIS2", 0))
    return (ny, nx)


def _observation_header_for_footprint(
    obs: ImageObservation,
    device: torch.device | str | None = None,
) -> dict:
    """Read the FITS header for *obs*, correcting compressed NAXIS cards."""

    hdr = _observation_header_for_area_scaling(obs)
    ny, nx = _true_image_shape(hdr)
    if ny > 0 and nx > 0:
        hdr["NAXIS1"] = nx
        hdr["NAXIS2"] = ny
    return hdr


def _coerce_target_header(
    target: Target | CubeTarget | Mapping[str, Any] | WCS | None,
) -> dict[str, Any] | None:
    if target is None:
        return None
    if isinstance(target, (PlannedImagePlaneTarget, PlannedImagePlaneCubeTarget)):
        return dict(target.header)
    if isinstance(target, WCS):
        return target.to_header()
    if isinstance(target, Mapping):
        return dict(target)
    raise TypeError(
        f"target must be a Target, CubeTarget, header dict, WCS, or None; got {type(target).__name__}"
    )
