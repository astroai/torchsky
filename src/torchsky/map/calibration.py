"""Observation calibration, variance weighting, and provenance helpers.

Split out of the former image-plane monolith; these are leaf helpers shared by
the coadd, sky-matching, and aperture paths.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from .observation import ImageObservation
from ..wcs import WCS
from .operators.mask_weight import load_image_observation_weight_map
from .operators.radiometry import apply_image_observation_calibration
from .io_utils import _observation_header_for_area_scaling
from .operators.execution import is_plain_tan_header, wcs_has_distortion
from .operators.pixel_area import representative_pixel_area_sr
from .operators.config import (
    MapResamplingKernel,
    MapStackingMethod,
    MapValueSemantics,
)


def _accumulate_variance_weight(
    target: Tensor,
    contribution: Tensor,
    target_window: tuple[int, int, int, int] | None,
) -> None:
    if target_window is None:
        target.add_(contribution)
        return
    y0, y1, x0, x1 = target_window
    if y1 > y0 and x1 > x0:
        target[..., y0:y1, x0:x1].add_(contribution)


def _set_context_bit(
    context: Tensor,
    observation_index: int,
    support: Tensor,
    target_window: tuple[int, int, int, int] | None,
) -> None:
    word = int(observation_index) // 32
    bit = int(observation_index) % 32
    encoded = support.to(dtype=torch.int64) * (1 << bit)
    plane = context[word]
    if target_window is None:
        plane.bitwise_or_(encoded)
        return
    y0, y1, x0, x1 = target_window
    if y1 > y0 and x1 > x0:
        plane[y0:y1, x0:x1].bitwise_or_(encoded)


def _observation_operator_provenance(
    observation: ImageObservation,
) -> dict[str, Any]:
    return {
        "path": observation.path,
        "hdu": observation.hdu,
        "weight": float(observation.weight),
        "weight_is_inverse_variance": bool(observation.weight_is_inverse_variance),
        "weight_path": observation.weight_path,
        "weight_hdu": observation.weight_hdu,
        "mask_path": observation.mask_path,
        "mask_hdu": observation.mask_hdu,
        "dq_path": observation.dq_path,
        "dq_hdu": observation.dq_hdu,
        "dq_policy": (
            None
            if observation.dq_policy is None
            else {
                "reject_bits": observation.dq_policy.reject_bits,
                "propagate_bits": observation.dq_policy.propagate_bits,
                "cosmic_ray_bit": observation.dq_policy.cosmic_ray_bit,
                "unknown_bit_policy": observation.dq_policy.unknown_bit_policy,
            }
        ),
        "primary_beam_path": observation.primary_beam_path,
        "primary_beam_hdu": observation.primary_beam_hdu,
        "photometric_scale": float(observation.photometric_scale),
        "background_offset": float(observation.background_offset),
    }


def _context_provenance(
    observations: Sequence[ImageObservation],
    *,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    stacking_method: MapStackingMethod,
    drizzle_pixfrac: float,
    surface_brightness_area_scaling: bool,
    data_dependent_background: bool,
    has_correlation_factor: bool = False,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "context_word_bits": 32,
        "context_observations": [
            {"index": index, "path": observation.path, "hdu": observation.hdu}
            for index, observation in enumerate(observations)
        ],
        "observation_operators": [
            _observation_operator_provenance(observation)
            for observation in observations
        ],
        "map_execution": {
            "resampling_kernel": resampling_kernel,
            "value_semantics": value_semantics,
            "stacking_method": stacking_method,
            "drizzle_pixfrac": float(drizzle_pixfrac),
            "surface_brightness_area_scaling": bool(surface_brightness_area_scaling),
            "data_dependent_background": bool(data_dependent_background),
        },
    }
    if has_correlation_factor:
        provenance["noise_correlation"] = {
            "kind": "local_inverse_enbw",
            "definition": "first_coefficient_weight/squared_coefficient_weight",
            "scope": "exact for locally shift-invariant large apertures; diagnostic otherwise",
        }
    return provenance


def _linear_inverse_variance(
    normalization: Tensor,
    variance_weight: Tensor,
    *,
    value_semantics: MapValueSemantics,
    pixel_area_sr: Tensor | None,
) -> Tensor:
    scaled_normalization = normalization
    if value_semantics == "flux":
        if pixel_area_sr is None:
            raise RuntimeError("target pixel area is required for flux uncertainty")
        scaled_normalization = normalization / torch.clamp(
            pixel_area_sr,
            min=torch.finfo(pixel_area_sr.dtype).tiny,
        )
    inverse_variance = torch.where(
        variance_weight > 0.0,
        torch.square(scaled_normalization) / variance_weight,
        torch.zeros_like(normalization),
    )
    return inverse_variance


def _local_noise_correlation_factor(
    correlation_weight: Tensor,
    variance_weight: Tensor,
) -> Tensor:
    """Return the local inverse-ENBW factor from first/squared coefficients.

    For locally shift-invariant resampling of independent detector noise this is
    the exact large-aperture variance inflation relative to the diagonal-only
    variance. On distorted or edge-truncated grids it remains a local kernel
    diagnostic, not a covariance matrix.
    """
    supported = (correlation_weight > 0.0) & (variance_weight > 0.0)
    return torch.where(
        supported,
        correlation_weight / variance_weight,
        torch.zeros_like(correlation_weight),
    )


def _should_apply_surface_brightness_area_scaling(
    observations: Sequence[ImageObservation],
    *,
    target_header: dict,
    target_shape: tuple[int, int],
    first_header: dict | None = None,
) -> bool:
    if not observations or not is_plain_tan_header(target_header):
        return True
    target_area_sr = representative_pixel_area_sr(
        WCS(target_header),
        target_shape,
        device=torch.device("cpu"),
    )
    inv_area_sum = 0.0
    for obs_idx, obs in enumerate(observations):
        header_dict_obs = _observation_header_for_area_scaling(
            obs,
            cached_header=first_header if obs_idx == 0 else None,
        )
        if not is_plain_tan_header(header_dict_obs):
            return True
        source_wcs = WCS(header_dict_obs)
        if wcs_has_distortion(source_wcs):
            return True
        source_shape = (
            int(header_dict_obs.get("NAXIS2", 0)),
            int(header_dict_obs.get("NAXIS1", 0)),
        )
        if source_shape[0] <= 0 or source_shape[1] <= 0:
            return True
        source_area_sr = representative_pixel_area_sr(
            source_wcs,
            source_shape,
            device=torch.device("cpu"),
        )
        inv_area_sum += 1.0 / max(float(source_area_sr), 1e-20)
    harmonic_mean_source_area_sr = float(len(observations)) / max(inv_area_sum, 1e-20)
    return float(target_area_sr) >= harmonic_mean_source_area_sr


def _load_observation_weight_map(
    obs: ImageObservation,
    image: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    return load_image_observation_weight_map(obs, image, device=device)


def _effective_observation_photometric_scale(
    obs: ImageObservation,
    hdr: dict | None = None,
) -> float:
    scale = float(obs.photometric_scale)
    flux_kw = getattr(obs, "flux_scale_keyword", None)
    exptime_kw = getattr(obs, "exposure_time_keyword", None)
    if flux_kw is not None or exptime_kw is not None:
        if not isinstance(hdr, dict):
            hdr = getattr(obs, "header", None)
        if not isinstance(hdr, dict):
            hdr = _observation_header_for_area_scaling(obs)
        if flux_kw is not None and flux_kw in hdr:
            scale = scale * float(hdr[flux_kw])
        if exptime_kw is not None and exptime_kw in hdr:
            exptime = float(hdr[exptime_kw])
            if not math.isfinite(exptime) or exptime <= 0.0:
                raise ValueError(f"Header {exptime_kw} must be finite and positive")
            scale = scale / exptime
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Observation photometric_scale must be finite and positive")
    return scale


def _apply_observation_calibration(
    obs: ImageObservation,
    image: Tensor,
    source_weight: Tensor,
) -> tuple[Tensor, Tensor]:
    scale = _effective_observation_photometric_scale(obs)
    return apply_image_observation_calibration(
        photometric_scale=scale,
        background_offset=obs.background_offset,
        image=image,
        source_weight=source_weight,
    )
