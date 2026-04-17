"""Executable operator helpers shared by image-plane mapmaking paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .bundle import ObservationOperatorBundle, image_observation_operator_bundle
from .geometry import (
    ConservativePolygonRemapOp,
    ExactDrizzleGeometryOp,
    ForwardSplatGeometryOp,
)
from .geometry_conservative import conservative_polygon_project
from .geometry_drizzle import exact_point_drizzle_project, exact_square_drizzle_project


LoadWeightMapFn = Callable[[object, Tensor, torch.device], Tensor]
ApplyCalibrationFn = Callable[[object, Tensor, Tensor], tuple[Tensor, Tensor]]
ResampleToTargetFn = Callable[..., tuple[Tensor, Tensor]]
ForwardProjectFn = Callable[..., tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class PreparedImageObservationData:
    image: Tensor
    source_weight: Tensor
    operator_bundle: ObservationOperatorBundle


def prepare_image_observation_data(
    observation: object,
    image: Tensor,
    *,
    device: torch.device,
    resampling_kernel: str,
    load_weight_map_fn: LoadWeightMapFn,
    apply_calibration_fn: ApplyCalibrationFn,
) -> PreparedImageObservationData:
    """Normalize image-plane observation inputs into operator-ready tensors."""
    operator_bundle = image_observation_operator_bundle(
        observation,
        resampling_kernel=resampling_kernel,
    )
    if operator_bundle.mask_weight is not None or operator_bundle.noise is not None:
        source_weight = load_weight_map_fn(observation, image, device=device)
    else:
        source_weight = torch.isfinite(image).to(dtype=torch.float32, device=device)
    if operator_bundle.radiometry is not None or operator_bundle.background is not None:
        prepared_image, prepared_weight = apply_calibration_fn(
            observation,
            image,
            source_weight,
        )
    else:
        prepared_image = torch.nan_to_num(
            image.to(dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        prepared_weight = source_weight.to(dtype=torch.float32)
    return PreparedImageObservationData(
        image=prepared_image,
        source_weight=prepared_weight,
        operator_bundle=operator_bundle,
    )


def project_image_observation_geometry(
    image: Tensor,
    source_weight: Tensor,
    source_wcs,
    target,
    *,
    geometry,
    resampling_kernel: str,
    value_semantics: str,
    drizzle_pixfrac: float,
    pixel_area_sr_map_fn,
    representative_pixel_area_fn,
    chunk_pixels: int,
    resample_weighted_to_target_fn: ResampleToTargetFn,
    forward_project_weighted_to_target_fn: ForwardProjectFn,
) -> tuple[Tensor, Tensor]:
    """Dispatch geometry execution through operator-family types instead of local branches."""
    if isinstance(geometry, ExactDrizzleGeometryOp):
        if resampling_kernel == "drizzle_square":
            return exact_square_drizzle_project(
                image,
                source_weight,
                source_wcs,
                target,
                value_semantics=value_semantics,
                drizzle_pixfrac=drizzle_pixfrac,
                pixel_area_sr_map_fn=pixel_area_sr_map_fn,
                representative_pixel_area_fn=representative_pixel_area_fn,
                chunk_pixels=chunk_pixels,
            )
        return exact_point_drizzle_project(
            image,
            source_weight,
            source_wcs,
            target,
            value_semantics=value_semantics,
            pixel_area_sr_map_fn=pixel_area_sr_map_fn,
            representative_pixel_area_fn=representative_pixel_area_fn,
            chunk_pixels=chunk_pixels,
        )
    if isinstance(geometry, ConservativePolygonRemapOp):
        return conservative_polygon_project(
            image,
            source_weight,
            source_wcs,
            target,
            value_semantics=value_semantics,
            pixel_area_sr_map_fn=pixel_area_sr_map_fn,
            representative_pixel_area_fn=representative_pixel_area_fn,
            chunk_pixels=chunk_pixels,
        )
    if isinstance(geometry, ForwardSplatGeometryOp):
        return forward_project_weighted_to_target_fn(
            image,
            source_weight,
            source_wcs,
            target,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
        )
    return resample_weighted_to_target_fn(
        image,
        source_weight,
        source_wcs,
        target,
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
    )
