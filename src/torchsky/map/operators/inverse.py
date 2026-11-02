"""Inverse-map operator binders built from existing image-plane projections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from ...wcs import WCS
from ..image_plane_projection import (
    _forward_project_weighted_to_target,
    _resample_weighted_to_target,
)
from ..observation import ImageObservation
from ..io_utils import _observation_header_for_area_scaling
from ..solvers import BoundObservationOperator, BoundObservationOperatorSet
from ..target import TargetRenderSpec
from ..windowing import _prepare_target
from .bundle import ObservationOperatorBundle, image_observation_operator_bundle
from .config import validate_map_resampling_kernel, validate_map_value_semantics
from .execution import (
    project_image_observation_geometry,
    stack_inputs_from_projected_observation,
)
from .image_grid import (
    image_shape_hw,
    target_shape_from_header,
    validate_image_plane_2d,
)
from .mask_weight import load_image_observation_weight_map
from .pixel_area import pixel_area_sr_map


def bind_image_plane_target_operator(
    observation: ImageObservation,
    scene_target: TargetRenderSpec,
    *,
    operator_bundle: ObservationOperatorBundle | None = None,
    observation_header: dict[str, Any] | None = None,
    resampling_kernel: str = "bilinear",
    value_semantics: str | None = None,
    drizzle_pixfrac: float = 1.0,
    chunk_pixels: int = 262_144,
) -> BoundObservationOperator:
    """Bind one image-plane observation to a latent target-grid scene."""
    bundle = (
        operator_bundle
        if operator_bundle is not None
        else image_observation_operator_bundle(
            observation,
            resampling_kernel=resampling_kernel,
        )
    )
    kernel = str(resampling_kernel)
    semantics = str(value_semantics or scene_target.value_semantics)
    header = (
        _observation_header_for_area_scaling(observation)
        if observation_header is None
        else dict(observation_header)
    )

    def _forward_scene_tensor(scene_tensor: Tensor) -> Tensor:
        scene_image = validate_image_plane_2d(torch.as_tensor(scene_tensor))
        device = scene_image.device
        source_wcs = WCS(scene_target.header).to(device)
        obs_wcs = WCS(header).to(device)
        obs_shape = target_shape_from_header(
            header,
            fallback=image_shape_hw(scene_target.shape),
        )
        prepared_observation = _prepare_target(
            header=header,
            target_wcs=obs_wcs,
            target_shape=obs_shape,
            device=device,
            include_pixel_area=(semantics == "flux"),
            include_representative_pixel_area=True,
        )
        source_weight = torch.ones_like(scene_image, dtype=torch.float32)
        projected_numer, projected_weight = project_image_observation_geometry(
            scene_image,
            source_weight,
            source_wcs,
            prepared_observation,
            operator_bundle=bundle,
            resampling_kernel=validate_map_resampling_kernel(kernel),
            value_semantics=validate_map_value_semantics(semantics),
            drizzle_pixfrac=float(drizzle_pixfrac),
            pixel_area_sr_map_fn=pixel_area_sr_map,
            chunk_pixels=int(chunk_pixels),
            resample_weighted_to_target_fn=_resample_weighted_to_target,
            forward_project_weighted_to_target_fn=_forward_project_weighted_to_target,
        )[:2]
        predicted, _ = stack_inputs_from_projected_observation(
            projected_numer,
            projected_weight,
            stacking_method="mean",
            value_semantics=semantics,
            pixel_area_sr=prepared_observation.pixel_area_sr,
        )
        return predicted

    def _apply_operator(scene_representation, target_spec):
        del target_spec
        scene_tensor = scene_representation.render_to_target(scene_target)
        return _forward_scene_tensor(scene_tensor)

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        dual = torch.as_tensor(tensor)
        probe = torch.zeros(
            tuple(scene_target.shape),
            dtype=dual.dtype if dual.is_floating_point() else torch.float32,
            device=dual.device,
            requires_grad=True,
        )
        prediction = _forward_scene_tensor(probe)
        if dual.shape != prediction.shape:
            raise ValueError(
                "adjoint input shape must match bound observation prediction shape"
            )
        weighted = torch.sum(prediction * dual.to(dtype=prediction.dtype))
        (grad,) = torch.autograd.grad(weighted, probe)
        return grad

    noise_cache: dict[tuple[torch.device, tuple[int, int]], Tensor] = {}

    def _apply_noise(tensor: Tensor) -> Tensor:
        tensor_t = torch.as_tensor(tensor)
        if tensor_t.ndim < 2:
            raise ValueError("inverse-noise input must have at least two image axes")
        device = tensor_t.device
        shape = image_shape_hw(tensor_t.shape)
        cache_key = (device, shape)
        weight_map = noise_cache.get(cache_key)
        if weight_map is None:
            reference = torch.ones(
                shape,
                dtype=torch.float32,
                device=device,
            )
            weight_map = load_image_observation_weight_map(
                observation,
                reference,
                device=device,
            )
            noise_cache[cache_key] = weight_map
        return tensor_t * weight_map.to(dtype=tensor_t.dtype, device=device)

    return BoundObservationOperator(
        operators=bundle,
        target=scene_target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=_apply_noise,
    )


def bind_image_plane_observations_to_target_operator_set(
    observations: Sequence[ImageObservation],
    scene_target: TargetRenderSpec,
    *,
    observation_headers: Sequence[dict[str, Any] | None] | None = None,
    resampling_kernel: str = "bilinear",
    value_semantics: str | None = None,
    drizzle_pixfrac: float = 1.0,
    chunk_pixels: int = 262_144,
    exposure_chunk_size: int = 1,
) -> BoundObservationOperatorSet:
    """Bind several image-plane observations to one latent target-grid scene."""
    if len(observations) == 0:
        raise ValueError("at least one ImageObservation is required")
    if observation_headers is None:
        headers: Sequence[dict[str, Any] | None] = (None,) * len(observations)
    else:
        headers = observation_headers
        if len(headers) != len(observations):
            raise ValueError("observation_headers length must match observations")
    operators = [
        bind_image_plane_target_operator(
            observation,
            scene_target,
            observation_header=header,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
            chunk_pixels=chunk_pixels,
        )
        for observation, header in zip(observations, headers, strict=True)
    ]
    return BoundObservationOperatorSet(
        tuple(operators),
        exposure_chunk_size=exposure_chunk_size,
    )


__all__ = [
    "bind_image_plane_observations_to_target_operator_set",
    "bind_image_plane_target_operator",
]
