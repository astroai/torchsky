"""Image-plane-first mapmaking primitives."""

from __future__ import annotations

import dataclasses
import math
import operator
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torchfits import read, write

from ..spectral import (
    SpectralAxis,
)
from ..sphere.moc import MOC
from ..wcs import WCS

# Header probe for area scaling: recoverable failures when reading metadata only.
from .observation import (
    ApertureUncertainty,
    CubeResult,
    ImageObservation,
    MapResult,
    MultibandResult,
    Observation,
    OutlierConfig,
    PlannedImagePlaneCubeTarget,
    ResampleConfig,
    SkyConfig,
    StackConfig,
    Target,
)
from .operators.config import (
    MapResamplingKernel,
    MapStackingMethod,
    MapValueSemantics,
)
from .operators.execution import (
    PreparedImageObservationData,
    accumulate_projected_observation,
    finalize_projected_stack,
    masked_median_stack,
    prepare_image_observation_data,
    require_exact_median_scratch_capacity,
    require_sigma_clipped_scratch_capacity,
    spilled_exact_median_stack,
    spilled_overlap_background_stack,
    spilled_sigma_clipped_stack,
    stack_inputs_from_projected_observation,
)
from .operators.image_grid import (
    headers_equivalent_for_direct_copy,
    normalized_header_for_shape,
    target_shape_from_header,
    validate_image_plane_2d,
)
from .operators.pixel_area import (
    pixel_area_sr_map,
    representative_pixel_area_sr,
)
from .operators.resampling import (
    sample_image_plane_batched_channels,
)
from .operators.mask_weight import load_image_observation_dq
from .quality import DQPolicy
from .sky import CrossExposureSkyCorrector, MeshSkyEstimator
from .spectral_projection import (
    _build_band_metadata,
    _build_spectral_projection,
    _observation_spectral_columns,
)
from .windowing import (
    _LARGE_TAN_WINDOWING_EDGE_SAMPLES,  # noqa: F401
    _LARGE_TAN_WINDOWING_MARGIN_PIXELS,  # noqa: F401
    _LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS,  # noqa: F401
    _build_target_world_grid,
    _prepare_target,
    _PreparedImagePlaneTarget,
    _subtarget_header_for_window,
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

from .image_plane_common import (
    _coerce_target_header,
    _convert_world_between_wcs,
    _resolve_map_device,
    _validated_drizzle_pixfrac,
    _validated_map_resampling_kernel,
    _validated_map_stacking_method,
    _validated_map_value_semantics,
    load_observation_image_and_header,
)
from .image_plane_projection import (
    _project_observation_to_target,
    _project_observation_to_target_chunk,
)
from .planning import (
    _obs_target_pixel_bbox,
    plan,
    plan_image_plane_cube_target,
)
from .sky_matching import _resolve_tile_hw, detect_outliers, match_sky

_FINITE_APERTURE_COVARIANCE_KERNELS = frozenset(
    {
        "nearest",
        "bilinear",
        "bicubic",
        "lanczos2",
        "lanczos3",
        "lanczos4",
        "lanczos5",
        "mitchell_netravali",
        "catmull_rom",
        "drizzle_point",
        "drizzle_square",
        "drizzle_turbo",
        "drizzle_gaussian",
        "drizzle_lanczos2",
        "drizzle_lanczos3",
        "conservative_polygon",
        "splat_point",
        "splat_box",
    }
)

_DEFAULT_TARGET_CACHE_CAPACITY = 0


def _validated_target_cache_capacity(capacity: int) -> int:
    try:
        resolved = operator.index(capacity)
    except TypeError as exc:
        raise TypeError("target_cache_capacity must be an integer") from exc
    if resolved < 0:
        raise ValueError("target_cache_capacity must be non-negative")
    return resolved


def _target_grid_cache_signature(
    header: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), repr(value)) for key, value in header.items()))


class _PreparedTargetTileCache:
    def __init__(
        self,
        *,
        capacity: int,
        target_header: Mapping[str, Any],
        canvas_shape: tuple[int, int],
        device: torch.device,
        value_semantics: MapValueSemantics,
        representative_pixel_area_sr: float | None,
    ) -> None:
        self.capacity = _validated_target_cache_capacity(capacity)
        self._grid_signature = _target_grid_cache_signature(target_header)
        self._canvas_shape = (int(canvas_shape[0]), int(canvas_shape[1]))
        self._device = str(device)
        self._coordinate_dtype = str(
            torch.float32 if device.type == "mps" else torch.float64
        )
        self._value_semantics = value_semantics
        self._representative_pixel_area_sr = representative_pixel_area_sr
        self._entries: OrderedDict[tuple[Any, ...], _PreparedImagePlaneTarget] = (
            OrderedDict()
        )
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.admission_skips = 0

    def get_or_prepare(
        self,
        sub_header: dict,
        tile_shape: tuple[int, int],
        *,
        bounds: tuple[int, int, int, int],
    ) -> _PreparedImagePlaneTarget:
        include_area = self._value_semantics == "flux"
        quadrature_offsets = (
            (
                (-0.5 / math.sqrt(3.0), -0.5 / math.sqrt(3.0)),
                (-0.5 / math.sqrt(3.0), 0.5 / math.sqrt(3.0)),
                (0.5 / math.sqrt(3.0), -0.5 / math.sqrt(3.0)),
                (0.5 / math.sqrt(3.0), 0.5 / math.sqrt(3.0)),
            )
            if include_area
            else ()
        )
        key = (
            self._grid_signature,
            tuple(int(value) for value in bounds),
            self._canvas_shape,
            (int(tile_shape[0]), int(tile_shape[1])),
            self._device,
            self._coordinate_dtype,
            self._value_semantics,
            include_area,
            self._representative_pixel_area_sr,
            quadrature_offsets,
        )
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            self._entries.move_to_end(key)
            return cached
        self.misses += 1
        prepared = _prepare_canvas_tile(
            sub_header,
            tile_shape,
            device=torch.device(self._device),
            canvas_shape=self._canvas_shape,
            origin_yx=(bounds[0], bounds[2]),
            value_semantics=self._value_semantics,
            representative_pixel_area_sr=self._representative_pixel_area_sr,
        )
        if self.capacity == 0:
            return prepared
        if len(self._entries) >= self.capacity:
            self.admission_skips += 1
            return prepared
        self._entries[key] = prepared
        return prepared

    def stats(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "size": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "admission_skips": self.admission_skips,
        }


def _prepare_canvas_tile(
    sub_header: dict,
    tile_shape: tuple[int, int],
    *,
    device: torch.device,
    canvas_shape: tuple[int, int],
    origin_yx: tuple[int, int],
    value_semantics: MapValueSemantics,
    representative_pixel_area_sr: float | None,
) -> _PreparedImagePlaneTarget:
    prepared = _prepare_target(
        header=sub_header,
        target_wcs=WCS(sub_header),
        target_shape=tile_shape,
        device=device,
        canvas_shape=canvas_shape,
        canvas_origin_yx=origin_yx,
        include_pixel_area=(value_semantics == "flux"),
        include_representative_pixel_area=False,
        include_quadrature=(value_semantics == "flux"),
        include_world_grid=False,
    )
    prepared.representative_pixel_area_sr = representative_pixel_area_sr
    return prepared


def _observations_have_quality(observations: Sequence[ImageObservation]) -> bool:
    for observation in observations:
        if observation.dq is not None or observation.dq_path is not None:
            return True
        components = [
            value
            for value in (
                observation.var_poisson,
                observation.var_rnoise,
                observation.var_flat,
            )
            if isinstance(value, Tensor)
        ]
        if components:
            total = torch.zeros_like(torch.as_tensor(components[0]), dtype=torch.float32)
            valid = torch.ones_like(total, dtype=torch.bool)
            for component in components:
                component_t = torch.as_tensor(component, dtype=torch.float32)
                valid = valid & torch.isfinite(component_t) & (component_t >= 0.0)
                total = total + torch.where(
                    torch.isfinite(component_t) & (component_t >= 0.0),
                    component_t,
                    torch.zeros_like(component_t),
                )
            if bool((~valid | (total <= 0.0)).any()):
                return True
    return False


def _coadd_image_plane_spatial_tiles(
    observations: Sequence[ImageObservation],
    *,
    first_image: Tensor,
    first_header: dict,
    header_dict: dict,
    target_shape: tuple[int, int],
    tile_height: int,
    tile_width: int,
    output_path: str | None,
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    stacking_method: MapStackingMethod,
    drizzle_pixfrac: float,
    clip_sigma: float,
    clip_iterations: int,
    map_device: torch.device,
    target_cache_capacity: int,
    scratch_dir: str | Path | None = None,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
) -> MapResult:
    """Accumulate a large map with tile-bounded projection intermediates."""
    if not observations:
        raise ValueError("At least one observation is required")
    H, W = int(target_shape[0]), int(target_shape[1])
    th = max(int(tile_height), 1)
    tw = max(int(tile_width), 1)
    device = map_device
    full_target_wcs = WCS(header_dict)
    full_context = torch.zeros(
        (math.ceil(len(observations) / 32), H, W),
        dtype=torch.int64,
        device=device,
    )
    mosaic_surface_brightness_area_scaling: bool | None = None
    if value_semantics == "surface_brightness" and stacking_method == "weighted_mean":
        mosaic_surface_brightness_area_scaling = (
            _should_apply_surface_brightness_area_scaling(
                observations,
                target_header=header_dict,
                target_shape=target_shape,
                first_header=dict(first_header),
            )
        )
    mosaic_representative_area_sr = (
        representative_pixel_area_sr(
            full_target_wcs, target_shape, device=device
        )
        if mosaic_surface_brightness_area_scaling
        else None
    )
    target_cache = _PreparedTargetTileCache(
        capacity=target_cache_capacity,
        target_header=header_dict,
        canvas_shape=(H, W),
        device=device,
        value_semantics=value_semantics,
        representative_pixel_area_sr=mosaic_representative_area_sr,
    )

    if cross_exposure_sky_corrector is not None and len(observations) == 1:
        cross_exposure_sky_corrector = None

    if cross_exposure_sky_corrector is not None:
        read_dev = str(device)
        full_pixel_area_sr = None
        if value_semantics == "flux":
            full_pixel_area_sr = pixel_area_sr_map(
                full_target_wcs, target_shape, device=device
            )

        def overlap_contributions():
            for obs_idx, obs in enumerate(observations):
                if obs_idx == 0:
                    image = validate_image_plane_2d(first_image).to(device=device)
                    header_dict_obs = dict(first_header)
                else:
                    image, header_dict_obs = load_observation_image_and_header(
                        obs, device=read_dev
                    )
                    image = image.to(device=device)
                prepared_observation = prepare_image_observation_data(
                    obs,
                    image,
                    device=device,
                    resampling_kernel=resampling_kernel,
                    load_weight_map_fn=_load_observation_weight_map,
                    apply_calibration_fn=_apply_observation_calibration,
                    background_estimator=background_estimator,
                )
                source_wcs = WCS(header_dict_obs)
                obs_bbox = _obs_target_pixel_bbox(
                    source_wcs, (int(image.shape[0]), int(image.shape[1])), full_target_wcs
                )
                for y0 in range(0, H, th):
                    for x0 in range(0, W, tw):
                        y1, x1 = min(y0 + th, H), min(x0 + tw, W)
                        if obs_bbox is not None and (
                            y1 <= obs_bbox[0]
                            or y0 >= obs_bbox[1]
                            or x1 <= obs_bbox[2]
                            or x0 >= obs_bbox[3]
                        ):
                            continue
                        tile_shape = (y1 - y0, x1 - x0)
                        sub_header, _ = _subtarget_header_for_window(
                            header_dict,
                            full_shape=(H, W),
                            y0=y0,
                            y1=y1,
                            x0=x0,
                            x1=x1,
                        )
                        need_prepared_target = (
                            value_semantics == "flux"
                            or tuple(image.shape) != tile_shape
                            or not headers_equivalent_for_direct_copy(
                                header_dict_obs, sub_header
                            )
                        )
                        prepared_target = (
                            target_cache.get_or_prepare(
                                sub_header,
                                tile_shape,
                                bounds=(y0, y1, x0, x1),
                            )
                            if need_prepared_target
                            else None
                        )
                        projected = _project_observation_to_target_chunk(
                            obs,
                            image=image,
                            header_dict_obs=header_dict_obs,
                            source_wcs=source_wcs,
                            prepared_observation=prepared_observation,
                            prepared_target=prepared_target,
                            target_header=sub_header,
                            target_shape=tile_shape,
                            device=device,
                            footprint_order=None,
                            resampling_kernel=resampling_kernel,
                            value_semantics=value_semantics,
                            drizzle_pixfrac=drizzle_pixfrac,
                            background_estimator=None,
                        )
                        if projected.target_window is None:
                            global_window = (y0, y1, x0, x1)
                            pixel_area = (
                                None
                                if prepared_target is None
                                else prepared_target.pixel_area_sr
                            )
                        else:
                            py0, py1, px0, px1 = projected.target_window
                            global_window = (
                                y0 + py0,
                                y0 + py1,
                                x0 + px0,
                                x0 + px1,
                            )
                            pixel_area = (
                                None
                                if prepared_target is None
                                or prepared_target.pixel_area_sr is None
                                else prepared_target.pixel_area_sr[py0:py1, px0:px1]
                            )
                        projected_numer = projected.numer
                        projected_weight = projected.weight
                        projected_variance = projected.variance_weight
                        if stacking_method == "median":
                            projected_numer, projected_weight = (
                                stack_inputs_from_projected_observation(
                                    projected.numer,
                                    projected.weight,
                                    stacking_method="median",
                                    value_semantics=value_semantics,
                                    pixel_area_sr=pixel_area,
                                )
                            )
                            projected_variance = None
                        yield (
                            obs_idx,
                            projected_numer,
                            projected_weight,
                            projected_variance,
                            projected.overlap,
                            global_window,
                        )
                del prepared_observation, image

        spilled = spilled_overlap_background_stack(
            overlap_contributions(),
            n_observations=len(observations),
            shape=target_shape,
            stacking_method=stacking_method,
            value_semantics=value_semantics,
            corrector=cross_exposure_sky_corrector,
            scratch_dir=scratch_dir,
            output_device=device,
            tile_hw=(th, tw),
            pixel_area_sr=full_pixel_area_sr,
        )
        full_data = spilled.data
        full_weight = spilled.weight
        full_overlap = spilled.overlap_count
        full_context = spilled.context
        full_inverse_variance = (
            _linear_inverse_variance(
                spilled.normalization,
                spilled.variance_weight,
                value_semantics=value_semantics,
                pixel_area_sr=full_pixel_area_sr,
            )
            if spilled.variance_weight is not None
            else None
        )
        full_correlation_factor = (
            _local_noise_correlation_factor(
                spilled.normalization,
                spilled.variance_weight,
            )
            if spilled.variance_weight is not None
            else None
        )
    elif stacking_method in ("median", "sigma_clipped"):
        if stacking_method == "median":
            require_exact_median_scratch_capacity(
                len(observations),
                (min(th, H), min(tw, W)),
                scratch_dir=scratch_dir,
            )
        else:
            require_sigma_clipped_scratch_capacity(
                len(observations),
                (min(th, H), min(tw, W)),
                scratch_dir=scratch_dir,
            )
        full_data = torch.zeros((H, W), dtype=torch.float32, device=device)
        full_weight = torch.zeros((H, W), dtype=torch.float32, device=device)
        full_inverse_variance: Tensor | None = None
        full_correlation_factor: Tensor | None = None
        full_overlap = torch.zeros((H, W), dtype=torch.int32, device=device)
        for y0 in range(0, H, th):
            for x0 in range(0, W, tw):
                y1, x1 = min(y0 + th, H), min(x0 + tw, W)
                sub_header, _ = _subtarget_header_for_window(
                    header_dict,
                    full_shape=(H, W),
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                )
                sub = coadd_image_plane(
                    observations,
                    target_header=sub_header,
                    output_path=None,
                    footprint_order=None,
                    resampling_kernel=resampling_kernel,
                    value_semantics=value_semantics,
                    stacking_method=stacking_method,
                    drizzle_pixfrac=drizzle_pixfrac,
                    clip_sigma=clip_sigma,
                    clip_iterations=clip_iterations,
                    spatial_tile_hw=None,
                    surface_brightness_area_scaling=(
                        mosaic_surface_brightness_area_scaling
                    ),
                    device=map_device,
                    scratch_dir=scratch_dir,
                    background_estimator=background_estimator,
                    cross_exposure_sky_corrector=None,
                )
                full_data[y0:y1, x0:x1] = sub.data.to(device=device)
                full_weight[y0:y1, x0:x1] = sub.weight.to(device=device)
                if sub.inverse_variance is not None:
                    if full_inverse_variance is None:
                        full_inverse_variance = torch.zeros_like(full_weight)
                    full_inverse_variance[y0:y1, x0:x1] = sub.inverse_variance.to(
                        device=device
                    )
                if sub.correlation_factor is not None:
                    if full_correlation_factor is None:
                        full_correlation_factor = torch.zeros_like(full_weight)
                    full_correlation_factor[y0:y1, x0:x1] = (
                        sub.correlation_factor.to(device=device)
                    )
                if sub.overlap_count is not None:
                    full_overlap[y0:y1, x0:x1] = sub.overlap_count.to(device=device)
                if sub.context is not None:
                    full_context[:, y0:y1, x0:x1] = sub.context.to(device=device)
    else:
        numer = torch.zeros((H, W), dtype=torch.float32, device=device)
        denom = torch.zeros((H, W), dtype=torch.float32, device=device)
        variance_weight = torch.zeros_like(denom)
        variance_available = stacking_method == "weighted_mean"
        full_overlap = torch.zeros((H, W), dtype=torch.int32, device=device)
        read_dev = str(device)
        for obs_idx, obs in enumerate(observations):
            if obs_idx == 0:
                image = validate_image_plane_2d(first_image).to(device=device)
                header_dict_obs = dict(first_header)
            else:
                image, header_dict_obs = load_observation_image_and_header(
                    obs, device=read_dev
                )
                image = image.to(device=device)
            prepared_observation = prepare_image_observation_data(
                obs,
                image,
                device=device,
                resampling_kernel=resampling_kernel,
                load_weight_map_fn=_load_observation_weight_map,
                apply_calibration_fn=_apply_observation_calibration,
                background_estimator=background_estimator,
            )
            source_wcs = WCS(header_dict_obs)
            obs_bbox = _obs_target_pixel_bbox(
                source_wcs, (int(image.shape[0]), int(image.shape[1])), full_target_wcs
            )
            for y0 in range(0, H, th):
                for x0 in range(0, W, tw):
                    y1, x1 = min(y0 + th, H), min(x0 + tw, W)
                    if obs_bbox is not None and (
                        y1 <= obs_bbox[0]
                        or y0 >= obs_bbox[1]
                        or x1 <= obs_bbox[2]
                        or x0 >= obs_bbox[3]
                    ):
                        continue
                    tile_shape = (y1 - y0, x1 - x0)
                    sub_header, _ = _subtarget_header_for_window(
                        header_dict,
                        full_shape=(H, W),
                        y0=y0,
                        y1=y1,
                        x0=x0,
                        x1=x1,
                    )
                    need_prepared_target = (
                        value_semantics == "flux"
                        or tuple(image.shape) != tile_shape
                        or not headers_equivalent_for_direct_copy(
                            header_dict_obs, sub_header
                        )
                    )
                    prepared_target = (
                        target_cache.get_or_prepare(
                            sub_header,
                            tile_shape,
                            bounds=(y0, y1, x0, x1),
                        )
                        if need_prepared_target
                        else None
                    )
                    projected = _project_observation_to_target_chunk(
                        obs,
                        image=image,
                        header_dict_obs=header_dict_obs,
                        source_wcs=source_wcs,
                        prepared_observation=prepared_observation,
                        prepared_target=prepared_target,
                        target_header=sub_header,
                        target_shape=tile_shape,
                        device=device,
                        footprint_order=None,
                        resampling_kernel=resampling_kernel,
                        value_semantics=value_semantics,
                        drizzle_pixfrac=drizzle_pixfrac,
                        background_estimator=None,
                    )
                    pixel_area_sr = None
                    if value_semantics == "flux":
                        assert prepared_target is not None
                        pixel_area_sr = prepared_target.pixel_area_sr
                        if pixel_area_sr is None:
                            raise RuntimeError(
                                "target pixel area is required for flux semantics"
                            )
                        if projected.target_window is not None:
                            py0, py1, px0, px1 = projected.target_window
                            pixel_area_sr = pixel_area_sr[py0:py1, px0:px1]
                    accumulate_projected_observation(
                        numer[y0:y1, x0:x1],
                        denom[y0:y1, x0:x1],
                        full_overlap[y0:y1, x0:x1],
                        projected_numer=projected.numer,
                        projected_weight=projected.weight,
                        projected_overlap=projected.overlap,
                        stacking_method=stacking_method,
                        value_semantics=value_semantics,
                        pixel_area_sr=pixel_area_sr,
                        target_window=projected.target_window,
                    )
                    if projected.variance_weight is None:
                        variance_available = False
                    elif variance_available:
                        _accumulate_variance_weight(
                            variance_weight[y0:y1, x0:x1],
                            projected.variance_weight,
                            projected.target_window,
                        )
                    _set_context_bit(
                        full_context[:, y0:y1, x0:x1],
                        obs_idx,
                        projected.overlap > 0,
                        projected.target_window,
                    )
            del prepared_observation, image

        full_pixel_area_sr = None
        if value_semantics == "flux":
            full_pixel_area_sr = pixel_area_sr_map(
                WCS(header_dict), target_shape, device=device
            )
        normalization = denom
        full_data, full_weight = finalize_projected_stack(
            numer,
            denom,
            stacking_method=stacking_method,
            value_semantics=value_semantics,
            pixel_area_sr=full_pixel_area_sr,
        )
        full_inverse_variance = (
            _linear_inverse_variance(
                normalization,
                variance_weight,
                value_semantics=value_semantics,
                pixel_area_sr=full_pixel_area_sr,
            )
            if variance_available
            else None
        )
        full_correlation_factor = (
            _local_noise_correlation_factor(normalization, variance_weight)
            if variance_available
            else None
        )
    output_dq = _propagate_observation_dq(
        observations,
        target_header=header_dict,
        target_shape=target_shape,
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
        drizzle_pixfrac=drizzle_pixfrac,
        output_weight=full_weight,
        device=device,
    )
    provenance = _context_provenance(
        observations,
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
        stacking_method=stacking_method,
        drizzle_pixfrac=drizzle_pixfrac,
        surface_brightness_area_scaling=bool(mosaic_surface_brightness_area_scaling),
        data_dependent_background=(
            background_estimator is not None or cross_exposure_sky_corrector is not None
        ),
        has_correlation_factor=full_correlation_factor is not None,
    )
    provenance["target_tile_cache"] = target_cache.stats()
    result = MapResult(
        data=full_data,
        weight=full_weight,
        header=header_dict,
        inverse_variance=full_inverse_variance,
        correlation_factor=full_correlation_factor,
        context=full_context,
        provenance=provenance,
        overlap_count=full_overlap,
        coverage_moc=None,
        observation_mocs=None,
        dq=output_dq,
        usability_mask=(full_weight > 0.0) if output_dq is not None else None,
        dq_policy=(
            next(
                (
                    observation.dq_policy
                    for observation in observations
                    if observation.dq_policy is not None
                ),
                DQPolicy(),
            )
            if output_dq is not None
            else None
        ),
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write(output_path, full_data, header=header_dict, overwrite=True)
    return result


def _propagate_observation_dq(
    observations: Sequence[ImageObservation],
    *,
    target_header: dict[str, Any],
    target_shape: tuple[int, int],
    resampling_kernel: MapResamplingKernel,
    value_semantics: MapValueSemantics,
    drizzle_pixfrac: float,
    output_weight: Tensor,
    device: torch.device,
) -> Tensor | None:
    if not _observations_have_quality(observations):
        return None
    output = torch.zeros(target_shape, dtype=torch.int64, device=device)
    prepared_target = _prepare_target(
        header=target_header,
        target_wcs=WCS(target_header),
        target_shape=target_shape,
        device=device,
        include_pixel_area=(value_semantics == "flux"),
        include_representative_pixel_area=False,
        include_quadrature=(value_semantics == "flux"),
        include_world_grid=False,
    )
    for observation in observations:
        image, observation_header = load_observation_image_and_header(
            observation,
            device=str(device),
        )
        image = image.to(device=device)
        dq = load_image_observation_dq(observation, image, device=device)
        if dq is None:
            continue
        policy = observation.dq_policy or DQPolicy()
        flags = policy.propagated(dq)
        source_weight = _load_observation_weight_map(
            observation,
            image,
            device=device,
        )
        usable = source_weight > 0.0
        present = 0
        if flags.numel():
            for value in torch.unique(flags).tolist():
                present |= int(value)
        bit = 1
        while bit <= present:
            if present & bit:
                bit_is_set = (flags & bit) != 0
                for support, rejected in (
                    (usable & bit_is_set, False),
                    ((~usable) & bit_is_set, True),
                ):
                    if not bool(support.any()):
                        continue
                    flag_observation = dataclasses.replace(
                        observation,
                        data=torch.ones_like(image, dtype=torch.float32),
                        header=dict(observation_header),
                        wcs=dict(observation_header),
                        weight=1.0,
                        photometric_scale=1.0,
                        background_offset=0.0,
                        weight_map=support.to(dtype=torch.float32),
                        weight_path=None,
                        weight_is_inverse_variance=False,
                        mask=None,
                        mask_path=None,
                        good_bits=None,
                        dq=None,
                        dq_path=None,
                        dq_policy=None,
                        primary_beam=None,
                        primary_beam_path=None,
                        var_poisson=None,
                        var_rnoise=None,
                        var_flat=None,
                    )
                    projected = _project_observation_to_target_chunk(
                        flag_observation,
                        image=flag_observation.data,
                        header_dict_obs=dict(observation_header),
                        source_wcs=WCS(observation_header),
                        prepared_target=prepared_target,
                        target_header=target_header,
                        target_shape=target_shape,
                        device=device,
                        footprint_order=None,
                        resampling_kernel=resampling_kernel,
                        value_semantics=value_semantics,
                        drizzle_pixfrac=drizzle_pixfrac,
                        background_estimator=None,
                    )
                    flagged = projected.weight > 0.0
                    if projected.target_window is None:
                        if rejected:
                            flagged = flagged & (output_weight <= 0.0)
                        output.bitwise_or_(flagged.to(torch.int64) * bit)
                    else:
                        y0, y1, x0, x1 = projected.target_window
                        if rejected:
                            flagged = flagged & (
                                output_weight[y0:y1, x0:x1] <= 0.0
                            )
                        output[y0:y1, x0:x1].bitwise_or_(
                            flagged.to(torch.int64) * bit
                        )
            bit <<= 1
    return output


def coadd_image_plane(
    observations: Sequence[ImageObservation],
    *,
    target_header: dict | None = None,
    output_path: str | None = None,
    footprint_order: int | None = None,
    resampling_kernel: MapResamplingKernel = "bilinear",
    value_semantics: MapValueSemantics = "surface_brightness",
    stacking_method: MapStackingMethod = "weighted_mean",
    drizzle_pixfrac: float = 1.0,
    clip_sigma: float = 4.0,
    clip_iterations: int = 3,
    spatial_tile_hw: tuple[int, int] | None = None,
    target_cache_capacity: int = _DEFAULT_TARGET_CACHE_CAPACITY,
    surface_brightness_area_scaling: bool | None = None,
    device: torch.device | str | None = None,
    scratch_dir: str | Path | None = None,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
) -> MapResult:
    if not observations:
        raise ValueError("At least one observation is required")
    if footprint_order is not None and footprint_order < 0:
        raise ValueError("footprint_order must be non-negative")
    resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
    value_semantics = _validated_map_value_semantics(value_semantics)
    stacking_method = _validated_map_stacking_method(stacking_method)
    drizzle_pixfrac = _validated_drizzle_pixfrac(drizzle_pixfrac)
    clip_sigma = float(clip_sigma)
    clip_iterations = int(clip_iterations)
    target_cache_capacity = _validated_target_cache_capacity(target_cache_capacity)
    if not math.isfinite(clip_sigma) or clip_sigma <= 0.0:
        raise ValueError("clip_sigma must be finite and positive")
    if clip_iterations <= 0:
        raise ValueError("clip_iterations must be positive")
    if stacking_method == "sigma_clipped" and cross_exposure_sky_corrector is not None:
        raise NotImplementedError(
            "sigma_clipped stacking with overlap background rectification is not "
            "yet supported; use background_mode='per-exposure'"
        )

    map_device = _resolve_map_device(device)
    read_dev = str(map_device)
    first = observations[0]
    first_image, first_header = load_observation_image_and_header(
        first, device=read_dev
    )
    if target_header is not None and "CRVAL1" in target_header:
        target_shape = target_shape_from_header(
            target_header,
            fallback=tuple(first_image.shape[-2:]),
        )
        header_dict = normalized_header_for_shape(
            dict(target_header),
            target_shape,
            overrides=None,
        )
    else:
        header_dict = normalized_header_for_shape(
            dict(first_header),
            tuple(first_image.shape[-2:]),
            overrides=target_header,
        )
        target_shape = target_shape_from_header(
            header_dict,
            fallback=tuple(first_image.shape[-2:]),
        )
    device = map_device
    if (
        spatial_tile_hw is None
        and footprint_order is None
        and cross_exposure_sky_corrector is not None
        and len(observations) > 1
    ):
        # ponytail: one internal tile policy keeps overlap execution bounded;
        # expose tuning only if scorecards show the 1024-pixel default is poor.
        spatial_tile_hw = (
            min(1024, int(target_shape[0])),
            min(1024, int(target_shape[1])),
        )
    if spatial_tile_hw is not None:
        if footprint_order is not None:
            raise ValueError(
                "spatial_tile_hw is incompatible with footprint_order=... "
                "(MOC accumulation is not defined per tile); omit one of them"
            )
        return _coadd_image_plane_spatial_tiles(
            observations,
            first_image=first_image,
            first_header=first_header,
            header_dict=header_dict,
            target_shape=target_shape,
            tile_height=int(spatial_tile_hw[0]),
            tile_width=int(spatial_tile_hw[1]),
            output_path=output_path,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            stacking_method=stacking_method,
            drizzle_pixfrac=drizzle_pixfrac,
            clip_sigma=clip_sigma,
            clip_iterations=clip_iterations,
            map_device=map_device,
            target_cache_capacity=target_cache_capacity,
            scratch_dir=scratch_dir,
            background_estimator=background_estimator,
            cross_exposure_sky_corrector=cross_exposure_sky_corrector,
        )
    if not (
        value_semantics == "surface_brightness" and stacking_method == "weighted_mean"
    ):
        apply_surface_brightness_area_scaling = False
    elif surface_brightness_area_scaling is not None:
        apply_surface_brightness_area_scaling = surface_brightness_area_scaling
    else:
        apply_surface_brightness_area_scaling = (
            _should_apply_surface_brightness_area_scaling(
                observations,
                target_header=header_dict,
                target_shape=target_shape,
                first_header=dict(first_header),
            )
        )
    prepared_target: _PreparedImagePlaneTarget | None = None

    def ensure_prepared_target() -> _PreparedImagePlaneTarget:
        nonlocal prepared_target
        if prepared_target is None:
            prepared_target = _prepare_target(
                header=header_dict,
                target_wcs=WCS(header_dict),
                target_shape=target_shape,
                device=device,
                include_pixel_area=(value_semantics == "flux"),
                include_representative_pixel_area=apply_surface_brightness_area_scaling,
                include_quadrature=(value_semantics == "flux"),
                include_world_grid=False,
            )
        return prepared_target

    numer = torch.zeros(target_shape, dtype=torch.float32, device=device)
    denom = torch.zeros(target_shape, dtype=torch.float32, device=device)
    variance_weight = torch.zeros_like(denom)
    variance_available = stacking_method == "weighted_mean"
    overlap_count = torch.zeros(target_shape, dtype=torch.int32, device=device)
    context = torch.zeros(
        (math.ceil(len(observations) / 32), *target_shape),
        dtype=torch.int64,
        device=device,
    )
    observation_mocs: list[MOC] | None = [] if footprint_order is not None else None
    use_spilled_median = (
        stacking_method == "median" and cross_exposure_sky_corrector is None
    )
    use_spilled_sigma = stacking_method == "sigma_clipped"
    median_images: list[Tensor] | None = (
        [] if stacking_method == "median" and not use_spilled_median else None
    )
    median_support: list[Tensor] | None = (
        [] if stacking_method == "median" and not use_spilled_median else None
    )
    projected_store: list[dict] | None = (
        []
        if cross_exposure_sky_corrector is not None and stacking_method != "median"
        else None
    )

    def project_median_observation(
        obs_idx: int,
        obs: ImageObservation,
    ) -> tuple[Tensor, Tensor, Tensor, MOC | None]:
        if obs_idx == 0:
            image = first_image
            header_dict_obs = dict(first_header)
        else:
            image, header_dict_obs = load_observation_image_and_header(
                obs, device=read_dev
            )
        sampled_numer, sampled_weight, overlap, moc = _project_observation_to_target(
            obs,
            image=image,
            header_dict_obs=header_dict_obs,
            prepare_target=ensure_prepared_target,
            target_header=header_dict,
            target_shape=target_shape,
            device=device,
            footprint_order=footprint_order,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
            background_estimator=background_estimator,
        )
        contrib_numer, contrib_weight = stack_inputs_from_projected_observation(
            sampled_numer,
            sampled_weight,
            stacking_method="mean",
            value_semantics=value_semantics,
            pixel_area_sr=(
                ensure_prepared_target().pixel_area_sr
                if value_semantics == "flux"
                else None
            ),
        )
        return contrib_numer, contrib_weight > 0.0, overlap, moc

    def spilled_contributions():
        for obs_idx, obs in enumerate(observations):
            contrib, support, overlap, moc = project_median_observation(obs_idx, obs)
            overlap_count.add_(overlap)
            _set_context_bit(context, obs_idx, overlap > 0, None)
            if observation_mocs is not None and moc is not None:
                observation_mocs.append(moc)
            yield contrib, support

    def project_sigma_observation(
        obs_idx: int,
        obs: ImageObservation,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor | None,
        Tensor,
        MOC | None,
        tuple[int, int, int, int] | None,
    ]:
        if obs_idx == 0:
            image = first_image
            header_dict_obs = dict(first_header)
        else:
            image, header_dict_obs = load_observation_image_and_header(
                obs, device=read_dev
            )
        projected = _project_observation_to_target_chunk(
            obs,
            image=image,
            header_dict_obs=header_dict_obs,
            prepare_target=ensure_prepared_target,
            target_header=header_dict,
            target_shape=target_shape,
            device=device,
            footprint_order=footprint_order,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
            background_estimator=background_estimator,
        )
        values = torch.where(
            projected.weight > 0.0,
            projected.numer / projected.weight,
            torch.zeros_like(projected.numer),
        )
        clip_weight = projected.weight
        clip_variance = projected.variance_weight
        if value_semantics == "flux":
            prepared = ensure_prepared_target()
            if prepared.pixel_area_sr is None:
                raise RuntimeError("target pixel area is required for flux semantics")
            pixel_area = prepared.pixel_area_sr
            if projected.target_window is not None:
                y0, y1, x0, x1 = projected.target_window
                pixel_area = pixel_area[y0:y1, x0:x1]
            pixel_area_squared = torch.square(pixel_area)
            safe_pixel_area_squared = torch.clamp(
                pixel_area_squared,
                min=torch.finfo(pixel_area_squared.dtype).tiny,
            )
            values = pixel_area * values
            clip_weight = clip_weight / safe_pixel_area_squared
            if clip_variance is not None:
                clip_variance = clip_variance / safe_pixel_area_squared
        return (
            values,
            clip_weight,
            clip_variance,
            projected.overlap,
            projected.moc,
            projected.target_window,
        )

    def sigma_clipped_contributions():
        for obs_idx, obs in enumerate(observations):
            values, weights, variance, _overlap, moc, target_window = (
                project_sigma_observation(obs_idx, obs)
            )
            if observation_mocs is not None and moc is not None:
                observation_mocs.append(moc)
            yield values, weights, variance, target_window

    spilled_median_result = None
    if use_spilled_median:
        spilled_median_result = spilled_exact_median_stack(
            spilled_contributions(),
            n_observations=len(observations),
            shape=target_shape,
            scratch_dir=scratch_dir,
            output_device=device,
        )
    spilled_sigma_result = None
    if use_spilled_sigma:
        spilled_sigma_result = spilled_sigma_clipped_stack(
            sigma_clipped_contributions(),
            n_observations=len(observations),
            shape=target_shape,
            clip_sigma=clip_sigma,
            clip_iterations=clip_iterations,
            scratch_dir=scratch_dir,
            output_device=device,
        )

    observation_loop = (
        () if use_spilled_median or use_spilled_sigma else enumerate(observations)
    )
    for obs_idx, obs in observation_loop:
        if stacking_method == "median":
            contrib_numer, support, overlap, moc = project_median_observation(
                obs_idx, obs
            )
            assert median_images is not None and median_support is not None
            median_images.append(contrib_numer)
            median_support.append(support)
            overlap_count = overlap_count + overlap
            _set_context_bit(context, obs_idx, overlap > 0, None)
            if observation_mocs is not None and moc is not None:
                observation_mocs.append(moc)
        else:
            if obs_idx == 0:
                image = first_image
                header_dict_obs = dict(first_header)
            else:
                image, header_dict_obs = load_observation_image_and_header(
                    obs, device=read_dev
                )
            projected = _project_observation_to_target_chunk(
                obs,
                image=image,
                header_dict_obs=header_dict_obs,
                prepare_target=ensure_prepared_target,
                target_header=header_dict,
                target_shape=target_shape,
                device=device,
                footprint_order=footprint_order,
                resampling_kernel=resampling_kernel,
                value_semantics=value_semantics,
                drizzle_pixfrac=drizzle_pixfrac,
                background_estimator=background_estimator,
            )
            pixel_area_sr = None
            if value_semantics == "flux":
                prepared = ensure_prepared_target()
                if prepared.pixel_area_sr is None:
                    raise RuntimeError("target pixel area is required for flux semantics")
                pixel_area_sr = prepared.pixel_area_sr
                if projected.target_window is not None:
                    y0, y1, x0, x1 = projected.target_window
                    pixel_area_sr = pixel_area_sr[y0:y1, x0:x1]
            accumulate_projected_observation(
                numer,
                denom,
                overlap_count,
                projected_numer=projected.numer,
                projected_weight=projected.weight,
                projected_overlap=projected.overlap,
                stacking_method=stacking_method,
                value_semantics=value_semantics,
                pixel_area_sr=pixel_area_sr,
                target_window=projected.target_window,
            )
            if projected.variance_weight is None:
                variance_available = False
            elif variance_available:
                _accumulate_variance_weight(
                    variance_weight,
                    projected.variance_weight,
                    projected.target_window,
                )
            _set_context_bit(
                context,
                obs_idx,
                projected.overlap > 0,
                projected.target_window,
            )
            if projected_store is not None:
                projected_store.append(
                    {
                        "numer": projected.numer,
                        "weight": projected.weight,
                        "overlap": projected.overlap,
                        "target_window": projected.target_window,
                    }
                )
            if observation_mocs is not None and projected.moc is not None:
                observation_mocs.append(projected.moc)

    if cross_exposure_sky_corrector is not None:
        if stacking_method == "median":
            assert median_images is not None and median_support is not None
            corrector_masks = [~(s > 0.0) for s in median_support]
            corrected = cross_exposure_sky_corrector.correct(
                median_images, corrector_masks
            )
            median_images.clear()
            median_images.extend(corrected)
        else:
            assert projected_store is not None
            images = []
            masks = []
            for ps in projected_store:
                n = ps["numer"]
                w = ps["weight"]
                tw = ps["target_window"]
                img = torch.where(w > 0.0, n / w, torch.zeros_like(n))
                if tw is None:
                    images.append(img)
                    masks.append(~(w > 0.0))
                else:
                    y0, y1, x0, x1 = tw
                    full_img = torch.zeros(
                        target_shape, dtype=img.dtype, device=img.device
                    )
                    full_valid = torch.zeros(
                        target_shape, dtype=torch.bool, device=img.device
                    )
                    if y1 > y0 and x1 > x0:
                        full_img[y0:y1, x0:x1] = img
                        full_valid[y0:y1, x0:x1] = w > 0.0
                    images.append(full_img)
                    masks.append(~full_valid)
            corrected = cross_exposure_sky_corrector.correct(images, masks)
            numer.zero_()
            denom.zero_()
            overlap_count.zero_()
            corr_pixel_area_sr = None
            if value_semantics == "flux":
                prepared = ensure_prepared_target()
                if prepared.pixel_area_sr is None:
                    raise RuntimeError(
                        "target pixel area is required for flux semantics"
                    )
                corr_pixel_area_sr = prepared.pixel_area_sr
            for i, ps in enumerate(projected_store):
                w = ps["weight"]
                overlap_contrib = ps["overlap"]
                tw = ps["target_window"]
                pa_slice = corr_pixel_area_sr
                if tw is None:
                    corr_numer = corrected[i] * w
                else:
                    y0, y1, x0, x1 = tw
                    corr_numer = corrected[i][y0:y1, x0:x1] * w
                    if pa_slice is not None:
                        pa_slice = pa_slice[y0:y1, x0:x1]
                accumulate_projected_observation(
                    numer,
                    denom,
                    overlap_count,
                    projected_numer=corr_numer,
                    projected_weight=w,
                    projected_overlap=overlap_contrib,
                    stacking_method=stacking_method,
                    value_semantics=value_semantics,
                    pixel_area_sr=pa_slice,
                    target_window=tw,
                )

    if stacking_method == "median":
        if spilled_median_result is not None:
            coadd, denom = spilled_median_result
        else:
            assert median_images is not None and median_support is not None
            coadd, denom = masked_median_stack(median_images, median_support)
    elif spilled_sigma_result is not None:
        coadd = spilled_sigma_result.data
        denom = spilled_sigma_result.weight
        normalization = denom
        variance_available = spilled_sigma_result.variance_weight is not None
        variance_weight = (
            torch.zeros_like(denom)
            if spilled_sigma_result.variance_weight is None
            else spilled_sigma_result.variance_weight
        )
        overlap_count = spilled_sigma_result.overlap_count
        context = spilled_sigma_result.context
        pixel_area_sr = None
    else:
        pixel_area_sr = None
        if value_semantics == "flux":
            prepared = ensure_prepared_target()
            if prepared.pixel_area_sr is None:
                raise RuntimeError("target pixel area is required for flux semantics")
            pixel_area_sr = prepared.pixel_area_sr
        normalization = denom
        coadd, denom = finalize_projected_stack(
            numer,
            denom,
            stacking_method=stacking_method,
            value_semantics=value_semantics,
            pixel_area_sr=pixel_area_sr,
        )
    inverse_variance = None
    correlation_factor = None
    if stacking_method != "median" and variance_available:
        inverse_variance = _linear_inverse_variance(
            normalization,
            variance_weight,
            value_semantics=(
                "surface_brightness"
                if stacking_method == "sigma_clipped"
                else value_semantics
            ),
            pixel_area_sr=pixel_area_sr,
        )
        correlation_factor = _local_noise_correlation_factor(
            normalization,
            variance_weight,
        )
    coverage_moc = None
    if observation_mocs is not None:
        coverage_moc = MOC(torch.empty((0,), dtype=torch.int64))
        for moc in observation_mocs:
            coverage_moc = coverage_moc.union(moc)

    output_dq = _propagate_observation_dq(
        observations,
        target_header=header_dict,
        target_shape=target_shape,
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
        drizzle_pixfrac=drizzle_pixfrac,
        output_weight=denom,
        device=device,
    )
    result = MapResult(
        data=coadd,
        weight=denom,
        header=header_dict,
        inverse_variance=inverse_variance,
        correlation_factor=correlation_factor,
        context=context,
        provenance=_context_provenance(
            observations,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            stacking_method=stacking_method,
            drizzle_pixfrac=drizzle_pixfrac,
            surface_brightness_area_scaling=apply_surface_brightness_area_scaling,
            data_dependent_background=(
                background_estimator is not None
                or cross_exposure_sky_corrector is not None
            ),
            has_correlation_factor=correlation_factor is not None,
        ),
        overlap_count=overlap_count,
        coverage_moc=coverage_moc,
        observation_mocs=tuple(observation_mocs)
        if observation_mocs is not None
        else None,
        dq=output_dq,
        usability_mask=(denom > 0.0) if output_dq is not None else None,
        dq_policy=(
            next(
                (
                    observation.dq_policy
                    for observation in observations
                    if observation.dq_policy is not None
                ),
                DQPolicy(),
            )
            if output_dq is not None
            else None
        ),
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write(output_path, coadd, header=header_dict, overwrite=True)
    return result


def evaluate_image_plane_aperture_uncertainty(
    observations: Sequence[ImageObservation],
    result: MapResult,
    aperture_weights: Tensor,
    *,
    resampling_kernel: MapResamplingKernel = "bilinear",
    value_semantics: MapValueSemantics = "surface_brightness",
    drizzle_pixfrac: float = 1.0,
    spatial_tile_hw: tuple[int, int] | None = None,
    target_cache_capacity: int = _DEFAULT_TARGET_CACHE_CAPACITY,
    surface_brightness_area_scaling: bool | None = None,
    device: torch.device | str | None = None,
) -> ApertureUncertainty:
    """Propagate independent detector noise through one finite map aperture.

    The aperture is evaluated with the adjoint of the same finite-support
    observation geometry used by :func:`coadd_image_plane`. Source-pixel
    coefficients are accumulated across output tiles before squaring, so shared
    detector pixels contribute their covariance exactly without materializing a
    map covariance matrix. The supplied ``result`` must be the matching weighted
    mean product and provides its already-computed normalization.
    """
    if not observations:
        raise ValueError("At least one observation is required")
    resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
    if resampling_kernel not in _FINITE_APERTURE_COVARIANCE_KERNELS:
        raise ValueError(
            "finite-aperture covariance requires a finite-support resampling kernel"
        )
    value_semantics = _validated_map_value_semantics(value_semantics)
    drizzle_pixfrac = _validated_drizzle_pixfrac(drizzle_pixfrac)
    target_cache_capacity = _validated_target_cache_capacity(target_cache_capacity)
    if result.data.ndim != 2 or result.weight.ndim != 2:
        raise ValueError("finite-aperture covariance requires a 2D MapResult")
    provenance = result.provenance
    execution = None if provenance is None else provenance.get("map_execution")
    if not isinstance(execution, dict):
        raise ValueError(
            "finite-aperture covariance requires MapResult execution provenance"
        )
    try:
        recorded_pixfrac = float(execution["drizzle_pixfrac"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MapResult execution provenance is incomplete") from exc
    if (
        execution.get("resampling_kernel") != resampling_kernel
        or execution.get("value_semantics") != value_semantics
        or execution.get("stacking_method") != "weighted_mean"
        or recorded_pixfrac != drizzle_pixfrac
    ):
        raise ValueError(
            "MapResult execution provenance does not match the aperture operator"
        )
    if execution.get("data_dependent_background") is not False:
        raise ValueError(
            "finite-aperture covariance is unavailable after data-dependent "
            "background estimation"
        )
    recorded_observations = provenance.get("context_observations")
    expected_observations = [
        {"index": index, "path": observation.path, "hdu": observation.hdu}
        for index, observation in enumerate(observations)
    ]
    if recorded_observations != expected_observations:
        raise ValueError(
            "MapResult observation provenance does not match the aperture operator"
        )
    recorded_operators = provenance.get("observation_operators")
    expected_operators = [
        _observation_operator_provenance(observation)
        for observation in observations
    ]
    if recorded_operators != expected_operators:
        raise ValueError(
            "MapResult observation-operator provenance does not match the aperture operator"
        )
    target_shape = (int(result.data.shape[0]), int(result.data.shape[1]))
    if tuple(result.weight.shape) != target_shape:
        raise ValueError("MapResult data and weight shapes must match")
    aperture = torch.as_tensor(aperture_weights)
    if not aperture.is_floating_point() or tuple(aperture.shape) != target_shape:
        raise ValueError("aperture_weights must be a floating map-shaped tensor")
    if not bool(torch.isfinite(aperture).all()) or not bool((aperture != 0.0).any()):
        raise ValueError("aperture_weights must be finite and nonzero")
    if result.inverse_variance is None:
        raise ValueError(
            "finite-aperture covariance requires explicit detector inverse variance"
        )
    inverse_variance = torch.as_tensor(result.inverse_variance)
    if tuple(inverse_variance.shape) != target_shape:
        raise ValueError("MapResult inverse variance must match its data shape")

    map_device = _resolve_map_device(device)
    aperture = aperture.to(device=map_device, dtype=torch.float32)
    map_weight = result.weight.to(device=map_device, dtype=torch.float32)
    inverse_variance = inverse_variance.to(device=map_device, dtype=torch.float32)
    accumulator_dtype = (
        torch.float32 if map_device.type == "mps" else torch.float64
    )
    requested = aperture != 0.0
    if bool((requested & ((map_weight <= 0.0) | (inverse_variance <= 0.0))).any()):
        raise ValueError(
            "aperture includes pixels without detector-noise-supported map coverage"
        )
    diagonal_variance = torch.sum(
        torch.square(aperture)
        * torch.where(
            inverse_variance > 0.0,
            inverse_variance.reciprocal(),
            torch.zeros_like(inverse_variance),
        ),
        dtype=accumulator_dtype,
    )

    header_dict = normalized_header_for_shape(
        dict(result.header), target_shape, overrides=None
    )
    if value_semantics == "flux":
        target_area_sr = pixel_area_sr_map(
            WCS(header_dict), target_shape, device=map_device
        ).to(dtype=torch.float32)
        normalization = map_weight * torch.square(target_area_sr)
        linear_form_weight = torch.where(
            normalization > 0.0,
            aperture * target_area_sr / normalization,
            torch.zeros_like(aperture),
        )
    else:
        target_area_sr = None
        linear_form_weight = torch.where(
            map_weight > 0.0,
            aperture / map_weight,
            torch.zeros_like(aperture),
        )

    read_device = str(map_device)
    first_image, first_header = load_observation_image_and_header(
        observations[0], device=read_device
    )
    first_image = first_image.to(device=map_device)
    first_header = dict(first_header)
    if value_semantics != "surface_brightness":
        apply_area_scaling = False
    elif surface_brightness_area_scaling is not None:
        apply_area_scaling = bool(surface_brightness_area_scaling)
    else:
        apply_area_scaling = _should_apply_surface_brightness_area_scaling(
            observations,
            target_header=header_dict,
            target_shape=target_shape,
            first_header=first_header,
        )
    if execution.get("surface_brightness_area_scaling") is not apply_area_scaling:
        raise ValueError(
            "MapResult area-scaling provenance does not match the aperture operator"
        )
    full_representative_area_sr = (
        representative_pixel_area_sr(
            WCS(header_dict), target_shape, device=map_device
        )
        if apply_area_scaling
        else None
    )

    if spatial_tile_hw is None:
        tile_height = min(1024, target_shape[0])
        tile_width = min(1024, target_shape[1])
    else:
        tile_height = int(spatial_tile_hw[0])
        tile_width = int(spatial_tile_hw[1])
        if tile_height <= 0 or tile_width <= 0:
            raise ValueError("spatial_tile_hw dimensions must be positive")

    active_tiles: list[tuple[int, int, int, int]] = []
    for y0 in range(0, target_shape[0], tile_height):
        for x0 in range(0, target_shape[1], tile_width):
            y1 = min(y0 + tile_height, target_shape[0])
            x1 = min(x0 + tile_width, target_shape[1])
            if bool((linear_form_weight[y0:y1, x0:x1] != 0.0).any()):
                active_tiles.append((y0, y1, x0, x1))

    target_cache = _PreparedTargetTileCache(
        capacity=target_cache_capacity,
        target_header=header_dict,
        canvas_shape=target_shape,
        device=map_device,
        value_semantics=value_semantics,
        representative_pixel_area_sr=full_representative_area_sr,
    )

    exact_variance = torch.zeros((), dtype=accumulator_dtype, device=map_device)
    for observation_index, observation in enumerate(observations):
        if observation_index == 0:
            image = first_image
            observation_header = first_header
        else:
            image, observation_header = load_observation_image_and_header(
                observation, device=read_device
            )
            image = image.to(device=map_device)
        prepared = prepare_image_observation_data(
            observation,
            image,
            device=map_device,
            resampling_kernel=resampling_kernel,
            load_weight_map_fn=_load_observation_weight_map,
            apply_calibration_fn=_apply_observation_calibration,
            background_estimator=None,
        )
        if prepared.operator_bundle.noise is None:
            raise ValueError(
                "finite-aperture covariance requires explicit detector inverse variance "
                "for every observation"
            )
        source_weight = prepared.source_weight.detach().to(dtype=torch.float32)
        source_image = (
            prepared.image.detach().to(dtype=torch.float32).clone().requires_grad_(True)
        )
        differentiable = PreparedImageObservationData(
            image=source_image,
            source_weight=source_weight,
            operator_bundle=prepared.operator_bundle,
        )
        source_wcs = WCS(observation_header)
        source_gradient = torch.zeros_like(source_image)
        for y0, y1, x0, x1 in active_tiles:
            aperture_tile = linear_form_weight[y0:y1, x0:x1]
            tile_shape = (y1 - y0, x1 - x0)
            sub_header, _ = _subtarget_header_for_window(
                header_dict,
                full_shape=target_shape,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
            )
            needs_target = (
                value_semantics == "flux"
                or tuple(source_image.shape) != tile_shape
                or not headers_equivalent_for_direct_copy(
                    observation_header, sub_header
                )
            )
            prepared_target = (
                target_cache.get_or_prepare(
                    sub_header,
                    tile_shape,
                    bounds=(y0, y1, x0, x1),
                )
                if needs_target
                else None
            )
            with torch.enable_grad():
                projected = _project_observation_to_target_chunk(
                    observation,
                    image=source_image,
                    header_dict_obs=observation_header,
                    source_wcs=source_wcs,
                    prepared_observation=differentiable,
                    prepared_target=prepared_target,
                    target_header=sub_header,
                    target_shape=tile_shape,
                    device=map_device,
                    footprint_order=None,
                    resampling_kernel=resampling_kernel,
                    value_semantics=value_semantics,
                    drizzle_pixfrac=drizzle_pixfrac,
                    background_estimator=None,
                )
                if projected.numer.numel() == 0:
                    continue
                if projected.target_window is None:
                    projected_aperture = aperture_tile
                else:
                    py0, py1, px0, px1 = projected.target_window
                    projected_aperture = aperture_tile[py0:py1, px0:px1]
                linear_form = torch.sum(projected.numer * projected_aperture)
                if not linear_form.requires_grad:
                    continue
                gradient = torch.autograd.grad(
                    linear_form,
                    source_image,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]
            if gradient is not None:
                source_gradient.add_(gradient.detach())
        source_variance = torch.where(
            source_weight > 0.0,
            source_weight.reciprocal(),
            torch.zeros_like(source_weight),
        )
        exact_variance.add_(
            torch.sum(
                torch.square(source_gradient) * source_variance,
                dtype=accumulator_dtype,
            )
        )
        if observation_index == 0:
            del first_image
        del differentiable, image, prepared, source_gradient, source_image
        del source_variance, source_weight

    return ApertureUncertainty(
        variance=exact_variance,
        diagonal_variance=diagonal_variance,
    )


class ImagePlaneMapMaker:
    """Differentiable image-plane coaddition baseline."""

    def __init__(
        self,
        *,
        target_header: dict | None = None,
        resampling_kernel: MapResamplingKernel = "bilinear",
        value_semantics: MapValueSemantics = "surface_brightness",
        stacking_method: MapStackingMethod = "weighted_mean",
        drizzle_pixfrac: float = 1.0,
        clip_sigma: float = 4.0,
        clip_iterations: int = 3,
        spatial_tile_hw: tuple[int, int] | None = None,
        target_cache_capacity: int = _DEFAULT_TARGET_CACHE_CAPACITY,
        device: torch.device | str | None = None,
        scratch_dir: str | Path | None = None,
        background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
        cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
    ) -> None:
        self.target_header = target_header
        self.resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
        self.value_semantics = _validated_map_value_semantics(value_semantics)
        self.stacking_method = _validated_map_stacking_method(stacking_method)
        self.drizzle_pixfrac = _validated_drizzle_pixfrac(drizzle_pixfrac)
        self.clip_sigma = float(clip_sigma)
        self.clip_iterations = int(clip_iterations)
        self.spatial_tile_hw = (
            None
            if spatial_tile_hw is None
            else (int(spatial_tile_hw[0]), int(spatial_tile_hw[1]))
        )
        self.target_cache_capacity = _validated_target_cache_capacity(
            target_cache_capacity
        )
        self.device = device
        self.scratch_dir = scratch_dir
        self.background_estimator = background_estimator
        self.cross_exposure_sky_corrector = cross_exposure_sky_corrector

    def run(
        self,
        observations: Sequence[ImageObservation],
        *,
        output_path: str | None = None,
        device: torch.device | str | None = None,
    ) -> MapResult:
        eff_device = device if device is not None else self.device
        return coadd_image_plane(
            observations,
            target_header=self.target_header,
            output_path=output_path,
            resampling_kernel=self.resampling_kernel,
            value_semantics=self.value_semantics,
            stacking_method=self.stacking_method,
            drizzle_pixfrac=self.drizzle_pixfrac,
            clip_sigma=self.clip_sigma,
            clip_iterations=self.clip_iterations,
            spatial_tile_hw=self.spatial_tile_hw,
            target_cache_capacity=self.target_cache_capacity,
            device=eff_device,
            scratch_dir=self.scratch_dir,
            background_estimator=self.background_estimator,
            cross_exposure_sky_corrector=self.cross_exposure_sky_corrector,
        )


def coadd_image_plane_cube(
    observations: Sequence[ImageObservation],
    *,
    target: PlannedImagePlaneCubeTarget | None = None,
    target_header: dict | None = None,
    footprint_order: int | None = None,
    resampling_kernel: MapResamplingKernel = "bilinear",
    value_semantics: MapValueSemantics = "surface_brightness",
    stacking_method: MapStackingMethod = "weighted_mean",
    drizzle_pixfrac: float = 1.0,
    spectral_axis: SpectralAxis | None = None,
    spectral_bin_edges: Tensor | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    reference_index: int = 0,
    device: torch.device | str | None = None,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
) -> CubeResult:
    if not observations:
        raise ValueError("At least one observation is required")
    if cross_exposure_sky_corrector is not None:
        raise NotImplementedError(
            "cross_exposure_sky_corrector is not yet implemented for cube coaddition; "
            "use coadd_image_plane with per-band observations"
        )
    if target is not None and (
        target_header is not None
        or spectral_axis is not None
        or spectral_bin_edges is not None
        or reference_index != 0
    ):
        raise ValueError(
            "target is mutually exclusive with target_header, spectral_axis, "
            "spectral_bin_edges, and reference_index"
        )
    map_device = _resolve_map_device(device)
    read_dev = str(map_device)
    resampling_kernel = _validated_map_resampling_kernel(resampling_kernel)
    value_semantics = _validated_map_value_semantics(value_semantics)
    stacking_method = _validated_map_stacking_method(stacking_method)
    drizzle_pixfrac = _validated_drizzle_pixfrac(drizzle_pixfrac)
    if stacking_method in ("median", "sigma_clipped"):
        raise ValueError(
            "coadd_image_plane_cube supports median and sigma_clipped only for "
            "single-band image-plane coadds"
        )

    if target is None:
        target = plan_image_plane_cube_target(
            observations,
            reference_index=reference_index,
            header_overrides=target_header,
            spectral_axis=spectral_axis,
            spectral_bin_edges=spectral_bin_edges,
            spectral_unit=spectral_unit,
            spectral_type=spectral_type,
            device=map_device,
        )

    (
        observation_center,
        observation_lower,
        observation_upper,
        observation_filter_center,
        observation_filter_sigma,
        observation_names,
    ) = _observation_spectral_columns(observations)
    projection = _build_spectral_projection(
        observation_center,
        observation_lower,
        observation_upper,
        observation_filter_center,
        observation_filter_sigma,
        target_axis=target.spectral_axis,
        target_edges=target.spectral_bin_edges,
    )
    band_metadata = _build_band_metadata(
        target.spectral_axis,
        target.spectral_bin_edges,
        projection,
        observation_names,
    )

    first_image, first_header = load_observation_image_and_header(
        observations[0],
        device=read_dev,
    )
    device = map_device
    header_dict = dict(target.header)
    target_shape = target.shape
    target_wcs = WCS(header_dict)
    apply_surface_brightness_area_scaling = (
        value_semantics == "surface_brightness"
        and stacking_method == "weighted_mean"
        and _should_apply_surface_brightness_area_scaling(
            observations,
            target_header=header_dict,
            target_shape=target_shape,
            first_header=dict(first_header),
        )
    )
    prepared_target: _PreparedImagePlaneTarget | None = None

    def ensure_prepared_target() -> _PreparedImagePlaneTarget:
        nonlocal prepared_target
        if prepared_target is None:
            prepared_target = _prepare_target(
                header=header_dict,
                target_wcs=target_wcs,
                target_shape=target_shape,
                device=device,
                include_pixel_area=(value_semantics == "flux"),
                include_representative_pixel_area=apply_surface_brightness_area_scaling,
                include_quadrature=(value_semantics == "flux"),
                include_world_grid=False,
            )
        return prepared_target

    n_band = int(target.spectral_axis.values.numel())
    numer = torch.zeros((n_band, *target_shape), dtype=torch.float32, device=device)
    denom = torch.zeros((n_band, *target_shape), dtype=torch.float32, device=device)
    variance_weight = torch.zeros_like(denom)
    correlation_weight = torch.zeros_like(denom)
    variance_available = stacking_method == "weighted_mean"
    overlap_count = torch.zeros(
        (n_band, *target_shape), dtype=torch.int32, device=device
    )
    coverage_mocs: list[MOC | None] | None = None
    if footprint_order is not None:
        coverage_mocs = [
            MOC(torch.empty((0,), dtype=torch.int64)) for _ in range(n_band)
        ]

    for obs_idx, obs in enumerate(observations):
        if obs_idx == 0:
            image = first_image
            header_dict_obs = dict(first_header)
        else:
            image, header_dict_obs = load_observation_image_and_header(
                obs, device=read_dev
            )
        projected = _project_observation_to_target_chunk(
            obs,
            image=image,
            header_dict_obs=header_dict_obs,
            prepare_target=ensure_prepared_target,
            target_header=header_dict,
            target_shape=target_shape,
            device=device,
            footprint_order=footprint_order,
            resampling_kernel=resampling_kernel,
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
            background_estimator=background_estimator,
        )
        coeffs = (
            projection[obs_idx]
            .to(device=device, dtype=torch.float32)
            .view(n_band, 1, 1)
        )
        pixel_area_sr = None
        if value_semantics == "flux":
            prepared = ensure_prepared_target()
            if prepared.pixel_area_sr is None:
                raise RuntimeError("target pixel area is required for flux semantics")
            pixel_area_sr = prepared.pixel_area_sr
            if projected.target_window is not None:
                y0, y1, x0, x1 = projected.target_window
                pixel_area_sr = pixel_area_sr[y0:y1, x0:x1]
        accumulate_projected_observation(
            numer,
            denom,
            overlap_count,
            projected_numer=projected.numer,
            projected_weight=projected.weight,
            projected_overlap=projected.overlap,
            stacking_method=stacking_method,
            value_semantics=value_semantics,
            pixel_area_sr=pixel_area_sr,
            target_window=projected.target_window,
            coeffs=coeffs,
        )
        if projected.variance_weight is None:
            variance_available = False
        elif variance_available:
            _accumulate_variance_weight(
                variance_weight,
                torch.square(coeffs) * projected.variance_weight.unsqueeze(0),
                projected.target_window,
            )
            _accumulate_variance_weight(
                correlation_weight,
                torch.square(coeffs) * projected.weight.unsqueeze(0),
                projected.target_window,
            )
        if coverage_mocs is not None and projected.moc is not None:
            band_indices = torch.nonzero(
                projection[obs_idx] > 0.0, as_tuple=False
            ).flatten()
            for band_idx in band_indices.tolist():
                current = coverage_mocs[band_idx]
                coverage_mocs[band_idx] = (
                    projected.moc if current is None else current.union(projected.moc)
                )

    pixel_area_sr = None
    if value_semantics == "flux":
        prepared = ensure_prepared_target()
        if prepared.pixel_area_sr is None:
            raise RuntimeError("target pixel area is required for flux semantics")
        pixel_area_sr = prepared.pixel_area_sr.unsqueeze(0)
    normalization = denom
    cube, denom = finalize_projected_stack(
        numer,
        denom,
        stacking_method=stacking_method,
        value_semantics=value_semantics,
        pixel_area_sr=pixel_area_sr,
    )
    inverse_variance = (
        _linear_inverse_variance(
            normalization,
            variance_weight,
            value_semantics=value_semantics,
            pixel_area_sr=pixel_area_sr,
        )
        if variance_available
        else None
    )
    correlation_factor = (
        _local_noise_correlation_factor(correlation_weight, variance_weight)
        if variance_available
        else None
    )
    return CubeResult(
        spectral_axis=target.spectral_axis,
        band_metadata=band_metadata,
        data=cube,
        weight=denom,
        header=header_dict,
        spatial_wcs=target_wcs,
        inverse_variance=inverse_variance,
        correlation_factor=correlation_factor,
        overlap_count=overlap_count,
        coverage_mocs=tuple(coverage_mocs) if coverage_mocs is not None else None,
    )


def coadd_image_plane_multiband(
    observations: Sequence[ImageObservation],
    *,
    target_header: dict | None = None,
    footprint_order: int | None = None,
    resampling_kernel: MapResamplingKernel = "bilinear",
    value_semantics: MapValueSemantics = "surface_brightness",
    stacking_method: MapStackingMethod = "weighted_mean",
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    device: torch.device | str | None = None,
) -> MultibandResult:
    cube = coadd_image_plane_cube(
        observations,
        target_header=target_header,
        footprint_order=footprint_order,
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
        stacking_method=stacking_method,
        spectral_axis=None,
        spectral_bin_edges=None,
        spectral_unit=spectral_unit,
        spectral_type=spectral_type,
        device=device,
    )
    return MultibandResult(
        spectral_axis=cube.spectral_axis,
        data=cube.data,
        weight=cube.weight,
        header=cube.header,
        inverse_variance=cube.inverse_variance,
        correlation_factor=cube.correlation_factor,
        overlap_count=cube.overlap_count,
        coverage_mocs=cube.coverage_mocs,
        band_observation_counts=cube.band_metadata.observation_count,
    )


def _propagate_observation_variance_components(
    observations: Sequence[Observation],
    *,
    target_header: dict[str, Any],
    target_shape: tuple[int, int],
    resampling_kernel: MapResamplingKernel,
    drizzle_pixfrac: float,
    value_semantics: MapValueSemantics,
    device: torch.device,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    has_poisson = any(obs.var_poisson is not None for obs in observations)
    has_rnoise = any(obs.var_rnoise is not None for obs in observations)
    has_flat = any(obs.var_flat is not None for obs in observations)
    if not (has_poisson or has_rnoise or has_flat):
        return None, None, None

    target_wcs = WCS(target_header)
    prepared_target = _prepare_target(
        header=target_header,
        target_wcs=target_wcs,
        target_shape=target_shape,
        device=device,
        include_pixel_area=(value_semantics == "flux"),
        include_representative_pixel_area=False,
        include_quadrature=(value_semantics == "flux"),
        include_world_grid=False,
    )

    def _accumulate_component(attr_name: str) -> Tensor:
        var_num = torch.zeros(target_shape, dtype=torch.float32, device=device)
        var_den = torch.zeros(target_shape, dtype=torch.float32, device=device)
        read_dev = str(device)
        for obs in observations:
            comp = getattr(obs, attr_name)
            image, header_dict_obs = load_observation_image_and_header(
                obs, device=read_dev
            )
            image_dev = image.to(device=device, dtype=torch.float32)
            if comp is None:
                comp_t = torch.zeros_like(image_dev)
            elif isinstance(comp, Tensor):
                comp_t = validate_image_plane_2d(comp).to(
                    device=device, dtype=torch.float32
                )
            else:
                comp_raw, _ = read(
                    str(comp), hdu=obs.hdu, return_header=True, device=read_dev
                )
                comp_t = validate_image_plane_2d(comp_raw).to(
                    device=device, dtype=torch.float32
                )
            source_wcs = WCS(header_dict_obs)
            s_eff = _effective_observation_photometric_scale(obs, header_dict_obs)
            calibrated_variance = comp_t * float(s_eff) ** 2
            prepared_observation = prepare_image_observation_data(
                obs,
                image_dev,
                device=device,
                resampling_kernel=resampling_kernel,
                load_weight_map_fn=_load_observation_weight_map,
                apply_calibration_fn=_apply_observation_calibration,
                background_estimator=None,
            )
            source_term = (
                torch.square(prepared_observation.source_weight)
                * torch.where(
                    torch.isfinite(calibrated_variance)
                    & (calibrated_variance >= 0.0),
                    calibrated_variance,
                    torch.zeros_like(calibrated_variance),
                )
            )
            if value_semantics == "flux":
                src_area = pixel_area_sr_map(
                    source_wcs,
                    (int(comp_t.shape[0]), int(comp_t.shape[1])),
                    device=device,
                )
                source_term = source_term * torch.square(src_area)
            source_term = torch.where(
                torch.isfinite(image_dev), source_term, torch.zeros_like(source_term)
            )
            var_obs = dataclasses.replace(
                obs,
                data=torch.zeros_like(image_dev),
                wcs=header_dict_obs,
                weight=1.0,
                photometric_scale=1.0,
                background_offset=0.0,
                weight_map=source_term,
                weight_path=None,
                weight_type="weight",
                weight_is_inverse_variance=True,
                mask=None,
                mask_path=None,
                good_bits=None,
                dq=None,
                dq_path=None,
                dq_policy=None,
                primary_beam=None,
                primary_beam_path=None,
                var_poisson=None,
                var_rnoise=None,
                var_flat=None,
            )
            actual = _project_observation_to_target_chunk(
                obs,
                image=image_dev,
                header_dict_obs=header_dict_obs,
                source_wcs=source_wcs,
                prepared_observation=prepared_observation,
                prepared_target=prepared_target,
                target_header=target_header,
                target_shape=target_shape,
                device=device,
                footprint_order=None,
                resampling_kernel=resampling_kernel,
                value_semantics=value_semantics,
                drizzle_pixfrac=drizzle_pixfrac,
                background_estimator=None,
            )
            squared = _project_observation_to_target_chunk(
                var_obs,
                image=var_obs.data,
                header_dict_obs=header_dict_obs,
                source_wcs=source_wcs,
                prepared_target=prepared_target,
                target_header=target_header,
                target_shape=target_shape,
                device=device,
                footprint_order=None,
                resampling_kernel=resampling_kernel,
                value_semantics="surface_brightness",
                drizzle_pixfrac=drizzle_pixfrac,
                background_estimator=None,
            )
            if squared.variance_weight is None:
                raise RuntimeError(
                    "squared response is unavailable for the selected variance kernel"
                )
            if actual.target_window != squared.target_window:
                raise RuntimeError("science and variance response support differ")
            if actual.target_window is None:
                var_num.add_(squared.variance_weight)
                var_den.add_(actual.weight)
            else:
                y0, y1, x0, x1 = actual.target_window
                if y1 > y0 and x1 > x0:
                    var_num[y0:y1, x0:x1].add_(squared.variance_weight)
                    var_den[y0:y1, x0:x1].add_(actual.weight)
        if value_semantics == "flux":
            assert prepared_target.pixel_area_sr is not None
            target_area_squared = torch.square(prepared_target.pixel_area_sr)
            var_num = var_num / torch.clamp(
                target_area_squared,
                min=torch.finfo(target_area_squared.dtype).tiny,
            )
            var_den = var_den / torch.clamp(
                target_area_squared,
                min=torch.finfo(target_area_squared.dtype).tiny,
            )
        safe_den_sq = torch.clamp(
            torch.square(var_den), min=torch.finfo(var_den.dtype).tiny
        )
        output = torch.where(
            var_den > 0.0, var_num / safe_den_sq, torch.zeros_like(var_num)
        )
        return output

    vp = _accumulate_component("var_poisson") if has_poisson else None
    vr = _accumulate_component("var_rnoise") if has_rnoise else None
    vf = _accumulate_component("var_flat") if has_flat else None
    return vp, vr, vf


def resample(
    observation: Observation | Sequence[Observation],
    target: Target | Mapping[str, Any] | WCS | None = None,
    *,
    config: ResampleConfig | None = None,
    kernel: MapResamplingKernel | None = None,
    pixfrac: float | None = None,
    semantics: MapValueSemantics | None = None,
    footprint_order: int | None = None,
    device: torch.device | str | None = None,
) -> MapResult | list[MapResult]:
    """Resample one or more observations onto a common target WCS grid without stacking."""
    cfg = config or ResampleConfig()
    resolved_kernel = kernel if kernel is not None else cfg.kernel
    resolved_pixfrac = pixfrac if pixfrac is not None else cfg.pixfrac
    resolved_semantics = semantics if semantics is not None else cfg.semantics
    single = isinstance(observation, ImageObservation)
    obs_list = [observation] if single else list(observation)
    if not obs_list:
        raise ValueError("At least one Observation is required")
    target_header = _coerce_target_header(target)
    if target_header is None and len(obs_list) > 1:
        target_header = dict(plan(obs_list, device=device).header)

    results: list[MapResult] = []
    for obs in obs_list:
        res = coadd_image_plane(
            [obs],
            target_header=target_header,
            footprint_order=footprint_order,
            resampling_kernel=resolved_kernel,
            value_semantics=resolved_semantics,
            stacking_method="weighted_mean",
            drizzle_pixfrac=resolved_pixfrac,
            device=device,
        )
        vp, vr, vf = _propagate_observation_variance_components(
            [obs],
            target_header=res.header,
            target_shape=tuple(res.data.shape),
            resampling_kernel=resolved_kernel,
            drizzle_pixfrac=resolved_pixfrac,
            value_semantics=resolved_semantics,
            device=res.data.device,
        )
        if vp is not None or vr is not None or vf is not None:
            res = dataclasses.replace(res, var_poisson=vp, var_rnoise=vr, var_flat=vf)
        results.append(res)
    return results[0] if single else results


def blot(
    coadd_or_data: MapResult | Tensor,
    source_or_target: Observation
    | Sequence[Observation]
    | Target
    | Mapping[str, Any]
    | WCS,
    *,
    target: Observation
    | Sequence[Observation]
    | Target
    | Mapping[str, Any]
    | WCS
    | None = None,
    kernel: Literal[
        "bilinear", "bicubic", "lanczos2", "lanczos3", "lanczos4", "nearest"
    ] = "bilinear",
    semantics: MapValueSemantics = "surface_brightness",
    device: torch.device | str | None = None,
) -> Tensor | list[Tensor]:
    """Inverse-project (blot) a clean coadd back into one or more distorted exposure frames.

    Supports both:
    - ``blot(coadd_result, observations, kernel=..., semantics=...)``
    - ``blot(coadd_tensor, coadd_wcs_or_header, target=observations, ...)``
    """
    semantics = _validated_map_value_semantics(semantics)
    if kernel not in (
        "bilinear",
        "bicubic",
        "lanczos2",
        "lanczos3",
        "lanczos4",
        "nearest",
    ):
        raise ValueError(
            "blot kernel must be one of 'bilinear', 'bicubic', 'lanczos2', 'lanczos3', 'lanczos4', or 'nearest'"
        )
    if isinstance(coadd_or_data, MapResult):
        coadd_tensor = coadd_or_data.data
        coadd_header = dict(coadd_or_data.header)
        blot_target = source_or_target if target is None else target
    else:
        coadd_tensor = validate_image_plane_2d(coadd_or_data)
        if target is None:
            raise ValueError(
                "When passing a raw Tensor to blot(), specify coadd WCS/header as second argument and target=... as keyword"
            )
        if isinstance(source_or_target, (ImageObservation, Sequence)) and not isinstance(
            source_or_target, Mapping
        ):
            raise TypeError(
                "Second argument must be coadd Target/header/WCS when first argument is a raw Tensor"
            )
        coerced = _coerce_target_header(source_or_target)
        assert coerced is not None
        coadd_header = coerced
        blot_target = target

    map_device = (
        coadd_tensor.device if device is None else _resolve_map_device(device)
    )
    coadd_img = coadd_tensor.to(device=map_device, dtype=torch.float32)
    coadd_wcs = WCS(coadd_header)
    coadd_shape = (int(coadd_img.shape[0]), int(coadd_img.shape[1]))
    h_c, w_c = coadd_shape

    def _blot_single(
        dest: Observation | Target | Mapping[str, Any] | WCS,
    ) -> Tensor:
        if isinstance(dest, ImageObservation):
            obs_img, obs_hdr = load_observation_image_and_header(
                dest, device=str(map_device)
            )
            dest_shape = (int(obs_img.shape[0]), int(obs_img.shape[1]))
            dest_hdr = dict(obs_hdr)
        else:
            coerced_dest = _coerce_target_header(dest)
            if coerced_dest is None:
                raise ValueError("blot destination cannot be None")
            dest_hdr = coerced_dest
            dest_shape = target_shape_from_header(dest_hdr, fallback=coadd_shape)
        dest_wcs = WCS(dest_hdr)
        dest_prep = _prepare_target(
            header=dest_hdr,
            target_wcs=dest_wcs,
            target_shape=dest_shape,
            device=map_device,
            include_pixel_area=(semantics == "flux"),
            include_representative_pixel_area=False,
            include_quadrature=False,
            include_world_grid=False,
        )
        ra_grid, dec_grid = _build_target_world_grid(
            dest_wcs, dest_shape[0], dest_shape[1], device=map_device
        )
        flat_ra = ra_grid.reshape(-1)
        flat_dec = dec_grid.reshape(-1)
        flat_ra, flat_dec = _convert_world_between_wcs(
            flat_ra, flat_dec, dest_wcs, coadd_wcs
        )
        src_x_flat, src_y_flat = coadd_wcs.world_to_pixel(flat_ra, flat_dec)
        src_x = src_x_flat.reshape(dest_shape)
        src_y = src_y_flat.reshape(dest_shape)
        in_bounds = (
            (src_x >= 0.0)
            & (src_x <= float(w_c - 1))
            & (src_y >= 0.0)
            & (src_y <= float(h_c - 1))
        )
        clean_coadd = (
            torch.nan_to_num(coadd_img, nan=0.0, posinf=0.0, neginf=0.0)
            if not bool(torch.isfinite(coadd_img).all())
            else coadd_img
        )
        x_norm = (2.0 * src_x / max(w_c - 1, 1)) - 1.0
        y_norm = (2.0 * src_y / max(h_c - 1, 1)) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).to(dtype=torch.float32)
        sampled = sample_image_plane_batched_channels(
            clean_coadd.unsqueeze(0),
            x_src=src_x,
            y_src=src_y,
            grid=grid,
            kernel=kernel,
        )[0]
        sampled = torch.where(in_bounds, sampled, torch.zeros_like(sampled))
        if semantics == "flux":
            assert dest_prep.pixel_area_sr is not None
            coadd_area = pixel_area_sr_map(
                coadd_wcs, coadd_shape, device=map_device
            )
            coadd_area_at_dest = (
                F.grid_sample(
                    coadd_area.unsqueeze(0).unsqueeze(0),
                    grid.unsqueeze(0),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .squeeze(0)
                .squeeze(0)
            )
            area_ratio = dest_prep.pixel_area_sr / torch.clamp(
                coadd_area_at_dest, min=torch.finfo(torch.float32).tiny
            )
            sampled = sampled * area_ratio
        return sampled

    if isinstance(blot_target, Sequence) and not isinstance(
        blot_target, (Mapping, str, bytes)
    ):
        return [_blot_single(item) for item in blot_target]
    return _blot_single(blot_target)  # type: ignore[arg-type]


def _blot_spatial_derivative(blotted: Tensor) -> Tensor:
    """Compute AstroDrizzle / JWST stcal local spatial gradient max(|dI/dx|, |dI/dy|)."""
    H, W = blotted.shape
    padded = F.pad(
        blotted.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate"
    ).squeeze(0).squeeze(0)
    dx_left = torch.abs(blotted - padded[1 : H + 1, 0:W])
    dx_right = torch.abs(blotted - padded[1 : H + 1, 2 : W + 2])
    dy_up = torch.abs(blotted - padded[0:H, 1 : W + 1])
    dy_down = torch.abs(blotted - padded[2 : H + 2, 1 : W + 1])
    return torch.maximum(
        torch.maximum(dx_left, dx_right), torch.maximum(dy_up, dy_down)
    )


def _apply_observation_astrometric_shifts(
    observations: Sequence[Observation],
    device: torch.device | str | None,
) -> list[Observation]:
    """Apply DCR and non-sidereal moving-target ephemeris offsets to observation headers."""
    from ..wcs.iau2006 import apply_dcr, apply_ephemeris_offset

    needs_shift = any(
        obs.dcr_zenith_deg is not None
        or (obs.tracking_rate is not None and obs.mjd_obs is not None and obs.reference_mjd is not None)
        for obs in observations
    )
    if not needs_shift:
        return list(observations)

    map_device = _resolve_map_device(device)
    read_dev = str(map_device)
    shifted: list[Observation] = []
    for obs in observations:
        if obs.dcr_zenith_deg is None and not (
            obs.tracking_rate is not None
            and obs.mjd_obs is not None
            and obs.reference_mjd is not None
        ):
            shifted.append(obs)
            continue
        img, hdr = load_observation_image_and_header(obs, device=read_dev)
        hdr_copy = dict(hdr)
        ra0 = torch.tensor(float(hdr_copy.get("CRVAL1", 0.0)), dtype=torch.float64)
        dec0 = torch.tensor(float(hdr_copy.get("CRVAL2", 0.0)), dtype=torch.float64)
        if obs.dcr_zenith_deg is not None and obs.wavelength is not None:
            ra0, dec0 = apply_dcr(
                ra0,
                dec0,
                float(obs.wavelength),
                zenith_angle_deg=float(obs.dcr_zenith_deg),
                parallactic_angle_deg=float(obs.dcr_parallactic_deg),
                reference_wavelength_angstrom=float(obs.dcr_ref_wavelength),
                pressure_hpa=float(obs.dcr_pressure_hpa),
                temperature_c=float(obs.dcr_temperature_c),
            )
        if (
            obs.tracking_rate is not None
            and obs.mjd_obs is not None
            and obs.reference_mjd is not None
        ):
            ra0, dec0 = apply_ephemeris_offset(
                ra0,
                dec0,
                d_ra_cosdec_arcsec_per_hr=float(obs.tracking_rate[0]),
                d_dec_arcsec_per_hr=float(obs.tracking_rate[1]),
                mjd_obs=float(obs.mjd_obs),
                mjd_ref=float(obs.reference_mjd),
            )
        hdr_copy["CRVAL1"] = float(ra0.item())
        hdr_copy["CRVAL2"] = float(dec0.item())
        shifted.append(
            dataclasses.replace(
                obs,
                data=img,
                header=hdr_copy,
                wcs=WCS(hdr_copy),
            )
        )
    return shifted


def coadd(
    observations: Sequence[Observation],
    target: Target | Mapping[str, Any] | WCS | None = None,
    *,
    resample: ResampleConfig | None = None,
    sky: SkyConfig | None = None,
    outliers: OutlierConfig | None = None,
    stack: StackConfig | None = None,
    kernel: MapResamplingKernel | None = None,
    pixfrac: float | None = None,
    semantics: MapValueSemantics | None = None,
    method: MapStackingMethod | None = None,
    clip_sigma: float | None = None,
    clip_iterations: int | None = None,
    footprint_order: int | None = None,
    tile_size: int | tuple[int, int] | None = None,
    spatial_tile_hw: tuple[int, int] | None = None,
    target_cache_capacity: int = _DEFAULT_TARGET_CACHE_CAPACITY,
    surface_brightness_area_scaling: bool | None = None,
    output_path: str | None = None,
    scratch_dir: str | Path | None = None,
    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    cross_exposure_sky_corrector: CrossExposureSkyCorrector | None = None,
    device: torch.device | str | None = None,
) -> MapResult:
    """Unified 2D image-plane mosaic / coadd entry point.

    Accepts either structured configuration objects (``resample``, ``sky``, ``outliers``, ``stack``)
    or direct keyword arguments (``kernel``, ``pixfrac``, ``semantics``, ``method``, ``tile_size``).
    """
    r_cfg = resample or ResampleConfig()
    s_cfg = stack or StackConfig()
    resolved_kernel = kernel if kernel is not None else r_cfg.kernel
    resolved_pixfrac = pixfrac if pixfrac is not None else r_cfg.pixfrac
    resolved_semantics = semantics if semantics is not None else r_cfg.semantics
    resolved_method = method if method is not None else s_cfg.method
    resolved_clip_sigma = clip_sigma if clip_sigma is not None else s_cfg.clip_sigma
    resolved_clip_iter = (
        clip_iterations if clip_iterations is not None else s_cfg.clip_iterations
    )
    resolved_area_scaling = (
        surface_brightness_area_scaling
        if surface_brightness_area_scaling is not None
        else r_cfg.area_scaling
    )
    resolved_tile_hw = _resolve_tile_hw(tile_size, spatial_tile_hw, s_cfg)
    resolved_footprint_order = (
        footprint_order if footprint_order is not None else s_cfg.footprint_order
    )
    resolved_scratch = scratch_dir if scratch_dir is not None else s_cfg.scratch_dir

    bg_est = background_estimator
    sky_corr = cross_exposure_sky_corrector
    obs_to_stack = _apply_observation_astrometric_shifts(observations, device)
    target_header = _coerce_target_header(target)
    rejection_log: tuple[dict[str, int], ...] = ()

    if sky is not None and sky.mode != "none" and sky.subtract:
        if sky.mode in ("per_exposure", "local"):
            bg_est = MeshSkyEstimator(
                block_size=sky.mesh_size,
                statistic=sky.stat,
                mesh_filter_size=sky.smooth_scale,
                self_mask_sigma=sky.clip_sigma,
            )
        elif (
            sky.mode in ("global", "global+match")
            or resolved_method not in ("weighted_mean", "mean", "median")
        ):
            obs_to_stack, _ = match_sky(
                obs_to_stack,
                target=target_header,
                config=sky,
                apply=True,
                device=device,
            )
        elif sky.mode in ("constant", "overlap_constant", "match"):
            sky_corr = CrossExposureSkyCorrector(
                mesh_size=sky.mesh_size,
                regularization=sky.regularization,
                mode="constant",
            )
        elif sky.mode in ("planar", "overlap_planar"):
            sky_corr = CrossExposureSkyCorrector(
                mesh_size=sky.mesh_size,
                regularization=sky.regularization,
                mode="planar",
            )
        elif sky.mode in ("mesh", "overlap_mesh"):
            sky_corr = CrossExposureSkyCorrector(
                mesh_size=sky.mesh_size,
                regularization=sky.regularization,
                mode="mesh",
            )
        else:
            raise ValueError(f"Unsupported SkyConfig.mode: {sky.mode!r}")

    if outliers is not None and outliers.enabled:
        if outliers.parity_mode and resolved_kernel not in {
            "drizzle_point",
            "drizzle_square",
        }:
            raise ValueError(
                "certified outlier parity requires drizzle_point or drizzle_square"
            )
        if outliers.parity_mode and resolved_method != "weighted_mean":
            raise ValueError(
                "certified outlier parity requires weighted_mean final stacking"
            )
        init_coadd = coadd_image_plane(
            obs_to_stack,
            target_header=target_header,
            resampling_kernel=(resolved_kernel if outliers.parity_mode else "bilinear"),
            value_semantics="surface_brightness",
            stacking_method="median" if len(obs_to_stack) >= 3 else "weighted_mean",
            drizzle_pixfrac=resolved_pixfrac,
            device=device,
        )
        if outliers.parity_mode:
            _, obs_to_stack, rejection_log = detect_outliers(
                obs_to_stack,
                init_coadd,
                config=outliers,
                update_masks=True,
                return_logs=True,
                device=device,
            )
        else:
            _, obs_to_stack = detect_outliers(
                obs_to_stack,
                init_coadd,
                config=outliers,
                update_masks=True,
                device=device,
            )

    result = coadd_image_plane(
        obs_to_stack,
        target_header=target_header,
        output_path=output_path,
        footprint_order=resolved_footprint_order,
        resampling_kernel=resolved_kernel,
        value_semantics=resolved_semantics,
        stacking_method=resolved_method,
        drizzle_pixfrac=resolved_pixfrac,
        clip_sigma=resolved_clip_sigma,
        clip_iterations=resolved_clip_iter,
        spatial_tile_hw=resolved_tile_hw,
        target_cache_capacity=target_cache_capacity,
        surface_brightness_area_scaling=resolved_area_scaling,
        device=device,
        scratch_dir=resolved_scratch,
        background_estimator=bg_est,
        cross_exposure_sky_corrector=sky_corr,
    )
    if resolved_method == "weighted_mean":
        vp, vr, vf = _propagate_observation_variance_components(
            obs_to_stack,
            target_header=result.header,
            target_shape=tuple(result.data.shape),
            resampling_kernel=resolved_kernel,
            drizzle_pixfrac=resolved_pixfrac,
            value_semantics=resolved_semantics,
            device=result.data.device,
        )
        if vp is not None or vr is not None or vf is not None:
            result = dataclasses.replace(
                result, var_poisson=vp, var_rnoise=vr, var_flat=vf
            )
    if rejection_log:
        provenance = dict(result.provenance or {})
        provenance["outlier_rejection"] = {
            "mode": "certified_drizzle_parity",
            "logs": list(rejection_log),
        }
        result = dataclasses.replace(
            result,
            rejection_log=rejection_log,
            usability_mask=result.weight > 0.0,
            provenance=provenance,
        )
    return result


def coadd_multiband(
    observations: Sequence[Observation],
    target: Target | Mapping[str, Any] | WCS | None = None,
    *,
    resample: ResampleConfig | None = None,
    stack: StackConfig | None = None,
    kernel: MapResamplingKernel | None = None,
    semantics: MapValueSemantics | None = None,
    method: MapStackingMethod | None = None,
    footprint_order: int | None = None,
    spectral_unit: str = "Angstrom",
    spectral_type: str = "WAVE",
    device: torch.device | str | None = None,
) -> MultibandResult:
    """Clean multi-band image-plane coadd entry point."""
    r_cfg = resample or ResampleConfig()
    s_cfg = stack or StackConfig()
    obs_shifted = _apply_observation_astrometric_shifts(observations, device)
    return coadd_image_plane_multiband(
        obs_shifted,
        target_header=_coerce_target_header(target),
        footprint_order=footprint_order
        if footprint_order is not None
        else s_cfg.footprint_order,
        resampling_kernel=kernel if kernel is not None else r_cfg.kernel,
        value_semantics=semantics if semantics is not None else r_cfg.semantics,
        stacking_method=method if method is not None else s_cfg.method,
        spectral_unit=spectral_unit,
        spectral_type=spectral_type,
        device=device,
    )


def aperture_uncertainty(
    observations: Sequence[Observation],
    result: MapResult,
    aperture_weights: Tensor,
    *,
    resample: ResampleConfig | None = None,
    kernel: MapResamplingKernel | None = None,
    pixfrac: float | None = None,
    semantics: MapValueSemantics | None = None,
    spatial_tile_hw: tuple[int, int] | None = None,
    target_cache_capacity: int = _DEFAULT_TARGET_CACHE_CAPACITY,
    surface_brightness_area_scaling: bool | None = None,
    device: torch.device | str | None = None,
) -> ApertureUncertainty:
    """Clean alias for ``evaluate_image_plane_aperture_uncertainty``."""
    r_cfg = resample or ResampleConfig()
    return evaluate_image_plane_aperture_uncertainty(
        observations,
        result,
        aperture_weights,
        resampling_kernel=kernel if kernel is not None else r_cfg.kernel,
        value_semantics=semantics if semantics is not None else r_cfg.semantics,
        drizzle_pixfrac=pixfrac if pixfrac is not None else r_cfg.pixfrac,
        spatial_tile_hw=spatial_tile_hw,
        target_cache_capacity=target_cache_capacity,
        surface_brightness_area_scaling=(
            surface_brightness_area_scaling
            if surface_brightness_area_scaling is not None
            else r_cfg.area_scaling
        ),
        device=device,
    )
