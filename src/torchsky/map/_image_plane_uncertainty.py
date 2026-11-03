"""DQ, aperture-uncertainty, and variance-component propagation for image-plane coaddition."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torchfits import read

from ..wcs import WCS

# Header probe for area scaling: recoverable failures when reading metadata only.
from .observation import (
    ApertureUncertainty,
    ImageObservation,
    MapResult,
    Observation,
)
from .operators.config import (
    MapResamplingKernel,
    MapValueSemantics,
)
from .operators.execution import (
    PreparedImageObservationData,
)
from .operators.image_grid import (
    headers_equivalent_for_direct_copy,
    normalized_header_for_shape,
    validate_image_plane_2d,
)
from .operators.pixel_area import (
    pixel_area_sr_map,
    representative_pixel_area_sr,
)
from .operators.mask_weight import load_image_observation_dq
from .quality import DQPolicy
from .windowing import (
    _LARGE_TAN_WINDOWING_EDGE_SAMPLES,  # noqa: F401
    _LARGE_TAN_WINDOWING_MARGIN_PIXELS,  # noqa: F401
    _LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS,  # noqa: F401
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
    _resolve_map_device,
    _validated_drizzle_pixfrac,
    _validated_map_resampling_kernel,
    _validated_map_value_semantics,
)

from ._image_plane_tiling import (
    _DEFAULT_TARGET_CACHE_CAPACITY,
    _PreparedTargetTileCache,
    _obs_target_footprint,
    _validated_target_cache_capacity,
)
from ..device import _to


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
            total = torch.zeros_like(
                torch.as_tensor(components[0]), dtype=torch.float32
            )
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
    tile_hw: tuple[int, int] | None = None,
    output_device: torch.device | None = None,
) -> Tensor | None:
    """OR-propagate observation DQ flags onto the target grid.

    With *tile_hw* the projection of each flag plane is bounded to one tile at a
    time (and skipped for tiles outside the observation footprint), and the int64
    output lives on *output_device* (the host for a large target on an accelerator).
    """
    if not _observations_have_quality(observations):
        return None
    out_device = device if output_device is None else output_device
    H, W = int(target_shape[0]), int(target_shape[1])
    output = torch.zeros(target_shape, dtype=torch.int64, device=out_device)
    full_target_wcs = WCS(target_header)
    if tile_hw is None:
        tiles = [(0, H, 0, W)]
    else:
        th, tw = max(int(tile_hw[0]), 1), max(int(tile_hw[1]), 1)
        tiles = [
            (y0, min(y0 + th, H), x0, min(x0 + tw, W))
            for y0 in range(0, H, th)
            for x0 in range(0, W, tw)
        ]
    whole_target: _PreparedImagePlaneTarget | None = None
    for observation in observations:
        image, observation_header = _core_mod.load_observation_image_and_header(
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
        source_wcs = WCS(observation_header)
        footprint = (
            _obs_target_footprint(
                source_wcs, (int(image.shape[0]), int(image.shape[1])), full_target_wcs
            )
            if tile_hw is not None
            else None
        )
        tile_targets: dict[
            tuple[int, int, int, int], tuple[dict, _PreparedImagePlaneTarget]
        ] = {}
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
                    for ty0, ty1, tx0, tx1 in tiles:
                        if footprint is not None and not footprint.intersects(
                            ty0, ty1, tx0, tx1
                        ):
                            continue
                        tile_shape = (ty1 - ty0, tx1 - tx0)
                        if tile_hw is None:
                            if whole_target is None:
                                whole_target = _core_mod._prepare_target(
                                    header=target_header,
                                    target_wcs=full_target_wcs,
                                    target_shape=target_shape,
                                    device=device,
                                    include_pixel_area=(value_semantics == "flux"),
                                    include_representative_pixel_area=False,
                                    include_quadrature=(value_semantics == "flux"),
                                    include_world_grid=False,
                                )
                            tile_header, prepared_target = target_header, whole_target
                        else:
                            key = (ty0, ty1, tx0, tx1)
                            if key not in tile_targets:
                                sub_header, _ = _subtarget_header_for_window(
                                    target_header,
                                    full_shape=(H, W),
                                    y0=ty0,
                                    y1=ty1,
                                    x0=tx0,
                                    x1=tx1,
                                )
                                tile_targets[key] = (
                                    sub_header,
                                    _core_mod._prepare_canvas_tile(
                                        sub_header,
                                        tile_shape,
                                        device=device,
                                        canvas_shape=(H, W),
                                        origin_yx=(ty0, tx0),
                                        value_semantics=value_semantics,
                                        representative_pixel_area_sr=None,
                                    ),
                                )
                            tile_header, prepared_target = tile_targets[key]
                        projected = _core_mod._project_observation_to_target_chunk(
                            flag_observation,
                            image=flag_observation.data,
                            header_dict_obs=dict(observation_header),
                            source_wcs=source_wcs,
                            prepared_target=prepared_target,
                            target_header=tile_header,
                            target_shape=tile_shape,
                            device=device,
                            footprint_order=None,
                            resampling_kernel=resampling_kernel,
                            value_semantics=value_semantics,
                            drizzle_pixfrac=drizzle_pixfrac,
                            background_estimator=None,
                            propagate_variance=False,
                        )
                        flagged = projected.weight > 0.0
                        if projected.target_window is None:
                            wy0, wy1, wx0, wx1 = ty0, ty1, tx0, tx1
                        else:
                            py0, py1, px0, px1 = projected.target_window
                            if py1 <= py0 or px1 <= px0:
                                continue
                            wy0, wy1 = ty0 + py0, ty0 + py1
                            wx0, wx1 = tx0 + px0, tx0 + px1
                        if rejected:
                            flagged = flagged & (
                                output_weight[wy0:wy1, wx0:wx1].to(
                                    device=flagged.device
                                )
                                <= 0.0
                            )
                        output[wy0:wy1, wx0:wx1].bitwise_or_(
                            (flagged.to(torch.int64) * bit).to(device=out_device)
                        )
            bit <<= 1
    return output


def _propagate_aperture_uncertainty(
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
    if resampling_kernel not in _core_mod._FINITE_APERTURE_COVARIANCE_KERNELS:
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
    assert provenance is not None  # ``execution`` was read from it
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
        _observation_operator_provenance(observation) for observation in observations
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

    map_device = _resolve_map_device(device, observations, result, aperture_weights)
    aperture = _to(aperture, device=map_device, dtype=torch.float32)
    map_weight = _to(result.weight, device=map_device, dtype=torch.float32)
    inverse_variance = _to(inverse_variance, device=map_device, dtype=torch.float32)
    accumulator_dtype = torch.float32 if map_device.type == "mps" else torch.float64
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
    first_image, first_header = _core_mod.load_observation_image_and_header(
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
        representative_pixel_area_sr(WCS(header_dict), target_shape, device=map_device)
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
            image, observation_header = _core_mod.load_observation_image_and_header(
                observation, device=read_device
            )
            image = image.to(device=map_device)
        prepared = _core_mod.prepare_image_observation_data(
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
                projected = _core_mod._project_observation_to_target_chunk(
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
    prepared_target = _core_mod._prepare_target(
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
            image, header_dict_obs = _core_mod.load_observation_image_and_header(
                obs, device=read_dev
            )
            image_dev = _to(image, device=device, dtype=torch.float32)
            if comp is None:
                comp_t = torch.zeros_like(image_dev)
            elif isinstance(comp, Tensor):
                comp_t = _to(
                    validate_image_plane_2d(comp), device=device, dtype=torch.float32
                )
            else:
                comp_raw, _ = read(
                    str(comp), hdu=obs.hdu, return_header=True, device=read_dev
                )
                comp_t = _to(
                    validate_image_plane_2d(comp_raw),
                    device=device,
                    dtype=torch.float32,
                )
            source_wcs = WCS(header_dict_obs)
            s_eff = _effective_observation_photometric_scale(obs, header_dict_obs)
            calibrated_variance = comp_t * float(s_eff) ** 2
            prepared_observation = _core_mod.prepare_image_observation_data(
                obs,
                image_dev,
                device=device,
                resampling_kernel=resampling_kernel,
                load_weight_map_fn=_load_observation_weight_map,
                apply_calibration_fn=_apply_observation_calibration,
                background_estimator=None,
            )
            source_term = torch.square(
                prepared_observation.source_weight
            ) * torch.where(
                torch.isfinite(calibrated_variance) & (calibrated_variance >= 0.0),
                calibrated_variance,
                torch.zeros_like(calibrated_variance),
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
            actual = _core_mod._project_observation_to_target_chunk(
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
            squared = _core_mod._project_observation_to_target_chunk(
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


# Late-bound so monkeypatching ``image_plane_coadd`` attributes still reaches this code.
from . import image_plane_coadd as _core_mod  # noqa: E402
