"""Known-component deblending on native calibrated exposures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
from typing import Literal

import torch
from torch import Tensor

from ..map.field.observation_ops import observation_support_mask
from ..map.field.types import FieldObservation
from ..scene.sources import (
    GAUSSIAN_MORPHOLOGY,
    POINT_MORPHOLOGY,
    SERSIC_MORPHOLOGY,
    TEMPLATE_MORPHOLOGY,
    SourceCatalogScene,
    SourceStamp,
)
from ..wcs import WCS
from .catalog import Catalog
from .photometry import (
    BackgroundModel,
    BlockSparseCovariance,
    ForwardPhotometryResult,
    SourceGrouping,
    _rank_analysis,
    fit_forced_photometry,
)


FailurePolicy = Literal["raise", "return"]


@dataclass(frozen=True)
class ParametricBounds:
    position_arcsec: float = 2.0
    gaussian_sigma_arcsec: tuple[float, float] = (0.05, 30.0)
    sersic_radius_arcsec: tuple[float, float] = (0.05, 60.0)
    sersic_axis_ratio: tuple[float, float] = (0.05, 1.0)
    sersic_index: tuple[float, float] = (0.3, 8.0)
    angle_delta_deg: float = 180.0

    def __post_init__(self) -> None:
        positive = (
            self.position_arcsec,
            *self.gaussian_sigma_arcsec,
            *self.sersic_radius_arcsec,
            *self.sersic_axis_ratio,
            *self.sersic_index,
            self.angle_delta_deg,
        )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0 for value in positive
        ):
            raise ValueError("all parametric bounds must be finite and positive")
        for lower, upper in (
            self.gaussian_sigma_arcsec,
            self.sersic_radius_arcsec,
            self.sersic_axis_ratio,
            self.sersic_index,
        ):
            if lower >= upper:
                raise ValueError(
                    "each parametric lower bound must be below its upper bound"
                )
        if self.sersic_axis_ratio[1] > 1.0:
            raise ValueError("Sersic axis-ratio upper bound cannot exceed one")
        if self.sersic_index[0] < 0.3 or self.sersic_index[1] > 8.0:
            raise ValueError("Sersic-index bounds must remain inside [0.3, 8]")


@dataclass(frozen=True)
class FitAdequacyDiagnostics:
    evaluated_pixel_count: int
    fitted_parameter_count: int
    degrees_of_freedom: int
    chi_square: Tensor
    reduced_chi_square: Tensor
    upper_reduced_chi_square: Tensor
    heldout_log_likelihood: Tensor | None
    evaluation: Literal["heldout", "in_sample"]
    inadequate: bool
    reason: str | None


@dataclass(frozen=True)
class ParametricBlendDiagnostics:
    group_index: int
    source_indices: Tensor
    component_ids: Tensor
    parameter_indices: Tensor
    parameter_labels: tuple[str, ...]
    converged: bool
    iterations: int
    initial_objective: Tensor
    final_objective: Tensor
    damping: float
    rank: int
    condition_number: float
    gauge_parameter_indices: Tensor
    covariance: Tensor
    catastrophic_swap: bool
    failure_reason: str | None


@dataclass(frozen=True)
class FactorizationBlendDiagnostics:
    group_index: int
    source_indices: Tensor
    component_ids: Tensor
    converged: bool
    iterations: int
    initial_objective: Tensor
    final_objective: Tensor
    accepted_steps: int
    final_step_size: float
    morphology_sums: Tensor
    zero_morphology_indices: Tensor
    spectrum_gauge_parameter_indices: Tensor
    failure_reason: str | None


@dataclass(frozen=True)
class ParametricDeblendResult:
    catalog: Catalog
    scene: SourceCatalogScene
    initial_scene: SourceCatalogScene
    coefficients: Tensor
    covariance: BlockSparseCovariance
    nonlinear_covariance: tuple[Tensor, ...]
    predictions: tuple[Tensor, ...]
    residuals: tuple[Tensor, ...]
    grouping: SourceGrouping
    diagnostics: tuple[ParametricBlendDiagnostics, ...]
    linear_diagnostics: tuple[object, ...]
    adequacy: FitAdequacyDiagnostics
    background_coefficients: tuple[Tensor, ...]
    provenance: dict[str, object]

    @property
    def fluxes(self) -> Tensor:
        return self.catalog.columns["flux"]

    @property
    def flux(self) -> Tensor:
        return self.fluxes


@dataclass(frozen=True)
class MorphologySpectrumDeblendResult:
    catalog: Catalog
    scene: SourceCatalogScene
    initial_scene: SourceCatalogScene
    spectra: Tensor
    initial_spectra: Tensor
    morphologies: tuple[Tensor, ...]
    initial_morphologies: tuple[Tensor, ...]
    covariance: BlockSparseCovariance
    predictions: tuple[Tensor, ...]
    residuals: tuple[Tensor, ...]
    grouping: SourceGrouping
    diagnostics: tuple[FactorizationBlendDiagnostics, ...]
    linear_diagnostics: tuple[object, ...]
    adequacy: FitAdequacyDiagnostics
    background_coefficients: tuple[Tensor, ...]
    provenance: dict[str, object]

    @property
    def fluxes(self) -> Tensor:
        return self.spectra

    @property
    def flux(self) -> Tensor:
        return self.spectra


class KnownComponentFitError(RuntimeError):
    def __init__(self, message: str, result: object) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class _ParameterEntry:
    component: int
    field: str
    offset: int
    label: str
    lower: float
    upper: float


def _coefficient_rows(
    scene: SourceCatalogScene, values: Tensor | None = None
) -> Tensor:
    state = scene.coefficients if values is None else values
    rows = [
        scene.component_parameters(index, state)
        for index in range(scene.component_count)
    ]
    width = (
        0
        if scene.component_count == 0
        else int(scene.parameter_slices[0, 1] - scene.parameter_slices[0, 0])
    )
    return torch.stack(rows) if rows else scene.lon.new_empty((0, width))


def _copy_scene(
    scene: SourceCatalogScene,
    *,
    lon: Tensor | None = None,
    lat: Tensor | None = None,
    gaussian_parameters: Tensor | None = None,
    sersic_parameters: Tensor | None = None,
    template_data: Tensor | None = None,
    coefficients: Tensor | None = None,
) -> SourceCatalogScene:
    return SourceCatalogScene(
        scene.lon if lon is None else lon,
        scene.lat if lat is None else lat,
        component_ids=scene.component_ids,
        morphology=scene.morphology_codes,
        gaussian_parameters=(
            scene.gaussian_parameters
            if gaussian_parameters is None
            else gaussian_parameters
        ),
        sersic_parameters=(
            scene.sersic_parameters if sersic_parameters is None else sersic_parameters
        ),
        template_data=scene.template_data if template_data is None else template_data,
        template_slices=scene.template_slices,
        template_shapes=scene.template_shapes,
        template_pixel_scale_arcsec=scene.template_pixel_scale_arcsec,
        coefficients=_coefficient_rows(scene) if coefficients is None else coefficients,
        band_labels=scene.band_labels,
        spectral_basis=scene.spectral_basis,
        parameter_slices=scene.parameter_slices,
        support_uniq=scene.support_uniq,
        support_slices=scene.support_slices,
        support_radius_arcsec=scene.support_radius_arcsec,
        footprint_order=scene.footprint_order,
    )


def _component_mask(
    value: bool | Tensor, scene: SourceCatalogScene, *, name: str
) -> Tensor:
    if isinstance(value, bool):
        return torch.full(
            (scene.component_count,), value, dtype=torch.bool, device=scene.device
        )
    mask = torch.as_tensor(value, dtype=torch.bool, device=scene.device).reshape(-1)
    if mask.numel() != scene.component_count:
        raise ValueError(f"{name} must be bool or match scene.component_count")
    return mask


def _validate_holdout_masks(
    observations: Sequence[FieldObservation],
    heldout_masks: Sequence[Tensor] | None,
) -> tuple[Tensor, ...] | None:
    if heldout_masks is None:
        return None
    if len(heldout_masks) != len(observations):
        raise ValueError("heldout_masks must match observations")
    result = []
    for observation, mask in zip(observations, heldout_masks, strict=True):
        value = torch.as_tensor(mask, dtype=torch.bool, device=observation.image.device)
        if value.shape != observation.image.shape:
            raise ValueError("each heldout mask must match its observation image")
        result.append(value)
    return tuple(result)


def _training_observations(
    observations: Sequence[FieldObservation],
    heldout_masks: tuple[Tensor, ...] | None,
) -> tuple[FieldObservation, ...]:
    if heldout_masks is None:
        return tuple(observations)
    result = []
    for observation, heldout in zip(observations, heldout_masks, strict=True):
        valid = observation_support_mask(observation) & ~heldout
        result.append(replace(observation, mask=valid))
    return tuple(result)


def _union_bounds(stamps: Sequence[SourceStamp]) -> tuple[int, int, int, int]:
    return (
        min(int(stamp.y_slice.start) for stamp in stamps),
        max(int(stamp.y_slice.stop) for stamp in stamps),
        min(int(stamp.x_slice.start) for stamp in stamps),
        max(int(stamp.x_slice.stop) for stamp in stamps),
    )


def _group_bounds(
    scene: SourceCatalogScene,
    observations: Sequence[FieldObservation],
    grouping: SourceGrouping,
    *,
    padding_arcsec: float,
    chunk_size: int | None,
) -> tuple[dict[int, tuple[int, int, int, int]], ...]:
    bounds_per_group = []
    for sources, exposures in zip(
        grouping.blend_groups, grouping.group_exposures, strict=True
    ):
        current: dict[int, tuple[int, int, int, int]] = {}
        for exposure_value in exposures.tolist():
            exposure_index = int(exposure_value)
            observation = observations[exposure_index]
            stamps = [
                stamp
                for source_value in sources.tolist()
                if (
                    stamp := scene.render_component(
                        int(source_value), observation, chunk_size=chunk_size
                    )
                )
                is not None
            ]
            if not stamps:
                continue
            y0, y1, x0, x1 = _union_bounds(stamps)
            height, width = observation.image.shape
            wcs = WCS(observation.header).to(scene.device)
            x = scene.lon.new_tensor(0.5 * (width - 1))
            y = scene.lon.new_tensor(0.5 * (height - 1))
            scale = scene._pixel_scale_arcsec(wcs, x, y)
            padding = int(math.ceil(padding_arcsec / scale)) + 2
            current[exposure_index] = (
                max(0, y0 - padding),
                min(int(height), y1 + padding),
                max(0, x0 - padding),
                min(int(width), x1 + padding),
            )
        bounds_per_group.append(current)
    return tuple(bounds_per_group)


def _weighted_residual(
    result: ForwardPhotometryResult,
    observations: Sequence[FieldObservation],
    bounds: dict[int, tuple[int, int, int, int]],
) -> Tensor:
    values = []
    for exposure_index, (y0, y1, x0, x1) in bounds.items():
        observation = observations[exposure_index]
        valid = observation_support_mask(observation)[y0:y1, x0:x1]
        residual = result.residuals[exposure_index][y0:y1, x0:x1]
        weight = (
            torch.ones_like(residual)
            if observation.weight is None
            else observation.weight[y0:y1, x0:x1].to(residual).clamp_min(0)
        )
        values.append((residual[valid] * torch.sqrt(weight[valid])).reshape(-1))
    if values:
        return torch.cat(values)
    return result.coefficients.new_empty((0,))


def _adequacy(
    observations: Sequence[FieldObservation],
    predictions: Sequence[Tensor],
    heldout_masks: tuple[Tensor, ...] | None,
    *,
    fitted_parameter_count: int,
    sigma: float,
) -> FitAdequacyDiagnostics:
    terms = []
    log_terms = []
    count = 0
    for index, (observation, prediction) in enumerate(
        zip(observations, predictions, strict=True)
    ):
        valid = observation_support_mask(observation)
        if heldout_masks is not None:
            valid = valid & heldout_masks[index]
        residual = observation.image - prediction.to(observation.image)
        weight = (
            torch.ones_like(residual)
            if observation.weight is None
            else observation.weight.to(residual).clamp_min(0)
        )
        selected_weight = weight[valid]
        selected_residual = residual[valid]
        terms.append(torch.sum(selected_weight * selected_residual.square()))
        if heldout_masks is not None:
            log_terms.append(
                0.5
                * torch.sum(
                    torch.log(selected_weight.clamp_min(torch.finfo(weight.dtype).tiny))
                    - math.log(2.0 * math.pi)
                    - selected_weight * selected_residual.square()
                )
            )
        count += int(valid.sum().item())
    reference = predictions[0] if predictions else torch.tensor(0.0)
    chi_square = torch.stack(terms).sum() if terms else reference.new_zeros(())
    dof = max(count - int(fitted_parameter_count), 1)
    reduced = chi_square / dof
    upper = reduced.new_tensor(1.0 + float(sigma) * math.sqrt(2.0 / dof))
    inadequate = not bool(torch.isfinite(reduced)) or bool(reduced > upper)
    reason = "chi_square_exceeds_expected_noise" if inadequate else None
    heldout_log_likelihood = torch.stack(log_terms).sum() if log_terms else None
    return FitAdequacyDiagnostics(
        evaluated_pixel_count=count,
        fitted_parameter_count=int(fitted_parameter_count),
        degrees_of_freedom=dof,
        chi_square=chi_square,
        reduced_chi_square=reduced,
        upper_reduced_chi_square=upper,
        heldout_log_likelihood=heldout_log_likelihood,
        evaluation="heldout" if heldout_masks is not None else "in_sample",
        inadequate=inadequate,
        reason=reason,
    )


def _parameter_layout(
    scene: SourceCatalogScene,
    bounds: ParametricBounds,
    refine_positions: Tensor,
    refine_shapes: Tensor,
) -> tuple[Tensor, Tensor, Tensor, tuple[_ParameterEntry, ...]]:
    values: list[Tensor] = []
    lower: list[float] = []
    upper: list[float] = []
    entries: list[_ParameterEntry] = []

    def append(
        component: int,
        field: str,
        offset: int,
        label: str,
        value: Tensor,
        low: float,
        high: float,
    ) -> None:
        scalar = float(value.detach().item())
        if scalar < low or scalar > high:
            raise ValueError(f"initial {label} is outside the requested bounds")
        values.append(value)
        lower.append(low)
        upper.append(high)
        entries.append(_ParameterEntry(component, field, offset, label, low, high))

    for component in range(scene.component_count):
        if bool(refine_positions[component]):
            for offset, direction in enumerate(("east", "north")):
                append(
                    component,
                    "position",
                    offset,
                    f"component[{component}].{direction}_arcsec",
                    scene.lon.new_zeros(()),
                    -bounds.position_arcsec,
                    bounds.position_arcsec,
                )
        if not bool(refine_shapes[component]):
            continue
        code = int(scene.morphology_codes[component])
        if code == GAUSSIAN_MORPHOLOGY:
            parameters = scene.gaussian_parameters[component]
            append(
                component,
                "gaussian",
                0,
                f"component[{component}].major_sigma_arcsec",
                parameters[0],
                *bounds.gaussian_sigma_arcsec,
            )
            append(
                component,
                "gaussian",
                1,
                f"component[{component}].minor_sigma_arcsec",
                parameters[1],
                *bounds.gaussian_sigma_arcsec,
            )
            append(
                component,
                "gaussian",
                2,
                f"component[{component}].position_angle_deg",
                parameters[2],
                float(parameters[2]) - bounds.angle_delta_deg,
                float(parameters[2]) + bounds.angle_delta_deg,
            )
        elif code == SERSIC_MORPHOLOGY:
            parameters = scene.sersic_parameters[component]
            append(
                component,
                "sersic",
                0,
                f"component[{component}].half_light_radius_arcsec",
                parameters[0],
                *bounds.sersic_radius_arcsec,
            )
            append(
                component,
                "sersic",
                1,
                f"component[{component}].axis_ratio",
                parameters[1],
                *bounds.sersic_axis_ratio,
            )
            append(
                component,
                "sersic",
                2,
                f"component[{component}].position_angle_deg",
                parameters[2],
                float(parameters[2]) - bounds.angle_delta_deg,
                float(parameters[2]) + bounds.angle_delta_deg,
            )
            append(
                component,
                "sersic",
                3,
                f"component[{component}].sersic_index",
                parameters[3],
                *bounds.sersic_index,
            )
    if values:
        return (
            torch.stack(values),
            scene.lon.new_tensor(lower),
            scene.lon.new_tensor(upper),
            tuple(entries),
        )
    empty = scene.lon.new_empty((0,))
    return empty, empty, empty, ()


def _project_parameters(
    values: Tensor,
    lower: Tensor,
    upper: Tensor,
    entries: tuple[_ParameterEntry, ...],
    bounds: ParametricBounds,
) -> Tensor:
    projected = torch.maximum(torch.minimum(values, upper), lower)
    positions: dict[int, list[int]] = {}
    gaussian: dict[int, dict[int, int]] = {}
    for index, entry in enumerate(entries):
        if entry.field == "position":
            positions.setdefault(entry.component, []).append(index)
        elif entry.field == "gaussian":
            gaussian.setdefault(entry.component, {})[entry.offset] = index
    for indices in positions.values():
        if len(indices) == 2:
            radius = torch.linalg.vector_norm(projected[indices])
            scale = torch.clamp(
                bounds.position_arcsec
                / radius.clamp_min(torch.finfo(projected.dtype).tiny),
                max=1.0,
            )
            projected[indices] = projected[indices] * scale
    for offsets in gaussian.values():
        if 0 in offsets and 1 in offsets:
            major = offsets[0]
            minor = offsets[1]
            projected[minor] = torch.minimum(projected[minor], projected[major])
    return projected


def _scene_from_parameters(
    scene: SourceCatalogScene,
    values: Tensor,
    entries: tuple[_ParameterEntry, ...],
) -> SourceCatalogScene:
    east = torch.zeros_like(scene.lon)
    north = torch.zeros_like(scene.lat)
    gaussian = scene.gaussian_parameters.clone()
    sersic = scene.sersic_parameters.clone()
    for value, entry in zip(values, entries, strict=True):
        if entry.field == "position":
            target = east if entry.offset == 0 else north
            target[entry.component] = value
        elif entry.field == "gaussian":
            gaussian[entry.component, entry.offset] = value
        else:
            sersic[entry.component, entry.offset] = value
    cos_lat = torch.cos(torch.deg2rad(scene.lat)).abs().clamp_min(1.0e-8)
    lon = torch.remainder(scene.lon + east / (3600.0 * cos_lat), 360.0)
    lat = scene.lat + north / 3600.0
    return _copy_scene(
        scene,
        lon=lon,
        lat=lat,
        gaussian_parameters=gaussian,
        sersic_parameters=sersic,
    )


def _fit_linear(
    observations: Sequence[FieldObservation],
    scene: SourceCatalogScene,
    *,
    solver: Literal["signed", "nnls"],
    background: BackgroundModel,
    index_nside: int | None,
    footprint_order: int | None,
    chunk_size: int | None,
    rank_rtol: float | None,
    rank_atol: float,
) -> ForwardPhotometryResult:
    return fit_forced_photometry(
        observations,
        scene,
        solver=solver,
        background=background,
        index_nside=index_nside,
        footprint_order=footprint_order,
        chunk_size=chunk_size,
        rank_rtol=rank_rtol,
        rank_atol=rank_atol,
    )


def _catastrophic_swap(
    initial: SourceCatalogScene,
    refined: SourceCatalogScene,
    sources: Tensor,
) -> bool:
    if sources.numel() < 2:
        return False
    initial_lon = initial.lon[sources]
    initial_lat = initial.lat[sources]
    refined_lon = refined.lon[sources]
    refined_lat = refined.lat[sources]
    cos_lat = torch.cos(torch.deg2rad(initial_lat)).abs().clamp_min(1.0e-8)
    east = (
        (refined_lon[:, None] - initial_lon[None, :] + 180.0).remainder(360.0) - 180.0
    ) * cos_lat[None, :]
    north = refined_lat[:, None] - initial_lat[None, :]
    nearest = torch.argmin(east.square() + north.square(), dim=1)
    return not bool(
        torch.equal(nearest, torch.arange(sources.numel(), device=sources.device))
    )


def _require_deblend_controls(
    *,
    max_iterations: int,
    max_group_size: int,
    adequacy_sigma: float,
    failure_policy: str,
) -> None:
    if max_iterations < 1 or max_group_size < 1:
        raise ValueError("max_iterations and max_group_size must be positive")
    if not math.isfinite(float(adequacy_sigma)) or adequacy_sigma <= 0:
        raise ValueError("adequacy_sigma must be finite and positive")
    if failure_policy not in {"raise", "return"}:
        raise ValueError("failure_policy must be 'raise' or 'return'")


def fit_parametric_deblending(
    observations: Sequence[FieldObservation],
    scene: SourceCatalogScene,
    *,
    refine_positions: bool | Tensor = True,
    refine_shapes: bool | Tensor = True,
    bounds: ParametricBounds | None = None,
    linear_solver: Literal["signed", "nnls"] = "nnls",
    background: BackgroundModel = "constant",
    max_iterations: int = 20,
    initial_damping: float = 1.0e-2,
    finite_difference_step: float = 1.0e-3,
    step_tolerance: float = 1.0e-4,
    objective_tolerance: float = 1.0e-8,
    max_group_size: int = 64,
    heldout_masks: Sequence[Tensor] | None = None,
    adequacy_sigma: float = 5.0,
    failure_policy: FailurePolicy = "raise",
    index_nside: int | None = None,
    footprint_order: int | None = None,
    chunk_size: int | None = None,
    rank_rtol: float | None = None,
    rank_atol: float = 0.0,
) -> ParametricDeblendResult:
    """Refine supplied point, Gaussian, and bounded-Sersic components by LM."""
    if not isinstance(scene, SourceCatalogScene):
        raise TypeError("scene must be a SourceCatalogScene")
    allowed = (
        (scene.morphology_codes == POINT_MORPHOLOGY)
        | (scene.morphology_codes == GAUSSIAN_MORPHOLOGY)
        | (scene.morphology_codes == SERSIC_MORPHOLOGY)
    )
    if not bool(allowed.all()):
        raise ValueError(
            "parametric deblending accepts only point, Gaussian, and Sersic components"
        )
    _require_deblend_controls(
        max_iterations=max_iterations,
        max_group_size=max_group_size,
        adequacy_sigma=adequacy_sigma,
        failure_policy=failure_policy,
    )
    for value, name in (
        (initial_damping, "initial_damping"),
        (finite_difference_step, "finite_difference_step"),
        (step_tolerance, "step_tolerance"),
        (objective_tolerance, "objective_tolerance"),
    ):
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    bounds = ParametricBounds() if bounds is None else bounds
    position_mask = _component_mask(refine_positions, scene, name="refine_positions")
    shape_mask = _component_mask(refine_shapes, scene, name="refine_shapes")
    holdout = _validate_holdout_masks(observations, heldout_masks)
    training = _training_observations(observations, holdout)
    values, lower, upper, entries = _parameter_layout(
        scene, bounds, position_mask, shape_mask
    )
    initial_scene = scene
    initial_fit = _fit_linear(
        training,
        scene,
        solver=linear_solver,
        background=background,
        index_nside=index_nside,
        footprint_order=footprint_order,
        chunk_size=chunk_size,
        rank_rtol=rank_rtol,
        rank_atol=rank_atol,
    )
    grouping = initial_fit.grouping
    if any(group.numel() > max_group_size for group in grouping.blend_groups):
        raise ValueError("a connected blend group exceeds max_group_size")
    fixed_bounds = _group_bounds(
        scene,
        training,
        grouping,
        padding_arcsec=bounds.position_arcsec,
        chunk_size=chunk_size,
    )
    diagnostics: list[ParametricBlendDiagnostics] = []
    current_fit = initial_fit
    for group_index, sources in enumerate(grouping.blend_groups):
        selected = torch.tensor(
            [
                index
                for index, entry in enumerate(entries)
                if entry.component in set(int(value) for value in sources.tolist())
            ],
            dtype=torch.int64,
            device=scene.device,
        )
        residual = _weighted_residual(current_fit, training, fixed_bounds[group_index])
        initial_objective = 0.5 * torch.dot(residual, residual)
        objective = initial_objective
        damping = float(initial_damping)
        converged = selected.numel() == 0
        failure_reason: str | None = None
        information = values.new_zeros((selected.numel(), selected.numel()))
        iterations = 0
        if residual.numel() == 0 and selected.numel():
            failure_reason = "no_valid_pixels"
        elif selected.numel():
            for iteration in range(max_iterations):
                iterations = iteration + 1
                columns = []
                for global_index in selected.tolist():
                    step = finite_difference_step * max(
                        abs(float(values[global_index].detach().item())), 1.0
                    )
                    plus = values.clone()
                    minus = values.clone()
                    plus[global_index] += step
                    minus[global_index] -= step
                    plus = _project_parameters(plus, lower, upper, entries, bounds)
                    minus = _project_parameters(minus, lower, upper, entries, bounds)
                    denominator = plus[global_index] - minus[global_index]
                    if bool(denominator.abs() <= torch.finfo(values.dtype).eps):
                        columns.append(torch.zeros_like(residual))
                        continue
                    plus_scene = _scene_from_parameters(scene, plus, entries)
                    minus_scene = _scene_from_parameters(scene, minus, entries)
                    plus_fit = _fit_linear(
                        training,
                        plus_scene,
                        solver=linear_solver,
                        background=background,
                        index_nside=index_nside,
                        footprint_order=footprint_order,
                        chunk_size=chunk_size,
                        rank_rtol=rank_rtol,
                        rank_atol=rank_atol,
                    )
                    minus_fit = _fit_linear(
                        training,
                        minus_scene,
                        solver=linear_solver,
                        background=background,
                        index_nside=index_nside,
                        footprint_order=footprint_order,
                        chunk_size=chunk_size,
                        rank_rtol=rank_rtol,
                        rank_atol=rank_atol,
                    )
                    plus_residual = _weighted_residual(
                        plus_fit, training, fixed_bounds[group_index]
                    )
                    minus_residual = _weighted_residual(
                        minus_fit, training, fixed_bounds[group_index]
                    )
                    columns.append((plus_residual - minus_residual) / denominator)
                jacobian = torch.stack(columns, dim=1)
                information = jacobian.T @ jacobian
                gradient = jacobian.T @ residual
                if bool(
                    torch.linalg.vector_norm(gradient)
                    <= step_tolerance
                    * torch.linalg.vector_norm(residual).clamp_min(1.0)
                ):
                    converged = True
                    break
                diagonal = torch.diagonal(information).clamp_min(1.0)
                system = information + damping * torch.diag(diagonal)
                analysis_system = (
                    system.to("cpu") if system.device.type == "mps" else system
                )
                analysis_gradient = (
                    gradient.to("cpu") if gradient.device.type == "mps" else gradient
                )
                try:
                    delta = torch.linalg.solve(analysis_system, -analysis_gradient).to(
                        values
                    )
                except RuntimeError:
                    delta = (
                        torch.linalg.pinv(analysis_system) @ -analysis_gradient
                    ).to(values)
                trial_values = values.clone()
                trial_values[selected] += delta
                trial_values = _project_parameters(
                    trial_values, lower, upper, entries, bounds
                )
                trial_scene = _scene_from_parameters(scene, trial_values, entries)
                trial_fit = _fit_linear(
                    training,
                    trial_scene,
                    solver=linear_solver,
                    background=background,
                    index_nside=index_nside,
                    footprint_order=footprint_order,
                    chunk_size=chunk_size,
                    rank_rtol=rank_rtol,
                    rank_atol=rank_atol,
                )
                trial_residual = _weighted_residual(
                    trial_fit, training, fixed_bounds[group_index]
                )
                trial_objective = 0.5 * torch.dot(trial_residual, trial_residual)
                if bool(torch.isfinite(trial_objective)) and bool(
                    trial_objective < objective
                ):
                    previous = objective
                    values = trial_values
                    current_fit = trial_fit
                    residual = trial_residual
                    objective = trial_objective
                    damping = max(damping / 3.0, torch.finfo(values.dtype).eps)
                    relative_step = torch.linalg.vector_norm(delta) / (
                        torch.linalg.vector_norm(values[selected]).clamp_min(1.0)
                    )
                    relative_objective = (
                        previous - objective
                    ).abs() / previous.abs().clamp_min(1.0)
                    if bool(relative_step <= step_tolerance) or bool(
                        relative_objective <= objective_tolerance
                    ):
                        converged = True
                        break
                else:
                    damping *= 10.0
                    if not math.isfinite(damping):
                        failure_reason = "non_finite_damping"
                        break
            if not converged and failure_reason is None:
                failure_reason = "maximum_iterations"
        eigenvalues, cutoff, rank = _rank_analysis(
            information,
            rank_rtol=rank_rtol,
            rank_atol=rank_atol,
        )
        gauge_mask = torch.diagonal(information).abs() <= cutoff
        if information.numel() and rank < information.shape[0]:
            analysis = (
                information.to("cpu")
                if information.device.type == "mps"
                else information
            )
            null_values, null_vectors = torch.linalg.eigh(0.5 * (analysis + analysis.T))
            null_vectors = null_vectors[:, null_values <= cutoff.to(analysis)]
            if null_vectors.numel():
                involved = null_vectors.abs().amax(dim=1) > math.sqrt(
                    torch.finfo(information.dtype).eps
                )
                gauge_mask = gauge_mask | involved.to(gauge_mask.device)
        gauge_local = torch.nonzero(gauge_mask, as_tuple=False).flatten()
        covariance = torch.full_like(information, float("nan"))
        if information.numel() and rank == information.shape[0]:
            analysis = (
                information.to("cpu")
                if information.device.type == "mps"
                else information
            )
            covariance = torch.linalg.inv(analysis).to(information)
            covariance = 0.5 * (covariance + covariance.T)
        positive = eigenvalues[eigenvalues > cutoff]
        condition = (
            math.inf
            if positive.numel() == 0 or rank < information.shape[0]
            else float((positive[-1] / positive[0]).detach().item())
        )
        refined_scene = _scene_from_parameters(scene, values, entries)
        swapped = _catastrophic_swap(initial_scene, refined_scene, sources)
        if swapped:
            converged = False
            failure_reason = "catastrophic_component_swap"
        diagnostics.append(
            ParametricBlendDiagnostics(
                group_index=group_index,
                source_indices=sources,
                component_ids=scene.component_ids[sources],
                parameter_indices=selected,
                parameter_labels=tuple(
                    entries[index].label for index in selected.tolist()
                ),
                converged=converged,
                iterations=iterations,
                initial_objective=initial_objective,
                final_objective=objective,
                damping=damping,
                rank=rank,
                condition_number=condition,
                gauge_parameter_indices=selected[gauge_local],
                covariance=covariance,
                catastrophic_swap=swapped,
                failure_reason=failure_reason,
            )
        )
    refined_geometry = _scene_from_parameters(scene, values, entries)
    final_fit = _fit_linear(
        training,
        refined_geometry,
        solver=linear_solver,
        background=background,
        index_nside=index_nside,
        footprint_order=footprint_order,
        chunk_size=chunk_size,
        rank_rtol=rank_rtol,
        rank_atol=rank_atol,
    )
    fitted_scene = _copy_scene(
        refined_geometry,
        coefficients=_coefficient_rows(
            refined_geometry, torch.nan_to_num(final_fit.coefficients, nan=0.0)
        ),
    )
    fitted_count = (
        final_fit.coefficients.numel()
        + values.numel()
        + sum(value.numel() for value in final_fit.background_coefficients)
    )
    adequacy = _adequacy(
        observations,
        final_fit.predictions,
        holdout,
        fitted_parameter_count=fitted_count,
        sigma=adequacy_sigma,
    )
    catalog = Catalog(
        lon=fitted_scene.lon,
        lat=fitted_scene.lat,
        columns={
            **final_fit.catalog.columns,
            "component_id": fitted_scene.component_ids,
            "morphology_code": fitted_scene.morphology_codes,
            "gaussian_parameters": fitted_scene.gaussian_parameters,
            "sersic_parameters": fitted_scene.sersic_parameters,
        },
    )
    provenance: dict[str, object] = {
        **final_fit.provenance,
        "method": "known-component-parametric-variable-projection-lm",
        "component_count_preserved": True,
        "component_ids_preserved": True,
        "source_detection": False,
        "source_birth_death": False,
        "nonlinear_solver": "finite-difference-variable-projection-levenberg-marquardt",
        "bounds": bounds,
        "heldout_evaluation": holdout is not None,
    }
    result = ParametricDeblendResult(
        catalog=catalog,
        scene=fitted_scene,
        initial_scene=initial_scene,
        coefficients=final_fit.coefficients,
        covariance=final_fit.covariance,
        nonlinear_covariance=tuple(item.covariance for item in diagnostics),
        predictions=final_fit.predictions,
        residuals=tuple(
            observation.image - prediction.to(observation.image)
            for observation, prediction in zip(
                observations, final_fit.predictions, strict=True
            )
        ),
        grouping=final_fit.grouping,
        diagnostics=tuple(diagnostics),
        linear_diagnostics=tuple(final_fit.diagnostics),
        adequacy=adequacy,
        background_coefficients=final_fit.background_coefficients,
        provenance=provenance,
    )
    failures = [item.failure_reason for item in diagnostics if item.failure_reason]
    if adequacy.inadequate:
        failures.append(adequacy.reason or "inadequate_model")
    if failures and failure_policy == "raise":
        raise KnownComponentFitError(
            "known-component parametric fit failed: " + ", ".join(failures), result
        )
    return result


def _template_tuple(scene: SourceCatalogScene, data: Tensor) -> tuple[Tensor, ...]:
    values = []
    for component in range(scene.component_count):
        start, stop = (
            int(value) for value in scene.template_slices[component].tolist()
        )
        height, width = (
            int(value) for value in scene.template_shapes[component].tolist()
        )
        values.append(data[start:stop].reshape(height, width))
    return tuple(values)


def _project_morphology(
    value: Tensor,
    *,
    monotonic: bool,
    symmetric: bool,
) -> Tensor:
    projected = value.clamp_min(0)
    if symmetric:
        projected = 0.5 * (projected + torch.flip(projected, dims=(0, 1)))
    if monotonic:
        height, width = projected.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=value.device),
            torch.arange(width, device=value.device),
            indexing="ij",
        )
        radius2 = (2 * yy - (height - 1)).square() + (2 * xx - (width - 1)).square()
        previous = projected.new_tensor(torch.inf)
        output = projected.clone()
        for radius in torch.unique(radius2, sorted=True):
            mask = radius2 == radius
            output[mask] = torch.minimum(output[mask], previous)
            previous = output[mask].amax()
        projected = output
    total = projected.sum()
    if not bool(torch.isfinite(total)) or not bool(total > 0):
        raise ValueError("a morphology projection produced zero or non-finite support")
    return projected / total


def _project_template_data(
    scene: SourceCatalogScene,
    data: Tensor,
    *,
    monotonic: Tensor,
    symmetric: Tensor,
    selected_sources: set[int] | None = None,
) -> Tensor:
    pieces = []
    cursor = 0
    for component in range(scene.component_count):
        start, stop = (
            int(value) for value in scene.template_slices[component].tolist()
        )
        if start > cursor:
            pieces.append(data[cursor:start])
        height, width = (
            int(value) for value in scene.template_shapes[component].tolist()
        )
        value = data[start:stop].reshape(height, width)
        if selected_sources is None or component in selected_sources:
            value = _project_morphology(
                value,
                monotonic=bool(monotonic[component]),
                symmetric=bool(symmetric[component]),
            )
        pieces.append(value.reshape(-1))
        cursor = stop
    if cursor < data.numel():
        pieces.append(data[cursor:])
    return torch.cat(pieces) if pieces else data.new_empty((0,))


def fit_morphology_spectrum_deblending(
    observations: Sequence[FieldObservation],
    scene: SourceCatalogScene,
    *,
    monotonic: bool | Tensor = False,
    symmetric: bool | Tensor = False,
    background: BackgroundModel = "constant",
    max_iterations: int = 50,
    step_size: float = 0.1,
    backtracking: float = 0.5,
    step_tolerance: float = 1.0e-5,
    objective_tolerance: float = 1.0e-8,
    max_group_size: int = 64,
    heldout_masks: Sequence[Tensor] | None = None,
    adequacy_sigma: float = 5.0,
    failure_policy: FailurePolicy = "raise",
    index_nside: int | None = None,
    footprint_order: int | None = None,
    chunk_size: int | None = None,
    rank_rtol: float | None = None,
    rank_atol: float = 0.0,
) -> MorphologySpectrumDeblendResult:
    """Fit fixed-count nonnegative component morphologies and spectra by blocks."""
    if not isinstance(scene, SourceCatalogScene):
        raise TypeError("scene must be a SourceCatalogScene")
    if not bool((scene.morphology_codes == TEMPLATE_MORPHOLOGY).all()):
        raise ValueError(
            "morphology-spectrum deblending requires supplied template components"
        )
    cursor = 0
    for start, stop in scene.template_slices.tolist():
        if start != cursor or stop <= start:
            raise ValueError(
                "morphology-spectrum deblending requires contiguous non-overlapping template slices"
            )
        cursor = stop
    if cursor != scene.template_data.numel():
        raise ValueError(
            "morphology-spectrum deblending requires template slices to cover template_data"
        )
    _require_deblend_controls(
        max_iterations=max_iterations,
        max_group_size=max_group_size,
        adequacy_sigma=adequacy_sigma,
        failure_policy=failure_policy,
    )
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    if not math.isfinite(backtracking) or not 0 < backtracking < 1:
        raise ValueError("backtracking must be finite and in (0, 1)")
    monotonic_mask = _component_mask(monotonic, scene, name="monotonic")
    symmetric_mask = _component_mask(symmetric, scene, name="symmetric")
    holdout = _validate_holdout_masks(observations, heldout_masks)
    training = _training_observations(observations, holdout)
    initial_scene = scene
    initial_data = scene.template_data.clone()
    current_data = _project_template_data(
        scene,
        initial_data,
        monotonic=monotonic_mask,
        symmetric=symmetric_mask,
    )
    current_scene = _copy_scene(scene, template_data=current_data)
    current_fit = _fit_linear(
        training,
        current_scene,
        solver="nnls",
        background=background,
        index_nside=index_nside,
        footprint_order=footprint_order,
        chunk_size=chunk_size,
        rank_rtol=rank_rtol,
        rank_atol=rank_atol,
    )
    grouping = current_fit.grouping
    if any(group.numel() > max_group_size for group in grouping.blend_groups):
        raise ValueError("a connected blend group exceeds max_group_size")
    fixed_bounds = _group_bounds(
        scene,
        training,
        grouping,
        padding_arcsec=0.0,
        chunk_size=chunk_size,
    )
    diagnostics: list[FactorizationBlendDiagnostics] = []
    for group_index, sources in enumerate(grouping.blend_groups):
        selected_sources = set(int(value) for value in sources.tolist())
        residual = _weighted_residual(current_fit, training, fixed_bounds[group_index])
        objective = 0.5 * torch.dot(residual, residual)
        initial_objective = objective
        local_step = float(step_size)
        converged = False
        failure_reason: str | None = None
        accepted_steps = 0
        iterations = 0
        if residual.numel() == 0:
            failure_reason = "no_valid_pixels"
        else:
            for iteration in range(max_iterations):
                iterations = iteration + 1
                differentiable_data = current_data.detach().requires_grad_(True)
                differentiable_scene = _copy_scene(
                    scene, template_data=differentiable_data
                )
                differentiable_fit = _fit_linear(
                    training,
                    differentiable_scene,
                    solver="nnls",
                    background=background,
                    index_nside=index_nside,
                    footprint_order=footprint_order,
                    chunk_size=chunk_size,
                    rank_rtol=rank_rtol,
                    rank_atol=rank_atol,
                )
                differentiable_residual = _weighted_residual(
                    differentiable_fit, training, fixed_bounds[group_index]
                )
                differentiable_objective = 0.5 * torch.dot(
                    differentiable_residual, differentiable_residual
                )
                gradient = torch.autograd.grad(
                    differentiable_objective,
                    differentiable_data,
                    allow_unused=True,
                )[0]
                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    failure_reason = "non_finite_morphology_gradient"
                    break
                active_gradient = torch.zeros_like(gradient)
                for component in selected_sources:
                    start, stop = (
                        int(value)
                        for value in scene.template_slices[component].tolist()
                    )
                    active_gradient[start:stop] = gradient[start:stop]
                gradient_norm = torch.linalg.vector_norm(active_gradient)
                if bool(gradient_norm <= step_tolerance):
                    converged = True
                    break
                accepted = False
                trial_step = local_step
                while trial_step >= torch.finfo(current_data.dtype).eps:
                    trial_data = current_data - trial_step * active_gradient
                    trial_data = _project_template_data(
                        scene,
                        trial_data,
                        monotonic=monotonic_mask,
                        symmetric=symmetric_mask,
                        selected_sources=selected_sources,
                    )
                    trial_scene = _copy_scene(scene, template_data=trial_data)
                    trial_fit = _fit_linear(
                        training,
                        trial_scene,
                        solver="nnls",
                        background=background,
                        index_nside=index_nside,
                        footprint_order=footprint_order,
                        chunk_size=chunk_size,
                        rank_rtol=rank_rtol,
                        rank_atol=rank_atol,
                    )
                    trial_residual = _weighted_residual(
                        trial_fit, training, fixed_bounds[group_index]
                    )
                    trial_objective = 0.5 * torch.dot(trial_residual, trial_residual)
                    if bool(torch.isfinite(trial_objective)) and bool(
                        trial_objective < objective
                    ):
                        previous = objective
                        current_data = trial_data.detach()
                        current_scene = trial_scene
                        current_fit = trial_fit
                        objective = trial_objective
                        local_step = trial_step / backtracking
                        accepted = True
                        accepted_steps += 1
                        relative_objective = (
                            previous - objective
                        ).abs() / previous.abs().clamp_min(1.0)
                        relative_step = trial_step * gradient_norm
                        if bool(relative_objective <= objective_tolerance) or bool(
                            relative_step <= step_tolerance
                        ):
                            converged = True
                        break
                    trial_step *= backtracking
                if converged:
                    break
                if not accepted:
                    failure_reason = "proximal_line_search_failed"
                    break
            if not converged and failure_reason is None:
                failure_reason = "maximum_iterations"
        sums = torch.stack(
            [
                current_data[start:stop].sum()
                for start, stop in scene.template_slices.tolist()
            ]
        )
        zero = torch.nonzero(sums <= 0, as_tuple=False).flatten()
        gauge_parts = [
            item.gauge_parameter_indices
            for item in current_fit.diagnostics
            if any(
                int(source) in selected_sources
                for source in item.source_indices.tolist()
            )
        ]
        gauge = (
            torch.unique(torch.cat(gauge_parts), sorted=True)
            if gauge_parts
            else torch.empty(0, dtype=torch.int64, device=scene.device)
        )
        diagnostics.append(
            FactorizationBlendDiagnostics(
                group_index=group_index,
                source_indices=sources,
                component_ids=scene.component_ids[sources],
                converged=converged,
                iterations=iterations,
                initial_objective=initial_objective,
                final_objective=objective,
                accepted_steps=accepted_steps,
                final_step_size=local_step,
                morphology_sums=sums[sources],
                zero_morphology_indices=zero,
                spectrum_gauge_parameter_indices=gauge,
                failure_reason=failure_reason,
            )
        )
    final_geometry = _copy_scene(scene, template_data=current_data)
    final_fit = _fit_linear(
        training,
        final_geometry,
        solver="nnls",
        background=background,
        index_nside=index_nside,
        footprint_order=footprint_order,
        chunk_size=chunk_size,
        rank_rtol=rank_rtol,
        rank_atol=rank_atol,
    )
    spectra = _coefficient_rows(final_geometry, final_fit.coefficients)
    fitted_scene = _copy_scene(
        final_geometry,
        coefficients=torch.nan_to_num(spectra, nan=0.0),
    )
    free_morphology = sum(
        max(int(stop - start) - 1, 0) for start, stop in scene.template_slices.tolist()
    )
    adequacy = _adequacy(
        observations,
        final_fit.predictions,
        holdout,
        fitted_parameter_count=(
            final_fit.coefficients.numel()
            + free_morphology
            + sum(value.numel() for value in final_fit.background_coefficients)
        ),
        sigma=adequacy_sigma,
    )
    catalog = Catalog(
        lon=fitted_scene.lon,
        lat=fitted_scene.lat,
        columns={
            **final_fit.catalog.columns,
            "component_id": fitted_scene.component_ids,
            "morphology_code": fitted_scene.morphology_codes,
            "morphology_sum": torch.ones(
                scene.component_count, dtype=scene.dtype, device=scene.device
            ),
        },
    )
    provenance: dict[str, object] = {
        **final_fit.provenance,
        "method": "known-component-nonnegative-morphology-spectrum-block-proximal",
        "component_count_preserved": True,
        "component_ids_preserved": True,
        "source_detection": False,
        "source_birth_death": False,
        "morphology_gauge": "unit-sum",
        "nonnegative_morphology": True,
        "nonnegative_spectrum": True,
        "monotonic_constraint": bool(monotonic_mask.any()),
        "symmetric_constraint": bool(symmetric_mask.any()),
        "heldout_evaluation": holdout is not None,
        "covariance": "conditional-fixed-morphology-active-spectrum-system",
    }
    result = MorphologySpectrumDeblendResult(
        catalog=catalog,
        scene=fitted_scene,
        initial_scene=initial_scene,
        spectra=spectra,
        initial_spectra=_coefficient_rows(scene),
        morphologies=_template_tuple(scene, current_data),
        initial_morphologies=_template_tuple(scene, initial_data),
        covariance=final_fit.covariance,
        predictions=final_fit.predictions,
        residuals=tuple(
            observation.image - prediction.to(observation.image)
            for observation, prediction in zip(
                observations, final_fit.predictions, strict=True
            )
        ),
        grouping=final_fit.grouping,
        diagnostics=tuple(diagnostics),
        linear_diagnostics=tuple(final_fit.diagnostics),
        adequacy=adequacy,
        background_coefficients=final_fit.background_coefficients,
        provenance=provenance,
    )
    failures = [item.failure_reason for item in diagnostics if item.failure_reason]
    if adequacy.inadequate:
        failures.append(adequacy.reason or "inadequate_model")
    if failures and failure_policy == "raise":
        raise KnownComponentFitError(
            "known-component morphology-spectrum fit failed: " + ", ".join(failures),
            result,
        )
    return result


__all__ = [
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
]
