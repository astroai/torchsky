from __future__ import annotations

from typing import Any as _Any, Literal as _Literal, overload as _overload

from torch import Tensor as _Tensor, nn as _nn

from ..map.field.adaptive import (
    ResidualTileBasisSpec,
    make_residual_tile_basis_field as residual_tile_basis,
    residual_tile_basis_spec_from_observation as residual_tile_basis_spec,
)
from ..map.field.artifacts import (
    FieldArtifactManifest as ArtifactManifest,
    LoadedFieldArtifact as LoadedArtifact,
    load_field_artifact as load_artifact,
    save_field_artifact as save_artifact,
)
from ..map.field.core import (
    FieldComponentDecomposition,
    estimate_field_domain as estimate_domain,
    make_field_component_decomposition as component_decomposition,
    make_field_model as model,
)
from ..map.field.observation_builders import (
    field_observations_from_bin_centers,
    field_observations_from_bin_edges,
    field_observations_from_ifu_cube,
    ifu_cube_spectral_to_leading,
    make_tangent_plane_header,
)
from ..map.field.presets import (
    FIELD_MODEL_PRESET_NAMES,
    FieldModelPreset,
    get_field_model_preset,
    make_field_model_preset as _make_field_model_preset,
)
from ..map.field.quality import (
    FieldResidualProduct as ResidualProduct,
    FieldResidualTile as ResidualTile,
    field_image_quality_metrics as metrics,
    field_image_residual_product as residual_product,
    render_field_quality as quality,
    render_field_residual as residual,
    render_field_residual_product as render_residual_product,
    select_field_residual_tiles as residual_tiles,
)
from ..map.field.registry import (
    FIELD_MODEL_KINDS,
    RECOMMENDED_FIELD_MODEL_KINDS,
    is_recommended_field_model,
)
from ..map.field.rendering import (
    render_field_cube as _render_field_cube,
    render_field_cube_via_healpix as _render_field_cube_via_healpix,
    render_field_healpix as _render_field_healpix,
    render_field_observation as _render_field_observation,
    render_field_observation_distribution as _render_field_observation_distribution,
    render_field_samples as _render_field_samples,
    render_field_samples_distribution as _render_field_samples_distribution,
)
from ..map.field.spectral_budget import (
    spectral_data_budget,
    suggest_num_spectral_samples,
    suggest_spectral_model_capacity,
)
from ..map.field.training import (
    fit_diffuse_foam_hybrid as fit_hybrid,
    fit_field_component_decomposition as fit_component_decomposition,
    fit_field_model as fit,
    fit_structured_noise_crossfit as fit_noise_crossfit,
)
from ..map.field.types import (
    FieldDomain as Domain,
    FieldFitResult as FitResult,
    FieldModelKind,
    FieldObservation as Observation,
    ForwardContext,
    PSFSpec,
    SpectralCapacityProfile,
    SpectralDataBudget,
    SpectralModelCapacityHints,
)

PresetName = _Literal["fast", "balanced", "spectral", "operator_constrained"]
RenderTarget = _Literal["observation", "samples", "cube", "healpix", "cube_via_healpix"]


def preset(name: str, domain: Domain, **overrides: _Any) -> _nn.Module:
    """Build a recommended field model by preset name."""
    return _make_field_model_preset(name, domain, **overrides)


@_overload
def render(
    field_model: _nn.Module,
    observation: Observation,
    *,
    target: _Literal["observation"] = "observation",
    **kwargs: _Any,
) -> _Tensor: ...


@_overload
def render(
    field_model: _nn.Module,
    observation: Observation,
    *,
    target: _Literal["samples"],
    ra_deg: _Tensor,
    dec_deg: _Tensor,
    **kwargs: _Any,
) -> _Tensor: ...


@_overload
def render(
    field_model: _nn.Module,
    observation: Observation | None = None,
    *,
    target: _Literal["cube", "healpix", "cube_via_healpix"],
    **kwargs: _Any,
) -> _Tensor: ...


def render(
    field_model: _nn.Module,
    observation: Observation | None = None,
    *,
    target: RenderTarget = "observation",
    **kwargs: _Any,
) -> _Tensor:
    """Render a field to an observation, coordinate samples, cube, or HEALPix map.

    Args:
        field_model: Trained or initialized field module.
        observation: Observation metadata required by observation and sample targets.
        target: Output geometry selector.
        **kwargs: Target-specific coordinates, shape, WCS, or HEALPix options.

    Returns:
        Rendered tensor on the model's execution device.

    Examples:
        >>> from torchsky.field import render
        >>> callable(render)
        True
    """
    if target == "observation":
        if observation is None:
            raise ValueError(
                "field.render(..., target='observation') requires observation"
            )
        return _render_field_observation(field_model, observation, **kwargs)
    if target == "samples":
        if observation is None:
            raise ValueError("field.render(..., target='samples') requires observation")
        return _render_field_samples(field_model, observation, **kwargs)
    if target == "cube":
        return _render_field_cube(field_model, **kwargs)
    if target == "healpix":
        return _render_field_healpix(field_model, **kwargs)
    if target == "cube_via_healpix":
        return _render_field_cube_via_healpix(field_model, **kwargs)
    raise ValueError(f"unsupported field render target {target!r}")


def sample(
    field_model: _nn.Module,
    observation: Observation,
    *,
    ra_deg: _Tensor,
    dec_deg: _Tensor,
    **kwargs: _Any,
) -> _Tensor:
    """Evaluate a field on explicit sky-coordinate samples for one observation."""
    return _render_field_samples(
        field_model,
        observation,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        **kwargs,
    )


def distribution(
    field_model: _nn.Module,
    observation: Observation,
    *,
    target: _Literal["observation", "samples"] = "observation",
    **kwargs: _Any,
) -> tuple[_Tensor, _Tensor]:
    """Render predictive mean and log-variance."""
    if target == "observation":
        return _render_field_observation_distribution(field_model, observation, **kwargs)
    if target == "samples":
        return _render_field_samples_distribution(field_model, observation, **kwargs)
    raise ValueError(f"unsupported field distribution target {target!r}")


__all__ = [
    "FIELD_MODEL_KINDS",
    "FIELD_MODEL_PRESET_NAMES",
    "RECOMMENDED_FIELD_MODEL_KINDS",
    "ArtifactManifest",
    "Domain",
    "FieldComponentDecomposition",
    "FieldModelKind",
    "FieldModelPreset",
    "FitResult",
    "ForwardContext",
    "LoadedArtifact",
    "Observation",
    "PSFSpec",
    "PresetName",
    "RenderTarget",
    "ResidualProduct",
    "ResidualTile",
    "ResidualTileBasisSpec",
    "SpectralCapacityProfile",
    "SpectralDataBudget",
    "SpectralModelCapacityHints",
    "component_decomposition",
    "distribution",
    "estimate_domain",
    "field_observations_from_bin_centers",
    "field_observations_from_bin_edges",
    "field_observations_from_ifu_cube",
    "fit",
    "fit_component_decomposition",
    "fit_hybrid",
    "fit_noise_crossfit",
    "get_field_model_preset",
    "ifu_cube_spectral_to_leading",
    "is_recommended_field_model",
    "load_artifact",
    "make_tangent_plane_header",
    "metrics",
    "model",
    "preset",
    "quality",
    "render",
    "render_residual_product",
    "residual",
    "residual_product",
    "residual_tile_basis",
    "residual_tile_basis_spec",
    "residual_tiles",
    "sample",
    "save_artifact",
    "spectral_data_budget",
    "suggest_num_spectral_samples",
    "suggest_spectral_model_capacity",
]
