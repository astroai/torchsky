"""Composable continuous-field modeling API.

This namespace keeps field workflows explicit:

``model(...)`` builds a field module, ``fit(...)`` optimizes it, and
``render(...)`` / ``sample(...)`` evaluate it on observations or coordinates.
Lower-level model classes remain available for users who want direct
composition instead of a string-selected factory.
"""

from __future__ import annotations

from typing import Any, Literal, overload

from torch import Tensor, nn

from ..map.field.core import (
    BandpassConditionedField as BandpassConditioned,
    ConditionedSpectralField as ConditionedSpectral,
    FieldComponentDecomposition,
    FourierMLPField as FourierMLP,
    GaussianBasisField as GaussianBasis,
    HashGridConditionedField as HashGridConditioned,
    HashGridMLPField as HashGridMLP,
    HashGridResMLPField as HashGridResMLP,
    HashGridSpectralField as HashGridSpectral,
    LearnedSpectralBasisField as LearnedSpectralBasis,
    OperatorConstrainedFeatureField as OperatorConstrainedFeatures,
    SineImplicitField as Sine,
    SkyEncoderBandpassField as SkyEncoderBandpass,
    SkyEncoderMLPField as SkyEncoderMLP,
    SkyEncoderResMLPField as SkyEncoderResMLP,
    SkyLambdaAtlasField as SkyLambdaAtlas,
    SkyLambdaFeatureAtlasField as SkyLambdaFeatureAtlas,
    SkyLambdaFeatureField as SkyLambdaFeatures,
    SkyLambdaKPlanesField as SkyLambdaKPlanes,
    SkyLambdaPlanesField as SkyLambdaPlanes,
    SpectralBasisField as SpectralBasis,
    StrainerSkyEncoderMLPField as StrainerSkyEncoderMLP,
    WireImplicitField as Wire,
    estimate_field_domain as estimate_domain,
    field_component_anchor_loss as component_anchor_loss,
    field_component_centroid_anchor_loss as centroid_anchor_loss,
    make_field_component_decomposition as component_decomposition,
    make_field_model as model,
)
from ..map.field.adaptive import (
    ResidualTileBasisField as ResidualTileBasis,
    ResidualTileBasisSpec,
    make_residual_tile_basis_field as residual_tile_basis,
    residual_tile_basis_spec_from_observation as residual_tile_basis_spec,
)
from ..map.field.rendering import (
    evaluate_field_coordinates_vmapped as evaluate,
    render_field_cube,
    render_field_cube_via_healpix,
    render_field_healpix,
    render_field_observation,
    render_field_observation_distribution,
    render_field_samples,
    render_field_samples_distribution,
)
from ..map.field.quality import (
    FieldResidualProduct as ResidualProduct,
    FieldResidualTile as ResidualTile,
    field_image_residual_product as residual_product,
    field_image_quality_metrics as metrics,
    field_quality_scorecard as scorecard,
    render_field_quality as quality,
    render_field_residual as residual,
    render_field_residual_product as render_residual_product,
    select_field_residual_tiles as residual_tiles,
)
from ..map.field.training import (
    fit_field_component_decomposition as fit_component_decomposition,
    fit_field_model as fit,
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
    FIELD_MODEL_PRESETS,
    FieldModelPreset,
    get_field_model_preset,
    make_field_model_preset,
)
from ..map.field.registry import (
    FIELD_MODEL_KINDS,
    RECOMMENDED_FIELD_MODEL_KINDS,
    is_recommended_field_model,
)
from ..map.field.spectral_budget import (
    spectral_data_budget,
    suggest_num_spectral_samples,
    suggest_spectral_model_capacity,
)
from ..map.field.types import (
    FieldCentroidAnchorTerm,
    FieldComponentAnchorTerm,
    FieldCoordinateMode,
    FieldCurriculum,
    FieldDomain as Domain,
    FieldFitResult as FitResult,
    FieldLossKind,
    FieldLossReduction,
    FieldLRSchedule,
    FieldModelKind,
    FieldNoisePretrainKind,
    FieldObservation as Observation,
    FieldObservationSampling,
    FieldSpectralQuantity,
    FieldSpectralSampling,
    ForwardContext,
    ObservationSpectralSummary,
    PSFSpec,
    SpectralCapacityProfile,
    SpectralDataBudget,
    SpectralModelCapacityHints,
)

PresetName = Literal["fast", "balanced", "spectral", "operator_constrained"]
RenderTarget = Literal["observation", "samples", "cube", "healpix", "cube_via_healpix"]


def preset(name: str, domain: Domain, **overrides: Any) -> nn.Module:
    """Build a recommended field model by preset name."""
    return make_field_model_preset(name, domain, **overrides)


@overload
def render(
    field_model: nn.Module,
    observation: Observation,
    *,
    target: Literal["observation"] = "observation",
    **kwargs: Any,
) -> Tensor: ...


@overload
def render(
    field_model: nn.Module,
    observation: Observation,
    *,
    target: Literal["samples"],
    ra_deg: Tensor,
    dec_deg: Tensor,
    **kwargs: Any,
) -> Tensor: ...


@overload
def render(
    field_model: nn.Module,
    observation: Observation | None = None,
    *,
    target: Literal["cube", "healpix", "cube_via_healpix"],
    **kwargs: Any,
) -> Tensor: ...


def render(
    field_model: nn.Module,
    observation: Observation | None = None,
    *,
    target: RenderTarget = "observation",
    **kwargs: Any,
) -> Tensor:
    """Render a field to an observation, coordinate samples, cube, or HEALPix map."""
    if target == "observation":
        if observation is None:
            raise ValueError(
                "field.render(..., target='observation') requires observation"
            )
        return render_field_observation(field_model, observation, **kwargs)
    if target == "samples":
        if observation is None:
            raise ValueError("field.render(..., target='samples') requires observation")
        return render_field_samples(field_model, observation, **kwargs)
    if target == "cube":
        return render_field_cube(field_model, **kwargs)
    if target == "healpix":
        return render_field_healpix(field_model, **kwargs)
    if target == "cube_via_healpix":
        return render_field_cube_via_healpix(field_model, **kwargs)
    raise ValueError(f"unsupported field render target {target!r}")


def sample(
    field_model: nn.Module,
    observation: Observation,
    *,
    ra_deg: Tensor,
    dec_deg: Tensor,
    **kwargs: Any,
) -> Tensor:
    """Evaluate a field on explicit sky-coordinate samples for one observation."""
    return render_field_samples(
        field_model,
        observation,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        **kwargs,
    )


def distribution(
    field_model: nn.Module,
    observation: Observation,
    *,
    target: Literal["observation", "samples"] = "observation",
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Render predictive mean and log-variance."""
    if target == "observation":
        return render_field_observation_distribution(field_model, observation, **kwargs)
    if target == "samples":
        return render_field_samples_distribution(field_model, observation, **kwargs)
    raise ValueError(f"unsupported field distribution target {target!r}")


__all__ = [
    "FIELD_MODEL_KINDS",
    "FIELD_MODEL_PRESET_NAMES",
    "FIELD_MODEL_PRESETS",
    "RECOMMENDED_FIELD_MODEL_KINDS",
    "BandpassConditioned",
    "ConditionedSpectral",
    "Domain",
    "FieldCentroidAnchorTerm",
    "FieldComponentAnchorTerm",
    "FieldComponentDecomposition",
    "FieldCoordinateMode",
    "FieldCurriculum",
    "FieldLossKind",
    "FieldLossReduction",
    "FieldLRSchedule",
    "FieldModelKind",
    "FieldModelPreset",
    "FieldNoisePretrainKind",
    "FieldObservationSampling",
    "FieldSpectralQuantity",
    "FieldSpectralSampling",
    "FitResult",
    "FourierMLP",
    "ForwardContext",
    "GaussianBasis",
    "HashGridConditioned",
    "HashGridMLP",
    "HashGridResMLP",
    "HashGridSpectral",
    "LearnedSpectralBasis",
    "Observation",
    "ObservationSpectralSummary",
    "OperatorConstrainedFeatures",
    "PSFSpec",
    "PresetName",
    "RenderTarget",
    "ResidualProduct",
    "ResidualTile",
    "ResidualTileBasis",
    "ResidualTileBasisSpec",
    "Sine",
    "SkyEncoderBandpass",
    "SkyEncoderMLP",
    "SkyEncoderResMLP",
    "SkyLambdaAtlas",
    "SkyLambdaFeatureAtlas",
    "SkyLambdaFeatures",
    "SkyLambdaKPlanes",
    "SkyLambdaPlanes",
    "SpectralBasis",
    "SpectralCapacityProfile",
    "SpectralDataBudget",
    "SpectralModelCapacityHints",
    "StrainerSkyEncoderMLP",
    "Wire",
    "centroid_anchor_loss",
    "component_anchor_loss",
    "component_decomposition",
    "distribution",
    "estimate_domain",
    "evaluate",
    "field_observations_from_bin_centers",
    "field_observations_from_bin_edges",
    "field_observations_from_ifu_cube",
    "fit",
    "fit_component_decomposition",
    "get_field_model_preset",
    "ifu_cube_spectral_to_leading",
    "is_recommended_field_model",
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
    "scorecard",
    "spectral_data_budget",
    "suggest_num_spectral_samples",
    "suggest_spectral_model_capacity",
]
