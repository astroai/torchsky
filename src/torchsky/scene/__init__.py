"""Promoted scene, observation, and product facade.

This namespace is an additive facade over the current map/field implementation.
It gives users the mission-level vocabulary while the mature implementation
continues to live under :mod:`torchsky.map`.
"""

from __future__ import annotations

from typing import Any, TypeAlias

from torch import Tensor, nn

from ..map import (
    ForwardContext,
    ImageObservation,
    KinematicState,
    MapBandpassSpec,
    MapProductResult,
    MapProductSpec,
    PSFSpec,
    SurveyDataTerm,
    field_observations_from_bin_centers,
    field_observations_from_bin_edges,
    field_observations_from_ifu_cube,
    fit_field_model,
    make_field_model,
    multi_survey_charbonnier_loss,
    observation_charbonnier_loss,
    predict_observation_mean,
    render_field_cube,
    render_field_healpix,
    render_field_observation,
    render_field_observation_distribution,
    render_field_product,
    scale_observation_for_rest_frame_scene,
    stack_map_product,
)
from ..map.field import FieldObservation
from ..map.scenes import (
    ContinuousFieldModel,
    DiscreteTargetImage,
    HarmonicAlmScene,
    HealpixMapScene,
    SceneRepresentation,
    SpinHarmonicAlmScene,
    as_scene_representation,
    target_to_field_observation,
)
from ..map.target import TargetRenderSpec
from .operators import (
    ObservationOperatorAdapter,
    ObservationOperatorBundle,
    ObservationOperatorBundleProtocol,
    operator_adapter_for_observation,
    operator_bundle_for_observation,
)

Observation: TypeAlias = FieldObservation | ImageObservation
Scene: TypeAlias = SceneRepresentation
ProductSpec: TypeAlias = MapProductSpec
BandpassSpec: TypeAlias = MapBandpassSpec
Target: TypeAlias = TargetRenderSpec


def image_observation(*args: Any, **kwargs: Any) -> ImageObservation:
    """Construct an image-plane observation for classical map products."""
    return ImageObservation(*args, **kwargs)


def field_observation(*args: Any, **kwargs: Any) -> FieldObservation:
    """Construct a tensor-backed field observation for differentiable rendering."""
    return FieldObservation(*args, **kwargs)


def target_from_header(
    header: dict[str, Any],
    *,
    value_semantics: str = "surface_brightness",
    **kwargs: Any,
) -> TargetRenderSpec:
    """Build a render target from a FITS-like WCS header."""
    return TargetRenderSpec.from_header(
        header,
        value_semantics=value_semantics,
        **kwargs,
    )


def target_from_product(
    product: MapProductSpec,
    *,
    band_index: int = 0,
    psf_spec: object | None = None,
) -> TargetRenderSpec:
    """Build a render target from one band of a map product specification."""
    return TargetRenderSpec.from_map_product(
        product,
        band_index=band_index,
        psf_spec=psf_spec,
    )


def product_spec(
    *,
    target_header: dict[str, Any],
    bands: tuple[MapBandpassSpec, ...],
    value_semantics: str = "surface_brightness",
    footprint_order: int | None = None,
) -> MapProductSpec:
    """Construct a map product specification using scene vocabulary."""
    return MapProductSpec(
        target_header=target_header,
        bands=bands,
        value_semantics=value_semantics,
        footprint_order=footprint_order,
    )


def render(scene: SceneRepresentation | nn.Module | Tensor, target: TargetRenderSpec, **kwargs: Any) -> Tensor:
    """Render any supported scene representation to a target grid."""
    return as_scene_representation(scene).render_to_target(target, **kwargs)


def predict(
    model: nn.Module,
    observation: FieldObservation,
    **kwargs: Any,
) -> Tensor:
    """Predict one field observation using the current deterministic operator path."""
    return predict_observation_mean(model, observation, **kwargs)


__all__ = [
    "BandpassSpec",
    "ContinuousFieldModel",
    "DiscreteTargetImage",
    "FieldObservation",
    "ForwardContext",
    "HarmonicAlmScene",
    "HealpixMapScene",
    "ImageObservation",
    "KinematicState",
    "MapBandpassSpec",
    "MapProductResult",
    "MapProductSpec",
    "Observation",
    "ObservationOperatorAdapter",
    "ObservationOperatorBundle",
    "ObservationOperatorBundleProtocol",
    "PSFSpec",
    "ProductSpec",
    "Scene",
    "SceneRepresentation",
    "SpinHarmonicAlmScene",
    "SurveyDataTerm",
    "Target",
    "TargetRenderSpec",
    "as_scene_representation",
    "field_observation",
    "field_observations_from_bin_centers",
    "field_observations_from_bin_edges",
    "field_observations_from_ifu_cube",
    "fit_field_model",
    "image_observation",
    "make_field_model",
    "multi_survey_charbonnier_loss",
    "observation_charbonnier_loss",
    "operator_adapter_for_observation",
    "operator_bundle_for_observation",
    "predict",
    "predict_observation_mean",
    "product_spec",
    "render",
    "render_field_cube",
    "render_field_healpix",
    "render_field_observation",
    "render_field_observation_distribution",
    "render_field_product",
    "scale_observation_for_rest_frame_scene",
    "stack_map_product",
    "target_from_header",
    "target_from_product",
    "target_to_field_observation",
]
