"""Experimental scene representation and operator facade.

This namespace is additive over the map/field implementation: it gives users
scene-level vocabulary for representations, target specifications, and operator
adapters. Production map verbs live in :mod:`torchsky.map`; field models and
fit/render/sample verbs live in :mod:`torchsky.field`.
"""

from __future__ import annotations

from typing import Any, TypeAlias

from torch import Tensor, nn

from ..map.field import (
    FieldObservation,
    ForwardContext,
    PSFSpec,
    predict_observation_mean,
)
from ..map.observation import ImageObservation
from ..map.operators.resampling import sample_image_plane_batched_channels
from ..map.product import BandpassSpec, ProductSpec
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
from ..map.splat import SplatDiagnostics, splat_to_native
from ..map.target import TargetRenderSpec
from .operators import (
    ObservationOperatorAdapter,
    ObservationOperatorBundle,
    ObservationOperatorBundleProtocol,
    bind_image_plane_observations_to_target,
    operator_adapter_for_observation,
    operator_bundle_for_observation,
    operator_bundles_for_observations,
)

Scene: TypeAlias = SceneRepresentation


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
    product: ProductSpec,
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
    bands: tuple[BandpassSpec, ...],
    value_semantics: str = "surface_brightness",
    footprint_order: int | None = None,
) -> ProductSpec:
    """Construct a map product specification using scene vocabulary."""
    return ProductSpec(
        target_header=target_header,
        bands=bands,
        value_semantics=value_semantics,
        footprint_order=footprint_order,
    )


def render(
    scene: SceneRepresentation | nn.Module | Tensor,
    target: TargetRenderSpec,
    **kwargs: Any,
) -> Tensor:
    """Render any supported scene representation to a target grid."""
    return as_scene_representation(scene).render_to_target(target, **kwargs)


def predict(
    model: nn.Module,
    observation: FieldObservation,
    **kwargs: Any,
) -> Tensor:
    """Predict one field observation using the deterministic operator path."""
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
    "ObservationOperatorAdapter",
    "ObservationOperatorBundle",
    "ObservationOperatorBundleProtocol",
    "PSFSpec",
    "ProductSpec",
    "Scene",
    "SceneRepresentation",
    "SpinHarmonicAlmScene",
    "SplatDiagnostics",
    "TargetRenderSpec",
    "as_scene_representation",
    "bind_image_plane_observations_to_target",
    "field_observation",
    "image_observation",
    "operator_adapter_for_observation",
    "operator_bundle_for_observation",
    "operator_bundles_for_observations",
    "predict",
    "product_spec",
    "render",
    "sample_image_plane_batched_channels",
    "splat_to_native",
    "target_from_header",
    "target_from_product",
    "target_to_field_observation",
]
