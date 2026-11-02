"""Experimental scene representation and operator facade.

This namespace is additive over the map/field implementation: it gives users
scene-level vocabulary for representations, target specifications, and operator
adapters. Production map verbs live in :mod:`torchsky.map`; field models and
fit/render/sample verbs live in :mod:`torchsky.field`.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from ..map.field.implicit_forward import predict_observation_mean
from ..map.field.types import FieldObservation, ForwardContext, PSFSpec
from ..map.observation import ImageObservation
from ..map.operators.config import MapValueSemantics
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
from ..map.splat import SplatDiagnostics
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
from .sources import (
    GAUSSIAN_MORPHOLOGY,
    POINT_MORPHOLOGY,
    SERSIC_MORPHOLOGY,
    TEMPLATE_MORPHOLOGY,
    SourceCatalogScene,
    SourceStamp,
)

def image_observation(*args: Any, **kwargs: Any) -> ImageObservation:
    """Construct an image-plane observation for classical map products.

    Args:
        *args: Positional ``ImageObservation`` constructor arguments.
        **kwargs: Image data, WCS, calibration, mask, and weight metadata.

    Returns:
        Validated image-observation metadata.

    Examples:
        >>> import torch
        >>> from torchsky.scene import image_observation
        >>> observation = image_observation(data=torch.ones(2, 2), header={})
        >>> tuple(observation.data.shape)
        (2, 2)
    """
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
    value_semantics: MapValueSemantics = "surface_brightness",
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
    """Render any supported scene representation to a target grid.

    Args:
        scene: Scene representation, field module, or target-aligned tensor.
        target: Output grid and observing metadata.
        **kwargs: Scene-specific rendering controls.

    Returns:
        Rendered tensor on the scene's execution device.

    Examples:
        >>> from torchsky.scene import render
        >>> callable(render)
        True
    """
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
    "GAUSSIAN_MORPHOLOGY",
    "HealpixMapScene",
    "ImageObservation",
    "ObservationOperatorAdapter",
    "ObservationOperatorBundle",
    "ObservationOperatorBundleProtocol",
    "POINT_MORPHOLOGY",
    "SERSIC_MORPHOLOGY",
    "PSFSpec",
    "ProductSpec",
    "SceneRepresentation",
    "SourceCatalogScene",
    "SourceStamp",
    "SpinHarmonicAlmScene",
    "SplatDiagnostics",
    "TargetRenderSpec",
    "TEMPLATE_MORPHOLOGY",
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
    "target_from_header",
    "target_from_product",
    "target_to_field_observation",
]
