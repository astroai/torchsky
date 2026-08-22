"""Optional live GWCS freeze wrapper. Core sampling never imports gwcs."""

from __future__ import annotations

import importlib.util
from typing import Any, Callable

from .gwcs_sampling import (
    GWCSGridSamplingResult,
    sample_gwcs_detector_to_tangent,
)


def freeze_live_gwcs(
    model: Any,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    **kwargs: Any,
) -> GWCSGridSamplingResult:
    """Evaluate a live GWCS graph, then freeze detector→tangent tables.

    Requires the ``gwcs`` extra. The production runtime path remains
    ``sample_gwcs_detector_to_tangent`` with a callable and no ``import gwcs``.
    """
    if importlib.util.find_spec("gwcs") is None:
        raise ImportError("freeze_live_gwcs requires gwcs; install torchsky[gwcs]")

    def _reference(x: Any, y: Any) -> Any:
        if hasattr(model, "pixel_to_world_values"):
            return model.pixel_to_world_values(x, y)
        return model(x, y)

    reference: Callable[..., Any] = _reference
    return sample_gwcs_detector_to_tangent(reference, bounds, **kwargs)
