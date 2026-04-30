"""Field model kind enumeration (for CLI, benches, and future plugins)."""

from __future__ import annotations

from typing import get_args

from .types import FieldModelKind

FIELD_MODEL_KINDS: tuple[str, ...] = tuple(get_args(FieldModelKind))

# Keep the full model zoo available for experiments while declaring the narrow
# user-facing recommendation set in code.
RECOMMENDED_FIELD_MODEL_KINDS: tuple[str, ...] = (
    "sky_lambda_feature_field",
    "sky_encoder_resmlp",
    "hashgrid_spectral_field",
)


def is_recommended_field_model(kind: str) -> bool:
    """Return whether a field model kind is in the recommended subset."""
    return kind in RECOMMENDED_FIELD_MODEL_KINDS


__all__ = [
    "FIELD_MODEL_KINDS",
    "RECOMMENDED_FIELD_MODEL_KINDS",
    "is_recommended_field_model",
]
