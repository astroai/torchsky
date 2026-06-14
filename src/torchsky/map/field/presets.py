"""User-facing presets over the continuous-field model zoo."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from torch import nn

from .core import make_field_model
from .types import FieldDomain, FieldModelKind


@dataclass(frozen=True)
class FieldModelPreset:
    """Named field-model recipe for users who do not need class-level details."""

    name: str
    kind: FieldModelKind
    description: str
    kwargs: Mapping[str, Any]

    def make(self, domain: FieldDomain, **overrides: Any) -> nn.Module:
        """Build this preset for ``domain``, with explicit keyword overrides."""
        merged = dict(self.kwargs)
        merged.update(overrides)
        return make_field_model(self.kind, domain, **merged)


_FIELD_MODEL_PRESETS: dict[str, FieldModelPreset] = {
    "fast": FieldModelPreset(
        name="fast",
        kind="sky_encoder_mlp",
        description="small sky-coordinate MLP for smoke tests and quick baselines",
        kwargs={},
    ),
    "balanced": FieldModelPreset(
        name="balanced",
        kind="sky_lambda_feature_field",
        description="default multi-band field model with explicit sky-lambda features",
        kwargs={},
    ),
    "spectral": FieldModelPreset(
        name="spectral",
        kind="hashgrid_spectral_field",
        description="higher-capacity spectral model for richer wavelength structure",
        kwargs={},
    ),
    "operator_constrained": FieldModelPreset(
        name="operator_constrained",
        kind="operator_constrained_feature_field",
        description="advanced preset with hooks for operator-side constraints",
        kwargs={},
    ),
}

FIELD_MODEL_PRESETS = MappingProxyType(_FIELD_MODEL_PRESETS)
FIELD_MODEL_PRESET_NAMES = tuple(FIELD_MODEL_PRESETS)


def get_field_model_preset(name: str) -> FieldModelPreset:
    """Return a named field-model preset."""
    key = str(name).strip().lower().replace("-", "_")
    try:
        return FIELD_MODEL_PRESETS[key]
    except KeyError as exc:
        allowed = ", ".join(FIELD_MODEL_PRESET_NAMES)
        raise ValueError(
            f"unknown field model preset {name!r}; choose one of: {allowed}"
        ) from exc


def make_field_model_preset(
    name: str,
    domain: FieldDomain,
    **overrides: Any,
) -> nn.Module:
    """Build a continuous-field model from a named preset."""
    return get_field_model_preset(name).make(domain, **overrides)


__all__ = [
    "FIELD_MODEL_PRESET_NAMES",
    "FIELD_MODEL_PRESETS",
    "FieldModelPreset",
    "get_field_model_preset",
    "make_field_model_preset",
]
