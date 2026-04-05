"""Field model kind enumeration (for CLI, benches, and future plugins)."""

from __future__ import annotations

from typing import get_args

from .types import FieldModelKind

FIELD_MODEL_KINDS: tuple[str, ...] = tuple(get_args(FieldModelKind))
