"""Scene-level observation-operator adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from torch import Tensor, nn

from ..map import ImageObservation
from ..map.field import FieldObservation
from ..map.field.implicit_forward import predict_observation_mean as _predict_mean
from ..map.operators.bundle import (
    ObservationOperatorBundle,
    field_observation_operator_bundle,
    image_observation_operator_bundle,
)


@runtime_checkable
class ObservationOperatorBundleProtocol(Protocol):
    """Structural protocol for bundled observation-law components."""

    geometry: object | None
    spectral_response: object | None
    psf: object | None
    radiometry: object | None
    background: object | None
    mask_weight: object | None
    noise: object | None

    def iter_enabled(self): ...


@dataclass(frozen=True)
class ObservationOperatorAdapter:
    """Executable adapter over the existing map observation-operator bundle."""

    bundle: ObservationOperatorBundle

    @classmethod
    def from_field_observation(
        cls,
        observation: FieldObservation,
        *,
        geometry: object | None = None,
    ) -> "ObservationOperatorAdapter":
        return cls(field_observation_operator_bundle(observation, geometry=geometry))

    @classmethod
    def from_image_observation(
        cls,
        observation: ImageObservation,
        *,
        resampling_kernel: str,
    ) -> "ObservationOperatorAdapter":
        return cls(
            image_observation_operator_bundle(
                observation,
                resampling_kernel=resampling_kernel,
            )
        )

    @classmethod
    def from_observation(
        cls,
        observation: FieldObservation | ImageObservation,
        *,
        resampling_kernel: str = "bilinear",
        geometry: object | None = None,
    ) -> "ObservationOperatorAdapter":
        if isinstance(observation, FieldObservation):
            return cls.from_field_observation(observation, geometry=geometry)
        if isinstance(observation, ImageObservation):
            return cls.from_image_observation(
                observation,
                resampling_kernel=resampling_kernel,
            )
        raise TypeError(
            "observation must be a scene.FieldObservation or scene.ImageObservation"
        )

    def iter_enabled(self):
        return self.bundle.iter_enabled()

    def enabled_names(self) -> tuple[str, ...]:
        return tuple(op.name for op in self.iter_enabled())

    def has(self, slot: str) -> bool:
        op = getattr(self.bundle, slot)
        return op is not None and bool(getattr(op, "enabled", False))

    def predict_observation_mean(
        self,
        model: nn.Module,
        observation: FieldObservation,
        **render_kw: Any,
    ) -> Tensor:
        """Execute the current deterministic field-observation prediction path."""
        return _predict_mean(model, observation, **render_kw)


def operator_adapter_for_observation(
    observation: FieldObservation | ImageObservation,
    *,
    resampling_kernel: str = "bilinear",
    geometry: object | None = None,
) -> ObservationOperatorAdapter:
    """Return the promoted scene adapter for an existing observation object."""
    return ObservationOperatorAdapter.from_observation(
        observation,
        resampling_kernel=resampling_kernel,
        geometry=geometry,
    )


def operator_bundle_for_observation(
    observation: FieldObservation | ImageObservation,
    *,
    resampling_kernel: str = "bilinear",
    geometry: object | None = None,
) -> ObservationOperatorBundle:
    """Return the underlying current map operator bundle for one observation."""
    return operator_adapter_for_observation(
        observation,
        resampling_kernel=resampling_kernel,
        geometry=geometry,
    ).bundle


__all__ = [
    "ObservationOperatorAdapter",
    "ObservationOperatorBundle",
    "ObservationOperatorBundleProtocol",
    "operator_adapter_for_observation",
    "operator_bundle_for_observation",
]
