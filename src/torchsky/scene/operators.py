"""Scene-level observation-operator adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from torch import Tensor, nn

from ..map.field import FieldObservation
from ..map.field.implicit_forward import predict_observation_mean as _predict_mean
from ..map.image_plane import ImageObservation
from ..map.operators.bundle import (
    ObservationOperatorBundle,
    field_observation_operator_bundle,
    image_observation_operator_bundle,
)
from ..map.operators.inverse import (
    bind_image_plane_observations_to_target_operator_set,
    bind_image_plane_target_operator,
)
from ..map.solvers import (
    AdjointOperatorFn,
    ApplyNoiseFn,
    ApplyOperatorFn,
    BindTargetFn,
    BoundObservationOperator,
    BoundObservationOperatorSet,
    NormalEquationProbe,
    SolverHooks,
    build_solver_hooks,
)
from ..map.target import TargetRenderSpec


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
    observation: FieldObservation | ImageObservation | None = None
    resampling_kernel: str | None = None

    @classmethod
    def from_field_observation(
        cls,
        observation: FieldObservation,
        *,
        geometry: object | None = None,
    ) -> "ObservationOperatorAdapter":
        return cls(
            field_observation_operator_bundle(observation, geometry=geometry),
            observation=observation,
        )

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
            ),
            observation=observation,
            resampling_kernel=resampling_kernel,
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

    def solver_hooks(
        self,
        *,
        apply_operator: ApplyOperatorFn | None = None,
        apply_adjoint: AdjointOperatorFn | None = None,
        apply_noise: ApplyNoiseFn | None = None,
        bind_target_fn: BindTargetFn | None = None,
    ) -> SolverHooks:
        """Build executable solver hooks backed by this observation bundle."""
        return build_solver_hooks(
            self.bundle,
            apply_operator=apply_operator,
            apply_adjoint=apply_adjoint,
            apply_noise=apply_noise,
            bind_target_fn=bind_target_fn,
        )

    def bind_target(
        self,
        target: TargetRenderSpec,
        *,
        apply_operator: ApplyOperatorFn | None = None,
        apply_adjoint: AdjointOperatorFn | None = None,
        apply_noise: ApplyNoiseFn | None = None,
        bind_target_fn: BindTargetFn | None = None,
    ) -> BoundObservationOperator:
        """Bind this observation law to one target render grid."""
        return self.solver_hooks(
            apply_operator=apply_operator,
            apply_adjoint=apply_adjoint,
            apply_noise=apply_noise,
            bind_target_fn=bind_target_fn,
        ).bind_target(target)

    def bind_targets(
        self,
        targets: Sequence[TargetRenderSpec],
        *,
        apply_operator: ApplyOperatorFn | None = None,
        apply_adjoint: AdjointOperatorFn | None = None,
        apply_noise: ApplyNoiseFn | None = None,
        bind_target_fn: BindTargetFn | None = None,
        exposure_chunk_size: int = 1,
    ) -> BoundObservationOperatorSet:
        """Bind this observation law to several target render grids."""
        return self.solver_hooks(
            apply_operator=apply_operator,
            apply_adjoint=apply_adjoint,
            apply_noise=apply_noise,
            bind_target_fn=bind_target_fn,
        ).bind_targets(targets, exposure_chunk_size=exposure_chunk_size)

    def normal_equation_probe(
        self,
        scene: Any,
        target: TargetRenderSpec,
        data: Tensor,
        *,
        apply_operator: ApplyOperatorFn | None = None,
        apply_adjoint: AdjointOperatorFn | None = None,
        apply_noise: ApplyNoiseFn | None = None,
        bind_target_fn: BindTargetFn | None = None,
    ) -> NormalEquationProbe:
        """Assemble ``A^T N^-1 A`` and ``A^T N^-1 y`` probes for one target."""
        return self.bind_target(
            target,
            apply_operator=apply_operator,
            apply_adjoint=apply_adjoint,
            apply_noise=apply_noise,
            bind_target_fn=bind_target_fn,
        ).normal_equation_probe(scene, data)

    def bind_image_plane_target(
        self,
        scene_target: TargetRenderSpec,
        *,
        observation_header: dict[str, Any] | None = None,
        resampling_kernel: str | None = None,
        value_semantics: str | None = None,
        drizzle_pixfrac: float = 1.0,
        chunk_pixels: int = 262_144,
    ) -> BoundObservationOperator:
        """Bind an image-plane observation to a latent target-grid scene.

        The returned operator maps a scene tensor on ``scene_target`` into the
        observation image grid. Its adjoint is the exact vector-Jacobian product
        of that same differentiable projection, so normal-equation probes and CG
        solves exercise one shared forward model.
        """
        if not isinstance(self.observation, ImageObservation):
            raise TypeError("bind_image_plane_target requires an ImageObservation")
        return bind_image_plane_target_operator(
            self.observation,
            scene_target,
            operator_bundle=self.bundle,
            observation_header=observation_header,
            resampling_kernel=str(
                resampling_kernel or self.resampling_kernel or "bilinear"
            ),
            value_semantics=value_semantics,
            drizzle_pixfrac=drizzle_pixfrac,
            chunk_pixels=chunk_pixels,
        )


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


def operator_bundles_for_observations(
    observations: Sequence[FieldObservation | ImageObservation],
    *,
    resampling_kernel: str = "bilinear",
    geometry: object | None = None,
) -> tuple[ObservationOperatorBundle, ...]:
    """Return one observation-operator bundle per observation, preserving order."""
    return tuple(
        operator_bundle_for_observation(
            observation,
            resampling_kernel=resampling_kernel,
            geometry=geometry,
        )
        for observation in observations
    )


def bind_image_plane_observations_to_target(
    observations: Sequence[ImageObservation],
    scene_target: TargetRenderSpec,
    *,
    observation_headers: Sequence[dict[str, Any] | None] | None = None,
    resampling_kernel: str = "bilinear",
    value_semantics: str | None = None,
    drizzle_pixfrac: float = 1.0,
    chunk_pixels: int = 262_144,
    exposure_chunk_size: int = 1,
) -> BoundObservationOperatorSet:
    """Bind several image-plane observations to one latent target-grid scene."""
    if len(observations) == 0:
        raise ValueError("at least one ImageObservation is required")
    return bind_image_plane_observations_to_target_operator_set(
        observations,
        scene_target,
        observation_headers=observation_headers,
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
        drizzle_pixfrac=drizzle_pixfrac,
        chunk_pixels=chunk_pixels,
        exposure_chunk_size=exposure_chunk_size,
    )


__all__ = [
    "ObservationOperatorAdapter",
    "ObservationOperatorBundle",
    "ObservationOperatorBundleProtocol",
    "bind_image_plane_observations_to_target",
    "operator_adapter_for_observation",
    "operator_bundle_for_observation",
    "operator_bundles_for_observations",
]
