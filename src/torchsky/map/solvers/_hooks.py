"""Internal solver hooks for field fitting and discrete inverse mapmaking."""

from __future__ import annotations

from ._core import (
    BoundObservationOperator,
    BoundObservationOperatorSet,
    NormalEquationProbe,
    identity_noise,
)
from ._types import (
    AdjointOperatorFn,
    ApplyNoiseFn,
    ApplyOperatorFn,
    BindTargetFn,
)

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from ..healpix_image_plane import accumulate_image_on_healpix_map_adjoint
from ..healpix_spectral import (
    accumulate_image_on_alm_adjoint,
    accumulate_image_on_spin_alm_adjoint,
)
from ..operators.bundle import ObservationOperatorBundle
from ..operators.bundle import spherical_spectral_operator_bundle
from ..scenes import (
    DiscreteTargetImage,
    HarmonicAlmScene,
    HealpixMapScene,
    SceneRepresentation,
    SpinHarmonicAlmScene,
    as_scene_representation,
)
from ..target import TargetRenderSpec


@dataclass
class SolverHooks:
    """Internal assembly surface shared by GLS and MAP solver experiments."""

    operators: ObservationOperatorBundle
    apply_operator: ApplyOperatorFn | None = None
    apply_adjoint: AdjointOperatorFn | None = None
    apply_noise: ApplyNoiseFn | None = None
    bind_target_fn: BindTargetFn | None = None

    def render_scene(
        self, scene: SceneRepresentation, target: TargetRenderSpec
    ) -> Tensor:
        return self.bind_target(target).apply(scene)

    def apply_noise_model(self, tensor: Tensor) -> Tensor:
        fn = self.apply_noise or identity_noise
        return fn(tensor)

    def apply_adjoint_model(self, tensor: Tensor) -> Tensor:
        if self.apply_adjoint is None:
            raise NotImplementedError("adjoint operator hook is not configured")
        return self.apply_adjoint(tensor)

    def normal_equation_probe(
        self,
        scene: SceneRepresentation,
        target: TargetRenderSpec,
        data: Tensor,
    ) -> NormalEquationProbe:
        """Assemble A^T N^-1 A and A^T N^-1 y probes without solving."""
        return self.bind_target(target).normal_equation_probe(scene, data)

    def bind_target(self, target: TargetRenderSpec) -> BoundObservationOperator:
        apply_operator = self.apply_operator
        apply_adjoint = self.apply_adjoint
        apply_noise = self.apply_noise
        if self.bind_target_fn is not None:
            apply_operator, apply_adjoint, apply_noise = self.bind_target_fn(
                self.operators,
                target,
            )
        return BoundObservationOperator(
            operators=self.operators,
            target=target,
            apply_operator=apply_operator,
            apply_adjoint=apply_adjoint,
            apply_noise=apply_noise,
        )

    def bind_targets(
        self,
        targets: Sequence[TargetRenderSpec],
        *,
        exposure_chunk_size: int = 1,
    ) -> BoundObservationOperatorSet:
        return BoundObservationOperatorSet(
            tuple(self.bind_target(target) for target in targets),
            exposure_chunk_size=exposure_chunk_size,
        )


def build_solver_hooks(
    operators: ObservationOperatorBundle,
    *,
    apply_operator: ApplyOperatorFn | None = None,
    apply_adjoint: AdjointOperatorFn | None = None,
    apply_noise: ApplyNoiseFn | None = None,
    bind_target_fn: BindTargetFn | None = None,
) -> SolverHooks:
    return SolverHooks(
        operators=operators,
        apply_operator=apply_operator,
        apply_adjoint=apply_adjoint,
        apply_noise=apply_noise,
        bind_target_fn=bind_target_fn,
    )


def build_spherical_healpix_solver_hooks(
    *,
    nside: int,
    nest: bool = False,
    mode: str = "pixel_bilinear",
    lmax: int | None = None,
    target_nside: int | None = None,
    fwhm_rad: float | None = None,
    beam_window: Tensor | None = None,
    iter: int = 0,
    apply_noise: ApplyNoiseFn | None = None,
) -> SolverHooks:
    """Build native sphere↔image solver hooks for HEALPix coefficient maps.

    `pixel_bilinear` is the first exact inverse-capable mode because its forward
    operator is the native HEALPix bilinear bridge and its adjoint is the
    corresponding scatter-add transpose.
    """

    if mode not in {"pixel_bilinear", "harmonic_direct", "healpix_bridge"}:
        raise ValueError(
            "mode must be 'pixel_bilinear', 'harmonic_direct', or 'healpix_bridge'"
        )
    operators = spherical_spectral_operator_bundle(mode=mode)

    def _bind_target(
        bound_operators: ObservationOperatorBundle,
        target: TargetRenderSpec,
    ) -> tuple[ApplyOperatorFn | None, AdjointOperatorFn | None, ApplyNoiseFn | None]:
        def _coerce_healpix_scene(
            scene: SceneRepresentation | nn.Module | Tensor,
        ) -> HealpixMapScene:
            scene_representation = as_scene_representation(scene)
            if isinstance(scene_representation, HealpixMapScene):
                return scene_representation
            if isinstance(scene_representation, DiscreteTargetImage):
                return HealpixMapScene(
                    scene_representation.data,
                    nside=int(nside),
                    nest=bool(nest),
                    lmax=lmax,
                    target_nside=target_nside,
                    fwhm_rad=fwhm_rad,
                    beam_window=beam_window,
                    iter=int(iter),
                    mode=mode,
                )
            raise TypeError(
                "spherical HEALPix solver hooks require a HealpixMapScene or tensor coefficients"
            )

        def _apply_operator(
            scene: SceneRepresentation, target_spec: TargetRenderSpec
        ) -> Tensor:
            healpix_scene = _coerce_healpix_scene(scene)
            return healpix_scene.render_to_target(
                target_spec,
                operators=bound_operators,
            )

        if mode == "pixel_bilinear":

            def _apply_adjoint(tensor: Tensor) -> Tensor:
                image = torch.as_tensor(tensor)
                if image.ndim == 2:
                    return accumulate_image_on_healpix_map_adjoint(
                        image,
                        nside=int(nside),
                        nest=bool(nest),
                        source_header=target.header,
                    )
                if image.ndim == 3:
                    return torch.stack(
                        [
                            accumulate_image_on_healpix_map_adjoint(
                                image[idx],
                                nside=int(nside),
                                nest=bool(nest),
                                source_header=target.header,
                            )
                            for idx in range(int(image.shape[0]))
                        ],
                        dim=0,
                    )
                raise ValueError("adjoint image must be 2D or band-first 3D")
        else:

            def _apply_adjoint(tensor: Tensor) -> Tensor:
                raise NotImplementedError(
                    "exact spherical adjoint is currently implemented for mode='pixel_bilinear' only"
                )

        return _apply_operator, _apply_adjoint, apply_noise

    return build_solver_hooks(
        operators,
        apply_noise=apply_noise,
        bind_target_fn=_bind_target,
    )


def build_spherical_harmonic_solver_hooks(
    *,
    lmax: int,
    mmax: int | None = None,
    apply_noise: ApplyNoiseFn | None = None,
) -> SolverHooks:
    """Build native image↔alm solver hooks for direct harmonic projection."""
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    operators = spherical_spectral_operator_bundle(mode="harmonic_direct")

    def _bind_target(
        bound_operators: ObservationOperatorBundle,
        target: TargetRenderSpec,
    ) -> tuple[ApplyOperatorFn | None, AdjointOperatorFn | None, ApplyNoiseFn | None]:
        def _coerce_alm_scene(
            scene: SceneRepresentation | nn.Module | Tensor,
        ) -> HarmonicAlmScene:
            scene_representation = as_scene_representation(scene)
            if isinstance(scene_representation, HarmonicAlmScene):
                return scene_representation
            if isinstance(scene_representation, DiscreteTargetImage):
                return HarmonicAlmScene(
                    scene_representation.data,
                    lmax=ll,
                    mmax=mm,
                )
            raise TypeError(
                "spherical harmonic solver hooks require a HarmonicAlmScene or tensor alm coefficients"
            )

        def _apply_operator(
            scene: SceneRepresentation, target_spec: TargetRenderSpec
        ) -> Tensor:
            alm_scene = _coerce_alm_scene(scene)
            return alm_scene.render_to_target(
                target_spec,
                operators=bound_operators,
            )

        def _apply_adjoint(tensor: Tensor) -> Tensor:
            image = torch.as_tensor(tensor)
            if image.ndim == 2:
                return accumulate_image_on_alm_adjoint(
                    image,
                    source_header=target.header,
                    lmax=ll,
                    mmax=mm,
                )
            if image.ndim == 3:
                return torch.stack(
                    [
                        accumulate_image_on_alm_adjoint(
                            image[idx],
                            source_header=target.header,
                            lmax=ll,
                            mmax=mm,
                        )
                        for idx in range(int(image.shape[0]))
                    ],
                    dim=0,
                )
            raise ValueError("adjoint image must be 2D or band-first 3D")

        return _apply_operator, _apply_adjoint, apply_noise

    return build_solver_hooks(
        operators,
        apply_noise=apply_noise,
        bind_target_fn=_bind_target,
    )


def build_spherical_spin_harmonic_solver_hooks(
    *,
    spin: int,
    lmax: int,
    mmax: int | None = None,
    apply_noise: ApplyNoiseFn | None = None,
) -> SolverHooks:
    """Build native image↔(E,B) alm solver hooks for direct spin harmonic projection."""
    raw_spin = float(torch.as_tensor(spin).item())
    spin_i = int(raw_spin)
    if float(spin_i) != raw_spin or spin_i < 0:
        raise ValueError("spin must be a non-negative integer")
    ll = int(lmax)
    mm = int(ll if mmax is None else mmax)
    if mm < 0 or mm > ll:
        raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
    operators = spherical_spectral_operator_bundle(mode="harmonic_direct")

    def _bind_target(
        bound_operators: ObservationOperatorBundle,
        target: TargetRenderSpec,
    ) -> tuple[ApplyOperatorFn | None, AdjointOperatorFn | None, ApplyNoiseFn | None]:
        def _coerce_spin_scene(
            scene: SceneRepresentation | nn.Module | Tensor,
        ) -> SpinHarmonicAlmScene:
            scene_representation = as_scene_representation(scene)
            if isinstance(scene_representation, SpinHarmonicAlmScene):
                return scene_representation
            if isinstance(scene_representation, DiscreteTargetImage):
                data = torch.as_tensor(scene_representation.data)
                if data.ndim != 2 or int(data.shape[0]) != 2:
                    raise TypeError(
                        "spin harmonic solver hooks require a SpinHarmonicAlmScene or a (2, nalm) tensor"
                    )
                return SpinHarmonicAlmScene(
                    data[0],
                    data[1],
                    spin=spin_i,
                    lmax=ll,
                    mmax=mm,
                )
            raise TypeError(
                "spin harmonic solver hooks require a SpinHarmonicAlmScene or a (2, nalm) tensor"
            )

        def _apply_operator(
            scene: SceneRepresentation, target_spec: TargetRenderSpec
        ) -> Tensor:
            spin_scene = _coerce_spin_scene(scene)
            return spin_scene.render_to_target(
                target_spec,
                operators=bound_operators,
            )

        def _apply_adjoint(tensor: Tensor) -> Tensor:
            maps = torch.as_tensor(tensor)
            if maps.ndim != 3 or int(maps.shape[0]) != 2:
                raise ValueError("adjoint maps must have shape (2, H, W)")
            alm_e, alm_b = accumulate_image_on_spin_alm_adjoint(
                maps,
                spin=spin_i,
                source_header=target.header,
                lmax=ll,
                mmax=mm,
            )
            return torch.stack([alm_e, alm_b], dim=0)

        return _apply_operator, _apply_adjoint, apply_noise

    return build_solver_hooks(
        operators,
        apply_noise=apply_noise,
        bind_target_fn=_bind_target,
    )
