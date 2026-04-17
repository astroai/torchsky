"""Internal learned-operator scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .background import LearnedBackgroundOp
from .base import ObservationOperator, OperatorConditioning
from .bundle import ObservationOperatorBundle
from .geometry import LearnedGeometryOp
from .noise import LearnedNoiseOp
from .psf import LearnedPSFOp
from .spectral import LearnedSpectralResponseOp


DEFAULT_LEARNED_OPERATOR_CONDITIONING = OperatorConditioning(
    detector_coordinates=True,
    wavelength=True,
    time=True,
    exposure_metadata_keys=("survey", "exposure_id"),
)


@dataclass
class LearnedOperatorScaffold:
    geometry: LearnedGeometryOp
    spectral_response: LearnedSpectralResponseOp
    psf: LearnedPSFOp
    background: LearnedBackgroundOp
    noise: LearnedNoiseOp


@dataclass(frozen=True)
class LearnedOperatorSlot:
    family: str
    operator_name: str
    conditioning: OperatorConditioning
    enabled: bool
    frozen: bool


@dataclass(frozen=True)
class LearnedOperatorManifest:
    slots: tuple[LearnedOperatorSlot, ...]

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(slot.family for slot in self.slots)

    @property
    def conditioning_union(self) -> OperatorConditioning:
        keys: list[str] = []
        for slot in self.slots:
            for key in slot.conditioning.exposure_metadata_keys:
                if key not in keys:
                    keys.append(key)
        return OperatorConditioning(
            detector_coordinates=any(
                slot.conditioning.detector_coordinates for slot in self.slots
            ),
            wavelength=any(slot.conditioning.wavelength for slot in self.slots),
            time=any(slot.conditioning.time for slot in self.slots),
            exposure_metadata_keys=tuple(keys),
        )


LEARNED_OPERATOR_FAMILIES = (
    "geometry",
    "spectral_response",
    "psf",
    "background",
    "noise",
)

LEARNED_OPERATOR_REGISTRY: dict[str, type[ObservationOperator]] = {
    "geometry": LearnedGeometryOp,
    "spectral_response": LearnedSpectralResponseOp,
    "psf": LearnedPSFOp,
    "background": LearnedBackgroundOp,
    "noise": LearnedNoiseOp,
}


def validate_learned_operator_families(families: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(families)
    unknown = tuple(name for name in selected if name not in LEARNED_OPERATOR_FAMILIES)
    if unknown:
        raise ValueError(f"unsupported learned operator families: {unknown!r}")
    return selected


def make_minimal_learned_operator_scaffold(
    *,
    conditioning: OperatorConditioning = DEFAULT_LEARNED_OPERATOR_CONDITIONING,
) -> LearnedOperatorScaffold:
    geometry = LearnedGeometryOp()
    geometry.conditioning = conditioning
    spectral = LearnedSpectralResponseOp()
    spectral.conditioning = conditioning
    psf = LearnedPSFOp()
    psf.conditioning = conditioning
    background = LearnedBackgroundOp()
    background.conditioning = conditioning
    noise = LearnedNoiseOp()
    noise.conditioning = conditioning
    return LearnedOperatorScaffold(
        geometry=geometry,
        spectral_response=spectral,
        psf=psf,
        background=background,
        noise=noise,
    )


def learned_operator_manifest(
    scaffold: LearnedOperatorScaffold,
    *,
    families: Iterable[str] | None = None,
) -> LearnedOperatorManifest:
    selected = validate_learned_operator_families(
        LEARNED_OPERATOR_FAMILIES if families is None else families
    )
    slots = tuple(
        LearnedOperatorSlot(
            family=family,
            operator_name=getattr(scaffold, family).name,
            conditioning=getattr(scaffold, family).conditioning,
            enabled=getattr(scaffold, family).enabled,
            frozen=getattr(scaffold, family).frozen,
        )
        for family in selected
    )
    return LearnedOperatorManifest(slots=slots)


def swap_bundle_operator_families(
    bundle: ObservationOperatorBundle,
    learned: LearnedOperatorScaffold,
    *,
    families: Iterable[str] | None = None,
) -> ObservationOperatorBundle:
    selected = validate_learned_operator_families(
        LEARNED_OPERATOR_FAMILIES if families is None else families
    )
    updates = {name: getattr(learned, name) for name in selected}
    return bundle.with_updates(**updates)
