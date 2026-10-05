"""Internal bundle of observation operators shared across mapmaking paths."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .background import DetectorBackgroundOp
from .base import (
    BackgroundOp,
    GeometryOp,
    MaskWeightOp,
    NoiseOp,
    PSFOp,
    RadiometryOp,
    SpectralResponseOp,
)
from .geometry import (
    ConservativePolygonRemapOp,
    ExactDrizzleGeometryOp,
    ForwardSplatGeometryOp,
    InterpolationGeometryOp,
    RestrictedAffineFFTGeometryOp,
    SphericalSpectralGeometryOp,
    SpectralGlobalGeometryOp,
)
from .mask_weight import ExplicitMaskWeightOp
from .noise import ExplicitNoiseOp
from .psf import ExplicitPSFOp
from .radiometry import ExplicitRadiometryOp
from .spectral import ExplicitSpectralResponseOp


OPERATOR_BUNDLE_SLOT_NAMES = (
    "geometry",
    "spectral_response",
    "psf",
    "radiometry",
    "background",
    "mask_weight",
    "noise",
)


@dataclass
class ObservationOperatorBundle:
    """Composable observation-law bundle used internally by map and INR paths."""

    geometry: GeometryOp | None = None
    spectral_response: SpectralResponseOp | None = None
    psf: PSFOp | None = None
    radiometry: RadiometryOp | None = None
    background: BackgroundOp | None = None
    mask_weight: MaskWeightOp | None = None
    noise: NoiseOp | None = None

    def iter_slots(self):
        for slot_name in OPERATOR_BUNDLE_SLOT_NAMES:
            yield slot_name, getattr(self, slot_name)

    def iter_enabled(self):
        for _, op in self.iter_slots():
            if op is not None and op.enabled:
                yield op

    def enabled_operator_names(self) -> dict[str, str]:
        return {
            slot_name: op.name
            for slot_name, op in self.iter_slots()
            if op is not None and op.enabled
        }

    def with_updates(self, **updates) -> "ObservationOperatorBundle":
        return replace(self, **updates)


def geometry_op_for_kernel(kernel: str) -> GeometryOp:
    if kernel in {
        "nearest",
        "bilinear",
        "bicubic",
        "lanczos2",
        "lanczos3",
        "lanczos4",
        "lanczos5",
        "mitchell_netravali",
        "catmull_rom",
    }:
        return InterpolationGeometryOp()
    if kernel in {"drizzle_point", "drizzle_square"}:
        return ExactDrizzleGeometryOp()
    if kernel == "conservative_polygon":
        return ConservativePolygonRemapOp()
    if kernel in {
        "splat_point",
        "splat_box",
        "drizzle_turbo",
        "drizzle_gaussian",
        "drizzle_lanczos2",
        "drizzle_lanczos3",
    }:
        return ForwardSplatGeometryOp()
    if kernel in {
        "fft_shift",
        "fft_diagonal_affine",
        "periodic_linear_affine",
        "fft_linear_affine",
    }:
        return RestrictedAffineFFTGeometryOp()
    if kernel in {
        "spectral_linear_affine",
        "spectral_nonuniform",
        "spectral_nonuniform_fast",
    }:
        return SpectralGlobalGeometryOp()
    raise ValueError(f"unsupported geometry kernel {kernel!r}")


def spherical_spectral_operator_bundle(
    *,
    mode: str = "harmonic_direct",
) -> ObservationOperatorBundle:
    return ObservationOperatorBundle(
        geometry=SphericalSpectralGeometryOp(mode=mode),
    )


def _merge_operator_slot(values: Iterable[object | None]) -> object | None:
    selected = None
    for value in values:
        if value is None:
            continue
        if selected is None:
            selected = value
            continue
        if getattr(value, "learned", False) and not getattr(selected, "learned", False):
            selected = value
    return selected


def merge_operator_bundles(
    *bundles: ObservationOperatorBundle,
) -> ObservationOperatorBundle:
    return ObservationOperatorBundle(
        geometry=_merge_operator_slot(bundle.geometry for bundle in bundles),
        spectral_response=_merge_operator_slot(
            bundle.spectral_response for bundle in bundles
        ),
        psf=_merge_operator_slot(bundle.psf for bundle in bundles),
        radiometry=_merge_operator_slot(bundle.radiometry for bundle in bundles),
        background=_merge_operator_slot(bundle.background for bundle in bundles),
        mask_weight=_merge_operator_slot(bundle.mask_weight for bundle in bundles),
        noise=_merge_operator_slot(bundle.noise for bundle in bundles),
    )


def image_observation_operator_bundle(
    observation: object,
    *,
    resampling_kernel: str,
) -> ObservationOperatorBundle:
    has_spectral = any(
        getattr(observation, name, None) is not None
        for name in (
            "wavelength",
            "spectral_min",
            "spectral_max",
            "transmission_center",
            "transmission_sigma",
        )
    )
    has_background = float(
        getattr(observation, "background_offset", 0.0)
    ) != 0.0 or any(
        getattr(observation, name, None) is not None
        for name in ("sky_background", "instrument_background", "detector_background")
    )
    has_radiometry = float(getattr(observation, "photometric_scale", 1.0)) != 1.0 or any(
        getattr(observation, name, None) is not None
        for name in ("flux_scale_keyword", "exposure_time_keyword", "gain_keyword")
    )
    has_mask_weight = float(getattr(observation, "weight", 1.0)) != 1.0 or any(
        getattr(observation, name, None) is not None
        for name in (
            "weight_path",
            "mask_path",
            "primary_beam_path",
            "weight_map",
            "mask",
            "primary_beam",
            "var_poisson",
            "var_rnoise",
            "var_flat",
            "weight_thresh",
        )
    ) or str(getattr(observation, "weight_type", "weight")) != "weight"
    has_noise = (
        getattr(observation, "weight_path", None) is not None
        or getattr(observation, "weight_map", None) is not None
        or getattr(observation, "var_poisson", None) is not None
        or getattr(observation, "var_rnoise", None) is not None
        or getattr(observation, "var_flat", None) is not None
        or bool(getattr(observation, "weight_is_inverse_variance", False))
        or str(getattr(observation, "weight_type", "weight")) in {"ivm", "rms", "variance", "background"}
    )
    return ObservationOperatorBundle(
        geometry=geometry_op_for_kernel(resampling_kernel),
        spectral_response=ExplicitSpectralResponseOp() if has_spectral else None,
        psf=None,
        radiometry=ExplicitRadiometryOp() if has_radiometry else None,
        background=DetectorBackgroundOp() if has_background else None,
        mask_weight=ExplicitMaskWeightOp() if has_mask_weight else None,
        noise=ExplicitNoiseOp() if has_noise else None,
    )


def image_observations_operator_bundle(
    observations: Iterable[object],
    *,
    resampling_kernel: str,
) -> ObservationOperatorBundle:
    bundles = [
        image_observation_operator_bundle(
            observation,
            resampling_kernel=resampling_kernel,
        )
        for observation in observations
    ]
    if not bundles:
        return ObservationOperatorBundle(
            geometry=geometry_op_for_kernel(resampling_kernel)
        )
    return merge_operator_bundles(*bundles)


def field_observation_operator_bundle(
    observation: object,
    *,
    geometry: GeometryOp | None = None,
) -> ObservationOperatorBundle:
    psf_spec = getattr(observation, "psf_spec", None)
    has_psf = psf_spec is not None and getattr(psf_spec, "mode", "none") != "none"
    has_spectral = any(
        getattr(observation, name, None) is not None
        for name in (
            "wavelength",
            "spectral_min",
            "spectral_max",
            "transmission_center",
            "transmission_sigma",
            "transmission_wavelength",
            "transmission_throughput",
        )
    )
    has_background = float(
        getattr(observation, "background_offset", 0.0)
    ) != 0.0 or any(
        getattr(observation, name, None) is not None
        for name in ("sky_background", "instrument_background", "detector_background")
    )
    has_mask_weight = any(
        getattr(observation, name, None) is not None for name in ("mask",)
    )
    has_noise = has_mask_weight or getattr(observation, "weight", None) is not None
    has_radiometry = any(
        getattr(observation, name, None) is not None
        for name in (
            "gain_electrons_per_adu",
            "exposure_time_s",
            "collecting_area_m2",
        )
    )
    return ObservationOperatorBundle(
        geometry=geometry,
        spectral_response=ExplicitSpectralResponseOp() if has_spectral else None,
        psf=(
            ExplicitPSFOp(
                pixel_response_included=getattr(
                    psf_spec, "pixel_response_included", None
                )
            )
            if has_psf
            else None
        ),
        radiometry=ExplicitRadiometryOp() if has_radiometry else None,
        background=DetectorBackgroundOp() if has_background else None,
        mask_weight=ExplicitMaskWeightOp() if has_mask_weight else None,
        noise=ExplicitNoiseOp() if has_noise else None,
    )


def field_observations_operator_bundle(
    observations: Iterable[object],
    *,
    geometry: GeometryOp | None = None,
) -> ObservationOperatorBundle:
    bundles = [
        field_observation_operator_bundle(
            observation,
            geometry=geometry,
        )
        for observation in observations
    ]
    if not bundles:
        return ObservationOperatorBundle(geometry=geometry)
    return merge_operator_bundles(*bundles)
