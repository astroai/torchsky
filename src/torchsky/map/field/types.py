"""Shared types and dataclasses for continuous-field (INR) models.

Implementation of ``nn.Module`` subclasses and training live in :mod:`torchsky.map.field.core`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

import math
import torch
import torch.nn as nn
from torch import Tensor

from ...spectrophotometry import Passband

if TYPE_CHECKING:
    from ..exposure_pose import ExposurePose
    from ..operators.bundle import ObservationOperatorBundle

SpectralCapacityProfile = Literal[
    "photometric_wide",
    "multiband",
    "narrowband_sweep",
    "spectroscopic",
]

FieldModelKind = Literal[
    "sine_implicit",
    "wire_implicit",
    "gaussian_basis",
    "fourier_mlp",
    "spectral_power_foam",
    "foam_initialized_inr",
    "diffuse_foam_hybrid",
    "spectral_basis_field",
    "learned_spectral_basis_field",
    "sky_encoder_bandpass_field",
    "sky_lambda_planes_field",
    "sky_lambda_kplanes_field",
    "sky_lambda_feature_field",
    "operator_constrained_feature_field",
    "sky_lambda_feature_field_hetero",
    "sky_lambda_feature_atlas_field",
    "sky_lambda_atlas_field",
    "bandpass_conditioned_field",
    "conditioned_spectral_field",
    "hashgrid_conditioned_field",
    "sky_encoder_mlp",
    "strainer_sky_encoder_mlp",
    "sky_encoder_resmlp",
    "hashgrid_mlp",
    "hashgrid_resmlp",
    "hashgrid_spectral_field",
]
FieldCoordinateMode = Literal["local_tangent", "sphere_xyz"]
FieldLossKind = Literal["mse", "charbonnier", "huber", "gaussian_nll", "structured_nll"]
FieldLRSchedule = Literal["constant", "cosine"]
FieldLossReduction = Literal["per_observation", "per_band", "global", "band_global_mix"]
FieldCurriculum = Literal["none", "spectral_frequency_ramp", "feature_scale_ramp"]
FieldSpectralSampling = Literal[
    "deterministic_quadrature",
    "bandpass_mc",
    "mixture_mc",
]
FieldSpatialSampling = Literal["center", "gauss_legendre_2x2"]
FieldAMPDtype = Literal["none", "float16", "bfloat16"]
FieldObservationSampling = Literal["uniform", "band_balanced", "epoch"]
FieldNoisePretrainKind = Literal["none", "uniform", "gaussian", "spectrum"]
# wavelength | energy | frequency — meaning of spectral_min / spectral_max numeric values
FieldSpectralQuantity = Literal["wavelength", "energy", "frequency"]
TimeScale = Literal["tai", "tcb", "tcg", "tdb", "tt", "ut1", "utc"]
DetectorLaw = Literal["gaussian", "poisson_gaussian", "student_t"]

_TIME_SCALES = {"tai", "tcb", "tcg", "tdb", "tt", "ut1", "utc"}
_DETECTOR_LAWS = {"gaussian", "poisson_gaussian", "student_t"}


def _validate_optional_identifier(value: str | None, *, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{name} must be a non-empty string when provided")


class DetectorToSkyTransform(Protocol):
    """Forward-only differentiable detector-to-sky transform.

    Coordinates passed to this protocol are full-detector, zero-based pixel
    coordinates. Implementations return ICRS-like longitude and latitude in
    degrees and may own trainable exposure-pose parameters.
    """

    def pixel_to_world(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True)
class ForwardContext:
    """Exposure / instrument context ``c`` (see ``implicit_forward.md`` §2–3).

    Rendering passes this to ``nn.Module.forward`` or ``render_bandpass`` **only when**
    the model declares a ``forward_context`` keyword parameter. Otherwise it is ignored
    so existing field models stay valid without changes.
    """

    survey: str | None = None
    instrument: str | None = None
    detector: str | None = None
    exposure_id: str | None = None
    time_mjd: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    time_scale: TimeScale | None = None
    calibration_id: str | None = None
    provenance_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "survey",
            "instrument",
            "detector",
            "exposure_id",
            "calibration_id",
            "provenance_id",
        ):
            _validate_optional_identifier(getattr(self, name), name=name)
        if self.time_mjd is not None and not math.isfinite(float(self.time_mjd)):
            raise ValueError("ForwardContext time_mjd must be finite")
        if self.time_scale is not None:
            scale = str(self.time_scale).lower()
            if scale not in _TIME_SCALES:
                raise ValueError(
                    f"unsupported ForwardContext time_scale {self.time_scale!r}"
                )
            if self.time_mjd is None:
                raise ValueError("ForwardContext time_scale requires time_mjd")
            object.__setattr__(self, "time_scale", scale)
        if not isinstance(self.extra, dict):
            raise TypeError("ForwardContext extra must be a dict")

    def validate_certified(self) -> "ForwardContext":
        """Fail closed unless the context identifies one calibrated exposure."""
        missing = [
            name
            for name in (
                "survey",
                "instrument",
                "detector",
                "exposure_id",
                "calibration_id",
                "provenance_id",
                "time_mjd",
                "time_scale",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                "certified ForwardContext is missing " + ", ".join(missing)
            )
        return self


FieldPSFMode = Literal["none", "gaussian_sigma_pix", "kernel2d", "learned_lowrank"]


@dataclass
class PSFSpec:
    """Public PSF description for field observations and target products.

    Learned PSFs are calibration products and are frozen while rendering a sky
    scene.  ``frozen=False`` is reserved for a future calibration-only execution
    path; the scene renderer rejects it instead of silently dropping gradients.
    """

    mode: FieldPSFMode = "none"
    sigma_pix: float = 0.0
    kernel2d: Tensor | None = None
    model: nn.Module | None = None
    cell_u_pix: Tensor | None = None
    cell_lam_norm: float = 0.5
    frozen: bool = True
    pixel_response_included: bool | None = None
    """Whether the calibrated kernel already includes detector pixel response.

    True requires center sampling; another pixel quadrature would integrate twice.
    False records an optical-only kernel; None leaves legacy calibration unknown.
    This metadata does not certify the accuracy of discrete detector-grid convolution.
    """

    @classmethod
    def gaussian(cls, sigma_pix: float) -> "PSFSpec":
        return cls(mode="gaussian_sigma_pix", sigma_pix=float(sigma_pix))

    @classmethod
    def kernel(cls, kernel2d: Tensor) -> "PSFSpec":
        return cls(mode="kernel2d", kernel2d=kernel2d)

    @classmethod
    def learned(
        cls,
        model: nn.Module,
        *,
        base_sigma_pix: float,
        cell_u_pix: Tensor | None = None,
        cell_lam_norm: float = 0.5,
        frozen: bool = True,
    ) -> "PSFSpec":
        return cls(
            mode="learned_lowrank",
            sigma_pix=float(base_sigma_pix),
            model=model,
            cell_u_pix=cell_u_pix,
            cell_lam_norm=float(cell_lam_norm),
            frozen=bool(frozen),
        )

    def __post_init__(self) -> None:
        if self.pixel_response_included is not None and not isinstance(
            self.pixel_response_included, bool
        ):
            raise TypeError("pixel_response_included must be bool or None")
        if self.mode == "none":
            return
        if self.mode == "gaussian_sigma_pix":
            sigma = float(self.sigma_pix)
            if not math.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(
                    "PSFSpec gaussian_sigma_pix requires finite sigma_pix > 0"
                )
            return
        if self.mode == "kernel2d":
            if self.kernel2d is None:
                raise ValueError("PSFSpec kernel2d mode requires kernel2d")
            if self.kernel2d.ndim != 2:
                raise ValueError("PSFSpec kernel2d must be 2D")
            if self.kernel2d.numel() == 0:
                raise ValueError("PSFSpec kernel2d must be non-empty")
            if not bool(self.kernel2d.isfinite().all()):
                raise ValueError("PSFSpec kernel2d must be finite")
            if bool((self.kernel2d < 0).any()):
                raise ValueError("PSFSpec kernel2d must be non-negative")
            if not bool(self.kernel2d.sum() > 0):
                raise ValueError("PSFSpec kernel2d must have positive sum")
            if any(int(size) % 2 == 0 for size in self.kernel2d.shape):
                raise ValueError("PSFSpec kernel2d dimensions must be odd")
            return
        if self.mode == "learned_lowrank":
            if self.model is None:
                raise ValueError("PSFSpec learned_lowrank mode requires a model")
            sigma = float(self.sigma_pix)
            if not math.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(
                    "PSFSpec learned_lowrank requires finite base sigma_pix > 0"
                )
            return
        raise ValueError(f"unsupported PSFSpec mode {self.mode!r}")

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "PSFSpec":
        """Move tensor payloads while retaining the declared PSF contract."""
        if dtype is not None and not dtype.is_floating_point:
            raise TypeError("PSFSpec dtype must be floating point")
        move_kwargs: dict[str, object] = {}
        if device is not None:
            move_kwargs["device"] = device
        if dtype is not None:
            move_kwargs["dtype"] = dtype
        kernel = self.kernel2d
        cells = self.cell_u_pix
        if move_kwargs:
            kernel = None if kernel is None else kernel.to(**move_kwargs)
            cells = None if cells is None else cells.to(**move_kwargs)
            if self.model is not None:
                self.model.to(**move_kwargs)
        return PSFSpec(
            mode=self.mode,
            sigma_pix=float(self.sigma_pix),
            kernel2d=kernel,
            model=self.model,
            cell_u_pix=cells,
            cell_lam_norm=float(self.cell_lam_norm),
            frozen=bool(self.frozen),
            pixel_response_included=self.pixel_response_included,
        )


@dataclass(frozen=True)
class FieldDomain:
    """Sky + wavelength normalization box for continuous field models.

    ``spatial_scale_deg`` is a footprint scale in degrees (from WCS pixel scale × extent).
    Sky-encoder fields use it when ``scale_fourier_to_domain=True`` to choose spherical
    Fourier ``min_frequency`` so bands resolve structure inside the patch.
    """

    center_ra_deg: float
    center_dec_deg: float
    spatial_scale_deg: float
    wavelength_center: float
    wavelength_scale: float
    coord_mode: FieldCoordinateMode = "local_tangent"


@dataclass(frozen=True)
class ObservationSpectralSummary:
    """Per-observation spectral-axis support for :func:`spectral_data_budget`."""

    lower: float
    upper: float
    span: float
    transmission_sigma: float | None
    """Gaussian width in the same units as the spectral axis, if defined and valid."""
    variation_scale: float
    """Scale used for quadrature spacing (``min(sigma, span)`` or delta fallback)."""
    band_name: str | None


@dataclass(frozen=True)
class SpectralDataBudget:
    """Aggregated spectral complexity from a list of :class:`FieldObservation`."""

    observations: tuple[ObservationSpectralSummary, ...]
    wavelength_min: float
    wavelength_max: float
    wavelength_span: float
    min_variation_scale: float
    max_variation_scale: float
    n_distinct_band_specs: int


@dataclass(frozen=True)
class SpectralModelCapacityHints:
    """Suggested encoder / model capacity from a :class:`SpectralDataBudget`."""

    spectroscopicity: float
    """``wavelength_span / min_variation_scale`` (large ⇒ finer λ structure)."""
    profile: SpectralCapacityProfile
    spectral_frequencies: int
    """Suggested ``spectral_frequencies`` for :class:`SkyCoordinateEncoder` / sky MLP fields."""
    num_spectral_frequencies_basis: int
    """Suggested ``num_spatial_frequencies``-style order for spectral-basis families."""
    recommended_model_kinds: tuple[str, ...]
    rationale: str


@dataclass
class FieldObservation:
    """One band-limited 2D map plus spectral metadata for field fitting.

    Use :mod:`torchsky.map.field.observation_builders` to turn binned 1D spectra,
    IFU channel cubes, or X-ray-style center/width summaries into observation lists.
    Set ``spectral_quantity`` when the axis is energy (e.g. keV) rather than wavelength.

    Optional ``transmission_wavelength`` / ``transmission_throughput`` give a tabulated
    throughput curve (same units as the spectral axis, dimensionless throughput).
    When both are set, bandpass quadrature integrates against this curve (trapezoid
    rule on the quadrature grid, with linear interpolation of throughput) instead of
    the uniform-in-band or Gaussian fallbacks from ``transmission_sigma``.
    """

    image: Tensor
    header: dict
    weight: Tensor | None = None
    mask: Tensor | None = None
    err: Tensor | None = None
    var_poisson: Tensor | None = None
    var_rnoise: Tensor | None = None
    var_flat: Tensor | None = None
    contamination_prior: Tensor | None = None
    amplifier_index: Tensor | None = None
    wavelength: float | None = None
    spectral_min: float | None = None
    spectral_max: float | None = None
    transmission_center: float | None = None
    transmission_sigma: float | None = None
    transmission_wavelength: Tensor | None = None
    transmission_throughput: Tensor | None = None
    psf_sigma_pix: float = 0.0
    psf_spec: PSFSpec | None = None
    sky_background: Tensor | None = None
    instrument_background: Tensor | None = None
    detector_background: Tensor | None = None
    background_offset: float = 0.0
    band_name: str | None = None
    spectral_quantity: FieldSpectralQuantity | None = None
    detector_x0_pix: float = 0.0
    detector_y0_pix: float = 0.0
    detector_to_sky: DetectorToSkyTransform | None = None
    passband: Passband | None = None
    value_unit: str | None = None
    value_semantics: str | None = None
    spectral_unit: str | None = None
    detector_law: DetectorLaw | None = None
    gain_electrons_per_adu: Tensor | float | None = None
    gain_provenance_id: str | None = None
    exposure_time_s: float | None = None
    collecting_area_m2: float | None = None

    _cached_ra_deg: Tensor | None = field(default=None, init=False, repr=False)
    _cached_dec_deg: Tensor | None = field(default=None, init=False, repr=False)
    _cached_coordinate_signature: tuple[Any, ...] | None = field(
        default=None, init=False, repr=False
    )

    def clear_coordinate_cache(self) -> None:
        """Drop cached WCS coordinate grids after mutating image shape or header."""
        self._cached_ra_deg = None
        self._cached_dec_deg = None
        self._cached_coordinate_signature = None

    def coordinate_cache_signature(self, shape: tuple[int, int]) -> tuple[Any, ...]:
        """Return a lightweight signature for cached WCS coordinate grids."""
        header_items = tuple(
            sorted((str(key), repr(value)) for key, value in self.header.items())
        )
        return (tuple(self.image.shape), tuple(shape), header_items)

    def __post_init__(self) -> None:
        if not isinstance(self.image, Tensor) or self.image.ndim != 2:
            raise ValueError("FieldObservation image must be a 2D tensor")
        if not self.image.is_floating_point():
            raise TypeError("FieldObservation image must be floating point")
        if not isinstance(self.header, dict):
            raise TypeError("FieldObservation header must be a dict")
        if self.psf_spec is None:
            if float(self.psf_sigma_pix) > 0.0:
                self.psf_spec = PSFSpec.gaussian(float(self.psf_sigma_pix))
            else:
                self.psf_spec = PSFSpec()
        elif (
            self.psf_spec.mode == "gaussian_sigma_pix"
            and float(self.psf_sigma_pix) > 0.0
        ):
            sigma = float(self.psf_sigma_pix)
            if not math.isclose(
                float(self.psf_spec.sigma_pix), sigma, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError(
                    "FieldObservation psf_sigma_pix must agree with psf_spec"
                )
        if not math.isfinite(float(self.background_offset)):
            raise ValueError("FieldObservation background_offset must be finite")
        if not all(
            math.isfinite(float(value))
            for value in (self.detector_x0_pix, self.detector_y0_pix)
        ):
            raise ValueError("FieldObservation detector origins must be finite")
        for name in (
            "weight",
            "mask",
            "err",
            "var_poisson",
            "var_rnoise",
            "var_flat",
            "contamination_prior",
            "amplifier_index",
            "sky_background",
            "instrument_background",
            "detector_background",
        ):
            component = getattr(self, name)
            if component is None:
                continue
            if component.ndim != 2:
                raise ValueError(f"FieldObservation {name} must be 2D")
            if tuple(component.shape) != tuple(self.image.shape):
                raise ValueError(
                    f"FieldObservation {name} shape must match image shape"
                )
        if self.detector_law is not None and self.detector_law not in _DETECTOR_LAWS:
            raise ValueError(
                f"unsupported FieldObservation detector_law {self.detector_law!r}"
            )
        for name in (
            "value_unit",
            "value_semantics",
            "spectral_unit",
            "gain_provenance_id",
        ):
            _validate_optional_identifier(getattr(self, name), name=name)
        for name in ("exposure_time_s", "collecting_area_m2"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) <= 0.0
            ):
                raise ValueError(f"FieldObservation {name} must be finite and positive")
        if self.gain_electrons_per_adu is not None:
            gain = torch.as_tensor(self.gain_electrons_per_adu)
            if not bool(torch.isfinite(gain).all()) or bool((gain <= 0.0).any()):
                raise ValueError(
                    "FieldObservation gain_electrons_per_adu must be finite and positive"
                )
        if self.contamination_prior is not None:
            prior = self.contamination_prior
            if not bool(torch.isfinite(prior).all()) or bool(
                ((prior < 0.0) | (prior > 1.0)).any()
            ):
                raise ValueError(
                    "FieldObservation contamination_prior must be finite in [0, 1]"
                )
        if self.amplifier_index is not None:
            if self.amplifier_index.is_floating_point() or bool(
                (self.amplifier_index < 0).any()
            ):
                raise ValueError(
                    "FieldObservation amplifier_index must be a non-negative integer tensor"
                )

        wl = self.transmission_wavelength
        tt = self.transmission_throughput
        if self.passband is not None:
            if (
                wl is not None
                or tt is not None
                or self.transmission_center is not None
                or self.transmission_sigma is not None
            ):
                raise ValueError(
                    "FieldObservation passband and alternate transmission metadata are mutually exclusive"
                )
            if self.spectral_quantity not in (None, "wavelength"):
                raise ValueError(
                    "Passband is defined only for wavelength spectral coordinates"
                )
            if (
                self.spectral_unit is not None
                and self.spectral_unit.strip().lower()
                not in {
                    "angstrom",
                    "angstroms",
                    "aa",
                }
            ):
                raise ValueError("Passband coordinates use Angstrom spectral units")
            return
        if (wl is None) ^ (tt is None):
            raise ValueError(
                "transmission_wavelength and transmission_throughput must both be set or both None"
            )
        if wl is None:
            return
        if wl.ndim != 1 or tt.ndim != 1:
            raise ValueError(
                "transmission_wavelength and transmission_throughput must be 1D"
            )
        if wl.shape != tt.shape:
            raise ValueError(
                "transmission_wavelength and transmission_throughput shapes must match"
            )
        if wl.numel() < 2:
            raise ValueError("tabulated transmission requires at least two samples")
        if not bool(torch.isfinite(wl).all()) or not bool(torch.isfinite(tt).all()):
            raise ValueError("tabulated transmission samples must be finite")
        if bool((tt < 0.0).any()):
            raise ValueError("tabulated transmission throughput must be non-negative")
        if not bool(tt.sum() > 0.0):
            raise ValueError(
                "tabulated transmission requires positive total throughput"
            )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "FieldObservation":
        """Return a moved observation with mask/index dtypes preserved.

        Floating detector planes follow ``dtype``. Boolean masks and integer
        amplifier indices retain their semantic dtype. Coordinate caches are
        intentionally dropped by :func:`dataclasses.replace`.
        """
        if dtype is not None and not dtype.is_floating_point:
            raise TypeError("FieldObservation dtype must be floating point")
        kwargs: dict[str, object] = {}
        floating_fields = (
            "image",
            "weight",
            "err",
            "var_poisson",
            "var_rnoise",
            "var_flat",
            "contamination_prior",
            "sky_background",
            "instrument_background",
            "detector_background",
            "transmission_wavelength",
            "transmission_throughput",
            "gain_electrons_per_adu",
        )
        for name in floating_fields:
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                continue
            target_dtype = dtype if dtype is not None else value.dtype
            kwargs[name] = value.to(device=device, dtype=target_dtype)
        for name in ("mask", "amplifier_index"):
            value = getattr(self, name)
            if isinstance(value, Tensor):
                kwargs[name] = value.to(device=device)
        if self.psf_spec is not None:
            kwargs["psf_spec"] = self.psf_spec.to(device=device, dtype=dtype)
        if self.passband is not None and device is not None:
            kwargs["passband"] = self.passband.to(device)
        transform = self.detector_to_sky
        if isinstance(transform, nn.Module):
            move_kwargs: dict[str, object] = {}
            if device is not None:
                move_kwargs["device"] = device
            if dtype is not None:
                move_kwargs["dtype"] = dtype
            if move_kwargs:
                transform.to(**move_kwargs)
                kwargs["detector_to_sky"] = transform
        return replace(self, **kwargs)

    def validate_certified(
        self,
        forward_context: ForwardContext,
    ) -> "FieldObservation":
        """Validate the strict native-observation contract used for claims.

        Legacy and exploratory observations remain valid without these fields;
        production adapters opt into this fail-closed check explicitly.
        """
        if not isinstance(forward_context, ForwardContext):
            raise TypeError("certified observation requires a ForwardContext")
        forward_context.validate_certified()
        if self.weight is None or self.mask is None:
            raise ValueError("certified observation requires weight and mask planes")
        if self.mask.dtype != torch.bool:
            raise TypeError(
                "certified observation mask must be boolean (True = usable)"
            )
        if not bool(self.mask.any()):
            raise ValueError("certified observation requires at least one usable pixel")
        if not bool(torch.isfinite(self.weight).all()) or bool(
            (self.weight < 0.0).any()
        ):
            raise ValueError(
                "certified inverse variance must be finite and non-negative"
            )
        if bool((self.weight[~self.mask] != 0.0).any()):
            raise ValueError(
                "certified unusable pixels must have zero inverse variance"
            )
        if not bool((self.weight[self.mask] > 0.0).all()):
            raise ValueError(
                "certified usable pixels require positive inverse variance"
            )
        if not bool(torch.isfinite(self.image[self.mask]).all()):
            raise ValueError("certified usable image pixels must be finite")
        if self.psf_spec is None or self.psf_spec.mode == "none":
            raise ValueError("certified observation requires an explicit PSF")
        if self.passband is None and not (
            self.transmission_wavelength is not None
            and self.transmission_throughput is not None
        ):
            raise ValueError("certified observation requires a tabulated passband")
        missing = [
            name
            for name in (
                "value_unit",
                "value_semantics",
                "spectral_unit",
                "spectral_quantity",
                "detector_law",
                "gain_electrons_per_adu",
                "gain_provenance_id",
                "exposure_time_s",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                "certified FieldObservation is missing " + ", ".join(missing)
            )
        from ...wcs import WCS

        height, width = self.image.shape
        centre = self.image.new_tensor([(width - 1) * 0.5])
        centre_y = self.image.new_tensor([(height - 1) * 0.5])
        ra, dec = WCS(self.header).pixel_to_world(centre, centre_y)
        if not bool(torch.isfinite(ra).all() & torch.isfinite(dec).all()):
            raise ValueError("certified observation WCS is not finite at image centre")
        return self


@dataclass(frozen=True)
class FieldComponentAnchorTerm:
    """Explicit calibration / standards anchor for selected field decomposition roles."""

    observation: FieldObservation
    target_image: Tensor | None = None
    target_values: Tensor | None = None
    reference_model: nn.Module | None = None
    roles: str | tuple[str, ...] = "shared"
    sample_ra_deg: Tensor | None = None
    sample_dec_deg: Tensor | None = None
    sample_x_pix: Tensor | None = None
    sample_y_pix: Tensor | None = None
    sample_weight: Tensor | None = None
    override_wavelengths: Tensor | None = None
    override_bandpass_weights: Tensor | None = None
    forward_context: ForwardContext | None = None
    loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.loss_weight < 0.0:
            raise ValueError("loss_weight must be non-negative")
        provided_target_count = (
            int(self.target_image is not None)
            + int(self.target_values is not None)
            + int(self.reference_model is not None)
        )
        if provided_target_count != 1:
            raise ValueError(
                "exactly one of target_image, target_values, or reference_model must be provided"
            )
        world_sample_mode = (
            self.sample_ra_deg is not None or self.sample_dec_deg is not None
        )
        pixel_sample_mode = (
            self.sample_x_pix is not None or self.sample_y_pix is not None
        )
        if world_sample_mode and pixel_sample_mode:
            raise ValueError(
                "sample_ra_deg/sample_dec_deg and sample_x_pix/sample_y_pix are mutually exclusive"
            )
        sample_mode = world_sample_mode or pixel_sample_mode
        sample_shape = None
        if world_sample_mode:
            if self.sample_ra_deg is None or self.sample_dec_deg is None:
                raise ValueError(
                    "sample_ra_deg and sample_dec_deg must be provided together"
                )
            if tuple(self.sample_ra_deg.shape) != tuple(self.sample_dec_deg.shape):
                raise ValueError("sample_ra_deg and sample_dec_deg must match in shape")
            sample_shape = tuple(self.sample_ra_deg.shape)
        if pixel_sample_mode:
            if self.sample_x_pix is None or self.sample_y_pix is None:
                raise ValueError(
                    "sample_x_pix and sample_y_pix must be provided together"
                )
            if tuple(self.sample_x_pix.shape) != tuple(self.sample_y_pix.shape):
                raise ValueError("sample_x_pix and sample_y_pix must match in shape")
            sample_shape = tuple(self.sample_x_pix.shape)
        if sample_mode:
            if self.target_image is not None:
                raise ValueError("target_image is only valid for full-image anchors")
            if (
                self.target_values is not None
                and tuple(self.target_values.shape) != sample_shape
            ):
                raise ValueError("target_values shape must match sample coordinates")
            if (
                self.sample_weight is not None
                and tuple(self.sample_weight.shape) != sample_shape
            ):
                raise ValueError("sample_weight shape must match sample coordinates")
        else:
            if self.target_values is not None:
                raise ValueError(
                    "target_values requires sample_ra_deg/sample_dec_deg or sample_x_pix/sample_y_pix"
                )
            if self.sample_weight is not None:
                raise ValueError(
                    "sample_weight requires sample_ra_deg/sample_dec_deg or sample_x_pix/sample_y_pix"
                )
        if self.target_image is not None:
            if self.target_image.ndim != 2:
                raise ValueError("target_image must be 2D")
            if tuple(self.target_image.shape) != tuple(self.observation.image.shape):
                raise ValueError("target_image shape must match observation.image")
        if (self.override_wavelengths is None) != (
            self.override_bandpass_weights is None
        ):
            raise ValueError(
                "override_wavelengths and override_bandpass_weights must be provided together"
            )
        if self.override_wavelengths is not None:
            if (
                self.override_wavelengths.ndim != 1
                or self.override_bandpass_weights is None
            ):
                raise ValueError("override_wavelengths must be 1D")
            if self.override_bandpass_weights.ndim != 1:
                raise ValueError("override_bandpass_weights must be 1D")
            if tuple(self.override_wavelengths.shape) != tuple(
                self.override_bandpass_weights.shape
            ):
                raise ValueError(
                    "override_wavelengths and override_bandpass_weights must match in shape"
                )


@dataclass(frozen=True)
class FieldCentroidAnchorTerm:
    """Astrometric anchor on the rendered centroid of selected decomposition roles."""

    observation: FieldObservation
    roles: str | tuple[str, ...] = "shared"
    target_x_pix: float | None = None
    target_y_pix: float | None = None
    target_ra_deg: float | None = None
    target_dec_deg: float | None = None
    forward_context: ForwardContext | None = None
    loss_weight: float = 1.0
    window_sigma_pix: float = 1.0

    def __post_init__(self) -> None:
        if self.loss_weight < 0.0:
            raise ValueError("loss_weight must be non-negative")
        pixel_pair = self.target_x_pix is not None or self.target_y_pix is not None
        world_pair = self.target_ra_deg is not None or self.target_dec_deg is not None
        if pixel_pair and world_pair:
            raise ValueError("pixel and world centroid targets are mutually exclusive")
        if not pixel_pair and not world_pair:
            raise ValueError(
                "one of (target_x_pix, target_y_pix) or (target_ra_deg, target_dec_deg) must be provided"
            )
        if pixel_pair and (self.target_x_pix is None or self.target_y_pix is None):
            raise ValueError("target_x_pix and target_y_pix must be provided together")
        if world_pair and (self.target_ra_deg is None or self.target_dec_deg is None):
            raise ValueError(
                "target_ra_deg and target_dec_deg must be provided together"
            )
        if (
            not math.isfinite(float(self.window_sigma_pix))
            or float(self.window_sigma_pix) <= 0.0
        ):
            raise ValueError("window_sigma_pix must be finite and positive")


@dataclass
class FieldFitResult:
    model: nn.Module
    loss_history: tuple[float, ...]
    final_loss: float
    diagnostics: dict[str, Any] | None = None
    exposure_poses: tuple[ExposurePose | None, ...] | None = None
    operator_bundle: ObservationOperatorBundle | None = None
    observation_operator_bundles: tuple[ObservationOperatorBundle, ...] | None = None
    optimizer_state: dict[str, Any] | None = None
    amp_scaler_state: dict[str, Any] | None = None
    sampling_rng_state: Tensor | None = None
    completed_steps: int = 0
    _pre_train_loss: float | None = None

    @property
    def initial_loss(self) -> float:
        """Loss on full observations before any training (when available),
        otherwise ``loss_history[0]`` (loss after the first optimizer step)."""
        if self._pre_train_loss is not None:
            return self._pre_train_loss
        return self.loss_history[0]

    def training_loss_after_step(self, step: int) -> float:
        """Training loss after optimizer step ``step`` (1-based, matches ``loss_history`` index + 1)."""
        if step < 1:
            raise ValueError("step must be >= 1")
        if step > len(self.loss_history):
            raise IndexError(
                f"step {step} out of range for loss_history length {len(self.loss_history)}"
            )
        return self.loss_history[step - 1]
