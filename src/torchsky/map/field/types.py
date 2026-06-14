"""Shared types and dataclasses for continuous-field (INR) models.

Implementation of ``nn.Module`` subclasses and training live in :mod:`torchsky.map.field.core`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import math
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from ..exposure_pose import ExposurePose

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
FieldLossKind = Literal["mse", "charbonnier", "huber", "gaussian_nll"]
FieldLRSchedule = Literal["constant", "cosine"]
FieldLossReduction = Literal["per_observation", "per_band", "global", "band_global_mix"]
FieldCurriculum = Literal["none", "spectral_frequency_ramp", "feature_scale_ramp"]
FieldSpectralSampling = Literal[
    "deterministic_quadrature",
    "bandpass_mc",
    "mixture_mc",
]
FieldObservationSampling = Literal["uniform", "band_balanced"]
FieldNoisePretrainKind = Literal["none", "uniform", "gaussian", "spectrum"]
# wavelength | energy | frequency — meaning of spectral_min / spectral_max numeric values
FieldSpectralQuantity = Literal["wavelength", "energy", "frequency"]


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


FieldPSFMode = Literal["none", "gaussian_sigma_pix", "kernel2d"]


@dataclass
class PSFSpec:
    """Public PSF description for field observations and target products."""

    mode: FieldPSFMode = "none"
    sigma_pix: float = 0.0
    kernel2d: Tensor | None = None

    @classmethod
    def gaussian(cls, sigma_pix: float) -> "PSFSpec":
        return cls(mode="gaussian_sigma_pix", sigma_pix=float(sigma_pix))

    @classmethod
    def kernel(cls, kernel2d: Tensor) -> "PSFSpec":
        return cls(mode="kernel2d", kernel2d=kernel2d)

    def __post_init__(self) -> None:
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
            return
        raise ValueError(f"unsupported PSFSpec mode {self.mode!r}")


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
        for name in (
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

        wl = self.transmission_wavelength
        tt = self.transmission_throughput
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

    @property
    def initial_loss(self) -> float:
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
