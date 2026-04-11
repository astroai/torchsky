"""Shared types and dataclasses for continuous-field (INR) models.

Implementation of ``nn.Module`` subclasses and training live in :mod:`torchsky.map.field.core`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch.nn as nn
from torch import Tensor

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
    exposure_id: str | None = None
    time_mjd: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


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
    band_name: str | None = None
    spectral_quantity: FieldSpectralQuantity | None = None

    def __post_init__(self) -> None:
        wl = self.transmission_wavelength
        tt = self.transmission_throughput
        if (wl is None) ^ (tt is None):
            raise ValueError(
                "transmission_wavelength and transmission_throughput must both be set or both None"
            )
        if wl is None:
            return
        if wl.ndim != 1 or tt.ndim != 1:
            raise ValueError("transmission_wavelength and transmission_throughput must be 1D")
        if wl.shape != tt.shape:
            raise ValueError("transmission_wavelength and transmission_throughput shapes must match")
        if wl.numel() < 2:
            raise ValueError("tabulated transmission requires at least two samples")


@dataclass
class FieldFitResult:
    model: nn.Module
    loss_history: tuple[float, ...]
    final_loss: float

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
