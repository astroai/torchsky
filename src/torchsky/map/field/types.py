"""Shared types and dataclasses for continuous-field (INR) models.

Implementation of ``nn.Module`` subclasses and training live in :mod:`torchsky.map.field.core`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch.nn as nn
from torch import Tensor

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
FieldLossReduction = Literal[
    "per_observation", "per_band", "global", "band_global_mix"
]
FieldCurriculum = Literal["none", "spectral_frequency_ramp", "feature_scale_ramp"]
FieldSpectralSampling = Literal[
    "deterministic_quadrature",
    "bandpass_mc",
    "mixture_mc",
]
FieldObservationSampling = Literal["uniform", "band_balanced"]
FieldNoisePretrainKind = Literal["none", "uniform", "gaussian", "spectrum"]


@dataclass(frozen=True)
class FieldDomain:
    center_ra_deg: float
    center_dec_deg: float
    spatial_scale_deg: float
    wavelength_center: float
    wavelength_scale: float
    coord_mode: FieldCoordinateMode = "local_tangent"


@dataclass
class FieldObservation:
    image: Tensor
    header: dict
    weight: Tensor | None = None
    mask: Tensor | None = None
    wavelength: float | None = None
    spectral_min: float | None = None
    spectral_max: float | None = None
    transmission_center: float | None = None
    transmission_sigma: float | None = None
    psf_sigma_pix: float = 0.0
    band_name: str | None = None


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
