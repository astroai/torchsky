"""Radiometric calibration helpers shared across mapmaking paths."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ...spectrophotometry import Passband
from .base import RadiometryOp


@dataclass
class ExplicitRadiometryOp(RadiometryOp):
    def __init__(self, *, name: str = "explicit_radiometry") -> None:
        super().__init__(name=name, learned=False)


def apply_image_observation_calibration(
    *,
    photometric_scale: float,
    background_offset: float,
    image: Tensor,
    source_weight: Tensor,
) -> tuple[Tensor, Tensor]:
    scale = float(photometric_scale)
    background = float(background_offset)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Observation photometric_scale must be finite and positive")
    if not math.isfinite(background):
        raise ValueError("Observation background_offset must be finite")
    dtype = torch.promote_types(image.dtype, source_weight.dtype)
    calibrated_image = scale * (
        torch.nan_to_num(image.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        - background
    )
    calibrated_weight = source_weight.to(dtype=dtype) / (scale * scale)
    return calibrated_image, calibrated_weight


def expected_electrons_from_photon_spectral_density(
    photon_spectral_density: Tensor,
    wavelength_angstrom: Tensor,
    throughput: Tensor | Passband,
    *,
    collecting_area_m2: Tensor | float,
    exposure_time_s: Tensor | float,
    spatial_measure: Tensor | float = 1.0,
    spectral_dim: int = -1,
) -> Tensor:
    """Integrate a photon spectral density into expected detector electrons.

    ``photon_spectral_density`` is in
    ``photons s^-1 m^-2 Angstrom^-1`` per spatial element. For a flux density,
    leave ``spatial_measure=1``. For a radiance, supply the detector-pixel solid
    angle in steradians. ``throughput`` is dimensionless total transmission/QE;
    it may be a trailing-axis tensor (including a spatially varying response) or
    a canonical :class:`~torchsky.spectrophotometry.Passband`.

    The output is expected electrons. The spectral integral uses the explicit
    non-uniform Angstrom grid and remains differentiable in all floating inputs;
    support/search decisions for a tabulated passband are piecewise fixed.
    """
    density = torch.as_tensor(photon_spectral_density)
    if not density.is_floating_point() or density.ndim < 1:
        raise TypeError("photon_spectral_density must be a floating tensor")
    dim = int(spectral_dim)
    if dim < 0:
        dim += density.ndim
    if dim < 0 or dim >= density.ndim:
        raise ValueError("spectral_dim is out of range")
    density = density.movedim(dim, -1)
    wavelength = torch.as_tensor(
        wavelength_angstrom,
        device=density.device,
        dtype=density.dtype,
    )
    if wavelength.ndim != 1 or wavelength.numel() != density.shape[-1]:
        raise ValueError(
            "wavelength_angstrom must be 1D and match the spectral dimension"
        )
    if not bool(torch.isfinite(wavelength).all()) or bool((wavelength <= 0.0).any()):
        raise ValueError("wavelength_angstrom must be finite and positive")
    if not bool((wavelength[1:] > wavelength[:-1]).all()):
        raise ValueError("wavelength_angstrom must be strictly increasing")
    if not bool(torch.isfinite(density).all()) or bool((density < 0.0).any()):
        raise ValueError("photon_spectral_density must be finite and non-negative")

    response = (
        throughput.evaluate(wavelength)
        if isinstance(throughput, Passband)
        else torch.as_tensor(throughput, device=density.device, dtype=density.dtype)
    )
    if not bool(torch.isfinite(response).all()) or bool((response < 0.0).any()):
        raise ValueError("throughput must be finite and non-negative")
    try:
        torch.broadcast_shapes(tuple(density.shape), tuple(response.shape))
    except RuntimeError as exc:
        raise ValueError(
            "throughput must broadcast against photon density with wavelength last"
        ) from exc

    spectral_rate = torch.trapezoid(
        density * response,
        wavelength,
        dim=-1,
    )
    area = torch.as_tensor(
        collecting_area_m2, device=density.device, dtype=density.dtype
    )
    duration = torch.as_tensor(
        exposure_time_s, device=density.device, dtype=density.dtype
    )
    measure = torch.as_tensor(
        spatial_measure, device=density.device, dtype=density.dtype
    )
    for value, name, strictly_positive in (
        (area, "collecting_area_m2", True),
        (duration, "exposure_time_s", True),
        (measure, "spatial_measure", False),
    ):
        invalid_sign = value <= 0.0 if strictly_positive else value < 0.0
        if not bool(torch.isfinite(value).all()) or bool(invalid_sign.any()):
            qualifier = "positive" if strictly_positive else "non-negative"
            raise ValueError(f"{name} must be finite and {qualifier}")
    try:
        return spectral_rate * area * duration * measure
    except RuntimeError as exc:
        raise ValueError(
            "area, exposure time, and spatial measure must broadcast to the output"
        ) from exc
