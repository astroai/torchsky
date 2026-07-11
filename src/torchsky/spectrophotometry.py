"""Unit-explicit passbands and deterministic synthetic photometry.

The public contract uses wavelength in Angstrom internally.  A passband is a
dimensionless throughput curve, and spectra are sampled ``F_lambda`` values.
Photon-counting integrations use the detector response ``throughput * lambda``
weight.  No optional astronomy package is required, so applications can add
their own FITS/table adapters without making this numerical surface depend on
file formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

_C_ANGSTROM_PER_SECOND = 2.99792458e18
_AB_ZERO_UJY = 3.631e9

WavelengthUnit = Literal["angstrom", "nm"]
PhotometryOutput = Literal["mean_f_lambda", "f_nu_microjy", "ab_mag"]


@dataclass(frozen=True)
class Passband:
    """Validated dimensionless throughput curve for one astronomical band."""

    wavelength: torch.Tensor
    throughput: torch.Tensor
    name: str = ""
    wavelength_unit: WavelengthUnit = "angstrom"

    def __post_init__(self) -> None:
        wavelength = torch.as_tensor(self.wavelength, dtype=torch.float64).reshape(-1)
        throughput = torch.as_tensor(self.throughput, dtype=torch.float64).reshape(-1)
        if self.wavelength_unit not in ("angstrom", "nm"):
            raise ValueError("wavelength_unit must be 'angstrom' or 'nm'")
        if wavelength.numel() < 2 or wavelength.shape != throughput.shape:
            raise ValueError(
                "wavelength and throughput must be matching vectors with at least two samples"
            )
        if not torch.all(torch.isfinite(wavelength)) or not torch.all(
            torch.isfinite(throughput)
        ):
            raise ValueError("wavelength and throughput must be finite")
        if not torch.all(wavelength[1:] > wavelength[:-1]):
            raise ValueError("passband wavelengths must be strictly increasing")
        if torch.any(throughput < 0) or not torch.any(throughput > 0):
            raise ValueError("throughput must be non-negative with a positive response")
        if self.wavelength_unit == "nm":
            wavelength = wavelength * 10.0
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "throughput", throughput)
        object.__setattr__(self, "wavelength_unit", "angstrom")

    @classmethod
    def from_samples(
        cls,
        wavelength: torch.Tensor,
        throughput: torch.Tensor,
        *,
        name: str = "",
        wavelength_unit: WavelengthUnit = "angstrom",
    ) -> Passband:
        """Canonicalize an unsorted curve, averaging duplicate wavelengths."""
        values = torch.as_tensor(wavelength, dtype=torch.float64).reshape(-1)
        response = torch.as_tensor(throughput, dtype=torch.float64).reshape(-1)
        if values.shape != response.shape:
            raise ValueError("wavelength and throughput must have matching shapes")
        order = torch.argsort(values)
        values, response = values[order], response[order]
        unique, inverse = torch.unique_consecutive(values, return_inverse=True)
        sums = torch.zeros_like(unique).scatter_add_(0, inverse, response)
        counts = torch.zeros_like(unique).scatter_add_(
            0, inverse, torch.ones_like(response)
        )
        return cls(unique, sums / counts, name=name, wavelength_unit=wavelength_unit)

    @property
    def effective_wavelength(self) -> torch.Tensor:
        """Photon-weighted effective wavelength in Angstrom."""
        weight = self.throughput * self.wavelength
        return torch.trapezoid(
            self.wavelength * weight, self.wavelength
        ) / torch.trapezoid(weight, self.wavelength)

    @property
    def pivot_wavelength(self) -> torch.Tensor:
        """Pivot wavelength in Angstrom for converting mean ``F_lambda`` and ``F_nu``."""
        numerator = torch.trapezoid(self.throughput * self.wavelength, self.wavelength)
        denominator = torch.trapezoid(
            self.throughput / self.wavelength, self.wavelength
        )
        return torch.sqrt(numerator / denominator)


@dataclass(frozen=True)
class PhotometryResult:
    """Synthetic photometry values and the coverage diagnostics that qualify them."""

    value: torch.Tensor
    unit: str
    coverage: torch.Tensor
    valid: torch.Tensor


def synthetic_photometry(
    wavelength: torch.Tensor,
    flux_lambda: torch.Tensor,
    passband: Passband,
    *,
    output: PhotometryOutput = "mean_f_lambda",
    min_coverage: float = 0.99,
) -> PhotometryResult:
    """Integrate one spectrum or a batch through a photon-counting passband.

    ``flux_lambda`` is per Angstrom in arbitrary units for ``mean_f_lambda``.
    For ``f_nu_microjy`` and ``ab_mag`` it must be cgs ``F_lambda`` in
    ``erg s^-1 cm^-2 Angstrom^-1``.  Missing or insufficiently covered bands
    return ``NaN`` values and ``valid=False`` rather than silently extrapolating.
    """
    if output not in ("mean_f_lambda", "f_nu_microjy", "ab_mag"):
        raise ValueError("output must be 'mean_f_lambda', 'f_nu_microjy', or 'ab_mag'")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between zero and one")
    wave = torch.as_tensor(wavelength, dtype=torch.float64)
    flux = torch.as_tensor(flux_lambda, dtype=torch.float64, device=wave.device)
    if wave.ndim != 1 or flux.ndim < 1 or flux.shape[-1] != wave.numel():
        raise ValueError(
            "wavelength must be 1D and flux_lambda must end with its length"
        )
    if (
        wave.numel() < 2
        or not torch.all(torch.isfinite(wave))
        or not torch.all(wave[1:] > wave[:-1])
    ):
        raise ValueError("spectrum wavelength must be finite and strictly increasing")
    if not torch.all(torch.isfinite(flux)):
        raise ValueError(
            "flux_lambda must be finite; represent missing bands by a missing spectrum"
        )

    wave = _to_angstrom(wave)
    batch = flux.reshape(-1, wave.numel())
    grid, response, coverage = _covered_grid(wave, passband)
    result_shape = flux.shape[:-1]
    values = torch.full(
        (batch.shape[0],), torch.nan, dtype=flux.dtype, device=flux.device
    )
    valid = (coverage >= min_coverage) & (grid.numel() >= 2)
    if bool(torch.any(valid)):
        sampled_flux = _linear_interpolate(wave, batch, grid)
        photon_weight = response * grid
        numerator = torch.trapezoid(sampled_flux * photon_weight, grid, dim=-1)
        if output == "mean_f_lambda":
            denominator = torch.trapezoid(photon_weight, grid)
            values = torch.where(valid, numerator / denominator, values)
            unit = "F_lambda"
        else:
            # For a photon-counting detector, F_nu = integral(F_lambda*T*lambda)
            # / [c * integral(T/lambda)].  This is exact for constant F_nu.
            denominator = _passband_fnu_denominator(
                passband, device=wave.device, dtype=wave.dtype
            )
            values = torch.where(valid, numerator / denominator * 1.0e29, values)
            if output == "ab_mag":
                values = torch.where(
                    values > 0, -2.5 * torch.log10(values / _AB_ZERO_UJY), torch.nan
                )
                unit = "AB mag"
            else:
                unit = "microJy"
    else:
        unit = (
            "F_lambda"
            if output == "mean_f_lambda"
            else ("AB mag" if output == "ab_mag" else "microJy")
        )
    return PhotometryResult(
        value=values.reshape(result_shape),
        unit=unit,
        coverage=coverage.expand(result_shape),
        valid=valid.expand(result_shape),
    )


def flux_to_magnitude(
    flux: torch.Tensor, zero_point: float | torch.Tensor
) -> torch.Tensor:
    """Convert positive fluxes to magnitudes for an explicit flux zero point."""
    values = torch.as_tensor(flux)
    zero = torch.as_tensor(zero_point, dtype=values.dtype, device=values.device)
    if torch.any(zero <= 0) or not torch.all(torch.isfinite(zero)):
        raise ValueError("zero_point must be finite and positive")
    return torch.where(values > 0, -2.5 * torch.log10(values / zero), torch.nan)


def magnitude_to_flux(
    magnitude: torch.Tensor, zero_point: float | torch.Tensor
) -> torch.Tensor:
    """Convert magnitudes to flux using an explicit zero point."""
    values = torch.as_tensor(magnitude)
    zero = torch.as_tensor(zero_point, dtype=values.dtype, device=values.device)
    if torch.any(zero <= 0) or not torch.all(torch.isfinite(zero)):
        raise ValueError("zero_point must be finite and positive")
    return zero * torch.pow(10.0, -0.4 * values)


def _to_angstrom(wavelength: torch.Tensor) -> torch.Tensor:
    """Validate the Angstrom-native spectral grid."""
    if wavelength.numel() and wavelength[-1] <= 0:
        raise ValueError("spectrum wavelengths must be positive")
    return wavelength


def _covered_grid(
    wave: torch.Tensor, passband: Passband
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pb_wave = passband.wavelength.to(device=wave.device, dtype=wave.dtype)
    pb_response = passband.throughput.to(device=wave.device, dtype=wave.dtype)
    full_weight = torch.trapezoid(pb_response * pb_wave, pb_wave)
    lo, hi = wave[0], wave[-1]
    inside = (pb_wave >= lo) & (pb_wave <= hi)
    if hi <= pb_wave[0] or lo >= pb_wave[-1]:
        return (
            pb_wave[:0],
            pb_response[:0],
            torch.zeros((), dtype=wave.dtype, device=wave.device),
        )
    grid = pb_wave[inside]
    left = torch.maximum(lo, pb_wave[0])
    right = torch.minimum(hi, pb_wave[-1])
    grid = torch.cat((grid, left[None], right[None]))
    grid = torch.unique(torch.sort(grid).values)
    response = _linear_interpolate(pb_wave, pb_response.unsqueeze(0), grid).squeeze(0)
    covered_weight = torch.trapezoid(response * grid, grid)
    coverage = covered_weight / full_weight
    return grid, response, coverage


def _passband_fnu_denominator(
    passband: Passband, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    wave = passband.wavelength.to(device=device, dtype=dtype)
    throughput = passband.throughput.to(device=device, dtype=dtype)
    return _C_ANGSTROM_PER_SECOND * torch.trapezoid(throughput / wave, wave)


def _linear_interpolate(
    x: torch.Tensor, y: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    index = torch.searchsorted(x, query).clamp(1, x.numel() - 1)
    x0, x1 = x[index - 1], x[index]
    fraction = (query - x0) / (x1 - x0)
    if y.ndim == 1:
        return y[index - 1] + fraction * (y[index] - y[index - 1])
    index0 = (index - 1).expand(y.shape[0], -1)
    index1 = index.expand(y.shape[0], -1)
    y0, y1 = torch.gather(y, 1, index0), torch.gather(y, 1, index1)
    return y0 + fraction.unsqueeze(0) * (y1 - y0)
