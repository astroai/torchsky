"""Unit-explicit passbands and deterministic synthetic photometry.

The public contract uses wavelength in Angstrom internally.  A passband is a
dimensionless throughput curve, and spectra are sampled ``F_lambda`` values.
Photon-counting integrations use the detector response ``throughput * lambda``
weight.  No optional astronomy package is required, so applications can add
their own FITS/table adapters without making this numerical surface depend on
file formats.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch

_C_ANGSTROM_PER_SECOND = 2.99792458e18
_AB_ZERO_UJY = 3.631e9

#: Devices whose framework implements ``float64``. MPS does not: asking for it
#: raises instead of rounding, so every tensor this module holds has to ask.
_FLOAT64_DEVICES = frozenset({"cpu", "cuda"})


def _working_dtype(tensor: torch.Tensor) -> torch.dtype:
    """The precision a passband or spectrum is held in on its own device.

    Double precision is the default -- narrow passbands need it and passband
    samples are calibration metadata -- but a device without ``float64`` gets the
    tensor's own floating dtype, which is the same integral at the precision the
    caller chose rather than an exception.
    """

    if tensor.device.type in _FLOAT64_DEVICES:
        return torch.float64
    return tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()


WavelengthUnit = Literal["angstrom", "nm"]
PhotometryOutput = Literal["mean_f_lambda", "f_nu_microjy", "ab_mag"]
Weighting = Literal["photon", "energy"]


def _check_weighting(weighting: str) -> None:
    if weighting not in ("photon", "energy"):
        raise ValueError("weighting must be 'photon' or 'energy'")


def _band_weights(
    response: torch.Tensor, grid: torch.Tensor, weighting: Weighting
) -> tuple[torch.Tensor, torch.Tensor]:
    """``F_lambda`` and ``F_nu`` weights (without ``c``) of a response on a grid.

    Photon counting weights ``F_lambda`` by ``R lambda`` and ``F_nu`` by
    ``R / lambda``; energy counting by ``R`` and ``R / lambda^2`` (Bessell &
    Murphy 2012, appendix A).
    """

    if weighting == "photon":
        return response * grid, response / grid
    return response, response / grid.square()


@dataclass(frozen=True)
class Passband:
    """Validated dimensionless throughput curve for one astronomical band.

    Args:
        wavelength: Strictly increasing one-dimensional wavelength samples.
        throughput: Dimensionless nonnegative response at each wavelength.
        name: Optional band identifier.
        wavelength_unit: ``"angstrom"`` or ``"nm"``.

    Examples:
        >>> import torch
        >>> from torchsky.spectrophotometry import Passband
        >>> band = Passband(torch.tensor([4000.0, 5000.0]), torch.tensor([0.0, 1.0]))
        >>> band.wavelength_unit
        'angstrom'
    """

    wavelength: torch.Tensor
    throughput: torch.Tensor
    name: str = ""
    wavelength_unit: WavelengthUnit = "angstrom"

    def __post_init__(self) -> None:
        wavelength = torch.as_tensor(self.wavelength).reshape(-1)
        throughput = torch.as_tensor(self.throughput).reshape(-1)
        dtype = _working_dtype(wavelength)
        wavelength = wavelength.to(dtype=dtype)
        throughput = throughput.to(dtype=dtype)
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
        values = torch.as_tensor(wavelength).reshape(-1)
        response = torch.as_tensor(throughput).reshape(-1)
        dtype = _working_dtype(values)
        values = values.to(dtype=dtype)
        response = response.to(dtype=dtype)
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

    def evaluate(self, wavelength: torch.Tensor) -> torch.Tensor:
        """Linear throughput at Angstrom-native ``wavelength``; zero outside support."""
        query = torch.as_tensor(wavelength)
        grid = self.wavelength.to(device=query.device, dtype=query.dtype)
        trans = self.throughput.to(device=query.device, dtype=query.dtype)
        idx = torch.searchsorted(grid, query)
        last = grid.numel() - 1
        idx0 = (idx - 1).clamp(0, last)
        idx1 = idx.clamp(0, last)
        x0, x1 = grid[idx0], grid[idx1]
        t0, t1 = trans[idx0], trans[idx1]
        denom = (x1 - x0).clamp_min(1e-12)
        frac = (query - x0) / denom
        in_range = (frac >= 0.0) & (frac <= 1.0)
        frac = torch.where(in_range, frac, frac.clamp(0.0, 1.0))
        value = t0 + frac * (t1 - t0)
        outside = (query < grid[0]) | (query > grid[-1])
        return torch.where(outside, torch.zeros_like(value), value)

    def to(self, device: torch.device | str) -> Passband:
        """Return the canonical passband on ``device``.

        Passband samples remain float64 calibration metadata on devices that
        support it. Evaluation casts them to the query dtype, so moving a scene
        or observation never silently lowers the precision of the stored
        response curve. A device without ``float64`` (MPS) holds the curve in
        ``float32``, the same rule ``Passband`` applies to tensors created there.
        """
        target = torch.device(device)
        dtype = (
            self.wavelength.dtype if target.type in _FLOAT64_DEVICES else torch.float32
        )
        return Passband(
            self.wavelength.to(device=target, dtype=dtype),
            self.throughput.to(device=target, dtype=dtype),
            name=self.name,
            wavelength_unit="angstrom",
        )


@dataclass(frozen=True)
class PhotometryResult:
    """Synthetic photometry values and the coverage diagnostics that qualify them."""

    value: torch.Tensor
    unit: str
    coverage: torch.Tensor
    valid: torch.Tensor


def evaluate_throughput(
    wavelength: torch.Tensor,
    *,
    passband: Passband | None = None,
    center: torch.Tensor | float | None = None,
    sigma: float | None = None,
    lambda_center_map: torch.Tensor | None = None,
    wavelength_unit: WavelengthUnit = "angstrom",
) -> torch.Tensor:
    """Evaluate tabulated or Gaussian throughput at ``wavelength``.

    Gaussian / LVF paths use the caller's wavelength unit. Tabulated
    :class:`Passband` queries are converted to Angstrom. ``lambda_center_map``
    makes the Gaussian centre position-dependent (linear-variable filters).
    """
    wave = torch.as_tensor(wavelength)
    if passband is not None:
        query = wave * 10.0 if wavelength_unit == "nm" else wave
        return passband.evaluate(query)
    if sigma is None or float(sigma) <= 0:
        raise ValueError("gaussian throughput requires positive sigma")
    if lambda_center_map is not None:
        center_t = torch.as_tensor(
            lambda_center_map, device=wave.device, dtype=wave.dtype
        )
        if center_t.ndim == wave.ndim - 1:
            center_t = center_t.unsqueeze(-1)
    elif center is not None:
        center_t = torch.as_tensor(center, device=wave.device, dtype=wave.dtype)
    else:
        raise ValueError("gaussian throughput needs center or lambda_center_map")
    return torch.exp(-0.5 * torch.square((wave - center_t) / float(sigma)))


def _first(spec: Mapping[str, Any], *keys: str) -> Any:
    """First present, non-``None`` value for ``keys`` (no truthiness testing)."""
    for key in keys:
        value = spec.get(key)
        if value is not None:
            return value
    return None


def throughput_from_mapping(
    band_id: str, spec: Mapping[str, Any]
) -> Passband | dict[str, Any]:
    """Build a :class:`Passband` or Gaussian/LVF dict from a filter-profile mapping."""
    wl = _first(spec, "transmission_wavelength_nm", "transmission_wavelength")
    tr = _first(spec, "transmission_value", "transmission")
    lvf = bool(_first(spec, "lvf", "position_dependent"))
    if wl is not None and tr is not None and not isinstance(tr, (int, float)):
        unit: WavelengthUnit = (
            "nm" if spec.get("transmission_wavelength_nm") is not None else "angstrom"
        )
        if lvf:
            raise ValueError("LVF bands must use a gaussian parameterisation")
        return Passband.from_samples(
            torch.as_tensor(wl),
            torch.as_tensor(tr),
            name=band_id,
            wavelength_unit=unit,
        )
    center = _first(spec, "transmission_center", "lambda_eff_nm", "wavelength")
    sigma = _first(spec, "transmission_sigma", "sigma_nm")
    if sigma is None and spec.get("fwhm_nm") is not None:
        sigma = float(spec["fwhm_nm"]) / 2.354820045
    if center is None and spec.get("wavelength_range") is not None:
        lo, hi = spec["wavelength_range"]
        center = 0.5 * (float(lo) + float(hi))
        if sigma is None:
            sigma = (float(hi) - float(lo)) / 2.354820045
    if center is None:
        raise ValueError(f"Cannot build throughput for '{band_id}'")
    if sigma is None:
        lo = spec.get("spectral_min")
        hi = spec.get("spectral_max")
        sigma = (
            (float(hi) - float(lo)) / 2.354820045
            if lo is not None and hi is not None
            else 0.1 * float(center)
        )
    return {
        "band_id": band_id,
        "kind": "gaussian",
        "center": float(center),
        "sigma": float(sigma),
        "position_dependent": lvf,
    }


def _photometry_unit(output: PhotometryOutput) -> str:
    if output == "mean_f_lambda":
        return "F_lambda"
    if output == "ab_mag":
        return "AB mag"
    return "microJy"


def _integrate_one_grid(
    wave: torch.Tensor,
    batch: torch.Tensor,
    passband: Passband,
    *,
    output: PhotometryOutput,
    min_coverage: float,
    weighting: Weighting = "photon",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    grid, response, coverage = _covered_grid(wave, passband, weighting=weighting)
    values = torch.full(
        (batch.shape[0],), torch.nan, dtype=batch.dtype, device=batch.device
    )
    valid = (coverage >= min_coverage) & (grid.numel() >= 2)
    unit = _photometry_unit(output)
    if bool(torch.any(valid)):
        sampled_flux = _linear_interpolate(wave, batch, grid)
        lambda_weight, nu_weight = _band_weights(response, grid, weighting)
        numerator = torch.trapezoid(sampled_flux * lambda_weight, grid, dim=-1)
        if output == "mean_f_lambda":
            denominator = torch.trapezoid(lambda_weight, grid)
            values = torch.where(valid, numerator / denominator, values)
        else:
            # Same quadrature as the numerator so a constant F_ν is recovered
            # after the spectrum samples are unioned into the passband grid.
            denominator = _C_ANGSTROM_PER_SECOND * torch.trapezoid(nu_weight, grid)
            values = torch.where(valid, numerator / denominator * 1.0e29, values)
            if output == "ab_mag":
                values = torch.where(
                    values > 0, -2.5 * torch.log10(values / _AB_ZERO_UJY), torch.nan
                )
    return values, coverage, valid, unit


def synthetic_photometry(
    wavelength: torch.Tensor,
    flux_lambda: torch.Tensor,
    passband: Passband,
    *,
    output: PhotometryOutput = "mean_f_lambda",
    min_coverage: float = 0.99,
    mask: torch.Tensor | None = None,
    weighting: Weighting = "photon",
) -> PhotometryResult:
    """Integrate one spectrum or a batch through a passband.

    ``wavelength`` may be 1D (shared grid) or match ``flux_lambda`` including a
    trailing spectral axis. ``flux_lambda`` is per Angstrom in arbitrary units
    for ``mean_f_lambda``. For ``f_nu_microjy`` and ``ab_mag`` it must be cgs
    ``F_lambda`` in ``erg s^-1 cm^-2 Angstrom^-1``. Missing or insufficiently
    covered bands return ``NaN`` values and ``valid=False``.

    ``mask``, when given, is True for samples that exist. Masked wavelengths
    are dropped from the integral and do not count toward filter coverage.

    ``weighting="photon"`` (default) is a photon-counting detector, mean
    ``F_lambda = int F R lambda dlambda / int R lambda dlambda``;
    ``"energy"`` is an energy-counting one (bolometers, some published
    curves), ``int F R dlambda / int R dlambda``. Coverage is measured with the
    same weight.

    Args:
        wavelength: Shared or per-spectrum wavelength samples.
        flux_lambda: Flux density per Angstrom on the trailing spectral axis.
        passband: Photon-counting throughput curve.
        output: Requested mean-flux, microjansky, or AB-magnitude representation.
        min_coverage: Minimum passband response fraction required for validity.
        mask: Optional valid-sample mask.
        weighting: ``"photon"`` or ``"energy"`` counting response.

    Returns:
        Photometry values, unit, passband coverage, and validity mask.

    Examples:
        >>> import torch
        >>> from torchsky.spectrophotometry import Passband, synthetic_photometry
        >>> band = Passband(torch.tensor([4000.0, 5000.0]), torch.ones(2))
        >>> synthetic_photometry(band.wavelength, torch.ones(2), band).value.item()
        1.0
    """
    if output not in ("mean_f_lambda", "f_nu_microjy", "ab_mag"):
        raise ValueError("output must be 'mean_f_lambda', 'f_nu_microjy', or 'ab_mag'")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between zero and one")
    _check_weighting(weighting)
    wave = torch.as_tensor(wavelength)
    flux = torch.as_tensor(flux_lambda, device=wave.device)
    dtype = _working_dtype(wave)
    wave = wave.to(dtype=dtype)
    flux = flux.to(dtype=dtype)
    if flux.ndim < 1:
        raise ValueError("flux_lambda must include a spectral axis")
    n_wave = flux.shape[-1]
    result_shape = flux.shape[:-1]
    if wave.ndim == 1:
        if wave.numel() != n_wave:
            raise ValueError("wavelength length must match flux_lambda's last axis")
        waves = wave.reshape(1, -1).expand(max(int(flux.numel() // n_wave), 1), -1)
        shared = flux.numel() > 0
    elif wave.shape == flux.shape:
        waves = wave.reshape(-1, n_wave)
        shared = waves.shape[0] > 0 and (
            waves.shape[0] == 1
            or (not waves.requires_grad and bool(torch.equal(waves, waves[:1])))
        )
    else:
        raise ValueError("wavelength must be 1D or the same shape as flux_lambda")
    batch = flux.reshape(-1, n_wave)
    mask_t: torch.Tensor | None = None
    if mask is not None:
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=wave.device)
        if mask_t.shape != flux.shape:
            raise ValueError("mask must have the same shape as flux_lambda")
        mask_t = mask_t.reshape(-1, n_wave)
        shared = shared and bool(mask_t.all())
    if (
        n_wave < 2
        or not torch.all(torch.isfinite(waves))
        or not torch.all(waves[:, 1:] > waves[:, :-1])
    ):
        raise ValueError("spectrum wavelength must be finite and strictly increasing")
    flux_ok = torch.isfinite(batch)
    if mask_t is not None:
        flux_ok = flux_ok | ~mask_t
    if not bool(torch.all(flux_ok)):
        raise ValueError(
            "flux_lambda must be finite; represent missing samples with mask=False"
        )

    if shared:
        values, coverage, valid, unit = _integrate_masked(
            _to_angstrom(waves[0]),
            batch,
            passband,
            output=output,
            min_coverage=min_coverage,
            mask=None if mask_t is None else mask_t[0],
            weighting=weighting,
        )
        cov_out = coverage.expand(result_shape)
        valid_out = valid.expand(result_shape)
    else:
        values = torch.empty(batch.shape[0], dtype=batch.dtype, device=batch.device)
        covs = torch.empty(batch.shape[0], dtype=batch.dtype, device=batch.device)
        valids = torch.empty(batch.shape[0], dtype=torch.bool, device=batch.device)
        unit = _photometry_unit(output)
        for i in range(batch.shape[0]):
            keep = mask_t[i] if mask_t is not None else None
            row_wave = waves[i]
            row_flux = batch[i : i + 1]
            if keep is not None and int(keep.sum()) < 2:
                values[i] = torch.nan
                covs[i] = 0.0
                valids[i] = False
                continue
            row, cov, ok, _ = _integrate_masked(
                _to_angstrom(row_wave),
                row_flux,
                passband,
                output=output,
                min_coverage=min_coverage,
                mask=keep,
                weighting=weighting,
            )
            values[i] = row[0]
            covs[i] = cov
            valids[i] = bool(ok)
        cov_out = covs.reshape(result_shape)
        valid_out = valids.reshape(result_shape)
    return PhotometryResult(
        value=values.reshape(result_shape),
        unit=unit,
        coverage=cov_out,
        valid=valid_out,
    )


def synthetic_photometry_bands(
    wavelength: torch.Tensor,
    flux_lambda: torch.Tensor,
    passbands: Sequence[Passband],
    *,
    output: PhotometryOutput = "mean_f_lambda",
    min_coverage: float = 0.99,
    mask: torch.Tensor | None = None,
    weighting: Weighting = "photon",
) -> PhotometryResult:
    """Integrate through many passbands; values have a trailing band axis."""
    if not passbands:
        empty = torch.as_tensor(flux_lambda)[..., :0]
        return PhotometryResult(
            value=empty,
            unit=_photometry_unit(output),
            coverage=torch.zeros_like(empty),
            valid=torch.zeros(empty.shape, dtype=torch.bool),
        )
    results = [
        synthetic_photometry(
            wavelength,
            flux_lambda,
            band,
            output=output,
            min_coverage=min_coverage,
            mask=mask,
            weighting=weighting,
        )
        for band in passbands
    ]
    return PhotometryResult(
        value=torch.stack([item.value for item in results], dim=-1),
        unit=results[0].unit,
        coverage=torch.stack([item.coverage for item in results], dim=-1),
        valid=torch.stack([item.valid for item in results], dim=-1),
    )


def flux_to_magnitude(
    flux: torch.Tensor, zero_point: float | torch.Tensor
) -> torch.Tensor:
    """Convert positive fluxes to magnitudes for an explicit flux zero point."""
    values = torch.as_tensor(flux)
    zero = torch.as_tensor(zero_point, dtype=values.dtype, device=values.device)
    if torch.any(zero <= 0) or not torch.all(torch.isfinite(zero)):
        raise ValueError("zero_point must be finite and positive")
    safe = torch.where(values > 0, values, torch.ones_like(values))
    return torch.where(values > 0, -2.5 * torch.log10(safe / zero), torch.nan)


def magnitude_to_flux(
    magnitude: torch.Tensor, zero_point: float | torch.Tensor
) -> torch.Tensor:
    """Convert magnitudes to flux using an explicit zero point."""
    values = torch.as_tensor(magnitude)
    zero = torch.as_tensor(zero_point, dtype=values.dtype, device=values.device)
    if torch.any(zero <= 0) or not torch.all(torch.isfinite(zero)):
        raise ValueError("zero_point must be finite and positive")
    return zero * torch.pow(10.0, -0.4 * values)


#: Packaged Vega spectrum: CALSPEC ``alpha_lyr_stis_012.fits`` (Bohlin, Hubeny
#: & Rauch 2020, AJ 160, 21; STScI CALSPEC, public domain), WAVELENGTH (vacuum
#: Angstrom, float64) and FLUX (erg s^-1 cm^-2 A^-1, float32) columns verbatim,
#: 900 A to 300 um. Source file sha256
#: 7201cd8566d53d384430c3d5fd2ae8d8122e9a7f765711748b49336f63237672.
_VEGA_SPECTRUM_SOURCE = "CALSPEC alpha_lyr_stis_012"
_VEGA_FILE = Path(__file__).parent / "_data" / "calspec_alpha_lyr_stis_012.npz"


@lru_cache(maxsize=1)
def _vega_arrays() -> tuple[np.ndarray, np.ndarray]:
    with np.load(_VEGA_FILE, allow_pickle=False) as data:
        wavelength = np.asarray(data["wavelength"], dtype=np.float64)
        flux = np.asarray(data["flux"], dtype=np.float64)
    wavelength.setflags(write=False)
    flux.setflags(write=False)
    return wavelength, flux


def vega_spectrum(
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The CALSPEC spectrum of Vega, packaged with torchsky.

    ``alpha_lyr_stis_012`` (Bohlin, Hubeny & Rauch 2020): HST/STIS data from
    1152 to 10200 A joined to a 9550 K model, normalised to
    ``3.47e-9 erg s^-1 cm^-2 A^-1`` at 5556 A. Public-domain STScI data,
    92 kB in the wheel.

    Args:
        device: Target device; float64 where the device supports it.

    Returns:
        Vacuum wavelength in Angstrom and ``F_lambda`` in cgs units.

    Examples:
        >>> from torchsky.spectrophotometry import vega_spectrum
        >>> wave, flux = vega_spectrum()
        >>> wave.shape == flux.shape, float(wave[0]) < 1000.0
        (True, True)
    """
    wavelength, flux = _vega_arrays()
    target = torch.device("cpu" if device is None else device)
    dtype = (
        torch.float64 if target.type in _FLOAT64_DEVICES else torch.get_default_dtype()
    )
    return (
        torch.tensor(wavelength, dtype=dtype, device=target),
        torch.tensor(flux, dtype=dtype, device=target),
    )


def vega_zero_point(
    passband: Passband,
    *,
    output: Literal["mean_f_lambda", "f_nu_microjy"] = "mean_f_lambda",
    weighting: Weighting = "photon",
) -> torch.Tensor:
    """Band-averaged flux of Vega, the zero point of a Vega magnitude system.

    ``m_Vega = -2.5 log10(<F> / vega_zero_point(band))`` with ``<F>`` from
    :func:`synthetic_photometry` in the same ``output`` and ``weighting``;
    pass it to :func:`flux_to_magnitude`. Vega has magnitude zero in every
    band by this definition (systems that assign Vega 0.03 mag add that
    offset). Uses :func:`vega_spectrum`.

    Args:
        passband: Band to integrate through.
        output: ``"mean_f_lambda"`` (cgs per Angstrom) or ``"f_nu_microjy"``.
        weighting: ``"photon"`` or ``"energy"`` counting response.

    Returns:
        Scalar zero-point flux.

    Raises:
        ValueError: If the passband is not covered by the Vega spectrum.

    Examples:
        >>> import torch
        >>> from torchsky.spectrophotometry import Passband, vega_zero_point
        >>> band = Passband(torch.tensor([5546.0, 5556.0, 5566.0]), torch.tensor([0.0, 1.0, 0.0]))
        >>> round(float(vega_zero_point(band)) * 1e9, 2)
        3.47
    """
    if output not in ("mean_f_lambda", "f_nu_microjy"):
        raise ValueError("output must be 'mean_f_lambda' or 'f_nu_microjy'")
    wave, flux = vega_spectrum(device=passband.wavelength.device)
    # The response is non-zero up to the samples bracketing its positive part.
    positive = torch.nonzero(passband.throughput > 0).flatten()
    last = passband.wavelength.numel() - 1
    low = passband.wavelength[max(int(positive[0]) - 1, 0)]
    high = passband.wavelength[min(int(positive[-1]) + 1, last)]
    if bool(low < wave[0]) or bool(high > wave[-1]):
        raise ValueError("passband extends beyond the Vega spectrum (900 A to 300 um)")
    return synthetic_photometry(
        wave, flux, passband, output=output, weighting=weighting, min_coverage=0.0
    ).value


def _boolean_run_slices(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Inclusive-exclusive ``[start, stop)`` runs of True in a 1-D mask."""
    if mask.ndim != 1:
        raise ValueError("mask runs are computed on a 1-D mask")
    padded = torch.cat(
        (
            mask.new_zeros((1,), dtype=torch.bool),
            mask,
            mask.new_zeros((1,), dtype=torch.bool),
        )
    )
    delta = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)
    starts = (delta == 1).nonzero(as_tuple=False).flatten()
    stops = (delta == -1).nonzero(as_tuple=False).flatten()
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _integrate_masked(
    wave: torch.Tensor,
    batch: torch.Tensor,
    passband: Passband,
    *,
    output: PhotometryOutput,
    min_coverage: float,
    mask: torch.Tensor | None,
    weighting: Weighting = "photon",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Integrate, splitting at mask holes so invalid samples are not interpolated."""
    if mask is None or bool(mask.all()):
        return _integrate_one_grid(
            wave,
            batch,
            passband,
            output=output,
            min_coverage=min_coverage,
            weighting=weighting,
        )
    unit = _photometry_unit(output)
    values = torch.full(
        (batch.shape[0],), torch.nan, dtype=batch.dtype, device=batch.device
    )
    zero = torch.zeros((), dtype=batch.dtype, device=batch.device)
    runs = [
        (start, stop) for start, stop in _boolean_run_slices(mask) if stop - start >= 2
    ]
    if not runs:
        return (
            values,
            zero,
            torch.zeros((), dtype=torch.bool, device=batch.device),
            unit,
        )

    numerator = torch.zeros(batch.shape[0], dtype=batch.dtype, device=batch.device)
    den_lambda = zero.clone()
    den_nu = zero.clone()
    covered = zero.clone()
    full_weight = _passband_weight(passband, wave, weighting)
    for start, stop in runs:
        grid, response, piece_coverage = _covered_grid(
            wave[start:stop], passband, full_weight, weighting=weighting
        )
        if grid.numel() < 2:
            continue
        sampled_flux = _linear_interpolate(wave[start:stop], batch[:, start:stop], grid)
        lambda_weight, nu_weight = _band_weights(response, grid, weighting)
        numerator = numerator + torch.trapezoid(
            sampled_flux * lambda_weight, grid, dim=-1
        )
        den_lambda = den_lambda + torch.trapezoid(lambda_weight, grid)
        den_nu = den_nu + _C_ANGSTROM_PER_SECOND * torch.trapezoid(nu_weight, grid)
        covered = covered + piece_coverage * full_weight
    coverage = covered / full_weight
    valid = (coverage >= min_coverage) & (den_lambda > 0)
    if output == "mean_f_lambda":
        values = torch.where(valid, numerator / den_lambda, values)
    else:
        values = torch.where(valid, numerator / den_nu * 1.0e29, values)
        if output == "ab_mag":
            values = torch.where(
                values > 0, -2.5 * torch.log10(values / _AB_ZERO_UJY), torch.nan
            )
    return values, coverage, valid, unit


def _to_angstrom(wavelength: torch.Tensor) -> torch.Tensor:
    """Validate the Angstrom-native spectral grid."""
    if wavelength.numel() and bool((wavelength <= 0).any()):
        raise ValueError("spectrum wavelengths must be positive")
    return wavelength


def _passband_weight(
    passband: Passband, wave: torch.Tensor, weighting: Weighting = "photon"
) -> torch.Tensor:
    """Full reference weight, ``int R lambda dlambda`` (photon) or ``int R dlambda``."""
    pb_wave = passband.wavelength.to(device=wave.device, dtype=wave.dtype)
    pb_response = passband.throughput.to(device=wave.device, dtype=wave.dtype)
    return torch.trapezoid(_band_weights(pb_response, pb_wave, weighting)[0], pb_wave)


def _covered_grid(
    wave: torch.Tensor,
    passband: Passband,
    full_weight: torch.Tensor | None = None,
    *,
    weighting: Weighting = "photon",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Union grid of the spectrum and passband support with linear responses."""
    pb_wave = passband.wavelength.to(device=wave.device, dtype=wave.dtype)
    pb_response = passband.throughput.to(device=wave.device, dtype=wave.dtype)
    if full_weight is None:
        full_weight = torch.trapezoid(
            _band_weights(pb_response, pb_wave, weighting)[0], pb_wave
        )
    left = torch.maximum(wave[0], pb_wave[0])
    right = torch.minimum(wave[-1], pb_wave[-1])
    if not bool(right > left):
        return (
            pb_wave[:0],
            pb_response[:0],
            torch.zeros((), dtype=wave.dtype, device=wave.device),
        )
    pb_inside = pb_wave[(pb_wave > left) & (pb_wave < right)]
    spectrum_inside = wave[(wave > left) & (wave < right)]
    # Sorted, not deduplicated: coincident spectrum and passband samples must
    # each keep their own gradient path into the integral.
    grid = torch.sort(
        torch.cat((left[None], pb_inside, spectrum_inside, right[None]))
    ).values
    response = _linear_interpolate(pb_wave, pb_response.unsqueeze(0), grid).squeeze(0)
    coverage = (
        torch.trapezoid(_band_weights(response, grid, weighting)[0], grid) / full_weight
    )
    return grid, response, coverage


def _linear_interpolate(
    x: torch.Tensor, y: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation of ``y(x)`` at ``query``.

    The batched branch indexes with the query positions directly rather than
    ``gather``ing an expanded copy of them. That is not a stylistic choice: the
    MPS backend silently returns the *unexpanded* row for an expanded index
    tensor, so the gallery of gathered values came back as zeros with no error
    raised -- a passband response of zero everywhere inside the band, which
    reads as a measured absence of flux rather than as a bug.
    """

    index = torch.searchsorted(x, query).clamp(1, x.numel() - 1)
    x0, x1 = x[index - 1], x[index]
    # A zero-width cell can only sit at the grid start, where both endpoints
    # hold the same value; the floor keeps that cell at fraction zero.
    fraction = (query - x0) / (x1 - x0).clamp_min(torch.finfo(x.dtype).tiny)
    if y.ndim == 1:
        return y[index - 1] + fraction * (y[index] - y[index - 1])
    y0, y1 = y[:, index - 1], y[:, index]
    return y0 + fraction.unsqueeze(0) * (y1 - y0)


__all__ = [
    "Passband",
    "PhotometryOutput",
    "PhotometryResult",
    "WavelengthUnit",
    "Weighting",
    "evaluate_throughput",
    "flux_to_magnitude",
    "magnitude_to_flux",
    "synthetic_photometry",
    "synthetic_photometry_bands",
    "throughput_from_mapping",
    "vega_spectrum",
    "vega_zero_point",
]
