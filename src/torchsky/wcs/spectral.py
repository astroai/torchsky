from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

SPEED_OF_LIGHT = 299792458.0  # m / s
PLANCK = 6.62607015e-34  # J s
ELECTRON_VOLT = 1.602176634e-19  # J

_SPECTRAL_ALIASES = {
    "F": "FREQ",
    "FREQ": "FREQ",
    "FREQUENCY": "FREQ",
    "W": "WAVE",
    "WAVE": "WAVE",
    "WAVELENGTH": "WAVE",
    "LAMBDA": "WAVE",
    "N": "WAVN",
    "WAVN": "WAVN",
    "WAVENUMBER": "WAVN",
    "WAVE_NUMBER": "WAVN",
    "AWAV": "AWAV",
    "AIR": "AWAV",
    "AIRWAVE": "AWAV",
    "AIR_WAVELENGTH": "AWAV",
    "E": "ENER",
    "ENER": "ENER",
    "ENERGY": "ENER",
    "Z": "ZOPT",
    "ZOPT": "ZOPT",
    "REDSHIFT": "ZOPT",
    "VRAD": "VRAD",
    "RADIO": "VRAD",
    "RADIO_VELOCITY": "VRAD",
    "VOPT": "VOPT",
    "OPTICAL": "VOPT",
    "OPTICAL_VELOCITY": "VOPT",
    "VELO": "VELO",
    "RELATIVISTIC": "VELO",
    "RELATIVISTIC_VELOCITY": "VELO",
    "BETA": "BETA",
}

_LENGTH_TO_M = {
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
    "mm": 1.0e-3,
    "um": 1.0e-6,
    "micron": 1.0e-6,
    "nm": 1.0e-9,
    "angstrom": 1.0e-10,
    "ang": 1.0e-10,
    "a": 1.0e-10,
}
_FREQ_TO_HZ = {"hz": 1.0, "khz": 1.0e3, "mhz": 1.0e6, "ghz": 1.0e9, "thz": 1.0e12}
_ENERGY_TO_J = {"j": 1.0, "ev": ELECTRON_VOLT, "kev": 1.0e3 * ELECTRON_VOLT, "mev": 1.0e6 * ELECTRON_VOLT}
_VELOCITY_TO_M_S = {"m/s": 1.0, "ms-1": 1.0, "km/s": 1.0e3, "kms-1": 1.0e3}
_WAVENUMBER_TO_INV_M = {
    "m-1": 1.0,
    "1/m": 1.0,
    "m**-1": 1.0,
    "m^-1": 1.0,
    "cm-1": 100.0,
    "1/cm": 100.0,
    "cm**-1": 100.0,
    "cm^-1": 100.0,
}


def canonical_spectral_type(spectral_type: str) -> str:
    """Return the canonical FITS spectral coordinate type."""
    key = str(spectral_type).strip().upper()
    if "-" in key:
        key = key.split("-", 1)[0]
    key = key.replace(" ", "_")
    try:
        return _SPECTRAL_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported spectral coordinate type {spectral_type!r}") from exc


def spectral_unit_scale(spectral_type: str, unit: str | None = None) -> float:
    """Return the multiplicative scale from ``unit`` to the SI unit for a type."""
    kind = canonical_spectral_type(spectral_type)
    if unit is None or str(unit).strip() == "":
        if kind in {"WAVE", "AWAV"}:
            return 1.0
        if kind == "WAVN":
            return 1.0
        if kind == "FREQ":
            return 1.0
        if kind == "ENER":
            return 1.0
        if kind in {"VRAD", "VOPT", "VELO"}:
            return 1.0
        return 1.0
    u = str(unit).strip().lower().replace(" ", "")
    if kind in {"WAVE", "AWAV"}:
        return _LENGTH_TO_M[u]
    if kind == "FREQ":
        return _FREQ_TO_HZ[u]
    if kind == "WAVN":
        return _WAVENUMBER_TO_INV_M[u]
    if kind == "ENER":
        return _ENERGY_TO_J[u]
    if kind in {"VRAD", "VOPT", "VELO"}:
        return _VELOCITY_TO_M_S[u]
    return 1.0


def _as_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.as_tensor(value, dtype=torch.float64)


def _rest_frequency(
    rest_frequency: float | Tensor | None = None,
    rest_wavelength: float | Tensor | None = None,
) -> Tensor:
    if rest_frequency is not None:
        return _as_tensor(rest_frequency).to(dtype=torch.float64)
    if rest_wavelength is not None:
        return SPEED_OF_LIGHT / _as_tensor(rest_wavelength).to(dtype=torch.float64)
    raise ValueError("rest_frequency or rest_wavelength is required")


def rest_frequency_from_header(header: dict[str, Any]) -> float | None:
    """Return rest frequency in Hz from FITS ``RESTFRQ/FREQ`` or ``RESTWAV``."""
    if "RESTFRQ" in header:
        return float(header["RESTFRQ"])
    if "RESTFREQ" in header:
        return float(header["RESTFREQ"])
    if "RESTWAV" in header:
        return SPEED_OF_LIGHT / float(header["RESTWAV"])
    return None


def air_wavelength_to_vacuum(wavelength_air: Any, unit: str | None = None) -> Tensor:
    """Convert standard-air wavelength to vacuum wavelength.

    Uses the Morton/IAU-style refractive-index approximation for dry standard air.
    Inputs and outputs use ``unit`` when provided; FITS default is meters.
    """
    scale = spectral_unit_scale("AWAV", unit)
    air_m = _as_tensor(wavelength_air).to(dtype=torch.float64) * scale
    air_a = air_m / 1.0e-10
    sigma2 = (1.0e4 / air_a) ** 2
    n = 1.0 + 1.0e-8 * (
        8342.54 + 2406147.0 / (130.0 - sigma2) + 15998.0 / (38.9 - sigma2)
    )
    vac_m = air_m * n
    return vac_m / scale


def vacuum_wavelength_to_air(wavelength_vacuum: Any, unit: str | None = None) -> Tensor:
    """Convert vacuum wavelength to standard-air wavelength."""
    scale = spectral_unit_scale("WAVE", unit)
    vac_m = _as_tensor(wavelength_vacuum).to(dtype=torch.float64) * scale
    air_m = vac_m.clone()
    for _ in range(3):
        air_a = air_m / 1.0e-10
        sigma2 = (1.0e4 / air_a) ** 2
        n = 1.0 + 1.0e-8 * (
            8342.54 + 2406147.0 / (130.0 - sigma2) + 15998.0 / (38.9 - sigma2)
        )
        air_m = vac_m / n
    return air_m / scale


def spectral_to_frequency(
    value: Any,
    spectral_type: str,
    *,
    unit: str | None = None,
    rest_frequency: float | Tensor | None = None,
    rest_wavelength: float | Tensor | None = None,
) -> Tensor:
    """Convert a spectral coordinate to frequency in Hz."""
    kind = canonical_spectral_type(spectral_type)
    x = _as_tensor(value).to(dtype=torch.float64) * spectral_unit_scale(kind, unit)
    if kind == "FREQ":
        return x
    if kind == "WAVE":
        return SPEED_OF_LIGHT / x
    if kind == "WAVN":
        return SPEED_OF_LIGHT * x
    if kind == "AWAV":
        vac = air_wavelength_to_vacuum(value, unit=unit) * spectral_unit_scale("WAVE", unit)
        return SPEED_OF_LIGHT / vac
    if kind == "ENER":
        return x / PLANCK

    f0 = _rest_frequency(rest_frequency, rest_wavelength).to(device=x.device)
    if kind == "ZOPT":
        return f0 / (1.0 + x)
    if kind == "VRAD":
        return f0 * (1.0 - x / SPEED_OF_LIGHT)
    if kind == "VOPT":
        return f0 / (1.0 + x / SPEED_OF_LIGHT)
    if kind == "VELO":
        beta = x / SPEED_OF_LIGHT
        return f0 * torch.sqrt((1.0 - beta) / (1.0 + beta))
    if kind == "BETA":
        return f0 * torch.sqrt((1.0 - x) / (1.0 + x))
    raise ValueError(f"unsupported spectral coordinate type {spectral_type!r}")


def frequency_to_spectral(
    frequency: Any,
    spectral_type: str,
    *,
    unit: str | None = None,
    rest_frequency: float | Tensor | None = None,
    rest_wavelength: float | Tensor | None = None,
) -> Tensor:
    """Convert frequency in Hz to a requested spectral coordinate."""
    kind = canonical_spectral_type(spectral_type)
    f = _as_tensor(frequency).to(dtype=torch.float64)
    if kind == "FREQ":
        y = f
    elif kind == "WAVE":
        y = SPEED_OF_LIGHT / f
    elif kind == "WAVN":
        y = f / SPEED_OF_LIGHT
    elif kind == "AWAV":
        y = vacuum_wavelength_to_air(SPEED_OF_LIGHT / f, unit="m")
    elif kind == "ENER":
        y = PLANCK * f
    else:
        f0 = _rest_frequency(rest_frequency, rest_wavelength).to(device=f.device)
        if kind == "ZOPT":
            y = f0 / f - 1.0
        elif kind == "VRAD":
            y = SPEED_OF_LIGHT * (1.0 - f / f0)
        elif kind == "VOPT":
            y = SPEED_OF_LIGHT * (f0 / f - 1.0)
        elif kind == "VELO":
            ratio2 = (f / f0) ** 2
            y = SPEED_OF_LIGHT * (1.0 - ratio2) / (1.0 + ratio2)
        elif kind == "BETA":
            ratio2 = (f / f0) ** 2
            y = (1.0 - ratio2) / (1.0 + ratio2)
        else:
            raise ValueError(f"unsupported spectral coordinate type {spectral_type!r}")
    return y / spectral_unit_scale(kind, unit)


def convert_spectral(
    value: Any,
    from_type: str,
    to_type: str,
    *,
    from_unit: str | None = None,
    to_unit: str | None = None,
    rest_frequency: float | Tensor | None = None,
    rest_wavelength: float | Tensor | None = None,
) -> Tensor:
    """Convert between supported spectral coordinate types through frequency."""
    freq = spectral_to_frequency(
        value,
        from_type,
        unit=from_unit,
        rest_frequency=rest_frequency,
        rest_wavelength=rest_wavelength,
    )
    return frequency_to_spectral(
        freq,
        to_type,
        unit=to_unit,
        rest_frequency=rest_frequency,
        rest_wavelength=rest_wavelength,
    )


def transform_spectral(
    freq: Tensor, rest_freq: float, convention: str = "RADIO"
) -> Tensor:
    """Backward-compatible frequency-to-velocity helper returning km/s."""
    target = {
        "RADIO": "VRAD",
        "OPTICAL": "VOPT",
        "RELATIVISTIC": "VELO",
    }.get(convention.upper())
    if target is None:
        raise ValueError(f"Unknown spectral convention: {convention}")
    return frequency_to_spectral(
        freq, target, unit="km/s", rest_frequency=rest_freq
    )
