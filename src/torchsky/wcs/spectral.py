from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .._dtype import as_work_tensor as _as_tensor

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
    "cm": 1.0e-2,
    "centimeter": 1.0e-2,
    "centimetre": 1.0e-2,
    "mm": 1.0e-3,
    "um": 1.0e-6,
    "micron": 1.0e-6,
    "micrometer": 1.0e-6,
    "micrometre": 1.0e-6,
    "nm": 1.0e-9,
    "nanometer": 1.0e-9,
    "nanometre": 1.0e-9,
    "angstrom": 1.0e-10,
    "ang": 1.0e-10,
    "a": 1.0e-10,
}
_FREQ_TO_HZ = {"hz": 1.0, "khz": 1.0e3, "mhz": 1.0e6, "ghz": 1.0e9, "thz": 1.0e12}
_ENERGY_TO_J = {
    "j": 1.0,
    "joule": 1.0,
    "ev": ELECTRON_VOLT,
    "kev": 1.0e3 * ELECTRON_VOLT,
    "mev": 1.0e6 * ELECTRON_VOLT,
    "gev": 1.0e9 * ELECTRON_VOLT,
}
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
        raise ValueError(
            f"unsupported spectral coordinate type {spectral_type!r}"
        ) from exc


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
    try:
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
    except KeyError as exc:
        raise ValueError(
            f"unsupported spectral unit {unit!r} for type {kind}"
        ) from exc
    return 1.0


def _rest_frequency(
    rest_frequency: float | Tensor | None = None,
    rest_wavelength: float | Tensor | None = None,
) -> Tensor:
    if rest_frequency is not None:
        t = _as_tensor(rest_frequency)
        return t
    if rest_wavelength is not None:
        t = _as_tensor(rest_wavelength)
        return SPEED_OF_LIGHT / t
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

    Uses the WCSLIB ``awavwave`` refractive index (FITS-WCS Paper III / Edlén
    form), matching astropy.wcs. Inputs and outputs use ``unit`` when provided;
    FITS default is meters.
    """
    scale = spectral_unit_scale("AWAV", unit)
    air_m = _as_tensor(wavelength_air) * scale
    s = torch.where(air_m != 0.0, 1.0 / air_m, torch.zeros_like(air_m))
    s2 = s * s
    n = 1.000064328 + 2.554e8 / (0.41e14 - s2) + 294.981e8 / (1.46e14 - s2)
    vac_m = air_m * n
    return torch.where(air_m != 0.0, vac_m, torch.zeros_like(vac_m)) / scale


def vacuum_wavelength_to_air(wavelength_vacuum: Any, unit: str | None = None) -> Tensor:
    """Convert vacuum wavelength to standard-air wavelength.

    Uses WCSLIB ``waveawav`` (four refractive-index iterations).
    """
    scale = spectral_unit_scale("WAVE", unit)
    vac_m = _as_tensor(wavelength_vacuum) * scale
    n = torch.ones_like(vac_m)
    for _ in range(4):
        s = torch.where(vac_m != 0.0, n / vac_m, torch.zeros_like(vac_m))
        s2 = s * s
        n = 1.000064328 + 2.554e8 / (0.41e14 - s2) + 294.981e8 / (1.46e14 - s2)
    air_m = torch.where(vac_m != 0.0, vac_m / n, torch.zeros_like(vac_m))
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
    x = _as_tensor(value) * spectral_unit_scale(kind, unit)
    if kind == "FREQ":
        return x
    if kind == "WAVE":
        return SPEED_OF_LIGHT / x
    if kind == "WAVN":
        return SPEED_OF_LIGHT * x
    if kind == "AWAV":
        vac = air_wavelength_to_vacuum(value, unit=unit) * spectral_unit_scale(
            "WAVE", unit
        )
        return SPEED_OF_LIGHT / vac
    if kind == "ENER":
        return x / PLANCK

    f0 = _rest_frequency(rest_frequency, rest_wavelength).to(
        device=x.device, dtype=x.dtype
    )
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
    f = _as_tensor(frequency)
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
        f0 = _rest_frequency(rest_frequency, rest_wavelength).to(
            device=f.device, dtype=f.dtype
        )
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


def _is_positive_spectral_type(spectral_type: str) -> bool:
    return canonical_spectral_type(spectral_type) in {
        "WAVE",
        "FREQ",
        "AWAV",
        "WAVN",
        "ENER",
    }


def _spectral_derivative_si(
    value_si: float,
    from_type: str,
    to_type: str,
    *,
    rest_frequency: float | None,
    rest_wavelength: float | None,
) -> float:
    if canonical_spectral_type(from_type) == canonical_spectral_type(to_type):
        return 1.0
    try:
        x = torch.tensor(value_si, dtype=torch.float64, requires_grad=True)
        y = convert_spectral(
            x,
            from_type,
            to_type,
            rest_frequency=rest_frequency,
            rest_wavelength=rest_wavelength,
        )
        y.backward()
        if x.grad is not None and torch.isfinite(x.grad):
            return float(x.grad.item())
    except RuntimeError:
        # The conversion is not autograd-differentiable (e.g. detached or non-tensor
        # ops); fall back to a central finite difference below.
        pass
    eps = 1e-6 * max(1.0, abs(value_si))
    plus = float(
        convert_spectral(
            value_si + eps,
            from_type,
            to_type,
            rest_frequency=rest_frequency,
            rest_wavelength=rest_wavelength,
        ).item()
    )
    minus = float(
        convert_spectral(
            value_si - eps,
            from_type,
            to_type,
            rest_frequency=rest_frequency,
            rest_wavelength=rest_wavelength,
        ).item()
    )
    return (plus - minus) / (2.0 * eps)


class SpectralTransform:
    """Implement FITS Paper III spectral algorithm transformations.

    Handles grism (GRI/GRA), logarithmic (LOG), and compound X2P algorithms.
    """

    def __init__(
        self,
        ctype: str,
        crval: float,
        cunit: str | None = None,
        restfrq: float | None = None,
        restwav: float | None = None,
        pv: list[float] | None = None,
    ):
        self.ctype = ctype.strip().upper()
        parts = self.ctype.split("-")
        self.stype = parts[0]
        self.algo = parts[1] if len(parts) > 1 else ""

        if cunit is None:
            if self.stype in {"WAVE", "AWAV"}:
                self.cunit = "m"
            elif self.stype == "FREQ":
                self.cunit = "Hz"
            elif self.stype == "WAVN":
                self.cunit = "m-1"
            elif self.stype == "ENER":
                self.cunit = "J"
            elif self.stype == "ZOPT":
                self.cunit = ""
            elif self.stype == "BETA":
                self.cunit = ""
            elif self.stype in {"VRAD", "VOPT", "VELO"}:
                self.cunit = "m/s"
            else:
                self.cunit = "m"
        else:
            self.cunit = cunit

        self.restfrq = restfrq or 0.0
        self.restwav = restwav or 0.0
        if self.restfrq == 0.0 and self.restwav != 0.0:
            self.restfrq = SPEED_OF_LIGHT / self.restwav
        elif self.restwav == 0.0 and self.restfrq != 0.0:
            self.restwav = SPEED_OF_LIGHT / self.restfrq

        self.scale = spectral_unit_scale(self.stype, self.cunit)
        self.crval_si = crval * self.scale

        self.pv = [0.0] * 7
        self.pv[3] = 1.0  # default n_r
        if pv is not None:
            for idx, val in enumerate(pv):
                if idx < 7:
                    self.pv[idx] = val

        self.is_grism = self.algo in {"GRI", "GRA"}
        self.is_log = self.algo == "LOG"

        is_compound = (
            len(self.algo) == 3
            and self.algo[1] == "2"
            and self.algo[0] in {"F", "W", "A", "V", "N", "E", "Z", "B"}
            and self.algo[2] in {"F", "W", "A", "V", "N", "E", "Z", "B"}
        )
        if self.algo == "TAB":
            # -TAB lookup is owned by tabular axes; spectral algorithm is linear.
            self.algo = ""
        elif self.algo and not is_compound and not self.is_log and not self.is_grism:
            raise ValueError(
                f"unsupported FITS spectral algorithm {self.algo!r} in CTYPE {self.ctype!r}"
            )

        if self.is_grism:
            G = self.pv[0]
            m = self.pv[1]
            alpha = self.pv[2]
            n_r = self.pv[3]
            dn_r = self.pv[4]
            epsilon = self.pv[5]
            theta = self.pv[6]
            self.grism_xtype = "AWAV" if self.algo == "GRA" else "WAVE"

            d2r = math.pi / 180.0
            r2d = 180.0 / math.pi

            if canonical_spectral_type(self.stype) == self.grism_xtype:
                self.lambda_r = self.crval_si
            else:
                self.lambda_r = float(
                    convert_spectral(
                        self.crval_si,
                        self.stype,
                        self.grism_xtype,
                        rest_frequency=self.restfrq,
                        rest_wavelength=self.restwav,
                    ).item()
                )

            self.t = G * m / math.cos(epsilon * d2r)
            sin_arg = self.t * self.lambda_r - n_r * math.sin(alpha * d2r)
            sin_arg_clamped = max(-1.0, min(1.0, sin_arg))
            self.beta_r = math.asin(sin_arg_clamped) * r2d

            self.t_prime = self.t - dn_r * math.sin(alpha * d2r)
            self.w1 = -math.tan(theta * d2r)

            self.dXdS = _spectral_derivative_si(
                self.crval_si,
                self.stype,
                self.grism_xtype,
                rest_frequency=self.restfrq,
                rest_wavelength=self.restwav,
            )

            cos_b = math.cos(self.beta_r * d2r)
            cos_t = math.cos(theta * d2r)
            if abs(cos_b) < 1e-12:
                cos_b = 1e-12 if cos_b >= 0 else -1e-12
            self.w2 = self.dXdS * self.t_prime / (cos_b * (cos_t**2))
            self.w2 = self.w2 * self.scale

            self.w3 = self.beta_r + theta
            self.w4 = (n_r - dn_r * self.lambda_r) * math.sin(alpha * d2r)
            self.w5 = 1.0 / self.t_prime

        elif self.is_log:
            pass

        elif self.algo:
            self.xtype_char = self.algo[0]
            self.ptype_char = self.algo[2]

            xtype_map = {
                "F": "FREQ",
                "W": "WAVE",
                "A": "AWAV",
                "V": "VELO",
                "N": "WAVN",
                "E": "ENER",
                "Z": "ZOPT",
                "B": "BETA",
            }
            self.xtype_name = xtype_map[self.xtype_char]
            self.ptype_name = xtype_map[self.ptype_char]

            self.crvalX_si = float(
                convert_spectral(
                    self.crval_si,
                    self.stype,
                    self.xtype_name,
                    rest_frequency=self.restfrq,
                    rest_wavelength=self.restwav,
                ).item()
            )

            try:
                s_tensor = torch.tensor(
                    self.crval_si, dtype=torch.float64, requires_grad=True
                )
                X_tensor = convert_spectral(
                    s_tensor,
                    self.stype,
                    self.xtype_name,
                    rest_frequency=self.restfrq,
                    rest_wavelength=self.restwav,
                )
                X_tensor.backward()
                self.dXdS_si = float(s_tensor.grad.item())
            except RuntimeError:
                # Not autograd-differentiable: use a central finite difference.
                eps = 1e-6 * max(1.0, abs(self.crval_si))
                val_plus = float(
                    convert_spectral(
                        self.crval_si + eps,
                        self.stype,
                        self.xtype_name,
                        rest_frequency=self.restfrq,
                        rest_wavelength=self.restwav,
                    ).item()
                )
                val_minus = float(
                    convert_spectral(
                        self.crval_si - eps,
                        self.stype,
                        self.xtype_name,
                        rest_frequency=self.restfrq,
                        rest_wavelength=self.restwav,
                    ).item()
                )
                self.dXdS_si = (val_plus - val_minus) / (2.0 * eps)
            self.dXdS = self.dXdS_si * self.scale

    def pixel_to_world(self, x: Tensor) -> Tensor:
        if self.is_grism:
            d2r = math.pi / 180.0
            r2d = 180.0 / math.pi

            val = self.w1 + x * self.w2
            beta = torch.atan(val) * r2d + self.w3
            lambda_val = (torch.sin(beta * d2r) + self.w4) * self.w5

            if canonical_spectral_type(self.stype) == self.grism_xtype:
                spec_si = lambda_val
            else:
                spec_si = convert_spectral(
                    lambda_val,
                    self.grism_xtype,
                    self.stype,
                    rest_frequency=self.restfrq,
                    rest_wavelength=self.restwav,
                )
            return spec_si / self.scale

        elif self.is_log:
            x_si = x * self.scale
            X = self.crval_si * torch.exp(x_si / self.crval_si)
            return X / self.scale

        elif self.algo:
            X = self.crvalX_si + x * self.dXdS
            if _is_positive_spectral_type(self.xtype_name):
                valid_x = X > 0.0
                X_safe = torch.where(valid_x, X, torch.full_like(X, torch.nan))
            else:
                valid_x = torch.ones_like(X, dtype=torch.bool)
                X_safe = X
            spec_si = convert_spectral(
                X_safe,
                self.xtype_name,
                self.stype,
                rest_frequency=self.restfrq,
                rest_wavelength=self.restwav,
            )
            res = spec_si / self.scale
            valid = valid_x & torch.isfinite(res)
            return torch.where(valid, res, torch.full_like(res, torch.nan))

        else:
            return self.crval_si / self.scale + x

    def world_to_pixel(self, spec: Tensor) -> Tensor:
        spec_si = spec * self.scale

        if self.is_grism:
            d2r = math.pi / 180.0
            r2d = 180.0 / math.pi

            if canonical_spectral_type(self.stype) == self.grism_xtype:
                lambda_val = spec_si
            else:
                lambda_val = convert_spectral(
                    spec_si,
                    self.stype,
                    self.grism_xtype,
                    rest_frequency=self.restfrq,
                    rest_wavelength=self.restwav,
                )

            s = lambda_val / self.w5 - self.w4
            s_clamped = torch.clamp(s, min=-1.0, max=1.0)
            beta = torch.asin(s_clamped) * r2d
            val = torch.tan((beta - self.w3) * d2r)

            x = (val - self.w1) / self.w2
            return torch.where(torch.abs(s) <= 1.0, x, torch.full_like(x, torch.nan))

        elif self.is_log:
            ratio = spec_si / self.crval_si
            x_si = self.crval_si * torch.log(ratio)
            x = x_si / self.scale
            return torch.where(ratio > 0.0, x, torch.full_like(x, torch.nan))

        elif self.algo:
            is_positive_type = _is_positive_spectral_type(self.stype)
            if is_positive_type:
                spec_safe = torch.where(
                    spec_si > 0.0, spec_si, torch.full_like(spec_si, torch.nan)
                )
            else:
                spec_safe = spec_si

            X = convert_spectral(
                spec_safe,
                self.stype,
                self.xtype_name,
                rest_frequency=self.restfrq,
                rest_wavelength=self.restwav,
            )
            res = (X - self.crvalX_si) / self.dXdS
            valid = torch.isfinite(X)
            if is_positive_type:
                valid = valid & (spec_si > 0.0)
            return torch.where(valid, res, torch.full_like(res, torch.nan))

        else:
            return spec - self.crval_si / self.scale
