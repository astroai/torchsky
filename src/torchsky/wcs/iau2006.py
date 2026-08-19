"""IAU 2006 precession-bias, truncated nutation, aberration, refraction.

This is a documented CIRS subset: Fukushima-Williams IAU 2006 precession-bias
plus the largest IAU 2000B lunisolar nutation terms (not the full 77-term
series). Stellar aberration uses a Keplerian Earth barycentric velocity.
Refraction is Saemundsson/Meeus and is never applied unless pressure and
temperature are supplied.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .frames import ARCSEC_TO_RAD, D2R

MJD_J2000 = 51544.5
AU_M = 1.495978707e11
C_M_S = 299792458.0
_EARTH_ECC = 0.01671
_SECONDS_PER_DAY = 86400.0

# Largest IAU 2000B lunisolar terms: (L, Lp, F, D, Omega, dpsi_sin_as, deps_cos_as)
_NUTATION_TERMS: tuple[tuple[int, int, int, int, int, float, float], ...] = (
    (0, 0, 0, 0, 1, -17.2064161, 9.2052331),
    (0, 0, 2, -2, 2, -1.3170905, 0.5730336),
    (0, 0, 2, 0, 2, -0.2276413, 0.0978459),
    (0, 0, 0, 0, 2, 0.2074554, -0.0897492),
    (0, 1, 0, 0, 0, 0.1475877, -0.0010690),
    (0, 1, 2, -2, 2, -0.0516821, 0.0224386),
    (1, 0, 0, 0, 0, 0.0711159, -0.0006750),
    (0, 0, 2, 0, 1, -0.0387300, 0.0200724),
    (1, 0, 2, 0, 2, -0.0301461, 0.0129026),
    (0, 1, -2, 2, -2, 0.0215811, -0.0095929),
)


def tt_mjd_from_utc(utc_mjd: Tensor, tt_minus_utc_s: Tensor) -> Tensor:
    return utc_mjd + tt_minus_utc_s / _SECONDS_PER_DAY


def _julian_centuries_tt(tt_mjd: Tensor) -> Tensor:
    return (tt_mjd - MJD_J2000) / 36525.0


def iau2006_fw_angles(tt_mjd: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fukushima-Williams IAU 2006 angles γ, φ, ψ, ε in radians (SOFA pfw06)."""
    t = _julian_centuries_tt(tt_mjd)

    def _poly(coeffs: tuple[float, ...]) -> Tensor:
        acc = torch.zeros_like(t)
        for c in reversed(coeffs):
            acc = acc * t + c
        return acc * ARCSEC_TO_RAD

    gamb = _poly((-0.052928, 10.556378, 0.4932044, -0.00031238, -2.788e-6, 2.60e-8))
    phib = _poly((84381.412819, -46.811016, 0.0511268, 0.00053289, -4.40e-7, -1.76e-8))
    psib = _poly((-0.041775, 5038.481484, 1.5584176, -0.00018522, -2.6452e-5, -1.72e-8))
    epsa = _poly((84381.406, -46.836769, -0.0001831, 0.00200340, -5.76e-7, -4.34e-8))
    return gamb, phib, psib, epsa


def _fundamental_args_rad(
    tt_mjd: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    t = _julian_centuries_tt(tt_mjd)
    # IAU 2000 mean arguments, degrees, then to rad (truncated polynomials).
    arg_l = (134.96340251 + (1717915923.2178 / 3600.0) * t) * D2R
    arg_lp = (357.52910918 + (129596581.0481 / 3600.0) * t) * D2R
    arg_f = (93.27209062 + (1739527262.8478 / 3600.0) * t) * D2R
    arg_d = (297.85019547 + (1602961601.2090 / 3600.0) * t) * D2R
    arg_om = (125.04455501 + (-6962890.5431 / 3600.0) * t) * D2R
    return arg_l, arg_lp, arg_f, arg_d, arg_om


def iau2000b_nutation_truncated(tt_mjd: Tensor) -> tuple[Tensor, Tensor]:
    """Largest IAU 2000B lunisolar terms; not the full 77-term series."""
    arg_l, arg_lp, arg_f, arg_d, arg_om = _fundamental_args_rad(tt_mjd)
    dpsi = torch.zeros_like(tt_mjd)
    deps = torch.zeros_like(tt_mjd)
    for nl, nlp, nf, nd, nom, s_as, c_as in _NUTATION_TERMS:
        arg = nl * arg_l + nlp * arg_lp + nf * arg_f + nd * arg_d + nom * arg_om
        dpsi = dpsi + (s_as * ARCSEC_TO_RAD) * torch.sin(arg)
        deps = deps + (c_as * ARCSEC_TO_RAD) * torch.cos(arg)
    return dpsi, deps


def _rx(angle: Tensor) -> Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    z = torch.zeros((), dtype=angle.dtype, device=angle.device)
    o = torch.ones((), dtype=angle.dtype, device=angle.device)
    return torch.stack(
        (
            torch.stack((o, z, z)),
            torch.stack((z, c, s)),
            torch.stack((z, -s, c)),
        )
    )


def _rz(angle: Tensor) -> Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    z = torch.zeros((), dtype=angle.dtype, device=angle.device)
    o = torch.ones((), dtype=angle.dtype, device=angle.device)
    return torch.stack(
        (
            torch.stack((c, s, z)),
            torch.stack((-s, c, z)),
            torch.stack((z, z, o)),
        )
    )


def npb_matrix(tt_mjd: Tensor, *, nutation: bool = True) -> Tensor:
    """GCRS→CIRS bias-precession-nutation (SOFA fw2m)."""
    gamb, phib, psib, epsa = iau2006_fw_angles(tt_mjd)
    if nutation:
        dpsi, deps = iau2000b_nutation_truncated(tt_mjd)
        psib = psib + dpsi
        epsa = epsa + deps
    return _rx(-epsa) @ _rz(-psib) @ _rx(phib) @ _rz(gamb)


def earth_barycentric_velocity_icrs(tt_mjd: Tensor) -> Tensor:
    """Keplerian Earth barycentric velocity in ICRS, metres per second."""
    t = _julian_centuries_tt(tt_mjd)
    # Mean longitude and anomaly, IAU 2000 (degrees).
    mean_lon = (280.4664567 + 36000.76982779 * t) * D2R
    g = (357.52910918 + 35999.050291 * t) * D2R
    # Ecliptic longitude of Earth (Sun + 180°) with first-order equation of center.
    lam = mean_lon + (2.0 * _EARTH_ECC * torch.sin(g))
    eps = 84381.406 * ARCSEC_TO_RAD
    cos_eps = math.cos(eps)
    sin_eps = math.sin(eps)
    # Circular-orbit speed scaled by (1 + e cos g).
    n = 2.0 * math.pi / (365.256363 * _SECONDS_PER_DAY)
    speed = n * AU_M * (1.0 + _EARTH_ECC * torch.cos(g))
    # Velocity is perpendicular to the heliocentric radius in the ecliptic.
    ecl = torch.stack(
        (-torch.sin(lam), torch.cos(lam) * cos_eps, torch.cos(lam) * sin_eps),
        dim=-1,
    )
    return speed.unsqueeze(-1) * ecl


def apply_stellar_aberration(unit_xyz: Tensor, velocity_m_s: Tensor) -> Tensor:
    """Classical stellar aberration: normalize(p + v/c)."""
    shifted = unit_xyz + velocity_m_s / C_M_S
    return shifted / torch.linalg.vector_norm(shifted, dim=-1, keepdim=True).clamp_min(
        1.0e-15
    )


def refraction_arcmin(
    alt_deg: Tensor, *, pressure_mbar: Tensor, temperature_c: Tensor
) -> Tensor:
    """Saemundsson/Meeus refraction in arcminutes. Not valid far below the horizon."""
    h = alt_deg
    inner = (h + 10.3 / (h + 5.11)) * D2R
    r = 1.02 / torch.tan(inner)
    scale = (pressure_mbar / 1010.0) * (283.0 / (273.0 + temperature_c))
    return r * scale


def apply_refraction(
    alt_deg: Tensor, *, pressure_mbar: Tensor | float, temperature_c: Tensor | float
) -> Tensor:
    p = torch.as_tensor(pressure_mbar, dtype=alt_deg.dtype, device=alt_deg.device)
    t = torch.as_tensor(temperature_c, dtype=alt_deg.dtype, device=alt_deg.device)
    return alt_deg + refraction_arcmin(alt_deg, pressure_mbar=p, temperature_c=t) / 60.0
