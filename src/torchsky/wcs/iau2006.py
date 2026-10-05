"""IAU 2006 precession-bias, IAU 2000A nutation, aberration, refraction.

Fukushima-Williams IAU 2006 precession-bias plus full IAU 2000A nutation
(with IAU 2006/P03 adjustments, SOFA nut06a). Stellar aberration uses a
Keplerian Earth barycentric velocity. Refraction is Saemundsson/Meeus and
is never applied unless pressure and temperature are supplied.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .frames import ARCSEC_TO_RAD, D2R
from .iau2000a import (
    fae03,
    faf03,
    fal03,
    falp03,
    faom03,
    fapa03,
    fave03,
    fad03,
    iau2006_nutation,
    julian_centuries_tt,
)

MJD_J2000 = 51544.5
AU_M = 1.495978707e11
C_M_S = 299792458.0
_EARTH_ECC = 0.01671
_SECONDS_PER_DAY = 86400.0


def tt_mjd_from_utc(utc_mjd: Tensor, tt_minus_utc_s: Tensor) -> Tensor:
    return utc_mjd + tt_minus_utc_s / _SECONDS_PER_DAY


def _julian_centuries_tt(tt_mjd: Tensor) -> Tensor:
    return julian_centuries_tt(tt_mjd)


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


def _rx(angle: Tensor) -> Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack(
        (
            torch.stack((o, z, z), dim=-1),
            torch.stack((z, c, s), dim=-1),
            torch.stack((z, -s, c), dim=-1),
        ),
        dim=-2,
    )


def _ry(angle: Tensor) -> Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack(
        (
            torch.stack((c, z, -s), dim=-1),
            torch.stack((z, o, z), dim=-1),
            torch.stack((s, z, c), dim=-1),
        ),
        dim=-2,
    )


def _rz(angle: Tensor) -> Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    return torch.stack(
        (
            torch.stack((c, s, z), dim=-1),
            torch.stack((-s, c, z), dim=-1),
            torch.stack((z, z, o), dim=-1),
        ),
        dim=-2,
    )


# SOFA s06 series for s+XY/2, coefficients in arcseconds (IERS 2003 / IAU 2006).
_S06_SP = (94.00e-6, 3808.65e-6, -122.68e-6, -72574.11e-6, 27.98e-6, 15.62e-6)
_S06_NFA = (
    torch.tensor(
        [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 2, -2, 3, 0, 0, 0],
            [0, 0, 2, -2, 1, 0, 0, 0],
            [0, 0, 2, -2, 2, 0, 0, 0],
            [0, 0, 2, 0, 3, 0, 0, 0],
            [0, 0, 2, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, -1, 0, 0, 0],
            [1, 0, 0, 0, -1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 2, -2, 3, 0, 0, 0],
            [0, 1, 2, -2, 1, 0, 0, 0],
            [0, 0, 4, -4, 4, 0, 0, 0],
            [0, 0, 1, -1, 1, -8, 12, 0],
            [0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 2, 0, 0, 0],
            [1, 0, 2, 0, 3, 0, 0, 0],
            [1, 0, 2, 0, 1, 0, 0, 0],
            [0, 0, 2, -2, 0, 0, 0, 0],
            [0, 1, -2, 2, -3, 0, 0, 0],
            [0, 1, -2, 2, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 8, -13, -1],
            [0, 0, 0, 2, 0, 0, 0, 0],
            [2, 0, -2, 0, -1, 0, 0, 0],
            [0, 1, 2, -2, 2, 0, 0, 0],
            [1, 0, 0, -2, 1, 0, 0, 0],
            [1, 0, 0, -2, -1, 0, 0, 0],
            [0, 0, 4, -2, 4, 0, 0, 0],
            [0, 0, 2, -2, 4, 0, 0, 0],
            [1, 0, -2, 0, -3, 0, 0, 0],
            [1, 0, -2, 0, -1, 0, 0, 0],
        ],
        dtype=torch.float64,
    ),
    torch.tensor(
        [[0, 0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 2, -2, 3, 0, 0, 0]],
        dtype=torch.float64,
    ),
    torch.tensor(
        [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 2, -2, 2, 0, 0, 0],
            [0, 0, 2, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, -2, 2, 0, 0, 0],
            [0, 0, 2, 0, 1, 0, 0, 0],
            [1, 0, 2, 0, 2, 0, 0, 0],
            [0, 1, -2, 2, -2, 0, 0, 0],
            [1, 0, 0, -2, 0, 0, 0, 0],
            [0, 0, 2, -2, 1, 0, 0, 0],
            [1, 0, -2, 0, -2, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, -2, -2, -2, 0, 0, 0],
            [1, 0, 0, 0, -1, 0, 0, 0],
            [1, 0, 2, 0, 1, 0, 0, 0],
            [2, 0, 0, -2, 0, 0, 0, 0],
            [2, 0, -2, 0, -1, 0, 0, 0],
            [0, 0, 2, 2, 2, 0, 0, 0],
            [2, 0, 2, 0, 2, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 2, -2, 2, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float64,
    ),
    torch.tensor(
        [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 2, -2, 2, 0, 0, 0],
            [0, 0, 2, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0],
        ],
        dtype=torch.float64,
    ),
    torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.float64),
)
_S06_SC = (
    torch.tensor(
        [
            [-2640.73e-6, 0.39e-6],
            [-63.53e-6, 0.02e-6],
            [-11.75e-6, -0.01e-6],
            [-11.21e-6, -0.01e-6],
            [4.57e-6, 0.00e-6],
            [-2.02e-6, 0.00e-6],
            [-1.98e-6, 0.00e-6],
            [1.72e-6, 0.00e-6],
            [1.41e-6, 0.01e-6],
            [1.26e-6, 0.01e-6],
            [0.63e-6, 0.00e-6],
            [0.63e-6, 0.00e-6],
            [-0.46e-6, 0.00e-6],
            [-0.45e-6, 0.00e-6],
            [-0.36e-6, 0.00e-6],
            [0.24e-6, 0.12e-6],
            [-0.32e-6, 0.00e-6],
            [-0.28e-6, 0.00e-6],
            [-0.27e-6, 0.00e-6],
            [-0.26e-6, 0.00e-6],
            [0.21e-6, 0.00e-6],
            [-0.19e-6, 0.00e-6],
            [-0.18e-6, 0.00e-6],
            [0.10e-6, -0.05e-6],
            [-0.15e-6, 0.00e-6],
            [0.14e-6, 0.00e-6],
            [0.14e-6, 0.00e-6],
            [-0.14e-6, 0.00e-6],
            [-0.14e-6, 0.00e-6],
            [-0.13e-6, 0.00e-6],
            [0.11e-6, 0.00e-6],
            [-0.11e-6, 0.00e-6],
            [-0.11e-6, 0.00e-6],
        ],
        dtype=torch.float64,
    ),
    torch.tensor(
        [[-0.07e-6, 3.57e-6], [1.73e-6, -0.03e-6], [0.00e-6, 0.48e-6]],
        dtype=torch.float64,
    ),
    torch.tensor(
        [
            [743.52e-6, -0.17e-6],
            [56.91e-6, 0.06e-6],
            [9.84e-6, -0.01e-6],
            [-8.85e-6, 0.01e-6],
            [-6.38e-6, -0.05e-6],
            [-3.07e-6, 0.00e-6],
            [2.23e-6, 0.00e-6],
            [1.67e-6, 0.00e-6],
            [1.30e-6, 0.00e-6],
            [0.93e-6, 0.00e-6],
            [0.68e-6, 0.00e-6],
            [-0.55e-6, 0.00e-6],
            [0.53e-6, 0.00e-6],
            [-0.27e-6, 0.00e-6],
            [-0.27e-6, 0.00e-6],
            [-0.26e-6, 0.00e-6],
            [-0.25e-6, 0.00e-6],
            [0.22e-6, 0.00e-6],
            [-0.21e-6, 0.00e-6],
            [0.20e-6, 0.00e-6],
            [0.17e-6, 0.00e-6],
            [0.13e-6, 0.00e-6],
            [-0.13e-6, 0.00e-6],
            [-0.12e-6, 0.00e-6],
            [-0.11e-6, 0.00e-6],
        ],
        dtype=torch.float64,
    ),
    torch.tensor(
        [
            [0.30e-6, -23.42e-6],
            [-0.03e-6, -1.46e-6],
            [-0.01e-6, -0.25e-6],
            [0.00e-6, 0.23e-6],
        ],
        dtype=torch.float64,
    ),
    torch.tensor([[-0.26e-6, -0.01e-6]], dtype=torch.float64),
)


def cio_locator_s06(tt_mjd: Tensor, cip_x: Tensor, cip_y: Tensor) -> Tensor:
    """CIO locator s (SOFA s06), radians."""
    t = _julian_centuries_tt(tt_mjd)
    fa = torch.stack(
        (
            fal03(t),
            falp03(t),
            faf03(t),
            fad03(t),
            faom03(t),
            fave03(t),
            fae03(t),
            fapa03(t),
        ),
        dim=-1,
    )
    w = [torch.as_tensor(c, dtype=t.dtype, device=t.device) for c in _S06_SP]
    for k, (nfa, sc) in enumerate(zip(_S06_NFA, _S06_SC, strict=True)):
        nfa_t = nfa.to(device=t.device, dtype=t.dtype)
        sc_t = sc.to(device=t.device, dtype=t.dtype)
        arg = (fa.unsqueeze(-2) * nfa_t).sum(dim=-1)
        w[k] = w[k] + (sc_t[:, 0] * torch.sin(arg) + sc_t[:, 1] * torch.cos(arg)).sum(
            dim=-1
        )
    s_as = w[0] + t * (w[1] + t * (w[2] + t * (w[3] + t * (w[4] + t * w[5]))))
    return s_as * ARCSEC_TO_RAD - cip_x * cip_y / 2.0


def npb_matrix(tt_mjd: Tensor, *, nutation: bool = True) -> Tensor:
    """GCRS → true equator and equinox of date (SOFA pnm06a / fw2m)."""
    gamb, phib, psib, epsa = iau2006_fw_angles(tt_mjd)
    if nutation:
        dpsi, deps = iau2006_nutation(tt_mjd)
        psib = psib + dpsi
        epsa = epsa + deps
    return _rx(-epsa) @ _rz(-psib) @ _rx(phib) @ _rz(gamb)


def _c2ixys(cip_x: Tensor, cip_y: Tensor, s: Tensor) -> Tensor:
    r2 = cip_x * cip_x + cip_y * cip_y
    e = torch.where(r2 > 0.0, torch.atan2(cip_y, cip_x), torch.zeros_like(r2))
    d = torch.atan(torch.sqrt(r2 / (1.0 - r2).clamp_min(1.0e-30)))
    return _rz(-(e + s)) @ _ry(d) @ _rz(e)


def cirs_matrix(tt_mjd: Tensor) -> Tensor:
    """GCRS → CIRS (SOFA c2i06a)."""
    rnpb = npb_matrix(tt_mjd, nutation=True)
    cip_x, cip_y = rnpb[..., 2, 0], rnpb[..., 2, 1]
    s = cio_locator_s06(tt_mjd, cip_x, cip_y)
    return _c2ixys(cip_x, cip_y, s)


def earth_barycentric_velocity_icrs(tt_mjd: Tensor) -> Tensor:
    """Keplerian Earth barycentric velocity in ICRS, metres per second."""
    t = _julian_centuries_tt(tt_mjd)
    # Mean longitude and anomaly, IAU 2000 (degrees).
    mean_lon = (280.4664567 + 36000.76982779 * t) * D2R
    g = (357.52910918 + 35999.050291 * t) * D2R
    # Ecliptic longitude of Earth (Sun + 180°) with first-order equation of center.
    lam = mean_lon + math.pi + (2.0 * _EARTH_ECC * torch.sin(g))
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
