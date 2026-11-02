"""IAU 2006 precession-bias, IAU 2000A nutation, aberration, refraction.

Fukushima-Williams IAU 2006 precession-bias plus full IAU 2000A nutation
(with IAU 2006/P03 adjustments, SOFA nut06a). Earth position and velocity use the
ERFA ``epv00`` series (:func:`earth_pv_icrs`). Refraction is the ERFA ``refco``
model by default (Saemundsson/Meeus on request) and is never applied unless
pressure and temperature are supplied.
"""

from __future__ import annotations

import functools

import torch
from torch import Tensor

from ._epv00_data import EPV00_COUNTS, epv00_terms
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


_DJY = 365.25
# Ecliptic (mean J2000) to ICRS, eraEpv00.
_EPV00_AM = (
    (1.0, 0.000000211284, -0.000000091603),
    (-0.000000230286, 0.917482137087, -0.397776982902),
    (0.0, 0.397776982902, 0.917482137087),
)
_SRS = 1.97412574336e-8  # 2 G M_sun / (c^2 au): Schwarzschild radius of the Sun, au
_C_AU_PER_DAY = _SECONDS_PER_DAY * C_M_S / AU_M


@functools.lru_cache(maxsize=1)
def _epv00_tables() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Concatenated epv00 terms: amp, phase, freq, power of T, (Earth|Sun) x axis."""
    flat = torch.tensor(epv00_terms(), dtype=torch.float64).reshape(-1, 3)
    power, weights = [], []
    for table, count in enumerate(EPV00_COUNTS):
        body, rest = divmod(table, 9)  # 0: heliocentric Earth, 1: barycentric Sun
        k, axis = divmod(rest, 3)
        w = torch.zeros(count, 6, dtype=torch.float64)
        w[:, body * 3 + axis] = 1.0
        power.append(torch.full((count,), float(k), dtype=torch.float64))
        weights.append(w)
    return (
        flat[:, 0].contiguous(),
        flat[:, 1].contiguous(),
        flat[:, 2].contiguous(),
        torch.cat(power),
        torch.cat(weights),
    )


def earth_pv_icrs(tt_mjd: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """ERFA ``epv00``: Earth heliocentric position, barycentric position and velocity.

    Returns ``(helio_pos_au, bary_pos_au, bary_vel_au_per_day)``, each ``(..., 3)``
    in ICRS. The argument is TT MJD (TDB differs by < 2 ms, i.e. < 0.1 m in
    position). Evaluated in float64 on the CPU for any input device and moved
    back. Agrees with ``erfa.epv00`` to ~1e-15 au (position) and ~1e-17 au/day.
    """
    t_in = torch.as_tensor(tt_mjd)
    dev = t_in.device
    t_all = ((t_in.detach().cpu().to(torch.float64) - MJD_J2000) / _DJY).reshape(-1)
    amp, phase, freq, power, weights = _epv00_tables()
    am = torch.tensor(_EPV00_AM, dtype=torch.float64)
    pos_chunks, vel_chunks, helio_chunks = [], [], []
    block = 2048
    for start in range(0, t_all.numel(), block):
        t = t_all[start : start + block, None]
        arg = phase + freq * t
        cp, sp = torch.cos(arg), torch.sin(arg)
        tk = t.pow(power)
        tkm1 = torch.where(power > 0.0, t.pow((power - 1.0).clamp_min(0.0)), 0.0 * t)
        pos_term = amp * tk * cp
        vel_term = amp * (power * tkm1 * cp - freq * tk * sp)
        pos = pos_term @ weights  # (n, 6): Earth xyz | Sun xyz (ecliptic)
        vel = vel_term @ weights
        helio_chunks.append(pos[:, :3] @ am.T)
        pos_chunks.append((pos[:, :3] + pos[:, 3:]) @ am.T)
        vel_chunks.append((vel[:, :3] + vel[:, 3:]) @ am.T / _DJY)
    shape = t_in.shape + (3,)
    if t_all.numel() == 0:
        empty = torch.zeros(shape, dtype=torch.float64)
        return empty.to(dev), empty.to(dev), empty.to(dev)
    out = [
        torch.cat(c).reshape(shape).to(dev)
        for c in (helio_chunks, pos_chunks, vel_chunks)
    ]
    return out[0], out[1], out[2]


def earth_barycentric_velocity_icrs(tt_mjd: Tensor) -> Tensor:
    """Earth barycentric velocity in ICRS, metres per second (ERFA ``epv00`` series)."""
    t_in = torch.as_tensor(tt_mjd)
    velocity = earth_pv_icrs(t_in)[2] * (AU_M / _SECONDS_PER_DAY)
    return velocity.to(t_in.dtype) if t_in.is_floating_point() else velocity


def apparent_gcrs_unit(
    unit_xyz: Tensor,
    tt_mjd: Tensor,
    observer_velocity_m_s: Tensor | None = None,
    observer_position_m: Tensor | None = None,
) -> Tensor:
    """Astrometric ICRS direction to apparent GCRS (ERFA ``atciq`` for a star at infinity).

    Applies solar light deflection (``ldsun``) and annual stellar aberration with
    the relativistic terms of ``eraAb``, using the ``epv00`` Earth ephemeris.
    ``observer_velocity_m_s`` is the observer's velocity relative to the geocentre
    in GCRS (diurnal aberration); it is added to the Earth's barycentric velocity
    exactly as ERFA ``apcs`` does, and ``observer_position_m`` (GCRS, relative to
    the geocentre) shifts the Sun-observer vector of the deflection and of the
    ``eraAb`` term. Time quantities are float64 on the CPU.
    """
    dev, dtype = unit_xyz.device, unit_xyz.dtype
    helio, _, vel_au_day = earth_pv_icrs(tt_mjd)
    helio = helio.to(device=dev, dtype=dtype)
    if observer_position_m is not None:
        helio = helio + observer_position_m.to(device=dev, dtype=dtype) / AU_M
    vel = vel_au_day.to(device=dev, dtype=dtype) / _C_AU_PER_DAY
    if observer_velocity_m_s is not None:
        vel = vel + observer_velocity_m_s.to(device=dev, dtype=dtype) / C_M_S
    em = torch.linalg.vector_norm(helio, dim=-1, keepdim=True)
    eh = helio / em
    unit_xyz, eh = torch.broadcast_tensors(unit_xyz, eh)
    # Light deflection by the Sun (eraLdsun -> eraLd with bm = 1).
    qdqpe = (unit_xyz * (unit_xyz + eh)).sum(dim=-1, keepdim=True)
    dlim = 1.0e-6 / em.square().clamp_min(1.0)
    w = _SRS / em / torch.maximum(qdqpe, dlim)
    p = unit_xyz + w * torch.cross(unit_xyz, torch.cross(eh, unit_xyz, dim=-1), dim=-1)
    # Stellar aberration (eraAb), velocity in units of c.
    v2 = (vel * vel).sum(dim=-1, keepdim=True)
    bm1 = torch.sqrt(1.0 - v2)
    pdv = (p * vel).sum(dim=-1, keepdim=True)
    w1 = 1.0 + pdv / (1.0 + bm1)
    w2 = _SRS / em
    shifted = p * bm1 + w1 * vel + w2 * (vel - pdv * p)
    return shifted / torch.linalg.vector_norm(shifted, dim=-1, keepdim=True)


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
    """Apply atmospheric refraction to altitude in degrees."""
    p = torch.as_tensor(pressure_mbar, dtype=alt_deg.dtype, device=alt_deg.device)
    t = torch.as_tensor(temperature_c, dtype=alt_deg.dtype, device=alt_deg.device)
    return alt_deg + refraction_arcmin(alt_deg, pressure_mbar=p, temperature_c=t) / 60.0


def refco(
    pressure_mbar: Tensor | float,
    temperature_c: Tensor | float,
    relative_humidity: Tensor | float = 0.0,
    wavelength_um: Tensor | float = 0.55,
    *,
    like: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """ERFA ``refco``: constants ``(A, B)`` of ``dZ = A tan z + B tan^3 z``.

    ``pressure_mbar`` (hPa), ``temperature_c`` (deg C), ``relative_humidity``
    (0..1) and ``wavelength_um`` (> 100 selects the radio formula). Inputs are
    clamped as in ERFA (T -150..200 C, p 0..10000 hPa, rh 0..1, wl >= 0.1 um).
    """
    ref = like if like is not None else torch.zeros((), dtype=torch.float64)
    kw = {"dtype": ref.dtype, "device": ref.device}
    pr = torch.as_tensor(pressure_mbar, **kw).clamp(0.0, 10000.0)
    tc = torch.as_tensor(temperature_c, **kw).clamp(-150.0, 200.0)
    rh = torch.as_tensor(relative_humidity, **kw).clamp(0.0, 1.0)
    wl = torch.as_tensor(wavelength_um, **kw)
    optic = wl <= 100.0
    wl = wl.clamp_min(0.1)
    tk = tc + 273.15
    safe_p = torch.where(pr > 0.0, pr, torch.ones_like(pr))
    ps = torch.pow(10.0, (0.7859 + 0.03477 * tc) / (1.0 + 0.00412 * tc)) * (
        1.0 + pr * (4.5e-6 + 6.0e-10 * tc * tc)
    )
    pw = torch.where(pr > 0.0, rh * ps / (1.0 - (1.0 - rh) * ps / safe_p), 0.0 * pr)
    wlsq = wl * wl
    gamma_opt = (
        (77.53484e-6 + (4.39108e-7 + 3.666e-9 / wlsq) / wlsq) * pr - 11.2684e-6 * pw
    ) / tk
    gamma_rad = (77.6890e-6 * pr - (6.3938e-6 - 0.375463 / tk) * pw) / tk
    gamma = torch.where(optic, gamma_opt, gamma_rad)
    beta = 4.4474e-6 * tk
    beta = torch.where(optic, beta, beta - 0.0074 * pw * beta)
    return gamma * (1.0 - beta), -gamma * (beta - gamma / 2.0)


def apply_refco_enu(
    east: Tensor, north: Tensor, up: Tensor, refa: Tensor, refb: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Refract a topocentric (east, north, up) direction as ERFA ``atioq``.

    The in-vacuo zenith distance ``z`` is lowered by
    ``dZ = (A + B tan^2 z) tan z / (1 + (A + 3 B tan^2 z) / cos^2 z)``, the
    one-step Newton solution of ``dZ = A tan z' + B tan^3 z'`` at the refracted
    ``z' = z - dZ`` that ERFA uses, applied as a rotation toward the zenith with
    ERFA's guards (horizontal radius >= 1e-6, ``cos z`` >= 0.05).
    """
    r = torch.hypot(east, north).clamp_min(1.0e-6)
    z = up.clamp_min(0.05)
    tz = r / z
    w = refb * tz * tz
    delta = (refa + w) * tz / (1.0 + (refa + 3.0 * w) / (z * z))
    cosdel = 1.0 - 0.5 * delta * delta
    f = cosdel - delta * z / r
    return east * f, north * f, cosdel * up + delta * r


def _air_refractive_index_minus_one(
    wavelength_angstrom: Tensor,
    *,
    pressure_mbar: Tensor,
    temperature_c: Tensor,
) -> Tensor:
    """Edlen (1966) / Filippenko (1982) air refractive index (n - 1)."""
    wave_um = wavelength_angstrom.clamp_min(1000.0) * 1.0e-4
    sigma2 = (1.0 / wave_um).square()
    n0_ppm = (
        64.328
        + 29498.1 / (146.0 - sigma2).clamp_min(1.0)
        + 255.4 / (41.0 - sigma2).clamp_min(1.0)
    )
    scale = (pressure_mbar / 1013.25) * (288.15 / (273.15 + temperature_c))
    return n0_ppm * 1.0e-6 * scale


def dcr_offset(
    wavelength_angstrom: Tensor | float,
    zenith_deg: Tensor | float,
    parallactic_deg: Tensor | float = 0.0,
    *,
    ref_wavelength_angstrom: float = 5000.0,
    pressure_mbar: float = 780.0,
    temperature_c: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """Compute Differential Chromatic Refraction (DCR) tangent-plane offset ``(d_east_deg, d_north_deg)``.

    Returns the apparent chromatic shift toward zenith at ``wavelength_angstrom``
    relative to ``ref_wavelength_angstrom`` in degrees along local East and North.
    """
    ref_tensor = (
        wavelength_angstrom
        if isinstance(wavelength_angstrom, Tensor)
        else (
            zenith_deg
            if isinstance(zenith_deg, Tensor)
            else torch.as_tensor(wavelength_angstrom, dtype=torch.float64)
        )
    )
    dt = torch.float32 if ref_tensor.device.type == "mps" else torch.float64
    dev = ref_tensor.device
    lam = torch.as_tensor(wavelength_angstrom, dtype=dt, device=dev)
    lam_ref = torch.as_tensor(ref_wavelength_angstrom, dtype=dt, device=dev)
    zen = torch.as_tensor(zenith_deg, dtype=dt, device=dev).clamp(0.0, 89.0)
    pa = torch.as_tensor(parallactic_deg, dtype=dt, device=dev)
    p = torch.as_tensor(pressure_mbar, dtype=dt, device=dev)
    t = torch.as_tensor(temperature_c, dtype=dt, device=dev)

    dn = _air_refractive_index_minus_one(
        lam, pressure_mbar=p, temperature_c=t
    ) - _air_refractive_index_minus_one(lam_ref, pressure_mbar=p, temperature_c=t)
    dr_deg = torch.rad2deg(dn * torch.tan(torch.deg2rad(zen)))
    pa_rad = torch.deg2rad(pa)
    d_east_deg = dr_deg * torch.sin(pa_rad)
    d_north_deg = dr_deg * torch.cos(pa_rad)
    return d_east_deg, d_north_deg


def apply_dcr(
    ra_deg: Tensor,
    dec_deg: Tensor,
    wavelength_angstrom: Tensor | float,
    zenith_deg: Tensor | float,
    parallactic_deg: Tensor | float = 0.0,
    *,
    ref_wavelength_angstrom: float = 5000.0,
    pressure_mbar: float = 780.0,
    temperature_c: float = 10.0,
    inverse: bool = False,
) -> tuple[Tensor, Tensor]:
    """Apply or remove Differential Chromatic Refraction on ``(ra_deg, dec_deg)``."""
    d_east, d_north = dcr_offset(
        wavelength_angstrom,
        zenith_deg,
        parallactic_deg,
        ref_wavelength_angstrom=ref_wavelength_angstrom,
        pressure_mbar=pressure_mbar,
        temperature_c=temperature_c,
    )
    sign = -1.0 if inverse else 1.0
    cos_dec = torch.cos(torch.deg2rad(dec_deg)).clamp_min(1e-12)
    ra_out = torch.remainder(
        ra_deg + sign * d_east.to(dtype=ra_deg.dtype, device=ra_deg.device) / cos_dec,
        360.0,
    )
    dec_out = (
        dec_deg + sign * d_north.to(dtype=dec_deg.dtype, device=dec_deg.device)
    ).clamp(-90.0, 90.0)
    return ra_out, dec_out


def apply_ephemeris_offset(
    ra_deg: Tensor,
    dec_deg: Tensor,
    *,
    mjd_obs: float,
    reference_mjd: float,
    tracking_rate: tuple[float, float] | None = None,
    dra_cosdec_arcsec_per_hr: float = 0.0,
    ddec_arcsec_per_hr: float = 0.0,
    inverse: bool = False,
) -> tuple[Tensor, Tensor]:
    """Shift sky coordinates into (or out of) a moving target's comoving frame.

    Parameters
    ----------
    tracking_rate
        ``(dra_cosdec_arcsec_per_hr, ddec_arcsec_per_hr)`` non-sidereal rate.
    inverse
        If ``False`` (default), maps sidereal sky coordinates at ``mjd_obs`` into
        the comoving target frame at ``reference_mjd`` (subtracting target motion).
        If ``True``, maps comoving coordinates back to sidereal coordinates at ``mjd_obs``.
    """
    if tracking_rate is not None:
        rate_ra, rate_dec = float(tracking_rate[0]), float(tracking_rate[1])
    else:
        rate_ra, rate_dec = float(dra_cosdec_arcsec_per_hr), float(ddec_arcsec_per_hr)
    dt_hours = (float(mjd_obs) - float(reference_mjd)) * 24.0
    dra_cosdec_deg = rate_ra * dt_hours / 3600.0
    ddec_deg = rate_dec * dt_hours / 3600.0
    sign = 1.0 if inverse else -1.0
    cos_dec = torch.cos(torch.deg2rad(dec_deg)).clamp_min(1e-12)
    ra_out = torch.remainder(ra_deg + sign * (dra_cosdec_deg / cos_dec), 360.0)
    dec_out = (dec_deg + sign * ddec_deg).clamp(-90.0, 90.0)
    return ra_out, dec_out
