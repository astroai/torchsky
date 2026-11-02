"""Observed-place transforms with explicit Earth orientation.

This module is separate from geometric frames. Earth orientation is never
inferred. One path serves every option: ICRS → (aberration) → celestial-to-
terrestrial rotation → geodetic AltAz. ``pn='iau2006'`` is Fukushima-Williams
precession-bias plus full IAU 2000A nutation (IAU 2006/P03 adjustments) on
the CIO/ERA path (GCRS → CIRS → ERA). ``pn='none'`` skips precession-nutation
and rotates by GMST (equinox hour angle), so it is only an approximation
(~0.3 deg in 2024) to a true observed place. Polar motion, when supplied, is
applied after the Earth rotation in both models.

All scalar time quantities (UT1, TT, ERA, GMST, PN/CIO matrices, Earth
velocity) are computed in float64 on the CPU regardless of the coordinate
device/dtype (MPS float32 MJD would lose ~0.4 day-fraction digits); only the
resulting 3x3 matrices and velocities move to the working device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from .catalog_state import CatalogState
from .frames import ARCSEC_TO_RAD, D2R, R2D, _as_tensor, _work_dtype, _wrap360
from .iau2006 import (
    _rx,
    _ry,
    _rz,
    apparent_gcrs_unit,
    apply_refco_enu,
    apply_refraction,
    cirs_matrix,
    npb_matrix,
    refco,
    tt_mjd_from_utc,
)

SECONDS_PER_DAY = 86400.0
MJD_J2000 = 51544.5
PNModel = Literal["none", "iau2006"]
RefractionModel = Literal["refco", "saemundsson"]
_EARTH_ROTATION_RAD_S = 1.00273781191135448 * 2.0 * 3.141592653589793 / SECONDS_PER_DAY


@dataclass(frozen=True)
class EarthOrientation:
    """IERS Earth orientation. ``dut1_s`` is required.

    Polar motion ``xp_arcsec`` / ``yp_arcsec`` is optional. If either is set,
    both are applied (missing member treated as 0) via SOFA ``pom00`` with
    TIO locator ``s' = 0``. ``tt_minus_utc_s`` is required for ``pn='iau2006'``
    (TT−UTC is never inferred from leap-second tables).
    """

    dut1_s: Tensor

    xp_arcsec: Tensor | None = None
    yp_arcsec: Tensor | None = None
    tt_minus_utc_s: Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dut1_s", _as_tensor(self.dut1_s))
        if self.xp_arcsec is not None:
            object.__setattr__(self, "xp_arcsec", _as_tensor(self.xp_arcsec))
        if self.yp_arcsec is not None:
            object.__setattr__(self, "yp_arcsec", _as_tensor(self.yp_arcsec))
        if self.tt_minus_utc_s is not None:
            object.__setattr__(self, "tt_minus_utc_s", _as_tensor(self.tt_minus_utc_s))


@dataclass(frozen=True)
class Site:
    """Geodetic site. Height is WGS84 ellipsoidal height (metres) on the ITRS path.

    ``relative_humidity`` (0..1) and ``wavelength_um`` (> 100 selects the radio
    formula) only enter the ERFA ``refco`` refraction model.
    """

    lon_deg: Tensor
    lat_deg: Tensor
    height_m: Tensor | float = 0.0
    pressure_mbar: Tensor | float | None = None
    temperature_c: Tensor | float | None = None
    relative_humidity: Tensor | float = 0.0
    wavelength_um: Tensor | float = 0.55

    def __post_init__(self) -> None:
        object.__setattr__(self, "lon_deg", _as_tensor(self.lon_deg))
        object.__setattr__(self, "lat_deg", _as_tensor(self.lat_deg))
        object.__setattr__(self, "height_m", _as_tensor(self.height_m))
        if self.pressure_mbar is not None:
            object.__setattr__(self, "pressure_mbar", _as_tensor(self.pressure_mbar))
        if self.temperature_c is not None:
            object.__setattr__(self, "temperature_c", _as_tensor(self.temperature_c))


_WGS84_A_M = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _wgs84_itrs_m(lon_rad: Tensor, lat_rad: Tensor, height_m: Tensor) -> Tensor:
    """WGS84 geodetic to ITRS metres. Stellar AltAz does not use the vector."""
    sin_lat = torch.sin(lat_rad)
    n = _WGS84_A_M / torch.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    cos_lat = torch.cos(lat_rad)
    return torch.stack(
        (
            (n + height_m) * cos_lat * torch.cos(lon_rad),
            (n + height_m) * cos_lat * torch.sin(lon_rad),
            (n * (1.0 - _WGS84_E2) + height_m) * sin_lat,
        ),
        dim=-1,
    )


def _era_rad_from_ut1_mjd(ut1_mjd: Tensor) -> Tensor:
    """IAU 2000 Earth rotation angle (SOFA era00), radians."""
    tu = ut1_mjd - MJD_J2000
    frac = 0.5 + torch.remainder(ut1_mjd, 1.0)
    era = 2.0 * torch.pi * (0.7790572732640 + 0.00273781191135448 * tu + frac)
    return torch.remainder(era, 2.0 * torch.pi)


def _polar_motion_matrix(xp_arcsec: Tensor, yp_arcsec: Tensor) -> Tensor:
    """SOFA pom00 with TIO locator s' = 0: Rz(0) Ry(-xp) Rx(-yp)."""
    xp, yp = torch.broadcast_tensors(
        xp_arcsec * ARCSEC_TO_RAD, yp_arcsec * ARCSEC_TO_RAD
    )
    return _ry(-xp) @ _rx(-yp)


def _geodetic_normal_east_north(
    lon_rad: Tensor, lat_rad: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    sin_lon = torch.sin(lon_rad)
    cos_lon = torch.cos(lon_rad)
    sin_lat = torch.sin(lat_rad)
    cos_lat = torch.cos(lat_rad)
    east = torch.stack((-sin_lon, cos_lon, torch.zeros_like(sin_lon)), dim=-1)
    north = torch.stack((-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat), dim=-1)
    up = torch.stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat), dim=-1)
    return east, north, up


def _unit_xyz_from_lonlat_deg(lon_deg: Tensor, lat_deg: Tensor) -> Tensor:
    lon = lon_deg * D2R
    lat = lat_deg * D2R
    clat = torch.cos(lat)
    return torch.stack(
        (clat * torch.cos(lon), clat * torch.sin(lon), torch.sin(lat)), dim=-1
    )


def _gmst_deg_from_ut1_mjd(ut1_mjd: Tensor) -> Tensor:
    """IAU 1982 GMST from UT1 MJD, in degrees."""
    du = ut1_mjd - MJD_J2000
    t = du / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * du
        + 0.000387933 * t * t
        - t * t * t / 38710000.0
    )
    return _wrap360(gmst)


def _time64(value: Any) -> Tensor:
    """Time/EOP scalar as float64 on CPU (never the accelerator working dtype)."""
    if isinstance(value, Tensor):
        # Two steps: MPS ``.to(device="cpu", dtype=float64)`` returns zeros.
        return value.cpu().to(torch.float64)
    return torch.as_tensor(value, dtype=torch.float64)


def _resolve_pn(pn: PNModel | None, eop: EarthOrientation) -> PNModel:
    """``None`` selects the astropy-matching IAU 2006 model when TT-UTC is known."""
    if pn is None:
        return "iau2006" if eop.tt_minus_utc_s is not None else "none"
    if pn not in ("none", "iau2006"):
        raise ValueError("pn must be 'none' or 'iau2006'")
    return pn


def _tt64(utc64: Tensor, eop: EarthOrientation, why: str) -> Tensor:
    if eop.tt_minus_utc_s is None:
        raise ValueError(f"{why} requires eop.tt_minus_utc_s")
    return tt_mjd_from_utc(utc64, _time64(eop.tt_minus_utc_s))


def _ut1_64(utc64: Tensor, eop: EarthOrientation) -> Tensor:
    return utc64 + _time64(eop.dut1_s) / SECONDS_PER_DAY


def _gcrs_to_tirs_and_pom(
    utc64: Tensor, eop: EarthOrientation, pn: PNModel
) -> tuple[Tensor, Tensor | None]:
    """GCRS -> TIRS rotation and the TIRS -> ITRS polar-motion matrix (or None)."""
    ut1 = _ut1_64(utc64, eop)
    if pn == "iau2006":
        tt = _tt64(utc64, eop, "pn='iau2006'")
        rot = _rz(_era_rad_from_ut1_mjd(ut1)) @ cirs_matrix(tt)
    else:
        rot = _rz(_gmst_deg_from_ut1_mjd(ut1) * D2R)
    pom = None
    if eop.xp_arcsec is not None or eop.yp_arcsec is not None:
        z = torch.zeros((), dtype=torch.float64)
        xp = z if eop.xp_arcsec is None else _time64(eop.xp_arcsec)
        yp = z if eop.yp_arcsec is None else _time64(eop.yp_arcsec)
        pom = _polar_motion_matrix(xp, yp)
    return rot, pom


def _gcrs_to_terrestrial_matrix(
    utc64: Tensor, eop: EarthOrientation, pn: PNModel
) -> Tensor:
    """GCRS -> ITRS rotation, float64 on CPU."""
    rot, pom = _gcrs_to_tirs_and_pom(utc64, eop, pn)
    return rot if pom is None else pom @ rot


def _observer_pv_gcrs(
    site: Site, utc64: Tensor, eop: EarthOrientation, pn: PNModel
) -> tuple[Tensor, Tensor]:
    """Observer position (m) and velocity (m/s) relative to the geocentre in GCRS.

    ERFA ``pvtob``: the station rotates about the TIRS pole at the sidereal rate
    ``1.00273781191135448 * 2 pi / 86400`` rad/s; the velocity is carried to GCRS
    by the transpose of the GCRS -> TIRS rotation. Float64 on the CPU.
    """
    lon = _time64(site.lon_deg) * D2R
    lat = _time64(site.lat_deg) * D2R
    assert isinstance(site.height_m, Tensor)
    r_itrs = _wgs84_itrs_m(lon, lat, _time64(site.height_m))
    rot, pom = _gcrs_to_tirs_and_pom(utc64, eop, pn)
    r_tirs = (
        r_itrs
        if pom is None
        else (pom.transpose(-1, -2) @ r_itrs.unsqueeze(-1)).squeeze(-1)
    )
    omega = _EARTH_ROTATION_RAD_S
    v_tirs = torch.stack(
        (
            -omega * r_tirs[..., 1],
            omega * r_tirs[..., 0],
            torch.zeros_like(r_tirs[..., 2]),
        ),
        dim=-1,
    )
    rt = rot.transpose(-1, -2)
    return (rt @ r_tirs.unsqueeze(-1)).squeeze(-1), (rt @ v_tirs.unsqueeze(-1)).squeeze(
        -1
    )


def _apply_matrix(m: Tensor, v: Tensor) -> Tensor:
    m = m.to(device=v.device, dtype=v.dtype)
    return (m @ v.unsqueeze(-1)).squeeze(-1)


def observed_altaz(
    state: CatalogState,
    *,
    obstime_mjd_utc: Any,
    site: Site,
    eop: EarthOrientation,
    pn: PNModel | None = None,
    aberration: bool = False,
    refraction: bool = False,
    refraction_model: RefractionModel = "refco",
) -> tuple[Tensor, Tensor]:
    """Return ``(az_deg, alt_deg)`` from north toward east.

    ``obstime_mjd_utc`` is UTC MJD. UT1 = UTC + DUT1. Polar motion is applied
    when ``xp``/``yp`` are set.

    ``pn=None`` (default) selects ``'iau2006'`` when ``eop.tt_minus_utc_s`` is
    given, which matches astropy ``AltAz`` (with ``aberration=True``) to about
    1 arcsec, and ``'none'`` otherwise. ``pn='iau2006'`` applies
    Fukushima-Williams bias-precession and IAU 2000A nutation via the CIRS
    matrix (requires ``eop.tt_minus_utc_s``). ``pn='none'`` omits
    precession-nutation and uses GMST; it is approximate (~0.3 deg off) and is
    continuous in ``xp``/``yp`` (zero polar motion equals no polar motion).
    ``aberration=True`` converts the astrometric ICRS direction to the apparent
    place as ERFA ``atciq`` does: solar light deflection plus annual and diurnal
    aberration (the observer's Earth-rotation velocity), with the ``epv00`` Earth
    ephemeris. Without refraction the result agrees with ``erfa.atco13`` to
    < 1 mas (residual: TIO locator s' = 0, observer offset in the solar
    deflection, TT used for TDB).
    ``refraction=True`` applies the ERFA ``refco`` model (``A tan z + B tan^3 z``,
    honouring ``site.relative_humidity`` and ``site.wavelength_um``) to the
    topocentric direction as ``atioq`` does; ``refraction_model="saemundsson"``
    selects the Saemundsson/Meeus altitude formula instead. Both are off unless
    requested and need ``site.pressure_mbar`` and ``site.temperature_c``.

    Time-only quantities are evaluated in float64 on CPU on every device; only
    the final rotation matrices move to the coordinate device (CUDA float64,
    MPS float32). Gradients flow through the coordinates. The kernels are
    ATen, not fused C++.
    """
    if eop is None:
        raise ValueError("EarthOrientation is required; IERS DUT1 is never inferred")
    pn = _resolve_pn(pn, eop)
    icrs = state if state.frame == "ICRS" else state.to_frame("ICRS")
    device = icrs.lon.device
    dtype = _work_dtype(device)
    v_itrs = _icrs_unit_to_itrs(
        icrs, obstime_mjd_utc, eop, pn=pn, aberration=aberration, site=site
    )
    lon_rad = site.lon_deg.to(device=device, dtype=dtype) * D2R
    lat_rad = site.lat_deg.to(device=device, dtype=dtype) * D2R
    assert isinstance(site.height_m, Tensor)  # normalised by the site's __post_init__
    height = site.height_m.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(height).all()):
        raise ValueError("site.height_m must be finite WGS84 ellipsoidal height")
    east, north, up = _geodetic_normal_east_north(lon_rad, lat_rad)
    if refraction:
        if site.pressure_mbar is None or site.temperature_c is None:
            raise ValueError(
                "refraction=True requires site.pressure_mbar and site.temperature_c"
            )
        if refraction_model not in ("refco", "saemundsson"):
            raise ValueError("refraction_model must be 'refco' or 'saemundsson'")
    e_c = (v_itrs * east).sum(dim=-1)
    n_c = (v_itrs * north).sum(dim=-1)
    u_c = (v_itrs * up).sum(dim=-1)
    if refraction and refraction_model == "refco":
        assert site.pressure_mbar is not None and site.temperature_c is not None
        refa, refb = refco(
            site.pressure_mbar,
            site.temperature_c,
            site.relative_humidity,
            site.wavelength_um,
            like=u_c,
        )
        e_c, n_c, u_c = apply_refco_enu(e_c, n_c, u_c, refa, refb)
    alt = torch.asin(torch.clamp(u_c, -1.0, 1.0)) * R2D
    az = _wrap360(torch.atan2(e_c, n_c) * R2D)
    if refraction and refraction_model == "saemundsson":
        assert site.pressure_mbar is not None and site.temperature_c is not None
        alt = apply_refraction(
            alt,
            pressure_mbar=site.pressure_mbar,
            temperature_c=site.temperature_c,
        )
    return az, alt


def _icrs_to_itrs_xyz(
    state: CatalogState,
    *,
    obstime_mjd_utc: Any,
    eop: EarthOrientation,
    pn: PNModel | None = None,
    aberration: bool = False,
) -> Tensor:
    """Unit ICRS direction to ITRS Cartesian (dimensionless)."""
    icrs = state if state.frame == "ICRS" else state.to_frame("ICRS")
    return _icrs_unit_to_itrs(
        icrs, obstime_mjd_utc, eop, pn=_resolve_pn(pn, eop), aberration=aberration
    )


def _itrs_xyz_to_lonlat_deg(xyz: Tensor) -> tuple[Tensor, Tensor]:
    """ITRS unit/vector to geocentric lon/lat in degrees."""
    norm = torch.linalg.vector_norm(xyz, dim=-1).clamp_min(1.0e-15)
    x = xyz[..., 0] / norm
    y = xyz[..., 1] / norm
    z = xyz[..., 2] / norm
    lon = _wrap360(torch.atan2(y, x) * R2D)
    lat = torch.asin(torch.clamp(z, -1.0, 1.0)) * R2D
    return lon, lat


def _icrs_to_gcrs_unit(
    icrs: CatalogState,
    obstime: Any,
    eop: EarthOrientation,
    *,
    aberration: bool,
    site: Site | None = None,
    pn: PNModel | None = None,
) -> Tensor:
    device = icrs.lon.device
    dtype = _work_dtype(device)
    v = _unit_xyz_from_lonlat_deg(icrs.lon.to(dtype=dtype), icrs.lat.to(dtype=dtype))
    if aberration:
        if eop.tt_minus_utc_s is None:
            raise ValueError("aberration requires eop.tt_minus_utc_s")
        utc64 = _time64(obstime)
        tt = _tt64(utc64, eop, "aberration")
        pobs = vobs = None
        if site is not None:
            pobs, vobs = _observer_pv_gcrs(site, utc64, eop, _resolve_pn(pn, eop))
        v = apparent_gcrs_unit(v, tt, vobs, pobs)
    return v


def icrs_to_gcrs_xyz(
    state: CatalogState,
    *,
    obstime_mjd_utc: Any,
    eop: EarthOrientation,
    aberration: bool = False,
) -> Tensor:
    """ICRS direction to GCRS. Optional apparent place (deflection + aberration); no PN."""
    icrs = state if state.frame == "ICRS" else state.to_frame("ICRS")
    return _icrs_to_gcrs_unit(icrs, obstime_mjd_utc, eop, aberration=aberration)


def icrs_to_cirs_xyz(
    state: CatalogState,
    *,
    obstime_mjd_utc: Any,
    eop: EarthOrientation,
    aberration: bool = False,
) -> Tensor:
    """GCRS then IAU 2006 CIRS matrix (SOFA c2i06a)."""
    icrs = state if state.frame == "ICRS" else state.to_frame("ICRS")
    v = _icrs_to_gcrs_unit(icrs, obstime_mjd_utc, eop, aberration=aberration)
    tt = _tt64(_time64(obstime_mjd_utc), eop, "CIRS")
    return _apply_matrix(cirs_matrix(tt), v)


def icrs_to_tete_xyz(
    state: CatalogState,
    *,
    obstime_mjd_utc: Any,
    eop: EarthOrientation,
    aberration: bool = False,
) -> Tensor:
    """GCRS then IAU 2006 true equator/equinox of date (SOFA pnm06a)."""
    icrs = state if state.frame == "ICRS" else state.to_frame("ICRS")
    v = _icrs_to_gcrs_unit(icrs, obstime_mjd_utc, eop, aberration=aberration)
    tt = _tt64(_time64(obstime_mjd_utc), eop, "TETE")
    return _apply_matrix(npb_matrix(tt), v)


def _icrs_unit_to_itrs(
    icrs: CatalogState,
    obstime: Any,
    eop: EarthOrientation,
    *,
    pn: PNModel,
    aberration: bool,
    site: Site | None = None,
) -> Tensor:
    v = _icrs_to_gcrs_unit(icrs, obstime, eop, aberration=aberration, site=site, pn=pn)
    rot = _gcrs_to_terrestrial_matrix(_time64(obstime), eop, pn)
    return _apply_matrix(rot, v)
