"""Observed-place transforms with explicit Earth orientation.

This module is separate from geometric frames. Earth orientation is never
inferred. Without polar motion the path is classical DUT1 hour-angle AltAz.
With ``xp``/``yp`` supplied, the path is ICRS → (optional IAU 2006 PN) → ERA →
polar-motion ITRS → geodetic AltAz. ``pn='iau2006'`` is a documented CIRS
subset, not the full IAU 2006/2000A series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from .catalog_state import CatalogState
from .frames import ARCSEC_TO_RAD, D2R, R2D, _as_tensor, _work_dtype, _wrap360
from .iau2006 import (
    apply_refraction,
    apply_stellar_aberration,
    earth_barycentric_velocity_icrs,
    npb_matrix,
    tt_mjd_from_utc,
)

SECONDS_PER_DAY = 86400.0
MJD_J2000 = 51544.5
PNModel = Literal["none", "iau2006"]


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
    """Geodetic site. Height is WGS84 ellipsoidal height (metres) on the ITRS path."""

    lon_deg: Tensor
    lat_deg: Tensor
    height_m: Tensor | float = 0.0
    pressure_mbar: Tensor | float | None = None
    temperature_c: Tensor | float | None = None

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


def _era_rad_from_ut1_mjd(ut1_mjd: Tensor) -> Tensor:
    """IAU 2000 Earth rotation angle (SOFA era00), radians."""
    tu = ut1_mjd - MJD_J2000
    frac = 0.5 + torch.remainder(ut1_mjd, 1.0)
    era = 2.0 * torch.pi * (0.7790572732640 + 0.00273781191135448 * tu + frac)
    return torch.remainder(era, 2.0 * torch.pi)


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


def _ry(angle: Tensor) -> Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    z = torch.zeros((), dtype=angle.dtype, device=angle.device)
    o = torch.ones((), dtype=angle.dtype, device=angle.device)
    return torch.stack(
        (
            torch.stack((c, z, -s)),
            torch.stack((z, o, z)),
            torch.stack((s, z, c)),
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


def _polar_motion_matrix(xp_arcsec: Tensor, yp_arcsec: Tensor) -> Tensor:
    """SOFA pom00 with TIO locator s' = 0: Rz(0) Ry(-xp) Rx(-yp)."""
    xp = xp_arcsec * ARCSEC_TO_RAD
    yp = yp_arcsec * ARCSEC_TO_RAD
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


def observed_altaz(
    state: CatalogState,
    *,
    obstime_mjd_utc: Any,
    site: Site,
    eop: EarthOrientation,
    pn: PNModel = "none",
    aberration: bool = False,
    refraction: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return ``(az_deg, alt_deg)`` from north toward east.

    ``obstime_mjd_utc`` is UTC MJD. UT1 = UTC + DUT1. Polar motion is applied
    when ``xp``/``yp`` are set. ``pn='iau2006'`` applies Fukushima-Williams
    bias-precession and truncated IAU 2000B nutation (requires
    ``eop.tt_minus_utc_s``). Stellar aberration and Saemundsson refraction are
    off unless requested. Refraction needs ``site.pressure_mbar`` and
    ``site.temperature_c``. CUDA keeps float64; MPS uses float32. The kernels
    are ATen, not fused C++.
    """
    if eop is None:
        raise ValueError("EarthOrientation is required; IERS DUT1 is never inferred")
    if pn not in ("none", "iau2006"):
        raise ValueError("pn must be 'none' or 'iau2006'")
    icrs = state if state.frame == "ICRS" else state.to_frame("ICRS")
    device = icrs.lon.device
    dtype = _work_dtype(device)
    utc = _as_tensor(obstime_mjd_utc).to(device=device, dtype=dtype)
    dut1 = eop.dut1_s.to(device=device, dtype=dtype)
    ut1 = utc + dut1 / SECONDS_PER_DAY
    use_vector = (
        pn != "none"
        or aberration
        or eop.xp_arcsec is not None
        or eop.yp_arcsec is not None
    )
    if not use_vector:
        az, alt = _classical_gmst_altaz(icrs, ut1, site)
    else:
        az, alt = _vector_altaz(
            icrs,
            utc,
            ut1,
            site,
            eop,
            pn=pn,
            aberration=aberration,
        )
    if refraction:
        if site.pressure_mbar is None or site.temperature_c is None:
            raise ValueError(
                "refraction=True requires site.pressure_mbar and site.temperature_c"
            )
        alt = apply_refraction(
            alt,
            pressure_mbar=site.pressure_mbar,
            temperature_c=site.temperature_c,
        )
    return az, alt


def _classical_gmst_altaz(
    icrs: CatalogState, ut1: Tensor, site: Site
) -> tuple[Tensor, Tensor]:
    dtype = ut1.dtype
    device = ut1.device
    lon = icrs.lon.to(device=device, dtype=dtype)
    lat = icrs.lat.to(device=device, dtype=dtype)
    gmst = _gmst_deg_from_ut1_mjd(ut1)
    last = _wrap360(gmst + site.lon_deg.to(device=device, dtype=dtype))
    ha_rad = (_wrap360(last - lon) - 180.0) * D2R + torch.pi
    dec_rad = lat * D2R
    lat_rad = site.lat_deg.to(device=device, dtype=dtype) * D2R
    sin_alt = torch.sin(lat_rad) * torch.sin(dec_rad) + torch.cos(lat_rad) * torch.cos(
        dec_rad
    ) * torch.cos(ha_rad)
    alt = torch.asin(torch.clamp(sin_alt, -1.0, 1.0)) * R2D
    az = (
        torch.atan2(
            -torch.cos(dec_rad) * torch.sin(ha_rad),
            torch.cos(lat_rad) * torch.sin(dec_rad)
            - torch.sin(lat_rad) * torch.cos(dec_rad) * torch.cos(ha_rad),
        )
        * R2D
    )
    return _wrap360(az), alt


def icrs_to_itrs_xyz(
    state: CatalogState,
    *,
    obstime_mjd_utc: Any,
    eop: EarthOrientation,
    pn: PNModel = "none",
    aberration: bool = False,
) -> Tensor:
    """Unit ICRS direction to ITRS Cartesian (dimensionless)."""
    icrs = state if state.frame == "ICRS" else state.to_frame("ICRS")
    device = icrs.lon.device
    dtype = _work_dtype(device)
    utc = _as_tensor(obstime_mjd_utc).to(device=device, dtype=dtype)
    ut1 = utc + eop.dut1_s.to(device=device, dtype=dtype) / SECONDS_PER_DAY
    return _icrs_unit_to_itrs(icrs, utc, ut1, eop, pn=pn, aberration=aberration)


def itrs_xyz_to_lonlat_deg(xyz: Tensor) -> tuple[Tensor, Tensor]:
    """ITRS unit/vector to geocentric lon/lat in degrees."""
    norm = torch.linalg.vector_norm(xyz, dim=-1).clamp_min(1.0e-15)
    x = xyz[..., 0] / norm
    y = xyz[..., 1] / norm
    z = xyz[..., 2] / norm
    lon = _wrap360(torch.atan2(y, x) * R2D)
    lat = torch.asin(torch.clamp(z, -1.0, 1.0)) * R2D
    return lon, lat


def _icrs_unit_to_itrs(
    icrs: CatalogState,
    utc: Tensor,
    ut1: Tensor,
    eop: EarthOrientation,
    *,
    pn: PNModel,
    aberration: bool,
) -> Tensor:
    device = icrs.lon.device
    dtype = _work_dtype(device)
    v = _unit_xyz_from_lonlat_deg(icrs.lon.to(dtype=dtype), icrs.lat.to(dtype=dtype))
    if aberration:
        if eop.tt_minus_utc_s is None:
            raise ValueError("aberration requires eop.tt_minus_utc_s")
        tt = tt_mjd_from_utc(
            utc.to(dtype=dtype),
            eop.tt_minus_utc_s.to(device=device, dtype=dtype),
        )
        vel = earth_barycentric_velocity_icrs(tt)
        v = apply_stellar_aberration(v, vel)
    if pn == "iau2006":
        if eop.tt_minus_utc_s is None:
            raise ValueError("pn='iau2006' requires eop.tt_minus_utc_s")
        tt = tt_mjd_from_utc(
            utc.to(dtype=dtype),
            eop.tt_minus_utc_s.to(device=device, dtype=dtype),
        )
        v = torch.matmul(npb_matrix(tt), v.reshape(-1, 3).T).T.reshape(v.shape)
    era = _era_rad_from_ut1_mjd(ut1.to(dtype=dtype))
    xp = eop.xp_arcsec
    yp = eop.yp_arcsec
    z = torch.zeros((), dtype=dtype, device=device)
    xp_t = z if xp is None else xp.to(device=device, dtype=dtype)
    yp_t = z if yp is None else yp.to(device=device, dtype=dtype)
    r_itrs = _polar_motion_matrix(xp_t, yp_t) @ _rz(era)
    return torch.matmul(r_itrs, v.reshape(-1, 3).T).T.reshape(v.shape)


def _vector_altaz(
    icrs: CatalogState,
    utc: Tensor,
    ut1: Tensor,
    site: Site,
    eop: EarthOrientation,
    *,
    pn: PNModel,
    aberration: bool,
) -> tuple[Tensor, Tensor]:
    device = icrs.lon.device
    dtype = _work_dtype(device)
    v_itrs = _icrs_unit_to_itrs(icrs, utc, ut1, eop, pn=pn, aberration=aberration)
    lon_rad = site.lon_deg.to(device=device, dtype=dtype) * D2R
    lat_rad = site.lat_deg.to(device=device, dtype=dtype) * D2R
    height = site.height_m.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(height).all()):
        raise ValueError("site.height_m must be finite WGS84 ellipsoidal height")
    east, north, up = _geodetic_normal_east_north(lon_rad, lat_rad)
    alt = torch.asin(torch.clamp((v_itrs * up).sum(dim=-1), -1.0, 1.0)) * R2D
    az = torch.atan2(-(v_itrs * east).sum(dim=-1), (v_itrs * north).sum(dim=-1)) * R2D
    return _wrap360(az), alt
