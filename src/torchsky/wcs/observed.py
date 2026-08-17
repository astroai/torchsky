"""Observed-place transforms with explicit Earth orientation.

This module is separate from geometric frames. Earth orientation is never
inferred. The default path is classical DUT1 hour-angle AltAz on CPU float64,
not IAU 2006 CIRS/TETE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .catalog_state import CatalogState
from .frames import D2R, R2D, _as_tensor, _wrap360

SECONDS_PER_DAY = 86400.0
MJD_J2000 = 51544.5


@dataclass(frozen=True)
class EarthOrientation:
    """IERS Earth orientation. ``dut1_s`` is required; polar motion is not yet applied."""

    dut1_s: Tensor

    xp_arcsec: Tensor | None = None
    yp_arcsec: Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dut1_s", _as_tensor(self.dut1_s))
        if self.xp_arcsec is not None or self.yp_arcsec is not None:
            raise NotImplementedError(
                "polar motion (xp/yp) is not part of the classical DUT1 AltAz "
                "path; omit xp and yp, or wait for a CIRS implementation"
            )


@dataclass(frozen=True)
class Site:
    """Geodetic site. Height is stored for provenance and is unused in HA/AltAz."""

    lon_deg: Tensor
    lat_deg: Tensor
    height_m: Tensor | float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lon_deg", _as_tensor(self.lon_deg))
        object.__setattr__(self, "lat_deg", _as_tensor(self.lat_deg))
        object.__setattr__(self, "height_m", _as_tensor(self.height_m))


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
) -> tuple[Tensor, Tensor]:
    """Return ``(az_deg, alt_deg)`` from north toward east.

    ``obstime_mjd_utc`` is UTC MJD. UT1 = UTC + DUT1. This is not apparent
    place: no aberration, refraction, or polar motion.
    """
    if eop is None:
        raise ValueError("EarthOrientation is required; IERS DUT1 is never inferred")
    icrs = state if state.frame == "ICRS" else state.to_frame("ICRS")
    utc = _as_tensor(obstime_mjd_utc).to(device=icrs.lon.device)
    dut1 = eop.dut1_s.to(device=icrs.lon.device, dtype=utc.dtype)
    ut1 = utc + dut1 / SECONDS_PER_DAY
    gmst = _gmst_deg_from_ut1_mjd(ut1)
    last = _wrap360(gmst + site.lon_deg.to(device=icrs.lon.device))
    ha_rad = (_wrap360(last - icrs.lon) - 180.0) * D2R + torch.pi
    dec_rad = icrs.lat * D2R
    lat_rad = site.lat_deg.to(device=icrs.lon.device) * D2R
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
