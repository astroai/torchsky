"""Thin coordinate transform facade. This is not a SkyCoord subclass."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from .catalog_state import CatalogState
from .observed import (
    EarthOrientation,
    PNModel,
    Site,
    icrs_to_itrs_xyz,
    itrs_xyz_to_lonlat_deg,
    observed_altaz,
)


def transform(
    lon: Any,
    lat: Any,
    *,
    from_frame: str,
    to_frame: str,
    obstime_mjd_utc: Any | None = None,
    site: Site | None = None,
    eop: EarthOrientation | None = None,
    pn: PNModel = "none",
    aberration: bool = False,
    refraction: bool = False,
) -> tuple[Tensor, Tensor]:
    """Rotate or observe ``(lon, lat)`` tensors. Returns ``(lon_out, lat_out)``.

    Geometric frames use ``CatalogState.to_frame``. ``ALTAZ`` and ``ITRS``
    require ``obstime_mjd_utc``, ``site`` (AltAz), and ``eop``.
    """
    state = CatalogState(lon, lat, frame=from_frame)
    dest = str(to_frame).strip().upper().replace(" ", "")
    if dest in {"ALTAZ", "HORIZON"}:
        if obstime_mjd_utc is None or site is None or eop is None:
            raise ValueError("ALTAZ requires obstime_mjd_utc, site, and eop")
        return observed_altaz(
            state,
            obstime_mjd_utc=obstime_mjd_utc,
            site=site,
            eop=eop,
            pn=pn,
            aberration=aberration,
            refraction=refraction,
        )
    if dest == "ITRS":
        if obstime_mjd_utc is None or eop is None:
            raise ValueError("ITRS requires obstime_mjd_utc and eop")
        xyz = icrs_to_itrs_xyz(
            state,
            obstime_mjd_utc=obstime_mjd_utc,
            eop=eop,
            pn=pn,
            aberration=aberration,
        )
        return itrs_xyz_to_lonlat_deg(xyz)
    out = state.to_frame(to_frame)
    return out.lon, out.lat
