"""Thin coordinate transform facade. This is not a SkyCoord subclass."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from .catalog_state import CatalogState
from .observed import (
    EarthOrientation,
    PNModel,
    Site,
    icrs_to_cirs_xyz,
    icrs_to_gcrs_xyz,
    icrs_to_itrs_xyz,
    icrs_to_tete_xyz,
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

    Geometric frames use ``CatalogState.to_frame``. ``ALTAZ``, ``ITRS``,
    ``GCRS``, ``CIRS``, and ``TETE`` require ``obstime_mjd_utc`` and ``eop``.
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
    if dest in {"ITRS", "GCRS", "CIRS", "TETE"}:
        if obstime_mjd_utc is None or eop is None:
            raise ValueError(f"{dest} requires obstime_mjd_utc and eop")
        if dest == "ITRS":
            xyz = icrs_to_itrs_xyz(
                state,
                obstime_mjd_utc=obstime_mjd_utc,
                eop=eop,
                pn=pn,
                aberration=aberration,
            )
        elif dest == "GCRS":
            xyz = icrs_to_gcrs_xyz(
                state,
                obstime_mjd_utc=obstime_mjd_utc,
                eop=eop,
                aberration=aberration,
            )
        elif dest == "CIRS":
            xyz = icrs_to_cirs_xyz(
                state,
                obstime_mjd_utc=obstime_mjd_utc,
                eop=eop,
                aberration=aberration,
            )
        else:
            xyz = icrs_to_tete_xyz(
                state,
                obstime_mjd_utc=obstime_mjd_utc,
                eop=eop,
                aberration=aberration,
            )
        return itrs_xyz_to_lonlat_deg(xyz)
    out = state.to_frame(to_frame)
    return out.lon, out.lat
