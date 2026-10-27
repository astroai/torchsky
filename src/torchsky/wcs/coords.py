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
    ``ECLIPTIC_OF_DATE`` uses TT MJD (``obstime_mjd_utc`` is treated as TT
    unless you pass a TT value through ``CatalogState.to_frame``).

    Args:
        lon: Longitude values in degrees.
        lat: Latitude values in degrees.
        from_frame: Input coordinate-frame name.
        to_frame: Output coordinate-frame name.
        obstime_mjd_utc: Observation time in UTC MJD, or TT MJD for ecliptic-of-date.
        site: Observing site required for altitude-azimuth output.
        eop: Earth-orientation parameters required for observed frames.
        pn: Precession-nutation model.
        aberration: Apply annual aberration for supported observed frames.
        refraction: Apply atmospheric refraction to altitude-azimuth output.

    Returns:
        Output longitude and latitude tensors in degrees.

    Examples:
        >>> import torch
        >>> from torchsky.coords import transform
        >>> lon, lat = transform(torch.tensor([0.0]), torch.tensor([0.0]), from_frame="ICRS", to_frame="GALACTIC")
        >>> lon.shape == lat.shape
        True
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
    if dest == "ECLIPTIC_OF_DATE":
        if obstime_mjd_utc is None:
            raise ValueError("ECLIPTIC_OF_DATE requires obstime_mjd_utc (TT MJD)")
        out = state.to_frame("ECLIPTIC_OF_DATE", tt_mjd=obstime_mjd_utc)
        return out.lon, out.lat
    out = state.to_frame(to_frame)
    return out.lon, out.lat
