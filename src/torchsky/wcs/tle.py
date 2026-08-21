"""Optional SGP4/TLE satellite state. Not imported by ``torchsky.wcs``."""

from __future__ import annotations

from datetime import datetime

import torch

from .catalog_state import CatalogState
from .frames import unit_to_radec

try:
    from sgp4.api import Satrec, jday
except ImportError:  # pragma: no cover - optional extra
    Satrec = None
    jday = None


def catalog_state_from_tle(
    line1: str,
    line2: str,
    utc: datetime,
) -> CatalogState:
    """Propagate a TLE to UTC and return TEME spherical coordinates.

    Requires the ``tle`` extra (``sgp4``). The frame tag is ``TEME``; this is
    not ICRS and is not converted by ``CatalogState.to_frame``.
    """
    if Satrec is None or jday is None:
        raise ImportError("catalog_state_from_tle requires sgp4; install torchsky[tle]")

    year = int(utc.year)
    month = int(utc.month)
    day = int(utc.day)
    hour = int(utc.hour)
    minute = int(utc.minute)
    second = float(utc.second) + float(getattr(utc, "microsecond", 0)) * 1.0e-6
    jd, fr = jday(year, month, day, hour, minute, second)
    sat = Satrec.twoline2rv(line1, line2)
    error, r_km, _v = sat.sgp4(jd, fr)
    if int(error) != 0:
        raise ValueError(f"SGP4 failed with error code {error}")
    vec = torch.as_tensor(r_km, dtype=torch.float64)
    lon, lat = unit_to_radec(vec)
    return CatalogState(lon=lon, lat=lat, frame="TEME", epoch=2000.0)
