"""Optional Astropy SkyCoord interchange. Not imported by ``torchsky.wcs``.

Astropy is loaded inside the converters so importing this module does not
pull Astropy into the default torchsky path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .catalog_state import CatalogState
from .frames import canonical_celestial_frame

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord


def catalog_state_from_astropy(coord: SkyCoord) -> CatalogState:
    """Copy lon/lat (and PM/parallax/RV when present) from a SkyCoord."""
    import astropy.units as u
    from astropy.units import UnitsError

    icrs = coord.icrs
    kwargs: dict[str, object] = {
        "lon": torch.as_tensor(icrs.ra.deg, dtype=torch.float64),
        "lat": torch.as_tensor(icrs.dec.deg, dtype=torch.float64),
        "frame": "ICRS",
    }
    obstime = getattr(coord, "obstime", None)
    if obstime is not None:
        kwargs["epoch"] = torch.as_tensor(obstime.tt.jyear, dtype=torch.float64)
    try:
        kwargs["pm_lon_coslat_mas_yr"] = torch.as_tensor(
            icrs.pm_ra_cosdec.to_value(u.mas / u.yr), dtype=torch.float64
        )
        kwargs["pm_lat_mas_yr"] = torch.as_tensor(
            icrs.pm_dec.to_value(u.mas / u.yr), dtype=torch.float64
        )
    except (AttributeError, TypeError, ValueError, UnitsError):
        pass
    try:
        kwargs["parallax_mas"] = torch.as_tensor(
            icrs.distance.parallax.to_value(u.mas), dtype=torch.float64
        )
    except (AttributeError, TypeError, ValueError, UnitsError):
        pass
    try:
        kwargs["radial_velocity_km_s"] = torch.as_tensor(
            icrs.radial_velocity.to_value(u.km / u.s), dtype=torch.float64
        )
    except (AttributeError, TypeError, ValueError, UnitsError):
        pass
    return CatalogState(**kwargs)  # type: ignore[arg-type]


def catalog_state_to_astropy(state: CatalogState) -> SkyCoord:
    """Build a SkyCoord from ICRS ``CatalogState`` (converts frames first)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    icrs = (
        state
        if canonical_celestial_frame(state.frame) == "ICRS"
        else state.to_frame("ICRS")
    )
    kwargs: dict[str, object] = {
        "ra": icrs.lon.detach().cpu().numpy() * u.deg,
        "dec": icrs.lat.detach().cpu().numpy() * u.deg,
        "frame": "icrs",
        "obstime": Time(
            icrs.epoch.detach().cpu().numpy(), format="jyear", scale="tt"
        ),
    }
    if icrs.pm_lon_coslat_mas_yr is not None and icrs.pm_lat_mas_yr is not None:
        kwargs["pm_ra_cosdec"] = (
            icrs.pm_lon_coslat_mas_yr.detach().cpu().numpy() * u.mas / u.yr
        )
        kwargs["pm_dec"] = icrs.pm_lat_mas_yr.detach().cpu().numpy() * u.mas / u.yr
    if icrs.parallax_mas is not None:
        plx_np = icrs.parallax_mas.detach().cpu().numpy()
        if (plx_np > 0.0).all():
            kwargs["distance"] = (1000.0 / plx_np) * u.pc
    if icrs.radial_velocity_km_s is not None:
        kwargs["radial_velocity"] = (
            icrs.radial_velocity_km_s.detach().cpu().numpy() * u.km / u.s
        )
    return SkyCoord(**kwargs)
