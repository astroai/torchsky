from __future__ import annotations

import pytest
import torch

from torchsky.wcs import CatalogState, EarthOrientation, Site, observed_altaz
from torchsky.wcs.observed import _gmst_deg_from_ut1_mjd


def test_catalog_state_frame_roundtrip() -> None:
    state = CatalogState(
        lon=torch.tensor([10.0, 120.0], dtype=torch.float64),
        lat=torch.tensor([20.0, -5.0], dtype=torch.float64),
        frame="ICRS",
    )
    gal = state.to_frame("GALACTIC")
    back = gal.to_frame("ICRS")
    torch.testing.assert_close(back.lon, state.lon, atol=1e-8, rtol=0.0)
    torch.testing.assert_close(back.lat, state.lat, atol=1e-8, rtol=0.0)


def test_catalog_state_refuses_incomplete_space_motion() -> None:
    state = CatalogState(
        lon=10.0,
        lat=20.0,
        pm_lon_coslat_mas_yr=1.0,
        pm_lat_mas_yr=2.0,
    )
    with pytest.raises(ValueError, match="not imputed"):
        state.propagate_physical(2010.0)


def test_observed_altaz_requires_eop_and_zenith() -> None:
    with pytest.raises(ValueError, match="EarthOrientation"):
        observed_altaz(
            CatalogState(lon=0.0, lat=45.0),
            obstime_mjd_utc=60000.0,
            site=Site(lon_deg=0.0, lat_deg=45.0),
            eop=None,  # type: ignore[arg-type]
        )
    with pytest.raises(NotImplementedError, match="polar motion"):
        EarthOrientation(dut1_s=0.0, xp_arcsec=0.1)

    eop = EarthOrientation(dut1_s=0.0)
    site = Site(lon_deg=0.0, lat_deg=45.0)
    utc = torch.tensor(51544.5, dtype=torch.float64)
    last = _gmst_deg_from_ut1_mjd(utc)
    state = CatalogState(lon=last, lat=torch.tensor(45.0, dtype=torch.float64))
    _az, alt = observed_altaz(state, obstime_mjd_utc=utc, site=site, eop=eop)
    torch.testing.assert_close(
        alt, torch.tensor(90.0, dtype=torch.float64), atol=1e-6, rtol=0.0
    )