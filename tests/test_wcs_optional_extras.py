"""Fail-closed optional extras: JPL DE, TLE, live GWCS. No SkyCoord class."""

from __future__ import annotations

import pytest
import torch
from astropy.coordinates import SkyCoord
import astropy.units as u

from torchsky.wcs.astropy_interop import (
    catalog_state_from_astropy,
    catalog_state_to_astropy,
)
from torchsky.wcs.catalog_state import CatalogState
from torchsky.wcs.ephemeris import get_body
from torchsky.wcs.gwcs_live import freeze_live_gwcs
from torchsky.wcs.tle import catalog_state_from_tle


def test_get_body_fails_closed_without_kernel() -> None:
    with pytest.raises((ImportError, FileNotFoundError, ValueError)):
        get_body("earth", 58000.0, path="/no/such/de440.bsp")


def test_tle_fails_closed_without_sgp4() -> None:
    from datetime import datetime, timezone

    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    try:
        state = catalog_state_from_tle(
            line1, line2, datetime(2008, 9, 20, tzinfo=timezone.utc)
        )
    except ImportError:
        return
    assert state.frame == "TEME"
    assert torch.isfinite(state.lon).all()
    assert torch.isfinite(state.lat).all()


def test_freeze_live_gwcs_fails_closed_without_gwcs() -> None:
    with pytest.raises((ImportError, TypeError, ValueError)):
        freeze_live_gwcs(object(), ((0.0, 1.0), (0.0, 1.0)))


def test_catalog_state_astropy_roundtrip_positions() -> None:
    sky = SkyCoord(ra=[10.0, 120.0] * u.deg, dec=[20.0, -5.0] * u.deg, frame="icrs")
    state = catalog_state_from_astropy(sky)
    back = catalog_state_to_astropy(state)
    assert torch.allclose(
        state.lon, torch.as_tensor(sky.ra.deg, dtype=torch.float64), atol=1e-12
    )
    assert abs(float(back.ra.deg[0] - sky.ra.deg[0])) < 1e-10


def test_no_skycoord_class_in_torchsky_wcs() -> None:
    import torchsky.wcs as wcs

    assert not hasattr(wcs, "SkyCoord")
    CatalogState(lon=0.0, lat=0.0)
