"""``observed_altaz`` never skips precession: TT-UTC defaults to the leap table.

Oracles: astropy ``Time`` for UTC -> TT (ERFA ``utctai``, including the
leap-day ramp) and ``erfa.atco13`` for the observed place.
"""

import numpy as np
import pytest
import torch

erfa = pytest.importorskip("erfa")
astropy_time = pytest.importorskip("astropy.time")

from torchsky.wcs import CatalogState, EarthOrientation, Site, observed_altaz  # noqa: E402
from torchsky.wcs.temporal import (  # noqa: E402
    _LEAP_TABLE_EXPIRES_MJD,
    _tt_minus_utc_seconds_checked,
)

MAS = 3.6e6


def test_leap_table_tt_minus_utc_matches_astropy() -> None:
    rng = np.random.default_rng(3)
    # Random epochs 1972..table expiry, plus the 2016-12-31 leap day itself.
    mjd = np.concatenate(
        [
            rng.uniform(41317.0, _LEAP_TABLE_EXPIRES_MJD - 1.0, 400),
            np.array([57753.0, 57753.25, 57753.999, 57754.0]),
        ]
    )
    ours = _tt_minus_utc_seconds_checked(torch.tensor(mjd)).numpy()
    t = astropy_time.Time(mjd, format="mjd", scale="utc")
    ref = (t.tt.mjd - mjd) * 86400.0
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    "mjd", [40000.0, 41316.9, _LEAP_TABLE_EXPIRES_MJD, 70000.0, float("nan")]
)
def test_tt_minus_utc_outside_table_raises(mjd: float) -> None:
    with pytest.raises(ValueError, match="TT-UTC"):
        observed_altaz(
            CatalogState(lon=10.0, lat=20.0),
            obstime_mjd_utc=torch.tensor([60000.0, mjd], dtype=torch.float64),
            site=Site(lon_deg=0.0, lat_deg=45.0),
            eop=EarthOrientation(dut1_s=0.0),
        )


def test_default_altaz_without_tt_matches_atco13() -> None:
    n = 500
    rng = np.random.default_rng(11)
    ra = rng.uniform(0.0, 360.0, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    mjd = rng.uniform(58000.0, 61000.0, n)
    dut1, xp, yp = 0.12, 0.1, 0.3
    lon, lat, h = -155.47, 19.83, 4200.0
    zeros = np.zeros(n)
    aob, zob, *_ = erfa.atco13(
        np.radians(ra), np.radians(dec), zeros, zeros, zeros, zeros,
        2400000.5 * np.ones(n), mjd, dut1, np.radians(lon), np.radians(lat), h,
        np.radians(xp / 3600.0), np.radians(yp / 3600.0), 0.0, 0.0, 0.0, 0.55,
    )  # fmt: skip
    az_e, alt_e = np.degrees(aob), 90.0 - np.degrees(zob)
    az, alt = observed_altaz(
        CatalogState(lon=torch.tensor(ra), lat=torch.tensor(dec)),
        obstime_mjd_utc=torch.tensor(mjd),
        site=Site(lon_deg=lon, lat_deg=lat, height_m=h),
        eop=EarthOrientation(dut1_s=dut1, xp_arcsec=xp, yp_arcsec=yp),
        aberration=True,
    )
    daz = ((az.numpy() - az_e + 180.0) % 360.0 - 180.0) * np.cos(np.radians(alt_e))
    sep = np.hypot(daz, alt.numpy() - alt_e) * MAS
    assert sep.max() < 0.05, f"max {sep.max():.4f} mas"


def test_default_applies_precession_and_none_is_explicit_opt_in() -> None:
    state = CatalogState(lon=180.0, lat=20.0)
    kwargs = dict(
        obstime_mjd_utc=torch.tensor(60000.0, dtype=torch.float64),
        site=Site(lon_deg=0.0, lat_deg=45.0),
    )
    derived = observed_altaz(state, eop=EarthOrientation(dut1_s=0.0), **kwargs)
    explicit = observed_altaz(
        state, eop=EarthOrientation(dut1_s=0.0, tt_minus_utc_s=69.184), **kwargs
    )
    torch.testing.assert_close(derived, explicit, atol=1e-12, rtol=0.0)
    az0, alt0 = observed_altaz(
        state, eop=EarthOrientation(dut1_s=0.0), pn="none", **kwargs
    )
    daz = torch.remainder(derived[0] - az0 + 180.0, 360.0) - 180.0
    assert 0.1 < float(torch.hypot(daz, derived[1] - alt0)) < 1.0
