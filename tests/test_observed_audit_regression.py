"""Regression tests: observed-place float64 time handling and pn consistency."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import torchsky.wcs.observed as obs
from torchsky.wcs import (
    CatalogState,
    EarthOrientation,
    Site,
    observed_altaz,
)

astropy = pytest.importorskip("astropy")
erfa = pytest.importorskip("erfa")

MJD = 60500.3
LON, LAT, H = -155.47, 19.82, 4200.0


def _stars(n: int = 300):
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    return ra, dec


def _tt_minus_utc() -> float:
    from astropy.time import Time

    t = Time(MJD, format="mjd", scale="utc")
    return float((t.tt.mjd - t.mjd) * 86400.0)


def _astropy_altaz(ra, dec):
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iers.conf.auto_download = False
        iers.conf.iers_degraded_accuracy = "ignore"
        loc = EarthLocation.from_geodetic(LON * u.deg, LAT * u.deg, H * u.m)
        t = Time(MJD, format="mjd", scale="utc", location=loc)
        t.delta_ut1_utc = 0.0
        aa = SkyCoord(ra * u.deg, dec * u.deg).transform_to(
            AltAz(obstime=t, location=loc, pressure=0 * u.hPa)
        )
    return aa.az.deg, aa.alt.deg


def _sep_arcsec(a1, l1, a2, l2):
    a1, l1, a2, l2 = (
        np.radians(np.asarray(x, dtype=np.float64)) for x in (a1, l1, a2, l2)
    )
    d = np.arctan2(
        np.hypot(
            np.cos(l2) * np.sin(a2 - a1),
            np.cos(l1) * np.sin(l2) - np.sin(l1) * np.cos(l2) * np.cos(a2 - a1),
        ),
        np.sin(l1) * np.sin(l2) + np.cos(l1) * np.cos(l2) * np.cos(a2 - a1),
    )
    return np.degrees(d) * 3600.0


def _site():
    return Site(lon_deg=LON, lat_deg=LAT, height_m=H)


def test_default_altaz_matches_astropy() -> None:
    ra, dec = _stars()
    eop = EarthOrientation(dut1_s=0.0, tt_minus_utc_s=_tt_minus_utc())
    az, alt = observed_altaz(
        CatalogState(lon=ra, lat=dec),
        obstime_mjd_utc=MJD,
        site=_site(),
        eop=eop,
        aberration=True,
    )
    az_a, alt_a = _astropy_altaz(ra, dec)
    assert _sep_arcsec(az.numpy(), alt.numpy(), az_a, alt_a).max() < 1.0


def test_pn_none_polar_motion_zero_is_continuous() -> None:
    ra, dec = _stars()
    state = CatalogState(lon=ra, lat=dec)
    eop0 = EarthOrientation(dut1_s=0.0)
    eopz = EarthOrientation(dut1_s=0.0, xp_arcsec=0.0, yp_arcsec=0.0)
    a0 = observed_altaz(state, obstime_mjd_utc=MJD, site=_site(), eop=eop0, pn="none")
    a1 = observed_altaz(state, obstime_mjd_utc=MJD, site=_site(), eop=eopz, pn="none")
    assert (
        _sep_arcsec(a0[0].numpy(), a0[1].numpy(), a1[0].numpy(), a1[1].numpy()).max()
        < 1e-6
    )


def test_time_quantities_are_float64_irrespective_of_working_dtype(monkeypatch) -> None:
    ra, dec = _stars(100)
    eop = EarthOrientation(dut1_s=0.0, tt_minus_utc_s=_tt_minus_utc())
    state = CatalogState(lon=ra, lat=dec)
    ref = observed_altaz(
        state, obstime_mjd_utc=MJD, site=_site(), eop=eop, aberration=True
    )
    monkeypatch.setattr(obs, "_work_dtype", lambda device: torch.float32)
    st32 = CatalogState(
        lon=torch.tensor(ra, dtype=torch.float32),
        lat=torch.tensor(dec, dtype=torch.float32),
    )
    t32 = torch.tensor(MJD, dtype=torch.float32)
    az, alt = observed_altaz(
        st32, obstime_mjd_utc=t32, site=_site(), eop=eop, aberration=True
    )
    # float32 *time* alone would move results by 0.3-0.5 deg; float32 angles
    # only cost O(arcsec).
    assert az.dtype == torch.float32
    t_exact = torch.tensor(float(t32), dtype=torch.float64)
    ref32t = observed_altaz(
        state, obstime_mjd_utc=t_exact, site=_site(), eop=eop, aberration=True
    )
    d = _sep_arcsec(
        az.double().numpy(), alt.double().numpy(), ref32t[0].numpy(), ref32t[1].numpy()
    )
    assert d.max() < 30.0
    # internal time matrices stay float64
    assert (
        obs._gcrs_to_terrestrial_matrix(
            torch.tensor(MJD, dtype=torch.float64), eop, "iau2006"
        ).dtype
        == torch.float64
    )
    assert ref[0].dtype == torch.float64


def test_cirs_tete_itrs_match_erfa() -> None:
    ra, dec = _stars(50)
    tt_off = _tt_minus_utc()
    eop = EarthOrientation(dut1_s=0.0, tt_minus_utc_s=tt_off)
    state = CatalogState(lon=ra, lat=dec)
    tt = MJD + tt_off / 86400.0
    lon, lat = np.radians(ra), np.radians(dec)
    v = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1
    )
    cirs = obs.icrs_to_cirs_xyz(state, obstime_mjd_utc=MJD, eop=eop).numpy()
    np.testing.assert_allclose(cirs, v @ erfa.c2i06a(2400000.5, tt).T, atol=1e-12)
    tete = obs.icrs_to_tete_xyz(state, obstime_mjd_utc=MJD, eop=eop).numpy()
    np.testing.assert_allclose(tete, v @ erfa.pnm06a(2400000.5, tt).T, atol=1e-12)
    itrs = obs._icrs_to_itrs_xyz(state, obstime_mjd_utc=MJD, eop=eop).numpy()
    rot = erfa.c2tcio(erfa.c2i06a(2400000.5, tt), erfa.era00(2400000.5, MJD), np.eye(3))
    np.testing.assert_allclose(itrs, v @ rot.T, atol=1e-12)


def test_coordinate_gradients_survive() -> None:
    eop = EarthOrientation(dut1_s=0.0, tt_minus_utc_s=_tt_minus_utc())
    lon = torch.tensor([10.0, 200.0], dtype=torch.float64, requires_grad=True)
    lat = torch.tensor([20.0, -30.0], dtype=torch.float64, requires_grad=True)
    az, alt = observed_altaz(
        CatalogState(lon=lon, lat=lat), obstime_mjd_utc=MJD, site=_site(), eop=eop
    )
    (az.sum() + alt.sum()).backward()
    assert lon.grad is not None and torch.isfinite(lon.grad).all()
    assert lat.grad is not None and torch.isfinite(lat.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_matches_cpu() -> None:
    ra, dec = _stars(200)
    eop = EarthOrientation(dut1_s=0.0, tt_minus_utc_s=_tt_minus_utc())
    cpu = observed_altaz(
        CatalogState(lon=ra, lat=dec),
        obstime_mjd_utc=MJD,
        site=_site(),
        eop=eop,
        aberration=True,
    )
    st = CatalogState(
        lon=torch.tensor(ra, dtype=torch.float32, device="mps"),
        lat=torch.tensor(dec, dtype=torch.float32, device="mps"),
    )
    az, alt = observed_altaz(
        st, obstime_mjd_utc=MJD, site=_site(), eop=eop, aberration=True
    )
    d = _sep_arcsec(
        az.cpu().double().numpy(),
        alt.cpu().double().numpy(),
        cpu[0].numpy(),
        cpu[1].numpy(),
    )
    assert d.max() < 30.0
