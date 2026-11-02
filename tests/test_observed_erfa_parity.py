"""Observed place against ERFA: epv00 ephemeris, apparent place and refco refraction."""

import numpy as np
import pytest
import torch

erfa = pytest.importorskip("erfa")

from torchsky.wcs import CatalogState, EarthOrientation, Site, observed_altaz  # noqa: E402
from torchsky.wcs.iau2006 import (  # noqa: E402
    earth_barycentric_velocity_icrs,
    earth_pv_icrs,
    refco,
)

MAS = 3.6e6  # degrees to milliarcseconds


def _sky(n, seed, mjd_lo=58000.0, mjd_hi=61000.0):
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    mjd = rng.uniform(mjd_lo, mjd_hi, n)
    yy, mm, dd, fd = erfa.jd2cal(2400000.5, mjd)
    dat = np.array(
        [erfa.dat(int(y), int(m), int(d), f) for y, m, d, f in zip(yy, mm, dd, fd)]
    )
    return ra, dec, mjd, dat + 32.184


def _compare(refraction_args, seed):
    n = 1500
    ra, dec, mjd, tt_utc = _sky(n, seed)
    dut1, xp, yp = 0.12, 0.1, 0.3
    lon, lat, h = -155.47, 19.83, 4200.0
    phpa, tc, rh, wl = refraction_args
    zeros = np.zeros(n)
    aob, zob, *_ = erfa.atco13(
        np.radians(ra), np.radians(dec), zeros, zeros, zeros, zeros,
        2400000.5 * np.ones(n), mjd, dut1, np.radians(lon), np.radians(lat), h,
        np.radians(xp / 3600.0), np.radians(yp / 3600.0), phpa, tc, rh, wl,
    )  # fmt: skip
    az_e, alt_e = np.degrees(aob), 90.0 - np.degrees(zob)
    refract = phpa > 0.0
    site = Site(
        lon_deg=lon, lat_deg=lat, height_m=h,
        pressure_mbar=phpa if refract else None,
        temperature_c=tc if refract else None,
        relative_humidity=rh, wavelength_um=wl,
    )  # fmt: skip
    eop = EarthOrientation(
        dut1_s=dut1, xp_arcsec=xp, yp_arcsec=yp, tt_minus_utc_s=torch.tensor(tt_utc)
    )
    state = CatalogState(lon=torch.tensor(ra), lat=torch.tensor(dec), frame="ICRS")
    az, alt = observed_altaz(
        state, obstime_mjd_utc=torch.tensor(mjd), site=site, eop=eop,
        aberration=True, refraction=refract,
    )  # fmt: skip
    daz = ((az.numpy() - az_e + 180.0) % 360.0 - 180.0) * np.cos(np.radians(alt_e))
    sep = np.hypot(daz, alt.numpy() - alt_e) * MAS
    return sep, 90.0 - alt_e


def test_epv00_matches_erfa():
    mjd = np.random.default_rng(0).uniform(40000.0, 70000.0, 500)
    helio, bary, vel = earth_pv_icrs(torch.tensor(mjd))
    pvh, pvb = erfa.epv00(2400000.5 * np.ones_like(mjd), mjd)
    np.testing.assert_allclose(helio.numpy(), pvh["p"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(bary.numpy(), pvb["p"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(vel.numpy(), pvb["v"], rtol=0, atol=1e-14)
    v_ms = earth_barycentric_velocity_icrs(torch.tensor(mjd)).numpy()
    np.testing.assert_allclose(v_ms, pvb["v"] * 149597870700.0 / 86400.0, atol=1e-8)


def test_observed_altaz_matches_atco13_without_refraction():
    sep, _ = _compare((0.0, 0.0, 0.0, 0.55), seed=1)
    assert sep.max() < 0.05, f"max {sep.max():.4f} mas"


@pytest.mark.parametrize(
    "met",
    [(800.0, 10.0, 0.3, 0.55), (1013.25, 15.0, 0.0, 0.55), (900.0, 20.0, 0.5, 3000.0)],
)
def test_refco_refraction_matches_atco13_below_z85(met):
    sep, zd = _compare(met, seed=2)
    sel = zd < 85.0
    assert sel.sum() > 300
    assert sep[sel].max() < 0.05, f"max {sep[sel].max():.4f} mas"


@pytest.mark.parametrize(
    "args",
    [(800.0, 10.0, 0.3, 0.55), (600.0, -5.0, 0.8, 1.6), (900.0, 20.0, 0.5, 3000.0)],
)
def test_refco_constants_match_erfa(args):
    a, b = refco(*args)
    ea, eb = erfa.refco(*args)
    assert abs(a.item() - ea) < 1e-15 and abs(b.item() - eb) < 1e-18


def test_saemundsson_remains_available_and_close():
    ra, dec, mjd, tt_utc = _sky(50, 3, 60000.0, 60001.0)
    site = Site(
        lon_deg=-155.47, lat_deg=19.83, pressure_mbar=1010.0, temperature_c=10.0
    )
    eop = EarthOrientation(dut1_s=0.1, tt_minus_utc_s=torch.tensor(tt_utc))
    state = CatalogState(lon=torch.tensor(ra), lat=torch.tensor(dec), frame="ICRS")
    kw = dict(obstime_mjd_utc=torch.tensor(mjd), site=site, eop=eop, refraction=True)
    _, alt_r = observed_altaz(state, **kw)
    _, alt_s = observed_altaz(state, refraction_model="saemundsson", **kw)
    _, alt_0 = observed_altaz(
        state, obstime_mjd_utc=torch.tensor(mjd), site=site, eop=eop
    )
    up = alt_0.numpy() > 20.0
    assert np.all(np.abs(alt_r.numpy() - alt_s.numpy())[up] < 5e-3)  # agree to ~18"
    assert np.all((alt_r.numpy() - alt_0.numpy())[up] > 0.0)
    with pytest.raises(ValueError, match="refraction_model"):
        observed_altaz(state, refraction_model="bogus", **kw)  # type: ignore[arg-type]


def test_refraction_model_and_site_humidity_reach_every_entry_point():
    import torchsky
    from torchsky.catalogs import Catalog
    from torchsky.coords import transform

    assert (
        torchsky.PrecisionWarning
        is __import__("torchsky.wcs", fromlist=["x"]).PrecisionWarning
    )
    ra, dec, mjd, tt_utc = _sky(20, 4, 60000.0, 60001.0)
    eop = EarthOrientation(dut1_s=0.1, tt_minus_utc_s=torch.tensor(tt_utc))
    site = Site(
        lon_deg=-155.47, lat_deg=19.83, pressure_mbar=800.0, temperature_c=5.0,
        relative_humidity=0.9, wavelength_um=3000.0,
    )  # fmt: skip
    dry = Site(lon_deg=-155.47, lat_deg=19.83, pressure_mbar=800.0, temperature_c=5.0)
    state = CatalogState(lon=torch.tensor(ra), lat=torch.tensor(dec), frame="ICRS")
    t = torch.tensor(mjd)
    kw = dict(obstime_mjd_utc=t, eop=eop, refraction=True)
    ref_az, ref_alt = observed_altaz(state, site=site, **kw)
    cat = Catalog(lon=state.lon, lat=state.lat, frame="ICRS")
    for model in ("refco", "saemundsson"):
        direct = observed_altaz(state, site=site, refraction_model=model, **kw)
        via_t = transform(
            state.lon, state.lat, from_frame="ICRS", to_frame="ALTAZ",
            site=site, refraction_model=model, **kw,
        )  # fmt: skip
        via_c = cat.observe("ALTAZ", site=site, refraction_model=model, **kw)
        assert torch.equal(direct[1], via_t[1]) and torch.equal(direct[1], via_c.lat)
    # default is refco; humidity/wavelength change the result
    assert torch.equal(
        ref_alt, observed_altaz(state, site=site, refraction_model="refco", **kw)[1]
    )
    dry_alt = observed_altaz(state, site=dry, **kw)[1]
    assert not torch.equal(dry_alt, ref_alt)
