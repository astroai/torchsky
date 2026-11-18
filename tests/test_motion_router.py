"""Per-row epoch propagation and motion Jacobians against astropy."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchsky.wcs import (
    propagate_proper_motion_with_jacobian,
    propagate_space_motion_with_jacobian,
)
from torchsky.wcs.catalog_state import CatalogState

coords = pytest.importorskip("astropy.coordinates")
units = pytest.importorskip("astropy.units")
atime = pytest.importorskip("astropy.time")
u = units
D = torch.float64

T0, T1 = 2016.0, 1990.0


def _astropy(ra, dec, pmra, pmdec, plx=None, rv=None):
    kw = {}
    if plx is not None:
        kw = dict(
            distance=coords.Distance(parallax=plx * u.mas),
            radial_velocity=rv * u.km / u.s,
        )
    c = coords.SkyCoord(
        ra * u.deg,
        dec * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        obstime=atime.Time(T0, format="jyear"),
        **kw,
    )
    return c.apply_space_motion(new_obstime=atime.Time(T1, format="jyear"))


def _rows():
    # Barnard-like, Gaia-typical, near both poles, across RA 0/360.
    ra = np.array([269.45, 10.0, 40.0, 220.0, 359.9999])
    dec = np.array([4.69, -30.0, 89.98, -89.98, 0.0])
    pmra = np.array([-798.6, 5.0, -798.6, 300.0, 1200.0])
    pmdec = np.array([10328.1, -3.0, 10328.1, -2000.0, -40.0])
    plx = np.array([547.45, 2.0, 30.0, 10.0, 100.0])
    rv = np.array([-110.5, 15.0, 50.0, -20.0, 0.0])
    return ra, dec, pmra, pmdec, plx, rv


def _sep_mas(ra1, dec1, ra2, dec2) -> np.ndarray:
    a = coords.SkyCoord(ra1 * u.deg, dec1 * u.deg)
    b = coords.SkyCoord(ra2 * u.deg, dec2 * u.deg)
    return a.separation(b).to_value(u.mas)


def test_angular_propagation_carries_proper_motion_like_astropy() -> None:
    ra, dec, pmra, pmdec, _, _ = _rows()
    state = CatalogState(
        lon=torch.tensor(ra),
        lat=torch.tensor(dec),
        epoch=T0,
        pm_lon_coslat_mas_yr=torch.tensor(pmra),
        pm_lat_mas_yr=torch.tensor(pmdec),
    )
    moved = state.propagate_angular(T1)
    want = _astropy(ra, dec, pmra, pmdec)
    assert (
        _sep_mas(moved.lon.numpy(), moved.lat.numpy(), want.ra.deg, want.dec.deg).max()
        < 1e-4
    )
    np.testing.assert_allclose(
        moved.pm_lon_coslat_mas_yr.numpy(),
        want.pm_ra_cosdec.to_value(u.mas / u.yr),
        rtol=1e-9,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        moved.pm_lat_mas_yr.numpy(),
        want.pm_dec.to_value(u.mas / u.yr),
        rtol=1e-9,
        atol=1e-7,
    )
    # Forward then back is not the identity without distance (the radial part
    # of the velocity is dropped at each normalisation); it matches astropy's
    # own round trip.
    back = moved.propagate_angular(T0)
    c = coords.SkyCoord(
        want.ra,
        want.dec,
        pm_ra_cosdec=want.pm_ra_cosdec,
        pm_dec=want.pm_dec,
        obstime=atime.Time(T1, format="jyear"),
    ).apply_space_motion(new_obstime=atime.Time(T0, format="jyear"))
    assert (
        _sep_mas(back.lon.numpy(), back.lat.numpy(), c.ra.deg, c.dec.deg).max() < 1e-4
    )


def test_moving_rows_without_proper_motion_fail_closed() -> None:
    state = CatalogState(
        lon=torch.tensor([10.0, 20.0], dtype=D),
        lat=torch.tensor([0.0, 0.0], dtype=D),
        epoch=torch.tensor([T0, T1], dtype=D),
        pm_lon_coslat_mas_yr=torch.tensor([math.nan, 1.0], dtype=D),
        pm_lat_mas_yr=torch.tensor([math.nan, 1.0], dtype=D),
    )
    with pytest.raises(ValueError, match="1 row"):
        state.propagate_angular(T1)
    # The NaN row is already at T0 when propagating to T0: allowed, kept as is.
    moved = state.propagate_angular(T0)
    assert moved.lon[0] == 10.0 and torch.isnan(moved.pm_lon_coslat_mas_yr[0])


def test_router_uses_the_richest_motion_per_row() -> None:
    ra, dec, pmra, pmdec, plx, rv = _rows()
    n = len(ra)
    # Row 1: negative parallax (level 4); row 3: no radial velocity (level 4);
    # row 4: no proper motion (level 2, prior).
    plx_t = torch.tensor(plx).clone()
    plx_t[1] = -0.5
    rv_t = torch.tensor(rv).clone()
    rv_t[3] = math.nan
    pm_lon = torch.tensor(pmra).clone()
    pm_lat = torch.tensor(pmdec).clone()
    pm_lon[4] = pm_lat[4] = math.nan
    state = CatalogState(
        lon=torch.tensor(ra),
        lat=torch.tensor(dec),
        epoch=T0,
        pm_lon_coslat_mas_yr=pm_lon,
        pm_lat_mas_yr=pm_lat,
        parallax_mas=plx_t,
        radial_velocity_km_s=rv_t,
    )
    assert state.motion_level().tolist() == [6, 4, 6, 4, 2]
    with pytest.raises(ValueError, match="prior_pm_mas_yr"):
        state.propagate(T1)
    prior = (torch.full((n,), 3.0, dtype=D), torch.full((n,), -4.0, dtype=D))
    moved = state.propagate(T1, prior_pm_mas_yr=prior)
    for i in (0, 2):
        want = _astropy(ra[i], dec[i], pmra[i], pmdec[i], plx[i], rv[i])
        assert (
            _sep_mas(
                moved.lon[i].item(), moved.lat[i].item(), want.ra.deg, want.dec.deg
            )
            < 1e-3
        )
    for i in (1, 3):
        want = _astropy(ra[i], dec[i], pmra[i], pmdec[i])
        assert (
            _sep_mas(
                moved.lon[i].item(), moved.lat[i].item(), want.ra.deg, want.dec.deg
            )
            < 1e-4
        )
    want = _astropy(ra[4], dec[4], 3.0, -4.0)
    assert (
        _sep_mas(moved.lon[4].item(), moved.lat[4].item(), want.ra.deg, want.dec.deg)
        < 1e-4
    )
    assert torch.isnan(moved.pm_lon_coslat_mas_yr[4])  # a prior is not a measurement


def _fd_jacobian(ra, dec, pmra, pmdec, plx=None, rv=None):
    """Central differences: offsets on the sky in, offsets on the sky out (mas)."""
    base = _astropy(ra, dec, pmra, pmdec, plx, rv)
    n_in = 4 if plx is None else 6
    steps = [1.0, 1.0, 1.0, 1.0, 1e-3, 1e-3][:n_in]
    jac = np.zeros((2, n_in))
    for k in range(n_in):
        out = []
        for s in (1.0, -1.0):
            x = [ra, dec, pmra, pmdec] + ([] if plx is None else [plx, rv])
            if k < 2:
                c = coords.SkyCoord(ra * u.deg, dec * u.deg).spherical_offsets_by(
                    s * steps[0] * u.mas * (k == 0), s * steps[1] * u.mas * (k == 1)
                )
                x[0], x[1] = c.ra.deg, c.dec.deg
            else:
                x[k] += s * steps[k]
            dl, db = base.spherical_offsets_to(_astropy(*x))
            out.append((dl.to_value(u.mas), db.to_value(u.mas)))
        jac[:, k] = (np.array(out[0]) - np.array(out[1])) / (2 * steps[k])
    return jac


@pytest.mark.parametrize("dec", [0.0, 60.0, 89.98, -89.98])
def test_angular_jacobian_matches_astropy_finite_differences(dec: float) -> None:
    ra, pmra, pmdec = (
        40.0,
        -798.6,
        10328.1,
    )  # 26 yr path crosses the pole at |dec| = 89.98
    _, jac = propagate_proper_motion_with_jacobian(
        torch.tensor(ra, dtype=D),
        torch.tensor(dec, dtype=D),
        torch.tensor(pmra, dtype=D),
        torch.tensor(pmdec, dtype=D),
        source_epoch_jyear=T0,
        target_epoch_jyear=T1,
    )
    want = _fd_jacobian(ra, dec, pmra, pmdec)
    assert np.abs(jac.numpy() - want).max() <= 1e-6 * np.abs(want).max()


@pytest.mark.parametrize("dec", [0.0, 89.98, -89.98])
def test_space_motion_jacobian_matches_astropy_finite_differences(dec: float) -> None:
    x = (40.0, dec, -798.6, 10328.1, 547.45, -110.5)
    _, jac = propagate_space_motion_with_jacobian(
        *[torch.tensor(v, dtype=D) for v in x],
        source_epoch_jyear=T0,
        target_epoch_jyear=T1,
    )
    want = _fd_jacobian(*x)
    # Finite-difference noise in the parallax / RV columns sets the tolerance.
    assert np.abs(jac.numpy()[:2, :6] - want).max() <= 5e-6 * np.abs(want).max()
