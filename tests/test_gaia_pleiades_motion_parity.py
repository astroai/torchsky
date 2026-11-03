"""Astropy parity for the pipeline of ``examples/gaia_pleiades_motion.py``.

Four Gaia DR3 Pleiades members (rows of the pinned example snapshot) are
rotated to the Galactic frame and propagated from J2016.0 to J2000.0 with
``CatalogState``; astropy ``SkyCoord`` is the oracle.
"""

from __future__ import annotations

import pytest
import torch

from torchsky.coords import CatalogState

# source_id, ra, dec, parallax [mas], pmra, pmdec [mas/yr], radial_velocity [km/s]
ROWS = (
    (64879402312818944, 56.21358052689, 23.26872443224, 7.3946, 20.546, -44.300, 6.55),
    (64913452814489216, 56.95058643018, 23.21788377590, 7.4112, 20.054, -45.848, 7.91),
    (64923279699744256, 57.25981898023, 23.25223911266, 7.2668, 19.731, -44.428, 5.03),
    (64929391436043264, 57.42060476340, 23.34136964836, 7.3159, 18.733, -45.133, 5.73),
)


def _state() -> CatalogState:
    cols = torch.tensor([row[1:] for row in ROWS], dtype=torch.float64).T
    return CatalogState(
        lon=cols[0],
        lat=cols[1],
        frame="ICRS",
        epoch=2016.0,
        parallax_mas=cols[2],
        pm_lon_coslat_mas_yr=cols[3],
        pm_lat_mas_yr=cols[4],
        radial_velocity_km_s=cols[5],
    )


def _skycoord(state: CatalogState):
    u = pytest.importorskip("astropy.units")
    coordinates = pytest.importorskip("astropy.coordinates")
    time = pytest.importorskip("astropy.time")
    return (
        coordinates.SkyCoord(
            ra=state.lon.numpy() * u.deg,
            dec=state.lat.numpy() * u.deg,
            distance=coordinates.Distance(parallax=state.parallax_mas.numpy() * u.mas),
            pm_ra_cosdec=state.pm_lon_coslat_mas_yr.numpy() * u.mas / u.yr,
            pm_dec=state.pm_lat_mas_yr.numpy() * u.mas / u.yr,
            radial_velocity=state.radial_velocity_km_s.numpy() * u.km / u.s,
            obstime=time.Time(2016.0, format="jyear", scale="tcb"),
            frame="icrs",
        ),
        u,
        time,
    )


def test_galactic_proper_motion_matches_astropy() -> None:
    state = _state()
    ref, u, _ = _skycoord(state)
    gal = state.to_frame("GALACTIC")
    expected = ref.galactic
    torch.testing.assert_close(
        gal.lon, torch.tensor(expected.l.deg), atol=1e-9, rtol=0.0
    )
    torch.testing.assert_close(
        gal.lat, torch.tensor(expected.b.deg), atol=1e-9, rtol=0.0
    )
    torch.testing.assert_close(
        gal.pm_lon_coslat_mas_yr,
        torch.tensor(expected.pm_l_cosb.to_value(u.mas / u.yr)),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        gal.pm_lat_mas_yr,
        torch.tensor(expected.pm_b.to_value(u.mas / u.yr)),
        atol=1e-6,
        rtol=0.0,
    )


def test_space_motion_to_j2000_matches_apply_space_motion() -> None:
    state = _state()
    ref, u, time = _skycoord(state)
    moved = ref.apply_space_motion(
        new_obstime=time.Time(2000.0, format="jyear", scale="tcb")
    )
    physical = state.propagate_physical(2000.0)
    dlon_uas = (
        (physical.lon - torch.tensor(moved.ra.deg))
        * physical.lat.deg2rad().cos()
        * 3.6e9
    )
    dlat_uas = (physical.lat - torch.tensor(moved.dec.deg)) * 3.6e9
    assert float(torch.hypot(dlon_uas, dlat_uas).max()) < 1.0
    torch.testing.assert_close(
        physical.pm_lon_coslat_mas_yr,
        torch.tensor(moved.pm_ra_cosdec.to_value(u.mas / u.yr)),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        physical.parallax_mas,
        torch.tensor(moved.distance.parallax.to_value(u.mas)),
        atol=1e-9,
        rtol=0.0,
    )
    # The proper-motion-only model the example uses for every member stays
    # within 50 microarcseconds of the rigorous motion at the Pleiades distance.
    angular = state.propagate_angular(2000.0)
    gap_uas = (
        torch.hypot(
            (angular.lon - physical.lon) * physical.lat.deg2rad().cos(),
            angular.lat - physical.lat,
        )
        * 3.6e9
    )
    assert float(gap_uas.max()) < 50.0
