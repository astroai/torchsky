from __future__ import annotations

import pytest
import torch

from torchsky.wcs import (
    CatalogState,
    EarthOrientation,
    Site,
    icrs_to_itrs_xyz,
    observed_altaz,
    transform,
)
from torchsky.wcs.observed import (
    _era_rad_from_ut1_mjd,
    _gmst_deg_from_ut1_mjd,
    _wgs84_itrs_m,
)


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


def test_catalog_state_rotates_proper_motion_with_the_frame() -> None:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    state = CatalogState(
        lon=torch.tensor([10.0, 120.0, 359.9, 45.0], dtype=torch.float64),
        lat=torch.tensor([20.0, -5.0, 89.9, -89.9], dtype=torch.float64),
        pm_lon_coslat_mas_yr=torch.tensor([10.0, -30.0, 100.0, 5.0]),
        pm_lat_mas_yr=torch.tensor([20.0, 40.0, -10.0, 15.0]),
    )
    expected = SkyCoord(
        ra=state.lon.numpy() * u.deg,
        dec=state.lat.numpy() * u.deg,
        pm_ra_cosdec=state.pm_lon_coslat_mas_yr.numpy() * u.mas / u.yr,
        pm_dec=state.pm_lat_mas_yr.numpy() * u.mas / u.yr,
    ).galactic
    galactic = state.to_frame("GALACTIC")
    torch.testing.assert_close(
        galactic.pm_lon_coslat_mas_yr,
        torch.tensor(expected.pm_l_cosb.to_value(u.mas / u.yr)),
        atol=2e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        galactic.pm_lat_mas_yr,
        torch.tensor(expected.pm_b.to_value(u.mas / u.yr)),
        atol=2e-5,
        rtol=0.0,
    )
    back = galactic.to_frame("ICRS")
    torch.testing.assert_close(back.pm_lon_coslat_mas_yr, state.pm_lon_coslat_mas_yr)
    torch.testing.assert_close(back.pm_lat_mas_yr, state.pm_lat_mas_yr)


@pytest.mark.parametrize("frame", ["GALACTIC", "ECLIPTIC", "ECLIPTIC_OF_DATE"])
def test_frame_motion_rotation_preserves_magnitude_and_gradients(frame) -> None:
    lon = torch.tensor([10.0, 120.0], dtype=torch.float64, requires_grad=True)
    pm_lon = torch.tensor([10.0, -30.0], dtype=torch.float64, requires_grad=True)
    state = CatalogState(
        lon=lon,
        lat=[20.0, -5.0],
        pm_lon_coslat_mas_yr=pm_lon,
        pm_lat_mas_yr=[20.0, 40.0],
    )
    out = state.to_frame(frame, tt_mjd=60000.0)
    torch.testing.assert_close(
        torch.hypot(out.pm_lon_coslat_mas_yr, out.pm_lat_mas_yr),
        torch.tensor([500.0, 2500.0], dtype=torch.float64).sqrt(),
    )
    grads = torch.autograd.grad(out.pm_lon_coslat_mas_yr.sum(), (lon, pm_lon))
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_catalog_state_refuses_partial_or_unsupported_motion_conversion() -> None:
    partial = CatalogState(lon=10.0, lat=20.0, pm_lat_mas_yr=1.0)
    with pytest.raises(ValueError, match="both proper.motion"):
        partial.to_frame("GALACTIC")
    complete = CatalogState(
        lon=10.0, lat=20.0, pm_lon_coslat_mas_yr=1.0, pm_lat_mas_yr=2.0
    )
    with pytest.raises(ValueError, match="FK4"):
        complete.to_frame("FK4")


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
    eop = EarthOrientation(dut1_s=0.0, xp_arcsec=0.1, yp_arcsec=-0.2)
    assert eop.xp_arcsec is not None

    eop = EarthOrientation(dut1_s=0.0)
    site = Site(lon_deg=0.0, lat_deg=45.0)
    utc = torch.tensor(51544.5, dtype=torch.float64)
    last = _gmst_deg_from_ut1_mjd(utc)
    state = CatalogState(lon=last, lat=torch.tensor(45.0, dtype=torch.float64))
    _az, alt = observed_altaz(state, obstime_mjd_utc=utc, site=site, eop=eop)
    torch.testing.assert_close(
        alt, torch.tensor(90.0, dtype=torch.float64), atol=1e-6, rtol=0.0
    )


def test_observed_altaz_polar_motion_shifts_at_mas_scale() -> None:
    utc = torch.tensor(60000.0, dtype=torch.float64)
    site = Site(lon_deg=0.0, lat_deg=45.0, height_m=0.0)
    state = CatalogState(lon=180.0, lat=20.0)
    az0, alt0 = observed_altaz(
        state,
        obstime_mjd_utc=utc,
        site=site,
        eop=EarthOrientation(dut1_s=0.0, xp_arcsec=0.0, yp_arcsec=0.0),
    )
    az1, alt1 = observed_altaz(
        state,
        obstime_mjd_utc=utc,
        site=site,
        eop=EarthOrientation(dut1_s=0.0, xp_arcsec=0.3, yp_arcsec=0.0),
    )
    d_arcsec = (az1 - az0).abs() * 3600.0
    # 0.3 arcsec of xp moves the CIP; the local az change is the same order.
    assert float(d_arcsec) > 0.05
    assert float(d_arcsec) < 2.0
    az_hi, alt_hi = observed_altaz(
        state,
        obstime_mjd_utc=utc,
        site=Site(lon_deg=0.0, lat_deg=45.0, height_m=2000.0),
        eop=EarthOrientation(dut1_s=0.0, xp_arcsec=0.3, yp_arcsec=0.0),
    )
    torch.testing.assert_close(az_hi, az1, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(alt_hi, alt1, atol=1e-12, rtol=0.0)


def test_era00_j2000_matches_sofa_constant() -> None:
    era = _era_rad_from_ut1_mjd(torch.tensor(51544.5, dtype=torch.float64))
    expected = torch.tensor(2.0 * torch.pi * 0.7790572732640, dtype=torch.float64)
    torch.testing.assert_close(era, expected, atol=1e-12, rtol=0.0)


def test_wgs84_height_increases_itrs_radius() -> None:
    lon = torch.tensor(0.0, dtype=torch.float64)
    lat = torch.tensor(0.0, dtype=torch.float64)
    r0 = torch.linalg.vector_norm(_wgs84_itrs_m(lon, lat, torch.tensor(0.0)))
    r1 = torch.linalg.vector_norm(_wgs84_itrs_m(lon, lat, torch.tensor(1000.0)))
    torch.testing.assert_close(
        r1 - r0, torch.tensor(1000.0, dtype=torch.float64), atol=1e-6, rtol=0.0
    )


def test_catalog_state_barycentric_au_from_parallax() -> None:
    state = CatalogState(
        lon=0.0,
        lat=0.0,
        parallax_mas=1000.0,
    )
    xyz = state.to_representation("barycentric_au")
    au_per_pc = 648000.0 / float(torch.pi)
    torch.testing.assert_close(
        xyz,
        torch.tensor([au_per_pc, 0.0, 0.0], dtype=torch.float64),
        atol=1e-6,
        rtol=0.0,
    )
    with pytest.raises(ValueError, match="parallax"):
        CatalogState(lon=0.0, lat=0.0).to_representation("barycentric_au")


def test_iau2006_pn_shifts_altaz_from_era_only() -> None:
    utc = torch.tensor(60000.0, dtype=torch.float64)
    site = Site(lon_deg=0.0, lat_deg=45.0)
    state = CatalogState(lon=180.0, lat=20.0)
    eop = EarthOrientation(
        dut1_s=0.0, xp_arcsec=0.0, yp_arcsec=0.0, tt_minus_utc_s=69.184
    )
    az0, alt0 = observed_altaz(
        state, obstime_mjd_utc=utc, site=site, eop=eop, pn="none"
    )
    az1, alt1 = observed_altaz(
        state, obstime_mjd_utc=utc, site=site, eop=eop, pn="iau2006"
    )
    daz = torch.remainder(az1 - az0 + 180.0, 360.0) - 180.0
    shift_deg = torch.hypot(daz, alt1 - alt0)
    assert float(shift_deg) > 0.05
    assert float(shift_deg) < 1.0
    with pytest.raises(ValueError, match="tt_minus_utc"):
        observed_altaz(
            state,
            obstime_mjd_utc=utc,
            site=site,
            eop=EarthOrientation(dut1_s=0.0),
            pn="iau2006",
        )


def test_aberration_and_refraction_are_opt_in() -> None:
    utc = torch.tensor(51544.5, dtype=torch.float64)
    last = _gmst_deg_from_ut1_mjd(utc)
    state = CatalogState(lon=last, lat=0.0)
    eop = EarthOrientation(
        dut1_s=0.0, xp_arcsec=0.0, yp_arcsec=0.0, tt_minus_utc_s=69.184
    )
    site = Site(lon_deg=0.0, lat_deg=45.0, pressure_mbar=1010.0, temperature_c=10.0)
    az0, alt0 = observed_altaz(state, obstime_mjd_utc=utc, site=site, eop=eop)
    assert float(alt0.item()) > 40.0
    az_ab, alt_ab = observed_altaz(
        state, obstime_mjd_utc=utc, site=site, eop=eop, aberration=True
    )
    assert float((torch.hypot(az_ab - az0, alt_ab - alt0) * 3600.0).item()) > 5.0
    az_r, alt_r = observed_altaz(
        state, obstime_mjd_utc=utc, site=site, eop=eop, refraction=True
    )
    torch.testing.assert_close(az_r, az0, atol=1e-12, rtol=0.0)
    assert float((alt_r - alt0).item()) > 0.005
    assert float((alt_r - alt0).item()) < 0.05
    with pytest.raises(ValueError, match="pressure"):
        observed_altaz(
            state,
            obstime_mjd_utc=utc,
            site=Site(lon_deg=0.0, lat_deg=45.0),
            eop=eop,
            refraction=True,
        )


def test_icrs_to_itrs_roundtrip_unit() -> None:
    state = CatalogState(lon=30.0, lat=10.0)
    eop = EarthOrientation(dut1_s=0.0)
    xyz = icrs_to_itrs_xyz(state, obstime_mjd_utc=60000.0, eop=eop)
    torch.testing.assert_close(
        torch.linalg.vector_norm(xyz),
        torch.tensor(1.0, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_coords_transform_geometric_and_altaz() -> None:
    lon, lat = transform(10.0, 20.0, from_frame="ICRS", to_frame="GALACTIC")
    back_lon, back_lat = transform(lon, lat, from_frame="GALACTIC", to_frame="ICRS")
    torch.testing.assert_close(
        back_lon, torch.tensor(10.0, dtype=torch.float64), atol=1e-8, rtol=0.0
    )
    torch.testing.assert_close(
        back_lat, torch.tensor(20.0, dtype=torch.float64), atol=1e-8, rtol=0.0
    )
    utc = torch.tensor(51544.5, dtype=torch.float64)
    last = _gmst_deg_from_ut1_mjd(utc)
    az, alt = transform(
        last,
        45.0,
        from_frame="ICRS",
        to_frame="ALTAZ",
        obstime_mjd_utc=utc,
        site=Site(lon_deg=0.0, lat_deg=45.0),
        eop=EarthOrientation(dut1_s=0.0),
    )
    torch.testing.assert_close(
        alt, torch.tensor(90.0, dtype=torch.float64), atol=1e-5, rtol=0.0
    )


def _accel_devices() -> list[str]:
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def _on_accel(value: torch.Tensor | float, device: str) -> torch.Tensor:
    dtype = torch.float32 if device == "mps" else torch.float64
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def test_observed_altaz_stays_on_accelerator() -> None:
    devices = _accel_devices()
    if not devices:
        pytest.skip("no CUDA or MPS")
    utc = torch.tensor(51544.5, dtype=torch.float64)
    last = _gmst_deg_from_ut1_mjd(utc)
    site = Site(lon_deg=0.0, lat_deg=45.0)
    eop = EarthOrientation(dut1_s=0.0)
    az_cpu, alt_cpu = observed_altaz(
        CatalogState(lon=last, lat=45.0),
        obstime_mjd_utc=utc,
        site=site,
        eop=eop,
    )
    torch.testing.assert_close(
        alt_cpu, torch.tensor(90.0, dtype=torch.float64), atol=1e-5, rtol=0.0
    )
    for device in devices:
        az, alt = observed_altaz(
            CatalogState(
                lon=_on_accel(last, device),
                lat=_on_accel(45.0, device),
            ),
            obstime_mjd_utc=_on_accel(utc, device),
            site=site,
            eop=eop,
        )
        assert az.device.type == device
        assert alt.device.type == device
        atol = 1.0e-4 if device == "mps" else 1.0e-8
        torch.testing.assert_close(
            alt.cpu().to(torch.float64), alt_cpu, atol=atol, rtol=0.0
        )
        daz = (
            torch.remainder(az.cpu().to(torch.float64) - az_cpu + 180.0, 360.0) - 180.0
        )
        torch.testing.assert_close(daz, torch.zeros_like(daz), atol=atol, rtol=0.0)


def test_iau2006_and_transform_stay_on_accelerator() -> None:
    devices = _accel_devices()
    if not devices:
        pytest.skip("no CUDA or MPS")
    utc = torch.tensor(60000.0, dtype=torch.float64)
    site = Site(lon_deg=0.0, lat_deg=45.0)
    eop = EarthOrientation(
        dut1_s=0.0, xp_arcsec=0.0, yp_arcsec=0.0, tt_minus_utc_s=69.184
    )
    state_cpu = CatalogState(lon=180.0, lat=20.0)
    az_cpu, alt_cpu = observed_altaz(
        state_cpu, obstime_mjd_utc=utc, site=site, eop=eop, pn="iau2006"
    )
    xyz_cpu = icrs_to_itrs_xyz(state_cpu, obstime_mjd_utc=utc, eop=eop, pn="iau2006")
    lon_cpu, lat_cpu = transform(10.0, 20.0, from_frame="ICRS", to_frame="GALACTIC")
    for device in devices:
        state = CatalogState(
            lon=_on_accel(180.0, device),
            lat=_on_accel(20.0, device),
        )
        az, alt = observed_altaz(
            state,
            obstime_mjd_utc=_on_accel(utc, device),
            site=site,
            eop=eop,
            pn="iau2006",
        )
        assert az.device.type == device
        atol = 1.0e-3 if device == "mps" else 1.0e-8
        torch.testing.assert_close(
            alt.cpu().to(torch.float64), alt_cpu, atol=atol, rtol=0.0
        )
        daz = (
            torch.remainder(az.cpu().to(torch.float64) - az_cpu + 180.0, 360.0) - 180.0
        )
        torch.testing.assert_close(daz, torch.zeros_like(daz), atol=atol, rtol=0.0)
        xyz = icrs_to_itrs_xyz(
            state, obstime_mjd_utc=_on_accel(utc, device), eop=eop, pn="iau2006"
        )
        assert xyz.device.type == device
        torch.testing.assert_close(
            xyz.cpu().to(torch.float64), xyz_cpu, atol=atol, rtol=0.0
        )
        lon, lat = transform(
            _on_accel(10.0, device),
            _on_accel(20.0, device),
            from_frame="ICRS",
            to_frame="GALACTIC",
        )
        assert lon.device.type == device
        torch.testing.assert_close(
            lon.cpu().to(torch.float64), lon_cpu, atol=atol, rtol=0.0
        )
        torch.testing.assert_close(
            lat.cpu().to(torch.float64), lat_cpu, atol=atol, rtol=0.0
        )
