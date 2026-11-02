from __future__ import annotations

import erfa
import numpy as np
import pytest
import torch

from torchsky.wcs import CatalogState, EarthOrientation, Site, observed_altaz, transform
from torchsky.wcs.frames import ARCSEC_TO_RAD
from torchsky.wcs.iau2000a import iau2000a_nutation, iau2006_nutation
from torchsky.wcs.iau2006 import cirs_matrix, npb_matrix

UTC_MJD = 60000.0
DUT1_S = 0.184
TT_MINUS_UTC_S = 69.184
TT_MJD = UTC_MJD + TT_MINUS_UTC_S / 86400.0
JD0 = 2400000.5


def _wrap_delta_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


def test_iau2000a_nutation_matches_erfa() -> None:
    tt = torch.tensor([51544.5, TT_MJD, 58000.0], dtype=torch.float64)
    dpsi, deps = iau2000a_nutation(tt)
    ref = np.array([erfa.nut00a(JD0, float(m)) for m in tt.numpy()])
    dpsi_as = (dpsi.numpy() - ref[:, 0]) / ARCSEC_TO_RAD
    deps_as = (deps.numpy() - ref[:, 1]) / ARCSEC_TO_RAD
    assert float(np.max(np.abs(dpsi_as))) < 1.0e-7
    assert float(np.max(np.abs(deps_as))) < 1.0e-7


def test_iau2006_nutation_matches_erfa() -> None:
    tt = torch.tensor(TT_MJD, dtype=torch.float64)
    dpsi, deps = iau2006_nutation(tt)
    r_dpsi, r_deps = erfa.nut06a(JD0, TT_MJD)
    assert abs(float(dpsi) - r_dpsi) / ARCSEC_TO_RAD < 1.0e-7
    assert abs(float(deps) - r_deps) / ARCSEC_TO_RAD < 1.0e-7


def test_npb_and_cirs_matrices_match_erfa() -> None:
    tt = torch.tensor(TT_MJD, dtype=torch.float64)
    np.testing.assert_allclose(
        npb_matrix(tt).numpy(), erfa.pnm06a(JD0, TT_MJD), atol=1e-13, rtol=0.0
    )
    np.testing.assert_allclose(
        cirs_matrix(tt).numpy(), erfa.c2i06a(JD0, TT_MJD), atol=1e-13, rtol=0.0
    )


def test_iau2006_altaz_handedness_matches_erfa_c2tcio() -> None:
    ra = np.array([10.0, 83.6331, 201.2983], dtype=np.float64)
    dec = np.array([20.0, 22.0145, -43.0191], dtype=np.float64)
    site_lon = np.array([-70.4, 0.0, 149.1], dtype=np.float64)
    site_lat = np.array([-24.6, 51.5, -35.3], dtype=np.float64)
    xp_arcsec = 0.12
    yp_arcsec = -0.21
    eop = EarthOrientation(
        dut1_s=DUT1_S,
        xp_arcsec=xp_arcsec,
        yp_arcsec=yp_arcsec,
        tt_minus_utc_s=TT_MINUS_UTC_S,
    )
    az, alt = observed_altaz(
        CatalogState(lon=ra, lat=dec),
        obstime_mjd_utc=UTC_MJD,
        site=Site(lon_deg=site_lon, lat_deg=site_lat),
        eop=eop,
        pn="iau2006",
    )

    ut1_mjd = UTC_MJD + DUT1_S / 86400.0
    rpom = erfa.pom00(
        xp_arcsec * ARCSEC_TO_RAD,
        yp_arcsec * ARCSEC_TO_RAD,
        0.0,
    )
    rotation = erfa.c2tcio(erfa.c2i06a(JD0, TT_MJD), erfa.era00(JD0, ut1_mjd), rpom)
    lon = np.deg2rad(ra)
    lat = np.deg2rad(dec)
    icrs = np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)),
        axis=-1,
    )
    itrs = np.einsum("ij,nj->ni", rotation, icrs)
    lon_site = np.deg2rad(site_lon)
    lat_site = np.deg2rad(site_lat)
    east = np.stack(
        (-np.sin(lon_site), np.cos(lon_site), np.zeros_like(lon_site)), axis=-1
    )
    north = np.stack(
        (
            -np.sin(lat_site) * np.cos(lon_site),
            -np.sin(lat_site) * np.sin(lon_site),
            np.cos(lat_site),
        ),
        axis=-1,
    )
    up = np.stack(
        (
            np.cos(lat_site) * np.cos(lon_site),
            np.cos(lat_site) * np.sin(lon_site),
            np.sin(lat_site),
        ),
        axis=-1,
    )
    expected_az = (
        np.rad2deg(
            np.arctan2(np.sum(itrs * east, axis=-1), np.sum(itrs * north, axis=-1))
        )
        % 360.0
    )
    expected_alt = np.rad2deg(np.arcsin(np.sum(itrs * up, axis=-1)))

    np.testing.assert_allclose(
        _wrap_delta_deg(az.numpy(), expected_az), 0.0, atol=2.0e-10
    )
    np.testing.assert_allclose(alt.numpy(), expected_alt, atol=2.0e-10)


def test_gcrs_without_aberration_is_icrs() -> None:
    eop = EarthOrientation(dut1_s=DUT1_S, tt_minus_utc_s=TT_MINUS_UTC_S)
    ra = torch.tensor([10.0, 83.6331, 201.2983], dtype=torch.float64)
    dec = torch.tensor([20.0, 22.0145, -43.0191], dtype=torch.float64)
    lon, lat = transform(
        ra, dec, from_frame="ICRS", to_frame="GCRS", obstime_mjd_utc=UTC_MJD, eop=eop
    )
    np.testing.assert_allclose(
        _wrap_delta_deg(lon.numpy(), ra.numpy()), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(lat.numpy(), dec.numpy(), atol=1e-12)


def test_cirs_tete_match_erfa_matrices() -> None:
    eop = EarthOrientation(dut1_s=DUT1_S, tt_minus_utc_s=TT_MINUS_UTC_S)
    ra = torch.tensor([10.0, 83.6331, 201.2983, 359.0], dtype=torch.float64)
    dec = torch.tensor([20.0, 22.0145, -43.0191, -20.0], dtype=torch.float64)
    lon = ra * np.pi / 180.0
    lat = dec * np.pi / 180.0
    clat = torch.cos(lat)
    xyz = torch.stack(
        (clat * torch.cos(lon), clat * torch.sin(lon), torch.sin(lat)), dim=-1
    )
    r_cirs = torch.as_tensor(erfa.c2i06a(JD0, TT_MJD), dtype=torch.float64)
    r_tete = torch.as_tensor(erfa.pnm06a(JD0, TT_MJD), dtype=torch.float64)
    pred_cirs = (r_cirs @ xyz.T).T
    pred_tete = (r_tete @ xyz.T).T

    lon_c, lat_c = transform(
        ra, dec, from_frame="ICRS", to_frame="CIRS", obstime_mjd_utc=UTC_MJD, eop=eop
    )
    lon_t, lat_t = transform(
        ra, dec, from_frame="ICRS", to_frame="TETE", obstime_mjd_utc=UTC_MJD, eop=eop
    )
    got_c = _lonlat_to_xyz(lon_c, lat_c)
    got_t = _lonlat_to_xyz(lon_t, lat_t)
    sep_cirs = _sep_arcsec(got_c, pred_cirs)
    sep_tete = _sep_arcsec(got_t, pred_tete)
    assert float(sep_cirs.max()) < 1.0e-6
    assert float(sep_tete.max()) < 1.0e-6


def test_observed_frames_require_eop() -> None:
    with pytest.raises(ValueError, match="GCRS requires"):
        transform(10.0, 20.0, from_frame="ICRS", to_frame="GCRS")
    with pytest.raises(ValueError, match="tt_minus_utc"):
        transform(
            10.0,
            20.0,
            from_frame="ICRS",
            to_frame="CIRS",
            obstime_mjd_utc=UTC_MJD,
            eop=EarthOrientation(dut1_s=DUT1_S),
        )


def test_gcrs_aberration_shifts_mas() -> None:
    eop = EarthOrientation(dut1_s=DUT1_S, tt_minus_utc_s=TT_MINUS_UTC_S)
    ra = torch.tensor(83.6331, dtype=torch.float64)
    dec = torch.tensor(22.0145, dtype=torch.float64)
    lon0, lat0 = transform(
        ra, dec, from_frame="ICRS", to_frame="GCRS", obstime_mjd_utc=UTC_MJD, eop=eop
    )
    lon1, lat1 = transform(
        ra,
        dec,
        from_frame="ICRS",
        to_frame="GCRS",
        obstime_mjd_utc=UTC_MJD,
        eop=eop,
        aberration=True,
    )
    daz = _wrap_delta_deg(lon1.numpy(), lon0.numpy())
    sep = float(
        np.hypot(daz * np.cos(np.deg2rad(lat0.numpy())), lat1.numpy() - lat0.numpy())
    )
    assert sep * 3600.0 > 5.0
    assert sep * 3600.0 < 30.0


def _lonlat_to_xyz(lon_deg: torch.Tensor, lat_deg: torch.Tensor) -> torch.Tensor:
    lon = lon_deg * np.pi / 180.0
    lat = lat_deg * np.pi / 180.0
    clat = torch.cos(lat)
    return torch.stack(
        (clat * torch.cos(lon), clat * torch.sin(lon), torch.sin(lat)), dim=-1
    )


def _sep_arcsec(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    chord = torch.linalg.vector_norm(a - b, dim=-1)
    return 2.0 * torch.asin((chord / 2.0).clamp(max=1.0)) / ARCSEC_TO_RAD


def test_catalog_state_stays_geometric() -> None:
    state = CatalogState(lon=10.0, lat=20.0, frame="ICRS")
    with pytest.raises(ValueError):
        state.to_frame("CIRS")


def test_cirs_tete_cuda_or_mps_matches_cpu() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float64
        atol_arcsec = 1.0e-6
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
        atol_arcsec = 5.0e-2
    else:
        pytest.skip("no CUDA/MPS device")

    eop = EarthOrientation(dut1_s=DUT1_S, tt_minus_utc_s=TT_MINUS_UTC_S)
    ra = torch.tensor([10.0, 83.6331, 201.2983, 359.0], dtype=dtype, device=device)
    dec = torch.tensor([20.0, 22.0145, -43.0191, -20.0], dtype=dtype, device=device)
    lon_c, lat_c = transform(
        ra, dec, from_frame="ICRS", to_frame="CIRS", obstime_mjd_utc=UTC_MJD, eop=eop
    )
    lon_t, lat_t = transform(
        ra, dec, from_frame="ICRS", to_frame="TETE", obstime_mjd_utc=UTC_MJD, eop=eop
    )
    assert lon_c.device.type == device.type
    ra_cpu = ra.detach().cpu().double()
    dec_cpu = dec.detach().cpu().double()
    lon_cpu_c, lat_cpu_c = transform(
        ra_cpu,
        dec_cpu,
        from_frame="ICRS",
        to_frame="CIRS",
        obstime_mjd_utc=UTC_MJD,
        eop=eop,
    )
    lon_cpu_t, lat_cpu_t = transform(
        ra_cpu,
        dec_cpu,
        from_frame="ICRS",
        to_frame="TETE",
        obstime_mjd_utc=UTC_MJD,
        eop=eop,
    )
    sep_c = _sep_arcsec(
        _lonlat_to_xyz(lon_c.cpu().double(), lat_c.cpu().double()),
        _lonlat_to_xyz(lon_cpu_c, lat_cpu_c),
    )
    sep_t = _sep_arcsec(
        _lonlat_to_xyz(lon_t.cpu().double(), lat_t.cpu().double()),
        _lonlat_to_xyz(lon_cpu_t, lat_cpu_t),
    )
    assert float(sep_c.max()) < atol_arcsec
    assert float(sep_t.max()) < atol_arcsec
