import numpy as np
import torch

import pytest
import torchsky

from torchsky.wcs.temporal import (
    TemporalWCS,
    canonical_time_reference_position,
    convert_time_scale,
    iso_to_mjd,
    tai_minus_utc_seconds,
    temporal_unit_to_days,
)


def test_temporal_seconds_roundtrip() -> None:
    twcs = TemporalWCS({"MJDREF": 50000.0, "TIMEUNIT": "s", "TIMESYS": "UTC"})

    t = torch.tensor([0.0, 86400.0, 43200.0], dtype=torch.float64)
    mjd = twcs.to_mjd(t)

    expected = torch.tensor([50000.0, 50001.0, 50000.5], dtype=torch.float64)
    assert torch.allclose(mjd, expected)  # type: ignore

    t_back = twcs.from_mjd(mjd)
    assert torch.allclose(t, t_back)  # type: ignore


def test_temporal_mjdrefi_mjdreff_support() -> None:
    twcs = TemporalWCS({"MJDREFI": 58000, "MJDREFF": 0.125, "TIMEUNIT": "d"})

    t = torch.tensor([0.0, 2.0], dtype=torch.float64)
    mjd = twcs.to_mjd(t)

    np.testing.assert_allclose(
        mjd.cpu().numpy(),
        np.array([58000.125, 58002.125]),
        atol=1e-12,  # type: ignore
    )


def test_temporal_year_and_century_units() -> None:
    year_wcs = TemporalWCS({"MJDREF": 0.0, "TIMEUNIT": "a"})
    century_wcs = TemporalWCS({"MJDREF": 0.0, "TIMEUNIT": "cy"})

    one_year = torch.tensor([1.0], dtype=torch.float64)
    one_century = torch.tensor([1.0], dtype=torch.float64)

    np.testing.assert_allclose(
        year_wcs.to_mjd(one_year).cpu().numpy(),
        np.array([365.25]),
        atol=1e-12,  # type: ignore
    )
    np.testing.assert_allclose(
        century_wcs.to_mjd(one_century).cpu().numpy(),  # type: ignore
        np.array([36525.0]),
        atol=1e-12,  # type: ignore
    )


def test_temporal_common_subday_units() -> None:
    hour_wcs = TemporalWCS({"MJDREF": 60000.0, "TIMEUNIT": "h"})
    minute_wcs = TemporalWCS({"MJDREF": 60000.0, "TIMEUNIT": "min"})
    ms_wcs = TemporalWCS({"MJDREF": 60000.0, "TIMEUNIT": "ms"})

    np.testing.assert_allclose(
        hour_wcs.to_mjd(torch.tensor([12.0], dtype=torch.float64)).numpy(),
        np.array([60000.5]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        minute_wcs.to_mjd(torch.tensor([720.0], dtype=torch.float64)).numpy(),
        np.array([60000.5]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ms_wcs.to_mjd(torch.tensor([1000.0], dtype=torch.float64)).numpy(),
        np.array([60000.0 + 1.0 / 86400.0]),
        atol=1e-12,
    )
    assert temporal_unit_to_days("hour") == 3600.0 / 86400.0
    with pytest.raises(ValueError, match="unsupported temporal unit"):
        temporal_unit_to_days("fortnight")


def test_temporal_reference_position_aliases_fail_closed() -> None:
    assert canonical_time_reference_position(None) == "TOPOCENTER"
    assert canonical_time_reference_position("topocentre") == "TOPOCENTER"
    assert canonical_time_reference_position("GEO") == "GEOCENTER"
    assert canonical_time_reference_position("BARYCENTRE") == "BARYCENTER"
    assert canonical_time_reference_position("SSB") == "BARYCENTER"
    assert canonical_time_reference_position("HELIOCENTRE") == "HELIOCENTER"
    assert canonical_time_reference_position("EMBARYCENTER") == "EMBARYCENTER"
    assert canonical_time_reference_position("MARS") == "MARS"
    assert canonical_time_reference_position("MOON") == "MOON"
    with pytest.raises(ValueError, match="unsupported temporal reference position"):
        canonical_time_reference_position("PLUTO")


def test_temporal_jd_mjd_conversion_and_iso_output() -> None:
    twcs = TemporalWCS({"MJDREF": 0.0})

    mjd = torch.tensor([0.0, 1.0], dtype=torch.float64)
    jd = twcs.mjd_to_jd(mjd)
    mjd_back = twcs.jd_to_mjd(jd)

    assert torch.allclose(mjd, mjd_back)  # type: ignore

    iso = twcs.to_iso8601(torch.tensor([0.0], dtype=torch.float64))
    assert iso[0].startswith("1858-11-17")  # type: ignore


def test_temporal_tai_minus_utc_leap_table_current_offset() -> None:
    mjd = torch.tensor([57753.999, 57754.0, 61049.0], dtype=torch.float64)
    offsets = tai_minus_utc_seconds(mjd)

    np.testing.assert_allclose(offsets.numpy(), np.array([36.999, 37.0, 37.0]))






def test_temporal_header_scale_helpers() -> None:
    twcs = TemporalWCS({"MJDREF": 57754.0, "TIMEUNIT": "s", "TIMESYS": "UTC"})
    rel = torch.tensor([0.0, 86400.0], dtype=torch.float64)

    tai = twcs.to_mjd_scale(rel, "TAI")
    rel_back = twcs.from_mjd_scale(tai, "TAI")

    np.testing.assert_allclose(
        tai.numpy(),
        np.array([57754.0, 57755.0]) + 37.0 / 86400.0,
        atol=2e-12,
    )
    torch.testing.assert_close(rel_back, rel)


def test_temporal_ephemeris_corrections_native() -> None:
    # Test topocenter (no corrections applied)
    twcs_topo = TemporalWCS({"MJDREF": 60000.0, "TREFPOS": "TOPOCENTER"})
    mjd = torch.tensor([60000.0])
    ra = torch.tensor([120.0])
    dec = torch.tensor([30.0])
    torch.testing.assert_close(twcs_topo.apply_corrections(mjd, ra, dec), mjd)

    # Test BARYCENTER (should apply light travel time corrections)
    twcs_bary = TemporalWCS(
        {
            "MJDREF": 60000.0,
            "TREFPOS": "BARYCENTER",
            "OBSGEO-X": 6378137.0,
            "OBSGEO-Y": 0.0,
            "OBSGEO-Z": 0.0,
        }
    )
    assert twcs_bary.requires_ephemeris
    mjd_corr = twcs_bary.apply_corrections(mjd, ra, dec)

    delta_days = (mjd_corr - mjd).item()
    assert abs(delta_days) < 0.01

    assert torch.isfinite(mjd_corr).all()


def test_temporal_geocenter_reference_requires_no_ephemeris() -> None:
    twcs = TemporalWCS({"MJDREF": 60000.0, "TREFPOS": "GEOCENTER"})
    mjd = torch.tensor([60000.0, 60001.0], dtype=torch.float64)
    ra = torch.tensor([0.0, 120.0], dtype=torch.float64)
    dec = torch.tensor([0.0, 30.0], dtype=torch.float64)

    assert not twcs.requires_ephemeris
    torch.testing.assert_close(twcs.apply_corrections(mjd, ra, dec), mjd)






def test_temporal_reference_metadata_and_observatory_location() -> None:
    twcs = TemporalWCS(
        {
            "MJDREF": 60000.0,
            "TIMESYS": "TT",
            "TREFPOS": "TOPOCENTER",
            "TREFDIR": "SOURCE",
            "PLEPHEM": "DE440",
            "OBSGEO-X": 1.0,
            "OBSGEO-Y": 2.0,
            "OBSGEO-Z": 3.0,
        }
    )

    assert twcs.reference_position == "TOPOCENTER"
    assert twcs.reference_direction == "SOURCE"
    assert twcs.ephemeris == "DE440"
    assert twcs.has_observatory_location
    assert twcs.observatory_xyz_m == (1.0, 2.0, 3.0)
    assert twcs.observatory_lon_lat_height_deg_m is None


def test_temporal_date_metadata_parses_mjd_and_iso_dates() -> None:
    twcs = TemporalWCS(
        {
            "DATE-OBS": "2020-01-01T12:00:00",
            "MJD-BEG": 58849.0,
            "DATE-END": "2020-01-02T00:00:00Z",
        }
    )

    np.testing.assert_allclose(twcs.mjd_obs, 58849.5, atol=1e-12)
    assert twcs.mjd_beg == 58849.0
    np.testing.assert_allclose(twcs.mjd_end, 58850.0, atol=1e-12)
    np.testing.assert_allclose(iso_to_mjd("2020-01-01"), 58849.0, atol=1e-12)


def test_temporal_duration_and_midpoint_metadata() -> None:
    interval = TemporalWCS({"MJD-BEG": 60000.0, "MJD-END": 60000.25})
    exposure = TemporalWCS({"MJD-OBS": 60000.0, "EXPOSURE": 3600.0})
    telapse = TemporalWCS({"MJD-OBS": 60000.0, "TIMEUNIT": "s", "TELAPSE": 7200.0})

    assert interval.duration_days == 0.25
    assert interval.midpoint_mjd == 60000.125
    np.testing.assert_allclose(exposure.duration_days, 3600.0 / 86400.0)
    np.testing.assert_allclose(exposure.midpoint_mjd, 60000.0 + 1800.0 / 86400.0)
    np.testing.assert_allclose(telapse.duration_days, 7200.0 / 86400.0)


def test_wcs_temporal_axis_metadata_view() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "TIME",
        "CUNIT1": "s",
        "CRPIX1": 1.0,
        "CRVAL1": 60000.0,
        "CDELT1": 1.0,
        "TIMESYS": "UTC",
        "DATE-OBS": "2023-01-01T00:00:00",
        "EXPOSURE": 30.0,
    }
    wcs = torchsky.WCS(header)
    twcs = wcs.temporal_wcs()

    assert wcs.temporal_axis_indices == (0,)
    assert twcs.timeunit == "s"
    assert twcs.timesys == "UTC"
    np.testing.assert_allclose(twcs.mjd_ref, 60000.0)
    np.testing.assert_allclose(twcs.mjd_obs, 59945.0)
    np.testing.assert_allclose(twcs.duration_days, 30.0 / 86400.0)


def test_trefpos_embarycenter_and_moon_are_geocenter() -> None:
    mjd = torch.tensor([60000.0], dtype=torch.float64)
    ra = torch.tensor([120.0], dtype=torch.float64)
    dec = torch.tensor([30.0], dtype=torch.float64)
    for name in ("EMBARYCENTER", "MOON", "CUSTOM", "BASECENTER"):
        twcs = TemporalWCS({"MJDREF": 60000.0, "TREFPOS": name})
        torch.testing.assert_close(twcs.apply_corrections(mjd, ra, dec), mjd)
        assert not twcs.requires_ephemeris


def test_trefpos_mars_keplerian_delay_differs_from_barycenter() -> None:
    mjd = torch.tensor([60000.0], dtype=torch.float64)
    ra = torch.tensor([0.0], dtype=torch.float64)
    dec = torch.tensor([0.0], dtype=torch.float64)
    mars = TemporalWCS({"MJDREF": 60000.0, "TREFPOS": "MARS"})
    bary = TemporalWCS({"MJDREF": 60000.0, "TREFPOS": "BARYCENTER"})
    assert mars.requires_ephemeris
    corr_m = mars.apply_corrections(mjd, ra, dec)
    corr_b = bary.apply_corrections(mjd, ra, dec)
    assert not torch.allclose(corr_m, corr_b, atol=1e-8, rtol=0.0)
    delta_s = float((corr_m - mjd).item()) * 86400.0
    assert abs(delta_s) < 3600.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_convert_time_scale_cuda_matches_cpu() -> None:
    mjd = torch.linspace(57204.0, 61049.0, 32, dtype=torch.float64)
    cpu = convert_time_scale(mjd, "UTC", "TT")
    gpu = convert_time_scale(mjd.cuda(), "UTC", "TT")
    assert gpu.device.type == "cuda"
    torch.testing.assert_close(gpu.cpu(), cpu, atol=1e-15, rtol=0.0)


@pytest.mark.parametrize(
    "trefpos,trefdir",
    [
        ("TOPOCENTER", None),
        ("GEOCENTER", None),
        ("BARYCENTER", "H"),
        ("HELIOCENTER", "H"),
        ("EMBARYCENTER", None),
    ],
)
def test_paper4_reference_position_fixtures_are_named(
    trefpos: str, trefdir: str | None
) -> None:
    header: dict[str, float | str] = {
        "MJDREF": 60000.0,
        "TIMESYS": "UTC",
        "TREFPOS": trefpos,
    }
    if trefdir is not None:
        header["TREFDIR"] = trefdir
    twcs = TemporalWCS(header)
    assert twcs.trefpos == trefpos
    assert twcs.trefdir == trefdir
    mjd = torch.tensor([60000.0], dtype=torch.float64)
    ra = torch.tensor([120.0], dtype=torch.float64)
    dec = torch.tensor([30.0], dtype=torch.float64)
    corr = twcs.apply_corrections(mjd, ra, dec)
    assert torch.isfinite(corr).all()
