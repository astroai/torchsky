import numpy as np
import torch
from astropy.time import Time

import pytest
import torchsky

from torchsky.wcs.temporal import (
    TemporalWCS,
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


def test_temporal_utc_leap_day_fraction_matches_astropy_time() -> None:
    mjd_utc = torch.tensor(
        [57203.0, 57203.5, 57203.99999, 57753.0, 57753.5, 57753.99999],
        dtype=torch.float64,
    )
    astropy_utc = Time(mjd_utc.numpy(), format="mjd", scale="utc")

    tai = convert_time_scale(mjd_utc, "UTC", "TAI")
    tt = convert_time_scale(mjd_utc, "UTC", "TT")
    utc_back = convert_time_scale(tai, "TAI", "UTC")

    np.testing.assert_allclose(tai.numpy(), astropy_utc.tai.mjd, atol=2e-12)
    np.testing.assert_allclose(tt.numpy(), astropy_utc.tt.mjd, atol=2e-12)
    np.testing.assert_allclose(utc_back.numpy(), mjd_utc.numpy(), atol=2e-12)


def test_temporal_scale_conversion_matches_astropy_time() -> None:
    mjd_utc = torch.tensor([57204.0, 57754.0, 60000.25], dtype=torch.float64)
    astropy_utc = Time(mjd_utc.numpy(), format="mjd", scale="utc")

    tai = convert_time_scale(mjd_utc, "UTC", "TAI")
    tt = convert_time_scale(mjd_utc, "UTC", "TT")
    utc_back = convert_time_scale(tai, "TAI", "UTC")

    np.testing.assert_allclose(tai.numpy(), astropy_utc.tai.mjd, atol=2e-12)
    np.testing.assert_allclose(tt.numpy(), astropy_utc.tt.mjd, atol=2e-12)
    np.testing.assert_allclose(utc_back.numpy(), mjd_utc.numpy(), atol=2e-12)


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


def test_temporal_ephemeris_corrections_fail_closed() -> None:
    twcs = TemporalWCS({"MJDREF": 60000.0, "TREFPOS": "BARYCENTER"})
    assert twcs.requires_ephemeris
    try:
        twcs.apply_corrections(
            torch.tensor([60000.0]), torch.tensor([0.0]), torch.tensor([0.0])
        )
    except NotImplementedError as exc:
        assert "ephemeris" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("barycentric temporal correction must fail closed")


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
