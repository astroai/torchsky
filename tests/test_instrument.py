from __future__ import annotations

import pytest

from torchsky.instrument import (
    NIRCAM_UPDEC_BANDS,
    jwst_exposure_timing,
    nircam_imaging_band,
)


def test_nircam_updec_metadata_uses_official_channel_sampling() -> None:
    short = nircam_imaging_band("f090w")
    long = nircam_imaging_band("F444W")

    assert NIRCAM_UPDEC_BANDS == (
        "F090W",
        "F150W",
        "F200W",
        "F277W",
        "F356W",
        "F444W",
    )
    assert short.detector_pixel_scale_arcsec == 0.031
    assert long.detector_pixel_scale_arcsec == 0.063
    assert short.detector_nyquist_period_arcsec == 0.062
    assert long.half_power_min_angstrom < long.pivot_wavelength_angstrom
    assert long.pivot_wavelength_angstrom < long.half_power_max_angstrom
    assert long.empirical_psf_fwhm_arcsec == 0.145
    assert long.empirical_psf_fwhm_pixels == pytest.approx(2.3015873)


def test_nircam_metadata_rejects_unknown_filter() -> None:
    with pytest.raises(KeyError, match="unsupported NIRCam"):
        nircam_imaging_band("F999W")


def test_jwst_exposure_timing_prefers_consistent_official_midpoint() -> None:
    timing = jwst_exposure_timing(
        {"EXPSTART": 60000.0, "EXPMID": 60000.0005, "EXPEND": 60000.001},
        {"MJD-BEG": 60000.0, "MJD-MID": 60000.0005, "MJD-END": 60000.001},
    )

    assert timing.midpoint_mjd_utc == 60000.0005
    assert timing.midpoint_source == "SCI:MJD-MID"
    assert timing.duration_seconds == pytest.approx(86.4)


def test_jwst_exposure_timing_derives_missing_midpoint() -> None:
    timing = jwst_exposure_timing(
        {"EXPSTART": 60000.0, "EXPEND": 60000.002},
    )

    assert timing.midpoint_mjd_utc == pytest.approx(60000.001)
    assert timing.midpoint_source == "derived_start_end"


@pytest.mark.parametrize(
    ("primary", "science", "message"),
    [
        (
            {"EXPMID": 60000.0},
            {"MJD-MID": 60001.0},
            "conflicting JWST midpoint aliases",
        ),
        (
            {"EXPSTART": 60001.0, "EXPEND": 60000.0},
            None,
            "end precedes start",
        ),
    ],
)
def test_jwst_exposure_timing_fails_closed(
    primary: dict[str, float],
    science: dict[str, float] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        jwst_exposure_timing(primary, science)
