from __future__ import annotations

import pytest

from torchsky.instrument import NIRCAM_UPDEC_BANDS, nircam_imaging_band


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
