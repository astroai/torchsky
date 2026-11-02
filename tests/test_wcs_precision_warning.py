"""PrecisionWarning for float32 (MPS) WCS whose grid float32 cannot resolve."""

import warnings

import pytest
import torch

from torchsky.wcs import WCS, PrecisionWarning


def _wcs(scale_arcsec: float, dec: float = 0.0, naxis: int = 2000) -> WCS:
    s = scale_arcsec / 3600.0
    return WCS(
        {
            "NAXIS": 2,
            "NAXIS1": naxis,
            "NAXIS2": naxis,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": naxis / 2.0,
            "CRPIX2": naxis / 2.0,
            "CRVAL1": 10.0,
            "CRVAL2": dec,
            "CDELT1": -s,
            "CDELT2": s,
        }
    )


def test_error_estimate_is_float32_ulp_over_pixel_scale():
    # ulp32(360 deg) = 2**-15 deg = 111 mas: 0.55 px at 0.2"/px, 0.11 px at 1"/px.
    assert _wcs(0.2)._mps_float32_error_pixels() == pytest.approx(0.55, abs=0.02)
    assert _wcs(1.0)._mps_float32_error_pixels() == pytest.approx(0.11, abs=0.02)
    # longitude error shrinks with cos(dec): 60 deg halves the angular error
    assert _wcs(0.2, dec=60.0)._mps_float32_error_pixels() < 0.30


def test_warns_once_when_error_reaches_half_a_pixel():
    wcs = _wcs(0.2)
    with pytest.warns(PrecisionWarning, match="float32"):
        wcs._warn_mps_precision()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wcs._warn_mps_precision()  # once per WCS


def test_silent_when_float32_resolves_the_pixel():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _wcs(1.0)._warn_mps_precision()
        _wcs(0.2, dec=60.0)._warn_mps_precision()


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
def test_moving_to_mps_warns_for_sub_arcsecond_pixels():
    with pytest.warns(PrecisionWarning):
        _wcs(0.2).to("mps")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        _wcs(1.0).to("mps")
