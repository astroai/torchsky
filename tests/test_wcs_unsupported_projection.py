import pytest

from torchsky.wcs import WCS


def _hdr(c1, c2):
    return {
        "NAXIS": 2,
        "CTYPE1": c1,
        "CTYPE2": c2,
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 10.0,
        "CRVAL2": 5.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }


@pytest.mark.parametrize(
    "c1,c2", [("RA---XYZ", "DEC--XYZ"), ("GLON-QQQ", "GLAT-QQQ")]
)
def test_unsupported_celestial_projection_raises(c1, c2):
    with pytest.raises(ValueError) as exc:
        WCS(_hdr(c1, c2))
    msg = str(exc.value)
    assert "CTYPE1" in msg and "TAN" in msg


@pytest.mark.parametrize(
    "c1,c2",
    [
        ("RA---TAN", "DEC--TAN"),
        ("RA---TAN-SIP", "DEC--TAN-SIP"),
        ("GLON-CAR", "GLAT-CAR"),
        ("", ""),
        ("LINEAR", "LINEAR"),
        ("X", "Y"),
        ("WAVE", "LINEAR"),
    ],
)
def test_supported_and_non_celestial_axes_construct(c1, c2):
    WCS(_hdr(c1, c2))
