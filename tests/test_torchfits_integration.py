"""Cross-repo integration: torchsky consumes torchfits for I/O."""

from __future__ import annotations

from pathlib import Path


_FIXTURE = str(Path(__file__).resolve().parent / "data" / "radio_mm_stub.fits")


def test_torchsky_reads_fixture_via_torchfits() -> None:
    """torchsky.io delegation to torchfits works."""
    from torchsky.io import get_header

    hdr = get_header(_FIXTURE, 0)
    assert dict(hdr)["SIMPLE"] is True


def test_torchsky_wcs_from_torchfits_header() -> None:
    """WCS constructed from a torchfits-read header works."""
    from torchsky.io import get_header
    from torchsky.wcs import WCS

    hdr = get_header(_FIXTURE, 0)
    d = dict(hdr)
    d.update({
        "NAXIS": 2,
        "NAXIS1": 100,
        "NAXIS2": 200,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 50.0,
        "CRPIX2": 100.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    })
    wcs = WCS(d)
    ra, dec = wcs.pixel_to_world(50.0, 100.0)
    # Verify the transform is sensible
    assert abs(float(ra) - 180.0) < 0.1
    assert abs(float(dec) - 0.0) < 0.1
