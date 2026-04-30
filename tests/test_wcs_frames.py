import numpy as np
import torch
from astropy.coordinates import FK5, Galactic, ICRS, SkyCoord
from astropy.time import Time
import astropy.units as u

from torchsky.wcs import WCS, convert_celestial, galactic_to_icrs, icrs_to_galactic, precess_fk5


def _wrap_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


def test_icrs_galactic_matches_astropy() -> None:
    ra = torch.tensor([0.0, 83.6331, 201.2983, 359.0], dtype=torch.float64)
    dec = torch.tensor([0.0, 22.0145, -43.0191, -20.0], dtype=torch.float64)

    l_t, b_t = icrs_to_galactic(ra, dec)
    sky = SkyCoord(ra=ra.numpy() * u.deg, dec=dec.numpy() * u.deg, frame=ICRS())

    np.testing.assert_allclose(
        _wrap_delta(l_t.numpy(), sky.galactic.l.deg), 0.0, atol=2e-5
    )
    np.testing.assert_allclose(b_t.numpy(), sky.galactic.b.deg, atol=2e-5)

    ra_back, dec_back = galactic_to_icrs(l_t, b_t)
    np.testing.assert_allclose(_wrap_delta(ra_back.numpy(), ra.numpy()), 0.0, atol=2e-10)
    np.testing.assert_allclose(dec_back.numpy(), dec.numpy(), atol=2e-10)


def test_fk5_precession_matches_astropy_common_epoch_range() -> None:
    ra = torch.tensor([10.0, 120.0, 250.0], dtype=torch.float64)
    dec = torch.tensor([-30.0, 5.0, 60.0], dtype=torch.float64)

    ra_t, dec_t = precess_fk5(ra, dec, from_equinox=2000.0, to_equinox=2050.0)
    sky = SkyCoord(
        ra=ra.numpy() * u.deg,
        dec=dec.numpy() * u.deg,
        frame=FK5(equinox=Time("J2000")),
    ).transform_to(FK5(equinox=Time("J2050")))

    np.testing.assert_allclose(_wrap_delta(ra_t.numpy(), sky.ra.deg), 0.0, atol=5e-5)
    np.testing.assert_allclose(dec_t.numpy(), sky.dec.deg, atol=5e-5)


def test_convert_celestial_fk5_to_galactic_matches_astropy() -> None:
    ra = torch.tensor([15.0, 155.0], dtype=torch.float64)
    dec = torch.tensor([-10.0, 35.0], dtype=torch.float64)

    l_t, b_t = convert_celestial(
        ra,
        dec,
        from_frame="FK5",
        to_frame="GALACTIC",
        from_equinox=2050.0,
    )
    sky = SkyCoord(
        ra=ra.numpy() * u.deg,
        dec=dec.numpy() * u.deg,
        frame=FK5(equinox=Time("J2050")),
    ).transform_to(Galactic())

    np.testing.assert_allclose(_wrap_delta(l_t.numpy(), sky.l.deg), 0.0, atol=2e-4)
    np.testing.assert_allclose(b_t.numpy(), sky.b.deg, atol=2e-4)


def test_wcs_celestial_frame_metadata_and_helpers() -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 10.0,
        "CRVAL2": -10.0,
        "CDELT1": -0.1,
        "CDELT2": 0.1,
        "RADESYS": "FK5",
        "EQUINOX": 2050.0,
    }
    wcs = WCS(header)

    assert wcs.celestial_frame == "FK5"
    assert wcs.celestial_equinox == 2050.0

    ra = torch.tensor([10.0], dtype=torch.float64)
    dec = torch.tensor([-10.0], dtype=torch.float64)
    ra_2000, dec_2000 = wcs.precess_celestial(ra, dec, to_equinox=2000.0)
    exp = SkyCoord(
        ra=ra.numpy() * u.deg,
        dec=dec.numpy() * u.deg,
        frame=FK5(equinox=Time("J2050")),
    ).transform_to(FK5(equinox=Time("J2000")))

    np.testing.assert_allclose(
        _wrap_delta(ra_2000.numpy(), exp.ra.deg), 0.0, atol=5e-5
    )
    np.testing.assert_allclose(dec_2000.numpy(), exp.dec.deg, atol=5e-5)
