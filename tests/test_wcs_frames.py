import erfa
import numpy as np
import pytest
import torch
from astropy.coordinates import FK4, FK5, Galactic, ICRS, SkyCoord
from astropy.time import Time
import astropy.units as u

from torchsky.wcs import (
    WCS,
    convert_celestial,
    galactic_to_icrs,
    icrs_to_galactic,
    precess_fk5,
    propagate_proper_motion,
    propagate_space_motion,
)


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
    np.testing.assert_allclose(
        _wrap_delta(ra_back.numpy(), ra.numpy()), 0.0, atol=2e-10
    )
    np.testing.assert_allclose(dec_back.numpy(), dec.numpy(), atol=2e-10)


def test_propagate_proper_motion_matches_astropy() -> None:
    ra = torch.tensor([10.0, 120.0, 359.0, 1.0], dtype=torch.float64)
    dec = torch.tensor([20.0, -45.0, 89.9, -89.9], dtype=torch.float64)
    pm_ra_cosdec = torch.tensor([100.0, -500.0, 10_000.0, -10_000.0])
    pm_dec = torch.tensor([50.0, 1_000.0, -5_000.0, 5_000.0])
    target_epoch = torch.tensor([2010.0, 2050.0, 2100.0, 2100.0])

    ra_t, dec_t = propagate_proper_motion(
        ra,
        dec,
        pm_ra_cosdec,
        pm_dec,
        source_epoch_jyear=2000.0,
        target_epoch_jyear=target_epoch,
    )
    sky = SkyCoord(
        ra=ra.numpy() * u.deg,
        dec=dec.numpy() * u.deg,
        pm_ra_cosdec=pm_ra_cosdec.numpy() * u.mas / u.yr,
        pm_dec=pm_dec.numpy() * u.mas / u.yr,
        obstime=Time(np.full(ra.shape, 2000.0), format="jyear", scale="tcb"),
        frame=ICRS(),
    ).apply_space_motion(
        new_obstime=Time(target_epoch.numpy(), format="jyear", scale="tcb")
    )

    np.testing.assert_allclose(_wrap_delta(ra_t.numpy(), sky.ra.deg), 0.0, atol=3e-7)
    np.testing.assert_allclose(dec_t.numpy(), sky.dec.deg, atol=1e-8)


def test_propagate_proper_motion_broadcasts_and_is_differentiable() -> None:
    ra = torch.tensor([10.0, 20.0], dtype=torch.float64, requires_grad=True)
    dec = torch.tensor([89.999, -89.999], dtype=torch.float64, requires_grad=True)
    ra_out, dec_out = propagate_proper_motion(
        ra,
        dec,
        1_000.0,
        -500.0,
        source_epoch_jyear=torch.tensor([2000.0, 2010.0]),
        target_epoch_jyear=2025.0,
    )

    assert ra_out.shape == dec_out.shape == (2,)
    (ra_out.sum() + dec_out.sum()).backward()
    assert ra.grad is not None and torch.isfinite(ra.grad).all()
    assert dec.grad is not None and torch.isfinite(dec.grad).all()


def test_propagate_space_motion_matches_erfa_starpm() -> None:
    ra = np.array([10.0, 269.452075, 42.0, 359.9, 339.50019800605236])
    dec = np.array([20.0, 4.693391, -89.5, 70.0, 27.605252088646363])
    pm_ra_cosdec = np.array(
        [100.0, -801.551, 50.0, -30.0, -64_880.75716316896]
    )
    pm_dec = np.array([50.0, 10_362.394, -20.0, 40.0, 16_145.147314046984])
    parallax = np.array([100.0, 548.31, 2.0, 0.5, 3.911354729847825])
    radial_velocity = np.array(
        [20.0, -110.6, 300.0, -200.0, 70_595.04657621654]
    )
    source_epoch = np.array([2000.0, 2000.0, 2050.0, 2030.0, 2000.0])
    target_epoch = np.array([2100.0, 2025.0, 1950.0, 2200.0, 2100.0])

    actual = propagate_space_motion(
        torch.from_numpy(ra),
        torch.from_numpy(dec),
        torch.from_numpy(pm_ra_cosdec),
        torch.from_numpy(pm_dec),
        torch.from_numpy(parallax),
        torch.from_numpy(radial_velocity),
        source_epoch_jyear=torch.from_numpy(source_epoch),
        target_epoch_jyear=torch.from_numpy(target_epoch),
    )

    degrees_to_radians = np.pi / 180.0
    mas_to_radians = degrees_to_radians / 3_600_000.0
    erfa_result = erfa.starpm(
        ra * degrees_to_radians,
        dec * degrees_to_radians,
        pm_ra_cosdec * mas_to_radians / np.cos(dec * degrees_to_radians),
        pm_dec * mas_to_radians,
        parallax / 1000.0,
        radial_velocity,
        2_451_545.0,
        (source_epoch - 2000.0) * 365.25,
        2_451_545.0,
        (target_epoch - 2000.0) * 365.25,
    )
    expected = (
        erfa_result[0] / degrees_to_radians,
        erfa_result[1] / degrees_to_radians,
        erfa_result[2] * np.cos(erfa_result[1]) / mas_to_radians,
        erfa_result[3] / mas_to_radians,
        erfa_result[4] * 1000.0,
        erfa_result[5],
    )

    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value.numpy(), reference, rtol=2e-13, atol=2e-10)


def test_propagate_space_motion_matches_astropy_for_physical_states() -> None:
    ra = torch.tensor([10.0, 269.452075], dtype=torch.float64)
    dec = torch.tensor([20.0, 4.693391], dtype=torch.float64)
    pm_ra_cosdec = torch.tensor([100.0, -801.551], dtype=torch.float64)
    pm_dec = torch.tensor([50.0, 10_362.394], dtype=torch.float64)
    parallax = torch.tensor([100.0, 548.31], dtype=torch.float64)
    radial_velocity = torch.tensor([20.0, -110.6], dtype=torch.float64)

    actual = propagate_space_motion(
        ra,
        dec,
        pm_ra_cosdec,
        pm_dec,
        parallax,
        radial_velocity,
        source_epoch_jyear=2000.0,
        target_epoch_jyear=torch.tensor([2100.0, 2025.0]),
    )
    sky = SkyCoord(
        ra=ra.numpy() * u.deg,
        dec=dec.numpy() * u.deg,
        pm_ra_cosdec=pm_ra_cosdec.numpy() * u.mas / u.yr,
        pm_dec=pm_dec.numpy() * u.mas / u.yr,
        distance=(1000.0 / parallax.numpy()) * u.pc,
        radial_velocity=radial_velocity.numpy() * u.km / u.s,
        obstime=Time(2000.0, format="jyear", scale="tcb"),
        frame=ICRS(),
    ).apply_space_motion(
        new_obstime=Time([2100.0, 2025.0], format="jyear", scale="tcb")
    )
    expected = (
        sky.ra.deg,
        sky.dec.deg,
        sky.pm_ra_cosdec.to_value(u.mas / u.yr),
        sky.pm_dec.to_value(u.mas / u.yr),
        1000.0 / sky.distance.to_value(u.pc),
        sky.radial_velocity.to_value(u.km / u.s),
    )

    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value.numpy(), reference, rtol=2e-10, atol=1e-7)


def test_propagate_space_motion_broadcasts_and_has_finite_jacobian() -> None:
    state = torch.tensor(
        [10.0, 20.0, 100.0, 50.0, 100.0, 20.0], dtype=torch.float64
    )

    def propagate(value: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            propagate_space_motion(
                *value,
                source_epoch_jyear=2000.0,
                target_epoch_jyear=2100.0,
            )
        )

    jacobian = torch.func.jacrev(propagate)(state)
    assert jacobian.shape == (6, 6)
    assert torch.isfinite(jacobian).all()

    broadcast = propagate_space_motion(
        torch.tensor([10.0, 20.0]),
        30.0,
        1.0,
        2.0,
        10.0,
        20.0,
        source_epoch_jyear=2025.0,
        target_epoch_jyear=torch.tensor([2000.0, 2050.0]),
    )
    assert all(value.shape == (2,) for value in broadcast)


def test_propagate_space_motion_rejects_incomplete_or_unphysical_states() -> None:
    with pytest.raises(ValueError, match="parallax_mas must be positive"):
        propagate_space_motion(
            10.0,
            20.0,
            1.0,
            2.0,
            0.0,
            20.0,
            source_epoch_jyear=2000.0,
            target_epoch_jyear=2025.0,
        )

    with pytest.raises(ValueError, match="at least 0.5c"):
        propagate_space_motion(
            10.0,
            20.0,
            1.0e9,
            2.0,
            1.0,
            20.0,
            source_epoch_jyear=2000.0,
            target_epoch_jyear=2025.0,
        )


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


def test_convert_celestial_fk4_to_fk5_matches_astropy() -> None:
    ra = torch.tensor([12.0, 150.0, 275.0], dtype=torch.float64)
    dec = torch.tensor([-20.0, 5.0, 55.0], dtype=torch.float64)

    ra_t, dec_t = convert_celestial(
        ra,
        dec,
        from_frame="FK4",
        to_frame="FK5",
        from_equinox=1950.0,
        to_equinox=2000.0,
    )
    sky = SkyCoord(
        ra=ra.numpy() * u.deg,
        dec=dec.numpy() * u.deg,
        frame=FK4(equinox=Time("B1950"), obstime=Time("B1950")),
    ).transform_to(FK5(equinox=Time("J2000")))

    np.testing.assert_allclose(_wrap_delta(ra_t.numpy(), sky.ra.deg), 0.0, atol=2e-4)
    np.testing.assert_allclose(dec_t.numpy(), sky.dec.deg, atol=2e-4)


def test_convert_celestial_fk5_to_fk4_matches_astropy() -> None:
    ra = torch.tensor([15.0, 155.0, 310.0], dtype=torch.float64)
    dec = torch.tensor([-10.0, 35.0, -65.0], dtype=torch.float64)

    ra_t, dec_t = convert_celestial(
        ra,
        dec,
        from_frame="FK5",
        to_frame="FK4",
        from_equinox=2000.0,
        to_equinox=1950.0,
    )
    sky = SkyCoord(
        ra=ra.numpy() * u.deg,
        dec=dec.numpy() * u.deg,
        frame=FK5(equinox=Time("J2000")),
    ).transform_to(FK4(equinox=Time("B1950"), obstime=Time("B1950")))

    np.testing.assert_allclose(_wrap_delta(ra_t.numpy(), sky.ra.deg), 0.0, atol=2e-4)
    np.testing.assert_allclose(dec_t.numpy(), sky.dec.deg, atol=2e-4)


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

    np.testing.assert_allclose(_wrap_delta(ra_2000.numpy(), exp.ra.deg), 0.0, atol=5e-5)
    np.testing.assert_allclose(dec_2000.numpy(), exp.dec.deg, atol=5e-5)


def test_wcs_convert_celestial_fk4_metadata_matches_astropy() -> None:
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
        "RADESYS": "FK4",
        "EQUINOX": 1950.0,
    }
    wcs = WCS(header)
    ra = torch.tensor([10.0], dtype=torch.float64)
    dec = torch.tensor([-10.0], dtype=torch.float64)

    ra_fk5, dec_fk5 = wcs.convert_celestial(ra, dec, "FK5", to_equinox=2000.0)
    exp = SkyCoord(
        ra=ra.numpy() * u.deg,
        dec=dec.numpy() * u.deg,
        frame=FK4(equinox=Time("B1950"), obstime=Time("B1950")),
    ).transform_to(FK5(equinox=Time("J2000")))

    np.testing.assert_allclose(_wrap_delta(ra_fk5.numpy(), exp.ra.deg), 0.0, atol=2e-4)
    np.testing.assert_allclose(dec_fk5.numpy(), exp.dec.deg, atol=2e-4)


def test_wcs_precess_celestial_fk4_matches_astropy() -> None:
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
        "RADESYS": "FK4",
        "EQUINOX": 1950.0,
    }
    wcs = WCS(header)
    ra = torch.tensor([12.0, 120.0], dtype=torch.float64)
    dec = torch.tensor([-20.0, 40.0], dtype=torch.float64)

    ra_b1975, dec_b1975 = wcs.precess_celestial(ra, dec, to_equinox=1975.0)
    exp = SkyCoord(
        ra=ra.numpy() * u.deg,
        dec=dec.numpy() * u.deg,
        frame=FK4(equinox=Time("B1950"), obstime=Time("B1950")),
    ).transform_to(FK4(equinox=Time("B1975"), obstime=Time("B1975")))

    np.testing.assert_allclose(
        _wrap_delta(ra_b1975.numpy(), exp.ra.deg), 0.0, atol=2e-4
    )
    np.testing.assert_allclose(dec_b1975.numpy(), exp.dec.deg, atol=2e-4)
