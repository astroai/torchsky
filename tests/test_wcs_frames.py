import erfa
import numpy as np
import pytest
import torch
from astropy.coordinates import (
    FK4,
    FK5,
    Galactic,
    Galactocentric,
    ICRS,
    LSR,
    SkyCoord,
    BarycentricMeanEcliptic,
)
from astropy.time import Time
import astropy.units as u

from torchsky.wcs import (
    WCS,
    convert_celestial,
    ecliptic_of_date_to_icrs,
    galactic_to_icrs,
    galactocentric_to_icrs,
    heliocentric_rv_to_lsr,
    icrs_to_ecliptic_of_date,
    icrs_to_galactocentric,
    icrs_to_galactic,
    precess_fk5,
    propagate_proper_motion,
    propagate_proper_motion_with_jacobian,
    propagate_space_motion,
    propagate_space_motion_with_jacobian,
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


def test_icrs_ecliptic_roundtrip_and_astropy() -> None:
    from torchsky.wcs import icrs_to_ecliptic, ecliptic_to_icrs

    ra = torch.tensor([0.0, 83.6331, 201.2983], dtype=torch.float64)
    dec = torch.tensor([0.0, 22.0145, -43.0191], dtype=torch.float64)
    lon, lat = icrs_to_ecliptic(ra, dec)
    ra_back, dec_back = ecliptic_to_icrs(lon, lat)
    np.testing.assert_allclose(
        _wrap_delta(ra_back.numpy(), ra.numpy()), 0.0, atol=1e-10
    )
    np.testing.assert_allclose(dec_back.numpy(), dec.numpy(), atol=1e-10)

    sky = SkyCoord(ra=ra.numpy() * u.deg, dec=dec.numpy() * u.deg, frame=ICRS())
    ecl = sky.transform_to(BarycentricMeanEcliptic(equinox="J2000"))
    np.testing.assert_allclose(
        _wrap_delta(lon.numpy(), ecl.lon.deg), 0.0, atol=2e-5
    )
    np.testing.assert_allclose(lat.numpy(), ecl.lat.deg, atol=2e-5)
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


def test_proper_motion_jacobian_matches_central_difference() -> None:
    state = torch.tensor([359.9999, 20.0, 100.0, 50.0], dtype=torch.float64)
    source_epoch = 2016.0
    target_epoch = 2041.0
    nominal, actual = propagate_proper_motion_with_jacobian(
        *state,
        source_epoch_jyear=source_epoch,
        target_epoch_jyear=target_epoch,
    )
    epsilons = torch.tensor([0.1, 0.1, 1e-3, 1e-3], dtype=torch.float64)
    columns = []
    for column, epsilon in enumerate(epsilons):
        plus = state.clone()
        minus = state.clone()
        if column == 0:
            delta_deg = epsilon / (3_600_000.0 * torch.cos(state[1] * np.pi / 180.0))
            plus[0] += delta_deg
            minus[0] -= delta_deg
        elif column == 1:
            plus[1] += epsilon / 3_600_000.0
            minus[1] -= epsilon / 3_600_000.0
        else:
            plus[column] += epsilon
            minus[column] -= epsilon
        moved_plus = propagate_proper_motion(
            *plus,
            source_epoch_jyear=source_epoch,
            target_epoch_jyear=target_epoch,
        )
        moved_minus = propagate_proper_motion(
            *minus,
            source_epoch_jyear=source_epoch,
            target_epoch_jyear=target_epoch,
        )
        east = (
            ((moved_plus[0] - moved_minus[0] + 180.0) % 360.0 - 180.0)
            * torch.cos(nominal[1] * np.pi / 180.0)
            * 3_600_000.0
            / (2.0 * epsilon)
        )
        north = (
            (moved_plus[1] - moved_minus[1])
            * 3_600_000.0
            / (2.0 * epsilon)
        )
        columns.append(torch.stack((east, north)))
    expected = torch.stack(columns, dim=-1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_proper_motion_jacobian_is_identity_at_same_epoch() -> None:
    propagated, jacobian = propagate_proper_motion_with_jacobian(
        torch.tensor([10.0, 359.999], dtype=torch.float64),
        torch.tensor([20.0, 89.9], dtype=torch.float64),
        torch.tensor([100.0, 5.0], dtype=torch.float64),
        torch.tensor([50.0, -2.0], dtype=torch.float64),
        source_epoch_jyear=torch.tensor([2016.0, 2020.0], dtype=torch.float64),
        target_epoch_jyear=torch.tensor([2016.0, 2020.0], dtype=torch.float64),
    )
    expected = torch.zeros((2, 2, 4), dtype=torch.float64)
    expected[:, 0, 0] = 1.0
    expected[:, 1, 1] = 1.0
    torch.testing.assert_close(jacobian, expected, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        torch.stack(propagated),
        torch.tensor([[10.0, 359.999], [20.0, 89.9]], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )


def test_proper_motion_zero_motion_jacobian_has_exact_time_baseline() -> None:
    _propagated, jacobian = propagate_proper_motion_with_jacobian(
        10.0,
        20.0,
        0.0,
        0.0,
        source_epoch_jyear=2000.0,
        target_epoch_jyear=2025.0,
    )
    expected = torch.tensor(
        [[1.0, 0.0, 25.0, 0.0], [0.0, 1.0, 0.0, 25.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(jacobian, expected, rtol=1e-9, atol=1e-9)


def test_proper_motion_jacobian_batches_like_scalar_calls() -> None:
    states = torch.tensor(
        [
            [10.0, 20.0, 100.0, 50.0],
            [359.9999, 89.9, 5.0, -2.0],
            [120.0, -89.9, -500.0, 1000.0],
        ],
        dtype=torch.float64,
    )
    source_epochs = torch.tensor([2000.0, 2016.0, 2020.0], dtype=torch.float64)
    target_epochs = torch.tensor([2025.0, 2040.0, 2000.0], dtype=torch.float64)
    _propagated, batched = propagate_proper_motion_with_jacobian(
        *states.unbind(dim=-1),
        source_epoch_jyear=source_epochs,
        target_epoch_jyear=target_epochs,
    )
    scalar = torch.stack(
        [
            propagate_proper_motion_with_jacobian(
                *state,
                source_epoch_jyear=source_epoch,
                target_epoch_jyear=target_epoch,
            )[1]
            for state, source_epoch, target_epoch in zip(
                states, source_epochs, target_epochs, strict=True
            )
        ]
    )

    assert batched.shape == (3, 2, 4)
    assert torch.isfinite(batched).all()
    torch.testing.assert_close(batched, scalar, rtol=2e-12, atol=2e-12)


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


def test_space_motion_jacobian_is_identity_at_same_epoch() -> None:
    propagated, jacobian = propagate_space_motion_with_jacobian(
        torch.tensor([10.0, 359.999], dtype=torch.float64),
        torch.tensor([20.0, 89.9], dtype=torch.float64),
        torch.tensor([100.0, 5.0], dtype=torch.float64),
        torch.tensor([50.0, -2.0], dtype=torch.float64),
        torch.tensor([100.0, 2.0], dtype=torch.float64),
        torch.tensor([20.0, 30.0], dtype=torch.float64),
        source_epoch_jyear=torch.tensor([2016.0, 2020.0], dtype=torch.float64),
        target_epoch_jyear=torch.tensor([2016.0, 2020.0], dtype=torch.float64),
    )

    expected = torch.eye(6, dtype=torch.float64).expand(2, -1, -1)
    torch.testing.assert_close(jacobian, expected, rtol=1e-9, atol=1e-9)
    expected_state = torch.tensor(
        [
            [10.0, 359.999],
            [20.0, 89.9],
            [100.0, 5.0],
            [50.0, -2.0],
            [100.0, 2.0],
            [20.0, 30.0],
        ],
        dtype=torch.float64,
    )
    for output, value in zip(propagated, expected_state, strict=True):
        torch.testing.assert_close(output, value, rtol=1e-10, atol=1e-10)


def test_space_motion_jacobian_matches_central_difference() -> None:
    state = torch.tensor(
        [359.9999, 20.0, 100.0, 50.0, 100.0, 20.0], dtype=torch.float64
    )
    source_epoch = 2016.0
    target_epoch = 2041.0
    _propagated, actual = propagate_space_motion_with_jacobian(
        *state,
        source_epoch_jyear=source_epoch,
        target_epoch_jyear=target_epoch,
    )
    nominal = propagate_space_motion(
        *state,
        source_epoch_jyear=source_epoch,
        target_epoch_jyear=target_epoch,
    )

    epsilons = torch.tensor([0.1, 0.1, 1e-3, 1e-3, 0.1, 1e-3], dtype=torch.float64)
    columns = []
    for column, epsilon in enumerate(epsilons):
        plus = state.clone()
        minus = state.clone()
        if column == 0:
            delta_deg = epsilon / (3_600_000.0 * torch.cos(state[1] * np.pi / 180.0))
            plus[0] += delta_deg
            minus[0] -= delta_deg
        elif column == 1:
            plus[1] += epsilon / 3_600_000.0
            minus[1] -= epsilon / 3_600_000.0
        else:
            plus[column] += epsilon
            minus[column] -= epsilon
        moved_plus = propagate_space_motion(
            *plus,
            source_epoch_jyear=source_epoch,
            target_epoch_jyear=target_epoch,
        )
        moved_minus = propagate_space_motion(
            *minus,
            source_epoch_jyear=source_epoch,
            target_epoch_jyear=target_epoch,
        )
        east = (
            ((moved_plus[0] - moved_minus[0] + 180.0) % 360.0 - 180.0)
            * torch.cos(nominal[1] * np.pi / 180.0)
            * 3_600_000.0
            / (2.0 * epsilon)
        )
        north = (
            (moved_plus[1] - moved_minus[1])
            * 3_600_000.0
            / (2.0 * epsilon)
        )
        columns.append(
            torch.stack(
                (
                    east,
                    north,
                    (moved_plus[2] - moved_minus[2]) / (2.0 * epsilon),
                    (moved_plus[3] - moved_minus[3]) / (2.0 * epsilon),
                    (moved_plus[4] - moved_minus[4]) / (2.0 * epsilon),
                    (moved_plus[5] - moved_minus[5]) / (2.0 * epsilon),
                )
            )
        )
    expected = torch.stack(columns, dim=-1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_space_motion_jacobian_batches_like_scalar_calls() -> None:
    states = torch.tensor(
        [
            [10.0, 20.0, 100.0, 50.0, 100.0, 20.0],
            [359.9999, 89.9, 5.0, -2.0, 2.0, 30.0],
            [120.0, -45.0, -500.0, 1000.0, 10.0, -50.0],
        ],
        dtype=torch.float64,
    )
    source_epochs = torch.tensor([2000.0, 2016.0, 2020.0], dtype=torch.float64)
    target_epochs = torch.tensor([2025.0, 2040.0, 2000.0], dtype=torch.float64)
    _propagated, batched = propagate_space_motion_with_jacobian(
        *states.unbind(dim=-1),
        source_epoch_jyear=source_epochs,
        target_epoch_jyear=target_epochs,
    )
    scalar = torch.stack(
        [
            propagate_space_motion_with_jacobian(
                *state,
                source_epoch_jyear=source_epoch,
                target_epoch_jyear=target_epoch,
            )[1]
            for state, source_epoch, target_epoch in zip(
                states, source_epochs, target_epochs, strict=True
            )
        ]
    )

    assert batched.shape == (3, 6, 6)
    assert torch.isfinite(batched).all()
    torch.testing.assert_close(batched, scalar, rtol=2e-12, atol=2e-12)


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


def test_ecliptic_of_date_roundtrip_and_erfa() -> None:
    ra = torch.tensor([0.0, 83.6331, 201.2983], dtype=torch.float64)
    dec = torch.tensor([0.0, 22.0145, -43.0191], dtype=torch.float64)
    tt_mjd = 58000.0
    lon, lat = icrs_to_ecliptic_of_date(ra, dec, tt_mjd)
    ra_back, dec_back = ecliptic_of_date_to_icrs(lon, lat, tt_mjd)
    np.testing.assert_allclose(_wrap_delta(ra_back.numpy(), ra.numpy()), 0.0, atol=1e-9)
    np.testing.assert_allclose(dec_back.numpy(), dec.numpy(), atol=1e-9)

    jd1, jd2 = 2400000.5, tt_mjd
    gamb, phib, psib, epsa = erfa.pfw06(jd1, jd2)
    rnpb = np.asarray(erfa.fw2m(gamb, phib, psib, epsa), dtype=np.float64)
    c = float(np.cos(epsa))
    s = float(np.sin(epsa))
    recl = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=np.float64)
    ra_r = np.deg2rad(ra.numpy())
    dec_r = np.deg2rad(dec.numpy())
    vec = np.stack(
        (
            np.cos(dec_r) * np.cos(ra_r),
            np.cos(dec_r) * np.sin(ra_r),
            np.sin(dec_r),
        ),
        axis=-1,
    )
    v_date = vec @ rnpb.T
    v_ecl = v_date @ recl.T
    lon_e = np.rad2deg(np.arctan2(v_ecl[:, 1], v_ecl[:, 0])) % 360.0
    lat_e = np.rad2deg(np.arcsin(np.clip(v_ecl[:, 2], -1.0, 1.0)))
    np.testing.assert_allclose(_wrap_delta(lon.numpy(), lon_e), 0.0, atol=1e-10)
    np.testing.assert_allclose(lat.numpy(), lat_e, atol=1e-10)


def test_galactocentric_roundtrip_and_astropy() -> None:
    ra = torch.tensor([266.4051, 83.6331], dtype=torch.float64)
    dec = torch.tensor([-28.936175, 22.0145], dtype=torch.float64)
    dist = torch.tensor([8.122, 0.14], dtype=torch.float64)
    xyz = icrs_to_galactocentric(ra, dec, dist)
    ra_b, dec_b, dist_b = galactocentric_to_icrs(xyz)
    np.testing.assert_allclose(_wrap_delta(ra_b.numpy(), ra.numpy()), 0.0, atol=1e-8)
    np.testing.assert_allclose(dec_b.numpy(), dec.numpy(), atol=1e-8)
    np.testing.assert_allclose(dist_b.numpy(), dist.numpy(), atol=1e-8)
    sky = SkyCoord(
        ra=ra.numpy() * u.deg,
        dec=dec.numpy() * u.deg,
        distance=dist.numpy() * u.kpc,
        frame=ICRS(),
    )
    galcen = sky.transform_to(
        Galactocentric(
            galcen_distance=8.122 * u.kpc,
            z_sun=20.8 * u.pc,
            roll=0 * u.deg,
        )
    )
    np.testing.assert_allclose(xyz[:, 0].numpy(), galcen.x.to_value(u.kpc), atol=1e-3)
    np.testing.assert_allclose(xyz[:, 1].numpy(), galcen.y.to_value(u.kpc), atol=1e-3)
    np.testing.assert_allclose(xyz[:, 2].numpy(), galcen.z.to_value(u.kpc), atol=1e-3)


def test_lsr_radial_velocity_matches_astropy_sign() -> None:
    ra = torch.tensor([180.0], dtype=torch.float64)
    dec = torch.tensor([0.0], dtype=torch.float64)
    rv = torch.tensor([10.0], dtype=torch.float64)
    lsr = heliocentric_rv_to_lsr(ra, dec, rv)
    sky = SkyCoord(
        ra=ra.numpy() * u.deg,
        dec=dec.numpy() * u.deg,
        distance=100.0 * u.pc,
        pm_ra_cosdec=0.0 * u.mas / u.yr,
        pm_dec=0.0 * u.mas / u.yr,
        radial_velocity=rv.numpy() * u.km / u.s,
        frame=ICRS(),
    )
    ast = sky.transform_to(LSR()).radial_velocity.to_value(u.km / u.s)
    np.testing.assert_allclose(lsr.numpy(), ast, atol=0.5)
