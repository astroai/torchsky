import erfa
import numpy as np
import pytest
import torch
from astropy.coordinates import (
    FK5,
    SkyCoord,
)
from astropy.time import Time
import astropy.units as u

from torchsky.wcs import (
    WCS,
    ecliptic_of_date_to_icrs,
    icrs_to_ecliptic_of_date,
    propagate_proper_motion,
    propagate_proper_motion_with_jacobian,
    propagate_space_motion,
    propagate_space_motion_with_jacobian,
)


def _wrap_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0






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
