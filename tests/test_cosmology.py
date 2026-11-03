"""Flat wCDM distance checks."""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.cosmology import (
    FLRW,
    PLANCK18_H0,
    SPEED_OF_LIGHT_KM_S,
    alcock_paczynski_scaling,
    angular_diameter_distance,
    comoving_distance,
    comoving_position,
    comoving_separation,
    differential_comoving_volume,
    distance_modulus,
    e_of_z,
    hubble_parameter,
    luminosity_distance,
    transverse_comoving_distance,
)


def test_low_z_luminosity_distance_is_cz_over_h0() -> None:
    z = torch.tensor(1.0e-4)
    d_l = luminosity_distance(z, H0=70.0, om0=0.3, w0=-1.0)
    expected = SPEED_OF_LIGHT_KM_S * float(z) / 70.0
    assert d_l.item() == pytest.approx(expected, rel=2e-3)


def test_etherington_distance_duality() -> None:
    z = torch.tensor([0.2, 1.0, 2.5])
    d_l = luminosity_distance(z)
    d_a = angular_diameter_distance(z)
    torch.testing.assert_close(d_l, (1.0 + z) ** 2 * d_a, rtol=1e-5, atol=1e-4)


def test_w0_minus_one_matches_lcdm_e() -> None:
    z = torch.linspace(0.0, 3.0, 8)
    torch.testing.assert_close(
        e_of_z(z, om0=0.3, w0=-1.0),
        torch.sqrt(0.3 * (1.0 + z) ** 3 + 0.7),
    )


def test_cpl_expansion_matches_astropy() -> None:
    astropy = pytest.importorskip("astropy.cosmology")
    z = torch.linspace(0.0, 3.0, 9, dtype=torch.float64)
    om0, ok0, w0, wa = 0.29, 0.03, -0.85, 0.2
    ref = astropy.w0waCDM(
        H0=70.0,
        Om0=om0,
        Ode0=1.0 - om0 - ok0,
        w0=w0,
        wa=wa,
    )
    got = e_of_z(z, om0=om0, ok0=ok0, w0=w0, wa=wa)
    want = torch.tensor(ref.efunc(z.numpy()), dtype=torch.float64)
    torch.testing.assert_close(got, want, rtol=5e-8, atol=5e-8)


def test_angular_diameter_zero_at_zero_redshift() -> None:
    z0 = torch.tensor(0.0)
    assert angular_diameter_distance(z0).item() == pytest.approx(0.0, abs=1e-12)
    assert luminosity_distance(z0).item() == pytest.approx(0.0, abs=1e-12)
    z_neg = torch.tensor([-0.01, 0.0, 0.2])
    d_a = angular_diameter_distance(z_neg)
    assert torch.isfinite(d_a).all()
    assert d_a[0].item() == pytest.approx(0.0, abs=1e-12)
    assert d_a[1].item() == pytest.approx(0.0, abs=1e-12)
    assert d_a[2].item() > 0.0
    mu = distance_modulus(torch.tensor(0.0))
    assert torch.isfinite(mu)
    d_c = comoving_distance(torch.tensor(0.0))
    assert d_c.item() == pytest.approx(0.0, abs=1e-12)
    integer_input = comoving_distance(torch.tensor(1))
    assert integer_input.is_floating_point()
    assert integer_input.item() > 0.0


def test_astropy_flat_lambda_cdm_parity() -> None:
    astropy = pytest.importorskip("astropy.cosmology")
    z = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    ref = astropy.FlatLambdaCDM(H0=PLANCK18_H0, Om0=0.3111)
    got = luminosity_distance(z, H0=PLANCK18_H0, om0=0.3111, w0=-1.0)
    want = torch.tensor(ref.luminosity_distance(z.numpy()).value, dtype=torch.float64)
    torch.testing.assert_close(got, want, rtol=2e-3, atol=0.5)


def test_differential_comoving_volume_matches_astropy_and_has_gradients() -> None:
    astropy = pytest.importorskip("astropy.cosmology")
    z = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    om0 = torch.tensor(0.3111, dtype=torch.float64, requires_grad=True)
    cosmology = FLRW(H0=PLANCK18_H0, Om0=om0)
    got = differential_comoving_volume(z, cosmology)
    reference = astropy.FlatLambdaCDM(H0=PLANCK18_H0, Om0=0.3111)
    want = torch.tensor(
        reference.differential_comoving_volume(z.numpy()).value,
        dtype=torch.float64,
    )
    torch.testing.assert_close(got, want, rtol=2e-3, atol=1.0)
    got.sum().backward()
    assert om0.grad is not None and torch.isfinite(om0.grad)


@pytest.mark.parametrize("ok0", [-0.08, 0.08])
def test_curved_transverse_distance_matches_astropy(ok0: float) -> None:
    astropy = pytest.importorskip("astropy.cosmology")
    z = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    om0 = 0.3
    ref = astropy.LambdaCDM(H0=70.0, Om0=om0, Ode0=1.0 - om0 - ok0)
    got = transverse_comoving_distance(z, H0=70.0, om0=om0, ok0=ok0, w0=-1.0)
    want = torch.tensor(
        ref.comoving_transverse_distance(z.numpy()).value, dtype=torch.float64
    )
    torch.testing.assert_close(got, want, rtol=4e-4, atol=0.25)


def test_flat_position_and_separation_are_rotation_invariant() -> None:
    dtype = torch.float64
    directions_a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype)
    directions_b = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    z_a = torch.tensor([0.4, 0.8], dtype=dtype)
    z_b = torch.tensor([0.7, 1.1], dtype=dtype)
    cosmology = FLRW(H0=70.0, Om0=0.3)

    positions = comoving_position(directions_a, z_a, cosmology)
    torch.testing.assert_close(
        torch.linalg.vector_norm(positions, dim=-1),
        cosmology.comoving_distance(z_a),
    )
    separation = comoving_separation(directions_a, z_a, directions_b, z_b, cosmology)

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype
    )
    rotated_a = directions_a @ rotation.T
    rotated_b = directions_b @ rotation.T
    rotated_separation = comoving_separation(rotated_a, z_a, rotated_b, z_b, cosmology)
    torch.testing.assert_close(separation, rotated_separation)


@pytest.mark.parametrize("ok0", [-0.08, 0.0, 0.08])
def test_curved_radial_separation_is_distance_difference(ok0: float) -> None:
    dtype = torch.float64
    direction = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    z_a = torch.tensor(0.4, dtype=dtype)
    z_b = torch.tensor(1.2, dtype=dtype)
    cosmology = FLRW(H0=70.0, Om0=0.3, Ok0=ok0)
    expected = torch.abs(
        cosmology.comoving_distance(z_b) - cosmology.comoving_distance(z_a)
    )
    got = comoving_separation(direction, z_a, direction, z_b, cosmology)
    torch.testing.assert_close(got, expected, rtol=2e-6, atol=2e-3)


@pytest.mark.parametrize("ok0", [-0.08, 0.08])
def test_curved_non_collinear_separation_matches_scalar_reference(ok0: float) -> None:
    dtype = torch.float64
    angle = 0.63
    direction_a = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    direction_b = torch.tensor([math.cos(angle), math.sin(angle), 0.0], dtype=dtype)
    z_a = torch.tensor(0.35, dtype=dtype)
    z_b = torch.tensor(1.1, dtype=dtype)
    cosmology = FLRW(H0=70.0, Om0=0.3, Ok0=ok0)

    got = comoving_separation(direction_a, z_a, direction_b, z_b, cosmology)
    d_a = cosmology.comoving_distance(z_a).item()
    d_b = cosmology.comoving_distance(z_b).item()
    radius = (SPEED_OF_LIGHT_KM_S / 70.0) / math.sqrt(abs(ok0))
    radial_a, radial_b = d_a / radius, d_b / radius
    half_chord_squared = (1.0 - math.cos(angle)) / 2.0
    if ok0 > 0.0:
        half_squared = math.sinh(0.5 * (radial_a - radial_b)) ** 2 + (
            math.sinh(radial_a) * math.sinh(radial_b) * half_chord_squared
        )
        expected = 2.0 * radius * math.asinh(math.sqrt(half_squared))
    else:
        half_squared = math.sin(0.5 * (radial_a - radial_b)) ** 2 + (
            math.sin(radial_a) * math.sin(radial_b) * half_chord_squared
        )
        expected = 2.0 * radius * math.asin(math.sqrt(half_squared))
    assert got.item() == pytest.approx(expected, rel=5e-11, abs=1e-7)


@pytest.mark.parametrize("ok0", [-1e-3, -1e-4, -1e-5, 1e-5, 1e-4, 1e-3])
def test_curved_arcminute_pair_is_stable_in_float32(ok0: float) -> None:
    angle = math.radians(1.0 / 60.0)
    direction32_a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    direction32_b = torch.tensor(
        [math.cos(angle), math.sin(angle), 0.0], dtype=torch.float32
    )
    direction64_a = direction32_a.to(torch.float64)
    direction64_b = torch.tensor(
        [math.cos(angle), math.sin(angle), 0.0], dtype=torch.float64
    )
    z32 = torch.tensor(0.8, dtype=torch.float32)
    z64 = z32.to(torch.float64)

    got = comoving_separation(
        direction32_a,
        z32,
        direction32_b,
        z32,
        FLRW(H0=70.0, Om0=0.3, Ok0=ok0),
    )
    reference = comoving_separation(
        direction64_a,
        z64,
        direction64_b,
        z64,
        FLRW(H0=70.0, Om0=0.3, Ok0=ok0),
    )
    assert got.item() > 0.0
    torch.testing.assert_close(got.to(torch.float64), reference, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_curvature_series_transition_is_continuous(sign: float) -> None:
    dtype = torch.float64
    direction_a = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    direction_b = torch.tensor([0.8, 0.6, 0.0], dtype=dtype)
    z_a = torch.tensor(0.4, dtype=dtype)
    z_b = torch.tensor(1.2, dtype=dtype)
    flat = FLRW(H0=70.0, Om0=0.3)
    d_h = SPEED_OF_LIGHT_KM_S / 70.0
    radial_scale = max(
        (flat.comoving_distance(z_a).item() / d_h) ** 2,
        (flat.comoving_distance(z_b).item() / d_h) ** 2,
    )
    transition = math.sqrt(torch.finfo(dtype).eps) / radial_scale

    values = []
    for factor in (0.9, 1.1):
        values.append(
            comoving_separation(
                direction_a,
                z_a,
                direction_b,
                z_b,
                FLRW(H0=70.0, Om0=0.3, Ok0=sign * factor * transition),
            )
        )
    torch.testing.assert_close(values[0], values[1], rtol=5e-8, atol=1e-5)


@pytest.mark.parametrize("ok0_value", [-1e-3, 1e-3])
def test_curved_pair_matrix_has_finite_diagonal_gradients(ok0_value: float) -> None:
    dtype = torch.float64
    directions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]], dtype=dtype
    )
    redshifts = torch.tensor([0.3, 0.7, 1.2], dtype=dtype, requires_grad=True)
    h0 = torch.tensor(70.0, dtype=dtype, requires_grad=True)
    ok0 = torch.tensor(ok0_value, dtype=dtype, requires_grad=True)
    separations = comoving_separation(
        directions[:, None, :],
        redshifts[:, None],
        directions[None, :, :],
        redshifts[None, :],
        FLRW(H0=h0, Om0=0.3, Ok0=ok0),
    )

    torch.testing.assert_close(
        separations.diagonal(), torch.zeros(3, dtype=dtype), rtol=0.0, atol=0.0
    )
    separations.sum().backward()
    for parameter in (redshifts, h0, ok0):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_ap_identity_and_hubble_definition() -> None:
    z = torch.tensor([0.2, 0.7, 1.4], dtype=torch.float64)
    cosmology = FLRW(H0=70.0, Om0=0.3, Ok0=0.01, w0=-0.9, wa=0.1)
    scaling = alcock_paczynski_scaling(z, cosmology, cosmology)
    torch.testing.assert_close(scaling.alpha_parallel, torch.ones_like(z))
    torch.testing.assert_close(scaling.alpha_perpendicular, torch.ones_like(z))
    torch.testing.assert_close(
        hubble_parameter(
            z,
            H0=cosmology.H0,
            om0=cosmology.Om0,
            ok0=cosmology.Ok0,
            w0=cosmology.w0,
            wa=cosmology.wa,
        ),
        float(cosmology.H0) * cosmology.e(z),
    )


def test_ap_nonidentity_scalings_and_absolute_observable() -> None:
    z = torch.tensor([0.3, 0.9, 1.5], dtype=torch.float64)
    cosmology = FLRW(H0=72.0, Om0=0.28, Ok0=0.02, w0=-0.9, wa=0.1)
    reference = FLRW(H0=67.66, Om0=0.3111)
    scaling = alcock_paczynski_scaling(z, cosmology, reference)
    d_m = cosmology.transverse_comoving_distance(z)
    d_m_reference = reference.transverse_comoving_distance(z)
    h = cosmology.hubble_parameter(z)
    h_reference = reference.hubble_parameter(z)

    torch.testing.assert_close(scaling.alpha_parallel, h_reference / h)
    torch.testing.assert_close(scaling.alpha_perpendicular, d_m / d_m_reference)
    torch.testing.assert_close(scaling.f_ap, d_m * h / SPEED_OF_LIGHT_KM_S)
    assert not torch.allclose(scaling.alpha_parallel, torch.ones_like(z))
    assert not torch.allclose(scaling.alpha_perpendicular, torch.ones_like(z))


def test_flrw_geometry_parameter_gradients_are_finite() -> None:
    dtype = torch.float64
    parameters = [
        torch.tensor(70.0, dtype=dtype, requires_grad=True),
        torch.tensor(0.3, dtype=dtype, requires_grad=True),
        torch.tensor(0.0, dtype=dtype, requires_grad=True),
        torch.tensor(-0.9, dtype=dtype, requires_grad=True),
        torch.tensor(0.1, dtype=dtype, requires_grad=True),
    ]
    cosmology = FLRW(*parameters)
    reference = FLRW(H0=67.66, Om0=0.3111)
    z = torch.tensor([0.3, 0.9, 1.5], dtype=dtype)
    unit_a = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    unit_b = torch.tensor([0.8, 0.6, 0.0], dtype=dtype)

    scaling = alcock_paczynski_scaling(z, cosmology, reference)
    loss = (
        cosmology.transverse_comoving_distance(z).sum()
        + comoving_separation(unit_a, z[0], unit_b, z[-1], cosmology)
        + scaling.alpha_parallel.sum()
        + scaling.alpha_perpendicular.sum()
    )
    loss.backward()
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad)


def test_flat_curvature_gradients_match_two_sided_limits() -> None:
    dtype = torch.float64
    epsilon = 1.0e-4
    z = torch.tensor(1.1, dtype=dtype)
    ok0 = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    distance = FLRW(H0=70.0, Om0=0.3, Ok0=ok0).transverse_comoving_distance(z)
    (distance_gradient,) = torch.autograd.grad(distance, ok0)
    distance_plus = FLRW(H0=70.0, Om0=0.3, Ok0=epsilon).transverse_comoving_distance(z)
    distance_minus = FLRW(H0=70.0, Om0=0.3, Ok0=-epsilon).transverse_comoving_distance(
        z
    )
    distance_finite_difference = (distance_plus - distance_minus) / (2.0 * epsilon)
    torch.testing.assert_close(
        distance_gradient, distance_finite_difference, rtol=3e-3, atol=0.2
    )

    unit_a = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    unit_b = torch.tensor([0.8, 0.6, 0.0], dtype=dtype)
    z_a = torch.tensor(0.4, dtype=dtype)
    z_b = torch.tensor(1.2, dtype=dtype)
    ok0 = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    separation = comoving_separation(
        unit_a, z_a, unit_b, z_b, FLRW(H0=70.0, Om0=0.3, Ok0=ok0)
    )
    (separation_gradient,) = torch.autograd.grad(separation, ok0)
    separation_plus = comoving_separation(
        unit_a,
        z_a,
        unit_b,
        z_b,
        FLRW(H0=70.0, Om0=0.3, Ok0=epsilon),
    )
    separation_minus = comoving_separation(
        unit_a,
        z_a,
        unit_b,
        z_b,
        FLRW(H0=70.0, Om0=0.3, Ok0=-epsilon),
    )
    separation_finite_difference = (separation_plus - separation_minus) / (
        2.0 * epsilon
    )
    torch.testing.assert_close(
        separation_gradient, separation_finite_difference, rtol=5e-3, atol=0.5
    )


def test_comoving_position_rejects_curvature_and_non_unit_directions() -> None:
    with pytest.raises(ValueError, match="Ok0 = 0"):
        comoving_position(
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor(0.5),
            FLRW(Ok0=0.01),
        )
    with pytest.raises(ValueError, match="unit vectors"):
        comoving_position(torch.tensor([2.0, 0.0, 0.0]), torch.tensor(0.5))


def test_python_float_parameters_keep_float64_precision() -> None:
    astropy = pytest.importorskip("astropy.cosmology")
    z = torch.tensor([0.1, 0.5, 1.0, 3.0], dtype=torch.float64)
    h0, om0 = 67.66, 0.3111
    ref = astropy.FlatLambdaCDM(H0=h0, Om0=om0, Tcmb0=0).comoving_distance(z.numpy())
    got = comoving_distance(z, H0=h0, om0=om0)
    assert got.dtype == torch.float64
    torch.testing.assert_close(got, torch.as_tensor(ref.value), rtol=1e-9, atol=0.0)


_TIME_CASES = [
    {"H0": 67.66, "om0": 0.3111},
    {"H0": 70.0, "om0": 0.3, "ok0": 0.08},
    {"H0": 70.0, "om0": 0.3, "ok0": -0.08},
    {"H0": 70.0, "om0": 0.29, "ok0": 0.03, "w0": -0.85, "wa": 0.2},
    {"H0": 70.0, "om0": 0.3, "w0": -0.9, "wa": -0.3},
]


def _astropy_reference(astropy, case):
    ode0 = 1.0 - case["om0"] - case.get("ok0", 0.0)
    # Radiation off (Tcmb0=0): torchsky.cosmology has no radiation or
    # massive-neutrino terms.
    return astropy.w0waCDM(
        H0=case["H0"],
        Om0=case["om0"],
        Ode0=ode0,
        w0=case.get("w0", -1.0),
        wa=case.get("wa", 0.0),
        Tcmb0=0.0,
    )


@pytest.mark.parametrize("case", _TIME_CASES)
def test_lookback_time_and_age_match_astropy(case) -> None:
    astropy = pytest.importorskip("astropy.cosmology")
    from torchsky.cosmology import age, lookback_time

    z = torch.tensor([0.0, 0.01, 0.5, 1.0, 3.0, 10.0, 100.0], dtype=torch.float64)
    ref = _astropy_reference(astropy, case)
    torch.testing.assert_close(
        lookback_time(z, **case),
        torch.tensor(ref.lookback_time(z.numpy()).value, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    # astropy integrates the age with quad at its default 1.5e-8 tolerance.
    torch.testing.assert_close(
        age(z, **case),
        torch.tensor(ref.age(z.numpy()).value, dtype=torch.float64),
        rtol=1e-7,
        atol=0.0,
    )


def test_age_matches_closed_forms() -> None:
    from torchsky.cosmology import HUBBLE_TIME_GYR, age, lookback_time

    z = torch.tensor([0.0, 1.0, 1000.0], dtype=torch.float64)
    # Einstein-de Sitter: t(z) = (2/3) t_H (1+z)^(-3/2).
    torch.testing.assert_close(
        age(z, H0=70.0, om0=1.0),
        2.0 / 3.0 * HUBBLE_TIME_GYR / 70.0 * (1.0 + z) ** -1.5,
        rtol=1e-14,
        atol=0.0,
    )
    # Flat LCDM: t0 = 2 t_H / (3 sqrt(OL)) asinh(sqrt(OL/Om)).
    om = 0.3111
    exact = (
        HUBBLE_TIME_GYR
        / PLANCK18_H0
        * 2.0
        / (3.0 * math.sqrt(1.0 - om))
        * math.asinh(math.sqrt((1.0 - om) / om))
    )
    assert float(age(torch.tensor(0.0, dtype=torch.float64))) == pytest.approx(
        exact, rel=1e-14
    )
    # lookback(z) + age(z) = age(0) for any model.
    z = torch.tensor([0.3, 2.0, 7.0], dtype=torch.float64)
    total = lookback_time(z, om0=0.3, ok0=0.05, w0=-0.9, wa=0.1) + age(
        z, om0=0.3, ok0=0.05, w0=-0.9, wa=0.1
    )
    torch.testing.assert_close(
        total,
        age(torch.zeros(3, dtype=torch.float64), om0=0.3, ok0=0.05, w0=-0.9, wa=0.1),
        rtol=1e-13,
        atol=0.0,
    )


def test_lookback_time_and_age_edges() -> None:
    from torchsky.cosmology import age, lookback_time

    assert float(lookback_time(torch.tensor(-0.01))) == 0.0
    assert lookback_time(torch.tensor([])).numel() == 0
    with pytest.raises(ValueError, match="Om0 > 0"):
        age(torch.tensor(0.0), om0=0.0, ok0=1.0)
    with pytest.raises(ValueError, match="H0"):
        lookback_time(torch.tensor(1.0), H0=-1.0)
    flrw = FLRW(H0=70.0, Om0=0.3)
    z = torch.tensor([0.5, 2.0], dtype=torch.float64)
    torch.testing.assert_close(flrw.age(z), age(z, H0=70.0, om0=0.3))
    torch.testing.assert_close(
        flrw.lookback_time(z), lookback_time(z, H0=70.0, om0=0.3)
    )


def test_lookback_time_and_age_gradients() -> None:
    from torchsky.cosmology import age, lookback_time

    z = torch.tensor([0.5, 2.0], dtype=torch.float64, requires_grad=True)
    params = (
        torch.tensor(68.0, dtype=torch.float64, requires_grad=True),
        torch.tensor(0.31, dtype=torch.float64, requires_grad=True),
        torch.tensor(0.02, dtype=torch.float64, requires_grad=True),
        torch.tensor(-0.95, dtype=torch.float64, requires_grad=True),
        torch.tensor(0.1, dtype=torch.float64, requires_grad=True),
    )
    for function in (lookback_time, age):
        assert torch.autograd.gradcheck(
            lambda zz, h0, om, ok, w0, wa, f=function: f(
                zz, H0=h0, om0=om, ok0=ok, w0=w0, wa=wa
            ),
            (z, *params),
        )
    # dt_L/dz = t_H / ((1+z) E(z)) analytically.
    zz = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(lookback_time(zz, H0=70.0, om0=0.3).sum(), zz)
    from torchsky.cosmology import HUBBLE_TIME_GYR

    expected = HUBBLE_TIME_GYR / 70.0 / (1.7 * e_of_z(zz.detach(), om0=0.3))
    torch.testing.assert_close(grad, expected, rtol=1e-12, atol=0.0)


def test_lookback_time_and_age_on_device(compute_device) -> None:
    from torchsky.cosmology import age, lookback_time

    dtype = torch.float32 if compute_device.type == "mps" else torch.float64
    z_cpu = torch.tensor([0.1, 1.0, 5.0], dtype=dtype)
    z = z_cpu.to(compute_device)
    for function in (lookback_time, age):
        out = function(z, H0=70.0, om0=0.3)
        assert out.device.type == compute_device.type and out.dtype == dtype
        torch.testing.assert_close(
            out.cpu(), function(z_cpu, H0=70.0, om0=0.3), rtol=1e-5, atol=1e-5
        )
