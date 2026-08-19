"""Named redshift dimming and velocity-kind converters."""

from __future__ import annotations

import pytest
import torch

from torchsky.redshift import (
    C_KM_S,
    apply_redshift,
    apply_redshift_f_lambda,
    compose_rapidities,
    convert_velocity_kind,
    dimming_f_lambda,
    dimming_tolman_sb,
    los_barycentric_beta,
    one_plus_z_from_beta,
    one_plus_z_from_velocity_kms,
    one_plus_z_from_zeta,
    require_one_plus_z,
    shift_ln_wavelength,
    shift_wavelength,
    zeta_from_beta,
    zeta_from_redshift,
    zeta_from_velocity_kms,
)
from torchsky.wcs.iau2006 import C_M_S, earth_barycentric_velocity_icrs
from torchsky.wcs.spectral import air_wavelength_to_vacuum


def test_zero_velocity_is_identity() -> None:
    flux = torch.ones(4)
    wave = torch.linspace(5000.0, 5003.0, 4)
    out_flux, out_wave = apply_redshift(flux, wave, torch.tensor(0.0), is_velocity=True)
    torch.testing.assert_close(out_flux, flux)
    torch.testing.assert_close(out_wave, wave)


def test_z_v_beta_round_trip() -> None:
    z = torch.tensor([0.0, 0.1, 1.0])
    v = convert_velocity_kind(z, "redshift", "km/s")
    beta = convert_velocity_kind(z, "redshift", "beta")
    torch.testing.assert_close(
        convert_velocity_kind(v, "km/s", "redshift"), z, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        convert_velocity_kind(beta, "beta", "redshift"), z, atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(beta * C_KM_S, v, atol=1e-4, rtol=1e-5)


def test_f_lambda_times_lambda_invariant() -> None:
    flux = torch.tensor([2.0, 3.0])
    wave = torch.tensor([1000.0, 2000.0])
    z = 1.0
    shifted, lam = apply_redshift_f_lambda(flux, wave, 1.0 + z)
    torch.testing.assert_close(shifted * lam, flux * wave)


def test_tolman_and_f_lambda_disagree_at_z_one() -> None:
    factor = torch.tensor(2.0)
    f_lam = dimming_f_lambda(factor)
    tolman = dimming_tolman_sb(factor)
    assert f_lam.item() == pytest.approx(0.5)
    assert tolman.item() == pytest.approx(1.0 / 16.0)
    assert f_lam.item() != pytest.approx(tolman.item())


def test_doppler_factor_matches_beta_formula() -> None:
    v = torch.tensor(30_000.0)
    beta = v / C_KM_S
    got = one_plus_z_from_velocity_kms(v)
    expected = torch.sqrt((1.0 + beta) / (1.0 - beta))
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(got, one_plus_z_from_beta(beta))


def test_classical_z_is_beta() -> None:
    z = torch.tensor([0.0, 0.01, 0.1])
    torch.testing.assert_close(
        convert_velocity_kind(z, "redshift", "beta", relativistic=False), z
    )
    torch.testing.assert_close(
        convert_velocity_kind(z, "beta", "redshift", relativistic=False), z
    )


def test_shift_wavelength_and_require() -> None:
    torch.testing.assert_close(
        shift_wavelength(torch.tensor([100.0]), 2.0), torch.tensor([200.0])
    )
    with pytest.raises(ValueError, match="positive"):
        require_one_plus_z(0.0)


def test_z_zero_rapidity_is_identity() -> None:
    z = torch.zeros(3)
    torch.testing.assert_close(zeta_from_redshift(z), torch.zeros(3))
    ln_lambda = torch.log(torch.tensor([5000.0, 6000.0]))
    torch.testing.assert_close(shift_ln_wavelength(ln_lambda, 0.0), ln_lambda)
    torch.testing.assert_close(
        one_plus_z_from_zeta(zeta_from_redshift(0.0)), torch.tensor(1.0)
    )


def test_rapidity_is_additive_versus_multiplicative_one_plus_z() -> None:
    z1 = torch.tensor([0.1, 0.3])
    z2 = torch.tensor([0.2, -0.05])
    composed = compose_rapidities(zeta_from_redshift(z1), zeta_from_redshift(z2))
    torch.testing.assert_close(one_plus_z_from_zeta(composed), (1.0 + z1) * (1.0 + z2))
    torch.testing.assert_close(
        zeta_from_beta(torch.tensor(0.01)),
        zeta_from_velocity_kms(0.01 * C_KM_S),
        atol=1e-6,
        rtol=1e-6,
    )


def test_los_barycentric_beta_round_trip_on_known_vector() -> None:
    tt_mjd = torch.tensor(58000.0)
    unit = torch.tensor([0.0, 0.0, 1.0])
    velocity = earth_barycentric_velocity_icrs(tt_mjd)
    expected = -(velocity * unit).sum() / C_M_S
    got = los_barycentric_beta(unit, tt_mjd)
    torch.testing.assert_close(got, expected)
    # Receding (positive β) when Earth velocity is anti-parallel to the source.
    approaching = torch.nn.functional.normalize(velocity, dim=-1)
    receding = -approaching
    assert float(los_barycentric_beta(receding, tt_mjd)) > 0
    assert float(los_barycentric_beta(approaching, tt_mjd)) < 0


def test_air_vac_then_rapidity_does_not_mix_frames() -> None:
    # Air→vac is a wavelength-dependent index. Convert the published
    # (observed) grid first; do not apply n(λ) after a rest-frame shift.
    air_obs = torch.tensor([1500.0, 3500.0])
    one_plus_z = 1.5
    vac_obs = air_wavelength_to_vacuum(air_obs, unit="Angstrom")
    rest_after_vac = vac_obs / one_plus_z
    rest_before_vac = air_wavelength_to_vacuum(air_obs / one_plus_z, unit="Angstrom")
    assert not torch.allclose(rest_after_vac, rest_before_vac, rtol=1e-6, atol=0)
    zeta = zeta_from_redshift(one_plus_z - 1.0)
    shifted = torch.exp(shift_ln_wavelength(torch.log(vac_obs), -zeta))
    torch.testing.assert_close(shifted, rest_after_vac)
