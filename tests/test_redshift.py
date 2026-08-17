"""Named redshift dimming and velocity-kind converters."""

from __future__ import annotations

import pytest
import torch

from torchsky.redshift import (
    C_KM_S,
    apply_redshift,
    apply_redshift_f_lambda,
    convert_velocity_kind,
    dimming_f_lambda,
    dimming_tolman_sb,
    one_plus_z_from_beta,
    one_plus_z_from_velocity_kms,
    require_one_plus_z,
    shift_wavelength,
)


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
    torch.testing.assert_close(convert_velocity_kind(v, "km/s", "redshift"), z, atol=1e-6, rtol=1e-5)
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
