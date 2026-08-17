"""Flat wCDM distance checks."""

from __future__ import annotations

import pytest
import torch

from torchsky.cosmology import (
    PLANCK18_H0,
    SPEED_OF_LIGHT_KM_S,
    angular_diameter_distance,
    comoving_distance,
    distance_modulus,
    e_of_z,
    luminosity_distance,
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


def test_astropy_flat_lambda_cdm_parity() -> None:
    astropy = pytest.importorskip("astropy.cosmology")
    z = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    ref = astropy.FlatLambdaCDM(H0=PLANCK18_H0, Om0=0.3111)
    got = luminosity_distance(z, H0=PLANCK18_H0, om0=0.3111, w0=-1.0, n_steps=4096)
    want = torch.tensor(ref.luminosity_distance(z.numpy()).value, dtype=torch.float64)
    torch.testing.assert_close(got, want, rtol=2e-3, atol=0.5)
