"""Analytic checks for CCM89 / O'Donnell extinction."""

from __future__ import annotations

import pytest
import torch

from torchsky.extinction import (
    a_lambda,
    apply_extinction,
    ccm89,
    ccm89_ab,
    odonnell94,
    transmission,
)


def test_ir_power_law_piece() -> None:
    wave = torch.tensor([20000.0])  # x = 0.5 inverse microns
    x = 10000.0 / 20000.0
    expected_a = 0.574 * x**1.61
    expected_b = -0.527 * x**1.61
    r_v = 3.1
    got = ccm89(wave, torch.tensor(1.0), r_v)
    assert got.item() == pytest.approx(expected_a + expected_b / r_v, rel=1e-5)


def test_optical_ccm89_differs_from_odonnell() -> None:
    wave = torch.tensor([4000.0, 5500.0, 7000.0])
    a_v = torch.tensor(1.0)
    r_v = torch.tensor(3.1)
    assert not torch.allclose(ccm89(wave, a_v, r_v), odonnell94(wave, a_v, r_v))


def test_zero_ebv_is_identity() -> None:
    wave = torch.tensor([3500.0, 5500.0, 8000.0])
    flux = torch.ones_like(wave)
    out = apply_extinction(flux, wave, torch.tensor(0.0))
    torch.testing.assert_close(out, flux)
    torch.testing.assert_close(transmission(wave, 0.0), torch.ones_like(wave))


def test_angstrom_and_nm_agree() -> None:
    wave_a = torch.tensor([5510.0])
    wave_nm = torch.tensor([551.0])
    mag_a = a_lambda(wave_a, ebv=1.0, r_v=3.1)
    mag_nm = a_lambda(wave_nm, ebv=1.0, r_v=3.1, wavelength_unit="nm")
    torch.testing.assert_close(mag_a, mag_nm, atol=1e-6, rtol=0)
    assert abs(float(mag_a) - 3.1) < 0.2


def test_requires_exactly_one_scale() -> None:
    wave = torch.tensor([5500.0])
    with pytest.raises(ValueError, match="exactly one"):
        a_lambda(wave)
    with pytest.raises(ValueError, match="exactly one"):
        a_lambda(wave, ebv=0.1, a_v=0.3)


def test_far_uv_uses_ccm89_cubic_not_mid_uv_edge() -> None:
    # x=9 is 1111 Å, inside the far-UV cubic; x=8 is the join with mid-UV.
    wave_fuv = torch.tensor([1.0e4 / 9.0])
    wave_join = torch.tensor([1.0e4 / 8.0])
    y = 1.0
    a_fuv = -1.073 - 0.628 * y + 0.137 * y**2 - 0.070 * y**3
    b_fuv = 13.670 + 4.257 * y - 0.420 * y**2 + 0.374 * y**3
    r_v = 3.1
    got = ccm89(wave_fuv, torch.tensor(1.0), r_v)
    assert got.item() == pytest.approx(a_fuv + b_fuv / r_v, rel=1e-5)
    join = ccm89(wave_join, torch.tensor(1.0), r_v)
    assert join.item() > 0
    # Frozen at x=10: 800 Å and 500 Å must agree.
    torch.testing.assert_close(
        ccm89(torch.tensor([800.0]), torch.tensor(1.0), r_v),
        ccm89(torch.tensor([500.0]), torch.tensor(1.0), r_v),
    )


def test_batched_ebv_broadcasts_over_wavelength() -> None:
    wave = torch.linspace(4000.0, 8000.0, 5)
    ebv = torch.tensor([0.0, 0.2, 1.0])
    mag = a_lambda(wave, ebv=ebv)
    assert mag.shape == (3, 5)
    torch.testing.assert_close(mag[0], torch.zeros(5), atol=1e-7, rtol=0)
    assert torch.all(mag[2] > mag[1])


def test_transmission_never_amplifies() -> None:
    wave = torch.linspace(2000.0, 20000.0, 32)
    t = transmission(wave, ebv=1.0)
    assert torch.all(t > 0)
    assert torch.all(t <= 1.0 + 1e-6)


def test_apply_extinction_accepts_uspm_wavelength_name() -> None:
    flux = torch.ones(3)
    wave = torch.tensor([4000.0, 5500.0, 8000.0])
    out = apply_extinction(flux, ext_params=torch.tensor(0.0), lambda_obs_angstrom=wave)
    torch.testing.assert_close(out, flux)


def test_inplace_wavelength_edit_is_not_stale() -> None:
    wave = torch.linspace(4000.0, 8000.0, 8)
    a_v = torch.tensor(1.0)
    first = ccm89(wave, a_v).clone()
    wave.mul_(2.0)
    second = ccm89(wave, a_v)
    assert not torch.allclose(first, second)


def test_optical_y2_coefficient_is_ccm89_published() -> None:
    # At y=1 (x=2.82 inverse microns) the y^2 term is the published 0.50447.
    x = torch.tensor([2.82])
    a, _b = ccm89_ab(x)
    y = float(x - 1.82)
    expected_a = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    assert a.item() == pytest.approx(expected_a, rel=1e-6)


def test_dust_extinction_v_band_point() -> None:
    pytest.importorskip("dust_extinction")
    pytest.importorskip("astropy.units")
    from astropy import units as u
    from dust_extinction.parameter_averages import CCM89

    wave_aa = 5470.0
    got = float(a_lambda(torch.tensor([wave_aa]), a_v=1.0, r_v=3.1, law="ccm89"))
    ref = float(CCM89(Rv=3.1)(wave_aa * u.Angstrom))
    assert got == pytest.approx(ref, rel=2e-2)
