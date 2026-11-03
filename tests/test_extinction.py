"""Analytic checks for CCM89 / O'Donnell extinction."""

from typing import Any, cast

import pytest
import torch

from torchsky.extinction import (
    a_lambda,
    apply_extinction,
    calzetti,
    ccm89,
    ccm89_ab,
    fitzpatrick99,
    k_lambda,
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


def test_apply_extinction_accepts_generic_wavelength_argument() -> None:
    flux = torch.ones(3)
    wave = torch.tensor([4000.0, 5500.0, 8000.0])
    out = apply_extinction(flux, wavelength=wave, ext_params=torch.tensor(0.0))
    torch.testing.assert_close(out, flux)


def test_apply_extinction_rejects_removed_wavelength_alias() -> None:
    with pytest.raises(TypeError, match="lambda_obs_angstrom"):
        cast(Any, apply_extinction)(
            torch.ones(3),
            ext_params=torch.tensor(0.0),
            lambda_obs_angstrom=torch.ones(3),
        )


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


def test_fitzpatrick99_matches_kbarbary_optical_anchors() -> None:
    # kbarbary/extinction Fitzpatrick99(3.1)([3000, 4000], 1.0)
    wave = torch.tensor([3000.0, 4000.0])
    got = fitzpatrick99(wave, torch.tensor(1.0), 3.1)
    torch.testing.assert_close(
        got,
        torch.tensor([1.79939955, 1.42338583]),
        atol=5e-3,
        rtol=0,
    )


def test_calzetti_v_is_a_v() -> None:
    wave = torch.tensor([5500.0])
    mag = calzetti(wave, torch.tensor(1.0))
    assert mag.item() == pytest.approx(1.0, abs=0.02)
    k = k_lambda(wave, law="calzetti", r_v=4.05)
    torch.testing.assert_close(k, mag)


def test_host_and_mw_laws_differ_in_uv() -> None:
    uv = torch.tensor([2000.0])
    f99 = fitzpatrick99(uv, torch.tensor(1.0), 3.1)
    cal = calzetti(uv, torch.tensor(1.0))
    ccm = ccm89(uv, torch.tensor(1.0), 3.1)
    assert f99.item() != pytest.approx(cal.item(), rel=0.02)
    assert f99.item() != pytest.approx(ccm.item(), rel=0.02)


def test_f99_optional_dust_extinction_parity() -> None:
    pytest.importorskip("dust_extinction")
    pytest.importorskip("astropy.units")
    from astropy import units as u
    from dust_extinction.parameter_averages import F99

    wave = torch.tensor([1500.0, 3000.0, 5500.0, 8000.0])
    got = fitzpatrick99(wave, torch.tensor(1.0), 3.1)
    ref = torch.tensor(
        [float(F99(Rv=3.1)(float(w) * u.Angstrom)) for w in wave],
        dtype=got.dtype,
    )
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


# A(λ)/A(V) from dust_extinction 1.7 (K. Gordon, BSD-3-Clause), the reference
# implementation of these laws, at R_V = 2.5, 3.1, 5.0.
_DUST_EXTINCTION_REFERENCE = {
    "g23": (
        [
            1000.0,
            1500.0,
            2175.0,
            3100.0,
            4400.0,
            5500.0,
            10000.0,
            22000.0,
            98000.0,
            200000.0,
        ],
        {
            2.5: [
                7.07162967271,
                3.61980669258,
                3.92073114306,
                2.07237900604,
                1.39544917216,
                0.998204954128,
                0.320757048028,
                0.0686489185088,
                0.077978345895,
                0.038275030712,
            ],
            3.1: [
                5.14408310829,
                2.82354704072,
                3.22035808515,
                1.81486758871,
                1.32041499934,
                0.998505968387,
                0.395680326042,
                0.103110029219,
                0.0850407954221,
                0.0415882981779,
            ],
            5.0: [
                2.0921343813,
                1.56280259195,
                2.11143407681,
                1.40714117794,
                1.20161089237,
                0.998982574297,
                0.514308849563,
                0.15767345451,
                0.0962230071733,
                0.0468343049989,
            ],
        },
    ),
    "f19": (
        [1200.0, 2175.0, 3100.0, 4400.0, 5500.0, 8000.0, 20000.0],
        {
            2.5: [
                4.4597106329,
                3.92197557679,
                2.00250148832,
                1.39979088583,
                1.00017660567,
                0.5606448,
                0.120563341736,
            ],
            3.1: [
                3.49750990775,
                3.18247982373,
                1.78945101289,
                1.32241332519,
                1.00015166451,
                0.607741935484,
                0.135020069516,
            ],
            5.0: [
                1.97402542627,
                2.01161154807,
                1.45212109344,
                1.19989885418,
                1.00011217434,
                0.6823124,
                0.157909888502,
            ],
        },
    ),
    "gcc09": (
        [950.0, 1200.0, 1500.0, 2175.0, 2800.0],
        {
            2.5: [
                8.59650371706,
                5.04129561792,
                3.62289086449,
                3.97859787129,
                2.3287431064,
            ],
            3.1: [
                5.89194126788,
                3.67644218099,
                2.79314950486,
                3.19329305505,
                1.98487743876,
            ],
            5.0: [1.60971739, 1.51542423918, 1.4793923521, 1.94989376266, 1.440423465],
        },
    ),
}


@pytest.mark.parametrize("law", ["g23", "f19", "gcc09"])
@pytest.mark.parametrize("r_v", [2.5, 3.1, 5.0])
def test_recent_laws_match_dust_extinction_values(law: str, r_v: float) -> None:
    waves, table = _DUST_EXTINCTION_REFERENCE[law]
    wave = torch.tensor(waves, dtype=torch.float64)
    got = k_lambda(wave, law=cast(Any, law), r_v=r_v)
    want = torch.tensor(table[r_v], dtype=torch.float64)
    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize(
    ("law", "cls_name", "lo_um", "hi_um", "r_vs"),
    [
        ("g23", "G23", 0.0912, 32.0, (2.3, 3.1, 4.0, 5.6)),
        ("f19", "F19", 1.0 / 8.7, 1.0 / 0.3, (2.0, 3.1, 4.0, 6.0)),
        ("gcc09", "GCC09", 1.0 / 11.0, 1.0 / 3.3, (2.0, 3.1, 4.0, 6.0)),
    ],
)
def test_recent_laws_match_live_dust_extinction(
    law, cls_name, lo_um, hi_um, r_vs
) -> None:
    pytest.importorskip("dust_extinction")
    import numpy as np
    from astropy import units as u
    from dust_extinction import parameter_averages

    x = np.linspace(1.0 / hi_um, 1.0 / lo_um, 2001)
    wave = torch.tensor(1.0e4 / x, dtype=torch.float64)
    for r_v in r_vs:
        ref = getattr(parameter_averages, cls_name)(Rv=r_v)(x / u.micron)
        got = k_lambda(wave, law=law, r_v=r_v)
        torch.testing.assert_close(
            got, torch.tensor(ref, dtype=torch.float64), rtol=1e-12, atol=1e-13
        )


@pytest.mark.parametrize("law", ["g23", "f19", "gcc09"])
def test_recent_laws_fail_closed_outside_their_domain(law: str) -> None:
    inside = {"g23": 5500.0, "f19": 5500.0, "gcc09": 2000.0}[law]
    outside = {"g23": 500.0, "f19": 40000.0, "gcc09": 5500.0}[law]
    with pytest.raises(ValueError, match=law):
        a_lambda(torch.tensor([outside], dtype=torch.float64), a_v=1.0, law=law)
    with pytest.raises(ValueError, match="R_V"):
        a_lambda(torch.tensor([inside], dtype=torch.float64), a_v=1.0, r_v=7.0, law=law)


@pytest.mark.parametrize("law", ["g23", "f19", "gcc09"])
def test_recent_laws_gradients(law: str) -> None:
    wave = {
        "g23": torch.tensor([1200.0, 3150.0, 5500.0, 10000.0, 98000.0]),
        "f19": torch.tensor([1500.0, 4400.0, 9000.0]),
        "gcc09": torch.tensor([1000.0, 2175.0, 3000.0]),
    }[law].to(torch.float64)
    ebv = torch.tensor([0.1, 0.4], dtype=torch.float64, requires_grad=True)
    r_v = torch.tensor(3.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda e, r: a_lambda(wave, ebv=e, r_v=r, law=cast(Any, law)), (ebv, r_v)
    )
    wave_grad = wave.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda w: a_lambda(w, a_v=1.0, r_v=3.1, law=cast(Any, law)), (wave_grad,)
    )


@pytest.mark.parametrize("law", ["g23", "f19", "gcc09"])
def test_recent_laws_on_device(law: str, compute_device) -> None:
    dtype = torch.float32 if compute_device.type == "mps" else torch.float64
    wave = {
        "g23": torch.tensor([1200.0, 5500.0, 22000.0]),
        "f19": torch.tensor([1500.0, 5500.0, 9000.0]),
        "gcc09": torch.tensor([1000.0, 2175.0, 3000.0]),
    }[law].to(dtype)
    ebv = torch.tensor([[0.1], [0.4]], dtype=dtype)
    expected = transmission(wave, ebv, law=cast(Any, law))
    got = transmission(
        wave.to(compute_device), ebv.to(compute_device), law=cast(Any, law)
    )
    assert got.device.type == compute_device.type and got.shape == (2, 3)
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-5, atol=1e-6)
