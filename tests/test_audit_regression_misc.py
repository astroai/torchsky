"""Regression tests for audit findings: cosmology, caches, clamps, redshift."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchsky.cosmology as C
import torchsky.extinction as E
from torchsky.redshift import one_plus_z_from_beta, zeta_from_beta
from torchsky.sphere import pairwise_angular_distance
from torchsky.sphere import spectral as S

F64 = torch.float64
ZS = [1e-5, 1e-4, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1100.0]


@pytest.mark.parametrize(
    ("om", "ok", "w0", "wa"),
    [(0.3, 0.0, -1.0, 0.0), (0.3, 0.05, -0.9, 0.2), (0.3, -0.05, -1.1, -0.3)],
)
def test_distances_match_astropy_and_are_batch_independent(om, ok, w0, wa) -> None:
    astropy_cosmo = pytest.importorskip("astropy.cosmology")
    ref = astropy_cosmo.w0waCDM(H0=70, Om0=om, Ode0=1 - om - ok, w0=w0, wa=wa, Tcmb0=0)
    kw = {"H0": 70.0, "om0": om, "ok0": ok, "w0": w0, "wa": wa}
    z = torch.tensor(ZS, dtype=F64)
    d_l = C.luminosity_distance(z, **kw).numpy()
    d_a = C.angular_diameter_distance(z, **kw).numpy()
    np.testing.assert_allclose(d_l, ref.luminosity_distance(ZS).value, rtol=1e-6)
    np.testing.assert_allclose(d_a, ref.angular_diameter_distance(ZS).value, rtol=1e-6)
    for i, zi in enumerate(ZS):
        alone = C.luminosity_distance(torch.tensor([zi], dtype=F64), **kw)
        # Independent of the batch to float64 rounding (vectorized reductions may differ by 1 ulp).
        assert alone.item() == pytest.approx(d_l[i], rel=1e-14)


def test_comoving_distance_gradient_at_zero_redshift() -> None:
    z = torch.tensor([0.0, 1e-3], dtype=F64, requires_grad=True)
    C.comoving_distance(z, H0=70.0, om0=0.3).sum().backward()
    hubble = C.SPEED_OF_LIGHT_KM_S / 70.0
    assert z.grad[0].item() == pytest.approx(hubble, rel=1e-12)


def test_extinction_transmission_gradient_at_zero_ebv() -> None:
    ebv = torch.tensor(0.0, dtype=F64, requires_grad=True)
    wave = torch.tensor([5000.0], dtype=F64)
    E.transmission(wave, ebv=ebv).sum().backward()
    a_per_ebv = E.a_lambda(wave, ebv=torch.tensor(1.0, dtype=F64)).item()
    assert ebv.grad.item() == pytest.approx(-0.4 * np.log(10.0) * a_per_ebv, rel=1e-10)


def test_caches_built_in_inference_mode_do_not_break_autograd() -> None:
    wave = torch.tensor([2000.0, 3000.0, 4000.0], dtype=F64)
    with torch.inference_mode():
        E.fitzpatrick99(wave, torch.tensor(1.0, dtype=F64), 3.1)
        S.map2alm(torch.randn(12 * 64, dtype=F64), lmax=10)
    rv = torch.tensor(3.1, dtype=F64, requires_grad=True)
    E.fitzpatrick99(wave, torch.tensor(1.0, dtype=F64), rv).sum().backward()
    assert torch.isfinite(rv.grad)
    m = torch.randn(12 * 64, dtype=F64, requires_grad=True)
    S.map2alm(m, lmax=10).abs().sum().backward()
    assert torch.isfinite(m.grad).all()


def test_pairwise_angular_distance_duplicate_points_have_finite_gradient() -> None:
    lon = torch.tensor([10.0, 10.0, 50.0], dtype=F64, requires_grad=True)
    lat = torch.tensor([20.0, 20.0, 0.0], dtype=F64, requires_grad=True)
    d = pairwise_angular_distance(lon, lat)
    assert d[0, 1].item() == 0.0
    d.sum().backward()
    assert torch.isfinite(lon.grad).all()
    assert torch.isfinite(lat.grad).all()


def test_superluminal_beta_raises_instead_of_clamping() -> None:
    for bad in (1.0, -1.0, 1.5):
        with pytest.raises(ValueError):
            one_plus_z_from_beta(torch.tensor([bad], dtype=F64))
        with pytest.raises(ValueError):
            zeta_from_beta(bad)
    z = one_plus_z_from_beta(torch.tensor([0.6], dtype=F64)).item()
    assert z == pytest.approx(2.0)


def test_passband_interpolation_gradient_at_grid_node() -> None:
    from torchsky.spectrophotometry import Passband

    grid = torch.tensor([1.0, 2.0, 3.0], dtype=F64)
    band = Passband(grid, torch.tensor([0.0, 1.0, 0.0], dtype=F64))
    q = torch.tensor([1.5, 2.0, 3.0], dtype=F64, requires_grad=True)
    band.evaluate(q).sum().backward()
    assert q.grad.tolist() == pytest.approx([1.0, 1.0, -1.0])
