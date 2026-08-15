"""Synthetic FOSSIL operator gates on the current Foam / spectral spine.

These checks cover adjoint identity, paired μ-scale oddness, smooth-foreground
leakage, and held-out isolation. They are not a TensorSpec / ProtectedSubspace
framework and do not replace Phase D cosmology benches.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchsky.map.field import (
    PowerPixelOperator,
    bounded_power_pixel_operator,
    render_power_spectral_samples,
)

# Standard primordial μ amplitude used as the weak-target scale.
_MU_AMP = 2.0e-8
# Photon-number conserving pivot for the μ template (Chluba / Fixsen).
_MU_X0 = 2.1923
_T_CMB_K = 2.7255
_H_SI = 6.62607015e-34
_K_SI = 1.380649e-23
# Noiseless orthogonal recovery should be essentially exact; keep a declared bound.
_MU_ATTENUATION_BOUND = 0.25
_FG_LEAKAGE_FRAC = 1.0e-4


def _two_pixel_operator() -> PowerPixelOperator:
    pixels = torch.tensor(
        [
            [[-1.0, -0.5], [0.0, -0.5], [0.0, 0.5], [-1.0, 0.5]],
            [[0.0, -0.5], [1.0, -0.5], [1.0, 0.5], [0.0, 0.5]],
        ],
        dtype=torch.float64,
    )
    sites = torch.tensor([[-0.5, 0.0], [0.5, 0.0]], dtype=torch.float64)
    radii = torch.full((2,), 2.0, dtype=torch.float64)
    return bounded_power_pixel_operator(pixels, sites, radii)


def _mu_template(nu_hz: np.ndarray) -> np.ndarray:
    """Number-conserving μ distortion shape M(x) = G(x) (x/x0 - 1).

    ``x = h ν / k T_CMB``. ``G(x)`` is the Planck thermodynamic weight
    ``x e^x / (e^x - 1)^2``. Dimensionless; multiply by a brightness scale
    to form a relative intensity perturbation.
    """
    x = (_H_SI * np.asarray(nu_hz, dtype=np.float64)) / (_K_SI * _T_CMB_K)
    x = np.clip(x, 1.0e-8, 50.0)
    ex = np.exp(x)
    planck_weight = x * ex / np.square(ex - 1.0)
    return planck_weight * (x / _MU_X0 - 1.0)


def _log_poly_basis(nu_hz: np.ndarray, degree: int) -> np.ndarray:
    x = np.log(nu_hz / np.median(nu_hz))
    return np.stack([np.power(x, k) for k in range(degree + 1)], axis=1)


def _foreground_in_subspace(
    nu_hz: np.ndarray, degree: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    basis = _log_poly_basis(nu_hz, degree)
    # Intensity stays in the linear nuisance span and positive on this grid.
    coeff = np.zeros(degree + 1, dtype=np.float64)
    coeff[0] = 2.0
    coeff[1] = -0.3
    coeff[2] = 0.05
    fg = basis @ coeff
    if float(np.min(fg)) <= 0.0:
        raise RuntimeError("synthetic foreground left the positive cone")
    return fg, basis


def _lstsq_fit(basis: np.ndarray, y: np.ndarray) -> np.ndarray:
    coeff, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return basis @ coeff


def _matched_amp(residual: np.ndarray, template: np.ndarray) -> float:
    denom = float(np.dot(template, template))
    assert denom > 0.0
    return float(np.dot(template, residual) / denom)


def _nu_grid() -> np.ndarray:
    # CMB-relevant frequencies; x spans below and above the μ null.
    return np.logspace(10.5, 11.9, 48)  # ~30 GHz to ~800 GHz


def test_mu_template_null_at_photon_number_pivot() -> None:
    nu = _MU_X0 * _K_SI * _T_CMB_K / _H_SI
    val = float(_mu_template(np.array([nu]))[0])
    assert abs(val) < 1.0e-12


def test_foam_spectral_mixing_adjoint_identity() -> None:
    operator = _two_pixel_operator()
    samples = torch.tensor(
        [[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75], [0.5, 0.5]],
        dtype=torch.float64,
    )
    responses = torch.tensor(
        [[0.5, 0.25, 0.25, 0.0], [0.0, 0.5, 0.0, 0.5]],
        dtype=torch.float64,
    )
    y = torch.tensor([[1.2, -0.7], [0.4, 0.9]], dtype=torch.float64)

    rendered = render_power_spectral_samples(operator, samples, responses)
    lhs = float(torch.sum(rendered * y))
    cell_adj = operator.rmatvec(y)
    sample_adj = torch.einsum("bs,bc->sc", responses, cell_adj)
    rhs = float(torch.sum(samples * sample_adj))
    assert lhs == pytest.approx(rhs, rel=0.0, abs=1.0e-12)

    # Independence: the same adjoint via explicit channel mix then rmatvec.
    channel_cells = torch.einsum("bs,sc->bc", responses, samples)
    mixed = operator.matvec(channel_cells)
    torch.testing.assert_close(rendered, mixed)
    assert float(torch.sum(mixed * y)) == pytest.approx(rhs, rel=0.0, abs=1.0e-12)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_mu_scale_paired_injection_oddness(sign: float) -> None:
    nu = _nu_grid()
    foreground, basis = _foreground_in_subspace(nu)
    shape = _mu_template(nu)
    scale = float(np.mean(foreground) / np.sqrt(np.mean(np.square(shape))))
    injected = sign * _MU_AMP * scale * shape
    obs = foreground + injected

    residual = obs - _lstsq_fit(basis, obs)
    template_perp = shape - _lstsq_fit(basis, shape)
    recoverable = _matched_amp(injected, template_perp)
    recovered = _matched_amp(residual, template_perp)

    assert recoverable != 0.0
    assert np.sign(recovered) == np.sign(sign)
    assert np.sign(recoverable) == np.sign(sign)
    attenuation = abs(recovered / recoverable - 1.0)
    assert attenuation <= _MU_ATTENUATION_BOUND


def test_smooth_foreground_leakage_bound() -> None:
    operator = _two_pixel_operator()
    nu = _nu_grid()
    fg, basis = _foreground_in_subspace(nu)
    shape = _mu_template(nu)
    scale = float(np.mean(fg) / np.sqrt(np.mean(np.square(shape))))
    target = _MU_AMP * scale * shape

    n_nu = int(nu.size)
    cell_samples = torch.from_numpy(np.repeat(fg[:, None], operator.cell_count, axis=1))
    responses = torch.eye(n_nu, dtype=torch.float64)
    rendered = render_power_spectral_samples(operator, cell_samples, responses)
    obs = rendered[:, 0].detach().cpu().numpy() + target

    residual = obs - _lstsq_fit(basis, obs)
    fg_power = float(np.dot(fg, fg))
    fg_in_residual = float(np.dot(residual, fg) ** 2 / fg_power)
    assert fg_in_residual / fg_power < _FG_LEAKAGE_FRAC

    template_perp = shape - _lstsq_fit(basis, shape)
    recovered = _matched_amp(residual, template_perp)
    assert recovered > 0.0


def test_held_out_isolation_does_not_see_target() -> None:
    operator = _two_pixel_operator()
    nu = _nu_grid()
    fg, basis = _foreground_in_subspace(nu)
    shape = _mu_template(nu)
    scale = float(np.mean(fg) / np.sqrt(np.mean(np.square(shape))))
    target = _MU_AMP * scale * shape

    n_nu = int(nu.size)
    cell_samples = torch.from_numpy(np.repeat(fg[:, None], operator.cell_count, axis=1))
    responses = torch.eye(n_nu, dtype=torch.float64)
    rendered = render_power_spectral_samples(operator, cell_samples, responses)
    y_train = rendered[:, 0].detach().cpu().numpy()
    y_hold = rendered[:, 1].detach().cpu().numpy() + target

    coeff, *_ = np.linalg.lstsq(basis, y_train, rcond=None)
    train_fit = basis @ coeff
    # Spatial gain from the μ null so the held-out target cannot set the scale.
    x = (_H_SI * nu) / (_K_SI * _T_CMB_K)
    i_null = int(np.argmin(np.abs(x - _MU_X0)))
    gain = float(y_hold[i_null] / y_train[i_null])
    train_residual = y_train - train_fit
    hold_residual = y_hold - gain * train_fit
    template_perp = shape - _lstsq_fit(basis, shape)

    mu_train = _matched_amp(train_residual, template_perp)
    mu_hold = _matched_amp(hold_residual, template_perp)
    recoverable = _matched_amp(target, template_perp)

    assert abs(mu_train) < 0.05 * abs(recoverable)
    assert np.sign(mu_hold) == np.sign(recoverable)
    assert abs(mu_hold / recoverable - 1.0) <= _MU_ATTENUATION_BOUND
