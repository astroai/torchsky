"""Tests for the differentiable lensing (magnification) operators."""

import pytest
import torch

from torchsky.map.operators import (
    ConvergenceMagnification,
    MagnificationResidual,
    magnification_bias,
    magnify_flux,
    scale_source_sizes,
)


def test_linear_magnification_law():
    kappa = torch.tensor([[[0.0, 0.1], [-0.2, 0.05]]])
    operator = ConvergenceMagnification(mode="linear", max_magnification=None)
    mu = operator(kappa)
    torch.testing.assert_close(mu, 1.0 + 2.0 * kappa, rtol=0, atol=0)


def test_exponential_magnification_is_positive_and_clamped():
    kappa = torch.tensor([-3.0, 0.0, 3.0])
    unclamped = ConvergenceMagnification(mode="exponential", max_magnification=None)
    mu = unclamped(kappa)
    assert bool((mu > 0).all())
    torch.testing.assert_close(mu, torch.exp(2.0 * kappa), rtol=1e-6, atol=0)
    clamped = ConvergenceMagnification(mode="exponential", max_magnification=10.0)
    assert float(clamped(kappa).max()) <= 10.0
    assert float(clamped(kappa).min()) >= 0.1


def test_flux_and_size_scaling():
    kappa = torch.tensor([0.1, -0.05])
    flux = torch.tensor([100.0, 50.0])
    torch.testing.assert_close(
        magnify_flux(flux, kappa), flux * (1.0 + 2.0 * kappa), rtol=1e-6, atol=0
    )
    sigma = torch.tensor([2.0, 3.0])
    torch.testing.assert_close(
        scale_source_sizes(sigma, kappa),
        sigma * torch.sqrt(1.0 + 2.0 * kappa),
        rtol=1e-6,
        atol=0,
    )


def test_magnification_bias_exponent():
    kappa = torch.tensor([0.1])
    density = torch.tensor([1.0])
    biased = magnification_bias(density, kappa, count_slope=1.0)
    torch.testing.assert_close(biased, density, rtol=1e-6, atol=0)
    biased = magnification_bias(density, kappa, count_slope=3.0)
    torch.testing.assert_close(
        biased, density * (1.0 + 2.0 * kappa).pow(2.0), rtol=1e-6, atol=0
    )


def test_operator_shape_handling_and_validation():
    operator = ConvergenceMagnification()
    kappa_3d = torch.zeros(2, 4, 4)
    assert operator(kappa_3d).shape == (2, 4, 4)
    assert operator(kappa_3d.unsqueeze(1)).shape == (2, 1, 4, 4)
    with pytest.raises(ValueError, match="shape"):
        operator(torch.zeros(4, 4))
    with pytest.raises(ValueError, match="finite"):
        operator(torch.full((1, 2, 2), float("nan")))
    with pytest.raises(ValueError, match="mode"):
        ConvergenceMagnification(mode="quadratic")
    with pytest.raises(ValueError, match="max_magnification"):
        ConvergenceMagnification(max_magnification=0.5)


def test_learned_residual_is_identity_at_initialisation():
    torch.manual_seed(0)
    operator = ConvergenceMagnification(residual=True)
    kappa = torch.randn(2, 1, 8, 8) * 0.05
    torch.testing.assert_close(operator(kappa), 1.0 + 2.0 * kappa, rtol=0, atol=0)
    assert float(operator.regularization()) == 0.0


def test_learned_residual_opens_and_receives_gradients():
    torch.manual_seed(0)
    operator = ConvergenceMagnification(residual=True)
    kappa = (torch.randn(2, 1, 8, 8) * 0.05).requires_grad_(True)
    mu = operator(kappa)
    target = 1.0 + 2.0 * kappa.detach() + 0.1
    loss = torch.nn.functional.mse_loss(mu, target) + 0.01 * operator.regularization()
    loss.backward()
    assert kappa.grad is not None and bool((kappa.grad != 0).any())
    residual_gate = operator.residual.gate
    assert residual_gate.grad is not None and float(residual_gate.grad.abs()) > 0.0
    with torch.no_grad():
        residual_gate.add_(0.1)
    moved = operator(kappa)
    assert not torch.allclose(mu.detach(), moved.detach())
    assert float(operator.regularization()) > 0.0


def test_residual_rejects_even_kernel():
    with pytest.raises(ValueError, match="odd"):
        MagnificationResidual(kernel_size=4)
    with pytest.raises(ValueError, match="channels"):
        MagnificationResidual(channels=0)
