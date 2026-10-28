"""Independent dense/closed-form AD checks for nuisance spectral placement."""

import pytest
import torch

from torchsky.map.operators.nuisance import (
    CalibratedNuisancePrecision,
    ProfiledNuisancePrecision,
)


def _parameters(kind, *, device, dtype):
    if kind == "singular_profile":
        data = ([1.0, 0.5, -0.3, 1.4], [1.2, 0.8, 2.0, 1.5], [0.4, -0.5, 1.0, 0.3])
    else:
        data = (
            [[1.0, -0.3], [0.5, 1.0], [-0.3, 0.7], [1.4, 0.2]],
            [1.2, 0.8, 2.0, 1.5],
            [0.4, -0.5, 1.0, 0.3],
        )
        if kind == "calibrated":
            # Three proportional columns exercise singular calibration covariance.
            data += ([[0.3], [-0.2]], [0.1, -0.2])
    return tuple(
        torch.tensor(value, dtype=dtype, device=device, requires_grad=True)
        for value in data
    )


def _loss(kind, args, *, reference):
    basis, weights, residual = args[:3]
    if kind == "singular_profile":
        if reference:
            # Only one independent column; no pseudoinverse in this reference.
            return (weights * residual.square()).sum() - (
                weights * basis * residual
            ).sum().square() / (weights * basis.square()).sum()
        basis = torch.stack((basis, 2 * basis, torch.zeros_like(basis)), dim=-1)
    if kind != "calibrated":
        if reference:
            weighted_basis = weights[:, None] * basis
            score = weighted_basis.T @ residual
            return (weights * residual.square()).sum() - score @ torch.linalg.solve(
                basis.T @ weighted_basis, score
            )
        return ProfiledNuisancePrecision(
            basis, apply_precision=lambda x: weights * x
        ).quadratic(residual)
    factor, mean = args[3:]
    factor = factor @ factor.new_tensor([[1.0, 2.0, -0.5]])
    if reference:
        covariance = (
            torch.diag(weights.reciprocal()) + basis @ factor @ factor.T @ basis.T
        )
        centered = residual - basis @ mean
        return (
            centered @ torch.linalg.solve(covariance, centered)
            + torch.linalg.slogdet(covariance).logabsdet
            + weights.log().sum()
        )
    operator = CalibratedNuisancePrecision(
        basis,
        apply_precision=lambda x: weights * x,
        covariance_factor=factor,
        mean=mean,
    )
    return (
        operator.quadratic(operator.center(residual))
        + operator.logdet_covariance_correction
    )


def _derivatives(kind, args, *, reference):
    value = _loss(kind, args, reference=reference)
    gradients = torch.autograd.grad(value, args, create_graph=True)
    direction = sum(
        (
            gradient
            * torch.linspace(
                0.2, 0.8, gradient.numel(), device=gradient.device, dtype=gradient.dtype
            ).reshape(gradient.shape)
        ).sum()
        for gradient in gradients
    )
    second = torch.autograd.grad(direction, args)
    return (value.detach().cpu().double(),) + tuple(
        t.detach().cpu().double() for t in (*gradients, *second)
    )


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("kind", ["full_profile", "singular_profile", "calibrated"])
def test_nuisance_first_second_device_ad_matches_independent_reference(device, kind):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    args = _parameters(kind, device=device, dtype=torch.float32)
    expected_args = tuple(x.detach().cpu().double().requires_grad_() for x in args)
    actual = _derivatives(kind, args, reference=False)
    expected = _derivatives(kind, expected_args, reference=True)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("kind", ["full_profile", "singular_profile", "calibrated"])
def test_nuisance_cpu_double_first_second_independent_reference(kind):
    args = _parameters(kind, device="cpu", dtype=torch.float64)
    expected_args = tuple(x.detach().clone().requires_grad_() for x in args)
    torch.testing.assert_close(
        _derivatives(kind, args, reference=False),
        _derivatives(kind, expected_args, reference=True),
        rtol=1e-10,
        atol=1e-11,
    )
