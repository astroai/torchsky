"""Trainable shared-geometry spectral Power Foam contracts."""

from __future__ import annotations

import pytest
import torch

from torchsky.map.field import (
    SpectralPowerFoam,
    bounded_power_pixel_operator,
    render_power_spectral_basis,
    smooth_bounded_power_weights,
)


def _model() -> SpectralPowerFoam:
    return SpectralPowerFoam(
        torch.tensor([[-0.5, 0.0], [0.5, 0.0]], dtype=torch.float64),
        torch.tensor([0.8, 0.9], dtype=torch.float64),
        torch.tensor([[1.0, 0.25], [2.0, -0.5]], dtype=torch.float64),
        minimum_radius=0.05,
    )


def test_spectral_power_foam_matches_functional_smooth_render() -> None:
    model = _model()
    points = torch.tensor(
        [[-0.6, 0.1], [0.4, -0.2], [1.5, 0.0]],
        dtype=torch.float64,
    )
    basis = torch.tensor(
        [[1.0, -0.5], [1.0, 0.2], [1.0, 0.8]],
        dtype=torch.float64,
    )

    actual = model(
        points,
        basis,
        power_temperature=7.0,
        support_temperature=9.0,
    )
    weights = smooth_bounded_power_weights(
        points,
        model.sites,
        model.radii,
        power_temperature=7.0,
        support_temperature=9.0,
    )
    cell_values = torch.einsum("pk,ck->pc", basis, model.cell_coefficients)
    expected = torch.sum(weights * cell_values, dim=-1)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        model.radii,
        torch.tensor([0.8, 0.9], dtype=torch.float64),
    )
    assert bool(torch.all(model.radii > 0.05))


def test_spectral_power_foam_has_finite_joint_geometry_and_spectral_gradients() -> None:
    model = _model()
    points = torch.tensor(
        [[-0.4, 0.2], [0.0, -0.1], [0.7, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    basis = torch.tensor(
        [[1.0, -0.7], [1.0, 0.0], [1.0, 0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch.square(model(points, basis) - 1.0).sum().backward()

    for parameter in (
        points,
        basis,
        model.sites,
        model.raw_radii,
        model.cell_coefficients,
    ):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())


@pytest.mark.parametrize("semantics", ["surface_brightness", "integrated_flux"])
def test_spectral_power_foam_exact_handoff_reuses_learned_coefficients(
    semantics: str,
) -> None:
    model = _model()
    pixels = torch.tensor(
        [[[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]]],
        dtype=torch.float64,
    )
    operator = bounded_power_pixel_operator(
        pixels,
        model.sites.detach(),
        model.radii.detach(),
    )
    basis_samples = torch.tensor(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    responses = torch.tensor(
        [[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]],
        dtype=torch.float64,
    )

    actual = model.render_exact(
        operator,
        basis_samples,
        responses,
        semantics=semantics,
    )
    expected = render_power_spectral_basis(
        operator,
        model.cell_coefficients,
        basis_samples,
        responses,
        semantics=semantics,
    )

    torch.testing.assert_close(actual, expected)


def test_spectral_power_foam_validation_fails_closed() -> None:
    sites = torch.zeros((2, 2), dtype=torch.float64)
    radii = torch.ones(2, dtype=torch.float64)
    coefficients = torch.ones((2, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="minimum_radius"):
        SpectralPowerFoam(
            sites,
            radii,
            coefficients,
            minimum_radius=-1.0,
        )
    with pytest.raises(ValueError, match="initial radii"):
        SpectralPowerFoam(
            sites,
            radii,
            coefficients,
            minimum_radius=2.0,
        )
    with pytest.raises(ValueError, match="cell_coefficients"):
        SpectralPowerFoam(sites, radii, torch.ones((3, 2), dtype=torch.float64))
    with pytest.raises(ValueError, match="basis_values"):
        _model()(torch.zeros((3, 2), dtype=torch.float64), torch.ones((3, 3)))
