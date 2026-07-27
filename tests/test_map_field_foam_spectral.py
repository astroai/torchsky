"""Shared-geometry spectral contracts for bounded Power Foam."""

from __future__ import annotations

import pytest
import torch

from torchsky.map.field import (
    PowerPixelOperator,
    bounded_power_pixel_operator,
    render_power_spectral_basis,
    render_power_spectral_samples,
)


def _two_cell_operator() -> PowerPixelOperator:
    pixel = torch.tensor(
        [[[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]]],
        dtype=torch.float64,
    )
    sites = torch.tensor([[-0.5, 0.0], [0.5, 0.0]], dtype=torch.float64)
    radii = torch.full((2,), 2.0, dtype=torch.float64)
    return bounded_power_pixel_operator(pixel, sites, radii)


@pytest.mark.parametrize("semantics", ["surface_brightness", "integrated_flux"])
def test_spectral_samples_share_exact_geometry_and_preserve_batches(
    semantics: str,
) -> None:
    operator = _two_cell_operator()
    cell_samples = torch.tensor(
        [
            [[1.0, 4.0], [2.0, 3.0], [5.0, 1.0], [7.0, -2.0]],
            [[-1.0, 2.0], [0.0, 3.0], [2.0, 4.0], [6.0, 8.0]],
        ],
        dtype=torch.float64,
    )
    responses = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.25, 0.5, 0.25],
        ],
        dtype=torch.float64,
    )

    rendered = render_power_spectral_samples(
        operator,
        cell_samples,
        responses,
        semantics=semantics,
    )
    channel_cells = torch.einsum("bs,...sc->...bc", responses, cell_samples)
    expected = operator.matvec(channel_cells, semantics=semantics)

    assert rendered.shape == (2, 4, 1)
    torch.testing.assert_close(rendered, expected)
    single = render_power_spectral_samples(
        operator,
        cell_samples,
        responses[2],
        semantics=semantics,
    )
    torch.testing.assert_close(single, expected[:, 2])


def test_spectral_basis_matches_explicit_samples_and_is_differentiable() -> None:
    operator = _two_cell_operator()
    coefficients = torch.tensor(
        [[1.0, 0.2, -0.1], [0.5, -0.3, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    basis = torch.tensor(
        [
            [1.0, -1.0, 1.0],
            [1.0, -0.25, 0.0625],
            [1.0, 0.5, 0.25],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    responses = torch.tensor(
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.25, 0.5, 0.25]],
        dtype=torch.float64,
    )

    rendered = render_power_spectral_basis(
        operator,
        coefficients,
        basis,
        responses,
    )
    samples = torch.einsum("ck,sk->sc", coefficients, basis)
    expected = render_power_spectral_samples(operator, samples, responses)
    torch.testing.assert_close(rendered, expected)

    torch.square(rendered).sum().backward()
    assert coefficients.grad is not None
    assert basis.grad is not None
    assert bool(torch.isfinite(coefficients.grad).all())
    assert bool(torch.isfinite(basis.grad).all())


def test_spectral_response_autograd_contains_exact_spatial_adjoint() -> None:
    operator = _two_cell_operator()
    samples = torch.tensor(
        [[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    responses = torch.tensor(
        [[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]],
        dtype=torch.float64,
    )
    pixel_dual = torch.tensor([[1.3], [-0.4]], dtype=torch.float64)

    rendered = render_power_spectral_samples(operator, samples, responses)
    torch.sum(rendered * pixel_dual).backward()

    band_cell_adjoint = operator.rmatvec(pixel_dual)
    expected = torch.einsum("bs,bc->sc", responses, band_cell_adjoint)
    torch.testing.assert_close(samples.grad, expected)


def test_spectral_rendering_validation_fails_closed() -> None:
    operator = _two_cell_operator()
    samples = torch.ones((3, 2), dtype=torch.float64)
    responses = torch.ones(3, dtype=torch.float64)

    with pytest.raises(ValueError, match="cell_samples"):
        render_power_spectral_samples(operator, torch.ones(3), responses)
    with pytest.raises(ValueError, match="sample count"):
        render_power_spectral_samples(operator, samples, torch.ones(4))
    with pytest.raises(ValueError, match="finite"):
        render_power_spectral_samples(
            operator,
            samples,
            torch.tensor([1.0, float("nan"), 0.0]),
        )
    with pytest.raises(ValueError, match="cell_coefficients"):
        render_power_spectral_basis(
            operator,
            torch.ones((3, 2), dtype=torch.float64),
            torch.ones((4, 2), dtype=torch.float64),
            torch.ones(4),
        )
    with pytest.raises(ValueError, match="basis_samples"):
        render_power_spectral_basis(
            operator,
            torch.ones((2, 3), dtype=torch.float64),
            torch.ones((4, 2), dtype=torch.float64),
            torch.ones(4),
        )
