"""Trainable shared-geometry spectral Power Foam contracts."""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.map.field import (
    FieldDomain,
    FieldObservation,
    DiffuseFoamHybridField,
    FoamAdaptationResult,
    FoamInitializedINRField,
    ForwardContext,
    SpectralPowerFoam,
    SpectralPowerFoamField,
    adapt_spectral_power_foam,
    bounded_power_pixel_operator,
    fit_field_model,
    fit_diffuse_foam_hybrid,
    make_field_model,
    make_tangent_plane_header,
    persistent_foam_site_initialization,
    render_power_spectral_basis,
    render_field_observation,
    render_diffuse_foam_hybrid_observation_exact,
    render_spectral_power_foam_observation_exact,
    smooth_bounded_power_weights,
    solve_power_pixel_least_squares,
    solve_power_pixel_affine_least_squares,
    solve_spectral_cell_coefficients,
    solve_spectral_power_pixel_affine_least_squares,
    spectral_basis_whitening_transform,
    soft_bounded_power_weights,
)
from torchsky.map.operators import (
    AdaptiveSpatioSpectralTemporalEncoder,
    ConditionedBackgroundField,
    ConditionedKernelPSF,
    LearnedBackgroundOp,
    LearnedPSFOp,
    ObservationOperatorBundle,
)


def test_spectral_basis_whitening_is_orthonormal_and_differentiable() -> None:
    response_basis = torch.tensor(
        [[1.0, -0.8], [1.0, -0.1], [1.0, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )

    transform = spectral_basis_whitening_transform(response_basis)
    whitened = response_basis @ transform

    torch.testing.assert_close(
        whitened.mT @ whitened,
        torch.eye(2, dtype=torch.float64),
    )
    torch.square(whitened).sum().backward()
    assert response_basis.grad is not None
    assert bool(torch.isfinite(response_basis.grad).all())


def test_persistent_foam_sites_require_coherent_multi_exposure_evidence() -> None:
    persistent = torch.tensor([[-0.55, -0.55], [-0.45, -0.45]])
    transient = torch.tensor([[0.45, 0.45], [0.55, 0.55]])
    coordinates = tuple(torch.cat([persistent, transient], dim=0) for _ in range(3))
    residuals = (
        torch.tensor([3.0, 2.0, 5.0, 4.0]),
        torch.tensor([2.5, 2.0, 0.0, -1.0]),
        torch.tensor([2.0, 2.5, -0.5, 0.0]),
    )

    result = persistent_foam_site_initialization(
        coordinates,
        residuals,
        num_sites=1,
        min_exposures=2,
        bin_shape=(2, 2),
        lower=(-1.0, -1.0),
        upper=(1.0, 1.0),
        initial_radius=0.4,
    )
    permuted = persistent_foam_site_initialization(
        tuple(reversed(coordinates)),
        tuple(reversed(residuals)),
        num_sites=1,
        min_exposures=2,
        bin_shape=(2, 2),
        lower=(-1.0, -1.0),
        upper=(1.0, 1.0),
        initial_radius=0.4,
    )

    torch.testing.assert_close(result.sites, torch.tensor([[-0.5, -0.5]]))
    torch.testing.assert_close(result.radii, torch.tensor([0.4]))
    assert result.exposure_counts.tolist() == [3]
    torch.testing.assert_close(result.sites, permuted.sites)
    torch.testing.assert_close(result.scores, permuted.scores)


def test_foam_heldout_statistics_are_ema_buffers() -> None:
    model = SpectralPowerFoam(
        torch.tensor([[-0.5, 0.0], [0.5, 0.0]]),
        torch.tensor([0.6, 0.6]),
        torch.ones((2, 1)),
    )
    with pytest.raises(ValueError, match="held_out"):
        model.update_adaptation_statistics(
            torch.tensor([0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([[-0.4, 0.0]]),
            split="train",
        )
    model.update_adaptation_statistics(
        torch.tensor([0, 0, 1]),
        torch.tensor([2.0, 4.0, 1.0]),
        torch.tensor([0.5, 1.5, 2.0]),
        torch.tensor([[-0.5, 0.0], [-0.3, 0.2], [0.4, 0.0]]),
        split="held_out",
        decay=0.9,
    )
    torch.testing.assert_close(model.heldout_residual_ema, torch.tensor([10.0, 1.0]))
    torch.testing.assert_close(model.contribution_ema, torch.tensor([1.0, 2.0]))
    assert model.evidence_updates.tolist() == [1, 1]
    assert {
        "heldout_residual_ema",
        "contribution_ema",
        "residual_position_ema",
        "evidence_updates",
    }.issubset(model.state_dict())

    model.update_adaptation_statistics(
        torch.tensor([0]),
        torch.tensor([2.0]),
        torch.tensor([3.0]),
        torch.tensor([[-0.2, 0.1]]),
        split="held_out",
        decay=0.5,
    )
    torch.testing.assert_close(model.heldout_residual_ema, torch.tensor([7.0, 1.0]))
    torch.testing.assert_close(model.contribution_ema, torch.tensor([2.0, 2.0]))
    assert model.evidence_updates.tolist() == [2, 1]


def test_foam_adaptation_is_deterministic_and_preserves_adam_state() -> None:
    model = SpectralPowerFoam(
        torch.tensor([[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]),
        torch.tensor([0.4, 0.4, 0.4]),
        torch.tensor([[1.0], [2.0], [3.0]]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    points = torch.tensor([[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]])
    model(points, torch.ones((3, 1))).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    old_sites_parameter = model.sites
    old_sites_momentum = optimizer.state[old_sites_parameter]["exp_avg"].clone()
    old_coefficients = model.cell_coefficients.detach().clone()
    old_radii = model.radii.detach().clone()

    model.update_adaptation_statistics(
        torch.tensor([0, 1, 2]),
        torch.tensor([5.0, 1.0, 0.1]),
        torch.tensor([0.5, 1.0, 0.01]),
        torch.tensor([[-0.35, 0.1], [0.0, 0.0], [0.5, 0.0]]),
        split="held_out",
    )
    result = adapt_spectral_power_foam(
        model,
        optimizer,
        densify_count=1,
        prune_count=1,
        min_updates=1,
    )

    assert isinstance(result, FoamAdaptationResult)
    assert result.densified_cell_ids == (0,)
    assert result.pruned_cell_ids == (2,)
    assert model.sites.shape == (3, 2)
    torch.testing.assert_close(model.sites[-1], torch.tensor([-0.35, 0.1]))
    torch.testing.assert_close(model.cell_coefficients[-1], old_coefficients[0])
    torch.testing.assert_close(model.radii[0], old_radii[0] / math.sqrt(2.0))
    torch.testing.assert_close(model.radii[-1], old_radii[0] / math.sqrt(2.0))
    assert old_sites_parameter not in optimizer.state
    new_sites_momentum = optimizer.state[model.sites]["exp_avg"]
    torch.testing.assert_close(new_sites_momentum[:2], old_sites_momentum[:2])
    torch.testing.assert_close(new_sites_momentum[2], torch.zeros(2))

    model(points, torch.ones((3, 1))).sum().backward()
    optimizer.step()
    assert torch.isfinite(model.sites).all()


def test_spectral_basis_whitening_rejects_rank_deficiency() -> None:
    with pytest.raises(ValueError, match="rank deficient"):
        spectral_basis_whitening_transform(
            torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64)
        )


def test_transformed_spectral_basis_is_shared_by_smooth_and_exact_rendering() -> None:
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    model = SpectralPowerFoamField(
        domain,
        sites=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        radii=torch.tensor([10.0], dtype=torch.float64),
        cell_coefficients=torch.tensor([[2.0, -0.5]], dtype=torch.float64),
    )
    model.set_spectral_basis_transform(
        torch.tensor([[0.5, 0.25], [-0.2, 2.0]], dtype=torch.float64)
    )
    wavelength = torch.tensor(2.4, dtype=torch.float64)
    smooth = model(
        torch.tensor(10.0, dtype=torch.float64),
        torch.tensor(20.0, dtype=torch.float64),
        wavelength,
    )
    expected = torch.dot(
        model.spectral_basis(wavelength),
        model.cell_coefficients[0],
    )

    torch.testing.assert_close(smooth, expected)


def test_joint_spectral_affine_inverse_recovers_heterogeneous_observations() -> None:
    sites = torch.tensor([[-0.5, 0.0], [0.5, 0.0]], dtype=torch.float64)
    radii = torch.full((2,), 2.0, dtype=torch.float64)
    left_right_pixels = torch.tensor(
        [
            [[-1.0, -0.5], [0.0, -0.5], [0.0, 0.5], [-1.0, 0.5]],
            [[0.0, -0.5], [1.0, -0.5], [1.0, 0.5], [0.0, 0.5]],
        ],
        dtype=torch.float64,
    )
    bottom_top_pixels = torch.tensor(
        [
            [[-1.0, -0.5], [1.0, -0.5], [1.0, 0.0], [-1.0, 0.0]],
            [[-1.0, 0.0], [1.0, 0.0], [1.0, 0.5], [-1.0, 0.5]],
        ],
        dtype=torch.float64,
    )
    operators = [
        bounded_power_pixel_operator(left_right_pixels, sites, radii),
        bounded_power_pixel_operator(bottom_top_pixels, sites, radii),
        bounded_power_pixel_operator(left_right_pixels, sites, radii),
        bounded_power_pixel_operator(bottom_top_pixels, sites, radii),
    ]
    response_basis = torch.tensor(
        [[1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    truth_coefficients = torch.tensor(
        [[2.0, 0.5], [3.0, -0.25]], dtype=torch.float64
    )
    truth_gradients = torch.tensor(
        [
            [[0.4, -0.2], [0.1, 0.05]],
            [[-0.3, 0.25], [0.05, -0.1]],
        ],
        dtype=torch.float64,
    )
    values = []
    for operator, basis in zip(operators, response_basis, strict=True):
        values.append(
            operator.matvec_affine(
                truth_coefficients @ basis,
                torch.einsum("ckd,k->cd", truth_gradients, basis),
            )
        )

    coefficients, gradients = solve_spectral_power_pixel_affine_least_squares(
        operators,
        response_basis,
        values,
        ridge=1.0e-10,
        max_iterations=200,
        tolerance=1.0e-10,
    )

    for operator, basis, expected in zip(
        operators, response_basis, values, strict=True
    ):
        actual = operator.matvec_affine(
            coefficients @ basis,
            torch.einsum("ckd,k->cd", gradients, basis),
        )
        torch.testing.assert_close(actual, expected, atol=2.0e-7, rtol=2.0e-7)


def test_joint_spectral_affine_inverse_observation_batching_is_exact() -> None:
    sites = torch.tensor([[-0.5, 0.0], [0.5, 0.0]], dtype=torch.float64)
    radii = torch.full((2,), 2.0, dtype=torch.float64)
    pixels = torch.tensor(
        [
            [[-1.0, -0.5], [0.0, -0.5], [0.0, 0.5], [-1.0, 0.5]],
            [[0.0, -0.5], [1.0, -0.5], [1.0, 0.5], [0.0, 0.5]],
        ],
        dtype=torch.float64,
    )
    operators = [bounded_power_pixel_operator(pixels, sites, radii) for _ in range(3)]
    response_basis = torch.tensor(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float64
    )
    truth_coefficients = torch.tensor(
        [[1.0, 0.25], [2.0, -0.5]], dtype=torch.float64
    )
    truth_gradients = torch.tensor(
        [[[0.1, 0.0], [0.0, 0.1]], [[-0.1, 0.0], [0.0, -0.1]]],
        dtype=torch.float64,
    )
    values = [
        operator.matvec_affine(
            truth_coefficients @ basis,
            torch.einsum("ckd,k->cd", truth_gradients, basis),
        )
        for operator, basis in zip(operators, response_basis, strict=True)
    ]

    expected = solve_spectral_power_pixel_affine_least_squares(
        operators,
        response_basis,
        values,
        ridge=1.0e-8,
        max_iterations=100,
        tolerance=1.0e-10,
    )
    diagnostics: dict[str, float] = {}
    actual = solve_spectral_power_pixel_affine_least_squares(
        operators,
        response_basis,
        values,
        ridge=1.0e-8,
        max_iterations=100,
        tolerance=1.0e-10,
        solve_device="cpu",
        observation_batch_size=1,
        diagnostics=diagnostics,
    )
    auto_batch_diagnostics: dict[str, float] = {}
    auto_batch = solve_spectral_power_pixel_affine_least_squares(
        operators,
        response_basis,
        values,
        ridge=1.0e-8,
        max_iterations=100,
        tolerance=1.0e-10,
        observation_batch_size="auto",
        diagnostics=auto_batch_diagnostics,
    )
    block_diagnostics: dict[str, float] = {}
    block = solve_spectral_power_pixel_affine_least_squares(
        operators,
        response_basis,
        values,
        ridge=1.0e-8,
        max_iterations=100,
        tolerance=1.0e-10,
        observation_batch_size=1,
        preconditioner="cell_block",
        diagnostics=block_diagnostics,
    )
    auto_diagnostics: dict[str, float] = {}
    auto = solve_spectral_power_pixel_affine_least_squares(
        operators,
        response_basis,
        values,
        ridge=1.0e-8,
        max_iterations=100,
        tolerance=1.0e-10,
        preconditioner="auto",
        diagnostics=auto_diagnostics,
    )
    fallback_diagnostics: dict[str, float] = {}
    fallback = solve_spectral_power_pixel_affine_least_squares(
        operators,
        response_basis,
        values,
        ridge=1.0e-8,
        max_iterations=100,
        tolerance=1.0e-10,
        preconditioner="auto",
        preconditioner_max_bytes=1,
        diagnostics=fallback_diagnostics,
    )
    ill_conditioned_basis = torch.tensor(
        [[1.0, -1.0], [1.0, -0.9], [1.0, -0.8]], dtype=torch.float64
    )
    ill_conditioned_values = [
        operator.matvec_affine(
            truth_coefficients @ basis,
            torch.einsum("ckd,k->cd", truth_gradients, basis),
        )
        for operator, basis in zip(
            operators, ill_conditioned_basis, strict=True
        )
    ]
    ill_diagnostics: dict[str, float] = {}
    solve_spectral_power_pixel_affine_least_squares(
        operators,
        ill_conditioned_basis,
        ill_conditioned_values,
        ridge=1.0e-8,
        max_iterations=100,
        tolerance=1.0e-10,
        preconditioner="auto",
        diagnostics=ill_diagnostics,
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(auto_batch[0], expected[0])
    torch.testing.assert_close(auto_batch[1], expected[1])
    torch.testing.assert_close(block[0], expected[0])
    torch.testing.assert_close(block[1], expected[1])
    torch.testing.assert_close(auto[0], expected[0])
    torch.testing.assert_close(auto[1], expected[1])
    torch.testing.assert_close(fallback[0], expected[0])
    torch.testing.assert_close(fallback[1], expected[1])
    moved = operators[0].to("cpu")
    assert moved is operators[0]
    assert moved.storage_bytes > 0
    assert diagnostics["observation_batch_size"] == 1.0
    assert auto_batch_diagnostics["observation_batch_auto"] == 1.0
    assert 1.0 <= auto_batch_diagnostics["observation_batch_size"] <= 3.0
    assert auto_batch_diagnostics["max_observation_transfer_bytes"] > 0.0
    assert diagnostics["operator_storage_bytes"] == 3.0 * moved.storage_bytes
    assert diagnostics["peak_operator_batch_bytes"] == moved.storage_bytes
    assert diagnostics["converged"] == 1.0
    assert diagnostics["relative_residual"] <= 1.0e-10
    assert diagnostics["initial_relative_residual"] > diagnostics[
        "relative_residual"
    ]
    assert diagnostics["normal_operator_applications"] == diagnostics[
        "iterations"
    ] + 1.0
    assert diagnostics["preconditioner_setup_seconds"] >= 0.0
    assert diagnostics["rhs_seconds"] >= 0.0
    assert diagnostics["initial_residual_seconds"] >= 0.0
    assert diagnostics["pcg_seconds"] >= 0.0
    assert block_diagnostics["preconditioner_cell_block"] == 1.0
    assert block_diagnostics["preconditioner_block_fallback_cells"] == 0.0
    assert block_diagnostics["preconditioner_storage_bytes"] > diagnostics[
        "preconditioner_storage_bytes"
    ]
    assert auto_diagnostics["preconditioner_auto"] == 1.0
    assert auto_diagnostics["preconditioner_cell_block"] == 0.0
    assert auto_diagnostics["preconditioner_response_condition"] < (
        auto_diagnostics["preconditioner_auto_condition_threshold"]
    )
    assert auto_diagnostics["preconditioner_block_construction_peak_bytes"] <= (
        auto_diagnostics["preconditioner_auto_budget_bytes"]
    )
    assert fallback_diagnostics["preconditioner_cell_block"] == 0.0
    assert fallback_diagnostics["preconditioner_auto_budget_bytes"] == 1.0
    assert ill_diagnostics["preconditioner_response_condition"] > ill_diagnostics[
        "preconditioner_auto_condition_threshold"
    ]
    assert ill_diagnostics["preconditioner_cell_block"] == 1.0


def test_joint_spectral_affine_refinement_centers_ridge_on_initial_model() -> None:
    sites = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    radii = torch.tensor([2.0], dtype=torch.float64)
    pixels = torch.tensor(
        [[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]],
        dtype=torch.float64,
    )
    operator = bounded_power_pixel_operator(pixels, sites, radii)
    basis = torch.ones((1, 1), dtype=torch.float64)
    initial_coefficients = torch.tensor([[5.0]], dtype=torch.float64)
    initial_gradients = torch.zeros((1, 1, 2), dtype=torch.float64)
    values = [
        operator.matvec_affine(
            initial_coefficients[:, 0], initial_coefficients.new_zeros((1, 2))
        )
    ]
    diagnostics: dict[str, float] = {}

    solved, gradients = solve_spectral_power_pixel_affine_least_squares(
        [operator],
        basis,
        values,
        ridge=100.0,
        max_iterations=20,
        tolerance=1.0e-12,
        initial_coefficients=initial_coefficients,
        initial_gradient_coefficients=initial_gradients,
        diagnostics=diagnostics,
    )

    torch.testing.assert_close(solved, initial_coefficients)
    torch.testing.assert_close(gradients, initial_gradients)
    assert diagnostics["ridge_centered_on_initial"] == 1.0


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


def test_spectral_power_foam_affine_basis_uses_exact_pixel_first_moments() -> None:
    model = SpectralPowerFoam(
        torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        torch.tensor([10.0], dtype=torch.float64),
        torch.tensor([[2.0]], dtype=torch.float64),
        cell_gradient_coefficients=torch.tensor([[[3.0, -4.0]]], dtype=torch.float64),
    )
    pixel = torch.tensor(
        [[[0.0, -1.0], [2.0, -1.0], [2.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float64,
    )
    operator = bounded_power_pixel_operator(pixel, model.sites.detach(), model.radii)

    rendered = model.render_exact(
        operator,
        torch.ones((1, 1), dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
    )

    torch.testing.assert_close(rendered, torch.tensor([20.0], dtype=torch.float64))
    rendered.sum().backward()
    assert model.cell_gradient_coefficients is not None
    torch.testing.assert_close(
        model.cell_gradient_coefficients.grad,
        torch.tensor([[[4.0, 0.0]]], dtype=torch.float64),
        atol=1.0e-14,
        rtol=0,
    )


def test_spectral_power_foam_compact_fixed_geometry_matches_dense_support() -> None:
    model = _model()
    model.sites.requires_grad_(False)
    model.raw_radii.requires_grad_(False)
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
        compact_fixed_geometry=True,
    )
    weights = soft_bounded_power_weights(
        points,
        model.sites,
        model.radii,
        temperature=7.0,
    )
    expected = torch.sum(
        weights * torch.einsum("pk,ck->pc", basis, model.cell_coefficients),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert model.cell_coefficients.grad is not None

    model.sites.requires_grad_(True)
    with pytest.raises(ValueError, match="requires frozen"):
        model(points, basis, compact_fixed_geometry=True)


def test_solve_spectral_cell_coefficients_reconstructs_observed_bands() -> None:
    basis = torch.tensor(
        [[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    values = torch.tensor(
        [[2.0, 5.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    weights = torch.tensor(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    coefficients = solve_spectral_cell_coefficients(
        basis,
        values,
        weights,
        ridge=1.0e-10,
    )
    reconstructed = basis @ coefficients.mT

    torch.testing.assert_close(
        reconstructed[weights > 0],
        values[weights > 0],
        atol=1.0e-8,
        rtol=0,
    )
    torch.testing.assert_close(coefficients[2], torch.zeros(4, dtype=torch.float64))
    with pytest.raises(ValueError, match="non-negative"):
        solve_spectral_cell_coefficients(basis, values, -weights)


def test_solve_power_pixel_least_squares_recovers_observed_cells() -> None:
    sites = torch.tensor([[-0.5, 0.0], [0.5, 0.0]], dtype=torch.float64)
    radii = torch.full((2,), 0.75, dtype=torch.float64)
    pixels = torch.tensor(
        [
            [[-1.0, -0.5], [0.0, -0.5], [0.0, 0.5], [-1.0, 0.5]],
            [[0.0, -0.5], [1.0, -0.5], [1.0, 0.5], [0.0, 0.5]],
        ],
        dtype=torch.float64,
    )
    operator = bounded_power_pixel_operator(pixels, sites, radii)
    expected = torch.tensor([2.0, 4.0], dtype=torch.float64)
    pixel_values = operator.matvec(expected)

    actual = solve_power_pixel_least_squares(
        operator,
        pixel_values,
        ridge=1.0e-12,
        tolerance=1.0e-10,
    )
    partial = solve_power_pixel_least_squares(
        operator,
        pixel_values,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        ridge=1.0e-12,
        tolerance=1.0e-10,
    )
    transformed = solve_power_pixel_least_squares(
        operator,
        2.0 * pixel_values,
        ridge=1.0e-12,
        tolerance=1.0e-10,
        pixel_matvec=lambda values: 2.0 * values,
        pixel_rmatvec=lambda values: 2.0 * values,
    )
    zero = solve_power_pixel_least_squares(
        operator,
        torch.zeros_like(pixel_values),
        initial=torch.ones_like(expected),
        ridge=1.0e-12,
        tolerance=1.0e-10,
    )

    torch.testing.assert_close(actual, expected, atol=1.0e-9, rtol=0)
    torch.testing.assert_close(transformed, expected, atol=1.0e-9, rtol=0)
    torch.testing.assert_close(partial[0], expected[0], atol=1.0e-9, rtol=0)
    assert partial[1] == 0
    torch.testing.assert_close(zero, torch.zeros_like(expected), atol=1.0e-9, rtol=0)
    with pytest.raises(ValueError, match="non-negative"):
        solve_power_pixel_least_squares(
            operator,
            pixel_values,
            -torch.ones(2, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="supplied together"):
        solve_power_pixel_least_squares(
            operator,
            pixel_values,
            pixel_matvec=lambda values: values,
        )


def test_solve_power_pixel_affine_least_squares_recovers_linear_field() -> None:
    site = torch.zeros((1, 2), dtype=torch.float64)
    radius = torch.full((1,), 10.0, dtype=torch.float64)
    pixels = torch.stack(
        [
            torch.tensor(
                [[x, y], [x + 1.0, y], [x + 1.0, y + 1.0], [x, y + 1.0]],
                dtype=torch.float64,
            )
            for y in (-1.0, 0.0)
            for x in (-1.0, 0.0)
        ]
    )
    operator = bounded_power_pixel_operator(pixels, site, radius)
    expected_value = torch.tensor([2.0], dtype=torch.float64)
    expected_gradient = torch.tensor([[3.0, -4.0]], dtype=torch.float64)
    pixel_values = operator.matvec_affine(expected_value, expected_gradient)

    actual_value, actual_gradient = solve_power_pixel_affine_least_squares(
        operator,
        pixel_values,
        ridge=1.0e-12,
        max_iterations=20,
        tolerance=1.0e-10,
    )

    torch.testing.assert_close(actual_value, expected_value, atol=1.0e-8, rtol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1.0e-8, rtol=0)


def test_affine_least_squares_is_invariant_to_spatial_units() -> None:
    def solve(scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        site = torch.zeros((1, 2), dtype=torch.float64)
        radius = torch.full((1,), 10.0 * scale, dtype=torch.float64)
        pixels = scale * torch.stack(
            [
                torch.tensor(
                    [[x, y], [x + 1.0, y], [x + 1.0, y + 1.0], [x, y + 1.0]],
                    dtype=torch.float64,
                )
                for y in (-1.0, 0.0)
                for x in (-1.0, 0.0)
            ]
        )
        operator = bounded_power_pixel_operator(pixels, site, radius)
        values = torch.tensor([2.0], dtype=torch.float64)
        gradients = torch.tensor([[3.0 / scale, -4.0 / scale]], dtype=torch.float64)
        integrated = operator.matvec_affine(values, gradients)
        return solve_power_pixel_affine_least_squares(
            operator,
            integrated,
            ridge=1.0e-3,
            max_iterations=20,
            tolerance=1.0e-12,
        )

    reference_value, reference_gradient = solve(1.0)
    scaled_value, scaled_gradient = solve(100.0)

    torch.testing.assert_close(scaled_value, reference_value, atol=1.0e-10, rtol=0)
    torch.testing.assert_close(
        100.0 * scaled_gradient,
        reference_gradient,
        atol=1.0e-10,
        rtol=0,
    )


def test_affine_least_squares_damps_weakly_observed_cell_variation() -> None:
    site = torch.zeros((1, 2), dtype=torch.float64)
    radius = torch.full((1,), 10.0, dtype=torch.float64)
    pixel = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float64,
    )
    operator = bounded_power_pixel_operator(pixel, site, radius)

    value, gradient = solve_power_pixel_affine_least_squares(
        operator,
        torch.ones(1, dtype=torch.float64),
        ridge=1.0e-3,
        max_iterations=20,
        tolerance=1.0e-12,
    )

    cell_scale = torch.sqrt(operator.cell_areas[0])
    assert torch.linalg.vector_norm(cell_scale * gradient[0]) < torch.abs(value[0])


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


def test_spectral_power_foam_field_reuses_smooth_and_exact_spectral_basis() -> None:
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    model = make_field_model(
        "spectral_power_foam",
        domain,
        sites=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        radii=torch.tensor([2.0], dtype=torch.float64),
        cell_coefficients=torch.tensor([[2.0, 0.5, -0.25]], dtype=torch.float64),
        temperature_start=4.0,
        temperature_end=16.0,
    )
    assert isinstance(model, SpectralPowerFoamField)
    model.set_curriculum_progress(0.5)

    wavelength = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    ra = torch.full_like(wavelength, 10.0)
    dec = torch.full_like(wavelength, 20.0)
    smooth = model(ra, dec, wavelength)
    expected = model.foam(
        torch.zeros((3, 2), dtype=torch.float64),
        model.spectral_basis(wavelength),
        power_temperature=8.0,
        support_temperature=8.0,
    )
    torch.testing.assert_close(smooth, expected)
    broadcast = model(
        torch.full((2, 3), 10.0, dtype=torch.float64),
        torch.full((2, 3), 20.0, dtype=torch.float64),
        torch.tensor([[1.0], [3.0]], dtype=torch.float64),
    )
    assert broadcast.shape == (2, 3)

    pixels = torch.tensor(
        [[[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]],
        dtype=torch.float64,
    )
    response = torch.tensor([[0.25, 0.5, 0.25]], dtype=torch.float64)
    exact = model.render_exact(pixels, wavelength, response)
    operator = bounded_power_pixel_operator(
        pixels,
        model.sites.detach(),
        model.radii.detach(),
    )
    expected_exact = model.foam.render_exact(
        operator,
        model.spectral_basis(wavelength),
        response,
    )
    torch.testing.assert_close(exact, expected_exact)


def test_foam_initialized_inr_is_positive_exact_and_zero_initialized() -> None:
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    foam = SpectralPowerFoamField(
        domain,
        sites=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        radii=torch.tensor([2.0], dtype=torch.float64),
        cell_coefficients=torch.tensor([[2.0, 0.0]], dtype=torch.float64),
        temperature_start=4.0,
        temperature_end=4.0,
    )
    model = make_field_model(
        "foam_initialized_inr",
        domain,
        foam=foam,
        hidden_dim=8,
        hidden_layers=2,
    )
    assert isinstance(model, FoamInitializedINRField)
    ra = torch.tensor([9.99, 10.0, 10.01], dtype=torch.float64)
    dec = torch.tensor([20.01, 20.0, 19.99], dtype=torch.float64)
    wavelength = torch.tensor([1.5, 2.0, 2.5], dtype=torch.float64)
    expected = foam(ra, dec, wavelength).clamp_min(model.intensity_floor)
    actual = model(ra, dec, wavelength)

    torch.testing.assert_close(actual, expected.to(dtype=actual.dtype))
    assert bool((actual > 0).all())
    assert not any(parameter.requires_grad for parameter in model.reference_foam.parameters())
    actual.sum().backward()
    assert model.residual_output.weight.grad is not None
    assert not torch.all(model.residual_output.weight.grad == 0)


def test_diffuse_foam_hybrid_is_combined_only_and_supports_staged_training() -> None:
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    foam = SpectralPowerFoamField(
        domain,
        sites=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        radii=torch.tensor([0.5], dtype=torch.float32),
        cell_coefficients=torch.tensor([[-2.0, 0.25]], dtype=torch.float32),
        log_intensity_spectra=True,
    )
    model = make_field_model(
        "diffuse_foam_hybrid",
        domain,
        foam=foam,
        diffuse_hidden_dim=8,
        diffuse_hidden_layers=1,
        diffuse_omega0=2.0,
        training_stage="foam",
    )
    assert isinstance(model, DiffuseFoamHybridField)
    assert model.component_identifiability == "combined_only"
    assert model.training_stage == "foam"
    assert not any(parameter.requires_grad for parameter in model.diffuse_field.parameters())
    assert all(parameter.requires_grad for parameter in model.localized_foam.parameters())

    ra = torch.tensor([9.99, 10.0, 10.01], dtype=torch.float32)
    dec = torch.tensor([20.01, 20.0, 19.99], dtype=torch.float32)
    wavelength = torch.tensor([1.5, 2.0, 2.5], dtype=torch.float32)
    diffuse, localized = model.diagnostic_components(ra, dec, wavelength)
    torch.testing.assert_close(model(ra, dec, wavelength), diffuse + localized)
    assert bool((model(ra, dec, wavelength) >= 0.0).all())

    model.set_training_stage("joint")
    assert all(parameter.requires_grad for parameter in model.parameters())
    regularization = model.component_regularization(
        ra,
        dec,
        wavelength,
        foam_sparsity_weight=0.1,
        correlation_weight=0.2,
    )
    assert regularization.ndim == 0
    assert torch.isfinite(regularization)
    (model(ra, dec, wavelength).sum() + regularization).backward()
    assert model.diffuse_field.output.weight.grad is not None
    assert model.localized_foam.foam.cell_coefficients.grad is not None

    model._training_stage_code.fill_(9)
    with pytest.raises(ValueError, match="stored hybrid training stage"):
        _ = model.training_stage


def test_diffuse_foam_hybrid_staged_fit_uses_joint_regularization() -> None:
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    foam = SpectralPowerFoamField(
        domain,
        sites=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        radii=torch.tensor([2.0], dtype=torch.float32),
        cell_coefficients=torch.tensor([[-2.0]], dtype=torch.float32),
        log_intensity_spectra=True,
    )
    model = make_field_model(
        "diffuse_foam_hybrid",
        domain,
        foam=foam,
        diffuse_hidden_dim=6,
        diffuse_hidden_layers=1,
    )
    assert isinstance(model, DiffuseFoamHybridField)
    observation = FieldObservation(
        image=torch.full((4, 4), 0.3, dtype=torch.float32),
        header=make_tangent_plane_header(
            4,
            4,
            crval_ra_deg=10.0,
            crval_dec_deg=20.0,
            cd11=-0.01,
            cd22=0.01,
        ),
        weight=torch.ones((4, 4), dtype=torch.float32),
        wavelength=2.0,
    )

    result = fit_diffuse_foam_hybrid(
        [observation],
        model,
        diffuse_steps=1,
        foam_steps=1,
        joint_steps=1,
        joint_foam_sparsity_weight=0.01,
        joint_correlation_weight=0.01,
        lr=1.0e-3,
        num_spectral_samples=1,
        device="cpu",
    )

    assert result.completed_steps == 3
    assert model.training_stage == "joint"
    assert result.diagnostics is not None
    assert [row["stage"] for row in result.diagnostics["hybrid_stages"]] == [
        "diffuse",
        "foam",
        "joint",
    ]


def test_diffuse_foam_hybrid_exact_render_keeps_foam_cells_hard() -> None:
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    foam = SpectralPowerFoamField(
        domain,
        sites=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        radii=torch.tensor([0.12], dtype=torch.float64),
        cell_coefficients=torch.tensor([[-1.0]], dtype=torch.float64),
        log_intensity_spectra=True,
    )
    model = make_field_model(
        "diffuse_foam_hybrid",
        domain,
        foam=foam,
        diffuse_hidden_dim=6,
        diffuse_hidden_layers=1,
    )
    assert isinstance(model, DiffuseFoamHybridField)
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=make_tangent_plane_header(
            4,
            4,
            crval_ra_deg=10.0,
            crval_dec_deg=20.0,
            cd11=-0.01,
            cd22=0.01,
        ),
        wavelength=2.0,
    )
    empty_bundle = ObservationOperatorBundle()

    exact = render_diffuse_foam_hybrid_observation_exact(
        model,
        observation,
        num_spectral_samples=1,
        operator_bundle=empty_bundle,
    )
    expected = render_field_observation(
        model.diffuse_field,
        observation,
        num_spectral_samples=1,
        operator_bundle=empty_bundle,
    ) + render_spectral_power_foam_observation_exact(
        model.localized_foam,
        observation,
        num_spectral_samples=1,
        operator_bundle=empty_bundle,
    )

    torch.testing.assert_close(exact, expected)
    assert bool(torch.isfinite(exact).all())


def test_log_intensity_foam_is_positive_and_exact_render_matches_points() -> None:
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    foam = SpectralPowerFoamField(
        domain,
        sites=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        radii=torch.tensor([2.0], dtype=torch.float64),
        cell_coefficients=torch.tensor([[-2.0, 0.5]], dtype=torch.float64),
        temperature_start=4.0,
        temperature_end=4.0,
        compact_fixed_geometry=True,
        log_intensity_spectra=True,
    )
    foam.foam.sites.requires_grad_(False)
    foam.foam.raw_radii.requires_grad_(False)
    wavelengths = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    ra = torch.full_like(wavelengths, 10.0)
    dec = torch.full_like(wavelengths, 20.0)
    point_values = foam(ra, dec, wavelengths)
    expected = torch.exp(torch.tensor([-2.5, -2.0, -1.5], dtype=torch.float64))
    torch.testing.assert_close(point_values, expected)
    assert bool((point_values > 0.0).all())

    pixels = torch.tensor(
        [[[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]],
        dtype=torch.float64,
    )
    response = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    exact = foam.render_exact(pixels, wavelengths, response)
    torch.testing.assert_close(exact, torch.sum(response * expected).view(1))


def test_log_intensity_foam_rejects_nonexact_affine_spatial_spectra() -> None:
    with pytest.raises(ValueError, match="does not yet support exact affine"):
        SpectralPowerFoam(
            torch.tensor([[0.0, 0.0]], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.float64),
            cell_gradient_coefficients=torch.zeros((1, 1, 2), dtype=torch.float64),
            log_intensity_spectra=True,
        )


def test_spectral_power_foam_field_uses_common_bandpass_psf_render_path() -> None:
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    model = SpectralPowerFoamField(
        domain,
        sites=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        radii=torch.tensor([1.0], dtype=torch.float32),
        cell_coefficients=torch.tensor([[1.0, 0.2]], dtype=torch.float32),
    )
    observation = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=make_tangent_plane_header(
            4,
            4,
            crval_ra_deg=10.0,
            crval_dec_deg=20.0,
        ),
        weight=torch.ones((4, 4), dtype=torch.float32),
        mask=torch.ones((4, 4), dtype=torch.bool),
        spectral_min=1.0,
        spectral_max=3.0,
        transmission_center=2.0,
        transmission_sigma=0.4,
        psf_sigma_pix=0.5,
        band_name="TEST",
    )

    rendered = render_field_observation(
        model,
        observation,
        num_spectral_samples=5,
    )
    assert rendered.shape == observation.image.shape
    assert bool(torch.isfinite(rendered).all())
    rendered.sum().backward()
    assert model.cell_coefficients.grad is not None
    assert bool(torch.isfinite(model.cell_coefficients.grad).all())


def test_spectral_power_foam_field_hard_renders_target_pixel_footprints() -> None:
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    model = SpectralPowerFoamField(
        domain,
        sites=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        radii=torch.tensor([10.0], dtype=torch.float64),
        cell_coefficients=torch.tensor([[2.0, 0.0]], dtype=torch.float64),
    )
    observation = FieldObservation(
        image=torch.zeros((3, 4), dtype=torch.float32),
        header=make_tangent_plane_header(
            3,
            4,
            crval_ra_deg=10.0,
            crval_dec_deg=20.0,
            cd11=-0.001,
            cd22=0.001,
        ),
        spectral_min=1.0,
        spectral_max=3.0,
        transmission_center=2.0,
        transmission_sigma=0.4,
        background_offset=0.25,
        band_name="TEST",
    )

    rendered = render_spectral_power_foam_observation_exact(
        model,
        observation,
        num_spectral_samples=5,
    )

    torch.testing.assert_close(rendered, torch.full((3, 4), 2.25))


def test_spectral_power_foam_hard_render_supports_conditioned_operators() -> None:
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    model = SpectralPowerFoamField(
        domain,
        sites=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        radii=torch.tensor([10.0], dtype=torch.float32),
        cell_coefficients=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
    )
    encoder = AdaptiveSpatioSpectralTemporalEncoder(
        spatial_center=(3.5, 3.5),
        spatial_scale=4.0,
        spatial_length_bounds=(2.0, 16.0),
        wavelength_center=2.0,
        spectral_scale=1.0,
        spectral_length_bounds=(0.2, 4.0),
        time_center_mjd=60000.0,
        temporal_scale_days=10.0,
        temporal_length_bounds_days=(0.5, 40.0),
        spatial_bands=1,
        spectral_bands=1,
        temporal_bands=1,
    )
    psf = ConditionedKernelPSF(
        encoder, n_basis=2, kernel_size=3, init_fwhm_pix=1.5, hidden_dim=8
    )
    background = ConditionedBackgroundField(encoder, hidden_dim=8, hidden_layers=1)
    with torch.no_grad():
        background.network[-1].bias.fill_(0.25)
    bundle = ObservationOperatorBundle(
        psf=LearnedPSFOp(
            psf,
            pixel_response_included=False,
            pixel_response_provenance="analytic Gaussian kernel",
        ),
        background=LearnedBackgroundOp(background, coordinate_frame="detector"),
    )
    observation = FieldObservation(
        image=torch.zeros((8, 8), dtype=torch.float32),
        header=make_tangent_plane_header(
            8,
            8,
            crval_ra_deg=10.0,
            crval_dec_deg=20.0,
            cd11=-0.001,
            cd22=0.001,
        ),
        spectral_min=1.0,
        spectral_max=3.0,
        transmission_center=2.0,
        transmission_sigma=0.4,
    )

    rendered = render_spectral_power_foam_observation_exact(
        model,
        observation,
        num_spectral_samples=3,
        forward_context=ForwardContext(time_mjd=60005.0),
        operator_bundle=bundle,
        spatial_tile_hw=(4, 4),
    )

    assert rendered.shape == (8, 8)
    assert bool(torch.isfinite(rendered).all())
    torch.testing.assert_close(rendered[4, 4], torch.tensor(2.25), atol=1e-5, rtol=1e-5)
    rendered.square().sum().backward()
    assert model.cell_coefficients.grad is not None
    assert psf.kernel_basis.raw_kernels.grad is not None
    assert background.network[-1].bias.grad is not None


def test_spectral_power_foam_fits_multiple_observations_then_hard_renders() -> None:
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        spatial_scale_deg=0.1,
        wavelength_center=2.0,
        wavelength_scale=1.0,
    )
    model = SpectralPowerFoamField(
        domain,
        sites=torch.tensor(
            [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]]
        ),
        radii=torch.full((4,), 1.0),
        cell_coefficients=torch.zeros((4, 2)),
    )
    header = make_tangent_plane_header(
        4,
        4,
        crval_ra_deg=10.0,
        crval_dec_deg=20.0,
        cd11=-0.001,
        cd22=0.001,
    )
    observations = [
        FieldObservation(
            image=torch.full((4, 4), level),
            header=dict(header),
            weight=torch.ones((4, 4)),
            mask=torch.ones((4, 4), dtype=torch.bool),
            spectral_min=center - 0.2,
            spectral_max=center + 0.2,
            transmission_center=center,
            transmission_sigma=0.1,
            band_name=name,
        )
        for level, center, name in ((1.0, 1.5, "BLUE"), (2.0, 2.5, "RED"))
    ]
    initial_coefficients = model.cell_coefficients.detach().clone()

    fit = fit_field_model(
        observations,
        model,
        steps=2,
        lr=1.0e-2,
        num_spectral_samples=3,
        loss_kind="mse",
        loss_reduction="global",
        observations_per_step=2,
        pixel_samples_per_observation=8,
        curriculum="feature_scale_ramp",
    )
    rendered = render_spectral_power_foam_observation_exact(
        fit.model,
        observations[0],
        num_spectral_samples=3,
    )

    assert len(fit.loss_history) == 2
    assert bool(torch.isfinite(rendered).all())
    assert not torch.equal(fit.model.cell_coefficients, initial_coefficients)
