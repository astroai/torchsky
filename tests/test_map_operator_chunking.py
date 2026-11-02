from __future__ import annotations

import pytest
import torch

from torchsky.map.solvers import (
    BoundObservationOperatorSet,
    build_state_diagonal_operator,
    build_state_matrix_operator,
    make_diagonal_preconditioner,
)


def _operator_set(*, exposure_chunk_size: int) -> BoundObservationOperatorSet:
    state = torch.zeros((2, 3), dtype=torch.float64)
    operators = tuple(
        build_state_diagonal_operator(
            state,
            torch.full_like(state, float(index + 1)),
            apply_noise=lambda tensor, scale=float(index + 2): tensor * scale,
        )
        for index in range(5)
    )
    return BoundObservationOperatorSet(
        operators,
        exposure_chunk_size=exposure_chunk_size,
    )


def test_multi_exposure_apply_yields_bounded_ordered_chunks() -> None:
    operator_set = _operator_set(exposure_chunk_size=2)
    state = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    chunks = tuple(operator_set.iter_apply_chunks(state))

    assert tuple(len(chunk) for chunk in chunks) == (2, 2, 1)
    assert all(
        torch.equal(prediction, expected)
        for prediction, expected in zip(
            (prediction for chunk in chunks for prediction in chunk),
            operator_set.apply(state),
            strict=True,
        )
    )


@pytest.mark.parametrize("exposure_chunk_size", [1, 2, 5])
def test_chunked_normal_action_matches_eager_exactly(
    exposure_chunk_size: int,
) -> None:
    operator_set = _operator_set(exposure_chunk_size=exposure_chunk_size)
    state = torch.linspace(-1.0, 2.0, 6, dtype=torch.float64).reshape(2, 3)
    predictions = operator_set.apply(state)
    eager = operator_set.adjoint(operator_set.apply_noise_model(predictions))

    chunked = operator_set.normal_operator_action(state)

    assert torch.equal(chunked, eager)


def test_chunk_size_preserves_exact_adjoint_and_deterministic_accumulation() -> None:
    state = torch.linspace(-1.0, 2.0, 6, dtype=torch.float64).reshape(2, 3)
    duals = tuple(
        torch.linspace(float(index), float(index + 1), 6, dtype=torch.float64).reshape(
            2, 3
        )
        for index in range(5)
    )
    reference: torch.Tensor | None = None

    for chunk_size in (1, 2, 3, 5):
        operator_set = _operator_set(exposure_chunk_size=chunk_size)
        predictions = operator_set.apply(state)
        lhs = sum(
            torch.sum(prediction * dual)
            for prediction, dual in zip(predictions, duals, strict=True)
        )
        rhs = torch.sum(state * operator_set.adjoint(duals))
        action_a = operator_set.normal_operator_action(state)
        action_b = operator_set.normal_operator_action(state)

        torch.testing.assert_close(lhs, rhs, rtol=0.0, atol=0.0)
        assert torch.equal(action_a, action_b)
        if reference is None:
            reference = action_a
        else:
            assert torch.equal(action_a, reference)


def test_chunked_cg_matches_across_exposure_chunk_sizes() -> None:
    truth = torch.linspace(-0.5, 0.75, 6, dtype=torch.float64).reshape(2, 3)
    reference_solution: torch.Tensor | None = None
    reference_history: tuple[float, ...] | None = None

    for chunk_size in (1, 2, 5):
        operator_set = _operator_set(exposure_chunk_size=chunk_size)
        data = operator_set.apply(truth)
        result = operator_set.solve_normal_equation_cg(
            data,
            initial_solution=torch.zeros_like(truth),
            iterations=4,
        )
        if reference_solution is None:
            reference_solution = result.solution
            reference_history = result.residual_norm_history
        else:
            assert torch.equal(result.solution, reference_solution)
            assert result.residual_norm_history == reference_history


def test_seeded_posterior_variance_matches_small_dense_inverse() -> None:
    state = torch.zeros(3, dtype=torch.float64)
    normal = torch.tensor(
        [[4.0, 0.20, -0.10], [0.20, 3.0, 0.15], [-0.10, 0.15, 2.0]],
        dtype=torch.float64,
    )
    forward = torch.linalg.cholesky(normal).mT
    operator = build_state_matrix_operator(state, forward)
    damping = 0.25
    expected = torch.linalg.inv(normal + damping * torch.eye(3)).diagonal()
    rhs = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)

    solve = operator.solve_normal_system_cg(
        rhs,
        initial_solution=torch.zeros(3, dtype=torch.float32),
        iterations=3,
        damping=damping,
        preconditioner=make_diagonal_preconditioner(normal.diagonal() + damping),
    )

    variance = operator.estimate_posterior_variance_diagonal(
        state,
        num_probes=512,
        seed=23,
        iterations=3,
        damping=damping,
    )
    repeated = operator.estimate_posterior_variance_diagonal(
        state,
        num_probes=512,
        seed=23,
        iterations=3,
        damping=damping,
    )

    torch.testing.assert_close(
        solve.solution,
        torch.linalg.solve(normal + damping * torch.eye(3), rhs),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert solve.solution.dtype == rhs.dtype
    torch.testing.assert_close(variance, expected, rtol=0.02, atol=0.0)
    assert torch.equal(variance, repeated)
    assert torch.all(torch.isfinite(variance))
    assert torch.all(variance > 0.0)


def test_posterior_variance_is_exposure_chunk_invariant() -> None:
    state = torch.zeros((2, 3), dtype=torch.float64)
    reference: torch.Tensor | None = None

    for chunk_size in (1, 2, 5):
        operator_set = _operator_set(exposure_chunk_size=chunk_size)
        variance = operator_set.estimate_posterior_variance_diagonal(
            state,
            num_probes=8,
            seed=91,
            iterations=2,
            damping=0.1,
        )
        if reference is None:
            reference = variance
        else:
            assert torch.equal(variance, reference)


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        (0, ValueError, "positive"),
        (1.5, TypeError, "integer"),
        (True, TypeError, "integer"),
    ],
)
def test_exposure_chunk_size_fails_closed(value, error, match: str) -> None:
    with pytest.raises(error, match=match):
        BoundObservationOperatorSet(
            _operator_set(exposure_chunk_size=1).operators,
            exposure_chunk_size=value,
        )
