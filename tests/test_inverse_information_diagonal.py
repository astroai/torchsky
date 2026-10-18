"""Residual and signed Monte Carlo contracts for inverse-curvature probes."""

from __future__ import annotations

import pytest
import torch

from torchsky.map.solvers import (
    BoundObservationOperatorSet,
    _estimate_inverse_diagonal_hutchinson,
    build_state_diagonal_operator,
    build_state_matrix_operator,
)


@pytest.mark.parametrize("operator_set", [False, True])
@pytest.mark.parametrize("singular", [False, True])
def test_incomplete_or_singular_inverse_solve_fails_closed(operator_set, singular):
    state = torch.zeros(2, dtype=torch.float64)
    diagonal = torch.tensor([1.0, 0.0 if singular else 3.0], dtype=state.dtype)
    operator = build_state_diagonal_operator(state, diagonal)
    if operator_set:
        operator = BoundObservationOperatorSet((operator,))
    with pytest.raises(RuntimeError, match="residual check|non-finite"):
        operator.estimate_posterior_variance_diagonal(
            state, num_probes=1, iterations=1, rtol=1e-10, atol=0.0
        )


def test_singular_undamped_system_does_not_return_finite_confidence():
    state = torch.zeros(3, dtype=torch.float64)
    operator = build_state_diagonal_operator(state, torch.zeros_like(state))
    with pytest.raises(RuntimeError, match="residual check"):
        operator.estimate_posterior_variance_diagonal(state, num_probes=1)
    damped = operator.estimate_posterior_variance_diagonal(
        state, num_probes=1, damping=2.0, iterations=1
    )
    torch.testing.assert_close(damped, torch.full_like(state, 0.5))


def test_signed_hutchinson_estimate_is_preserved_and_exhaustive_mean_is_unbiased(
    monkeypatch,
):
    inverse = torch.tensor([[1.0, 2.0], [2.0, 5.0]], dtype=torch.float64)
    state = torch.zeros(2, dtype=inverse.dtype)
    bit_patterns = iter([[1, 0]])

    def randint(low, high, shape, *, dtype, device, generator):
        assert low == 0 and high == 2 and shape == state.shape
        return torch.tensor(next(bit_patterns), dtype=dtype, device=device)

    monkeypatch.setattr(torch, "randint", randint)

    def estimate(count):
        return _estimate_inverse_diagonal_hutchinson(
            lambda z: inverse @ z,
            state,
            num_probes=count,
            seed=0,
            normal_action=lambda x: torch.linalg.solve(inverse, x),
            rtol=1e-12,
        )

    signed = estimate(1)
    torch.testing.assert_close(signed, torch.tensor([-1.0, 3.0], dtype=state.dtype))
    assert signed[0] < 0.0
    bit_patterns = iter([[0, 0], [0, 1], [1, 0], [1, 1]])
    torch.testing.assert_close(estimate(4), inverse.diagonal())


@pytest.mark.parametrize("operator_set", [False, True])
def test_public_estimator_returns_negative_finite_probe_without_floor(
    monkeypatch, operator_set
):
    inverse = torch.tensor([[1.0, 2.0], [2.0, 5.0]], dtype=torch.float64)
    normal = torch.linalg.solve(inverse, torch.eye(2, dtype=inverse.dtype))
    state = torch.zeros(2, dtype=inverse.dtype)
    operator = build_state_matrix_operator(state, torch.linalg.cholesky(normal).T)
    if operator_set:
        operator = BoundObservationOperatorSet((operator,))

    def randint(low, high, shape, *, dtype, device, generator):
        return torch.tensor([1, 0], dtype=dtype, device=device)

    monkeypatch.setattr(torch, "randint", randint)
    result = operator.estimate_posterior_variance_diagonal(
        state, num_probes=1, iterations=2, rtol=1e-12
    )
    torch.testing.assert_close(result, torch.tensor([-1.0, 3.0], dtype=state.dtype))


def test_actual_residual_is_checked_after_solve():
    state = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="actual residual check"):
        _estimate_inverse_diagonal_hutchinson(
            lambda z: torch.zeros_like(z),
            state,
            num_probes=1,
            seed=0,
            normal_action=lambda x: x,
            rtol=0.0,
            atol=0.0,
        )


def test_absolute_residual_tolerance_has_declared_max_convention():
    state = torch.zeros(2, dtype=torch.float64)

    def estimate(atol):
        return _estimate_inverse_diagonal_hutchinson(
            lambda z: z * 0.9,
            state,
            num_probes=1,
            seed=0,
            normal_action=lambda x: x,
            rtol=0.0,
            atol=atol,
        )

    with pytest.raises(RuntimeError, match="residual check"):
        estimate(0.1)
    torch.testing.assert_close(estimate(0.15), torch.full_like(state, 0.9))


@pytest.mark.parametrize("name", ["damping", "rtol", "atol"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_nonfinite_or_negative_solve_controls_fail(name, value):
    state = torch.zeros(2, dtype=torch.float64)
    operator = build_state_diagonal_operator(state, torch.ones_like(state))
    with pytest.raises(ValueError, match=name):
        operator.estimate_posterior_variance_diagonal(
            state, num_probes=1, **{name: value}
        )


@pytest.mark.parametrize("name", ["damping", "rtol", "atol"])
@pytest.mark.parametrize("value", [True, "small", 1j])
def test_nonnumeric_or_boolean_solve_controls_fail(name, value):
    state = torch.zeros(2, dtype=torch.float64)
    operator = build_state_diagonal_operator(state, torch.ones_like(state))
    with pytest.raises(TypeError, match=name):
        operator.estimate_posterior_variance_diagonal(
            state, num_probes=1, **{name: value}
        )


@pytest.mark.parametrize("operator_set", [False, True])
@pytest.mark.parametrize(
    "iterations,error", [(True, TypeError), (1.5, TypeError), (0, ValueError)]
)
def test_invalid_inverse_probe_iterations_fail(operator_set, iterations, error):
    state = torch.zeros(2, dtype=torch.float64)
    operator = build_state_diagonal_operator(state, torch.ones_like(state))
    if operator_set:
        operator = BoundObservationOperatorSet((operator,))
    with pytest.raises(error, match="iterations"):
        operator.estimate_posterior_variance_diagonal(
            state, num_probes=1, iterations=iterations
        )


@pytest.mark.parametrize("callback_name", ["solve", "normal"])
@pytest.mark.parametrize(
    "bad_output, error, match",
    [
        (lambda x: x[:, None], ValueError, "shape"),
        (lambda x: x.float(), ValueError, "dtype and device"),
        (lambda x: torch.full_like(x, float("nan")), RuntimeError, "non-finite"),
        (lambda x: x.tolist(), TypeError, "tensor"),
    ],
)
def test_callback_shape_dtype_finiteness_and_type_fail_closed(
    callback_name, bad_output, error, match
):
    state = torch.zeros(2, dtype=torch.float64)
    kwargs = {"solve": lambda x: x, "normal_action": lambda x: x}
    kwargs["solve" if callback_name == "solve" else "normal_action"] = bad_output
    with pytest.raises(error, match=match):
        _estimate_inverse_diagonal_hutchinson(
            reference_state=state, num_probes=1, seed=0, **kwargs
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_callback_device_mismatch_fails_closed():
    state = torch.zeros(2, device="cuda")
    with pytest.raises(ValueError, match="dtype and device"):
        _estimate_inverse_diagonal_hutchinson(
            lambda z: z.cpu(), state, num_probes=1, seed=0, normal_action=lambda x: x
        )


@pytest.mark.parametrize("callback_name", ["solve", "normal_action"])
def test_callback_device_contract_is_checked_before_reading_output(callback_name):
    state = torch.zeros(2, dtype=torch.float64)
    callbacks = {"solve": lambda x: x, "normal_action": lambda x: x}
    callbacks[callback_name] = lambda x: x.to(device="meta")
    with pytest.raises(ValueError, match="dtype and device"):
        _estimate_inverse_diagonal_hutchinson(
            reference_state=state, num_probes=1, seed=0, **callbacks
        )


@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
        ("cpu", torch.complex64),
        ("cpu", torch.complex128),
        pytest.param(
            "cuda",
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            torch.float64,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_spd_diagonal_dense_dtype_device_and_damping_contract(device, dtype):
    state = torch.zeros((2, 3), device=device, dtype=dtype)
    diagonal = (
        torch.arange(1, 7, device=device, dtype=state.real.dtype)
        .reshape(2, 3)
        .to(dtype)
    )
    operator = build_state_diagonal_operator(state, diagonal)
    damping = 0.5
    expected = (diagonal.abs().square() + damping).reciprocal()
    estimate = operator.estimate_posterior_variance_diagonal(
        state,
        num_probes=2,
        seed=12,
        iterations=6,
        damping=damping,
        preconditioner=expected,
        rtol=1e-5 if state.real.dtype == torch.float32 else 1e-12,
    )
    torch.testing.assert_close(estimate, expected)
    assert estimate.device == state.device
    assert estimate.dtype == state.real.dtype


def test_damped_inverse_curvature_is_not_regularized_sampling_covariance():
    state = torch.zeros(2, dtype=torch.float64)
    forward = torch.tensor([1.0, 2.0], dtype=state.dtype)
    operator = build_state_diagonal_operator(state, forward)
    damping = 2.0
    curvature = operator.estimate_posterior_variance_diagonal(
        state, num_probes=1, iterations=2, damping=damping, rtol=1e-12
    )
    normal = forward.square()
    sampling_covariance = normal / (normal + damping).square()
    torch.testing.assert_close(curvature, (normal + damping).reciprocal())
    assert not torch.allclose(curvature, sampling_covariance)
