"""Bounded statistical contracts for linear Gaussian nuisance elimination."""

import pytest
import torch

from torchsky.map.operators.nuisance import (
    CalibratedNuisancePrecision,
    ProfiledNuisancePrecision,
)


DTYPE = torch.float64


def _basis():
    return torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, -1.0]], dtype=DTYPE)


def _dense_action(operator):
    size = operator._flat_basis.shape[0]
    eye = torch.eye(size, dtype=operator.dtype, device=operator.device)
    return torch.stack(
        [
            operator(eye[:, index].reshape(operator.data_shape)).reshape(-1)
            for index in range(size)
        ],
        dim=1,
    )


def test_profile_matches_independent_weighted_least_squares():
    basis = _basis()
    weights = torch.tensor([2.0, 0.5, 0.0, 3.0], dtype=DTYPE)
    residual = torch.tensor([2.0, -1.0, 8.0, 3.0], dtype=DTYPE)
    operator = ProfiledNuisancePrecision(basis, apply_precision=lambda x: weights * x)
    whitened_basis = weights.sqrt()[:, None] * basis
    coefficients = torch.linalg.lstsq(
        whitened_basis, weights.sqrt() * residual
    ).solution
    fitted_residual = residual - basis @ coefficients
    torch.testing.assert_close(operator(residual), weights * fitted_residual)
    torch.testing.assert_close(
        operator.quadratic(residual), (weights * fitted_residual.square()).sum()
    )
    assert operator.nuisance_rank == 2
    assert operator.treatment == "profiled_unknown"


def test_profile_correlated_precision_and_projection_gauge():
    basis = _basis()
    covariance = torch.eye(4, dtype=DTYPE) + 0.3 * torch.ones(4, 4, dtype=DTYPE)

    def base(x):
        return torch.linalg.solve(covariance, x)

    operator = ProfiledNuisancePrecision(basis, apply_precision=base)
    actual = _dense_action(operator)
    weights = torch.linalg.solve(covariance, torch.eye(4, dtype=DTYPE))
    projection = torch.eye(4, dtype=DTYPE) - basis @ torch.linalg.solve(
        basis.T @ weights @ basis, basis.T @ weights
    )
    torch.testing.assert_close(actual, projection.T @ weights @ projection)
    torch.testing.assert_close(
        actual @ basis, torch.zeros_like(basis), atol=1e-14, rtol=0.0
    )
    torch.testing.assert_close(projection @ projection, projection)
    torch.testing.assert_close(actual, actual.T)
    assert torch.linalg.eigvalsh(actual).min() > -1e-14
    assert abs(operator.quadratic(torch.ones(4, dtype=DTYPE))) < 1e-14


@pytest.mark.parametrize(
    "factor",
    [
        torch.tensor([[0.3, 0.0], [0.2, 0.7]], dtype=DTYPE),
        torch.tensor([[0.3], [0.2]], dtype=DTYPE),
    ],
)
def test_calibrated_matches_dense_covariance_and_normalization(factor):
    basis = _basis()
    covariance = torch.eye(4, dtype=DTYPE) + 0.2 * torch.ones(4, 4, dtype=DTYPE)
    mean = torch.tensor([2.0, -0.3], dtype=DTYPE)
    operator = CalibratedNuisancePrecision(
        basis,
        apply_precision=lambda x: torch.linalg.solve(covariance, x),
        covariance_factor=factor,
        mean=mean,
    )
    full_covariance = covariance + basis @ factor @ factor.T @ basis.T
    expected = torch.linalg.solve(full_covariance, torch.eye(4, dtype=DTYPE))
    torch.testing.assert_close(_dense_action(operator), expected)
    data = torch.tensor([3.0, 4.0, 2.0, -1.0], dtype=DTYPE)
    torch.testing.assert_close(operator.center(data), data - basis @ mean)
    correction = (
        torch.linalg.slogdet(full_covariance).logabsdet
        - torch.linalg.slogdet(covariance).logabsdet
    )
    torch.testing.assert_close(operator.logdet_covariance_correction, correction)
    assert operator.treatment == "calibrated_random_effect"
    assert operator.quadratic(torch.ones(4, dtype=DTYPE)) > 0.0


def test_mask_uses_observed_correlated_precision_and_ignores_invalid_pixels():
    valid = torch.tensor([True, False, True, True])
    basis = _basis()
    basis[1] = float("nan")
    observed_covariance = torch.tensor(
        [[2.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.5]], dtype=DTYPE
    )

    def base(x):
        assert x[1] == 0
        result = torch.zeros_like(x)
        result[valid] = torch.linalg.solve(observed_covariance, x[valid])
        return result

    operator = ProfiledNuisancePrecision(basis, apply_precision=base, valid=valid)
    observed_operator = ProfiledNuisancePrecision(
        basis[valid],
        apply_precision=lambda x: torch.linalg.solve(observed_covariance, x),
    )
    residual = torch.tensor([0.5, float("nan"), -0.3, 2.0], dtype=DTYPE)
    torch.testing.assert_close(
        operator(residual)[valid], observed_operator(residual[valid])
    )
    assert operator(residual)[1] == 0.0
    torch.testing.assert_close(
        operator.quadratic(residual), observed_operator.quadratic(residual[valid])
    )
    calibrated = CalibratedNuisancePrecision(
        basis,
        apply_precision=base,
        covariance_factor=torch.eye(2, dtype=DTYPE),
        mean=torch.ones(2, dtype=DTYPE),
        valid=valid,
    )
    assert calibrated.center(residual)[1] == 0.0
    assert calibrated(residual)[1] == 0.0


def test_masking_full_correlated_precision_is_not_missing_data_marginalization():
    covariance = torch.tensor([[2.0, 0.8], [0.8, 1.0]], dtype=DTYPE)
    basis = torch.empty(2, 0, dtype=DTYPE)
    valid = torch.tensor([True, False])
    wrong = ProfiledNuisancePrecision(
        basis, apply_precision=lambda x: torch.linalg.solve(covariance, x), valid=valid
    )
    observed = ProfiledNuisancePrecision(
        basis, apply_precision=lambda x: x / 2.0, valid=valid
    )
    residual = torch.tensor([1.0, float("nan")], dtype=DTYPE)
    assert wrong.quadratic(residual) != observed.quadratic(residual)
    torch.testing.assert_close(
        observed.quadratic(residual), torch.tensor(0.5, dtype=DTYPE)
    )


def test_redundant_basis_reports_rank_and_preserves_projection():
    column = _basis()[:, 0]
    basis = torch.stack([column, 2 * column, torch.zeros_like(column)], dim=1)
    weights = torch.tensor([1.0, 2.0, 0.0, 4.0], dtype=DTYPE)
    operator = ProfiledNuisancePrecision(basis, apply_precision=lambda x: weights * x)
    assert operator.nuisance_rank == 1
    residual = torch.tensor([3.0, -1.0, 20.0, 2.0], dtype=DTYPE)
    offset = (weights * residual).sum() / weights.sum()
    torch.testing.assert_close(operator(residual), weights * (residual - offset))
    torch.testing.assert_close(
        operator(column), torch.zeros_like(column), atol=1e-14, rtol=0.0
    )


def test_declared_rank_tolerance_removes_modes_without_information_floor():
    basis = torch.diag(torch.tensor([1.0, 0.01], dtype=DTYPE))
    complete = ProfiledNuisancePrecision(
        basis, apply_precision=lambda x: x, rank_rtol=0.0
    )
    truncated = ProfiledNuisancePrecision(
        basis, apply_precision=lambda x: x, rank_atol=1e-3
    )
    assert complete.nuisance_rank == 2
    assert truncated.nuisance_rank == 1
    torch.testing.assert_close(
        complete(torch.ones(2, dtype=DTYPE)), torch.zeros(2, dtype=DTYPE)
    )
    torch.testing.assert_close(
        truncated(torch.ones(2, dtype=DTYPE)), torch.tensor([0.0, 1.0], dtype=DTYPE)
    )


@pytest.mark.parametrize("treatment", ["profiled", "calibrated"])
def test_all_zero_weights_and_empty_observed_support(treatment):
    basis = _basis()
    kwargs = {"apply_precision": lambda x: torch.zeros_like(x)}
    if treatment == "profiled":
        operator = ProfiledNuisancePrecision(basis, **kwargs)
        assert operator.nuisance_rank == 0
    else:
        operator = CalibratedNuisancePrecision(
            basis, covariance_factor=torch.eye(2, dtype=DTYPE), **kwargs
        )
        assert operator.logdet_covariance_correction == 0.0
    torch.testing.assert_close(
        operator(torch.ones(4, dtype=DTYPE)), torch.zeros(4, dtype=DTYPE)
    )
    masked = ProfiledNuisancePrecision(
        torch.full_like(basis, float("nan")),
        apply_precision=lambda x: x,
        valid=torch.zeros(4, dtype=torch.bool),
    )
    assert masked.nuisance_rank == 0
    assert masked.quadratic(torch.full((4,), float("nan"), dtype=DTYPE)) == 0.0


@pytest.mark.parametrize("shape", [(3,), (2, 3), (0,)])
def test_empty_nuisance_and_zero_calibration_covariance_are_exact_base_limits(shape):
    residual = torch.ones(shape, dtype=DTYPE)
    basis = torch.empty(*shape, 0, dtype=DTYPE)
    profiled = ProfiledNuisancePrecision(basis, apply_precision=lambda x: 2 * x)
    calibrated = CalibratedNuisancePrecision(
        basis,
        apply_precision=lambda x: 2 * x,
        covariance_factor=torch.empty(0, 0, dtype=DTYPE),
    )
    for operator in (profiled, calibrated):
        torch.testing.assert_close(operator(residual), 2 * residual)
    assert calibrated.logdet_covariance_correction == 0.0
    nonempty = torch.ones(*shape, 1, dtype=DTYPE)
    for factor in (torch.empty(1, 0, dtype=DTYPE), torch.zeros(1, 2, dtype=DTYPE)):
        zero_covariance = CalibratedNuisancePrecision(
            nonempty, apply_precision=lambda x: 2 * x, covariance_factor=factor
        )
        torch.testing.assert_close(zero_covariance(residual), 2 * residual)
        assert zero_covariance.logdet_covariance_correction == 0.0


@pytest.mark.parametrize("calibrated", [False, True])
def test_adjoint_and_residual_quadratic_gradient(calibrated):
    basis = _basis().reshape(2, 2, 2)
    weights = torch.tensor([[1.0, 3.0], [2.0, 0.0]], dtype=DTYPE)
    kwargs = {"apply_precision": lambda x: weights * x}
    operator = (
        CalibratedNuisancePrecision(
            basis, covariance_factor=torch.eye(2, dtype=DTYPE), **kwargs
        )
        if calibrated
        else ProfiledNuisancePrecision(basis, **kwargs)
    )
    left = torch.tensor([[0.5, 2.0], [-0.3, 1.0]], dtype=DTYPE, requires_grad=True)
    right = torch.tensor([[1.0, -0.1], [2.0, 0.5]], dtype=DTYPE)
    torch.testing.assert_close(
        (left * operator(right)).sum(), (operator(left) * right).sum()
    )
    (gradient,) = torch.autograd.grad(operator.quadratic(left), left)
    torch.testing.assert_close(gradient, 2 * operator(left))


def test_profile_gradcheck_basis_precision_and_residual():
    basis = _basis().requires_grad_()
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE, requires_grad=True)
    residual = torch.tensor([0.5, 2.0, -0.3, 1.0], dtype=DTYPE, requires_grad=True)

    def loss(b, w, r):
        return ProfiledNuisancePrecision(b, apply_precision=lambda x: w * x).quadratic(
            r
        )

    assert torch.autograd.gradcheck(loss, (basis, weights, residual))


def test_redundant_profile_gradcheck_on_fixed_rank_stratum():
    column = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE, requires_grad=True)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE, requires_grad=True)
    residual = torch.tensor([0.5, 2.0, -0.3, 1.0], dtype=DTYPE, requires_grad=True)

    def loss(c, w, r):
        basis = torch.stack([c, 2 * c, torch.zeros_like(c)], dim=1)
        return ProfiledNuisancePrecision(
            basis, apply_precision=lambda x: w * x
        ).quadratic(r)

    assert torch.autograd.gradcheck(loss, (column, weights, residual))


def test_calibrated_gradcheck_includes_covariance_normalization_and_mean():
    basis = _basis().requires_grad_()
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE, requires_grad=True)
    factor = torch.tensor([[0.5], [0.2]], dtype=DTYPE, requires_grad=True)
    mean = torch.tensor([0.1, -0.2], dtype=DTYPE, requires_grad=True)
    data = torch.tensor([0.5, 2.0, -0.3, 1.0], dtype=DTYPE, requires_grad=True)

    def loss(b, w, f, m, y):
        operator = CalibratedNuisancePrecision(
            b, apply_precision=lambda x: w * x, covariance_factor=f, mean=m
        )
        return (
            operator.quadratic(operator.center(y))
            + operator.logdet_covariance_correction
        )

    assert torch.autograd.gradcheck(loss, (basis, weights, factor, mean, data))


@pytest.mark.parametrize(
    "precision",
    [
        torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=DTYPE),
        torch.tensor([[2.0, 1.0], [0.0, 2.0]], dtype=DTYPE),
    ],
)
@pytest.mark.parametrize("calibrated", [False, True])
def test_invalid_base_precision_rejects_negative_or_asymmetric_gram(
    precision, calibrated
):
    kwargs = {"apply_precision": lambda x: precision @ x}
    with pytest.raises(ValueError, match="negative|asymmetric"):
        if calibrated:
            CalibratedNuisancePrecision(
                torch.eye(2, dtype=DTYPE),
                covariance_factor=torch.eye(2, dtype=DTYPE),
                **kwargs,
            )
        else:
            ProfiledNuisancePrecision(
                torch.eye(2, dtype=DTYPE), rank_rtol=100, **kwargs
            )


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.complex64])
def test_unsupported_dtypes_fail_explicitly(dtype):
    with pytest.raises(TypeError, match="float32 or float64"):
        ProfiledNuisancePrecision(
            torch.ones(2, 1, dtype=dtype), apply_precision=lambda x: x
        )


def test_input_callback_and_calibration_validation():
    basis = _basis()
    operator = ProfiledNuisancePrecision(basis, apply_precision=lambda x: x)
    with pytest.raises(ValueError, match="dtype and device"):
        operator(torch.ones(4, dtype=torch.float32))
    with pytest.raises(ValueError, match="data shape"):
        operator(torch.ones(2, 2, dtype=DTYPE))
    with pytest.raises(ValueError, match="finite"):
        operator(torch.full((4,), float("nan"), dtype=DTYPE))
    with pytest.raises(ValueError, match="dtype and device"):
        ProfiledNuisancePrecision(basis, apply_precision=lambda x: x.float())
    with pytest.raises(ValueError, match="data shape"):
        ProfiledNuisancePrecision(basis, apply_precision=lambda x: x[:, None])
    with pytest.raises(TypeError, match="boolean"):
        ProfiledNuisancePrecision(
            basis, apply_precision=lambda x: x, valid=torch.ones(4)
        )
    with pytest.raises(ValueError, match="shape and device"):
        ProfiledNuisancePrecision(
            basis, apply_precision=lambda x: x, valid=torch.ones(3, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="finite"):
        ProfiledNuisancePrecision(
            torch.full_like(basis, float("inf")), apply_precision=lambda x: x
        )
    for tolerance in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="rank_rtol"):
            ProfiledNuisancePrecision(
                basis, apply_precision=lambda x: x, rank_rtol=tolerance
            )
        with pytest.raises(ValueError, match="rank_atol"):
            ProfiledNuisancePrecision(
                basis, apply_precision=lambda x: x, rank_atol=tolerance
            )
    with pytest.raises(ValueError, match="calibration_rank"):
        CalibratedNuisancePrecision(
            basis,
            apply_precision=lambda x: x,
            covariance_factor=torch.ones(3, 1, dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="dtype and device"):
        CalibratedNuisancePrecision(
            basis, apply_precision=lambda x: x, covariance_factor=torch.ones(2, 1)
        )
    with pytest.raises(ValueError, match="finite"):
        CalibratedNuisancePrecision(
            basis,
            apply_precision=lambda x: x,
            covariance_factor=torch.full((2, 1), float("nan"), dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="mean must have shape"):
        CalibratedNuisancePrecision(
            basis,
            apply_precision=lambda x: x,
            covariance_factor=torch.eye(2, dtype=DTYPE),
            mean=torch.zeros(3, dtype=DTYPE),
        )


@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
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
        pytest.param(
            "mps",
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_available_device_dtype_contract(device, dtype):
    basis = _basis().to(device=device, dtype=dtype)
    residual = torch.arange(4, device=device, dtype=dtype)
    for operator in (
        ProfiledNuisancePrecision(basis, apply_precision=lambda x: x),
        CalibratedNuisancePrecision(
            basis,
            apply_precision=lambda x: x,
            covariance_factor=torch.eye(2, device=device, dtype=dtype),
        ),
    ):
        result = operator(residual)
        assert result.device == basis.device
        assert result.dtype == dtype
        assert torch.isfinite(result).all()
        dense = _dense_action(operator)
        torch.testing.assert_close(dense, dense.T)


def test_matrix_free_construction_only_calls_base_on_data_vectors():
    basis = torch.stack(
        [torch.ones(4096, dtype=DTYPE), torch.linspace(-1, 1, 4096, dtype=DTYPE)],
        dim=-1,
    )
    calls = []

    def base(x):
        calls.append(tuple(x.shape))
        return x * 2.0

    operator = ProfiledNuisancePrecision(basis, apply_precision=base)
    assert calls == [(4096,), (4096,)]
    operator(torch.ones(4096, dtype=DTYPE))
    assert calls == [(4096,)] * 3
