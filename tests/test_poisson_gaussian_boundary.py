"""Independent right-boundary checks for the finite Poisson/Gaussian law."""

import math

import numpy as np
import pytest
from scipy.stats import norm, poisson
import torch

from pg_oracle import law_cdf, window
from torchsky.map.operators.noise import (
    poisson_gaussian_cdf,
    poisson_gaussian_nll,
    reduce_detector_nll,
)


def _law(kind, parameters, *, tail_sigma):
    rate, variance, gain, background = parameters.unbind()
    value = parameters.new_tensor(1.3)
    fn = poisson_gaussian_nll if kind == "nll" else poisson_gaussian_cdf
    return fn(
        value,
        rate,
        variance,
        gain=gain,
        background=background,
        tail_sigma=tail_sigma,
    )


def _polynomial_law(kind, parameters, *, count_two):
    # Independent direct finite probabilities; no logarithmic zero weight.
    rate, variance, gain, background = parameters.unbind()
    counts = torch.arange(3 if count_two else 2, dtype=parameters.dtype)
    terms = [torch.ones_like(rate), rate]
    if count_two:
        terms.append(rate.square() / 2)
    weights = torch.stack(terms)
    weights = weights / weights.sum()
    conditional = torch.distributions.Normal(
        background + gain * counts, variance.sqrt()
    )
    value = parameters.new_tensor(1.3)
    if kind == "nll":
        return -(weights * conditional.log_prob(value).exp()).sum().log()
    return (weights * conditional.cdf(value)).sum()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize(
    "tail_sigma,count_two",
    ((0.5, False), ((math.sqrt(5) - 1) / 2, True), (0.8, True), (12.0, True)),
)
@pytest.mark.parametrize("kind", ("nll", "cdf"))
def test_zero_rate_first_second_and_mixed_derivatives(
    kind, tail_sigma, count_two, dtype
):
    parameters = torch.tensor([0.0, 0.36, 0.7, -0.2], dtype=dtype, requires_grad=True)

    def fn(p):
        return _law(kind, p, tail_sigma=tail_sigma)

    def oracle(p):
        return _polynomial_law(kind, p, count_two=count_two)

    torch.testing.assert_close(fn(parameters), oracle(parameters))
    actual_gradient = torch.autograd.functional.jacobian(fn, parameters)
    actual_hessian = torch.autograd.functional.hessian(fn, parameters)
    expected_gradient = torch.autograd.functional.jacobian(oracle, parameters)
    expected_hessian = torch.autograd.functional.hessian(oracle, parameters)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        actual_hessian, expected_hessian, atol=tolerance, rtol=tolerance
    )


def test_independent_scipy_one_sided_scores_and_finite_mixture_values():
    y, variance, gain, background = 1.3, 0.36, 0.7, -0.2
    sigma = math.sqrt(variance)
    counts = np.array([0, 1])
    density = norm.pdf(y, background + gain * counts, sigma)
    cdf = norm.cdf(y, background + gain * counts, sigma)
    scores = (1 - density[1] / density[0], cdf[1] - cdf[0])
    for fn, score in zip(
        (poisson_gaussian_nll, poisson_gaussian_cdf), scores, strict=True
    ):
        rate = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        value = fn(
            torch.tensor(y, dtype=rate.dtype),
            rate,
            torch.tensor(variance, dtype=rate.dtype),
            gain=gain,
            background=background,
            tail_sigma=0.5,
            max_terms=2,
        )
        actual = torch.autograd.grad(value, rate)[0].item()
        assert actual == pytest.approx(score, rel=1e-13, abs=1e-13)
        step = 1e-7
        weights = poisson.pmf(counts, step)
        weights /= weights.sum()
        independent = (
            -np.log(weights @ density) if fn is poisson_gaussian_nll else weights @ cdf
        )
        assert (independent - value.item()) / step == pytest.approx(
            score, rel=1e-5, abs=1e-8
        )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_mixed_zero_and_positive_rates_do_not_evaluate_unselected_overflow(dtype):
    value = torch.tensor([0.3, 100.0], dtype=dtype)
    rate = torch.tensor([0.0, 20.0], dtype=dtype, requires_grad=True)
    variance = torch.ones_like(value)
    for fn in (poisson_gaussian_nll, poisson_gaussian_cdf):
        options = {"reduction": "none"} if fn is poisson_gaussian_nll else {}
        output = fn(value, rate, variance, **options)
        ordinary = fn(value, rate.detach(), variance, **options)
        torch.testing.assert_close(output, ordinary)
        gradient = torch.autograd.grad(output.sum(), rate, retain_graph=True)[0]
        assert bool(torch.isfinite(gradient).all())


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_inference_zero_value_survives_unrepresentable_boundary_derivatives(dtype):
    value = torch.tensor(5.0, dtype=dtype)
    variance = torch.tensor(0.01, dtype=dtype)
    rate = torch.tensor(0.0, dtype=dtype)
    actual = poisson_gaussian_nll(value, rate, variance)
    reference = -torch.distributions.Normal(
        torch.zeros_like(value), variance.sqrt()
    ).log_prob(value)
    torch.testing.assert_close(actual, reference)
    assert bool(torch.isfinite(actual))
    # First-order right derivative -expm1(450) is representable in float64 only;
    # the second-order jet (any-order path) overflows in both dtypes.
    rate.requires_grad_()
    if dtype == torch.float32:
        with pytest.raises(ValueError, match="derivatives exceed the dtype range"):
            poisson_gaussian_nll(value, rate, variance)
    else:
        loss = poisson_gaussian_nll(value, rate, variance)
        assert bool(
            torch.isfinite(torch.autograd.grad(loss, rate, retain_graph=True)[0])
        )
        with pytest.raises(ValueError, match="derivatives exceed the dtype range"):
            torch.autograd.grad(loss, rate, create_graph=True)
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            torch.testing.assert_close(
                poisson_gaussian_nll(value, rate, variance), reference
            )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_zero_rate_normalization_and_cdf_endpoints(dtype):
    value = torch.linspace(-8.0, 8.0, 2001, dtype=dtype)
    rate = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    variance = torch.tensor(0.36, dtype=dtype)
    density = (-poisson_gaussian_nll(value, rate, variance, reduction="none")).exp()
    torch.testing.assert_close(
        torch.trapezoid(density, value),
        torch.ones((), dtype=dtype),
        atol=2e-6,
        rtol=2e-6,
    )
    cdf = poisson_gaussian_cdf(value, rate, variance)
    assert bool((cdf[1:] >= cdf[:-1]).all())
    torch.testing.assert_close(
        cdf[0],
        cdf.new_tensor(norm.cdf(-8.0 / 0.6)),
        rtol=2e-4 if dtype == torch.float32 else 3e-13,
        atol=0,
    )
    assert cdf[-1] == 1
    assert bool(torch.isfinite(torch.autograd.grad(cdf.sum(), rate)[0]))


@pytest.mark.parametrize("kind", ("nll", "cdf"))
def test_empty_and_masked_reduction_contracts(kind):
    fn = poisson_gaussian_nll if kind == "nll" else poisson_gaussian_cdf
    empty = torch.empty(0, dtype=torch.float64, requires_grad=True)
    options = {"reduction": "none"} if kind == "nll" else {}
    assert fn(empty, empty, empty, **options).shape == (0,)
    if kind == "nll":
        for reduction in ("mean", "sum"):
            with pytest.raises(ValueError, match="empty"):
                fn(empty, empty, empty, reduction=reduction)
        pixel_nll = fn(
            torch.tensor([0.2, 1.0]),
            torch.tensor([0.0, 2.0]),
            torch.ones(2),
            reduction="none",
        )
        mask = torch.tensor([True, False])
        torch.testing.assert_close(
            reduce_detector_nll(pixel_nll, mask=mask, reduction="sum"), pixel_nll[0]
        )
        with pytest.raises(ValueError, match="no usable"):
            reduce_detector_nll(pixel_nll, mask=~torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize(
    "dtype", (torch.int64, torch.complex64, torch.float16, torch.bfloat16)
)
def test_unsupported_observed_dtypes_fail_closed(dtype):
    for fn in (poisson_gaussian_nll, poisson_gaussian_cdf):
        with pytest.raises(TypeError, match="real float32 or float64"):
            fn(torch.ones(1, dtype=dtype), torch.ones(1), torch.ones(1))


@pytest.mark.parametrize("device", ("cpu", "cuda", "mps"))
def test_device_zero_rate_score(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    rate = torch.tensor(0.0, device=device, requires_grad=True)
    result = poisson_gaussian_nll(
        torch.tensor(0.3, device=device), rate, torch.tensor(1.0, device=device)
    )
    gradient = torch.autograd.grad(result, rate)[0]
    torch.testing.assert_close(
        gradient, rate.new_tensor(1 - math.exp(-0.2)), rtol=2e-6, atol=2e-6
    )
    assert result.device == rate.device


@pytest.mark.parametrize("kind", ("nll", "cdf"))
def test_zero_rate_forward_mode_uses_right_hand_score(kind):
    rate = torch.tensor(0.0, dtype=torch.float64)
    value, variance = rate.new_tensor(1.3), rate.new_tensor(0.36)
    fn = poisson_gaussian_nll if kind == "nll" else poisson_gaussian_cdf

    def evaluate(r):
        return fn(value, r, variance, gain=0.7, background=-0.2, tail_sigma=0.5)

    sigma = math.sqrt(0.36)
    expected = (
        1 - norm.pdf(1.3, 0.5, sigma) / norm.pdf(1.3, -0.2, sigma)
        if kind == "nll"
        else norm.cdf(1.3, 0.5, sigma) - norm.cdf(1.3, -0.2, sigma)
    )
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(rate, torch.ones_like(rate))
        unpacked = torch.autograd.forward_ad.unpack_dual(evaluate(dual))
        assert unpacked.tangent.item() == pytest.approx(expected, rel=1e-13, abs=1e-13)
    _, tangent = torch.func.jvp(evaluate, (rate,), (torch.ones_like(rate),))
    assert tangent.item() == pytest.approx(expected, rel=1e-13, abs=1e-13)
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(rate, torch.ones_like(rate))
        unpacked = torch.autograd.forward_ad.unpack_dual(evaluate(dual))
        assert unpacked.tangent.item() == pytest.approx(expected, rel=1e-13, abs=1e-13)


@pytest.mark.parametrize("kind", ("nll", "cdf"))
@pytest.mark.parametrize("tail_sigma,count_two", ((0.61803398, True), (1e-46, False)))
def test_float32_rounded_and_underflowed_zero_support_uses_right_jet(
    kind, tail_sigma, count_two
):
    parameters = torch.tensor([0.0, 0.36, 0.7, -0.2], requires_grad=True)

    def fn(p):
        return _law(kind, p, tail_sigma=tail_sigma)

    def oracle(p):
        return _polynomial_law(kind, p, count_two=count_two)

    torch.testing.assert_close(fn(parameters), oracle(parameters))
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(fn, parameters),
        torch.autograd.functional.jacobian(oracle, parameters),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(fn, parameters),
        torch.autograd.functional.hessian(oracle, parameters),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("value", (-10.0, 10.0))
@pytest.mark.parametrize("tail_sigma", (0.5, 12.0))
def test_cdf_tail_boundary_scores_curvature_and_mixed_derivatives(
    dtype, value, tail_sigma
):
    parameters = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=dtype, requires_grad=True)
    rate, variance, gain, background = parameters.unbind()
    actual = poisson_gaussian_cdf(
        parameters.new_tensor(value),
        rate,
        variance,
        gain=gain,
        background=background,
        tail_sigma=tail_sigma,
    )
    gradient = torch.autograd.grad(actual, parameters, create_graph=True)[0]
    rate_row = torch.autograd.grad(gradient[0], parameters)[0]
    z0, z1, z2 = value, value - 1, value - 2
    if value > 0:
        score = norm.sf(z0) - norm.sf(z1)
        next_score = norm.sf(z1) - norm.sf(z2)
    else:
        score = norm.cdf(z1) - norm.cdf(z0)
        next_score = norm.cdf(z2) - norm.cdf(z1)
    curvature = next_score - score if tail_sigma == 12.0 else -2 * score
    p0, p1 = norm.pdf(z0), norm.pdf(z1)
    reference_row = parameters.new_tensor(
        [
            curvature,
            (z0 * p0 - z1 * p1) / 2,
            -p1,
            p0 - p1,
        ]
    )
    tolerance = 3e-5 if dtype == torch.float32 else 3e-13
    torch.testing.assert_close(
        gradient[0], parameters.new_tensor(score), rtol=tolerance, atol=0
    )
    torch.testing.assert_close(rate_row, reference_row, rtol=tolerance, atol=0)
    torch.testing.assert_close(
        actual, parameters.new_tensor(norm.cdf(z0)), rtol=tolerance, atol=0
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("rate", (0.0, 1e-6, 0.2))
def test_positive_and_zero_rate_lower_cdf_tail_matches_scipy(dtype, rate):
    value = torch.tensor(-10.0, dtype=dtype)
    actual = poisson_gaussian_cdf(
        value, value.new_tensor(rate), value.new_tensor(1.0), tail_sigma=0.5
    )
    lower, upper, side = window(-10.0, rate, 1.0, 1.0, 0.0, 0.5, cdf=True)
    # Far-below pixels use their own window and the exact unrenormalised pmf;
    # otherwise the prior window is merged in and renormalised.
    counts = np.arange(lower, upper + 1)
    weights = poisson.pmf(counts, rate)
    if not side:
        weights /= weights.sum()
    expected = weights @ norm.cdf(-10.0 - counts)
    torch.testing.assert_close(
        actual,
        value.new_tensor(expected),
        rtol=3e-5 if dtype == torch.float32 else 3e-13,
        atol=0,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("value", (-10.0, 10.0))
@pytest.mark.parametrize("rate", (1e-40, 1e-30, 1e-6, 0.2))
def test_positive_rate_cdf_tail_first_second_derivatives(dtype, value, rate):
    r = torch.tensor(rate, dtype=dtype, requires_grad=True)
    actual = poisson_gaussian_cdf(
        r.new_tensor(value), r, r.new_tensor(1.0), tail_sigma=0.5
    )
    score = torch.autograd.grad(actual, r, create_graph=True)[0]
    curvature = torch.autograd.grad(score, r)[0]
    lower, upper, side = window(value, float(r.detach()), 1.0, 1.0, 0.0, 0.5, cdf=True)
    if (lower, upper, side) != (0, 1, 0):
        r64 = r.detach().double().requires_grad_()
        one = r64.new_tensor(1.0)
        reference = law_cdf(
            r64.new_tensor(value), r64, one, one, one * 0, lower, upper, side
        )
        ref_score = torch.autograd.grad(reference, r64, create_graph=True)[0]
        ref_curv = torch.autograd.grad(ref_score, r64)[0]
        tol = 3e-5 if dtype == torch.float32 else 3e-12
        for got, want in (
            (actual, reference),
            (score, ref_score),
            (curvature, ref_curv),
        ):
            torch.testing.assert_close(
                got, r.new_tensor(float(want.detach())), rtol=tol, atol=0
            )
        return
    difference = (
        norm.sf(value) - norm.sf(value - 1)
        if value > 0
        else norm.cdf(value - 1) - norm.cdf(value)
    )
    tolerance = 3e-5 if dtype == torch.float32 else 3e-13
    torch.testing.assert_close(
        score, r.new_tensor(difference / (1 + rate) ** 2), rtol=tolerance, atol=0
    )
    torch.testing.assert_close(
        curvature,
        r.new_tensor(-2 * difference / (1 + rate) ** 3),
        rtol=tolerance,
        atol=0,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("value", (-10.0, 10.0))
def test_tiny_positive_rate_cdf_tail_first_derivative(dtype, value):
    rate = torch.tensor(1e-6, dtype=dtype, requires_grad=True)
    cdf = poisson_gaussian_cdf(
        rate.new_tensor(value), rate, rate.new_tensor(1.0), tail_sigma=0.5
    )
    score = torch.autograd.grad(cdf, rate)[0]
    difference = (
        norm.sf(value) - norm.sf(value - 1)
        if value > 0
        else norm.cdf(value - 1) - norm.cdf(value)
    )
    torch.testing.assert_close(
        score,
        rate.new_tensor(difference / (1 + 1e-6) ** 2),
        rtol=3e-5 if dtype == torch.float32 else 3e-13,
        atol=0,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("rate", (0.2, 3.0))
def test_central_cdf_derivative_matches_normalized_density(dtype, rate):
    value = torch.tensor([-3.0, -0.1, 0.4, 2.0, 4.0], dtype=dtype, requires_grad=True)
    r, variance = value.new_tensor(rate), value.new_tensor(0.64)
    options = dict(gain=0.7, background=0.1, tail_sigma=0.5)
    cdf = poisson_gaussian_cdf(value, r, variance, **options)
    derivative = torch.autograd.grad(cdf.sum(), value)[0]
    density = (
        -poisson_gaussian_nll(value, r, variance, reduction="none", **options)
    ).exp()
    torch.testing.assert_close(
        derivative, density, rtol=3e-5 if dtype == torch.float32 else 3e-13, atol=0
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("rate", (1e-6, 0.3, 3.4, 7.3))
def test_positive_cdf_mixed_hessian_and_jvp_match_finite_polynomial(dtype, rate):
    parameters = torch.tensor([rate, 0.64, 0.7, -0.2], dtype=dtype)
    value = parameters.new_tensor(0.7 * rate + 0.4)
    lower, upper, side = window(float(value), rate, 0.64, 0.7, -0.2, 0.5, cdf=True)
    assert side == 0
    counts = torch.arange(lower, upper + 1, dtype=dtype)
    factorials = parameters.new_tensor(
        [math.factorial(k) for k in range(lower, upper + 1)]
    )

    def actual(p):
        return poisson_gaussian_cdf(
            value, p[0], p[1], gain=p[2], background=p[3], tail_sigma=0.5
        )

    def oracle(p):
        # Independent finite polynomial, including supports starting above zero.
        weights = p[0].pow(counts) / factorials
        weights = weights / weights.sum()
        cdf = torch.distributions.Normal(p[3] + p[2] * counts, p[1].sqrt()).cdf(value)
        return (weights * cdf).sum()

    tolerance = 3e-5 if dtype == torch.float32 else 3e-12
    absolute = 1e-6 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(actual(parameters), oracle(parameters))
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(actual, parameters),
        torch.autograd.functional.jacobian(oracle, parameters),
        rtol=tolerance,
        atol=absolute,
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(actual, parameters),
        torch.autograd.functional.hessian(oracle, parameters),
        rtol=tolerance,
        atol=absolute,
    )
    direction = parameters.new_tensor([0.7, -0.2, 0.3, -0.4])
    torch.testing.assert_close(
        torch.func.jvp(actual, (parameters,), (direction,))[1],
        torch.func.jvp(oracle, (parameters,), (direction,))[1],
        rtol=tolerance,
        atol=absolute,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("rate,tail_sigma", ((3.4, 0.5), (100.3, 0.5), (10000.3, 12)))
def test_mode_scaled_cdf_matches_scipy_for_shifted_and_wide_support(
    dtype, rate, tail_sigma
):
    r = torch.tensor(rate, dtype=dtype)
    # Compare exactly the supplied floating-point rate and retained support.
    values = r + torch.tensor([-2.0, 0.0, 2.0], dtype=dtype) * (r + 1).sqrt()
    for index, value in enumerate(values):
        lower, upper, side = window(
            float(value), float(r), 1.0, 1.0, 0.0, tail_sigma, cdf=True
        )
        counts = np.arange(lower, upper + 1)
        weights = poisson.pmf(counts, float(r))
        if side == 0:
            weights /= weights.sum()
        expected = norm.cdf(float(value) - counts) @ weights
        torch.testing.assert_close(
            poisson_gaussian_cdf(value, r, torch.ones_like(r), tail_sigma=tail_sigma),
            value.new_tensor(expected),
            rtol=3e-5 if dtype == torch.float32 else 3e-12,
            atol=0,
        )


def test_batched_cdf_support_masks_preserve_scalar_values_and_rate_derivatives():
    rates = torch.tensor([0.0, 1e-30, 0.2, 3.4, 100.3], requires_grad=True)
    values = rates.detach() + 0.4
    cdf = poisson_gaussian_cdf(values, rates, torch.ones_like(rates), tail_sigma=0.5)
    gradient = torch.autograd.grad(cdf.sum(), rates, create_graph=True)[0]
    curvature = torch.autograd.grad(gradient.sum(), rates)[0]
    for index, rate in enumerate(rates.detach()):
        scalar_rate = rate.clone().requires_grad_()
        scalar_cdf = poisson_gaussian_cdf(
            values[index], scalar_rate, rate.new_tensor(1.0), tail_sigma=0.5
        )
        scalar_gradient = torch.autograd.grad(
            scalar_cdf, scalar_rate, create_graph=True
        )[0]
        scalar_curvature = torch.autograd.grad(scalar_gradient, scalar_rate)[0]
        torch.testing.assert_close(cdf[index], scalar_cdf)
        torch.testing.assert_close(gradient[index], scalar_gradient)
        torch.testing.assert_close(curvature[index], scalar_curvature)


@pytest.mark.parametrize("device", ("cpu", "cuda", "mps"))
def test_device_positive_cdf_rate_score_and_curvature(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    rates = torch.full((2,), 1e-6, device=device, requires_grad=True)
    values = rates.new_tensor([-0.4, 1.2])
    cdf = poisson_gaussian_cdf(values, rates, torch.ones_like(rates), tail_sigma=0.5)
    score = torch.autograd.grad(cdf.sum(), rates, create_graph=True)[0]
    curvature = torch.autograd.grad(score.sum(), rates)[0]
    differences = rates.new_tensor(
        norm.cdf(np.array([-1.4, 0.2])) - norm.cdf([-0.4, 1.2])
    )
    torch.testing.assert_close(score, differences / (1 + rates) ** 2)
    torch.testing.assert_close(curvature, -2 * differences / (1 + rates) ** 3)
    assert cdf.device == rates.device
