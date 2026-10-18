"""Independent finite-law checks for positive-rate detector likelihood AD."""

import math

import numpy as np
import pytest
from scipy.special import gammaln, logsumexp
from scipy.stats import norm
import torch

from torchsky.map.operators.noise import poisson_gaussian_cdf, poisson_gaussian_nll


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("value", (-10.0, 0.3, 1.3, 10.0))
@pytest.mark.parametrize("rate", (1e-40, 1e-30, 1e-6, 0.2))
def test_two_count_positive_nll_value_score_and_curvature(dtype, value, rate):
    r = torch.tensor(rate, dtype=dtype, requires_grad=True)
    actual = poisson_gaussian_nll(
        r.new_tensor(value), r, r.new_tensor(1.0), tail_sigma=0.5
    )
    score = torch.autograd.grad(actual, r, create_graph=True)[0]
    curvature = torch.autograd.grad(score, r)[0]
    represented_rate = float(r.detach())
    represented_value = float(r.new_tensor(value))
    ratio = math.exp(represented_value - 0.5)
    expected = -norm.logpdf(represented_value) + math.log1p(represented_rate)
    expected -= math.log1p(represented_rate * ratio)
    expected_score = (1 - ratio) / (
        (1 + represented_rate) * (1 + represented_rate * ratio)
    )
    expected_curvature = (ratio / (1 + represented_rate * ratio)) ** 2
    expected_curvature -= 1 / (1 + represented_rate) ** 2
    tolerance = 3e-5 if dtype == torch.float32 else 3e-13
    for a, b in (
        (actual, expected),
        (score, expected_score),
        (curvature, expected_curvature),
    ):
        torch.testing.assert_close(a, r.new_tensor(b), rtol=tolerance, atol=0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("rate", (1e-30, 1e-6, 0.2, 3.4, 7.3))
def test_positive_nll_full_mixed_hessian_and_jvp_match_finite_polynomial(dtype, rate):
    parameters = torch.tensor([rate, 0.64, 0.7, -0.2, 0.7 * rate + 0.4], dtype=dtype)
    radius = 0.5 * (parameters[0] + 1).sqrt() + 0.25
    lower = int((parameters[0] - radius).clamp_min(0).floor())
    upper = int((parameters[0] + radius).ceil())
    counts = torch.arange(lower, upper + 1, dtype=dtype)

    def actual(p):
        return poisson_gaussian_nll(
            p[4], p[0], p[1], gain=p[2], background=p[3], tail_sigma=0.5
        )

    def oracle(p):
        terms = [
            torch.ones_like(p[0])
            if k == 0
            else p[0]
            if k == 1
            else p[0].pow(k) / math.factorial(k)
            for k in range(lower, upper + 1)
        ]
        weights = torch.stack(terms)
        weights = weights / weights.sum()
        density = torch.distributions.Normal(p[3] + p[2] * counts, p[1].sqrt())
        return -(weights * density.log_prob(p[4]).exp()).sum().log()

    tolerance = 4e-5 if dtype == torch.float32 else 3e-12
    absolute = 1e-6 if dtype == torch.float32 else 2e-14
    for transform in (
        lambda f, p: f(p),
        torch.autograd.functional.jacobian,
        torch.autograd.functional.hessian,
    ):
        torch.testing.assert_close(
            transform(actual, parameters),
            transform(oracle, parameters),
            rtol=tolerance,
            atol=absolute,
        )
    direction = parameters.new_tensor([0.7, -0.2, 0.3, -0.4, 0.2])
    with torch.no_grad():
        torch.testing.assert_close(
            torch.func.jvp(actual, (parameters,), (direction,))[1],
            torch.func.jvp(oracle, (parameters,), (direction,))[1],
            rtol=tolerance,
            atol=absolute,
        )


def _scipy_finite_nll(value, rate, counts):
    log_terms = counts * math.log(rate) - gammaln(counts + 1)
    return logsumexp(log_terms) - logsumexp(log_terms + norm.logpdf(value - counts))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("rate,tail_sigma", ((100.3, 0.5), (10000.3, 12.0)))
def test_nll_wide_shifted_support_values_match_scipy(dtype, rate, tail_sigma):
    r = torch.tensor(rate, dtype=dtype)
    radius = tail_sigma * (r + 1).sqrt() + tail_sigma**2
    counts = np.arange(
        int((r - radius).clamp_min(0).floor()), int((r + radius).ceil()) + 1
    )
    values = r + torch.tensor([-2.0, 0.3, 2.0], dtype=dtype)
    expected = [_scipy_finite_nll(float(v), float(r), counts) for v in values]
    torch.testing.assert_close(
        poisson_gaussian_nll(
            values, r, torch.ones_like(r), tail_sigma=tail_sigma, reduction="none"
        ),
        values.new_tensor(expected),
        rtol=3e-5 if dtype == torch.float32 else 3e-12,
        atol=0,
    )


def test_joint_mode_retains_density_when_prior_and_gaussian_terms_underflow():
    rate, value = 1e-30, 71.0
    counts = np.arange(3)
    expected = _scipy_finite_nll(value, rate, counts)
    # All absolute Gaussian densities underflow, and the Float32 count-two
    # Poisson term underflows too. Their relative joint contribution matters.
    assert np.all(norm.pdf(value - counts) == 0)
    for dtype in (torch.float32, torch.float64):
        r = torch.tensor(rate, dtype=dtype)
        torch.testing.assert_close(
            poisson_gaussian_nll(
                r.new_tensor(value), r, r.new_tensor(1.0), tail_sigma=0.8
            ),
            r.new_tensor(expected),
            rtol=3e-5 if dtype == torch.float32 else 3e-13,
            atol=0,
        )
    r = torch.tensor(rate, dtype=torch.float64, requires_grad=True)
    nll = poisson_gaussian_nll(
        r.new_tensor(value), r, r.new_tensor(1.0), tail_sigma=0.8
    )
    score = torch.autograd.grad(nll, r, create_graph=True)[0]
    curvature = torch.autograd.grad(score, r)[0]
    log_terms = counts * math.log(rate) - gammaln(counts + 1)
    prior = np.exp(log_terms - logsumexp(log_terms))
    joint = log_terms + norm.logpdf(value - counts)
    conditional = np.exp(joint - logsumexp(joint))
    prior_score, joint_score = (prior @ counts) / rate, (conditional @ counts) / rate
    factorial = counts * (counts - 1)
    expected_curvature = joint_score**2 - (conditional @ factorial) / rate**2
    expected_curvature += (prior @ factorial) / rate**2 - prior_score**2
    torch.testing.assert_close(
        score, r.new_tensor(prior_score - joint_score), rtol=3e-12, atol=0
    )
    torch.testing.assert_close(
        curvature, r.new_tensor(expected_curvature), rtol=3e-12, atol=0
    )
    with pytest.raises(ValueError, match="modal derivative intermediates"):
        poisson_gaussian_nll(
            torch.tensor(value),
            torch.tensor(rate, requires_grad=True),
            torch.tensor(1.0),
            tail_sigma=0.8,
        )


def test_modal_range_value_only_and_forward_ad_contract():
    rate = torch.tensor(1e-40)
    # Subnormal modal reference scaling may round before division. The value
    # uses log sums; a requested unrepresentable derivative fails closed.
    value = torch.tensor(93.0)
    expected = _scipy_finite_nll(float(value), float(rate), np.arange(2))
    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            torch.testing.assert_close(
                poisson_gaussian_nll(
                    value, rate.requires_grad_(), torch.tensor(1.0), tail_sigma=0.5
                ),
                value.new_tensor(expected),
            )
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(rate.detach(), torch.ones_like(rate))
        with pytest.raises(ValueError, match="modal derivative intermediates"):
            poisson_gaussian_nll(value, dual, torch.tensor(1.0), tail_sigma=0.5)


@pytest.mark.parametrize("device", ("cpu", "cuda", "mps"))
def test_device_positive_nll_curvature(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    rate = torch.tensor(1e-30, device=device, requires_grad=True)
    loss = poisson_gaussian_nll(
        rate.new_tensor(1.3), rate, rate.new_tensor(1.0), tail_sigma=0.5
    )
    score = torch.autograd.grad(loss, rate, create_graph=True)[0]
    curvature = torch.autograd.grad(score, rate)[0]
    torch.testing.assert_close(
        curvature, rate.new_tensor(math.exp(1.6) - 1), rtol=3e-5, atol=0
    )
    assert loss.device == rate.device


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("value", (-1000.0, 1000.0))
def test_underflowing_modal_reference_conservatively_rejects_derivatives(dtype, value):
    rate = torch.tensor(1e-6, dtype=dtype, requires_grad=True)
    with pytest.raises(ValueError, match="modal derivative intermediates"):
        poisson_gaussian_nll(
            rate.new_tensor(value), rate, rate.new_tensor(1.0), tail_sigma=0.5
        )
    with torch.no_grad():
        torch.testing.assert_close(
            poisson_gaussian_nll(
                rate.new_tensor(value), rate, rate.new_tensor(1.0), tail_sigma=0.5
            ),
            rate.new_tensor(_scipy_finite_nll(value, float(rate), np.arange(2))),
        )


def test_shifted_power_intermediate_overflow_fails_closed_for_finite_final_curvature():
    r = torch.tensor(1.2e-19, requires_grad=True)
    with pytest.raises(ValueError, match="modal derivative intermediates"):
        poisson_gaussian_nll(r.new_tensor(50.0), r, r.new_tensor(1.0), tail_sigma=2.0)
    with torch.no_grad():
        torch.testing.assert_close(
            poisson_gaussian_nll(
                r.new_tensor(50.0), r, r.new_tensor(1.0), tail_sigma=2.0
            ),
            r.new_tensor(_scipy_finite_nll(50.0, float(r), np.arange(7))),
        )
    wider = r.detach().double().requires_grad_()
    loss = poisson_gaussian_nll(
        wider.new_tensor(50.0), wider, wider.new_tensor(1.0), tail_sigma=2.0
    )
    score = torch.autograd.grad(loss, wider, create_graph=True)[0]
    curvature = torch.autograd.grad(score, wider)[0]
    counts = np.arange(7)
    logits = counts * math.log(float(wider.detach())) - gammaln(counts + 1)
    prior = np.exp(logits - logsumexp(logits))
    joint = logits + norm.logpdf(50.0 - counts)
    post = np.exp(joint - logsumexp(joint))
    prior_mean, post_mean = prior @ counts, post @ counts
    variance_prior = prior @ (counts - prior_mean) ** 2
    variance_post = post @ (counts - post_mean) ** 2
    expected = (variance_prior - variance_post + post_mean - prior_mean) / float(
        wider.detach()
    ) ** 2
    assert expected < torch.finfo(torch.float32).max
    torch.testing.assert_close(
        curvature, wider.new_tensor(expected), rtol=3e-12, atol=0
    )


@pytest.mark.parametrize(
    "dtype,variance,gain", ((torch.float32, 1e38, 1e20), (torch.float64, 1e308, 1e155))
)
def test_standardization_prevents_intermediate_square_overflow(dtype, variance, gain):
    rate = torch.tensor(0.2, dtype=dtype, requires_grad=True)
    v, g = rate.new_tensor(variance), rate.new_tensor(gain)
    loss = poisson_gaussian_nll(rate.new_tensor(0.0), rate, v, gain=g, tail_sigma=0.5)
    score = torch.autograd.grad(loss, rate, create_graph=True)[0]
    curvature = torch.autograd.grad(score, rate)[0]
    ratio = math.exp(-0.5 * (float(g) / math.sqrt(float(v))) ** 2)
    r = float(rate.detach())
    expected = 0.5 * math.log(2 * math.pi) + 0.5 * math.log(float(v))
    expected += math.log1p(r) - math.log1p(r * ratio)
    expected_score = (1 - ratio) / ((1 + r) * (1 + r * ratio))
    expected_curvature = (ratio / (1 + r * ratio)) ** 2 - 1 / (1 + r) ** 2
    for a, b in (
        (loss, expected),
        (score, expected_score),
        (curvature, expected_curvature),
    ):
        torch.testing.assert_close(
            a,
            rate.new_tensor(b),
            rtol=3e-5 if dtype == torch.float32 else 3e-13,
            atol=0,
        )


@pytest.mark.parametrize(
    "dtype,rate", ((torch.float32, 1e20), (torch.float64, 1e20), (torch.float64, 1e16))
)
def test_support_rejects_before_integer_cast_aliases_counts(dtype, rate):
    for function in (poisson_gaussian_nll, poisson_gaussian_cdf):
        with pytest.raises(ValueError, match="support.*count range"):
            function(
                torch.tensor(0.0, dtype=dtype),
                torch.tensor(rate, dtype=dtype),
                torch.tensor(1.0, dtype=dtype),
            )


@pytest.mark.parametrize(
    "dtype,tail_sigma", ((torch.float32, 1e30), (torch.float64, 1e200))
)
def test_support_radius_rejects_before_scalar_square_overflows(dtype, tail_sigma):
    for function in (poisson_gaussian_nll, poisson_gaussian_cdf):
        with pytest.raises(ValueError, match="support radius"):
            function(
                torch.tensor(0.0, dtype=dtype),
                torch.tensor(0.2, dtype=dtype),
                torch.tensor(1.0, dtype=dtype),
                tail_sigma=tail_sigma,
            )
