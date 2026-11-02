"""Autograd oracle for the finite Poisson--Gaussian law on an explicit window."""

import torch

from torchsky.map.operators.noise import _poisson_gaussian_window


def window(value, rate, var, gain, bg, tail_sigma, *, cdf=False):
    """Library window ``(lower, upper, side)`` as Python ints."""
    args = [
        torch.as_tensor(float(x), dtype=torch.float64).reshape(1)
        for x in (value, rate, var, gain, bg)
    ]
    lower, upper, side = _poisson_gaussian_window(*args, tail_sigma=tail_sigma, cdf=cdf)
    return int(lower), int(upper), int(side)


def _log_weights(rate, lower, upper, side):
    counts = torch.arange(lower, upper + 1, dtype=rate.dtype)
    if float(rate.detach()) < 1.0:
        # Polynomial form: no tiny-rate logarithmic curvature cancellation.
        weight = rate.pow(counts) / torch.exp(torch.lgamma(counts + 1.0))
        log_p = weight.log()
        log_p = log_p - rate if side else log_p - weight.sum().log()
        return counts, log_p
    log_p = counts * rate.log() - torch.lgamma(counts + 1.0)
    # Near rows renormalise over the window; far rows use the exact pmf.
    log_p = log_p - rate if side else log_p - torch.logsumexp(log_p, 0)
    return counts, log_p


def law_nll(value, rate, var, gain, bg, lower, upper, side):
    counts, log_p = _log_weights(rate, lower, upper, side)
    dist = torch.distributions.Normal(bg + gain * counts, var.sqrt())
    return -torch.logsumexp(log_p + dist.log_prob(value), 0)


def law_cdf(value, rate, var, gain, bg, lower, upper, side):
    counts, log_p = _log_weights(rate, lower, upper, side)
    z = (value - bg - gain * counts) / var.sqrt()
    return (log_p.exp() * 0.5 * torch.erfc(-z / 2**0.5)).sum()
