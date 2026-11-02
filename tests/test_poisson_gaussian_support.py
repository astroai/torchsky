"""Observation-aware Poisson support window, chunking and gradients."""

import numpy as np
import pytest
from scipy.special import gammaln, logsumexp
from scipy.stats import norm
import torch

from torchsky.map.operators import noise
from torchsky.map.operators.noise import poisson_gaussian_cdf, poisson_gaussian_nll


def _true_nll(value, rate, var, gain, bg, n_max=4000):
    counts = np.arange(n_max)
    log_p = counts * np.log(rate) - rate - gammaln(counts + 1)
    log_g = norm.logpdf(value, bg + gain * counts, np.sqrt(var))
    return -logsumexp(log_p + log_g)


def _true_cdf(value, rate, var, gain, bg, n_max=4000):
    counts = np.arange(n_max)
    log_p = counts * np.log(rate) - rate - gammaln(counts + 1)
    return float(np.exp(log_p) @ norm.cdf(value, bg + gain * counts, np.sqrt(var)))


@pytest.mark.parametrize(
    "rate,value,var,gain",
    [
        (100.0, 100.0, 4.0, 1.0),
        (100.0, 400.0, 4.0, 1.0),  # cosmic-ray-like upward outlier
        (100.0, 700.0, 25.0, 2.0),
        (30.0, -50.0, 9.0, 1.0),
        (1e-3, 60.0, 1.0, 1.0),
        (1e3, 3000.0, 16.0, 1.5),
    ],
)
def test_outlier_nll_matches_untruncated_mixture(rate, value, var, gain):
    r = torch.tensor(rate, dtype=torch.float64)
    got = poisson_gaussian_nll(
        r.new_tensor(value), r, r.new_tensor(var), gain=gain, background=0.0
    )
    want = _true_nll(value, rate, var, gain, 0.0)
    assert float(got) == pytest.approx(want, rel=1e-10, abs=1e-8)


@pytest.mark.parametrize("value", (-60.0, 40.0, 100.0, 160.0, 400.0))
def test_cdf_matches_untruncated_mixture(value):
    rate, var = 100.0, 4.0
    r = torch.tensor(rate, dtype=torch.float64)
    got = float(poisson_gaussian_cdf(r.new_tensor(value), r, r.new_tensor(var)))
    want = _true_cdf(value, rate, var, 1.0, 0.0)
    assert got == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_chunking_is_exact_and_matches_single_chunk(monkeypatch):
    g = torch.Generator().manual_seed(0)
    rate = torch.rand(517, generator=g, dtype=torch.float64) * 300
    value = rate + torch.randn(517, generator=g, dtype=torch.float64) * 20
    value[::50] += 500.0  # outliers
    var = torch.full_like(rate, 9.0)
    whole = poisson_gaussian_nll(value, rate, var, reduction="none")
    cdf_whole = poisson_gaussian_cdf(value, rate, var)
    monkeypatch.setattr(noise, "_POISSON_GAUSSIAN_CHUNK_BYTES", 400_000)
    assert noise._chunk_plan(rate, 12.0, False)[0] < 10
    torch.testing.assert_close(
        poisson_gaussian_nll(value, rate, var, reduction="none"), whole, rtol=0, atol=0
    )
    torch.testing.assert_close(
        poisson_gaussian_cdf(value, rate, var), cdf_whole, rtol=0, atol=0
    )


def test_gradcheck_including_far_rows_and_checkpointing(monkeypatch):
    monkeypatch.setattr(noise, "_POISSON_GAUSSIAN_CHUNK_BYTES", 50_000)
    monkeypatch.setattr(noise, "_POISSON_GAUSSIAN_SAVE_BYTES", 0)
    rate = torch.tensor([5.0, 40.0, 40.0], dtype=torch.float64, requires_grad=True)
    value = torch.tensor([6.0, 41.0, 160.0], dtype=torch.float64)  # last is far
    var = torch.tensor([2.0, 3.0, 3.0], dtype=torch.float64, requires_grad=True)
    assert bool(
        (
            noise._poisson_gaussian_window(
                value,
                rate.detach(),
                var.detach(),
                torch.ones_like(rate),
                torch.zeros_like(rate),
                tail_sigma=2.0,
                cdf=False,
            )[2]
            != 0
        ).any()
    )
    torch.autograd.gradcheck(
        lambda r, v: poisson_gaussian_nll(
            value, r, v, tail_sigma=2.0, reduction="none"
        ),
        (rate, var),
    )
    torch.autograd.gradcheck(
        lambda r, v: poisson_gaussian_cdf(value - 150, r, v, tail_sigma=2.0),
        (rate, var),
    )


def test_prior_support_beyond_count_range_still_rejects():
    with pytest.raises(ValueError, match="support.*count range"):
        poisson_gaussian_nll(torch.tensor(0.0), torch.tensor(1e20), torch.tensor(1.0))


def _streaming_inputs(dtype=torch.float64):
    y = torch.tensor([3.0, -60.0, 400.0, 0.2, 1.5, 5.0], dtype=dtype)
    r = torch.tensor([2.5, 5.0, 3.0, 1e-3, 0.7, 0.0], dtype=dtype)
    v = torch.tensor([1.5, 2.0, 1.0, 0.8, 1.2, 1.0], dtype=dtype)
    g = torch.tensor([1.1, 0.9, 1.0, 1.2, 1.0, 1.3], dtype=dtype)
    b = torch.tensor([0.1, 0.0, -0.2, 0.0, 0.3, 0.0], dtype=dtype)
    return [t.clone().requires_grad_() for t in (y, r, v, g, b)]


@pytest.mark.parametrize("cdf", (False, True))
def test_streaming_gradients_match_modal_path_and_gradcheck(cdf):
    from torchsky.map.operators import noise

    def run(y, r, v, g, b):
        if cdf:
            return noise.poisson_gaussian_cdf(y, r, v, gain=g, background=b)
        return noise.poisson_gaussian_nll(
            y, r, v, gain=g, background=b, reduction="none"
        )

    args = _streaming_inputs()
    out = run(*args)
    weight = torch.linspace(0.5, 1.5, out.numel(), dtype=out.dtype)
    got = torch.autograd.grad((out * weight).sum(), args)
    modal = (
        noise._poisson_gaussian_cdf_modal_path
        if cdf
        else noise._poisson_gaussian_nll_modal_path
    )(tuple(args), 12.0, 4096)
    ref = torch.autograd.grad((modal * weight).sum(), args)
    torch.testing.assert_close(out.detach(), modal.detach(), rtol=1e-11, atol=0)
    for a, e in zip(got, ref):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-12)
    # gradcheck needs strictly positive rates (finite differences cross zero).
    pos = [t.detach().clone() for t in args]
    pos[1][-1] = 0.4
    pos = [t.requires_grad_() for t in pos]
    assert torch.autograd.gradcheck(run, pos, eps=1e-7, atol=1e-6)
    assert torch.autograd.gradgradcheck(run, pos, eps=1e-7, atol=1e-5, rtol=1e-4)
