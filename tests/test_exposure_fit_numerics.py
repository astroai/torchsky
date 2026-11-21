"""Numerical contracts of the moved renderer pieces: the FFT correlation against direct convolution and its adjoint; item sky flux and gradients."""

import math

import torch
import torch.nn.functional as F

from torchsky.map.exposure_fit._items import ItemSky, RBFBasis
from torchsky.map.exposure_fit._render import correlate_same


def test_correlate_same_matches_direct_convolution():
    g = torch.Generator().manual_seed(0)
    img = torch.randn(2, 21, 17, generator=g, dtype=torch.float64)
    ker = torch.randn(2, 7, 5, generator=g, dtype=torch.float64)
    out = correlate_same(img, ker)
    ref = torch.stack(
        [
            F.conv2d(img[i][None, None], ker[i][None, None], padding=(3, 2))[0, 0]
            for i in range(2)
        ]
    )
    assert torch.allclose(out, ref, rtol=0, atol=1e-11)


def test_correlate_same_adjoint():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(19, 23, generator=g, dtype=torch.float64)
    y = torch.randn(19, 23, generator=g, dtype=torch.float64)
    ker = torch.randn(9, 7, generator=g, dtype=torch.float64)
    lhs = (correlate_same(x, ker) * y).sum()
    rhs = (
        x * correlate_same(y, ker.flip(-2, -1))
    ).sum()  # adjoint of correlation with an odd kernel: correlation with the flipped kernel
    assert abs(float(lhs - rhs)) < 1e-10 * max(1.0, abs(float(lhs)))


def _one_item_sky(N, x, y, a, th, K=3):
    s = ItemSky(
        torch.zeros(K, N, N, dtype=torch.float64), 0.04, RBFBasis(K), 1, n_levels=1
    )
    with torch.no_grad():
        s.pos.data[0, 0] = torch.tensor([x, y], dtype=torch.float64)
        fl = s.floor
        s.lw.data[0, 0] = torch.full(
            (2,), 0.5 * math.log(a * a - fl * fl), dtype=torch.float64
        )
        s.ang.data[0, 0] = th
        s.lc.data[0, 0] = 0.0
        s._mut += 1
    return s


def test_item_flux_conserved():
    N, p = 61, 0.04
    for x, y in [(0.0, 0.0), (0.01, 0.02), (0.3, -0.2)]:
        s = _one_item_sky(N, x, y, 0.12, 0.5)
        A = s.coeffs()
        assert abs(float(p * p * A[0].detach().sum()) - 1.0) < 1e-6


def test_item_gradients_against_finite_differences():
    N = 41
    s = _one_item_sky(N, 0.05, -0.03, 0.15, 0.4)
    w = torch.randn(
        3, N, N, generator=torch.Generator().manual_seed(2), dtype=torch.float64
    )

    def val():
        s._mut += 1
        return (s.coeffs() * w).sum()

    for par in (s.pos, s.lw, s.ang, s.lc):
        par.grad = None
    val().backward()
    for par in (s.pos, s.lw, s.ang, s.lc):
        g = par.grad.reshape(-1).clone()
        for j in range(min(3, g.numel())):
            h = 1e-6
            with torch.no_grad():
                par.data.reshape(-1)[j] += h
            fp = float(val())
            with torch.no_grad():
                par.data.reshape(-1)[j] -= 2 * h
            fm = float(val())
            with torch.no_grad():
                par.data.reshape(-1)[j] += h
            fd = (fp - fm) / (2 * h)
            assert abs(fd - float(g[j])) < 1e-5 * max(1.0, abs(fd))
