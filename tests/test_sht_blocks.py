"""Pure-torch scalar Legendre transforms (sphere._sht_blocks) against the native kernels."""

from __future__ import annotations

import math

import pytest
import torch

from torchsky import _ops
from torchsky.sphere import _sht_blocks as blocks
from torchsky.sphere._spectral_basis import _ring_layout_for_nside
from torchsky.sphere.spectral import alm_size, alm2map, map2alm

pytestmark = pytest.mark.skipif(
    not _ops.has_spectral_dispatcher(), reason="native spectral kernels unavailable"
)


def _inputs(nside: int, nmaps: int = 2):
    lmax = mmax = 2 * nside
    _, _, theta, _ = _ring_layout_for_nside(nside)
    gen = torch.Generator().manual_seed(nside)
    nrings = int(theta.numel())
    modes = torch.complex(
        torch.randn(nmaps, mmax + 1, nrings, generator=gen, dtype=torch.float64),
        torch.randn(nmaps, mmax + 1, nrings, generator=gen, dtype=torch.float64),
    )
    alm = torch.complex(
        torch.randn(nmaps, alm_size(lmax, mmax), generator=gen, dtype=torch.float64),
        torch.randn(nmaps, alm_size(lmax, mmax), generator=gen, dtype=torch.float64),
    )
    return lmax, mmax, theta, modes, alm


def _native(theta, modes, alm, lmax, mmax, pix_w):
    weight = torch.tensor([pix_w], dtype=torch.float64)
    a = torch.stack(
        [
            _ops.spectral_scalar_integrate_recurrence(m, theta, weight, lmax, mmax)
            for m in modes
        ]
    )
    f = torch.stack(
        [_ops.spectral_scalar_interpolate_recurrence(x, theta, lmax, mmax) for x in alm]
    )
    return a, f


def _rel(x: torch.Tensor, ref: torch.Tensor) -> float:
    return float((x - ref).abs().max() / ref.abs().max())


@pytest.mark.parametrize("nside", [2, 8, 32])
def test_recurrence_and_table_match_native_float64(nside: int) -> None:
    lmax, mmax, theta, modes, alm = _inputs(nside)
    pix_w = 4.0 * math.pi / (12 * nside * nside)
    ref_a, ref_f = _native(theta, modes, alm, lmax, mmax, pix_w)
    rec_a = blocks._integrate_recurrence(
        modes, theta, lmax=lmax, mmax=mmax, pix_w=pix_w
    )
    rec_f = blocks._synthesize_recurrence(alm, theta, lmax=lmax, mmax=mmax)
    table = blocks._LegendreTable(theta, lmax, mmax, torch.device("cpu"), torch.float64)
    tab_a = blocks._integrate_table(modes, table, pix_w)
    tab_f = blocks._synthesize_table(alm, table)
    for got, ref in ((rec_a, ref_a), (rec_f, ref_f), (tab_a, ref_a), (tab_f, ref_f)):
        assert _rel(got, ref) < 1e-12


def test_float32_table_is_built_in_float64() -> None:
    # The float32 table is rounded once from float64 values, so its error is
    # float32 rounding (~1e-7), not an accumulated float32 recurrence.
    nside = 32
    lmax, mmax, theta, modes, alm = _inputs(nside)
    pix_w = 4.0 * math.pi / (12 * nside * nside)
    ref_a, ref_f = _native(theta, modes, alm, lmax, mmax, pix_w)
    table = blocks._LegendreTable(theta, lmax, mmax, torch.device("cpu"), torch.float32)
    got_a = blocks._integrate_table(modes.to(torch.complex64), table, pix_w)
    got_f = blocks._synthesize_table(alm.to(torch.complex64), table)
    assert _rel(got_a.to(torch.complex128), ref_a) < 2e-6
    assert _rel(got_f.to(torch.complex128), ref_f) < 2e-6


def test_table_over_budget_falls_back_to_the_recurrence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nside = 8
    lmax, mmax, theta, modes, alm = _inputs(nside)
    monkeypatch.setenv("TORCHSKY_SHT_TABLE", "1")
    monkeypatch.setenv("TORCHSKY_SHT_TABLE_BYTES", "1")
    blocks._TABLE_CACHE.clear()
    assert (
        blocks._legendre_table(theta, lmax, mmax, torch.device("cpu"), torch.float64)
        is None
    )
    got = blocks._scalar_ring_legendre_integrate(
        modes, theta, lmax=lmax, mmax=mmax, pix_w=1.0
    )
    ref = blocks._integrate_recurrence(modes, theta, lmax=lmax, mmax=mmax, pix_w=1.0)
    assert torch.equal(got, ref)
    assert not blocks._TABLE_CACHE


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("budget", ["auto", "1"])
def test_mps_scalar_transforms_match_cpu(
    budget: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "auto": float32 table on the device; "1" byte: native float64 on the host.
    if budget != "auto":
        monkeypatch.setenv("TORCHSKY_SHT_TABLE_BYTES", budget)
    blocks._TABLE_CACHE.clear()
    nside, lmax = 32, 64
    gen = torch.Generator().manual_seed(5)
    sky = torch.randn(12 * nside * nside, generator=gen, dtype=torch.float64)
    ref_alm = map2alm(sky, lmax=lmax, mmax=lmax)
    ref_map = alm2map(ref_alm, nside, lmax=lmax, mmax=lmax)
    got_alm = map2alm(sky.to(torch.float32).to("mps"), lmax=lmax, mmax=lmax)
    got_map = alm2map(
        ref_alm.to(torch.complex64).to("mps"), nside, lmax=lmax, mmax=lmax
    )
    assert got_alm.device.type == "mps" and got_map.device.type == "mps"
    assert _rel(got_alm.cpu().to(torch.complex128), ref_alm) < 1e-5
    assert _rel(got_map.cpu().to(torch.float64), ref_map) < 1e-5


@pytest.mark.parametrize("nside", [1, 2, 8, 32])
def test_bluestein_polar_rings_match_per_length_ffts(
    nside: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torchsky.sphere import _spectral_ring_fourier as rf

    starts, lengths, _, _ = _ring_layout_for_nside(nside)
    npix, mmax = 12 * nside * nside, 3 * nside
    gen = torch.Generator().manual_seed(nside)
    rows = torch.complex(
        torch.randn(2, npix, generator=gen, dtype=torch.float64),
        torch.randn(2, npix, generator=gen, dtype=torch.float64),
    ).requires_grad_(True)  # requires_grad keeps the native CPU kernels out
    modes_in = torch.complex(
        torch.randn(
            2, mmax + 1, int(lengths.numel()), generator=gen, dtype=torch.float64
        ),
        torch.randn(
            2, mmax + 1, int(lengths.numel()), generator=gen, dtype=torch.float64
        ),
    ).requires_grad_(True)
    out = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("TORCHSKY_RING_BLUESTEIN", mode)
        out[mode] = (
            rf._ring_fourier_modes(
                rows, starts_cpu=starts, lengths_cpu=lengths, mmax=mmax, nside=nside
            ),
            rf._ring_fourier_synthesis(
                modes_in, starts_cpu=starts, lengths_cpu=lengths, nside=nside
            ),
        )
    for got, ref in zip(out["1"], out["0"]):
        assert _rel(got.detach(), ref.detach()) < 1e-13
    # Differentiable: the gradient of a real functional agrees with the loop.
    grads = []
    for mode in ("0", "1"):
        monkeypatch.setenv("TORCHSKY_RING_BLUESTEIN", mode)
        val = rf._ring_fourier_modes(
            rows, starts_cpu=starts, lengths_cpu=lengths, mmax=mmax, nside=nside
        )
        (g,) = torch.autograd.grad(val.abs().square().sum(), rows)
        grads.append(g)
    assert _rel(grads[1], grads[0]) < 1e-12
