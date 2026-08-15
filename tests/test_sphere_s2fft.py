"""Optional s2fft HEALPix SHT peer. healpy remains the science oracle.

Native ``sampling="healpix"`` only. ``method="jax_healpy"`` wraps healpy and
must not be treated as a second oracle. HEALPix has no sampling theorem, so
torchsky vs s2fft residuals are allowed to be looser than torchsky vs healpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.sht_peer_layout import (  # noqa: E402
    lmax_to_s2fft_L,
    packed_alm_to_s2fft,
    s2fft_alm_to_packed,
)
from torchsky.sphere import healpix as hp  # noqa: E402
from torchsky.sphere.spectral import (  # noqa: E402
    alm2map,
    alm2map_spin,
    alm_index,
    alm_size,
    map2alm,
    map2alm_spin,
)

# Independent HEALPix implementations need not match at 1e-10.
_S2FFT_VS_TORCHSKY_ATOL = 5e-2
_NATIVE_S2FFT_METHODS = ("numpy", "jax", "torch")


def _random_packed_alm(lmax: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.zeros(alm_size(lmax), dtype=np.complex128)
    for m in range(lmax + 1):
        for ell in range(m, lmax + 1):
            scale = 1.0 / ((ell + 1) ** 2)
            if m == 0:
                val = scale * rng.normal()
            else:
                val = scale * (rng.normal() + 1j * rng.normal())
            arr[alm_index(ell, m, lmax)] = val
    return arr


def test_packed_s2fft_layout_roundtrip() -> None:
    lmax = 8
    packed = _random_packed_alm(lmax, seed=7)
    flm = packed_alm_to_s2fft(packed, lmax, reality=True)
    back = s2fft_alm_to_packed(flm, lmax)
    np.testing.assert_allclose(back, packed, atol=1e-15, rtol=0.0)
    assert flm.shape == (lmax + 1, 2 * lmax + 1)
    # m=0 column is the packed m=0 block.
    m0 = lmax
    for ell in range(lmax + 1):
        assert flm[ell, m0] == packed[alm_index(ell, 0, lmax)]


def test_s2fft_bandlimit_is_lmax_plus_one() -> None:
    assert lmax_to_s2fft_L(0) == 1
    assert lmax_to_s2fft_L(31) == 32


def _s2fft_native(kind: str, arr, *, L: int, nside: int, spin: int, reality: bool):
    s2fft = pytest.importorskip("s2fft")
    fn = s2fft.forward if kind == "forward" else s2fft.inverse
    last_exc: Exception | None = None
    for method in _NATIVE_S2FFT_METHODS:
        try:
            return fn(
                arr,
                L,
                nside=nside,
                sampling="healpix",
                method=method,
                spin=spin,
                reality=reality,
            )
        except Exception as exc:
            last_exc = exc
    pytest.skip(f"native s2fft HEALPix {kind} unavailable: {last_exc}")


def test_s2fft_layout_matches_upstream_helper() -> None:
    s2fft = pytest.importorskip("s2fft")
    samples = getattr(getattr(s2fft, "sampling", None), "s2_samples", None)
    if samples is None or not hasattr(samples, "flm_hp_to_2d"):
        pytest.skip("s2fft.sampling.s2_samples.flm_hp_to_2d missing")
    lmax = 6
    packed = _random_packed_alm(lmax, seed=11)
    flm = packed_alm_to_s2fft(packed, lmax, reality=True)
    ref = np.asarray(samples.flm_hp_to_2d(packed, lmax + 1))
    np.testing.assert_allclose(flm, ref, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("nside", [8, 16])
def test_s2fft_scalar_roundtrip_vs_torchsky(nside: int) -> None:
    pytest.importorskip("s2fft")
    lmax = nside
    L = lmax + 1
    rng = np.random.default_rng(21 + nside)
    npix = hp.nside2npix(nside)
    m_np = rng.normal(size=npix).astype(np.float64)
    m_t = torch.from_numpy(m_np)

    flm = _s2fft_native("forward", m_np, L=L, nside=nside, spin=0, reality=True)
    packed_s2 = s2fft_alm_to_packed(np.asarray(flm), lmax)
    packed_ts = map2alm(m_t, nside=nside, lmax=lmax).detach().cpu().numpy()
    denom = max(float(np.linalg.norm(packed_ts)), 1e-12)
    rel = float(np.linalg.norm(packed_s2 - packed_ts) / denom)
    assert rel < _S2FFT_VS_TORCHSKY_ATOL

    rec_s2 = np.asarray(
        _s2fft_native("inverse", flm, L=L, nside=nside, spin=0, reality=True)
    ).reshape(-1)
    rec_ts = (
        alm2map(torch.from_numpy(packed_ts), nside=nside, lmax=lmax)
        .detach()
        .cpu()
        .numpy()
    )
    rec_denom = max(float(np.linalg.norm(rec_ts)), 1e-12)
    rec_rel = float(np.linalg.norm(rec_s2.real - rec_ts) / rec_denom)
    assert rec_rel < _S2FFT_VS_TORCHSKY_ATOL


@pytest.mark.parametrize("nside", [8, 16])
def test_s2fft_spin_roundtrip_vs_torchsky(nside: int) -> None:
    pytest.importorskip("s2fft")
    lmax = nside
    L = lmax + 1
    spin = 2
    rng = np.random.default_rng(33 + nside)
    npix = hp.nside2npix(nside)
    qu = rng.normal(size=(2, npix)).astype(np.float64)
    f = qu[0] + 1j * qu[1]

    flm = _s2fft_native("forward", f, L=L, nside=nside, spin=spin, reality=False)
    rec = np.asarray(
        _s2fft_native("inverse", flm, L=L, nside=nside, spin=spin, reality=False)
    ).reshape(-1)
    ae, ab = map2alm_spin(torch.from_numpy(qu), spin=spin, nside=nside, lmax=lmax)
    qu_ts = alm2map_spin((ae, ab), nside=nside, spin=spin, lmax=lmax)
    rec_ts = qu_ts[0].detach().cpu().numpy() + 1j * qu_ts[1].detach().cpu().numpy()
    denom = max(float(np.linalg.norm(rec_ts)), 1e-12)
    rel = float(np.linalg.norm(rec - rec_ts) / denom)
    assert rel < _S2FFT_VS_TORCHSKY_ATOL


def test_s2fft_vs_healpy_residual_is_allowed() -> None:
    """Document that native s2fft HEALPix need not match healpy at 1e-10."""
    pytest.importorskip("s2fft")
    healpy = pytest.importorskip("healpy")
    nside = 8
    lmax = 8
    L = lmax + 1
    rng = np.random.default_rng(41)
    m_np = rng.normal(size=hp.nside2npix(nside)).astype(np.float64)
    flm = _s2fft_native("forward", m_np, L=L, nside=nside, spin=0, reality=True)
    packed_s2 = s2fft_alm_to_packed(np.asarray(flm), lmax)
    packed_hp = healpy.map2alm(m_np, lmax=lmax, iter=0)
    packed_ts = (
        map2alm(torch.from_numpy(m_np), nside=nside, lmax=lmax).detach().cpu().numpy()
    )
    rel_oracle = float(
        np.linalg.norm(packed_ts - packed_hp)
        / max(float(np.linalg.norm(packed_hp)), 1e-12)
    )
    rel_peer = float(
        np.linalg.norm(packed_s2 - packed_hp)
        / max(float(np.linalg.norm(packed_hp)), 1e-12)
    )
    assert rel_oracle < 1.0e-8
    assert np.isfinite(rel_peer)
