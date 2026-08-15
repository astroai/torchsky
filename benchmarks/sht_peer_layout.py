"""Packed (healpy/torchsky) ↔ s2fft 2D alm layout. Benches/tests only.

s2fft uses bandlimit ``L = lmax + 1`` and a ``(L, 2L-1)`` complex array with
``flm[ell, L-1+m] = a_{ell,m}``. Torchsky and healpy store packed ``m >= 0``.

Negative-m fill uses the real-scalar conjugation ``a_{l,-m} = (-1)^m conj(a_{l,m})``.
Do not use that fill for spin-weighted fields.
"""

from __future__ import annotations

import numpy as np

from torchsky.sphere.spectral import alm_index, alm_size


def lmax_to_s2fft_L(lmax: int) -> int:
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    return int(lmax) + 1


def packed_alm_to_s2fft(
    alm: np.ndarray, lmax: int, *, reality: bool = True
) -> np.ndarray:
    """Packed healpy/torchsky alm → s2fft ``(L, 2L-1)``."""
    ll = int(lmax)
    L = lmax_to_s2fft_L(ll)
    packed = np.asarray(alm, dtype=np.complex128).reshape(-1)
    nalm = alm_size(ll)
    if packed.size != nalm:
        raise ValueError(f"packed alm length {packed.size} != alm_size({ll})={nalm}")
    out = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    m0 = L - 1
    for m in range(L):
        for ell in range(m, L):
            val = packed[alm_index(ell, m, ll)]
            out[ell, m0 + m] = val
            if reality and m > 0:
                out[ell, m0 - m] = ((-1) ** m) * np.conjugate(val)
    return out


def s2fft_alm_to_packed(flm: np.ndarray, lmax: int) -> np.ndarray:
    """s2fft ``(L, 2L-1)`` → packed healpy/torchsky alm (``m >= 0`` only)."""
    ll = int(lmax)
    L = lmax_to_s2fft_L(ll)
    arr = np.asarray(flm, dtype=np.complex128)
    if arr.shape != (L, 2 * L - 1):
        raise ValueError(f"expected flm shape {(L, 2 * L - 1)}, got {arr.shape}")
    out = np.zeros(alm_size(ll), dtype=np.complex128)
    m0 = L - 1
    for m in range(L):
        for ell in range(m, L):
            out[alm_index(ell, m, ll)] = arr[ell, m0 + m]
    return out
