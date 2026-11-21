"""Device-independent scalar Legendre transforms for HEALPix rings (pure torch).

This is the path every device without the native CPU kernels takes (CUDA,
MPS, CPU builds without the extension). It computes, for the ring Fourier
modes ``F_m(r)`` of a map,

    map2alm:  a_lm   = w * sum_r lambda_lm(theta_r) F_m(r)
    alm2map:  F_m(r) = sum_l a_lm lambda_lm(theta_r)

with ``lambda_lm`` the orthonormal associated Legendre functions, evaluated
by the three-term recurrence in ``l``. The layout follows libsharp/ducc and
torchsky's native kernels, but is expressed as whole-array tensor programs:

* North/south ring pairs are folded: ``lambda_lm(pi - theta) =
  (-1)^(l+m) lambda_lm(theta)``, so a pair contributes through its even part
  ``F(r) + F(r')`` or odd part ``F(r) - F(r')`` and only half the rings are
  recurred (the equator ring, when present, enters both parts once).
* ``m`` is processed in fixed-width blocks whose ``l`` loop starts at the
  block's first ``m``; rows with ``m > l`` stay zero until their diagonal, so
  the triangular ``m <= l`` structure costs at most half a block of extra rows.
* Each call advances ``K`` recurrence steps for a whole block and folds the
  per-ring sums into the same call. All shapes are fixed (the last block is
  padded with inert rows, ``l`` enters as a tensor), so on CUDA the step
  function compiles once into a fused kernel per call instead of ~10 kernels
  per ``l`` step; eager execution runs the identical code.
* Underflow of the sectoral seed ``lambda_mm ~ sin(theta)^m`` is handled as in
  libsharp: each (m, ring) carries an integer exponent in units of ``2^S``
  (S = 256 in float64, 64 in float32, which has no wider exponent range), the
  recurrence runs on the mantissa, and a value contributes only once its
  exponent reaches 0 (anything still scaled is below ``2^-S/2`` of O(1)).
"""

from __future__ import annotations

import math
import os
from typing import Callable

import torch
from torch import Tensor

from ..device import _to

_K_STEPS = 16  # recurrence steps per call
_BLOCK_ELEMS = 1 << 18  # target (m rows x rings) elements per block
_SCALE_BITS = {torch.float64: 256, torch.float32: 64}

_COMPILED: dict[tuple, Callable] = {}


def _use_compile(device: torch.device) -> bool:
    """``TORCHSKY_SHT_COMPILE``: ``auto`` (default: CUDA only), ``1`` or ``0``."""
    mode = os.environ.get("TORCHSKY_SHT_COMPILE", "auto").strip().lower()
    if mode in {"0", "false", "off", "no"}:
        return False
    if mode in {"1", "true", "on", "yes"}:
        return True
    return device.type == "cuda"


def _stepper(fn: Callable, device: torch.device, name: str) -> Callable:
    if not _use_compile(device):
        return fn
    key = (name, device.type)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = torch.compile(fn, dynamic=False, fullgraph=True)
        _COMPILED[key] = compiled
    return compiled


def _recurrence_coefficients(ell: Tensor, m_rows: Tensor) -> tuple[Tensor, Tensor]:
    """``lambda_l = a * cos(theta) * lambda_{l-1} - b * lambda_{l-2}`` for l > m.

    Rows with ``l <= m`` get ``a = b = 0`` (their value is zero below the
    diagonal and seeded on it); ``b`` vanishes at ``l = m + 1`` by itself.
    """
    l2 = ell * ell
    m2 = m_rows * m_rows
    active = ell > m_rows
    den = torch.where(active, l2 - m2, torch.ones_like(m2))
    a = torch.where(active, torch.sqrt((4.0 * l2 - 1.0) / den), torch.zeros_like(m2))
    lm1 = (ell - 1.0) * (ell - 1.0)
    two_l_m3 = torch.where(ell > 1.0, 2.0 * ell - 3.0, torch.ones_like(ell))
    b_arg = (2.0 * ell + 1.0) / two_l_m3 * (lm1 - m2) / den
    b = torch.where(
        active & (ell > m_rows + 1.0),
        torch.sqrt(torch.clamp(b_arg, min=0.0)),
        torch.zeros_like(m2),
    )
    return a, b


def _advance(
    ell: Tensor,
    m_rows: Tensor,
    ct: Tensor,
    lam1: Tensor,
    lam2: Tensor,
    ex: Tensor,
    seed_mant: Tensor,
    seed_ex: Tensor,
    thr: float,
    inv: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One recurrence step; returns (lambda_l mantissa, rescaled lambda_{l-1}, ex, value)."""
    a, b = _recurrence_coefficients(ell, m_rows)
    new = a[:, None] * ct[None, :] * lam1 - b[:, None] * lam2
    diag = (ell == m_rows)[:, None]
    new = torch.where(diag, seed_mant, new)
    ex = torch.where(diag, seed_ex, ex)
    big = new.abs() > thr
    new = torch.where(big, new * inv, new)
    lam1 = torch.where(big, lam1 * inv, lam1)
    ex = ex + big.to(ex.dtype)
    value = torch.where(ex == 0.0, new, torch.zeros_like(new))
    return new, lam1, ex, value


def _integrate_steps(
    ell0: Tensor,
    m_rows: Tensor,
    ct: Tensor,
    lam1: Tensor,
    lam2: Tensor,
    ex: Tensor,
    seed_mant: Tensor,
    seed_ex: Tensor,
    w_even: Tensor,
    w_odd: Tensor,
    thr: float,
    inv: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """K steps of the map2alm recurrence; returns sums (K, rows, channels)."""
    outs = []
    for k in range(_K_STEPS):
        ell = ell0 + float(k)
        new, lam1, ex, value = _advance(
            ell, m_rows, ct, lam1, lam2, ex, seed_mant, seed_ex, thr, inv
        )
        even = (torch.remainder(ell + m_rows, 2.0) == 0.0)[:, None, None]
        # weights are (rows, channels, rings): each (row, channel) sum runs over
        # a contiguous ring axis, identical whatever the number of maps.
        weights = torch.where(even, w_even, w_odd)
        outs.append((value[:, None, :] * weights).sum(dim=-1))
        lam2, lam1 = lam1, new
    return lam1, lam2, ex, torch.stack(outs)


def _synthesize_steps(
    ell0: Tensor,
    m_rows: Tensor,
    ct: Tensor,
    lam1: Tensor,
    lam2: Tensor,
    ex: Tensor,
    seed_mant: Tensor,
    seed_ex: Tensor,
    coeffs: Tensor,
    acc_even: Tensor,
    acc_odd: Tensor,
    thr: float,
    inv: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """K steps of the alm2map recurrence, accumulating even/odd ring sums."""
    for k in range(_K_STEPS):
        ell = ell0 + float(k)
        new, lam1, ex, value = _advance(
            ell, m_rows, ct, lam1, lam2, ex, seed_mant, seed_ex, thr, inv
        )
        even = (torch.remainder(ell + m_rows, 2.0) == 0.0)[:, None, None]
        contrib = value[:, :, None] * coeffs[k][:, None, :]
        acc_even = acc_even + torch.where(even, contrib, torch.zeros_like(contrib))
        acc_odd = acc_odd + torch.where(even, torch.zeros_like(contrib), contrib)
        lam2, lam1 = lam1, new
    return lam1, lam2, ex, acc_even, acc_odd


class _RingPairs:
    """North/south folding of the HEALPix rings (theta symmetric about pi/2)."""

    def __init__(self, theta_ring: Tensor, device: torch.device, dtype: torch.dtype):
        th = _to(theta_ring, device="cpu", dtype=torch.float64)
        nrings = int(th.numel())
        self.nrings = nrings
        self.n_pairs = nrings // 2
        self.has_equator = nrings % 2 == 1
        self.n_eff = self.n_pairs + int(self.has_equator)
        th_eff = th[: self.n_eff]
        self.theta = th_eff
        self.log_sin = torch.log(torch.clamp(torch.sin(th_eff), min=1e-300))
        self.ct = _to(torch.cos(th_eff), device=device, dtype=dtype)

    def fold(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(M, nrings, C) -> even, odd parts (M, n_eff, C)."""
        north = x[:, : self.n_pairs]
        south = x[:, self.nrings - self.n_pairs :].flip(1)
        even = north + south
        odd = north - south
        if self.has_equator:
            eq = x[:, self.n_pairs : self.n_pairs + 1]
            even = torch.cat([even, eq], dim=1)
            odd = torch.cat([odd, eq], dim=1)
        return even, odd

    def unfold(self, even: Tensor, odd: Tensor) -> Tensor:
        """Even/odd ring sums (M, n_eff, C) -> values on all rings (M, nrings, C)."""
        north = even[:, : self.n_pairs] + odd[:, : self.n_pairs]
        south = (even[:, : self.n_pairs] - odd[:, : self.n_pairs]).flip(1)
        parts = [north]
        if self.has_equator:
            parts.append(even[:, self.n_pairs :] + odd[:, self.n_pairs :])
        parts.append(south)
        return torch.cat(parts, dim=1)


def _block_rows(n_eff: int, mmax: int) -> int:
    rows = max(16, min(_BLOCK_ELEMS // max(n_eff, 1), mmax + 1))
    return int(rows)


def _seeds(
    pairs: _RingPairs,
    m_rows_cpu: Tensor,
    mmax: int,
    scale_bits: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Sectoral seeds lambda_mm(theta) = c_m (-sin theta)^m as mantissa * 2^(S*ex)."""
    mv = m_rows_cpu
    valid = mv <= mmax
    mv_safe = torch.where(valid, mv, torch.zeros_like(mv))
    log_c = 0.5 * (
        torch.log(2 * mv_safe + 1)
        - math.log(4.0 * math.pi)
        + torch.lgamma(2 * mv_safe + 1)
        - 2 * (mv_safe * math.log(2.0) + torch.lgamma(mv_safe + 1))
    )
    seed_log = log_c[:, None] + mv_safe[:, None] * pairs.log_sin[None, :]
    ln_scale = scale_bits * math.log(2.0)
    ex = torch.clamp(torch.floor(seed_log / ln_scale), max=0.0)
    mant = torch.exp(seed_log - ex * ln_scale)
    sign = torch.where(torch.remainder(mv_safe, 2.0) == 0.0, 1.0, -1.0)
    mant = mant * sign[:, None]
    mant = torch.where(valid[:, None], mant, torch.zeros_like(mant))
    return _to(mant, device=device, dtype=dtype), _to(ex, device=device, dtype=dtype)


def _alm_base(lmax: int, mmax: int) -> Tensor:
    m = torch.arange(mmax + 1, dtype=torch.int64)
    return m * (lmax + 1) - m * (m - 1) // 2 - m


def _block_index(
    m0: int, rows: int, lmax: int, mmax: int, length: int, base: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """For block rows m0..m0+rows-1 and l = m0 + t (t < length): valid mask and alm index."""
    t = torch.arange(length, dtype=torch.int64)[:, None]
    m = torch.arange(m0, m0 + rows, dtype=torch.int64)[None, :]
    ell = m0 + t
    valid = (ell >= m) & (ell <= lmax) & (m <= mmax)
    idx = torch.where(
        valid, base[torch.clamp(m, max=mmax)] + ell, torch.zeros_like(ell)
    )
    return valid, idx, ell


def _integrate_recurrence(
    s_all: Tensor, theta_ring: Tensor, *, lmax: int, mmax: int, pix_w: float
) -> Tensor:
    """alm[b, (l,m)] = pix_w * sum_r s_all[b, m, r] lambda_lm(theta_r), streamed."""
    nmaps = int(s_all.shape[0])
    device = s_all.device
    dtype = s_all.real.dtype
    scale_bits = _SCALE_BITS[dtype]
    thr = 2.0 ** (scale_bits // 2)
    inv = 2.0 ** (-scale_bits)
    pairs = _RingPairs(theta_ring, device, dtype)
    # (M, rings, channels): channels = map x (re, im)
    x = torch.view_as_real(s_all.contiguous()).permute(1, 2, 0, 3)
    x = x.reshape(mmax + 1, pairs.nrings, nmaps * 2)
    w_even, w_odd = pairs.fold(x)
    rows = _block_rows(pairs.n_eff, mmax)
    nalm = int(_alm_base(lmax, mmax)[-1]) + lmax + 1 if mmax >= 0 else 0
    out = torch.zeros((nalm, nmaps * 2), dtype=dtype, device=device)
    base = _alm_base(lmax, mmax)
    step = _stepper(_integrate_steps, device, "integrate")
    channels = nmaps * 2
    for m0 in range(0, mmax + 1, rows):
        m_rows_cpu = torch.arange(m0, m0 + rows, dtype=torch.float64)
        # Padded rows (m > mmax) never reach their diagonal.
        m_rows_cpu = torch.where(
            m_rows_cpu <= mmax, m_rows_cpu, torch.full_like(m_rows_cpu, lmax + 2.0)
        )
        seed_mant, seed_ex = _seeds(pairs, m_rows_cpu, mmax, scale_bits, device, dtype)
        m_rows = _to(m_rows_cpu, device=device, dtype=dtype)
        hi = min(m0 + rows, mmax + 1)
        we = torch.zeros((rows, channels, pairs.n_eff), dtype=dtype, device=device)
        wo = torch.zeros_like(we)
        we[: hi - m0] = w_even[m0:hi].transpose(1, 2)
        wo[: hi - m0] = w_odd[m0:hi].transpose(1, 2)
        length = -(-(lmax - m0 + 1) // _K_STEPS) * _K_STEPS
        sums = torch.empty((length, rows, channels), dtype=dtype, device=device)
        lam1 = torch.zeros((rows, pairs.n_eff), dtype=dtype, device=device)
        lam2 = torch.zeros_like(lam1)
        ex = torch.zeros_like(lam1)
        for t0 in range(0, length, _K_STEPS):
            ell0 = torch.tensor(float(m0 + t0), dtype=dtype, device=device)
            lam1, lam2, ex, chunk = step(
                ell0,
                m_rows,
                pairs.ct,
                lam1,
                lam2,
                ex,
                seed_mant,
                seed_ex,
                we,
                wo,
                thr,
                inv,
            )
            sums[t0 : t0 + _K_STEPS] = chunk
        valid, idx, _ = _block_index(m0, rows, lmax, mmax, length, base)
        flat_valid = valid.reshape(-1)
        out[_to(idx.reshape(-1)[flat_valid], device=device)] = sums.reshape(
            length * rows, channels
        )[_to(flat_valid, device=device)]
    out = (out * pix_w).reshape(nalm, nmaps, 2).permute(1, 0, 2).contiguous()
    return torch.view_as_complex(out)


def _synthesize_recurrence(
    alm: Tensor, theta_ring: Tensor, *, lmax: int, mmax: int
) -> Tensor:
    """f[b, m, r] = sum_l alm[b, (l,m)] lambda_lm(theta_r), streamed."""
    nmaps = int(alm.shape[0])
    device = alm.device
    dtype = alm.real.dtype
    scale_bits = _SCALE_BITS[dtype]
    thr = 2.0 ** (scale_bits // 2)
    inv = 2.0 ** (-scale_bits)
    pairs = _RingPairs(theta_ring, device, dtype)
    channels = nmaps * 2
    a = torch.view_as_real(alm.contiguous()).permute(1, 0, 2).reshape(-1, channels)
    rows = _block_rows(pairs.n_eff, mmax)
    base = _alm_base(lmax, mmax)
    step = _stepper(_synthesize_steps, device, "synthesize")
    even_all = torch.zeros(
        (mmax + 1, pairs.n_eff, channels), dtype=dtype, device=device
    )
    odd_all = torch.zeros_like(even_all)
    for m0 in range(0, mmax + 1, rows):
        m_rows_cpu = torch.arange(m0, m0 + rows, dtype=torch.float64)
        m_rows_cpu = torch.where(
            m_rows_cpu <= mmax, m_rows_cpu, torch.full_like(m_rows_cpu, lmax + 2.0)
        )
        seed_mant, seed_ex = _seeds(pairs, m_rows_cpu, mmax, scale_bits, device, dtype)
        m_rows = _to(m_rows_cpu, device=device, dtype=dtype)
        length = -(-(lmax - m0 + 1) // _K_STEPS) * _K_STEPS
        valid, idx, _ = _block_index(m0, rows, lmax, mmax, length, base)
        dense = a[_to(idx.reshape(-1), device=device)].reshape(length, rows, channels)
        dense = dense * _to(valid, device=device, dtype=dtype)[:, :, None]
        lam1 = torch.zeros((rows, pairs.n_eff), dtype=dtype, device=device)
        lam2 = torch.zeros_like(lam1)
        ex = torch.zeros_like(lam1)
        acc_e = torch.zeros((rows, pairs.n_eff, channels), dtype=dtype, device=device)
        acc_o = torch.zeros_like(acc_e)
        for t0 in range(0, length, _K_STEPS):
            ell0 = torch.tensor(float(m0 + t0), dtype=dtype, device=device)
            lam1, lam2, ex, acc_e, acc_o = step(
                ell0,
                m_rows,
                pairs.ct,
                lam1,
                lam2,
                ex,
                seed_mant,
                seed_ex,
                dense[t0 : t0 + _K_STEPS],
                acc_e,
                acc_o,
                thr,
                inv,
            )
        hi = min(m0 + rows, mmax + 1)
        even_all[m0:hi] = acc_e[: hi - m0]
        odd_all[m0:hi] = acc_o[: hi - m0]
    full = pairs.unfold(even_all, odd_all)  # (M, nrings, channels)
    full = (
        full.reshape(mmax + 1, pairs.nrings, nmaps, 2).permute(2, 0, 1, 3).contiguous()
    )
    return torch.view_as_complex(full)


# --------------------------------------------------------------------------
# Table mode: precomputed Legendre blocks, transforms as batched matmuls.
#
# For repeated transforms at a fixed (nside, lmax) the recurrence state need
# not stream through memory on every l step: lambda_lm on the folded rings is
# computed once (in float64) and each transform reads it once. Rows are stored
# relative to their own diagonal, l = m + s, so the parity of s is the parity
# of l + m and the even/odd halves pair with the even/odd ring sums uniformly:
#
#   map2alm   A[m, s] = sum_p T[m, s, p] X_parity(s)[m, p]      (bmm)
#   alm2map   X_e[m, p] = sum_s even A[m, s] T[m, s, p]           (bmm)
#
# m is grouped in _TABLE_BLOCKS blocks; a block's s axis is padded to the
# longest row (its first m), so storage is ~(1 + 1/_TABLE_BLOCKS) of the
# triangle while each transform issues only a few large matmuls.

_TABLE_BLOCKS = 8  # m blocks per table: few large matmuls, ~1/8 padding
_TABLE_CACHE: dict[tuple, "_LegendreTable"] = {}
_TABLE_CACHE_MAX = 2


def _table_budget(device: torch.device) -> int:
    """Bytes a cached Legendre table may take (``TORCHSKY_SHT_TABLE_BYTES``)."""
    env = os.environ.get("TORCHSKY_SHT_TABLE_BYTES")
    if env is not None and env.strip():
        return int(float(env))
    if device.type == "cuda":
        # Half the device: nside 512 / lmax 1024 in float64 (4.5 GB) fits a
        # 10.75 GiB H100 MIG slice, where the recurrence fallback measured
        # ~5x slower per transform and ~10 s cold.
        return int(torch.cuda.get_device_properties(device).total_memory // 2)
    if device.type == "mps":
        recommended = getattr(torch.mps, "recommended_max_memory", None)
        if recommended is not None:
            return int(recommended() // 4)
        return 2 << 30
    return 2 << 30


def _table_rows(mmax: int) -> int:
    return max(16, -(-(mmax + 1) // _TABLE_BLOCKS))


def _table_bytes(n_eff: int, lmax: int, mmax: int, itemsize: int) -> int:
    rows = _table_rows(mmax)
    total = 0
    for m0 in range(0, mmax + 1, rows):
        total += rows * (lmax - m0 + 1) * n_eff
    return total * itemsize


def _values_steps(
    ell0: Tensor,
    m_rows: Tensor,
    ct: Tensor,
    lam1: Tensor,
    lam2: Tensor,
    ex: Tensor,
    seed_mant: Tensor,
    seed_ex: Tensor,
    thr: float,
    inv: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """K steps of the recurrence returning lambda values (K, rows, rings)."""
    outs = []
    for k in range(_K_STEPS):
        ell = ell0 + float(k)
        new, lam1, ex, value = _advance(
            ell, m_rows, ct, lam1, lam2, ex, seed_mant, seed_ex, thr, inv
        )
        outs.append(value)
        lam2, lam1 = lam1, new
    return lam1, lam2, ex, torch.stack(outs)


class _LegendreTable:
    def __init__(
        self,
        theta_ring: Tensor,
        lmax: int,
        mmax: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.lmax, self.mmax = int(lmax), int(mmax)
        self.pairs = _RingPairs(theta_ring, device, dtype)
        n_eff = self.pairs.n_eff
        # Build in float64 (on the device when it has float64, else the host).
        build_dev = torch.device("cpu") if device.type == "mps" else device
        build_pairs = _RingPairs(theta_ring, build_dev, torch.float64)
        thr = 2.0 ** (_SCALE_BITS[torch.float64] // 2)
        inv = 2.0 ** (-_SCALE_BITS[torch.float64])
        base = _alm_base(lmax, mmax)
        rows = _table_rows(mmax)
        # Eager on purpose: compiling the build costs ~9 s per shape on CUDA
        # and measured no faster at nside 512 (10.7 s vs 10.4 s cold).
        values_step = _values_steps
        self.blocks: list[tuple[int, int, Tensor, Tensor, Tensor, Tensor]] = []
        for m0 in range(0, mmax + 1, rows):
            hi = min(m0 + rows, mmax + 1)
            m_rows_cpu = torch.arange(m0, m0 + rows, dtype=torch.float64)
            m_rows_cpu = torch.where(
                m_rows_cpu <= mmax, m_rows_cpu, torch.full_like(m_rows_cpu, lmax + 2.0)
            )
            seed_mant, seed_ex = _seeds(
                build_pairs,
                m_rows_cpu,
                mmax,
                _SCALE_BITS[torch.float64],
                build_dev,
                torch.float64,
            )
            m_rows = _to(m_rows_cpu, device=build_dev, dtype=torch.float64)
            length = -(-(lmax - m0 + 1) // _K_STEPS) * _K_STEPS
            vals = torch.empty(
                (length, rows, n_eff), dtype=torch.float64, device=build_dev
            )
            lam1 = torch.zeros((rows, n_eff), dtype=torch.float64, device=build_dev)
            lam2 = torch.zeros_like(lam1)
            ex = torch.zeros_like(lam1)
            for t0 in range(0, length, _K_STEPS):
                ell0 = torch.tensor(
                    float(m0 + t0), dtype=torch.float64, device=build_dev
                )
                lam1, lam2, ex, chunk = values_step(
                    ell0,
                    m_rows,
                    build_pairs.ct,
                    lam1,
                    lam2,
                    ex,
                    seed_mant,
                    seed_ex,
                    thr,
                    inv,
                )
                vals[t0 : t0 + _K_STEPS] = chunk
            # Re-index l = m0 + t to s = l - m (row-relative), s < lmax - m0 + 1.
            n_s = lmax - m0 + 1
            r = torch.arange(rows)[:, None]
            s = torch.arange(n_s)[None, :]
            t_idx = torch.clamp(s + r, max=length - 1)
            row_valid = (s + r + m0 <= lmax) & (r + m0 <= mmax)
            tab = vals[t_idx.to(build_dev), r.to(build_dev)]  # (rows, n_s, n_eff)
            tab = (
                tab * _to(row_valid, device=build_dev, dtype=torch.float64)[:, :, None]
            )
            even = _to(tab[:, 0::2].contiguous(), device=device, dtype=dtype)
            odd = _to(tab[:, 1::2].contiguous(), device=device, dtype=dtype)
            m_idx = torch.clamp(torch.arange(m0, m0 + rows), max=mmax)[:, None]
            alm_idx = base[m_idx] + m_idx + s  # l = m + s
            idx_e = alm_idx[:, 0::2]
            idx_o = alm_idx[:, 1::2]
            ok_e = row_valid[:, 0::2]
            ok_o = row_valid[:, 1::2]
            self.blocks.append(
                (
                    m0,
                    hi,
                    even,
                    odd,
                    _to(
                        torch.where(ok_e, idx_e, torch.full_like(idx_e, -1)),
                        device=device,
                    ),
                    _to(
                        torch.where(ok_o, idx_o, torch.full_like(idx_o, -1)),
                        device=device,
                    ),
                )
            )


def _legendre_table(
    theta_ring: Tensor, lmax: int, mmax: int, device: torch.device, dtype: torch.dtype
) -> "_LegendreTable | None":
    """Cached table, or None when it would exceed the device budget."""
    th = _to(theta_ring, device="cpu", dtype=torch.float64)
    nrings = int(th.numel())
    n_eff = nrings // 2 + nrings % 2
    itemsize = torch.empty((), dtype=dtype).element_size()
    if _table_bytes(n_eff, int(lmax), int(mmax), itemsize) > _table_budget(device):
        return None
    key = (nrings, float(th[0]), int(lmax), int(mmax), str(device), dtype)
    table = _TABLE_CACHE.get(key)
    if table is None:
        while len(_TABLE_CACHE) >= _TABLE_CACHE_MAX:
            _TABLE_CACHE.pop(next(iter(_TABLE_CACHE)))
        table = _LegendreTable(th, int(lmax), int(mmax), device, dtype)
        _TABLE_CACHE[key] = table
    return table


def _use_table(device: torch.device) -> bool:
    """``TORCHSKY_SHT_TABLE``: ``auto`` (default: GPU and MPS), ``1`` or ``0``."""
    mode = os.environ.get("TORCHSKY_SHT_TABLE", "auto").strip().lower()
    if mode in {"0", "false", "off", "no"}:
        return False
    if mode in {"1", "true", "on", "yes"}:
        return True
    return device.type in {"cuda", "mps"}


def _integrate_table(s_all: Tensor, table: _LegendreTable, pix_w: float) -> Tensor:
    nmaps = int(s_all.shape[0])
    dtype = s_all.real.dtype
    device = s_all.device
    pairs = table.pairs
    channels = nmaps * 2
    x = torch.view_as_real(s_all.contiguous()).permute(1, 2, 0, 3)
    x = x.reshape(table.mmax + 1, pairs.nrings, channels)
    w_even, w_odd = pairs.fold(x)
    nalm = int(_alm_base(table.lmax, table.mmax)[-1]) + table.lmax + 1
    out = torch.zeros((nalm + 1, channels), dtype=dtype, device=device)
    for m0, hi, even, odd, idx_e, idx_o in table.blocks:
        rows = even.shape[0]
        xe = torch.zeros((rows, pairs.n_eff, channels), dtype=dtype, device=device)
        xo = torch.zeros_like(xe)
        xe[: hi - m0] = w_even[m0:hi]
        xo[: hi - m0] = w_odd[m0:hi]
        ae = torch.bmm(even, xe)  # (rows, n_s_even, channels)
        ao = torch.bmm(odd, xo)
        # index -1 -> scratch slot nalm (discarded)
        ie = torch.where(idx_e >= 0, idx_e, torch.full_like(idx_e, nalm)).reshape(-1)
        io = torch.where(idx_o >= 0, idx_o, torch.full_like(idx_o, nalm)).reshape(-1)
        out.index_copy_(0, ie, ae.reshape(-1, channels))
        out.index_copy_(0, io, ao.reshape(-1, channels))
    out = out[:nalm] * pix_w
    out = out.reshape(nalm, nmaps, 2).permute(1, 0, 2).contiguous()
    return torch.view_as_complex(out)


def _synthesize_table(alm: Tensor, table: _LegendreTable) -> Tensor:
    nmaps = int(alm.shape[0])
    dtype = alm.real.dtype
    device = alm.device
    pairs = table.pairs
    channels = nmaps * 2
    a = torch.view_as_real(alm.contiguous()).permute(1, 0, 2).reshape(-1, channels)
    a = torch.cat([a, torch.zeros((1, channels), dtype=dtype, device=device)])
    scratch = a.shape[0] - 1
    even_all = torch.zeros(
        (table.mmax + 1, pairs.n_eff, channels), dtype=dtype, device=device
    )
    odd_all = torch.zeros_like(even_all)
    for m0, hi, even, odd, idx_e, idx_o in table.blocks:
        ie = torch.where(idx_e >= 0, idx_e, torch.full_like(idx_e, scratch))
        io = torch.where(idx_o >= 0, idx_o, torch.full_like(idx_o, scratch))
        ce = a[ie]  # (rows, n_s_even, channels)
        co = a[io]
        # (rows, channels, n_s) @ (rows, n_s, rings) -> (rows, channels, rings)
        xe = torch.bmm(ce.transpose(1, 2), even).transpose(1, 2)
        xo = torch.bmm(co.transpose(1, 2), odd).transpose(1, 2)
        even_all[m0:hi] = xe[: hi - m0]
        odd_all[m0:hi] = xo[: hi - m0]
    full = pairs.unfold(even_all, odd_all)
    full = (
        full.reshape(table.mmax + 1, pairs.nrings, nmaps, 2)
        .permute(2, 0, 1, 3)
        .contiguous()
    )
    return torch.view_as_complex(full)


def _native_host(device: torch.device) -> bool:
    """MPS without a table: run the Legendre step in the native CPU kernel.

    MPS has no float64, and a float32 recurrence loses ~1e-3 relative accuracy
    by lmax ~1000; the native float64 kernel is fast and only the (m, ring)
    mode arrays cross the device boundary.
    """
    from .. import _ops

    return device.type == "mps" and _ops.has_spectral_dispatcher()


def _integrate_native_host(
    s_all: Tensor, theta_ring: Tensor, *, lmax: int, mmax: int, pix_w: float
) -> Tensor:
    from .. import _ops

    th = _to(theta_ring, device="cpu", dtype=torch.float64)
    weight = torch.tensor([pix_w], dtype=torch.float64)
    host = s_all.detach().cpu().to(torch.complex128)
    alms = torch.stack(
        [
            _ops.spectral_scalar_integrate_recurrence(
                host[i].contiguous(), th, weight, int(lmax), int(mmax)
            )
            for i in range(int(host.shape[0]))
        ]
    )
    return alms.to(s_all.dtype).to(s_all.device)


def _synthesize_native_host(
    alm: Tensor, theta_ring: Tensor, *, lmax: int, mmax: int
) -> Tensor:
    from .. import _ops

    th = _to(theta_ring, device="cpu", dtype=torch.float64)
    host = alm.detach().cpu().to(torch.complex128)
    modes = torch.stack(
        [
            _ops.spectral_scalar_interpolate_recurrence(
                host[i].contiguous(), th, int(lmax), int(mmax)
            )
            for i in range(int(host.shape[0]))
        ]
    )
    return modes.to(alm.dtype).to(alm.device)


def _scalar_ring_legendre_integrate(
    s_all: Tensor, theta_ring: Tensor, *, lmax: int, mmax: int, pix_w: float
) -> Tensor:
    """alm[b, (l,m)] = pix_w * sum_r s_all[b, m, r] lambda_lm(theta_r).

    Table mode when the cached Legendre table fits the device budget (CUDA,
    MPS); on MPS beyond it the native float64 CPU kernel; otherwise the
    streamed recurrence. The native path is not differentiable through the
    modes, so it is used only when no gradient is required.
    """
    device = s_all.device
    if _use_table(device):
        table = _legendre_table(theta_ring, lmax, mmax, device, s_all.real.dtype)
        if table is not None:
            return _integrate_table(s_all, table, pix_w)
    needs_grad = torch.is_grad_enabled() and s_all.requires_grad
    if not needs_grad and _native_host(device):
        return _integrate_native_host(
            s_all, theta_ring, lmax=lmax, mmax=mmax, pix_w=pix_w
        )
    return _integrate_recurrence(s_all, theta_ring, lmax=lmax, mmax=mmax, pix_w=pix_w)


def _scalar_ring_legendre_synthesize(
    alm: Tensor, theta_ring: Tensor, *, lmax: int, mmax: int
) -> Tensor:
    """f[b, m, r] = sum_l alm[b, (l,m)] lambda_lm(theta_r) (dispatch as integrate)."""
    device = alm.device
    if _use_table(device):
        table = _legendre_table(theta_ring, lmax, mmax, device, alm.real.dtype)
        if table is not None:
            return _synthesize_table(alm, table)
    needs_grad = torch.is_grad_enabled() and alm.requires_grad
    if not needs_grad and _native_host(device):
        return _synthesize_native_host(alm, theta_ring, lmax=lmax, mmax=mmax)
    return _synthesize_recurrence(alm, theta_ring, lmax=lmax, mmax=mmax)
