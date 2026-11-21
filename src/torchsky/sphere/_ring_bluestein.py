"""All HEALPix polar-ring DFTs in a few batched FFTs (Bluestein / chirp-z).

Polar rings have distinct lengths (4, 8, ..., 4(nside-1) pixels), so a plain
FFT needs one call per length: ~nside small calls, which dominate a transform
on GPUs and MPS once the Legendre step is fast. Bluestein's identity
``j k = (j^2 + k^2 - (k - j)^2) / 2`` turns a length-n DFT into a convolution,
which every ring can evaluate at one common padded length L >= 2 n_max - 1:

    X_k = conj(b_k) * sum_j (x_j conj(b_j)) b_{k-j},     b_t = exp(i pi t^2 / n)

The per-ring chirps conj(b_j) and the FFTs of the wrapped kernels b_{t} are
precomputed once per (rings, L, device, dtype); a transform is then a gather,
two FFTs of size L over all rings at once, and elementwise products. The
result is exact up to rounding (no approximation of the DFT).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..device import _to

_PLAN_CACHE: dict[tuple, "_BluesteinPlan"] = {}
_PLAN_CACHE_MAX = 4


class _BluesteinPlan:
    def __init__(
        self,
        groups: tuple[tuple[int, Tensor, Tensor], ...],
        npix: int,
        mmax: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        lengths: list[int] = []
        ring_ids: list[int] = []
        starts: list[int] = []
        for nph, ring_idx, pix_idx in groups:
            for i in range(int(ring_idx.numel())):
                lengths.append(int(nph))
                ring_ids.append(int(ring_idx[i]))
                starts.append(int(pix_idx[i, 0]))
        n = torch.tensor(lengths, dtype=torch.int64)
        n_max = int(n.max())
        size = 1 << int(math.ceil(math.log2(2 * n_max - 1)))
        self.size = size
        self.n_rings = len(lengths)
        self.npix = npix
        real = torch.float64 if dtype == torch.complex128 else torch.float32

        j = torch.arange(size, dtype=torch.int64)[None, :]
        nn = n[:, None]
        inside = j < nn
        # Pixel gather (padded slots point to a zero sentinel at index npix).
        pix = torch.tensor(starts, dtype=torch.int64)[:, None] + j
        self.pix = _to(
            torch.where(inside, pix, torch.full_like(pix, npix)), device=device
        )
        self.inside = _to(inside, device=device)

        # Chirps b_t = exp(i pi t^2 / n), with t^2 reduced mod 2n in integers so
        # the phase stays exact for large t.
        t2 = torch.remainder(j * j, 2 * nn)
        phase = math.pi * t2.to(torch.float64) / nn.to(torch.float64)
        b = torch.polar(torch.ones_like(phase), phase)  # (rings, size)
        zero = torch.zeros_like(b)
        self.chirp = _to(
            torch.where(inside, b.conj(), zero), device=device, dtype=dtype
        )
        self.chirp_out = _to(b.conj(), device=device, dtype=dtype)  # conj(b_k)
        self.chirp_syn = _to(torch.where(inside, b, zero), device=device, dtype=dtype)
        self.chirp_syn_out = _to(b, device=device, dtype=dtype)
        # Wrapped kernels: h[t] = b_t for t < n, h[L - t] = b_t for 0 < t < n.
        tail = (size - j) < nn
        tail_t = torch.remainder(size - j, size)
        t2_tail = torch.remainder(tail_t * tail_t, 2 * nn)
        b_tail = torch.polar(
            torch.ones_like(phase),
            math.pi * t2_tail.to(torch.float64) / nn.to(torch.float64),
        )
        h = torch.where(inside, b, torch.where(tail & (j > 0), b_tail, zero))
        self.kernel = _to(torch.fft.fft(h, dim=-1), device=device, dtype=dtype)
        self.kernel_syn = _to(
            torch.fft.fft(h.conj(), dim=-1), device=device, dtype=dtype
        )

        # Mode readout k = m mod n, and alias folding for synthesis.
        m = torch.arange(int(mmax) + 1, dtype=torch.int64)[None, :]
        k = torch.remainder(m, nn)  # (rings, M)
        self.mode_idx = _to(k, device=device)
        self.fold_idx = _to(
            k + torch.arange(self.n_rings, dtype=torch.int64)[:, None] * size,
            device=device,
        )
        self.ring_idx = _to(torch.tensor(ring_ids, dtype=torch.int64), device=device)
        del real

    def forward(self, rows: Tensor) -> Tensor:
        """rows (nrows, npix) -> modes (nrows, M, n_polar_rings)."""
        nrows = int(rows.shape[0])
        padded = torch.cat(
            [rows, torch.zeros((nrows, 1), dtype=rows.dtype, device=rows.device)], dim=1
        )
        x = padded[:, self.pix]  # (nrows, rings, size)
        y = torch.fft.ifft(torch.fft.fft(x * self.chirp, dim=-1) * self.kernel, dim=-1)
        y = y * self.chirp_out
        idx = self.mode_idx.unsqueeze(0).expand(nrows, -1, -1)
        modes = torch.gather(y, 2, idx)  # (nrows, rings, M)
        return modes.transpose(1, 2)

    def inverse(self, s_m: Tensor) -> Tensor:
        """Mode coefficients (nrows, M, n_polar_rings) -> pixel values (nrows, npix + 1).

        x_j = sum_k S_k exp(+2 pi i j k / n) with S_k = sum_{m = k mod n} s_m;
        the last column is the sentinel (discard).
        """
        nrows = int(s_m.shape[0])
        coeff = s_m.transpose(1, 2)  # (nrows, rings, M)
        folded = torch.zeros(
            (nrows, self.n_rings * self.size), dtype=s_m.dtype, device=s_m.device
        )
        folded.index_add_(1, self.fold_idx.reshape(-1), coeff.reshape(nrows, -1))
        folded = folded.view(nrows, self.n_rings, self.size)
        y = torch.fft.ifft(
            torch.fft.fft(folded * self.chirp_syn, dim=-1) * self.kernel_syn, dim=-1
        )
        x = y * self.chirp_syn_out
        out = torch.zeros((nrows, self.npix + 1), dtype=s_m.dtype, device=s_m.device)
        flat_pix = self.pix.reshape(-1)
        out.index_copy_(1, flat_pix, (x * self.inside).reshape(nrows, -1))
        return out


def bluestein_plan(
    groups: tuple[tuple[int, Tensor, Tensor], ...],
    npix: int,
    mmax: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _BluesteinPlan:
    key = (
        tuple(
            (int(nph), int(ring_idx[0]), int(ring_idx.numel()))
            for nph, ring_idx, _ in groups
        ),
        int(npix),
        int(mmax),
        str(device),
        dtype,
    )
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        while len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
        plan = _BluesteinPlan(groups, npix, mmax, device, dtype)
        _PLAN_CACHE[key] = plan
    return plan
