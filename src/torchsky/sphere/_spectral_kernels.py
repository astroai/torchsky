"""Legendre scans and ring spin/scalar transform kernels for HEALPix spectral transforms."""

import os
import math

import torch
from torch import Tensor

from .. import _ops as _native_ops
from .._dtype import work_complex_dtype as _preferred_complex_dtype
from .._dtype import work_dtype as _preferred_real_dtype

from . import healpix as _healpix
from ._spectral_cache import SpectralCacheNamespace

from ._spectral_autograd import (
    SpinSHTInterpolate,
    _accumulate_add,
    _scalar_ring_integrate_recurrence_cpu,
    _scalar_ring_interpolate_recurrence_cpu,
)
from ._spectral_basis import (
    _ring_layout_for_nside,
    _ring_phi0_phase_for_nside,
    _ring_spin_bases_for_nside,
    _ring_spin_basis_concat,
)
from ._spectral_config import (
    _HAS_HEALPIX_SPIN_ALM2MAP_RING_REC,
    _HAS_HEALPIX_SPIN_INTERPOLATE_REC,
    _RING_PIX2RING_CACHE,
    _SPECTRAL_CACHE,
    _SPIN_RING_INTEGRATE_CONCAT_FAST_MAX_BYTES,
    _SPIN_RING_INTEGRATE_CONCAT_MAX_NALM,
    _SPIN_RING_INTERP_CONCAT_FAST_MAX_BYTES,
)
from ._spectral_layout import (
    _alm_m_array,
)
from ._spectral_ring_fourier import (
    _ring_fourier_modes,
    _ring_fourier_synthesis,
)
from ..device import _to


_M_PER_ALM_CACHE: SpectralCacheNamespace[Tensor] = _SPECTRAL_CACHE.namespace(
    "m_per_alm", device="cpu", dtype=torch.int64
)


def _m_per_alm_cached(lmax: int, mmax: int) -> Tensor:
    """Return (nalm,) int64 tensor where entry j holds the m-index for alm j."""
    key = (int(lmax), int(mmax))

    def build() -> Tensor:
        parts = [
            torch.full((lmax - m + 1,), m, dtype=torch.int64) for m in range(mmax + 1)
        ]
        return torch.cat(parts)

    return _M_PER_ALM_CACHE.get_or_build(key, build)


_LEGENDRE_SCAN_MAX_ELEMS = 1 << 22


def _scalar_legendre_chunks(theta_ring: Tensor, mmax: int) -> list[tuple[int, int]]:
    """Split ``0..mmax`` into m-chunks so a chunk holds ~``2**22`` (m, ring) values."""
    nrings = max(int(theta_ring.numel()), 1)
    width = max(1, _LEGENDRE_SCAN_MAX_ELEMS // nrings)
    return [
        (m0, min(m0 + width, int(mmax) + 1)) for m0 in range(0, int(mmax) + 1, width)
    ]


def _scan_device(device: torch.device) -> torch.device:
    """float64 recurrence device: MPS has no float64, so scan on the host."""
    return torch.device("cpu") if device.type == "mps" else device


def _scalar_legendre_scan(
    theta_ring: Tensor,
    *,
    m0: int,
    m1: int,
    lmax: int,
    device: torch.device,
):
    """
    Yield ``(l, k, lam)`` with ``lam[i, r] = lambda_{l, m0+i}(theta_r)`` for the
    ``k = min(l, m1 - 1) - m0 + 1`` rows with ``m <= l``.

    Orthonormal Legendre three-term recurrence in ``l``, vectorised over a chunk
    of m and all rings; memory is ``O((m1 - m0) * nrings)``, never the dense
    ``(nalm, nrings)`` basis. The sectoral seed is formed in log space and every
    (m, ring) carries an integer power-of-``2**S`` exponent, so the recurrence
    neither overflows nor underflows at any degree (underflowed values lie below
    ``2**-2S`` relative to O(1) and are dropped, as in libsharp).
    """
    nm = m1 - m0
    dtype = torch.float64
    scale_bits = 256
    ln_scale = scale_bits * math.log(2.0)
    thr = 2.0 ** (scale_bits // 2)
    inv = 2.0 ** (-scale_bits)

    th = _to(theta_ring, device="cpu", dtype=torch.float64)
    ct = _to(torch.cos(th), device=device, dtype=dtype)
    log_sin = torch.log(torch.clamp(torch.sin(th), min=1e-300))
    mv = torch.arange(m0, m1, dtype=torch.float64)
    log_c = 0.5 * (
        torch.log(2 * mv + 1)
        - math.log(4.0 * math.pi)
        + torch.lgamma(2 * mv + 1)
        - 2 * (mv * math.log(2.0) + torch.lgamma(mv + 1))
    )
    seed_log = log_c[:, None] + mv[:, None] * log_sin[None, :]
    ex64 = torch.clamp(torch.floor(seed_log / ln_scale), max=0.0)
    mant = torch.exp(seed_log - ex64 * ln_scale)
    sign = torch.where(mv % 2 == 0, 1.0, -1.0)
    mant = _to(mant * sign[:, None], device=device, dtype=dtype)
    ex = _to(ex64, device=device, dtype=dtype)

    lam1 = torch.zeros((nm, ct.numel()), dtype=dtype, device=device)
    lam2 = torch.zeros_like(lam1)
    for ell in range(m0, int(lmax) + 1):
        kp = min(ell - 1, m1 - 1) - m0 + 1
        new = torch.zeros_like(lam1)
        if kp > 0:
            ms = mv[:kp]
            l2 = float(ell) * float(ell)
            den = l2 - ms * ms
            a = torch.sqrt((4.0 * l2 - 1.0) / den)
            b = torch.sqrt(
                (2.0 * ell + 1.0)
                / (2.0 * ell - 3.0 if ell > 1 else 1.0)
                * (((ell - 1.0) ** 2) - ms * ms)
                / den
            )
            a_t = _to(a, device=device, dtype=dtype)[:, None]
            b_t = _to(b, device=device, dtype=dtype)[:, None]
            new[:kp] = a_t * ct[None, :] * lam1[:kp] - b_t * lam2[:kp]
        if ell < m1:
            new[ell - m0] = mant[ell - m0]
        big = new.abs() > thr
        new = torch.where(big, new * inv, new)
        lam1 = torch.where(big, lam1 * inv, lam1)
        ex = ex + big.to(dtype)
        k = min(ell, m1 - 1) - m0 + 1
        yield ell, k, new[:k] * torch.exp2(scale_bits * ex[:k])
        lam2, lam1 = lam1, new


def _scalar_ring_legendre_integrate(
    s_all: Tensor, theta_ring: Tensor, *, lmax: int, mmax: int, pix_w: float
) -> Tensor:
    """alm[b, (l,m)] = pix_w * sum_r s_all[b, m, r] lambda_lm(theta_r); chunked scan."""
    nmaps = int(s_all.shape[0])
    device = s_all.device
    real_dtype = s_all.real.dtype
    s_ri = torch.view_as_real(s_all.contiguous())
    out = torch.zeros(
        (nmaps, _core_mod.alm_size(lmax, mmax), 2), dtype=real_dtype, device=device
    )
    marr = torch.arange(int(mmax) + 1, dtype=torch.int64)
    base = (marr * (lmax + 1) - marr * (marr - 1) // 2 - marr).to(device)
    for m0, m1 in _scalar_legendre_chunks(theta_ring, mmax):
        s_c = s_ri[:, m0:m1]
        for ell, k, lam in _scalar_legendre_scan(
            theta_ring, m0=m0, m1=m1, lmax=lmax, device=_scan_device(device)
        ):
            lam = _to(lam, device=device, dtype=real_dtype)
            coef = torch.einsum("bmrc,mr->bmc", s_c[:, :k], lam)
            out[:, base[m0 : m0 + k] + ell] = coef * pix_w
    return torch.view_as_complex(out.contiguous())


def _scalar_ring_legendre_synthesize(
    alm: Tensor, theta_ring: Tensor, *, lmax: int, mmax: int
) -> Tensor:
    """f[b, m, r] = sum_l alm[b, (l,m)] lambda_lm(theta_r); chunked scan."""
    nmaps = int(alm.shape[0])
    device = alm.device
    real_dtype = alm.real.dtype
    nrings = int(theta_ring.numel())
    a_ri = torch.view_as_real(alm.contiguous())
    f = torch.zeros((nmaps, mmax + 1, nrings, 2), dtype=real_dtype, device=device)
    marr = torch.arange(int(mmax) + 1, dtype=torch.int64)
    base = (marr * (lmax + 1) - marr * (marr - 1) // 2 - marr).to(device)
    for m0, m1 in _scalar_legendre_chunks(theta_ring, mmax):
        for ell, k, lam in _scalar_legendre_scan(
            theta_ring, m0=m0, m1=m1, lmax=lmax, device=_scan_device(device)
        ):
            lam = _to(lam, device=device, dtype=real_dtype)
            coef = a_ri[:, base[m0 : m0 + k] + ell]
            f[:, m0 : m0 + k] += coef[:, :, None, :] * lam[None, :, :, None]
    return torch.view_as_complex(f.contiguous())


def _scalar_alm2map_ring_torch(
    rows: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
) -> Tensor:
    starts_cpu, lengths_cpu, _, _ = _ring_layout_for_nside(int(nside))
    device = rows.device
    dtype_real = _preferred_real_dtype(device)
    dtype_cplx = _preferred_complex_dtype(device)

    nmaps = int(rows.shape[0])

    use_cpp_recurrence = (
        _core_mod._SCALAR_INTERP_CPP_ENABLE
        and _native_ops.has_spectral_dispatcher()
        and device.type == "cpu"
    )

    if use_cpp_recurrence:
        out_s_m = []
        for i in range(nmaps):
            s_modes = _scalar_ring_interpolate_recurrence_cpu(
                rows[i],
                nside=int(nside),
                lmax=int(lmax),
                mmax=int(mmax),
            )
            out_s_m.append(s_modes)
        s_modes_batch = torch.stack(out_s_m, dim=0)

        phase0_pos = _to(
            _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=1),
            dtype=torch.complex128,
            device=s_modes_batch.device,
        )
        c_all = s_modes_batch * phase0_pos.unsqueeze(0)

        if mmax > 0:
            c_all = c_all.clone()
            c_all[:, 1:, :] = 2.0 * c_all[:, 1:, :]
    else:
        # Chunked Legendre recurrence scan: O(chunk * nrings) memory.
        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(int(nside))
        f_m_ring = _scalar_ring_legendre_synthesize(
            rows.to(dtype=dtype_cplx),
            theta_ring_cpu,
            lmax=int(lmax),
            mmax=int(mmax),
        )

        phi0_phase = _to(
            _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=1),
            device=device,
            dtype=dtype_cplx,
        )
        # Apply phi0 phase per m
        f_m_ring = f_m_ring * phi0_phase.unsqueeze(0)

        c_all = f_m_ring
        if mmax > 0:
            c_all = c_all.clone()
            c_all[:, 1:, :] = 2.0 * c_all[:, 1:, :]

    out = _ring_fourier_synthesis(
        c_all,
        starts_cpu=starts_cpu,
        lengths_cpu=lengths_cpu,
        nside=int(nside),
    )
    if out.is_complex():
        out = out.real
    return out.to(dtype=dtype_real)


def _scalar_map2alm_ring_torch(
    rows: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
) -> Tensor:
    starts_cpu, lengths_cpu, _, _ = _ring_layout_for_nside(int(nside))
    device = rows.device
    dtype_cplx = _preferred_complex_dtype(device)

    nmaps = int(rows.shape[0])
    npix = int(rows.shape[1])

    pix_w = (4.0 * math.pi) / float(npix)
    s_all = _ring_fourier_modes(
        rows.to(dtype=dtype_cplx),
        starts_cpu=starts_cpu,
        lengths_cpu=lengths_cpu,
        mmax=mmax,
        nside=int(nside),
    )

    phase0_neg = _to(
        _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=-1),
        device=device,
        dtype=dtype_cplx,
    )
    s_all = s_all * phase0_neg.unsqueeze(0)

    use_cpp_recurrence = (
        _core_mod._SCALAR_RECURRENCE_CPP_ENABLE
        and _native_ops.has_spectral_dispatcher()
        and device.type == "cpu"
    )

    if use_cpp_recurrence:
        out_alms = []
        for i in range(nmaps):
            alm = _scalar_ring_integrate_recurrence_cpu(
                s_all[i],
                nside=int(nside),
                lmax=int(lmax),
                mmax=int(mmax),
                pix_w=pix_w,
            )
            out_alms.append(alm)
        return torch.stack(out_alms, dim=0)

    _, _, theta_ring_cpu, _ = _ring_layout_for_nside(int(nside))
    return _scalar_ring_legendre_integrate(
        s_all, theta_ring_cpu, lmax=int(lmax), mmax=int(mmax), pix_w=pix_w
    )


def _ring_pix2ring_index(nside: int) -> Tensor:
    key = int(nside)

    def build() -> Tensor:
        _, lengths_cpu, _, _ = _ring_layout_for_nside(key)
        nrings = int(lengths_cpu.numel())
        ring_ids = torch.arange(nrings, dtype=torch.int64)
        return torch.repeat_interleave(ring_ids, lengths_cpu)

    return _RING_PIX2RING_CACHE.get_or_build(key, build)


def _ring_expand_m0(s0: Tensor, *, nside: int) -> Tensor:
    """
    Expand per-ring m=0 values to per-pixel map.

    s0: (nrings,) complex
    returns: (npix,) complex
    """
    ring_idx_cpu = _ring_pix2ring_index(int(nside))
    ring_idx = (
        ring_idx_cpu if s0.device.type == "cpu" else ring_idx_cpu.to(device=s0.device)
    )
    if s0.ndim == 1:
        return torch.index_select(s0, 0, ring_idx)
    if s0.ndim == 2:
        return torch.index_select(s0, 1, ring_idx)
    raise ValueError("s0 must have shape (nrings,) or (nrows, nrings)")


def _ring_spin_interpolate_blocks(
    coeff_plus: Tensor,
    coeff_minus: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor]:
    nrings = (4 * int(nside)) - 1
    nalm = _core_mod.alm_size(int(lmax), int(mmax))
    bytes_concat = int(nalm * nrings * 32)  # two complex128 concat bases
    use_concat_fast = bytes_concat <= _SPIN_RING_INTERP_CONCAT_FAST_MAX_BYTES
    if use_concat_fast:
        y_plus, y_minus = _ring_spin_basis_concat(
            int(nside), int(lmax), int(mmax), int(spin)
        )
        device = coeff_plus.device
        dtype = coeff_plus.dtype
        if y_plus.numel() == 0:
            empty = torch.zeros((int(mmax) + 1, 0), dtype=dtype, device=device)
            return empty, empty
        y_plus = _to(y_plus, device=device, dtype=dtype)
        y_minus = _to(y_minus, device=device, dtype=dtype)
        use_cpp = (
            os.environ.get("TORCHSKY_SPIN_RING_INTERP_CONCAT_CPP", "1") != "0"
            and _native_ops.has_spectral_dispatcher()
            and device.type == "cpu"
            and dtype == torch.complex128
            and (not coeff_plus.requires_grad)
            and (not coeff_minus.requires_grad)
        )
        if use_cpp:
            s_plus_cpp, s_minus_cpp = _native_ops.spectral_spin_interpolate_concat(
                coeff_plus.contiguous(),
                coeff_minus.contiguous(),
                y_plus.contiguous(),
                y_minus.contiguous(),
                int(lmax),
                int(mmax),
            )
            return s_plus_cpp, s_minus_cpp
        m_idx = _alm_m_array(int(lmax), int(mmax))
        m_idx = m_idx if device.type == "cpu" else m_idx.to(device=device)
        term_plus = coeff_plus.unsqueeze(1) * y_plus
        term_minus = coeff_minus.unsqueeze(1) * y_minus
        s_plus = torch.zeros(
            (int(mmax) + 1, y_plus.shape[1]), dtype=dtype, device=device
        )
        s_minus = torch.zeros_like(s_plus)
        s_plus = _accumulate_add(s_plus, 0, m_idx, term_plus, scatter=False)
        s_minus = _accumulate_add(s_minus, 0, m_idx, term_minus, scatter=False)
        return s_plus, s_minus

    y_plus_blocks = _ring_spin_bases_for_nside(
        int(nside), int(lmax), int(mmax), int(spin)
    )
    y_minus_blocks = _ring_spin_bases_for_nside(
        int(nside), int(lmax), int(mmax), -int(spin)
    )
    nrings = int(y_plus_blocks[0].shape[1]) if y_plus_blocks else 0
    device = coeff_plus.device
    dtype = coeff_plus.dtype
    s_plus = torch.zeros((int(mmax) + 1, nrings), dtype=dtype, device=device)
    s_minus = torch.zeros_like(s_plus)
    idx = 0
    for m in range(int(mmax) + 1):
        l_count = int(lmax) - m + 1
        y_plus = _to(y_plus_blocks[m], device=device, dtype=dtype)
        y_minus = _to(y_minus_blocks[m], device=device, dtype=dtype)
        c_plus_m = coeff_plus[idx : idx + l_count]
        c_minus_m = coeff_minus[idx : idx + l_count]
        s_plus[m] = c_plus_m @ y_plus
        s_minus[m] = c_minus_m @ y_minus
        idx += l_count
    return s_plus, s_minus


def _ring_spin_integrate_blocks(
    s_plus: Tensor,
    s_minus: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
    spin: int,
    pix_w: float,
) -> tuple[Tensor, Tensor]:
    nrings = (4 * int(nside)) - 1
    nalm = _core_mod.alm_size(int(lmax), int(mmax))
    bytes_concat = int(nalm * nrings * 32)  # two complex128 concat bases
    use_concat_fast = (
        bytes_concat <= _SPIN_RING_INTEGRATE_CONCAT_FAST_MAX_BYTES
        and nalm <= _SPIN_RING_INTEGRATE_CONCAT_MAX_NALM
    )
    if use_concat_fast:
        y_plus, y_minus = _ring_spin_basis_concat(
            int(nside), int(lmax), int(mmax), int(spin)
        )
        device = s_plus.device
        dtype = s_plus.dtype
        if y_plus.numel() == 0:
            empty = torch.zeros((nalm,), dtype=dtype, device=device)
            return empty, empty
        y_plus = _to(y_plus, device=device, dtype=dtype)
        y_minus = _to(y_minus, device=device, dtype=dtype)
        use_cpp = (
            os.environ.get("TORCHSKY_SPIN_RING_INTEGRATE_CONCAT_CPP", "1") != "0"
            and _native_ops.has_spectral_dispatcher()
            and device.type == "cpu"
            and dtype == torch.complex128
            and (not s_plus.requires_grad)
            and (not s_minus.requires_grad)
        )
        if use_cpp:
            c_plus_cpp, c_minus_cpp = _native_ops.spectral_spin_integrate_concat(
                s_plus.contiguous(),
                s_minus.contiguous(),
                y_plus.contiguous(),
                y_minus.contiguous(),
                int(lmax),
                int(mmax),
                float(pix_w),
            )
            return c_plus_cpp, c_minus_cpp
        m_idx = _alm_m_array(int(lmax), int(mmax))
        m_idx = m_idx if device.type == "cpu" else m_idx.to(device=device)
        s_plus_exp = torch.index_select(s_plus, 0, m_idx)
        s_minus_exp = torch.index_select(s_minus, 0, m_idx)
        c_plus = torch.sum(s_plus_exp * y_plus, dim=1) * pix_w
        c_minus = torch.sum(s_minus_exp * y_minus, dim=1) * pix_w
        return c_plus, c_minus

    y_plus_blocks = _ring_spin_bases_for_nside(
        int(nside), int(lmax), int(mmax), int(spin)
    )
    y_minus_blocks = _ring_spin_bases_for_nside(
        int(nside), int(lmax), int(mmax), -int(spin)
    )
    device = s_plus.device
    dtype = s_plus.dtype
    nalm = _core_mod.alm_size(int(lmax), int(mmax))
    c_plus = torch.empty((nalm,), dtype=dtype, device=device)
    c_minus = torch.empty_like(c_plus)
    idx = 0
    for m in range(int(mmax) + 1):
        l_count = int(lmax) - m + 1
        y_plus = _to(y_plus_blocks[m], device=device, dtype=dtype)
        y_minus = _to(y_minus_blocks[m], device=device, dtype=dtype)
        c_plus[idx : idx + l_count] = (y_plus @ s_plus[m]) * pix_w
        c_minus[idx : idx + l_count] = (y_minus @ s_minus[m]) * pix_w
        idx += l_count
    return c_plus, c_minus


def _spin_alm2map_ring_torch(
    coeff_plus: Tensor,
    coeff_minus: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
    spin: int,
) -> Tensor:
    """
    Helper for ring-based alm2map_spin.
    """
    use_cpp = (
        _HAS_HEALPIX_SPIN_INTERPOLATE_REC
        and os.environ.get("TORCHSKY_SPIN_ALM2MAP_RECURRENCE_CPP", "1") != "0"
        and coeff_plus.device.type == "cpu"
        and coeff_minus.device.type == "cpu"
    )

    if use_cpp:
        starts_cpu, lengths_cpu, theta_ring_cpu, _ = _ring_layout_for_nside(nside)
        fused_rec = (
            _HAS_HEALPIX_SPIN_ALM2MAP_RING_REC
            and coeff_plus.device.type == "cpu"
            and (not coeff_plus.requires_grad)
            and (not coeff_minus.requires_grad)
        )
        if fused_rec:
            cp_cpu = (
                coeff_plus
                if coeff_plus.dtype == torch.complex128
                else coeff_plus.to(dtype=torch.complex128)
            )
            cm_cpu = (
                coeff_minus
                if coeff_minus.dtype == torch.complex128
                else coeff_minus.to(dtype=torch.complex128)
            )
            phase0_pos = _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=1)
            return _native_ops.spectral_spin_alm2map_ring_recurrence(
                cp_cpu,
                starts_cpu,
                lengths_cpu,
                theta_ring_cpu,
                phase0_pos,
                int(lmax),
                int(mmax),
                int(spin),
                cm_cpu,
                _healpix.nside2npix(int(nside)),
            )
        if coeff_plus.requires_grad or coeff_minus.requires_grad:
            s_plus, s_minus = SpinSHTInterpolate.apply(
                coeff_plus, coeff_minus, int(nside), int(lmax), int(mmax), int(spin)
            )
        else:
            # Fast path: call C++ directly and avoid stack/detach overhead
            _, _, theta_ring_cpu, _ = _ring_layout_for_nside(nside)
            # We still need complex128 CPU tensors for the current C++ impl
            cp_cpu = _to(coeff_plus, dtype=torch.complex128, device="cpu").contiguous()
            cm_cpu = _to(coeff_minus, dtype=torch.complex128, device="cpu").contiguous()
            alm_stack = torch.stack([cp_cpu, cm_cpu], dim=0)

            s_modes_stacked = _native_ops.spectral_spin_interpolate_recurrence(
                alm_stack,
                theta_ring_cpu.contiguous(),
                int(lmax),
                int(mmax),
                int(spin),
            )
            s_plus = s_modes_stacked[0].to(device=coeff_plus.device)
            s_minus = s_modes_stacked[1].to(device=coeff_minus.device)
    else:
        s_plus, s_minus = _ring_spin_interpolate_blocks(
            coeff_plus,
            coeff_minus,
            nside=int(nside),
            lmax=int(lmax),
            mmax=int(mmax),
            spin=int(spin),
        )

    # 2. Apply phase factors and inverse FFT to get ring values
    phase0_pos = _to(
        _ring_phi0_phase_for_nside(int(nside), int(mmax), sign=1),
        device=s_plus.device,
        dtype=s_plus.dtype,
    )
    s_plus = s_plus * phase0_pos
    s_minus = s_minus * phase0_pos

    # 3. Inverse FFT (synthesis) + final spin combination
    start_pix, lengths, _, _ = _ring_layout_for_nside(int(nside))
    use_cpp_finalize = (
        os.environ.get("TORCHSKY_SPIN_RING_FINALIZE_CPP", "1") != "0"
        and _native_ops.has_spectral_dispatcher()
        and s_plus.device.type == "cpu"
        and (not s_plus.requires_grad)
        and (not s_minus.requires_grad)
    )
    if use_cpp_finalize:
        return _native_ops.spectral_spin_ring_finalize(
            s_plus.contiguous(),
            s_minus.contiguous(),
            start_pix.contiguous(),
            lengths.contiguous(),
            _healpix.nside2npix(int(nside)),
        )

    p_pm_pos = _ring_fourier_synthesis(
        torch.stack([s_plus, s_minus], dim=0),
        starts_cpu=start_pix,
        lengths_cpu=lengths,
        nside=int(nside),
    )
    p_plus_pos = p_pm_pos[0]
    p_minus_pos = p_pm_pos[1]
    p_pm_m0 = _ring_expand_m0(
        torch.stack([s_plus[0], s_minus[0]], dim=0), nside=int(nside)
    )
    p_plus_m0 = p_pm_m0[0]
    p_minus_m0 = p_pm_m0[1]

    if mmax == 0:
        p_plus = p_plus_pos
        p_minus = p_minus_pos
    else:
        p_plus = p_plus_pos + torch.conj(p_minus_pos - p_minus_m0)
        p_minus = p_minus_pos + torch.conj(p_plus_pos - p_plus_m0)

    q = 0.5 * (p_plus + p_minus)
    u = -0.5j * (p_plus - p_minus)
    real_dtype = _preferred_real_dtype(q.device)
    return torch.stack([(-q).real.to(real_dtype), (-u).real.to(real_dtype)], dim=0)


# Late-bound so monkeypatching ``spectral`` attributes still reaches this code.
from . import spectral as _core_mod  # noqa: E402
