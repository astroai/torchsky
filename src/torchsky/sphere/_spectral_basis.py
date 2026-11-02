"""Legendre, Wigner-d, spherical-harmonic basis and ring layout/phase helpers for HEALPix spectral transforms."""

import math

import torch
from torch import Tensor


from . import healpix as _healpix
from ._spectral_cache import SpectralCacheNamespace

from ._spectral_config import (
    _LEGENDRE_REC_CACHE,
    _RING_ALIAS_INDEX_CACHE,
    _RING_AZIMUTH_PHASE_CACHE,
    _RING_GROUP_CACHE,
    _RING_LAYOUT_CACHE,
    _RING_PHI0_PHASE_CACHE,
    _RING_SCALAR_BASIS_CACHE,
    _RING_SCALAR_BASIS_CONCAT_CACHE,
    _RING_SPIN_BASIS_CACHE,
    _RING_SPIN_CONJ_INDEX_CACHE,
    _SCALAR_CACHE_ENTRY_LIMIT,
    _SCALAR_MAP2ALM_MAT_CACHE,
    _SPECTRAL_CACHE,
    _SPIN_ALM2MAP_MAT_CACHE,
    _SPIN_CACHE_ENTRY_LIMIT,
    _SPIN_MAP2ALM_MAT_CACHE,
    _SPIN_YLM_CACHE,
)
from ._spectral_layout import (
    _angles_for_nside,
)


def _legendre_rec_coeffs(
    m: int, lmax: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return recurrence coefficients a_lm, b_lm for l=m+2..lmax."""
    key = (m, lmax)

    def build() -> tuple[tuple[float, ...], tuple[float, ...]]:
        a_list = []
        b_list = []
        for ell in range(m + 2, lmax + 1):
            a_lm = math.sqrt((4.0 * ell * ell - 1.0) / (ell * ell - m * m))
            b_lm = math.sqrt(
                (2.0 * ell + 1.0)
                / (2.0 * ell - 3.0)
                * (((ell - 1.0) * (ell - 1.0)) - m * m)
                / (ell * ell - m * m)
            )
            a_list.append(a_lm)
            b_list.append(b_lm)
        return tuple(a_list), tuple(b_list)

    return _LEGENDRE_REC_CACHE.get_or_build(key, build)


def _legendre_l_sequence_loop(
    x: Tensor,
    prev2: Tensor,
    prev1: Tensor,
    a_list: tuple[float, ...],
    b_list: tuple[float, ...],
) -> list[Tensor]:
    """Inner recurrence loop, potentially compilable."""
    res = []
    p2 = prev2
    p1 = prev1
    for i in range(len(a_list)):
        cur = a_list[i] * x * p1 - b_list[i] * p2
        res.append(cur)
        p2 = p1
        p1 = cur
    return res


def _legendre_l_sequence(m: int, x: Tensor, lmax: int) -> Tensor:
    """Fully normalized Associated Legendre P_l^m(x) for l=m..lmax. shape [lmax-m+1, npix]."""
    # Standard normalization for Ylm includes 1/sqrt(4pi)
    # Pmm = (-1)^m * sqrt( (2m+1)!! / (2m)!! ) * (1-x^2)^(m/2) * sqrt(1/4pi)

    x.shape[0]

    sin2theta = torch.clamp(1.0 - x * x, min=1e-30)

    ln_4pi = math.log(4.0 * math.pi)
    ln2 = math.log(2.0)

    # log_norm_mm = 0.5 * (log(2m+1) - log(4pi) + lgamma(2m+1) - 2*(m*log(2) + lgamma(m+1)))
    log_pmm = 0.5 * (
        math.log(2 * m + 1)
        - ln_4pi
        + math.lgamma(2 * m + 1)
        - 2 * (m * ln2 + math.lgamma(m + 1))
    )
    log_sin_m = 0.5 * m * torch.log(sin2theta)

    pmm = torch.exp(log_pmm + log_sin_m)
    if (m % 2) != 0:
        pmm = -pmm

    if lmax == m:
        return pmm.unsqueeze(0)

    pm1 = x * math.sqrt(2 * m + 3) * pmm

    if lmax == m + 1:
        return torch.stack([pmm, pm1], dim=0)

    a_list, b_list = _legendre_rec_coeffs(m, lmax)

    # Note: torch.compile overhead is often not worth it for small L,
    # but for large L (>512) it can be beneficial.
    # We use a simple loop here which is already faster than the old version.
    seq = [pmm, pm1]
    seq.extend(_legendre_l_sequence_loop(x, pmm, pm1, a_list, b_list))

    return torch.stack(seq, dim=0)


def _wigner_d_element(ell: int, m: int, mp: int, theta: Tensor) -> Tensor:
    """
    Compute Wigner small-d element d^ell_{m,mp}(theta).
    Direct implementation using Wigner's formula.
    WARNING: Numerically unstable for large ell. Use for small ell only (e.g. bootstrapping).
    """
    if abs(m) > ell or abs(mp) > ell:
        return torch.zeros_like(theta)

    ct = torch.cos(theta * 0.5)
    st = torch.sin(theta * 0.5)
    out = torch.zeros_like(theta)

    log_pref = 0.5 * (
        math.lgamma(float(ell + m + 1))
        + math.lgamma(float(ell - m + 1))
        + math.lgamma(float(ell + mp + 1))
        + math.lgamma(float(ell - mp + 1))
    )
    kmin = max(0, m - mp)
    kmax = min(ell + m, ell - mp)
    if kmin > kmax:
        return out

    for k in range(kmin, kmax + 1):
        a = ell + m - k
        b = k
        c = mp - m + k
        d = ell - mp - k
        if min(a, b, c, d) < 0:
            continue
        sign = -1.0 if ((k + mp - m) & 1) else 1.0
        log_den = (
            math.lgamma(float(a + 1))
            + math.lgamma(float(b + 1))
            + math.lgamma(float(c + 1))
            + math.lgamma(float(d + 1))
        )
        coeff = sign * math.exp(log_pref - log_den)
        p_ct = 2 * ell + m - mp - 2 * k
        p_st = mp - m + 2 * k
        out = out + coeff * (ct**p_ct) * (st**p_st)
    return out


def _build_spin_ylm_basis(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    theta, phi, _, _ = _angles_for_nside(nside, nest)
    y_blocks: list[Tensor] = []
    y_conj_blocks: list[Tensor] = []
    # ⚡ Bolt: Vectorized phase multipliers for all m
    m_range = torch.arange(mmax + 1, device=phi.device, dtype=phi.dtype)
    # [mmax+1, npix]
    all_phases = torch.exp(
        torch.complex(
            torch.zeros(1, device=phi.device, dtype=phi.dtype),
            m_range.unsqueeze(1) * phi.unsqueeze(0),
        )
    )
    spin_phase = -1.0 if (spin & 1) else 1.0  # (-1)^spin

    # ⚡ Bolt: Pre-calculate all possible ell norms for the given lmax and spin
    abs_spin = abs(spin)
    ell_all = torch.arange(abs_spin, lmax + 1, dtype=torch.float64, device=theta.device)
    all_ell_norms = torch.sqrt((2 * ell_all + 1) / (4.0 * math.pi))

    for m in range(mmax + 1):
        phase = all_phases[m]
        m_phase = -1.0 if (m & 1) else 1.0  # Condon-Shortley convention

        # ⚡ Bolt: Use pre-calculated norms instead of re-calculating for every m
        ell_start = max(m, abs(spin))
        if ell_start > lmax:
            y_blocks.append(
                torch.zeros(
                    (0, theta.shape[0]), dtype=torch.complex128, device=theta.device
                )
            )
            y_conj_blocks.append(
                torch.zeros(
                    (0, theta.shape[0]), dtype=torch.complex128, device=theta.device
                )
            )
            continue

        # Get d-matrix sequence for all required l
        d_seq = _core_mod._wigner_d_l_sequence(ell_start, lmax, m, -spin, theta)

        # ⚡ Bolt: Fast slice from pre-calculated norms
        # ell_start - abs_spin is the offset into our pre-calculated norms tensor
        norms = all_ell_norms[ell_start - abs_spin :]

        # Broadcast and multiply: (spin_phase * m_phase) * norm * d * phase
        combined_scale = (spin_phase * m_phase) * norms.unsqueeze(1)
        y = (combined_scale * d_seq).to(torch.complex128) * phase.unsqueeze(0)

        if ell_start > m:
            n_pre = ell_start - m
            # We need to prepend n_pre rows of zeros
            # y currently has shape [ell_max - ell_start + 1, npix]
            # We want [ell_max - m + 1, npix]
            pre_zeros = torch.zeros(
                (n_pre, theta.shape[0]), dtype=torch.complex128, device=theta.device
            )
            y = torch.cat([pre_zeros, y], dim=0)

        y_blocks.append(y)
        y_conj_blocks.append(torch.conj(y))

    return tuple(y_blocks), tuple(y_conj_blocks)


def _spin_ylm_basis(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    key = (int(nside), bool(nest), int(lmax), int(mmax), int(spin))
    estimated_bytes = (
        2
        * _core_mod.alm_size(int(lmax), int(mmax))
        * _healpix.nside2npix(int(nside))
        * 16
    )
    return _SPIN_YLM_CACHE.get_or_build(
        key,
        lambda: _build_spin_ylm_basis(nside, nest, lmax, mmax, spin),
        estimated_bytes=estimated_bytes,
    )


_SPIN_YLM_CONCAT_CACHE: SpectralCacheNamespace[tuple[Tensor, Tensor]] = (
    _SPECTRAL_CACHE.namespace(
        "spin_ylm_concat",
        device="cpu",
        dtype=torch.complex128,
        max_entry_bytes=_SPIN_CACHE_ENTRY_LIMIT,
    )
)


def _build_spin_ylm_basis_concat(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor]:
    """
    Returns concatenated (nalm, npix) matrices for the spin-weighted basis.
    (Y, Y_conj)
    """
    npix = _healpix.nside2npix(int(nside))
    y_blocks, y_conj_blocks = _spin_ylm_basis(nside, nest, lmax, mmax, spin)

    if not y_blocks:
        device = torch.device("cpu")
        dtype = torch.complex128
        return (
            torch.empty((0, npix), dtype=dtype, device=device),
            torch.empty((0, npix), dtype=dtype, device=device),
        )
    return torch.cat(y_blocks, dim=0), torch.cat(y_conj_blocks, dim=0)


def _spin_map2alm_mats(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor] | None:
    """
    Optionally return cached concatenated map2alm matrices for spin transforms.

    Returns `(Yc_plus_t, Yc_minus_t)` with shape `[npix, nalm]` each, where
    `Yc` is the conjugated spin-weighted basis.
    """
    npix = _healpix.nside2npix(int(nside))
    nalm = _core_mod.alm_size(int(lmax), int(mmax))
    bytes_needed = int(npix * nalm * 16 * 2)
    if bytes_needed > _SPIN_CACHE_ENTRY_LIMIT:
        return None

    key = (int(nside), bool(nest), int(lmax), int(mmax), int(spin))

    def build() -> tuple[Tensor, Tensor]:
        _, y_conj_plus_blocks = _spin_ylm_basis(nside, nest, lmax, mmax, spin)
        _, y_conj_minus_blocks = _spin_ylm_basis(nside, nest, lmax, mmax, -spin)
        yc_plus_t = torch.cat(
            [blk.transpose(0, 1) for blk in y_conj_plus_blocks], dim=1
        ).contiguous()
        yc_minus_t = torch.cat(
            [blk.transpose(0, 1) for blk in y_conj_minus_blocks], dim=1
        ).contiguous()
        return yc_plus_t, yc_minus_t

    return _SPIN_MAP2ALM_MAT_CACHE.get_or_build(
        key, build, estimated_bytes=bytes_needed
    )


def _spin_alm2map_mats(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
    spin: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    """
    Optionally return cached concatenated alm2map matrices for spin transforms.

    Returns `(Y_plus, Y_minus, Y_plus_m0, Y_minus_m0)` where:
    - `Y_plus`, `Y_minus` have shape `[nalm, npix]`.
    - `Y_plus_m0`, `Y_minus_m0` are the `m=0` blocks with shape `[lmax+1, npix]`.
    """
    npix = _healpix.nside2npix(int(nside))
    nalm = _core_mod.alm_size(int(lmax), int(mmax))
    bytes_needed = int(npix * nalm * 16 * 2)
    if bytes_needed > _SPIN_CACHE_ENTRY_LIMIT:
        return None

    key = (int(nside), bool(nest), int(lmax), int(mmax), int(spin))

    def build() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        y_plus_blocks, _ = _spin_ylm_basis(nside, nest, lmax, mmax, spin)
        y_minus_blocks, _ = _spin_ylm_basis(nside, nest, lmax, mmax, -spin)
        y_plus = torch.cat(y_plus_blocks, dim=0).contiguous()
        y_minus = torch.cat(y_minus_blocks, dim=0).contiguous()
        return y_plus, y_minus, y_plus_blocks[0], y_minus_blocks[0]

    return _SPIN_ALM2MAP_MAT_CACHE.get_or_build(
        key, build, estimated_bytes=bytes_needed
    )


def _scalar_map2alm_mat(
    nside: int,
    nest: bool,
    lmax: int,
    mmax: int,
) -> Tensor | None:
    npix = _healpix.nside2npix(int(nside))
    nalm = _core_mod.alm_size(int(lmax), int(mmax))
    bytes_needed = int(npix * nalm * 16)
    if bytes_needed > _SCALAR_CACHE_ENTRY_LIMIT:
        return None
    key = (int(nside), bool(nest), int(lmax), int(mmax))

    def build() -> Tensor:
        _, y_conj_blocks = _core_mod._ylm_basis(nside, nest, lmax, mmax)
        return torch.cat(
            [blk.transpose(0, 1) for blk in y_conj_blocks], dim=1
        ).contiguous()

    return _SCALAR_MAP2ALM_MAT_CACHE.get_or_build(
        key, build, estimated_bytes=bytes_needed
    )


def _build_ring_layout_for_nside(
    nside: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    ns = int(nside)
    npix = _healpix.nside2npix(ns)
    nrings = (4 * ns) - 1
    theta, phi, _, _ = _angles_for_nside(ns, False)

    starts: list[int] = []
    lengths: list[int] = []
    theta_ring: list[float] = []
    phi0_ring: list[float] = []
    cursor = 0
    for ring in range(1, nrings + 1):
        if ring < ns:
            nph = 4 * ring
        elif ring <= (3 * ns):
            nph = 4 * ns
        else:
            nph = 4 * ((4 * ns) - ring)
        starts.append(cursor)
        lengths.append(nph)
        theta_ring.append(float(theta[cursor]))
        phi0_ring.append(float(phi[cursor]))
        cursor += nph
    if cursor != npix:
        raise RuntimeError(f"ring layout mismatch for nside={ns}: {cursor} != {npix}")

    return (
        torch.tensor(starts, dtype=torch.int64),
        torch.tensor(lengths, dtype=torch.int64),
        torch.tensor(theta_ring, dtype=torch.float64),
        torch.tensor(phi0_ring, dtype=torch.float64),
    )


def _ring_layout_for_nside(nside: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    ns = int(nside)
    return _RING_LAYOUT_CACHE.get_or_build(ns, lambda: _build_ring_layout_for_nside(ns))


def _build_ring_grouping(
    starts_cpu: Tensor,
    lengths_cpu: Tensor,
) -> tuple[
    tuple[int, int, int, int, int] | None, tuple[tuple[int, Tensor, Tensor], ...]
]:
    lengths_list = [int(v) for v in lengths_cpu.tolist()]
    if not lengths_list:
        return None, ()

    counts: dict[int, int] = {}
    for nph in lengths_list:
        counts[nph] = counts.get(nph, 0) + 1
    common_len = max(counts, key=lambda length: counts[length])
    common_rings = [i for i, nph in enumerate(lengths_list) if nph == common_len]

    common_block: tuple[int, int, int, int, int] | None = None
    processed = [False] * len(lengths_list)
    if common_rings and all(
        (b - a) == 1 for a, b in zip(common_rings, common_rings[1:])
    ):
        start_ring = int(common_rings[0])
        end_ring = int(common_rings[-1])
        pix_start = int(starts_cpu[start_ring].item())
        pix_end = int(starts_cpu[end_ring].item()) + int(common_len)
        common_block = (int(common_len), start_ring, end_ring, pix_start, pix_end)
        for rid in range(start_ring, end_ring + 1):
            processed[rid] = True

    by_len: dict[int, list[int]] = {}
    for rid, nph in enumerate(lengths_list):
        if processed[rid]:
            continue
        by_len.setdefault(int(nph), []).append(int(rid))

    groups: list[tuple[int, Tensor, Tensor]] = []
    for nph, ring_ids in sorted(by_len.items()):
        ring_idx = torch.tensor(ring_ids, dtype=torch.int64)
        starts_sel = torch.index_select(starts_cpu, 0, ring_idx).unsqueeze(1)
        pix_offsets = torch.arange(int(nph), dtype=torch.int64).unsqueeze(0)
        pix_idx = starts_sel + pix_offsets
        groups.append((int(nph), ring_idx, pix_idx))
    return common_block, tuple(groups)


def _ring_grouping_for_nside(
    nside: int,
) -> tuple[
    tuple[int, int, int, int, int] | None, tuple[tuple[int, Tensor, Tensor], ...]
]:
    key = int(nside)

    def build() -> tuple[
        tuple[int, int, int, int, int] | None,
        tuple[tuple[int, Tensor, Tensor], ...],
    ]:
        starts_cpu, lengths_cpu, _, _ = _ring_layout_for_nside(key)
        return _build_ring_grouping(starts_cpu, lengths_cpu)

    return _RING_GROUP_CACHE.get_or_build(key, build)


def _build_ring_scalar_bases_for_nside(
    nside: int, lmax: int, mmax: int
) -> tuple[Tensor, ...]:
    _, _, theta_ring_cpu, _ = _ring_layout_for_nside(int(nside))
    x_ring = torch.cos(theta_ring_cpu)
    blocks: list[Tensor] = []
    for m in range(int(mmax) + 1):
        p_seq = _legendre_l_sequence(m, x_ring, int(lmax))
        blocks.append(p_seq.to(torch.complex128))
    return tuple(blocks)


def _ring_scalar_bases_for_nside(
    nside: int, lmax: int, mmax: int
) -> tuple[Tensor, ...]:
    key = (int(nside), int(lmax), int(mmax))
    estimated_bytes = (
        _core_mod.alm_size(int(lmax), int(mmax)) * ((4 * int(nside)) - 1) * 16
    )
    return _RING_SCALAR_BASIS_CACHE.get_or_build(
        key,
        lambda: _build_ring_scalar_bases_for_nside(nside, lmax, mmax),
        estimated_bytes=estimated_bytes,
    )


def _ring_scalar_basis_concat(nside: int, lmax: int, mmax: int) -> Tensor:
    """
    Returns concatenated (N_alm, N_rings) basis matrix for scalar Y_lm.
    Uses LRU cache with size limit.
    """
    key = (int(nside), int(lmax), int(mmax))
    bytes_needed = (
        _core_mod.alm_size(int(lmax), int(mmax)) * ((4 * int(nside)) - 1) * 16
    )

    def build() -> Tensor:
        blocks = _ring_scalar_bases_for_nside(nside, lmax, mmax)
        if not blocks:
            return torch.empty(
                (0, 0), dtype=torch.complex128, device=torch.device("cpu")
            )
        return torch.cat(blocks, dim=0)

    return _RING_SCALAR_BASIS_CONCAT_CACHE.get_or_build(
        key, build, estimated_bytes=bytes_needed
    )


def _ring_phi0_phase_for_nside(nside: int, mmax: int, *, sign: int) -> Tensor:
    s = 1 if int(sign) >= 0 else -1
    key = (int(nside), int(mmax), int(s))

    def build() -> Tensor:
        _, _, _, phi0_ring_cpu = _ring_layout_for_nside(int(nside))
        m_vals = torch.arange(int(mmax) + 1, dtype=torch.float64)
        ang = (float(s) * m_vals.unsqueeze(1)) * phi0_ring_cpu.unsqueeze(0)
        return torch.exp(torch.complex(torch.zeros_like(ang), ang)).to(torch.complex128)

    return _RING_PHI0_PHASE_CACHE.get_or_build(key, build)


def _ring_azimuth_phase(nph: int, mmax: int) -> Tensor:
    key = (int(nph), int(mmax))

    def build() -> Tensor:
        n = int(nph)
        m_vals = torch.arange(int(mmax) + 1, dtype=torch.float64).unsqueeze(1)
        j = torch.arange(n, dtype=torch.float64).unsqueeze(0)
        ang = m_vals * (2.0 * math.pi / float(n)) * j
        return torch.exp(torch.complex(torch.zeros_like(ang), ang)).to(torch.complex128)

    return _RING_AZIMUTH_PHASE_CACHE.get_or_build(key, build)


def _ring_alias_indices(nph: int, mmax: int) -> Tensor:
    key = (int(nph), int(mmax))
    return _RING_ALIAS_INDEX_CACHE.get_or_build(
        key,
        lambda: torch.remainder(
            torch.arange(int(mmax) + 1, dtype=torch.int64), int(nph)
        ),
    )


def _ring_spin_conj_indices(nph: int, mmax: int) -> tuple[Tensor, Tensor, Tensor]:
    """
    Cached index vectors for spin conjugate mode extraction:
      +m indices, -m indices, and concatenated (+m|-m) indices.
    """
    key = (int(nph), int(mmax))

    def build() -> tuple[Tensor, Tensor, Tensor]:
        n = int(nph)
        mm = int(mmax)
        if n > mm:
            idx_plus = torch.arange(mm + 1, dtype=torch.int64)
        else:
            idx_plus = _ring_alias_indices(n, mm)
        idx_minus = torch.remainder(-idx_plus, n)
        return idx_plus, idx_minus, torch.cat([idx_plus, idx_minus], dim=0)

    return _RING_SPIN_CONJ_INDEX_CACHE.get_or_build(key, build)


def _build_ring_spin_bases_for_nside(
    nside: int, lmax: int, mmax: int, spin: int
) -> tuple[Tensor, ...]:
    _, _, theta_ring_cpu, _ = _ring_layout_for_nside(int(nside))
    sp = int(spin)
    spin_phase = -1.0 if (sp & 1) else 1.0
    blocks: list[Tensor] = []
    for m in range(int(mmax) + 1):
        m_phase = -1.0 if (m & 1) else 1.0

        ell_start = max(m, abs(sp))
        if ell_start > int(lmax):
            blocks.append(
                torch.zeros((0, theta_ring_cpu.shape[0]), dtype=torch.complex128)
            )
            continue

        d_seq = _core_mod._wigner_d_l_sequence(
            ell_start, int(lmax), m, -sp, theta_ring_cpu
        )

        ell_vec = torch.arange(
            ell_start, int(lmax) + 1, dtype=torch.float64, device=theta_ring_cpu.device
        )
        norms = torch.sqrt((2 * ell_vec + 1) / (4.0 * math.pi))

        # norms: [L], d_seq: [L, N]
        block = (spin_phase * m_phase * norms.unsqueeze(1) * d_seq).to(torch.complex128)

        if ell_start > m:
            n_pre = ell_start - m
            pre_zeros = torch.zeros(
                (n_pre, theta_ring_cpu.shape[0]), dtype=torch.complex128
            )
            block = torch.cat([pre_zeros, block], dim=0)

        blocks.append(block)
    return tuple(blocks)


def _ring_spin_bases_for_nside(
    nside: int, lmax: int, mmax: int, spin: int
) -> tuple[Tensor, ...]:
    key = (int(nside), int(lmax), int(mmax), int(spin))
    estimated_bytes = (
        _core_mod.alm_size(int(lmax), int(mmax)) * ((4 * int(nside)) - 1) * 16
    )
    return _RING_SPIN_BASIS_CACHE.get_or_build(
        key,
        lambda: _build_ring_spin_bases_for_nside(nside, lmax, mmax, spin),
        estimated_bytes=estimated_bytes,
    )


_RING_SPIN_CONCAT_CACHE: SpectralCacheNamespace[tuple[Tensor, Tensor]] = (
    _SPECTRAL_CACHE.namespace(
        "ring_spin_concat",
        device="cpu",
        dtype=torch.complex128,
        max_entry_bytes=_SPIN_CACHE_ENTRY_LIMIT,
    )
)


def _ring_spin_basis_concat(
    nside: int, lmax: int, mmax: int, spin: int
) -> tuple[Tensor, Tensor]:
    """
    Returns concatenated (N_alm, N_rings) basis matrices for +spin and -spin.
    """
    key = (int(nside), int(lmax), int(mmax), int(spin))
    bytes_needed = (
        2 * _core_mod.alm_size(int(lmax), int(mmax)) * ((4 * int(nside)) - 1) * 16
    )

    def build() -> tuple[Tensor, Tensor]:
        y_plus_blocks = _ring_spin_bases_for_nside(nside, lmax, mmax, spin)
        y_minus_blocks = _ring_spin_bases_for_nside(nside, lmax, mmax, -spin)
        if not y_plus_blocks:
            device = torch.device("cpu")
            dtype = torch.complex128
            return (
                torch.empty((0, 0), dtype=dtype, device=device),
                torch.empty((0, 0), dtype=dtype, device=device),
            )
        return torch.cat(y_plus_blocks, dim=0), torch.cat(y_minus_blocks, dim=0)

    return _RING_SPIN_CONCAT_CACHE.get_or_build(
        key, build, estimated_bytes=bytes_needed
    )


# Late-bound so monkeypatching ``spectral`` attributes still reaches this code.
from . import spectral as _core_mod  # noqa: E402
