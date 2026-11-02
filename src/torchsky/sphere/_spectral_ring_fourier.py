"""Ring Fourier mode analysis and synthesis for HEALPix spectral transforms."""

import torch
from torch import Tensor

from .. import _ops as _native_ops


from ._spectral_autograd import (
    _accumulate_add,
)
from ._spectral_basis import (
    _build_ring_grouping,
    _ring_alias_indices,
    _ring_azimuth_phase,
    _ring_grouping_for_nside,
    _ring_spin_conj_indices,
)
from ._spectral_config import (
    _RING_FOURIER_MODES_CPP_MAX_M,
    _RING_FOURIER_SYNTH_CPP_MAX_M,
    _RING_SMALL_DFT_MAX_NPH,
)


def _ring_fourier_modes(
    rows: Tensor,
    *,
    starts_cpu: Tensor,
    lengths_cpu: Tensor,
    mmax: int,
    nside: int | None = None,
) -> Tensor:
    # rows: [nrows, npix] complex tensor.
    device = rows.device
    dtype = rows.dtype
    nrings = int(lengths_cpu.numel())

    use_cpp = (
        (_core_mod._RING_FOURIER_MODES_CPP_ENABLE or _core_mod._RING_FOURIER_CPP_ENABLE)
        and _native_ops.has_spectral_dispatcher()
        and device.type == "cpu"
        and int(mmax) <= _RING_FOURIER_MODES_CPP_MAX_M
        and (not rows.requires_grad)
    )
    if use_cpp:
        return _native_ops.spectral_ring_fourier_modes(
            rows.contiguous(),
            starts_cpu.contiguous(),
            lengths_cpu.contiguous(),
            int(mmax),
        )

    if nside is not None:
        common_block, groups = _ring_grouping_for_nside(int(nside))
    else:
        common_block, groups = _build_ring_grouping(starts_cpu, lengths_cpu)

    out = torch.zeros(
        (int(rows.shape[0]), int(mmax) + 1, nrings), dtype=dtype, device=device
    )
    m_int = torch.arange(int(mmax) + 1, dtype=torch.int64, device=device)

    if common_block is not None:
        common_len, start_ring, end_ring, pix_start, pix_end = common_block
        n_common = end_ring - start_ring + 1
        belt_pixels = rows[:, pix_start:pix_end]
        belt_stack = belt_pixels.view(rows.shape[0], n_common, common_len)
        if int(common_len) <= _RING_SMALL_DFT_MAX_NPH:
            if int(common_len) > int(mmax):
                phase = _ring_azimuth_phase(int(common_len), int(mmax)).to(
                    device=device, dtype=dtype
                )
                dft_sel = torch.matmul(belt_stack, torch.conj(phase).transpose(0, 1))
            else:
                phase_base = _ring_azimuth_phase(
                    int(common_len), int(common_len) - 1
                ).to(device=device, dtype=dtype)
                dft_base = torch.matmul(
                    belt_stack, torch.conj(phase_base).transpose(0, 1)
                )
                col_idx = torch.remainder(m_int, int(common_len))
                dft_sel = torch.index_select(dft_base, 2, col_idx)
            out[:, :, start_ring : end_ring + 1] = dft_sel.permute(0, 2, 1)
        else:
            fft_block = torch.fft.fft(belt_stack, dim=-1)
            if int(common_len) > int(mmax):
                # No aliasing when nph > mmax: use direct contiguous frequency slice.
                fft_sel = fft_block[:, :, : int(mmax) + 1]
            else:
                col_idx = torch.remainder(m_int, int(common_len))
                fft_sel = torch.index_select(fft_block, 2, col_idx)
            out[:, :, start_ring : end_ring + 1] = fft_sel.permute(0, 2, 1)

    for nph, ring_idx_cpu, pix_idx_cpu in groups:
        ring_idx = (
            ring_idx_cpu if device.type == "cpu" else ring_idx_cpu.to(device=device)
        )
        pix_idx = pix_idx_cpu if device.type == "cpu" else pix_idx_cpu.to(device=device)
        ring_stack = rows[:, pix_idx]
        if int(nph) <= _RING_SMALL_DFT_MAX_NPH:
            if int(nph) > int(mmax):
                phase = _ring_azimuth_phase(int(nph), int(mmax)).to(
                    device=device, dtype=dtype
                )
                dft_sel = torch.matmul(ring_stack, torch.conj(phase).transpose(0, 1))
            else:
                phase_base = _ring_azimuth_phase(int(nph), int(nph) - 1).to(
                    device=device, dtype=dtype
                )
                dft_base = torch.matmul(
                    ring_stack, torch.conj(phase_base).transpose(0, 1)
                )
                dft_sel = torch.index_select(
                    dft_base, 2, torch.remainder(m_int, int(nph))
                )
            out.index_copy_(2, ring_idx, dft_sel.permute(0, 2, 1))
        else:
            fft_block = torch.fft.fft(ring_stack, dim=-1)
            if int(nph) > int(mmax):
                fft_sel = fft_block[:, :, : int(mmax) + 1]
            else:
                fft_sel = torch.index_select(
                    fft_block, 2, torch.remainder(m_int, int(nph))
                )
            out.index_copy_(2, ring_idx, fft_sel.permute(0, 2, 1))

    return out


def _ring_fourier_modes_spin_conj(
    p_plus: Tensor,
    *,
    starts_cpu: Tensor,
    lengths_cpu: Tensor,
    mmax: int,
    nside: int | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Specialized spin helper for p_minus = conj(p_plus).

    Returns (s_plus, s_minus) with shape (mmax+1, nrings), where
    s_plus[m, r]  = FFT(p_plus_ring_r)[m mod nph_r]
    s_minus[m, r] = FFT(conj(p_plus_ring_r))[m mod nph_r]
                  = conj(FFT(p_plus_ring_r)[(-m) mod nph_r]).
    """
    if p_plus.ndim != 1:
        raise ValueError("p_plus must be a 1D complex tensor")
    device = p_plus.device
    dtype = p_plus.dtype
    nrings = int(lengths_cpu.numel())

    use_cpp = (
        (_core_mod._RING_FOURIER_MODES_CPP_ENABLE or _core_mod._RING_FOURIER_CPP_ENABLE)
        and _native_ops.has_spectral_dispatcher()
        and device.type == "cpu"
        and int(mmax) <= _RING_FOURIER_MODES_CPP_MAX_M
        and (not p_plus.requires_grad)
    )
    if use_cpp:
        out_cpp = _native_ops.spectral_ring_fourier_modes_spin_conj(
            p_plus.contiguous(),
            starts_cpu.contiguous(),
            lengths_cpu.contiguous(),
            int(mmax),
        )
        return out_cpp

    if nside is not None:
        common_block, groups = _ring_grouping_for_nside(int(nside))
    else:
        common_block, groups = _build_ring_grouping(starts_cpu, lengths_cpu)

    s_plus = torch.zeros((int(mmax) + 1, nrings), dtype=dtype, device=device)
    s_minus = torch.zeros_like(s_plus)
    k = int(mmax) + 1

    if common_block is not None:
        common_len, start_ring, end_ring, pix_start, pix_end = common_block
        n_common = end_ring - start_ring + 1
        belt_pixels = p_plus[pix_start:pix_end]
        belt_stack = belt_pixels.view(n_common, common_len)
        fft_block = torch.fft.fft(belt_stack, dim=-1)
        idx_plus_cpu, idx_minus_cpu, idx_both_cpu = _ring_spin_conj_indices(
            int(common_len), int(mmax)
        )
        if device.type == "cpu":
            idx_minus = idx_minus_cpu
            idx_both = idx_both_cpu
        else:
            idx_minus = idx_minus_cpu.to(device=device)
            idx_both = idx_both_cpu.to(device=device)
        if int(common_len) > int(mmax):
            plus_sel = fft_block[:, :k]
            minus_sel = torch.conj(torch.index_select(fft_block, 1, idx_minus))
        else:
            both_sel = torch.index_select(fft_block, 1, idx_both)
            plus_sel = both_sel[:, :k]
            minus_sel = torch.conj(both_sel[:, k:])
        s_plus[:, start_ring : end_ring + 1] = plus_sel.transpose(0, 1)
        s_minus[:, start_ring : end_ring + 1] = minus_sel.transpose(0, 1)

    n_group_rings = 0
    for _, ring_idx_cpu, _ in groups:
        n_group_rings += int(ring_idx_cpu.numel())
    if n_group_rings > 0:
        plus_all = torch.empty((k, n_group_rings), dtype=dtype, device=device)
        minus_all = torch.empty_like(plus_all)
        if device.type == "cpu":
            ring_idx_all = torch.empty((n_group_rings,), dtype=torch.int64)
        else:
            ring_idx_all = torch.empty(
                (n_group_rings,), dtype=torch.int64, device=device
            )
    else:
        plus_all = None
        minus_all = None
        ring_idx_all = None
    write_off = 0

    for nph, ring_idx_cpu, pix_idx_cpu in groups:
        ring_idx = (
            ring_idx_cpu if device.type == "cpu" else ring_idx_cpu.to(device=device)
        )
        pix_idx = pix_idx_cpu if device.type == "cpu" else pix_idx_cpu.to(device=device)
        ring_stack = p_plus[pix_idx]
        fft_block = torch.fft.fft(ring_stack, dim=-1)
        idx_plus_cpu, idx_minus_cpu, idx_both_cpu = _ring_spin_conj_indices(
            int(nph), int(mmax)
        )
        if device.type == "cpu":
            idx_minus = idx_minus_cpu
            idx_both = idx_both_cpu
        else:
            idx_minus = idx_minus_cpu.to(device=device)
            idx_both = idx_both_cpu.to(device=device)
        if int(nph) > int(mmax):
            plus_sel = fft_block[:, :k]
            minus_sel = torch.conj(torch.index_select(fft_block, 1, idx_minus))
        else:
            both_sel = torch.index_select(fft_block, 1, idx_both)
            plus_sel = both_sel[:, :k]
            minus_sel = torch.conj(both_sel[:, k:])
        ng = int(ring_idx.numel())
        if (
            ng > 0
            and plus_all is not None
            and minus_all is not None
            and ring_idx_all is not None
        ):
            plus_all[:, write_off : write_off + ng] = plus_sel.transpose(0, 1)
            minus_all[:, write_off : write_off + ng] = minus_sel.transpose(0, 1)
            ring_idx_all[write_off : write_off + ng] = ring_idx
            write_off += ng

    if (
        n_group_rings > 0
        and plus_all is not None
        and minus_all is not None
        and ring_idx_all is not None
    ):
        s_plus.index_copy_(1, ring_idx_all, plus_all)
        s_minus.index_copy_(1, ring_idx_all, minus_all)

    return s_plus, s_minus


def _ring_fourier_synthesis(
    s_m: Tensor,
    *,
    starts_cpu: Tensor,
    lengths_cpu: Tensor,
    nside: int | None = None,
) -> Tensor:
    """
    Evaluate ring Fourier series from m>=0 coefficients.
    s_m: (mmax+1, nrings), complex128
    Returns: (npix,), complex128
    """
    if s_m.ndim == 2:
        s_m_batched = s_m.unsqueeze(0)
        squeeze_out = True
    elif s_m.ndim == 3:
        s_m_batched = s_m
        squeeze_out = False
    else:
        raise ValueError(
            "s_m must have shape (mmax+1, nrings) or (nrows, mmax+1, nrings)"
        )

    mmax = int(s_m_batched.shape[1]) - 1
    npix = int(starts_cpu[-1].item()) + int(lengths_cpu[-1].item())
    device = s_m_batched.device
    dtype = s_m_batched.dtype
    nrows = int(s_m_batched.shape[0])

    use_cpp = (
        (_core_mod._RING_FOURIER_SYNTH_CPP_ENABLE or _core_mod._RING_FOURIER_CPP_ENABLE)
        and _native_ops.has_spectral_dispatcher()
        and device.type == "cpu"
        and int(mmax) <= _RING_FOURIER_SYNTH_CPP_MAX_M
        and (not s_m_batched.requires_grad)
    )
    if use_cpp:
        out_cpp = _native_ops.spectral_ring_fourier_synthesis(
            s_m_batched.contiguous(),
            starts_cpu.contiguous(),
            lengths_cpu.contiguous(),
            npix,
        )
        if squeeze_out:
            return out_cpp[0]
        return out_cpp

    if nside is not None:
        common_block, groups = _ring_grouping_for_nside(int(nside))
    else:
        common_block, groups = _build_ring_grouping(starts_cpu, lengths_cpu)

    out_map = torch.zeros((nrows, npix), dtype=dtype, device=device)

    if common_block is not None:
        common_len, start_ring, end_ring, pix_start, pix_end = common_block
        s_group = s_m_batched[:, :, start_ring : end_ring + 1]
        s_group_perm = s_group.permute(0, 2, 1)
        if int(common_len) <= _RING_SMALL_DFT_MAX_NPH:
            phase = _ring_azimuth_phase(int(common_len), int(mmax)).to(
                device=device, dtype=dtype
            )
            map_vals = torch.matmul(s_group_perm, phase)
        else:
            if int(common_len) > int(mmax):
                map_vals = torch.fft.ifft(
                    s_group_perm * float(common_len), n=int(common_len), dim=-1
                )
            else:
                s_fold = torch.zeros(
                    (s_group_perm.shape[0], s_group_perm.shape[1], int(common_len)),
                    dtype=dtype,
                    device=device,
                )
                alias_idx = _ring_alias_indices(int(common_len), int(mmax))
                alias_idx = (
                    alias_idx if device.type == "cpu" else alias_idx.to(device=device)
                )
                s_fold = _accumulate_add(
                    s_fold, 2, alias_idx, s_group_perm, scatter=False
                )
                map_vals = torch.fft.ifft(s_fold * float(common_len), dim=-1)
        out_map[:, pix_start:pix_end] = map_vals.reshape(nrows, -1)

    for nph, ring_idx_cpu, pix_idx_cpu in groups:
        ring_idx = (
            ring_idx_cpu if device.type == "cpu" else ring_idx_cpu.to(device=device)
        )
        pix_idx = pix_idx_cpu if device.type == "cpu" else pix_idx_cpu.to(device=device)
        s_group = torch.index_select(s_m_batched, 2, ring_idx)
        s_group_perm = s_group.permute(0, 2, 1)
        if int(nph) <= _RING_SMALL_DFT_MAX_NPH:
            phase = _ring_azimuth_phase(int(nph), int(mmax)).to(
                device=device, dtype=dtype
            )
            map_vals = torch.matmul(s_group_perm, phase)
        else:
            if int(nph) > int(mmax):
                map_vals = torch.fft.ifft(s_group_perm * float(nph), n=int(nph), dim=-1)
            else:
                s_fold = torch.zeros(
                    (s_group_perm.shape[0], s_group_perm.shape[1], int(nph)),
                    dtype=dtype,
                    device=device,
                )
                alias_idx = _ring_alias_indices(int(nph), int(mmax))
                alias_idx = (
                    alias_idx if device.type == "cpu" else alias_idx.to(device=device)
                )
                s_fold = _accumulate_add(
                    s_fold, 2, alias_idx, s_group_perm, scatter=False
                )
                map_vals = torch.fft.ifft(s_fold * float(nph), dim=-1)
        out_map[:, pix_idx] = map_vals

    if squeeze_out:
        return out_map[0]
    return out_map


# Late-bound so monkeypatching ``spectral`` attributes still reaches this code.
from . import spectral as _core_mod  # noqa: E402
