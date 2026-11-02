"""Autograd functions and CPU recurrences for ring-based HEALPix spectral transforms."""

import torch
from torch import Tensor

from .. import _ops as _native_ops


from ._spectral_basis import (
    _ring_layout_for_nside,
)


def _accumulate_add(
    dst: Tensor, dim: int, index: Tensor, src: Tensor, *, scatter: bool
) -> Tensor:
    """In-place add. MPS has no complex scatter/index_add; split real/imag."""
    if src.is_complex() and src.device.type == "mps":
        real = torch.zeros(dst.shape, dtype=src.real.dtype, device=src.device)
        imag = torch.zeros_like(real)
        if scatter:
            real.scatter_add_(dim, index, src.real)
            imag.scatter_add_(dim, index, src.imag)
        else:
            real.index_add_(dim, index, src.real)
            imag.index_add_(dim, index, src.imag)
        return torch.complex(real, imag)
    if scatter:
        dst.scatter_add_(dim, index, src)
    else:
        dst.index_add_(dim, index, src)
    return dst


class ScalarSHTIntegrate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, s_modes: Tensor, nside: int, lmax: int, mmax: int, pix_w: float
    ) -> Tensor:
        ctx.nside = nside
        ctx.lmax = lmax
        ctx.mmax = mmax
        ctx.pix_w = pix_w

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(nside)
        nrings = int(theta_ring_cpu.numel())
        weight = torch.full((nrings,), pix_w, dtype=torch.float64, device="cpu")

        s_modes_cpu = s_modes.detach().to(dtype=torch.complex128, device="cpu")

        if _native_ops.has_spectral_dispatcher():
            out = _native_ops.spectral_scalar_integrate_recurrence(
                s_modes_cpu.contiguous(),
                theta_ring_cpu.contiguous(),
                weight.contiguous(),
                int(lmax),
                int(mmax),
            )
        else:
            s_minus = s_modes_cpu.clone()
            alms_stacked = _native_ops.spectral_spin_integrate_recurrence(
                s_modes_cpu.contiguous(),
                s_minus.contiguous(),
                theta_ring_cpu.contiguous(),
                weight.contiguous(),
                int(lmax),
                int(mmax),
                0,
            )
            out = (alms_stacked[0] + alms_stacked[1]) * 0.5

        return out.to(device=s_modes.device)

    @staticmethod
    def backward(ctx, grad_alm: Tensor):
        grad_alm_cpu = grad_alm.detach().to(device="cpu", dtype=torch.complex128)
        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(ctx.nside)

        if _native_ops.has_spectral_dispatcher():
            grad_s = _native_ops.spectral_scalar_interpolate_recurrence(
                grad_alm_cpu.contiguous(),
                theta_ring_cpu.contiguous(),
                int(ctx.lmax),
                int(ctx.mmax),
            )
        else:
            alm_stack = torch.stack([grad_alm_cpu, grad_alm_cpu], dim=0)
            s_modes_stacked = _native_ops.spectral_spin_interpolate_recurrence(
                alm_stack.contiguous(),
                theta_ring_cpu.contiguous(),
                int(ctx.lmax),
                int(ctx.mmax),
                0,
            )
            grad_s = s_modes_stacked[0]

        grad_s = grad_s * ctx.pix_w
        return grad_s.to(device=grad_alm.device), None, None, None, None


class ScalarSHTInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alm: Tensor, nside: int, lmax: int, mmax: int) -> Tensor:
        ctx.nside = nside
        ctx.lmax = lmax
        ctx.mmax = mmax

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(nside)
        alm_cpu = alm.detach().to(dtype=torch.complex128, device="cpu")

        if _native_ops.has_spectral_dispatcher():
            out = _native_ops.spectral_scalar_interpolate_recurrence(
                alm_cpu.contiguous(),
                theta_ring_cpu.contiguous(),
                int(lmax),
                int(mmax),
            )
        else:
            alm_stack = torch.stack([alm_cpu, alm_cpu], dim=0)
            s_modes_stacked = _native_ops.spectral_spin_interpolate_recurrence(
                alm_stack.contiguous(),
                theta_ring_cpu.contiguous(),
                int(lmax),
                int(mmax),
                0,
            )
            out = s_modes_stacked[0]

        return out.to(device=alm.device)

    @staticmethod
    def backward(ctx, grad_s_modes: Tensor):
        grad_s_cpu = grad_s_modes.detach().to(device="cpu", dtype=torch.complex128)

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(ctx.nside)
        nrings = int(theta_ring_cpu.numel())
        weight = torch.ones((nrings,), dtype=torch.float64, device="cpu")

        if _native_ops.has_spectral_dispatcher():
            grad_alm = _native_ops.spectral_scalar_integrate_recurrence(
                grad_s_cpu.contiguous(),
                theta_ring_cpu.contiguous(),
                weight.contiguous(),
                int(ctx.lmax),
                int(ctx.mmax),
            )
        else:
            s_minus = grad_s_cpu.clone()
            alms_stacked = _native_ops.spectral_spin_integrate_recurrence(
                grad_s_cpu.contiguous(),
                s_minus.contiguous(),
                theta_ring_cpu.contiguous(),
                weight.contiguous(),
                int(ctx.lmax),
                int(ctx.mmax),
                0,
            )
            grad_alm = (alms_stacked[0] + alms_stacked[1]) * 0.5

        return grad_alm.to(device=grad_s_modes.device), None, None, None


class SpinSHTIntegrate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        s_plus: Tensor,
        s_minus: Tensor,
        nside: int,
        lmax: int,
        mmax: int,
        spin: int,
        pix_w: float,
    ) -> tuple[Tensor, Tensor]:
        ctx.nside = nside
        ctx.lmax = lmax
        ctx.mmax = mmax
        ctx.spin = spin
        ctx.pix_w = pix_w

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(nside)
        nrings = int(theta_ring_cpu.numel())
        w_vec = torch.full((nrings,), pix_w, dtype=torch.float64, device="cpu")

        sp = s_plus.detach().to(dtype=torch.complex128, device="cpu")
        sm = s_minus.detach().to(dtype=torch.complex128, device="cpu")

        alms_stacked = _native_ops.spectral_spin_integrate_recurrence(
            sp.contiguous(),
            sm.contiguous(),
            theta_ring_cpu.contiguous(),
            w_vec.contiguous(),
            int(lmax),
            int(mmax),
            int(spin),
        )
        return alms_stacked[0].to(device=s_plus.device), alms_stacked[1].to(
            device=s_minus.device
        )

    @staticmethod
    def backward(ctx, grad_a_plus: Tensor, grad_a_minus: Tensor):
        gap = grad_a_plus.detach().to(dtype=torch.complex128, device="cpu")
        gam = grad_a_minus.detach().to(dtype=torch.complex128, device="cpu")

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(int(ctx.nside))
        alm_stack = torch.stack([gap, gam], dim=0)

        s_modes_stacked = _native_ops.spectral_spin_interpolate_recurrence(
            alm_stack.contiguous(),
            theta_ring_cpu.contiguous(),
            int(ctx.lmax),
            int(ctx.mmax),
            int(ctx.spin),
        )

        gsp = s_modes_stacked[0] * ctx.pix_w
        gsm = s_modes_stacked[1] * ctx.pix_w

        return (
            gsp.to(device=grad_a_plus.device),
            gsm.to(device=grad_a_plus.device),
            None,
            None,
            None,
            None,
            None,
        )


class SpinSHTInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a_plus: Tensor,
        a_minus: Tensor,
        nside: int,
        lmax: int,
        mmax: int,
        spin: int,
    ) -> tuple[Tensor, Tensor]:
        ctx.nside = nside
        ctx.lmax = lmax
        ctx.mmax = mmax
        ctx.spin = spin

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(nside)

        ap = a_plus.detach().to(dtype=torch.complex128, device="cpu")
        am = a_minus.detach().to(dtype=torch.complex128, device="cpu")
        alm_stack = torch.stack([ap, am], dim=0)

        s_modes_stacked = _native_ops.spectral_spin_interpolate_recurrence(
            alm_stack.contiguous(),
            theta_ring_cpu.contiguous(),
            int(lmax),
            int(mmax),
            int(spin),
        )
        return s_modes_stacked[0].to(device=a_plus.device), s_modes_stacked[1].to(
            device=a_minus.device
        )

    @staticmethod
    def backward(ctx, grad_s_plus: Tensor, grad_s_minus: Tensor):
        gsp = grad_s_plus.detach().to(dtype=torch.complex128, device="cpu")
        gsm = grad_s_minus.detach().to(dtype=torch.complex128, device="cpu")

        _, _, theta_ring_cpu, _ = _ring_layout_for_nside(ctx.nside)
        nrings = int(theta_ring_cpu.numel())
        w_vec = torch.ones((nrings,), dtype=torch.float64, device="cpu")

        alms_stacked = _native_ops.spectral_spin_integrate_recurrence(
            gsp.contiguous(),
            gsm.contiguous(),
            theta_ring_cpu.contiguous(),
            w_vec.contiguous(),
            int(ctx.lmax),
            int(ctx.mmax),
            int(ctx.spin),
        )

        return (
            alms_stacked[0].to(device=grad_s_plus.device),
            alms_stacked[1].to(device=grad_s_minus.device),
            None,
            None,
            None,
            None,
        )


def _scalar_ring_integrate_recurrence_cpu(
    s_modes: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
    pix_w: float,
) -> Tensor:
    """
    Integrate scalar Fourier modes to alms using C++ recurrence (spin=0).

    s_modes: (mmax+1, nrings) complex tensor of Fourier modes per ring
    Returns: (nalm,) complex tensor of alm coefficients
    """
    return ScalarSHTIntegrate.apply(
        s_modes, int(nside), int(lmax), int(mmax), float(pix_w)
    )


def _scalar_ring_interpolate_recurrence_cpu(
    alm: Tensor,
    *,
    nside: int,
    lmax: int,
    mmax: int,
) -> Tensor:
    """
    Interpolate scalar alms to Fourier modes using C++ recurrence (spin=0).

    alm: (nalm,) complex tensor of alm coefficients
    Returns: (mmax+1, nrings) complex tensor of Fourier modes per ring
    """
    return ScalarSHTInterpolate.apply(alm, int(nside), int(lmax), int(mmax))
