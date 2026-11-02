from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


def _nalm(lmax: int, mmax: int) -> int:
    torch._check(lmax >= 0, lambda: "lmax must be non-negative")
    torch._check(mmax >= 0, lambda: "mmax must be non-negative")
    torch._check(mmax <= lmax, lambda: "mmax must not exceed lmax")
    return ((mmax + 1) * (lmax + 1)) - ((mmax * (mmax + 1)) // 2)


def _check_cpu_metadata(*tensors: Tensor) -> None:
    for tensor in tensors:
        torch._check(tensor.dtype == torch.int64, lambda: "ring metadata must be int64")
        torch._check(tensor.ndim == 1, lambda: "ring metadata must be rank 1")


def _check_layout(starts: Tensor, lengths: Tensor) -> None:
    _check_cpu_metadata(starts, lengths)
    torch._check(
        starts.shape == lengths.shape, lambda: "ring metadata shapes must match"
    )


def _check_c128(tensor: Tensor, rank: int, name: str) -> None:
    torch._check(tensor.dtype == torch.complex128, lambda: f"{name} must be complex128")
    torch._check(tensor.ndim == rank, lambda: f"{name} must be rank {rank}")


def _check_f64(tensor: Tensor, rank: int, name: str) -> None:
    torch._check(tensor.dtype == torch.float64, lambda: f"{name} must be float64")
    torch._check(tensor.ndim == rank, lambda: f"{name} must be rank {rank}")


def _fake_scalar_direct(
    alm: Tensor, cos_theta: Tensor, phi: Tensor, lmax: int, mmax: int
) -> Tensor:
    _check_c128(alm, 2, "alm")
    _check_f64(cos_theta, 1, "cos_theta")
    _check_f64(phi, 1, "phi")
    torch._check(
        cos_theta.shape == phi.shape, lambda: "angular coordinate shapes must match"
    )
    torch._check(alm.shape[1] == _nalm(lmax, mmax), lambda: "alm length mismatch")
    return torch.empty(
        (alm.shape[0], cos_theta.shape[0]), dtype=torch.float64, device=alm.device
    )


def _fake_fourier_modes(
    rows: Tensor, starts: Tensor, lengths: Tensor, mmax: int
) -> Tensor:
    _check_c128(rows, 2, "rows")
    _check_layout(starts, lengths)
    torch._check(mmax >= 0, lambda: "mmax must be non-negative")
    return torch.empty(
        (rows.shape[0], mmax + 1, lengths.shape[0]),
        dtype=rows.dtype,
        device=rows.device,
    )


def _fake_fourier_modes_spin_conj(
    row: Tensor, starts: Tensor, lengths: Tensor, mmax: int
) -> tuple[Tensor, Tensor]:
    _check_c128(row, 1, "row")
    _check_layout(starts, lengths)
    torch._check(mmax >= 0, lambda: "mmax must be non-negative")
    shape = (mmax + 1, lengths.shape[0])
    return (
        torch.empty(shape, dtype=row.dtype, device=row.device),
        torch.empty(shape, dtype=row.dtype, device=row.device),
    )


def _fake_fourier_synthesis(
    modes: Tensor, starts: Tensor, lengths: Tensor, npix: int
) -> Tensor:
    _check_c128(modes, 3, "modes")
    _check_layout(starts, lengths)
    torch._check(modes.shape[2] == lengths.shape[0], lambda: "ring dimension mismatch")
    torch._check(npix >= 0, lambda: "npix must be non-negative")
    return torch.empty((modes.shape[0], npix), dtype=modes.dtype, device=modes.device)


def _fake_spin_finalize(
    s_plus: Tensor, s_minus: Tensor, starts: Tensor, lengths: Tensor, npix: int
) -> Tensor:
    _check_c128(s_plus, 2, "s_plus")
    _check_c128(s_minus, 2, "s_minus")
    _check_layout(starts, lengths)
    torch._check(s_plus.shape == s_minus.shape, lambda: "spin mode shapes must match")
    torch._check(s_plus.shape[1] == lengths.shape[0], lambda: "ring dimension mismatch")
    torch._check(npix >= 0, lambda: "npix must be non-negative")
    return torch.empty((2, npix), dtype=torch.float64, device=s_plus.device)


def _fake_spin_map2alm_concat(
    p_plus: Tensor,
    starts: Tensor,
    lengths: Tensor,
    phase0_neg: Tensor,
    y_plus: Tensor,
    y_minus: Tensor,
    lmax: int,
    mmax: int,
    pixel_weight: float,
) -> tuple[Tensor, Tensor]:
    del pixel_weight
    _check_c128(p_plus, 1, "p_plus")
    _check_layout(starts, lengths)
    _check_c128(phase0_neg, 2, "phase0_neg")
    _check_c128(y_plus, 2, "y_plus")
    _check_c128(y_minus, 2, "y_minus")
    nalm = _nalm(lmax, mmax)
    nrings = lengths.shape[0]
    torch._check(phase0_neg.shape == (mmax + 1, nrings), lambda: "phase shape mismatch")
    torch._check(y_plus.shape == (nalm, nrings), lambda: "basis shape mismatch")
    torch._check(y_minus.shape == y_plus.shape, lambda: "basis shapes must match")
    output = torch.empty((nalm,), dtype=p_plus.dtype, device=p_plus.device)
    return output, torch.empty_like(output)


def _fake_spin_interpolate_concat(
    coeff_plus: Tensor,
    coeff_minus: Tensor,
    y_plus: Tensor,
    y_minus: Tensor,
    lmax: int,
    mmax: int,
) -> tuple[Tensor, Tensor]:
    nalm = _nalm(lmax, mmax)
    _check_c128(coeff_plus, 1, "coeff_plus")
    _check_c128(coeff_minus, 1, "coeff_minus")
    _check_c128(y_plus, 2, "y_plus")
    _check_c128(y_minus, 2, "y_minus")
    torch._check(coeff_plus.shape == (nalm,), lambda: "coefficient length mismatch")
    torch._check(
        coeff_minus.shape == coeff_plus.shape, lambda: "coefficient shapes must match"
    )
    torch._check(y_plus.shape[0] == nalm, lambda: "basis row count mismatch")
    torch._check(y_minus.shape == y_plus.shape, lambda: "basis shapes must match")
    output = torch.empty(
        (mmax + 1, y_plus.shape[1]), dtype=coeff_plus.dtype, device=coeff_plus.device
    )
    return output, torch.empty_like(output)


def _fake_spin_integrate_concat(
    s_plus: Tensor,
    s_minus: Tensor,
    y_plus: Tensor,
    y_minus: Tensor,
    lmax: int,
    mmax: int,
    pixel_weight: float,
) -> tuple[Tensor, Tensor]:
    del pixel_weight
    nalm = _nalm(lmax, mmax)
    _check_c128(s_plus, 2, "s_plus")
    _check_c128(s_minus, 2, "s_minus")
    _check_c128(y_plus, 2, "y_plus")
    _check_c128(y_minus, 2, "y_minus")
    torch._check(s_plus.shape == s_minus.shape, lambda: "spin mode shapes must match")
    torch._check(s_plus.shape[0] == mmax + 1, lambda: "mode count mismatch")
    torch._check(
        y_plus.shape == (nalm, s_plus.shape[1]), lambda: "basis shape mismatch"
    )
    torch._check(y_minus.shape == y_plus.shape, lambda: "basis shapes must match")
    output = torch.empty((nalm,), dtype=s_plus.dtype, device=s_plus.device)
    return output, torch.empty_like(output)


def _fake_spin_integrate_recurrence(
    s_plus: Tensor,
    s_minus: Tensor,
    theta: Tensor,
    weight: Tensor,
    lmax: int,
    mmax: int,
    spin: int,
) -> Tensor:
    del spin
    _check_c128(s_plus, 2, "s_plus")
    _check_c128(s_minus, 2, "s_minus")
    _check_f64(theta, 1, "theta")
    _check_f64(weight, 1, "weight")
    torch._check(s_plus.shape == s_minus.shape, lambda: "spin mode shapes must match")
    torch._check(
        s_plus.shape == (mmax + 1, theta.shape[0]), lambda: "mode shape mismatch"
    )
    torch._check(weight.shape == theta.shape, lambda: "weight shape mismatch")
    return torch.empty((2, _nalm(lmax, mmax)), dtype=s_plus.dtype, device=s_plus.device)


def _fake_spin_interpolate_recurrence(
    alms: Tensor, theta: Tensor, lmax: int, mmax: int, spin: int
) -> Tensor:
    del spin
    _check_c128(alms, 2, "alms")
    _check_f64(theta, 1, "theta")
    torch._check(alms.shape == (2, _nalm(lmax, mmax)), lambda: "alm shape mismatch")
    return torch.empty(
        (2, mmax + 1, theta.shape[0]), dtype=alms.dtype, device=alms.device
    )


def _fake_spin_map2alm_recurrence(
    maps: Tensor,
    starts: Tensor,
    lengths: Tensor,
    theta: Tensor,
    phase0_neg: Tensor,
    lmax: int,
    mmax: int,
    spin: int,
    pixel_weight: float,
) -> Tensor:
    del spin, pixel_weight
    _check_f64(maps, 2, "maps")
    _check_layout(starts, lengths)
    _check_f64(theta, 1, "theta")
    _check_c128(phase0_neg, 2, "phase0_neg")
    torch._check(maps.shape[0] == 2, lambda: "maps must have shape [2, npix]")
    torch._check(theta.shape == lengths.shape, lambda: "theta ring count mismatch")
    torch._check(
        phase0_neg.shape == (mmax + 1, lengths.shape[0]), lambda: "phase shape mismatch"
    )
    return torch.empty(
        (2, _nalm(lmax, mmax)), dtype=torch.complex128, device=maps.device
    )


def _fake_spin_alm2map_recurrence(
    alm_plus: Tensor,
    starts: Tensor,
    lengths: Tensor,
    theta: Tensor,
    phase0_pos: Tensor,
    lmax: int,
    mmax: int,
    spin: int,
    alm_minus: Tensor,
    npix: int,
) -> Tensor:
    del spin
    _check_c128(alm_plus, 1, "alm_plus")
    _check_c128(alm_minus, 1, "alm_minus")
    _check_layout(starts, lengths)
    _check_f64(theta, 1, "theta")
    _check_c128(phase0_pos, 2, "phase0_pos")
    nalm = _nalm(lmax, mmax)
    torch._check(alm_plus.shape == (nalm,), lambda: "alm length mismatch")
    torch._check(alm_minus.shape == alm_plus.shape, lambda: "alm shapes must match")
    torch._check(theta.shape == lengths.shape, lambda: "theta ring count mismatch")
    torch._check(
        phase0_pos.shape == (mmax + 1, lengths.shape[0]), lambda: "phase shape mismatch"
    )
    torch._check(npix >= 0, lambda: "npix must be non-negative")
    return torch.empty((2, npix), dtype=torch.float64, device=alm_plus.device)


def _fake_scalar_integrate_recurrence(
    modes: Tensor, theta: Tensor, weight: Tensor, lmax: int, mmax: int
) -> Tensor:
    _check_c128(modes, 2, "modes")
    _check_f64(theta, 1, "theta")
    _check_f64(weight, 1, "weight")
    torch._check(
        modes.shape == (mmax + 1, theta.shape[0]), lambda: "mode shape mismatch"
    )
    torch._check(weight.shape == theta.shape, lambda: "weight shape mismatch")
    return torch.empty((_nalm(lmax, mmax),), dtype=modes.dtype, device=modes.device)


def _fake_scalar_interpolate_recurrence(
    alm: Tensor, theta: Tensor, lmax: int, mmax: int
) -> Tensor:
    _check_c128(alm, 1, "alm")
    _check_f64(theta, 1, "theta")
    torch._check(alm.shape == (_nalm(lmax, mmax),), lambda: "alm length mismatch")
    return torch.empty((mmax + 1, theta.shape[0]), dtype=alm.dtype, device=alm.device)


def _fake_spin_map2alm_from_basis(
    q: Tensor,
    u: Tensor,
    ycp_t: Tensor,
    ycm_t: Tensor,
    pixel_weight: float,
) -> tuple[Tensor, Tensor]:
    del pixel_weight
    _check_f64(q, 1, "q")
    _check_f64(u, 1, "u")
    _check_c128(ycp_t, 2, "ycp_t")
    _check_c128(ycm_t, 2, "ycm_t")
    torch._check(q.shape == u.shape, lambda: "q/u shapes must match")
    torch._check(ycp_t.shape == ycm_t.shape, lambda: "basis shapes must match")
    torch._check(
        ycp_t.shape[0] == q.shape[0],
        lambda: "basis first dimension must match q/u length",
    )
    output = torch.empty((ycp_t.shape[1],), dtype=ycp_t.dtype, device=q.device)
    return output, torch.empty_like(output)


def _fake_spin_alm2map_from_basis(
    coeff_plus: Tensor,
    coeff_minus: Tensor,
    y_plus: Tensor,
    y_minus: Tensor,
    y_plus_m0: Tensor,
    y_minus_m0: Tensor,
) -> Tensor:
    _check_c128(coeff_plus, 1, "coeff_plus")
    _check_c128(coeff_minus, 1, "coeff_minus")
    for tensor, name in (
        (y_plus, "y_plus"),
        (y_minus, "y_minus"),
        (y_plus_m0, "y_plus_m0"),
        (y_minus_m0, "y_minus_m0"),
    ):
        _check_c128(tensor, 2, name)
    torch._check(
        coeff_plus.shape == coeff_minus.shape,
        lambda: "coefficient shapes must match",
    )
    torch._check(y_plus.shape == y_minus.shape, lambda: "basis shapes must match")
    torch._check(
        y_plus_m0.shape == y_minus_m0.shape,
        lambda: "m0 basis shapes must match",
    )
    torch._check(
        y_plus.shape[0] == coeff_plus.shape[0],
        lambda: "basis row count must match coefficient length",
    )
    torch._check(
        y_plus_m0.shape[0] <= coeff_plus.shape[0],
        lambda: "m0 basis row count cannot exceed coefficient length",
    )
    torch._check(
        y_plus_m0.shape[1] == y_plus.shape[1],
        lambda: "basis tensors must have the same pixel dimension",
    )
    return torch.empty(
        (2, y_plus.shape[1]), dtype=torch.float64, device=coeff_plus.device
    )


def _register_fake(name: str, implementation: Callable[..., Any]) -> None:
    torch.library.register_fake(f"torchsky::{name}")(implementation)


def _setup_fourier_modes(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
    del output
    _, starts, lengths, _ = inputs
    ctx.save_for_backward(starts, lengths)
    ctx.npix = inputs[0].shape[1]


def _backward_fourier_modes(ctx: Any, grad: Tensor) -> tuple[Tensor, None, None, None]:
    starts, lengths = ctx.saved_tensors
    return (
        torch.ops.torchsky.spectral_ring_fourier_synthesis(
            grad, starts, lengths, ctx.npix
        ),
        None,
        None,
        None,
    )


def _setup_fourier_synthesis(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
    del output
    _, starts, lengths, _ = inputs
    ctx.save_for_backward(starts, lengths)
    ctx.mmax = inputs[0].shape[1] - 1


def _backward_fourier_synthesis(
    ctx: Any, grad: Tensor
) -> tuple[Tensor, None, None, None]:
    starts, lengths = ctx.saved_tensors
    return (
        torch.ops.torchsky.spectral_ring_fourier_modes(grad, starts, lengths, ctx.mmax),
        None,
        None,
        None,
    )


def _setup_scalar_integrate(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
    del output
    _, theta, weight, lmax, mmax = inputs
    ctx.save_for_backward(theta, weight)
    ctx.lmax = lmax
    ctx.mmax = mmax


def _backward_scalar_integrate(
    ctx: Any, grad: Tensor
) -> tuple[Tensor, None, None, None, None]:
    theta, weight = ctx.saved_tensors
    modes = torch.ops.torchsky.spectral_scalar_interpolate_recurrence(
        grad, theta, ctx.lmax, ctx.mmax
    )
    return modes * weight.unsqueeze(0), None, None, None, None


def _setup_scalar_interpolate(
    ctx: Any, inputs: tuple[Any, ...], output: Tensor
) -> None:
    del output
    _, theta, lmax, mmax = inputs
    ctx.save_for_backward(theta)
    ctx.lmax = lmax
    ctx.mmax = mmax


def _backward_scalar_interpolate(
    ctx: Any, grad: Tensor
) -> tuple[Tensor, None, None, None]:
    (theta,) = ctx.saved_tensors
    weight = torch.ones_like(theta)
    return (
        torch.ops.torchsky.spectral_scalar_integrate_recurrence(
            grad, theta, weight, ctx.lmax, ctx.mmax
        ),
        None,
        None,
        None,
    )


def _setup_spin_integrate(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
    del output
    _, _, theta, weight, lmax, mmax, spin = inputs
    ctx.save_for_backward(theta, weight)
    ctx.lmax = lmax
    ctx.mmax = mmax
    ctx.spin = spin


def _backward_spin_integrate(
    ctx: Any, grad: Tensor
) -> tuple[Tensor, Tensor, None, None, None, None, None]:
    theta, weight = ctx.saved_tensors
    modes = torch.ops.torchsky.spectral_spin_interpolate_recurrence(
        grad, theta, ctx.lmax, ctx.mmax, ctx.spin
    )
    scaled = modes * weight.reshape(1, 1, -1)
    return scaled[0], scaled[1], None, None, None, None, None


def _setup_spin_interpolate(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
    del output
    _, theta, lmax, mmax, spin = inputs
    ctx.save_for_backward(theta)
    ctx.lmax = lmax
    ctx.mmax = mmax
    ctx.spin = spin


def _backward_spin_interpolate(
    ctx: Any, grad: Tensor
) -> tuple[Tensor, None, None, None, None]:
    (theta,) = ctx.saved_tensors
    weight = torch.ones_like(theta)
    return (
        torch.ops.torchsky.spectral_spin_integrate_recurrence(
            grad[0], grad[1], theta, weight, ctx.lmax, ctx.mmax, ctx.spin
        ),
        None,
        None,
        None,
        None,
    )


def _setup_concat_interpolate(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    del output
    _, _, y_plus, y_minus, lmax, mmax = inputs
    ctx.save_for_backward(y_plus, y_minus)
    ctx.lmax = lmax
    ctx.mmax = mmax


def _backward_concat_interpolate(
    ctx: Any, grad_plus: Tensor, grad_minus: Tensor
) -> tuple[Tensor, Tensor, None, None, None, None]:
    y_plus, y_minus = ctx.saved_tensors
    coeff_plus, coeff_minus = torch.ops.torchsky.spectral_spin_integrate_concat(
        grad_plus,
        grad_minus,
        y_plus.conj(),
        y_minus.conj(),
        ctx.lmax,
        ctx.mmax,
        1.0,
    )
    return coeff_plus, coeff_minus, None, None, None, None


def _setup_concat_integrate(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    del output
    _, _, y_plus, y_minus, lmax, mmax, pixel_weight = inputs
    ctx.save_for_backward(y_plus, y_minus)
    ctx.lmax = lmax
    ctx.mmax = mmax
    ctx.pixel_weight = pixel_weight


def _backward_concat_integrate(
    ctx: Any, grad_plus: Tensor, grad_minus: Tensor
) -> tuple[Tensor, Tensor, None, None, None, None, None]:
    y_plus, y_minus = ctx.saved_tensors
    modes_plus, modes_minus = torch.ops.torchsky.spectral_spin_interpolate_concat(
        grad_plus,
        grad_minus,
        y_plus.conj(),
        y_minus.conj(),
        ctx.lmax,
        ctx.mmax,
    )
    return (
        modes_plus * ctx.pixel_weight,
        modes_minus * ctx.pixel_weight,
        None,
        None,
        None,
        None,
        None,
    )


def _setup_basis_map2alm(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    del output
    _, _, ycp_t, ycm_t, pixel_weight = inputs
    ctx.save_for_backward(ycp_t, ycm_t)
    ctx.pixel_weight = pixel_weight


def _backward_basis_map2alm(
    ctx: Any, grad_plus: Tensor, grad_minus: Tensor
) -> tuple[Tensor, Tensor, None, None, None]:
    ycp_t, ycm_t = ctx.saved_tensors
    plus = ycp_t.conj() @ grad_plus
    minus = ycm_t.conj() @ grad_minus
    scale = ctx.pixel_weight
    return scale * (plus + minus).real, scale * (plus - minus).imag, None, None, None


def _setup_basis_alm2map(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    del output
    _, _, y_plus, y_minus, y_plus_m0, y_minus_m0 = inputs
    ctx.save_for_backward(y_plus, y_minus, y_plus_m0, y_minus_m0)


def _backward_basis_alm2map(
    ctx: Any, grad: Tensor
) -> tuple[Tensor, Tensor, None, None, None, None]:
    y_plus, y_minus, y_plus_m0, y_minus_m0 = ctx.saved_tensors
    grad_plus_map = -0.5 * torch.complex(grad[0], grad[1])
    grad_minus_map = -0.5 * torch.complex(grad[0], -grad[1])
    grad_plus = grad_plus_map @ y_plus.conj().T
    grad_minus = grad_minus_map @ y_minus.conj().T
    l0 = y_plus_m0.shape[0]
    if l0 != y_plus.shape[0]:
        grad_plus = grad_plus + grad_minus_map.conj() @ y_plus.conj().T
        grad_minus = grad_minus + grad_plus_map.conj() @ y_minus.conj().T
        plus_correction = grad_minus_map.conj() @ y_plus_m0.conj().T
        minus_correction = grad_plus_map.conj() @ y_minus_m0.conj().T
        grad_plus = grad_plus - torch.cat(
            (plus_correction, torch.zeros_like(grad_plus[l0:]))
        )
        grad_minus = grad_minus - torch.cat(
            (minus_correction, torch.zeros_like(grad_minus[l0:]))
        )
    return grad_plus, grad_minus, None, None, None, None


def _register() -> None:
    fake_implementations: dict[str, Callable[..., Any]] = {
        "spectral_scalar_alm2map_direct": _fake_scalar_direct,
        "spectral_ring_fourier_modes": _fake_fourier_modes,
        "spectral_ring_fourier_modes_spin_conj": _fake_fourier_modes_spin_conj,
        "spectral_ring_fourier_synthesis": _fake_fourier_synthesis,
        "spectral_spin_ring_finalize": _fake_spin_finalize,
        "spectral_spin_map2alm_ring_concat": _fake_spin_map2alm_concat,
        "spectral_spin_interpolate_concat": _fake_spin_interpolate_concat,
        "spectral_spin_integrate_concat": _fake_spin_integrate_concat,
        "spectral_spin_integrate_recurrence": _fake_spin_integrate_recurrence,
        "spectral_spin_interpolate_recurrence": _fake_spin_interpolate_recurrence,
        "spectral_spin_map2alm_ring_recurrence": _fake_spin_map2alm_recurrence,
        "spectral_spin_alm2map_ring_recurrence": _fake_spin_alm2map_recurrence,
        "spectral_scalar_integrate_recurrence": _fake_scalar_integrate_recurrence,
        "spectral_scalar_interpolate_recurrence": _fake_scalar_interpolate_recurrence,
        "_spectral_spin_map2alm_from_basis": _fake_spin_map2alm_from_basis,
        "_spectral_spin_alm2map_from_basis": _fake_spin_alm2map_from_basis,
    }
    for name, implementation in fake_implementations.items():
        _register_fake(name, implementation)

    torch.library.register_autograd(
        "torchsky::spectral_ring_fourier_modes",
        _backward_fourier_modes,
        setup_context=_setup_fourier_modes,
    )
    torch.library.register_autograd(
        "torchsky::spectral_ring_fourier_synthesis",
        _backward_fourier_synthesis,
        setup_context=_setup_fourier_synthesis,
    )
    torch.library.register_autograd(
        "torchsky::spectral_scalar_integrate_recurrence",
        _backward_scalar_integrate,
        setup_context=_setup_scalar_integrate,
    )
    torch.library.register_autograd(
        "torchsky::spectral_scalar_interpolate_recurrence",
        _backward_scalar_interpolate,
        setup_context=_setup_scalar_interpolate,
    )
    torch.library.register_autograd(
        "torchsky::spectral_spin_integrate_recurrence",
        _backward_spin_integrate,
        setup_context=_setup_spin_integrate,
    )
    torch.library.register_autograd(
        "torchsky::spectral_spin_interpolate_recurrence",
        _backward_spin_interpolate,
        setup_context=_setup_spin_interpolate,
    )
    torch.library.register_autograd(
        "torchsky::spectral_spin_interpolate_concat",
        _backward_concat_interpolate,
        setup_context=_setup_concat_interpolate,
    )
    torch.library.register_autograd(
        "torchsky::spectral_spin_integrate_concat",
        _backward_concat_integrate,
        setup_context=_setup_concat_integrate,
    )
    torch.library.register_autograd(
        "torchsky::_spectral_spin_map2alm_from_basis",
        _backward_basis_map2alm,
        setup_context=_setup_basis_map2alm,
    )
    torch.library.register_autograd(
        "torchsky::_spectral_spin_alm2map_from_basis",
        _backward_basis_alm2map,
        setup_context=_setup_basis_alm2map,
    )


_register()
