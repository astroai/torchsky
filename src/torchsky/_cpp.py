"""Compatibility surface for historical native-extension symbols."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import warnings

import torch

from . import _cpp_ext as _ext

MIGRATED_COMPAT_SYMBOLS = (
    "wcs_pixel_to_world_fused_cpu",
    "wcs_world_to_pixel_fused_cpu",
    "healpix_ring2nest_torch_cpu",
    "healpix_nest2ring_torch_cpu",
    "healpix_ang2pix_ring_torch_cpu",
    "healpix_ang2pix_nested_torch_cpu",
    "healpix_pix2ang_ring_torch_cpu",
    "healpix_pix2ang_nested_torch_cpu",
    "healpix_neighbors_ring_torch_cpu",
    "healpix_neighbors_nested_torch_cpu",
    "healpix_get_interp_weights_ring_torch_cpu",
    "healpix_get_interp_weights_nested_torch_cpu",
    "healpix_get_interp_val_ring_torch_cpu",
    "healpix_get_interp_val_nested_torch_cpu",
    "_sphere_planar_point_in_polygon_cpu",
    "_sphere_arc_intersections_matrix_cpu",
    "_sparse_get_values_blocks_cpu",
    "_healpix_scalar_alm2map_direct_cpu",
    "_healpix_ring_fourier_modes_cpu",
    "_healpix_ring_fourier_modes_spin_conj_cpu",
    "_healpix_ring_fourier_synthesis_cpu",
    "_healpix_spin_ring_finalize_cpu",
    "_healpix_spin_map2alm_ring_concat_cpu",
    "_healpix_spin_interpolate_concat_cpu",
    "_healpix_spin_integrate_concat_cpu",
    "_healpix_spin_integrate_recurrence_cpu",
    "_healpix_spin_interpolate_recurrence_cpu",
    "_healpix_spin_map2alm_ring_recurrence_cpu",
    "_healpix_spin_alm2map_ring_recurrence_cpu",
    "_healpix_scalar_integrate_recurrence_cpu",
    "_healpix_scalar_interpolate_recurrence_cpu",
)

DEFERRED_NATIVE_SYMBOLS = (
    "healpix_query_disc_torch_cpu",
    "_healpix_spin_map2alm_from_basis_cpu",
    "_healpix_spin_alm2map_from_basis_cpu",
)

PYTHON_COMPAT_SYMBOLS = (
    "wcs_zenithal_project",
    "wcs_zenithal_deproject",
    "wcs_cylindrical_project",
    "wcs_cylindrical_deproject",
    "wcs_hpx_project",
    "wcs_hpx_deproject",
    "wcs_tpv_distort",
    "wcs_tpv_invert",
)


def _warn(name: str) -> None:
    warnings.warn(
        f"torchsky._cpp.{name} is deprecated; use torchsky._ops or torch.ops.torchsky",
        DeprecationWarning,
        stacklevel=2,
    )


def _operator_adapter(old_name: str, operator_name: str) -> Callable[..., Any]:
    def adapter(*args: Any, **kwargs: Any) -> Any:
        _warn(old_name)
        return getattr(torch.ops.torchsky, operator_name)(*args, **kwargs)

    adapter.__name__ = old_name
    adapter.__qualname__ = old_name
    return adapter


def _ring_npix(lengths: torch.Tensor) -> int:
    return int(lengths.sum().item())


def wcs_zenithal_project(
    xi: torch.Tensor, eta: torch.Tensor, proj_code: str
) -> tuple[torch.Tensor, torch.Tensor]:
    _warn("wcs_zenithal_project")
    from .wcs.zenithal import project_zenithal

    return project_zenithal(xi, eta, proj_code)


def wcs_zenithal_deproject(
    phi: torch.Tensor, theta: torch.Tensor, proj_code: str
) -> tuple[torch.Tensor, torch.Tensor]:
    _warn("wcs_zenithal_deproject")
    from .wcs.zenithal import deproject_zenithal

    return deproject_zenithal(phi, theta, proj_code)


def wcs_cylindrical_project(
    xi: torch.Tensor, eta: torch.Tensor, proj_code: str, lambda_param: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    _warn("wcs_cylindrical_project")
    from .wcs.cylindrical import project_cylindrical

    pv2 = xi.new_tensor([0.0, lambda_param])
    try:
        return project_cylindrical(xi, eta, proj_code, None, pv2)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def wcs_cylindrical_deproject(
    phi: torch.Tensor,
    theta: torch.Tensor,
    proj_code: str,
    lambda_param: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    _warn("wcs_cylindrical_deproject")
    from .wcs.cylindrical import deproject_cylindrical

    pv2 = phi.new_tensor([0.0, lambda_param])
    try:
        return deproject_cylindrical(phi, theta, proj_code, None, pv2)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def _check_hpx_parameters(h: float, k: float) -> None:
    if not (
        isinstance(h, (int, float))
        and isinstance(k, (int, float))
        and torch.isfinite(torch.tensor(h))
        and torch.isfinite(torch.tensor(k))
        and h > 0
        and k > 0
        and float(h).is_integer()
        and float(k).is_integer()
    ):
        raise RuntimeError("HPX H must be a finite positive integer")


def wcs_hpx_project(
    xi: torch.Tensor, eta: torch.Tensor, H: float = 4.0, K: float = 3.0
) -> tuple[torch.Tensor, torch.Tensor]:
    _warn("wcs_hpx_project")
    _check_hpx_parameters(H, K)
    from .wcs.allsky import project_allsky

    pv = xi.new_tensor([0.0, H, K])
    return project_allsky(xi, eta, "HPX", pv, None)


def wcs_hpx_deproject(
    phi: torch.Tensor, theta: torch.Tensor, H: float = 4.0, K: float = 3.0
) -> tuple[torch.Tensor, torch.Tensor]:
    _warn("wcs_hpx_deproject")
    _check_hpx_parameters(H, K)
    from .wcs.allsky import deproject_allsky

    pv = phi.new_tensor([0.0, H, K])
    return deproject_allsky(phi, theta, "HPX", pv, None)


def _validate_tpv_metadata(idx: torch.Tensor, coeff: torch.Tensor) -> None:
    if idx.dtype != torch.int64 or idx.ndim != 2 or idx.shape[1] != 3:
        raise RuntimeError("TPV exponents must be int64 with shape [N, 3]")
    if coeff.dtype != torch.float64 or coeff.ndim != 1:
        raise RuntimeError("TPV coefficients must be float64 rank 1")
    if idx.shape[0] != coeff.shape[0]:
        raise RuntimeError("TPV metadata lengths must match")
    if bool((idx < 0).any()) or bool((idx > 7).any()):
        raise RuntimeError("TPV exponents must be between zero and seven")
    if not bool(torch.isfinite(coeff).all()):
        raise RuntimeError("TPV coefficients must be finite")


def wcs_tpv_distort(
    u: torch.Tensor,
    v: torch.Tensor,
    idx1: torch.Tensor,
    c1: torch.Tensor,
    idx2: torch.Tensor,
    c2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _warn("wcs_tpv_distort")
    _validate_tpv_metadata(idx1, c1)
    _validate_tpv_metadata(idx2, c2)
    from .wcs.functional import _tpv_distort

    first, second = _tpv_distort(u, v, idx1, c1, idx2, c2)
    valid = torch.isfinite(u) & torch.isfinite(v)
    nan = torch.full_like(first, torch.nan)
    return torch.where(valid, first, nan), torch.where(valid, second, nan)


def wcs_tpv_invert(
    xi: torch.Tensor,
    eta: torch.Tensor,
    idx1: torch.Tensor,
    c1: torch.Tensor,
    idx2: torch.Tensor,
    c2: torch.Tensor,
    max_iter: int = 10,
    tol: float = 1e-11,
) -> tuple[torch.Tensor, torch.Tensor]:
    _warn("wcs_tpv_invert")
    _validate_tpv_metadata(idx1, c1)
    _validate_tpv_metadata(idx2, c2)
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise RuntimeError("max_iter must be positive")
    if not torch.isfinite(torch.tensor(tol)) or tol <= 0.0:
        raise RuntimeError("tol must be finite and positive")
    from .wcs.functional import _tpv_eval

    u, v = xi.clone(), eta.clone()
    for _ in range(max_iter):
        if c1.numel():
            f1, j00, j01 = _tpv_eval(u, v, idx1, c1)
        else:
            f1, j00, j01 = u, torch.ones_like(u), torch.zeros_like(u)
        if c2.numel():
            f2, j10, j11 = _tpv_eval(u, v, idx2, c2, swap=True)
        else:
            f2, j10, j11 = v, torch.zeros_like(v), torch.ones_like(v)
        determinant = j00 * j11 - j01 * j10
        scale = torch.maximum(
            torch.maximum(j00.abs(), j01.abs()), torch.maximum(j10.abs(), j11.abs())
        )
        conditioned = (
            torch.isfinite(scale)
            & torch.isfinite(determinant)
            & (scale > 0.0)
            & (determinant.abs() > 64.0 * torch.finfo(xi.dtype).eps * scale * scale)
        )
        safe = torch.where(conditioned, determinant, torch.ones_like(determinant))
        r1, r2 = f1 - xi, f2 - eta
        u = u - torch.clamp((j11 * r1 - j01 * r2) / safe, -1.0, 1.0)
        v = v - torch.clamp((-j10 * r1 + j00 * r2) / safe, -1.0, 1.0)
    first, j00, j01 = (
        _tpv_eval(u, v, idx1, c1)
        if c1.numel()
        else (u, torch.ones_like(u), torch.zeros_like(u))
    )
    second, j10, j11 = (
        _tpv_eval(u, v, idx2, c2, swap=True)
        if c2.numel()
        else (v, torch.zeros_like(v), torch.ones_like(v))
    )
    determinant = j00 * j11 - j01 * j10
    scale = torch.maximum(
        torch.maximum(j00.abs(), j01.abs()), torch.maximum(j10.abs(), j11.abs())
    )
    valid = (
        torch.isfinite(xi)
        & torch.isfinite(eta)
        & torch.isfinite(u)
        & torch.isfinite(v)
        & (torch.hypot(first - xi, second - eta) <= tol)
        & torch.isfinite(scale)
        & torch.isfinite(determinant)
        & (scale > 0.0)
        & (determinant.abs() > 64.0 * torch.finfo(xi.dtype).eps * scale * scale)
    )
    nan = torch.full_like(u, torch.nan)
    return torch.where(valid, u, nan), torch.where(valid, v, nan)


def _healpix_ring_fourier_modes_spin_conj_cpu(
    row: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    mmax: int,
) -> torch.Tensor:
    _warn("_healpix_ring_fourier_modes_spin_conj_cpu")
    return torch.stack(
        torch.ops.torchsky.spectral_ring_fourier_modes_spin_conj(
            row, starts, lengths, mmax
        )
    )


def _healpix_ring_fourier_synthesis_cpu(
    modes: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    _warn("_healpix_ring_fourier_synthesis_cpu")
    return torch.ops.torchsky.spectral_ring_fourier_synthesis(
        modes, starts, lengths, _ring_npix(lengths)
    )


def _healpix_spin_ring_finalize_cpu(
    s_plus: torch.Tensor,
    s_minus: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    _warn("_healpix_spin_ring_finalize_cpu")
    return torch.ops.torchsky.spectral_spin_ring_finalize(
        s_plus, s_minus, starts, lengths, _ring_npix(lengths)
    )


def _healpix_spin_alm2map_ring_recurrence_cpu(
    alm_plus: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    theta: torch.Tensor,
    phase0_pos: torch.Tensor,
    lmax: int,
    mmax: int,
    spin: int,
    alm_minus: torch.Tensor,
) -> torch.Tensor:
    _warn("_healpix_spin_alm2map_ring_recurrence_cpu")
    return torch.ops.torchsky.spectral_spin_alm2map_ring_recurrence(
        alm_plus,
        starts,
        lengths,
        theta,
        phase0_pos,
        lmax,
        mmax,
        spin,
        alm_minus,
        _ring_npix(lengths),
    )


def _empty_tpv(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((0, 3), dtype=torch.int64, device=reference.device),
        torch.empty(0, dtype=torch.float64, device=reference.device),
    )


def _validate_legacy_wcs_buffers(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if tensor.dtype != torch.float64:
            raise RuntimeError("fused WCS tensors must be float64")
        if tensor.device.type != "cpu":
            raise RuntimeError("fused WCS tensors must be on CPU")
        if not tensor.is_contiguous():
            raise RuntimeError("fused WCS tensors must be contiguous")
    if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
        raise RuntimeError("fused WCS input and output tensors must have identical shapes")
    for index, first in enumerate(tensors):
        for second in tensors[index + 1 :]:
            if torch._C._overlaps(first, second):
                raise RuntimeError("fused WCS input and output tensors must not overlap")


def _legacy_tpv_arguments(
    reference: torch.Tensor,
    is_tpv: bool,
    tpv_idx1: torch.Tensor | None,
    tpv_c1: torch.Tensor | None,
    tpv_idx2: torch.Tensor | None,
    tpv_c2: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    supplied = (tpv_idx1, tpv_c1, tpv_idx2, tpv_c2)
    if is_tpv and any(value is None for value in supplied):
        raise RuntimeError(
            "TPV index and coefficient tensors are required when is_tpv is true"
        )
    empty_idx, empty_coeff = _empty_tpv(reference)
    return (
        empty_idx if tpv_idx1 is None else tpv_idx1,
        empty_coeff if tpv_c1 is None else tpv_c1,
        empty_idx if tpv_idx2 is None else tpv_idx2,
        empty_coeff if tpv_c2 is None else tpv_c2,
    )


def wcs_pixel_to_world_fused_cpu(
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    ra_tensor: torch.Tensor,
    dec_tensor: torch.Tensor,
    crpix0: float,
    crpix1: float,
    cd00: float,
    cd01: float,
    cd10: float,
    cd11: float,
    proj_code: str,
    lambda_param: float,
    hpx_H: float,
    hpx_K: float,
    east_x: float,
    east_y: float,
    north_x: float,
    north_y: float,
    north_z: float,
    radial_x: float,
    radial_y: float,
    radial_z: float,
    ra_p_rad: float,
    dec_p_rad: float,
    phi_p_rad: float,
    native_type: str,
    alpha0: float,
    delta0: float,
    is_tpv: bool = False,
    tpv_idx1: torch.Tensor | None = None,
    tpv_c1: torch.Tensor | None = None,
    tpv_idx2: torch.Tensor | None = None,
    tpv_c2: torch.Tensor | None = None,
) -> None:
    _warn("wcs_pixel_to_world_fused_cpu")
    _validate_legacy_wcs_buffers(x_tensor, y_tensor, ra_tensor, dec_tensor)
    idx1, coeff1, idx2, coeff2 = _legacy_tpv_arguments(
        x_tensor, is_tpv, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2
    )
    out = torch.ops.torchsky.wcs_pixel_to_world(
        x_tensor,
        y_tensor,
        crpix0,
        crpix1,
        cd00,
        cd01,
        cd10,
        cd11,
        proj_code,
        lambda_param,
        hpx_H,
        hpx_K,
        east_x,
        east_y,
        north_x,
        north_y,
        north_z,
        radial_x,
        radial_y,
        radial_z,
        ra_p_rad,
        dec_p_rad,
        phi_p_rad,
        native_type,
        alpha0,
        delta0,
        is_tpv,
        idx1,
        coeff1,
        idx2,
        coeff2,
    )
    ra_tensor.copy_(out[0])
    dec_tensor.copy_(out[1])


def wcs_world_to_pixel_fused_cpu(
    ra_tensor: torch.Tensor,
    dec_tensor: torch.Tensor,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    crpix0: float,
    crpix1: float,
    cdi00: float,
    cdi01: float,
    cdi10: float,
    cdi11: float,
    proj_code: str,
    lambda_param: float,
    hpx_H: float,
    hpx_K: float,
    east_x: float,
    east_y: float,
    north_x: float,
    north_y: float,
    north_z: float,
    radial_x: float,
    radial_y: float,
    radial_z: float,
    ra_p_rad: float,
    dec_p_rad: float,
    phi_p_rad: float,
    native_type: str,
    alpha0: float,
    delta0: float,
    is_tpv: bool = False,
    tpv_idx1: torch.Tensor | None = None,
    tpv_c1: torch.Tensor | None = None,
    tpv_idx2: torch.Tensor | None = None,
    tpv_c2: torch.Tensor | None = None,
) -> None:
    _warn("wcs_world_to_pixel_fused_cpu")
    _validate_legacy_wcs_buffers(ra_tensor, dec_tensor, x_tensor, y_tensor)
    idx1, coeff1, idx2, coeff2 = _legacy_tpv_arguments(
        ra_tensor, is_tpv, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2
    )
    out = torch.ops.torchsky.wcs_world_to_pixel(
        ra_tensor,
        dec_tensor,
        crpix0,
        crpix1,
        cdi00,
        cdi01,
        cdi10,
        cdi11,
        proj_code,
        lambda_param,
        hpx_H,
        hpx_K,
        east_x,
        east_y,
        north_x,
        north_y,
        north_z,
        radial_x,
        radial_y,
        radial_z,
        ra_p_rad,
        dec_p_rad,
        phi_p_rad,
        native_type,
        alpha0,
        delta0,
        is_tpv,
        idx1,
        coeff1,
        idx2,
        coeff2,
    )
    x_tensor.copy_(out[0])
    y_tensor.copy_(out[1])


_SIMPLE_ADAPTERS = {
    "healpix_ring2nest_torch_cpu": "healpix_ring2nest",
    "healpix_nest2ring_torch_cpu": "healpix_nest2ring",
    "healpix_ang2pix_ring_torch_cpu": "healpix_ang2pix_ring",
    "healpix_ang2pix_nested_torch_cpu": "healpix_ang2pix_nested",
    "healpix_pix2ang_ring_torch_cpu": "healpix_pix2ang_ring",
    "healpix_pix2ang_nested_torch_cpu": "healpix_pix2ang_nested",
    "healpix_neighbors_ring_torch_cpu": "healpix_neighbors_ring",
    "healpix_neighbors_nested_torch_cpu": "healpix_neighbors_nested",
    "healpix_get_interp_weights_ring_torch_cpu": "healpix_get_interp_weights_ring",
    "healpix_get_interp_weights_nested_torch_cpu": "healpix_get_interp_weights_nested",
    "healpix_get_interp_val_ring_torch_cpu": "healpix_get_interp_val_ring",
    "healpix_get_interp_val_nested_torch_cpu": "healpix_get_interp_val_nested",
    "_sphere_planar_point_in_polygon_cpu": "sphere_planar_point_in_polygon",
    "_sphere_arc_intersections_matrix_cpu": "sphere_arc_intersections_matrix",
    "_sparse_get_values_blocks_cpu": "sparse_get_values_blocks",
    "_healpix_scalar_alm2map_direct_cpu": "spectral_scalar_alm2map_direct",
    "_healpix_ring_fourier_modes_cpu": "spectral_ring_fourier_modes",
    "_healpix_spin_map2alm_ring_concat_cpu": "spectral_spin_map2alm_ring_concat",
    "_healpix_spin_interpolate_concat_cpu": "spectral_spin_interpolate_concat",
    "_healpix_spin_integrate_concat_cpu": "spectral_spin_integrate_concat",
    "_healpix_spin_integrate_recurrence_cpu": "spectral_spin_integrate_recurrence",
    "_healpix_spin_interpolate_recurrence_cpu": "spectral_spin_interpolate_recurrence",
    "_healpix_spin_map2alm_ring_recurrence_cpu": "spectral_spin_map2alm_ring_recurrence",
    "_healpix_scalar_integrate_recurrence_cpu": "spectral_scalar_integrate_recurrence",
    "_healpix_scalar_interpolate_recurrence_cpu": "spectral_scalar_interpolate_recurrence",
}

for _old_name, _operator_name in _SIMPLE_ADAPTERS.items():
    globals()[_old_name] = _operator_adapter(_old_name, _operator_name)

for _deferred_name in DEFERRED_NATIVE_SYMBOLS:
    globals()[_deferred_name] = getattr(_ext, _deferred_name)


def selected_cpu_isa() -> str:
    return str(_ext.selected_cpu_isa())


__all__ = [
    *MIGRATED_COMPAT_SYMBOLS,
    *PYTHON_COMPAT_SYMBOLS,
    *DEFERRED_NATIVE_SYMBOLS,
    "selected_cpu_isa",
]

del _deferred_name, _old_name, _operator_name
