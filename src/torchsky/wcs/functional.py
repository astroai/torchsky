from __future__ import annotations

import torch
from torch import Tensor

from torchsky import _ops


_D2R = torch.pi / 180.0
_R2D = 180.0 / torch.pi


def _wrap180(value: Tensor) -> Tensor:
    return torch.remainder(value + 180.0, 360.0) - 180.0


def _wrap360(value: Tensor) -> Tensor:
    return torch.remainder(value, 360.0)


def _tpv_eval(
    u: Tensor, v: Tensor, idx: Tensor, coeff: Tensor, *, swap: bool = False
) -> tuple[Tensor, Tensor, Tensor]:
    if swap:
        result, derivative_v, derivative_u = _tpv_eval(v, u, idx, coeff)
        return result, derivative_u, derivative_v
    if coeff.numel() == 0:
        zero = torch.zeros_like(u)
        return u, torch.ones_like(u), zero
    radius = torch.hypot(u, v)
    px = idx[:, 0]
    py = idx[:, 1]
    pr = idx[:, 2]
    values = (
        u.unsqueeze(-1).pow(px) * v.unsqueeze(-1).pow(py) * radius.unsqueeze(-1).pow(pr)
    )
    result = torch.sum(values * coeff, dim=-1)
    safe_radius = torch.where(radius == 0.0, torch.ones_like(radius), radius)
    du = torch.where(
        px > 0,
        px
        * u.unsqueeze(-1).pow(torch.clamp(px - 1, min=0))
        * v.unsqueeze(-1).pow(py)
        * radius.unsqueeze(-1).pow(pr),
        torch.zeros_like(values),
    )
    dv = torch.where(
        py > 0,
        py
        * u.unsqueeze(-1).pow(px)
        * v.unsqueeze(-1).pow(torch.clamp(py - 1, min=0))
        * radius.unsqueeze(-1).pow(pr),
        torch.zeros_like(values),
    )
    radial = torch.where(
        pr > 0,
        pr
        * u.unsqueeze(-1).pow(px)
        * v.unsqueeze(-1).pow(py)
        * radius.unsqueeze(-1).pow(torch.clamp(pr - 1, min=0)),
        torch.zeros_like(values),
    )
    du = du + radial * (u / safe_radius).unsqueeze(-1)
    dv = dv + radial * (v / safe_radius).unsqueeze(-1)
    return result, torch.sum(du * coeff, dim=-1), torch.sum(dv * coeff, dim=-1)


def _tpv_distort(
    u: Tensor,
    v: Tensor,
    idx1: Tensor,
    c1: Tensor,
    idx2: Tensor,
    c2: Tensor,
) -> tuple[Tensor, Tensor]:
    if c1.numel() == 0 and c2.numel() == 0:
        return u, v
    first = _tpv_eval(u, v, idx1, c1)[0] if c1.numel() else u
    second = _tpv_eval(u, v, idx2, c2, swap=True)[0] if c2.numel() else v
    return first, second


def _tpv_invert(
    xi: Tensor,
    eta: Tensor,
    idx1: Tensor,
    c1: Tensor,
    idx2: Tensor,
    c2: Tensor,
) -> tuple[Tensor, Tensor]:
    if c1.numel() == 0 and c2.numel() == 0:
        return xi, eta
    u, v = xi, eta
    for _ in range(20):
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
            torch.maximum(j00.abs(), j01.abs()),
            torch.maximum(j10.abs(), j11.abs()),
        )
        conditioned = (
            torch.isfinite(scale)
            & torch.isfinite(determinant)
            & (scale > 0.0)
            & (determinant.abs() > 64.0 * torch.finfo(xi.dtype).eps * scale * scale)
        )
        safe = torch.where(
            conditioned,
            determinant,
            torch.ones_like(determinant),
        )
        r1, r2 = f1 - xi, f2 - eta
        du = torch.clamp((j11 * r1 - j01 * r2) / safe, -1.0, 1.0)
        dv = torch.clamp((-j10 * r1 + j00 * r2) / safe, -1.0, 1.0)
        u = u - du
        v = v - dv
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
        torch.maximum(j00.abs(), j01.abs()),
        torch.maximum(j10.abs(), j11.abs()),
    )
    valid = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & (torch.hypot(f1 - xi, f2 - eta) <= 1e-11)
        & torch.isfinite(scale)
        & torch.isfinite(determinant)
        & (scale > 0.0)
        & (determinant.abs() > 64.0 * torch.finfo(xi.dtype).eps * scale * scale)
    )
    nan = torch.full_like(u, torch.nan)
    return torch.where(valid, u, nan), torch.where(valid, v, nan)


def _plane_to_native(
    xi: Tensor,
    eta: Tensor,
    proj_code: str,
    lambda_param: float,
    hpx_h: float,
    hpx_k: float,
) -> tuple[Tensor, Tensor]:
    from .allsky import project_allsky
    from .cylindrical import project_cylindrical
    from .quadcube import project_quadcube
    from .zenithal import project_zenithal

    if proj_code in {"TAN", "SIN", "ARC"}:
        return project_zenithal(xi, eta, proj_code)
    if proj_code in {"CEA", "CAR", "SFL"}:
        pv2 = xi.new_tensor([0.0, lambda_param])
        return project_cylindrical(xi, eta, proj_code, None, pv2)
    if proj_code in {"AIT", "MOL", "PCO", "XPH", "HPX"}:
        pv1 = xi.new_tensor([0.0, hpx_h])
        pv2 = xi.new_tensor([0.0, hpx_k])
        return project_allsky(xi, eta, proj_code, pv1, pv2)
    if proj_code in {"TSC", "CSC", "QSC"}:
        return project_quadcube(xi, eta, proj_code)
    raise RuntimeError(f"unsupported fused WCS projection: {proj_code}")


def _native_to_plane(
    phi: Tensor,
    theta: Tensor,
    proj_code: str,
    lambda_param: float,
    hpx_h: float,
    hpx_k: float,
) -> tuple[Tensor, Tensor]:
    from .allsky import deproject_allsky
    from .cylindrical import deproject_cylindrical
    from .quadcube import deproject_quadcube
    from .zenithal import deproject_zenithal

    if proj_code in {"TAN", "SIN", "ARC"}:
        return deproject_zenithal(phi, theta, proj_code)
    if proj_code in {"CEA", "CAR", "SFL"}:
        pv2 = phi.new_tensor([0.0, lambda_param])
        return deproject_cylindrical(phi, theta, proj_code, None, pv2)
    if proj_code in {"AIT", "MOL", "PCO", "XPH", "HPX"}:
        pv1 = phi.new_tensor([0.0, hpx_h])
        pv2 = phi.new_tensor([0.0, hpx_k])
        return deproject_allsky(phi, theta, proj_code, pv1, pv2)
    if proj_code in {"TSC", "CSC", "QSC"}:
        return deproject_quadcube(phi, theta, proj_code)
    raise RuntimeError(f"unsupported fused WCS projection: {proj_code}")


def _native_to_world(
    phi: Tensor,
    theta: Tensor,
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
) -> tuple[Tensor, Tensor]:
    if (
        native_type == "center"
        and abs(delta0) < 1e-12
        and abs(phi_p_rad % (2.0 * torch.pi)) < 1e-12
    ):
        return _wrap360(phi + alpha0), theta
    phi_rad = phi * _D2R
    theta_rad = theta * _D2R
    cos_theta = torch.cos(theta_rad)
    sin_theta = torch.sin(theta_rad)
    if native_type == "pole":
        cos_delta_p = torch.cos(phi.new_tensor(dec_p_rad))
        sin_delta_p = torch.sin(phi.new_tensor(dec_p_rad))
        difference = phi_rad - phi_p_rad
        common = cos_theta * torch.cos(difference)
        dec = (
            torch.asin(
                torch.clamp(sin_theta * sin_delta_p + common * cos_delta_p, -1.0, 1.0)
            )
            * _R2D
        )
        ra = (
            torch.atan2(
                cos_theta * torch.sin(difference),
                sin_theta * cos_delta_p - common * sin_delta_p,
            )
            * _R2D
            + ra_p_rad * _R2D
        )
        return _wrap360(ra), dec
    u = cos_theta * torch.sin(phi_rad)
    v = -cos_theta * torch.cos(phi_rad)
    w = sin_theta
    world_x = u * east_x - v * radial_x + w * north_x
    world_y = u * east_y - v * radial_y + w * north_y
    world_z = -v * radial_z + w * north_z
    return _wrap360(torch.atan2(world_y, world_x) * _R2D), torch.asin(
        torch.clamp(world_z, -1.0, 1.0)
    ) * _R2D


def _world_to_native(
    ra: Tensor,
    dec: Tensor,
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
) -> tuple[Tensor, Tensor]:
    if (
        native_type == "center"
        and abs(delta0) < 1e-12
        and abs(phi_p_rad % (2.0 * torch.pi)) < 1e-12
    ):
        return _wrap180(ra - alpha0), dec
    ra_rad = ra * _D2R
    dec_rad = dec * _D2R
    cos_dec = torch.cos(dec_rad)
    x = cos_dec * torch.cos(ra_rad)
    y = cos_dec * torch.sin(ra_rad)
    z = torch.sin(dec_rad)
    if native_type == "pole":
        u = x * east_x + y * east_y
        n = x * north_x + y * north_y + z * north_z
        w = x * radial_x + y * radial_y + z * radial_z
        return _wrap360(torch.atan2(u, n) * _R2D + phi_p_rad * _R2D), torch.asin(
            torch.clamp(w, -1.0, 1.0)
        ) * _R2D
    u = x * east_x + y * east_y
    v = -(x * radial_x + y * radial_y + z * radial_z)
    w = x * north_x + y * north_y + z * north_z
    return _wrap180(torch.atan2(u, -v) * _R2D), torch.asin(
        torch.clamp(w, -1.0, 1.0)
    ) * _R2D


def _wcs_reference(direction: str, *args: object) -> tuple[Tensor, Tensor]:
    first, second = args[:2]
    assert isinstance(first, Tensor) and isinstance(second, Tensor)
    (
        crpix0,
        crpix1,
        m00,
        m01,
        m10,
        m11,
        proj_code,
        lambda_param,
        hpx_h,
        hpx_k,
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
        tpv_idx1,
        tpv_c1,
        tpv_idx2,
        tpv_c2,
    ) = args[2:]
    assert isinstance(proj_code, str) and isinstance(native_type, str)
    assert isinstance(tpv_idx1, Tensor) and isinstance(tpv_c1, Tensor)
    assert isinstance(tpv_idx2, Tensor) and isinstance(tpv_c2, Tensor)
    rotation = (
        float(east_x),
        float(east_y),
        float(north_x),
        float(north_y),
        float(north_z),
        float(radial_x),
        float(radial_y),
        float(radial_z),
        float(ra_p_rad),
        float(dec_p_rad),
        float(phi_p_rad),
        native_type,
        float(alpha0),
        float(delta0),
    )
    if direction == "pixel_to_world":
        dx, dy = first - float(crpix0), second - float(crpix1)
        xi = float(m00) * dx + float(m01) * dy
        eta = float(m10) * dx + float(m11) * dy
        if bool(is_tpv):
            xi, eta = _tpv_distort(xi, eta, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2)
        phi, theta = _plane_to_native(
            xi, eta, proj_code, float(lambda_param), float(hpx_h), float(hpx_k)
        )
        return _native_to_world(phi, theta, *rotation)
    phi, theta = _world_to_native(first, second, *rotation)
    xi, eta = _native_to_plane(
        phi, theta, proj_code, float(lambda_param), float(hpx_h), float(hpx_k)
    )
    if bool(is_tpv):
        xi, eta = _tpv_invert(xi, eta, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2)
    return (
        float(m00) * xi + float(m01) * eta + float(crpix0),
        float(m10) * xi + float(m11) * eta + float(crpix1),
    )


def _tpv_tensors(
    reference: Tensor,
    tpv_idx1: Tensor | None,
    tpv_c1: Tensor | None,
    tpv_idx2: Tensor | None,
    tpv_c2: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if all(value is None for value in (tpv_idx1, tpv_c1, tpv_idx2, tpv_c2)):
        empty_idx = torch.empty((0, 3), dtype=torch.int64, device=reference.device)
        empty_coeff = torch.empty((0,), dtype=torch.float64, device=reference.device)
        return empty_idx, empty_coeff, empty_idx.clone(), empty_coeff.clone()
    if any(value is None for value in (tpv_idx1, tpv_c1, tpv_idx2, tpv_c2)):
        raise ValueError(
            "all TPV index and coefficient tensors must be provided together"
        )
    assert tpv_idx1 is not None
    assert tpv_c1 is not None
    assert tpv_idx2 is not None
    assert tpv_c2 is not None
    return tpv_idx1, tpv_c1, tpv_idx2, tpv_c2


def pixel_to_world_fused(
    x: Tensor,
    y: Tensor,
    *,
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
    tpv_idx1: Tensor | None = None,
    tpv_c1: Tensor | None = None,
    tpv_idx2: Tensor | None = None,
    tpv_c2: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the fused dispatcher path; invalid projection domains produce NaNs."""
    idx1, c1, idx2, c2 = _tpv_tensors(x, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2)
    return _ops.wcs_pixel_to_world(
        x,
        y,
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
        c1,
        idx2,
        c2,
    )


def world_to_pixel_fused(
    ra: Tensor,
    dec: Tensor,
    *,
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
    tpv_idx1: Tensor | None = None,
    tpv_c1: Tensor | None = None,
    tpv_idx2: Tensor | None = None,
    tpv_c2: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run the inverse fused path; singular or invalid coordinates produce NaNs."""
    idx1, c1, idx2, c2 = _tpv_tensors(ra, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2)
    return _ops.wcs_world_to_pixel(
        ra,
        dec,
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
        c1,
        idx2,
        c2,
    )
