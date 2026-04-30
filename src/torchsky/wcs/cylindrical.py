import math

import torch
from torch import Tensor
from typing import Tuple, Optional

D2R = 0.017453292519943295
R2D = 57.29577951308232


def _wrap_lon180_checked(angle: Tensor) -> Tensor:
    return torch.remainder(angle + 180.0, 360.0) - 180.0


def _pv_param(
    params: Tensor | None,
    idx: int,
    default: float,
    *,
    like: Tensor,
) -> Tensor:
    if params is None or idx >= int(params.numel()):
        return torch.as_tensor(default, dtype=like.dtype, device=like.device)
    value = params[idx].to(dtype=like.dtype, device=like.device)
    return torch.where(
        torch.isclose(value, torch.zeros_like(value)),
        torch.as_tensor(default, dtype=like.dtype, device=like.device),
        value,
    )


def project_cylindrical(
    xi: Tensor,
    eta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Project intermediate (xi, eta) to spherical (phi, theta) for Cylindrical projections.
    """
    pv2_1 = _pv_param(pv2, 1, 1.0, like=xi)
    pv2_2 = _pv_param(pv2, 2, 1.0, like=xi)

    if projection_code == "CEA":
        phi = xi
        val = torch.clamp(eta * (pv2_1 * D2R), -1.0, 1.0)
        theta = torch.asin(val) * R2D

    elif projection_code == "CAR":
        phi = xi
        theta = eta

    elif projection_code == "MER":
        phi = xi
        arg = eta * D2R
        arg = torch.clamp(arg, -20.0, 20.0)
        exp_val = torch.exp(arg)
        theta = 2.0 * (torch.atan(exp_val) * R2D - 45.0)

    elif projection_code == "CYP":
        mu = pv2_1
        lam = pv2_2
        if bool(torch.any(torch.isclose(lam, torch.zeros_like(lam)))):
            raise ValueError("CYP PV2_2 (lambda) must be non-zero")
        phi = xi / lam
        eta_rad = eta * D2R
        theta_rad = torch.atan2(eta_rad, torch.ones_like(eta_rad))
        amplitude = mu + lam
        for _ in range(8):
            sin_t = torch.sin(theta_rad)
            cos_t = torch.cos(theta_rad)
            denom = torch.clamp(mu + cos_t, min=1e-12)
            f = amplitude * sin_t / denom - eta_rad
            df = amplitude * (mu * cos_t + 1.0) / torch.square(denom)
            step = f / torch.clamp(df, min=1e-12)
            theta_rad = torch.clamp(
                theta_rad - step,
                min=-0.5 * math.pi + 1e-12,
                max=0.5 * math.pi - 1e-12,
            )
        theta = theta_rad * R2D

    else:
        raise NotImplementedError(
            f"Cylindrical code {projection_code} not implemented."
        )

    return phi, theta


def deproject_cylindrical(
    phi: Tensor,
    theta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Forward cylindrical projection: native spherical (phi, theta) -> intermediate (xi, eta).

    phi, theta: Degrees. For cylindrical projections, these are typically native coords
                where phi is in [0, 360) and needs to be wrapped to [-180, 180] for xi.
    Returns: xi, eta in degrees.
    """
    pv2_1 = _pv_param(pv2, 1, 1.0, like=phi)
    pv2_2 = _pv_param(pv2, 2, 1.0, like=phi)

    # Wrap phi to [-180, 180] range for cylindrical projections
    phi_wrapped = _wrap_lon180_checked(phi)

    if projection_code == "CEA":
        xi = phi_wrapped
        eta = (R2D / pv2_1) * torch.sin(theta * D2R)
        return xi, eta

    elif projection_code == "CAR":
        xi = phi_wrapped
        eta = theta
        return xi, eta

    elif projection_code == "MER":
        xi = phi_wrapped
        half_theta = theta / 2.0
        tan_arg = torch.tan((45.0 + half_theta) * D2R)
        tan_arg = torch.clamp(tan_arg, min=1e-12)
        eta = R2D * torch.log(tan_arg)
        return xi, eta

    elif projection_code == "CYP":
        mu = pv2_1
        lam = pv2_2
        if bool(torch.any(torch.isclose(lam, torch.zeros_like(lam)))):
            raise ValueError("CYP PV2_2 (lambda) must be non-zero")
        theta_rad = theta * D2R
        xi = lam * phi_wrapped
        denom = torch.clamp(mu + torch.cos(theta_rad), min=1e-12)
        eta = R2D * (mu + lam) * torch.sin(theta_rad) / denom
        return xi, eta

    else:
        raise NotImplementedError(
            f"Cylindrical forward projection {projection_code} not implemented."
        )
