import math

import torch
from torch import Tensor
from typing import Tuple

from ._math import PVParams, _pv_value

D2R = 0.017453292519943295
R2D = 57.29577951308232


def _wrap_lon180_checked(angle: Tensor) -> Tensor:
    return torch.remainder(angle + 180.0, 360.0) - 180.0


def _cylindrical_parameters(projection_code: str, pv2: PVParams) -> tuple[float, float]:
    """``(PV2_1, PV2_2)`` with defaults 1.0, validated for CEA and CYP."""
    pv2_1 = _pv_value(pv2, 1, 1.0)
    pv2_2 = _pv_value(pv2, 2, 1.0)
    if projection_code == "CEA" and (not math.isfinite(pv2_1) or pv2_1 == 0.0):
        raise ValueError("CEA PV2_1 (lambda) must be finite and non-zero")
    if projection_code == "CYP" and math.isclose(pv2_2, 0.0, abs_tol=1e-8):
        raise ValueError("CYP PV2_2 (lambda) must be non-zero")
    return pv2_1, pv2_2


def project_cylindrical(
    xi: Tensor,
    eta: Tensor,
    projection_code: str,
    pv1: PVParams = None,
    pv2: PVParams = None,
) -> Tuple[Tensor, Tensor]:
    """
    Project intermediate (xi, eta) to spherical (phi, theta) for Cylindrical projections.
    """
    pv2_1, pv2_2 = _cylindrical_parameters(projection_code, pv2)

    if projection_code == "CEA":
        phi = xi
        val = eta * (pv2_1 * D2R)
        valid = torch.isfinite(val) & (val.abs() <= 1.0 + 1e-12)
        theta = torch.asin(torch.clamp(val, -1.0, 1.0)) * R2D
        nan = torch.full_like(theta, float("nan"))
        phi = torch.where(valid, phi, nan)
        theta = torch.where(valid, theta, nan)

    elif projection_code == "CAR":
        phi = xi
        theta = eta

    elif projection_code == "MER":
        phi = xi
        valid = torch.isfinite(xi) & torch.isfinite(eta)
        arg = eta * D2R
        arg = torch.clamp(arg, -20.0, 20.0)
        exp_val = torch.exp(arg)
        theta = 2.0 * (torch.atan(exp_val) * R2D - 45.0)
        nan = torch.full_like(theta, float("nan"))
        phi = torch.where(valid, phi, nan)
        theta = torch.where(valid, theta, nan)

    elif projection_code == "CYP":
        mu = pv2_1
        lam = pv2_2
        phi = xi / lam
        # Closed form of WCSLIB cypx2s: with q = y / (r0 (mu + lambda)),
        # theta = atan(q) + asin(q mu / sqrt(q^2 + 1)). An asin argument
        # beyond 1 (outside the domain) is NaN, as in WCSLIB.
        q = eta * (D2R / (mu + lam))
        arg = q * mu / torch.sqrt(q * q + 1.0)
        snapped = torch.where(arg.abs() <= 1.0 + 1e-10, arg.clamp(-1.0, 1.0), arg)
        theta = (torch.atan(q) + torch.asin(snapped)) * R2D

    else:
        raise NotImplementedError(
            f"Cylindrical code {projection_code} not implemented."
        )

    return phi, theta


def deproject_cylindrical(
    phi: Tensor,
    theta: Tensor,
    projection_code: str,
    pv1: PVParams = None,
    pv2: PVParams = None,
) -> Tuple[Tensor, Tensor]:
    """
    Forward cylindrical projection: native spherical (phi, theta) -> intermediate (xi, eta).

    phi, theta: Degrees. For cylindrical projections, these are typically native coords
                where phi is in [0, 360) and needs to be wrapped to [-180, 180] for xi.
    Returns: xi, eta in degrees.
    """
    pv2_1, pv2_2 = _cylindrical_parameters(projection_code, pv2)

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
        valid = torch.isfinite(phi) & torch.isfinite(theta) & (theta.abs() < 90.0)
        half_theta = theta / 2.0
        tan_arg = torch.tan((45.0 + half_theta) * D2R)
        eta = R2D * torch.log(tan_arg)
        nan = torch.full_like(eta, float("nan"))
        xi = torch.where(valid, phi_wrapped, nan)
        eta = torch.where(valid, eta, nan)
        return xi, eta

    elif projection_code == "CYP":
        mu = pv2_1
        lam = pv2_2
        theta_rad = theta * D2R
        xi = lam * phi_wrapped
        denom = torch.clamp(mu + torch.cos(theta_rad), min=1e-12)
        eta = R2D * (mu + lam) * torch.sin(theta_rad) / denom
        return xi, eta

    else:
        raise NotImplementedError(
            f"Cylindrical forward projection {projection_code} not implemented."
        )
