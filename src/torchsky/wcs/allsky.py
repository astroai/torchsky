import torch
from torch import Tensor
from typing import Tuple, Optional
import math

from ._math import _sincos, _wrap_lon180

D2R = 0.017453292519943295
R2D = 57.29577951308232
SQRT2 = 1.4142135623730951


def _hpx_parameters(
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[float, float, float, float, float, float, float]:
    h = 4.0
    k = 3.0
    if pv1 is not None:
        if float(pv1[1]) != 0.0:
            h = float(pv1[1])
        if float(pv1[2]) != 0.0:
            k = float(pv1[2])
    if pv2 is not None:
        if float(pv2[1]) != 0.0:
            h = float(pv2[1])
        if float(pv2[2]) != 0.0:
            k = float(pv2[2])
    if not math.isfinite(h) or h <= 0.0 or not h.is_integer():
        raise ValueError("HPX H must be a finite positive integer")
    if not math.isfinite(k) or k <= 0.0 or not k.is_integer():
        raise ValueError("HPX K must be a finite positive integer")
    facet_spacing = 360.0 / h
    facet_half_width = 0.5 * facet_spacing
    return (
        h,
        k,
        (k - 1.0) / k,
        90.0 * k / h,
        90.0 * (k - 1.0) / h,
        90.0 * (k + 1.0) / h,
        facet_half_width,
    )


def _hpx_facet_center(
    coordinate: Tensor, south: Tensor, k: float, facet_half_width: float
) -> Tensor:
    facet_spacing = 2.0 * facet_half_width
    offset = torch.full_like(coordinate, facet_half_width)
    if k % 2.0 == 0.0:
        offset = torch.where(south, torch.zeros_like(coordinate), offset)
    return (
        torch.floor(
            (coordinate + 180.0 - offset + facet_half_width) / facet_spacing
        )
        * facet_spacing
        - 180.0
        + offset
    )


def project_allsky(
    xi: Tensor,
    eta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Project intermediate world coordinates (xi, eta) to native spherical coordinates (phi, theta).
    NOTE: This is the INVERSE projection (Plane -> Sphere) relative to standard FITS definition.

    xi, eta: Degrees (Standard FITS Intermediate).
    Returns: phi, theta (Degrees)
    """
    if projection_code == "AIT":
        x = xi * D2R
        y = eta * D2R
        X = x
        Y = y
        # Bolt performance optimization: Explicit multiplication (x*x) is faster than
        # exponentiation (x**2) on PyTorch tensors due to avoiding pow overhead.
        xq = X * 0.25
        yh = Y * 0.5
        r2 = xq * xq + yh * yh
        mask = r2 <= 1.0
        z = torch.sqrt(torch.clamp(1.0 - r2, min=0.0))
        phi_rad = 2.0 * torch.atan2(0.5 * z * X, 2.0 * z * z - 1.0)
        sin_theta = torch.clamp(z * Y, -1.0, 1.0)
        theta_rad = torch.asin(sin_theta)
        phi = phi_rad * R2D
        theta = theta_rad * R2D
        nan_phi = torch.full_like(phi, float("nan"))
        nan_theta = torch.full_like(theta, float("nan"))
        phi = torch.where(mask, phi, nan_phi)
        theta = torch.where(mask, theta, nan_theta)

    elif projection_code == "MOL":
        x = xi * D2R
        y = eta * D2R
        X = x
        Y = y
        tol = max(1e-13, 4.0 * torch.finfo(X.dtype).eps)
        ellipse = X * X / 8.0 + Y * Y / 2.0
        valid = torch.isfinite(ellipse) & (ellipse <= 1.0 + tol)
        sin_gamma = torch.clamp(Y / SQRT2, -1.0, 1.0)
        gamma = torch.asin(sin_gamma)
        cos_gamma = torch.sqrt(torch.clamp(1.0 - sin_gamma * sin_gamma, min=0.0))
        t_val = (2.0 * gamma + 2.0 * sin_gamma * cos_gamma) / math.pi
        valid = valid & torch.isfinite(t_val) & (t_val.abs() <= 1.0 + tol)
        theta_rad = torch.asin(torch.clamp(t_val, -1.0, 1.0))
        theta = theta_rad * R2D
        denom = 2.0 * SQRT2 * cos_gamma
        denom_nonzero = denom.abs() > 0.0
        valid = valid & (denom_nonzero | (X == 0.0))
        phi_rad = math.pi * X / torch.where(
            denom_nonzero, denom, torch.ones_like(denom)
        )
        phi_rad = torch.where(denom_nonzero, phi_rad, torch.zeros_like(phi_rad))
        valid = valid & torch.isfinite(phi_rad) & (phi_rad.abs() <= math.pi + tol)
        phi = torch.clamp(phi_rad, -math.pi, math.pi) * R2D
        nan = torch.full_like(phi, float("nan"))
        phi = torch.where(valid, phi, nan)
        theta = torch.where(valid, theta, nan)

    elif projection_code == "HPX":
        _, K, _, eta_scale, eta_boundary, eta_pole, facet_half_width = (
            _hpx_parameters(pv1, pv2)
        )
        inv_eta_scale = 1.0 / eta_scale
        inv_polar_denom = 1.0 / facet_half_width
        abs_eta = torch.abs(eta)
        mask_eq = abs_eta <= eta_boundary
        sigma = (eta_pole - abs_eta) * inv_polar_denom
        s_theta = torch.where(
            mask_eq,
            eta * inv_eta_scale,
            torch.sign(eta) * (1.0 - (sigma * sigma) / K),
        )
        theta = torch.asin(torch.clamp(s_theta, -1.0, 1.0)) * R2D
        xc = _hpx_facet_center(xi, eta < 0.0, K, facet_half_width)
        phi = torch.where(mask_eq, xi, xc + (xi - xc) / torch.clamp(sigma, min=1e-9))
        invalid = (~torch.isfinite(xi)) | (~torch.isfinite(eta)) | (abs_eta > eta_pole) | (
            (~mask_eq) & (torch.abs(xi - xc) > facet_half_width * sigma + 1e-8)
        )
        if bool(invalid.any()):
            nan_phi = torch.full_like(phi, float("nan"))
            nan_theta = torch.full_like(theta, float("nan"))
            phi = torch.where(invalid, nan_phi, phi)
            theta = torch.where(invalid, nan_theta, theta)

    elif projection_code == "SFL":
        theta = eta
        # SFL: phi = xi / cos(theta)
        phi = xi / torch.clamp(torch.cos(theta * D2R), min=1e-12)

    elif projection_code == "PAR":
        # Parabolic pseudocylindrical (WCSLIB parx2s).
        # x = w1 * xi, y = w3 * eta  (w1 = 1/r0_rad, w3 = 1/(pi*r0_deg))
        r0 = R2D
        w1 = 1.0 / (r0 * D2R)  # = 1.0 when r0 = R2D
        w3 = 1.0 / (math.pi * r0)
        s = w1 * xi
        r = w3 * eta
        valid = (r <= 1.0) & (r >= -1.0)
        denom = 1.0 - 4.0 * r * r
        # At |r|=0.5, denom=0; WCSLIB uses a deferred test on theta sign.
        # phi = s / (1 - 4r^2), theta = 3 * asin(r)
        safe_denom = torch.where(denom == 0.0, torch.ones_like(denom), denom)
        phi = s / safe_denom
        theta = 3.0 * torch.asin(torch.clamp(r, -1.0, 1.0)) * R2D
        # Handle the denom==0 edge: theta must be < 0 for validity
        edge_valid = theta < 0.0
        valid = torch.where(denom == 0.0, edge_valid, valid)
        phi = torch.where(valid, phi, torch.full_like(phi, float("nan")))
        theta = torch.where(valid, theta, torch.full_like(theta, float("nan")))

    elif projection_code == "XPH":
        phi, theta = _xph_x2s(xi, eta)

    else:
        raise NotImplementedError(
            f"All-sky projection code {projection_code!r} is not implemented."
        )

    return phi, theta


def deproject_allsky(
    phi: Tensor,
    theta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward all-sky projection."""
    if projection_code == "AIT":
        phi_wrapped = _wrap_lon180(phi)
        phi_rad = phi_wrapped * D2R
        theta_rad = theta * D2R
        sin_theta, cos_theta = _sincos(theta_rad)
        sin_half_phi, cos_half_phi = _sincos(phi_rad * 0.5)
        denom = torch.sqrt(0.5 * (1.0 + cos_theta * cos_half_phi))
        xi = 2.0 * cos_theta * sin_half_phi / denom * R2D
        eta = sin_theta / denom * R2D
        return xi, eta

    elif projection_code == "MOL":
        phi_wrapped = _wrap_lon180(phi)
        phi_rad = phi_wrapped * D2R
        domain_tol = max(1e-12, 4.0 * torch.finfo(theta.dtype).eps)
        valid = (
            torch.isfinite(phi)
            & torch.isfinite(theta)
            & (theta.abs() <= 90.0 + domain_tol)
        )
        theta_clamped = torch.clamp(theta, -90.0, 90.0)
        theta_rad = theta_clamped * D2R
        sin_theta = torch.sin(theta_rad)
        target = math.pi * sin_theta
        pole = theta_clamped.abs() == 90.0
        lo = torch.full_like(theta_rad, -0.5 * math.pi)
        hi = torch.full_like(theta_rad, 0.5 * math.pi)
        gamma = theta_rad.clone()
        for _ in range(64):
            res = 2.0 * gamma + torch.sin(2.0 * gamma) - target
            lo = torch.where(res < 0.0, gamma, lo)
            hi = torch.where(res >= 0.0, gamma, hi)
            derivative = 2.0 + 2.0 * torch.cos(2.0 * gamma)
            derivative_safe = torch.where(
                derivative > 0.0, derivative, torch.ones_like(derivative)
            )
            newton = gamma - res / derivative_safe
            use_newton = torch.isfinite(newton) & (newton > lo) & (newton < hi)
            gamma = torch.where(use_newton, newton, 0.5 * (lo + hi))
        gamma = torch.where(
            pole,
            torch.copysign(torch.full_like(gamma, 0.5 * math.pi), theta_clamped),
            gamma,
        )
        residual = 2.0 * gamma + torch.sin(2.0 * gamma) - target
        residual_tol = max(1e-12, 64.0 * torch.finfo(theta.dtype).eps)
        valid = valid & (pole | (residual.abs() <= residual_tol))
        xi = 2.0 * SQRT2 * phi_rad * torch.cos(gamma) / math.pi * R2D
        eta = SQRT2 * torch.sin(gamma) * R2D
        xi = torch.where(pole, torch.zeros_like(xi), xi)
        nan = torch.full_like(xi, float("nan"))
        xi = torch.where(valid, xi, nan)
        eta = torch.where(valid, eta, nan)
        return xi, eta

    elif projection_code == "HPX":
        _, K, transition_sin, eta_scale, _, eta_pole, facet_half_width = (
            _hpx_parameters(pv1, pv2)
        )
        phi_w = _wrap_lon180(phi)
        s_theta = torch.sin(theta * D2R)
        abs_s = torch.abs(s_theta)
        mask_eq = abs_s <= transition_sin
        sigma = torch.sqrt(torch.clamp(K * (1.0 - abs_s), min=0.0))
        eta_pol = torch.sign(s_theta) * (eta_pole - facet_half_width * sigma)
        xc = _hpx_facet_center(phi_w, s_theta < 0.0, K, facet_half_width)
        xi = torch.where(mask_eq, phi_w, xc + sigma * (phi_w - xc))
        eta = torch.where(mask_eq, eta_scale * s_theta, eta_pol)
        valid = torch.isfinite(phi) & torch.isfinite(theta) & (theta.abs() <= 90.0 + 1e-12)
        nan = torch.full_like(xi, float("nan"))
        xi = torch.where(valid, xi, nan)
        eta = torch.where(valid, eta, nan)
        return xi, eta

    elif projection_code == "SFL":
        phi_wrapped = _wrap_lon180(phi)
        xi = phi_wrapped * torch.cos(theta * D2R)
        eta = theta
        return xi, eta

    elif projection_code == "PAR":
        # Forward PAR (WCSLIB pars2x).
        r0 = R2D
        w0 = r0 * D2R  # = 1.0
        w2 = math.pi * r0
        phi_wrapped = _wrap_lon180(phi)
        xi_raw = w0 * phi_wrapped
        s = torch.sin(theta * D2R / 3.0)
        xi = (1.0 - 4.0 * s * s) * xi_raw
        eta = w2 * s
        return xi, eta

    elif projection_code == "XPH":
        return _xph_s2x(phi, theta)

    return phi, theta


# ---------------------------------------------------------------------------
# XPH — HEALPix polar / butterfly (Calabretta & Lowe 2013; WCSLIB xph*)
# ---------------------------------------------------------------------------

_XPH_SINTHE_X = 2.0 / 3.0
_XPH_POLE_TOL = 1.0e-4
_XPH_INV_SQRT2 = 0.7071067811865476


def _xph_s2x(phi: Tensor, theta: Tensor) -> Tuple[Tensor, Tensor]:
    """Forward XPH: native (phi, theta) -> plane (xi, eta)."""
    # Torchsky pole-rotation phi has opposite sign to WCSLIB; undo before XPH.
    phi = -phi
    # Normalize phi to [-180, 180), then chi in [0, 360), psi = chi mod 90.
    chi = _wrap_lon180(phi) + 180.0
    psi = torch.remainder(chi, 90.0)
    phi_n = chi - 180.0

    sinthe = torch.sin(theta * D2R)
    abssin = sinthe.abs()
    mask_eq = abssin <= _XPH_SINTHE_X

    # Equatorial regime
    xi_eq = psi
    eta_eq = 67.5 * sinthe

    # Polar regime: sigma = sqrt(3*(1-|sin theta|)), linearized near poles
    pole_lim = 90.0 - _XPH_POLE_TOL * (math.sqrt(_XPH_SINTHE_X) * R2D)
    sigma_std = torch.sqrt(torch.clamp(3.0 * (1.0 - abssin), min=0.0))
    sigma_lin = (90.0 - theta.abs()) * math.sqrt(1.5) * D2R
    sigma = torch.where(theta.abs() < pole_lim, sigma_std, sigma_lin)
    xi_pol = 45.0 + (psi - 45.0) * sigma
    eta_pol = 45.0 * (2.0 - sigma)
    eta_pol = torch.where(theta < 0.0, -eta_pol, eta_pol)

    xi = torch.where(mask_eq, xi_eq, xi_pol) - 45.0
    eta = torch.where(mask_eq, eta_eq, eta_pol) - 90.0

    s = _XPH_INV_SQRT2
    # Quadrant rotation from rounded phi (WCSLIB xphs2x)
    q1 = phi_n < -90.0
    q2 = (~q1) & (phi_n < 0.0)
    q3 = (~q1) & (~q2) & (phi_n < 90.0)
    # q4: else
    x = torch.where(
        q1,
        s * (-xi + eta),
        torch.where(
            q2,
            s * (xi + eta),
            torch.where(q3, s * (xi - eta), s * (-xi - eta)),
        ),
    )
    y = torch.where(
        q1,
        s * (-xi - eta),
        torch.where(
            q2,
            s * (-xi + eta),
            torch.where(q3, s * (xi + eta), s * (xi - eta)),
        ),
    )
    return x, y


def _xph_x2s(xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
    """Inverse XPH: plane (xi, eta) -> native (phi, theta)."""
    s = _XPH_INV_SQRT2
    xr = xi / s
    yr = eta / s

    # Quadrant -> (xi1, eta1, base phi)
    c1 = (xr <= 0.0) & (yr > 0.0)
    c2 = (xr < 0.0) & (yr <= 0.0)
    c3 = (xr >= 0.0) & (yr < 0.0)
    # c4: else
    xi1 = torch.where(
        c1,
        (-xr - yr) * 0.5,
        torch.where(
            c2,
            (xr - yr) * 0.5,
            torch.where(c3, (xr + yr) * 0.5, (-xr + yr) * 0.5),
        ),
    )
    eta1 = torch.where(
        c1,
        (xr - yr) * 0.5,
        torch.where(
            c2,
            (xr + yr) * 0.5,
            torch.where(c3, (-xr + yr) * 0.5, (-xr - yr) * 0.5),
        ),
    )
    phi = torch.where(
        c1,
        torch.full_like(xr, -180.0),
        torch.where(
            c2,
            torch.full_like(xr, -90.0),
            torch.where(c3, torch.zeros_like(xr), torch.full_like(xr, 90.0)),
        ),
    )

    xi_f = xi1 + 45.0
    eta_f = eta1 + 90.0
    abseta = eta_f.abs()
    invalid = abseta > 90.0

    mask_eq = abseta <= 45.0
    # Equatorial
    phi_eq = phi + xi_f
    theta_eq = torch.asin(torch.clamp(eta_f / 67.5, -1.0, 1.0)) * R2D

    # Polar
    sigma = (90.0 - abseta) / 45.0
    safe_sigma = torch.clamp(sigma, min=1e-12)
    phi_pol = phi + 45.0 + xi1 / safe_sigma
    # Snap on axes
    phi_pol = torch.where(
        xr == 0.0,
        torch.where(yr <= 0.0, torch.zeros_like(phi_pol), torch.full_like(phi_pol, 180.0)),
        torch.where(
            yr == 0.0,
            torch.where(xr < 0.0, torch.full_like(phi_pol, -90.0), torch.full_like(phi_pol, 90.0)),
            phi_pol,
        ),
    )
    th_std = torch.asin(torch.clamp(1.0 - sigma * sigma / 3.0, -1.0, 1.0)) * R2D
    th_lin = 90.0 - sigma * (math.sqrt(_XPH_SINTHE_X) * R2D)
    th = torch.where(sigma < _XPH_POLE_TOL, th_lin, th_std)
    th = torch.where(eta_f < 0.0, -th, th)

    phi_out = torch.where(mask_eq, phi_eq, phi_pol)
    theta_out = torch.where(mask_eq, theta_eq, th)
    # Torchsky zenithal/pole rotation uses the opposite phi sign from WCSLIB
    # (cf. zenithal atan2(-xi,-eta) vs prjx2s). Flip so celestial matches.
    phi_out = _wrap_lon180(-phi_out)
    nan = torch.full_like(phi_out, float("nan"))
    phi_out = torch.where(invalid, nan, phi_out)
    theta_out = torch.where(invalid, nan, theta_out)
    return phi_out, theta_out
