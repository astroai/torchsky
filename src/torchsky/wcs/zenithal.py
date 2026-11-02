import math

import torch
from torch import Tensor
from typing import Tuple, Optional

from ._math import _sincos

D2R = 0.017453292519943295
R2D = 57.29577951308232


def _pv_param(
    params: Tensor | None,
    idx: int,
    default: float,
    *,
    like: Tensor,
) -> Tensor:
    """Return PV parameter *idx* with a default, matching cylindrical.py."""
    if params is None or idx >= int(params.numel()):
        return torch.as_tensor(default, dtype=like.dtype, device=like.device)
    return params[idx].to(dtype=like.dtype, device=like.device)


def project_zenithal(
    xi: Tensor,
    eta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Project intermediate world coordinates (xi, eta) to spherical coordinates (phi, theta)
    for Zenithal projections.
    """
    # 1. R and Phi
    r = torch.hypot(xi, eta)

    # Handle R=0 case to avoid division/singularities
    # If R=0, phi is undefined (usually 0), theta = 90.

    # Convert to degrees
    # WCS Paper II: phi increases in the direction of increasing RA (East).
    # For zenithal projections, the native longitude phi is measured from the
    # direction toward the celestial pole. With phi_p = 180°, the formula is:
    # phi = atan2(-xi, -eta)
    # 57.29577951308232 = 180.0 / math.pi
    phi = torch.atan2(-xi, -eta) * R2D

    theta = torch.zeros_like(r)

    # Zenithal Functions: R = f(theta) or theta = f_inv(R)
    # Here (xi, eta) -> (phi, theta), so we need theta = f_inv(R).
    # theta is latitude (90 at pole).
    # native coordinates: (phi, theta). Pole at (0, 90).

    # We compute (90 - theta) often called 'u' or colatitude?
    # No, usually we solve for theta directly.

    if projection_code == "TAN":
        r_rad = r * D2R
        theta = torch.atan2(torch.ones_like(r_rad), r_rad) * R2D

        # Wait, atan2(1, r_rad) is behavior of cot(theta) = r_rad?
        # If R_true = tan(90-theta)
        # then 90-theta = atan(R_true)
        # theta = 90 - atan(R_true)
        # atan2(1, R) is atan(1/R) = acot(R) = 90 - atan(R).
        # So yes, theta = rad2deg(atan2(1, r_rad)) is correct for TAN.

    elif projection_code == "SIN":
        x = xi * D2R
        y = eta * D2R
        a = pv2[1] if pv2 is not None and pv2.numel() > 1 else 0.0
        b = pv2[2] if pv2 is not None and pv2.numel() > 2 else 0.0
        if pv2 is not None and bool(
            (torch.as_tensor(a).abs() > 1e-15) | (torch.as_tensor(b).abs() > 1e-15)
        ):
            # Slant orthographic SIN (WCS Paper II): solve for z = sin(theta)
            # from the sphere constraint after subtracting the slant terms.
            x0 = x - a
            y0 = y - b
            aa = 1.0 + a * a + b * b
            bb = a * x0 + b * y0
            cc = x0 * x0 + y0 * y0 - 1.0
            disc = bb * bb - aa * cc
            valid = disc >= 0.0
            sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
            z = (-bb + sqrt_disc) / aa
            valid = valid & (z >= -1.0) & (z <= 1.0)
            x_comp = x0 + a * z
            y_comp = y0 + b * z
            phi = torch.atan2(-x_comp, -y_comp) * R2D
            theta = torch.asin(torch.clamp(z, -1.0, 1.0)) * R2D
            theta = theta.masked_fill(~valid, float("nan"))
            phi = phi.masked_fill(~valid, float("nan"))
        else:
            r_dim = r * D2R
            mask_out = r_dim > 1.0
            theta = torch.acos(torch.clamp(r_dim, -1.0, 1.0)) * R2D
            theta = theta.masked_fill(mask_out, float("nan"))

    elif projection_code == "ZEA":
        # Zenithal Equal Area
        # R = 2 * (180/pi) * sin(theta_co / 2)
        # 90 - theta = 2 * asin( R / (2 * 180/pi) )
        r_rad = r * D2R
        # ZEA is defined for r_rad <= 2.0 (theta_co <= 180 deg)
        mask_out = (r_rad / 2.0) > 1.0
        val = torch.clamp(r_rad / 2.0, -1.0, 1.0)
        theta = 90.0 - 2.0 * torch.asin(val) * R2D
        theta = theta.masked_fill(mask_out, float("nan"))

    elif projection_code == "STG":
        # Stereographic
        # R = 2 * (180/pi) * tan(theta_co / 2)
        # 90 - theta = 2 * atan( R / (2 * 180/pi) )
        r_rad = r * D2R
        theta = 90.0 - 2.0 * torch.atan(r_rad / 2.0) * R2D

    elif projection_code == "ARC":
        # Zenithal Equidistant
        # R = 90 - theta (linear distance)
        # theta = 90 - R
        valid = torch.isfinite(r) & (r <= math.nextafter(180.0, math.inf))
        theta = 90.0 - torch.clamp(r, max=180.0)
        nan = torch.full_like(theta, float("nan"))
        phi = torch.where(valid, phi, nan)
        theta = torch.where(valid, theta, nan)

    elif projection_code == "ZPN":
        if pv2 is None:
            w = r * D2R
        else:
            # ZPN stores the radial polynomial as PV2_m coefficients:
            # r(rad) = sum_m PV2_m * w(rad)**m, where w is colatitude.
            p_coeffs = pv2[:31]
            mask_valid = p_coeffs != 0.0
            if not mask_valid.any():
                w = r * D2R
            else:
                # Find last valid coefficient for order
                # Use a small epsilon to detect non-zero coefficients
                valid_indices = torch.nonzero(p_coeffs.abs() > 1e-15, as_tuple=True)[0]
                if valid_indices.numel() == 0:
                    w = r * D2R
                else:
                    max_m = valid_indices[-1].item()
                    coeffs = p_coeffs[: max_m + 1]

                    r_rad_target = r * D2R
                    if max_m == 1 and coeffs[1].abs() > 1e-15:
                        w = (r_rad_target - coeffs[0]) / coeffs[1]
                    else:
                        rev_c = torch.flip(coeffs, dims=(0,))

                        def _poly_and_slope(wv: Tensor) -> Tuple[Tensor, Tensor]:
                            val = torch.full_like(wv, rev_c[0].item())
                            der = torch.zeros_like(wv)
                            for c in rev_c[1:]:
                                der = der * wv + val
                                val = val * wv + c
                            return val, der

                        # Newton-Raphson on a detached copy (no in-place updates, so
                        # autograd never sees the iteration), then one differentiable
                        # out-of-place Newton step: dw/dr = 1/P'(w) at the root.
                        w0 = r_rad_target.detach().clone()
                        for _ in range(12):
                            val, der = _poly_and_slope(w0)
                            safe = der.abs() > 1e-15
                            step = torch.where(
                                safe,
                                (val - r_rad_target.detach())
                                / torch.where(safe, der, torch.ones_like(der)),
                                torch.zeros_like(der),
                            )
                            w0 = w0 - step
                        val, der = _poly_and_slope(w0)
                        safe = der.abs() > 1e-15
                        w = w0 - torch.where(
                            safe,
                            (val - r_rad_target)
                            / torch.where(safe, der, torch.ones_like(der)),
                            torch.zeros_like(der),
                        )

                    val = torch.zeros_like(w)
                    for c in torch.flip(coeffs, dims=(0,)):
                        val.mul_(w).add_(c)
                    valid = (
                        torch.isfinite(w)
                        & (w >= 0.0)
                        & (w <= torch.pi)
                        & ((val - r_rad_target).abs() <= 1e-10)
                    )
                    w = w.masked_fill(~valid, float("nan"))

        # 57.29577951308232 = 180.0 / math.pi
        theta = 90.0 - w * R2D

    elif projection_code == "AZP":
        # Zenithal perspective (WCS Paper II / WCSLIB azpx2s).
        # Parameters: mu = PV2_1 (distance), gamma = PV2_2 (tilt deg).
        mu = _pv_param(pv2, 1, 0.0, like=xi)
        gamma = _pv_param(pv2, 2, 0.0, like=eta)
        cos_g = torch.cos(gamma * D2R)
        sin_g = torch.sin(gamma * D2R)
        r0 = R2D
        w0 = r0 * (mu + 1.0)
        yc = eta * cos_g
        q = w0 + eta * sin_g
        r = torch.hypot(xi, yc)
        # Torchsky zenithal convention: phi = atan2(-xi, -yc) (not atan2(xi, -yc))
        phi = torch.where(r == 0.0, torch.zeros_like(r), torch.atan2(-xi, -yc) * R2D)
        s = r / q
        t = s * mu / torch.sqrt(s * s + 1.0)
        s_angle = torch.atan2(torch.ones_like(s), s) * R2D
        t_clamped = torch.clamp(t, -1.0 - 1e-13, 1.0 + 1e-13)
        t_angle = torch.asin(torch.clamp(t_clamped, -1.0, 1.0)) * R2D
        t_angle = torch.where(
            t_clamped.abs() > 1.0 + 1e-13,
            torch.sign(t_clamped) * 90.0,
            t_angle,
        )
        a = s_angle - t_angle
        b = s_angle + t_angle + 180.0
        a = torch.where(a > 90.0, a - 360.0, a)
        b = torch.where(b > 90.0, b - 360.0, b)
        theta = torch.where(a > b, a, b)
        valid = (r != 0.0) & torch.isfinite(theta) & (t_clamped.abs() <= 1.0 + 1e-13)
        theta = torch.where(r == 0.0, torch.full_like(r, 90.0), theta)
        theta = torch.where(
            valid | (r == 0.0), theta, torch.full_like(theta, float("nan"))
        )
        phi = torch.where(valid | (r == 0.0), phi, torch.full_like(phi, float("nan")))

    elif projection_code == "SZP":
        # Slant zenithal perspective (WCSLIB szpx2s).
        # Parameters: mu = PV2_1, phi_c = PV2_2, theta_c = PV2_3.
        mu = _pv_param(pv2, 1, 0.0, like=xi)
        phi_c = _pv_param(pv2, 2, 0.0, like=xi)
        theta_c = _pv_param(pv2, 3, 90.0, like=xi)
        inv_r0 = 1.0 / R2D
        zp = mu * torch.sin(theta_c * D2R) + 1.0
        xp = mu * torch.cos(theta_c * D2R) * torch.sin(phi_c * D2R)
        yp = mu * torch.cos(theta_c * D2R) * torch.cos(phi_c * D2R)
        # Torchsky zenithal convention: x_wcslib = -xi, so xr = -xi/r0
        xr = -xi * inv_r0
        yr = eta * inv_r0
        r2 = xr * xr + yr * yr
        x1 = (xr - xp) / zp
        y1 = (yr - yp) / zp
        xy = xr * x1 + yr * y1
        t_val = x1 * x1 + y1 * y1
        a_coef = t_val + 1.0
        b_coef = xy - t_val
        c_coef = r2 - xy - xy + t_val - 1.0
        disc = b_coef * b_coef - a_coef * c_coef
        valid = (disc >= 0.0) & (r2 >= 1e-10)
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        sinth1 = (-b_coef + sqrt_disc) / a_coef
        sinth2 = (-b_coef - sqrt_disc) / a_coef
        sinthe = torch.maximum(sinth1, sinth2)
        sinthe = torch.clamp(sinthe, -1.0, 1.0)
        theta_val = torch.asin(sinthe) * R2D
        z = 1.0 - sinthe
        phi_val = torch.atan2(xr - x1 * z, -(yr - y1 * z)) * R2D
        # Small-angle formula for r2 < 1e-10
        theta_small = 90.0 - R2D * torch.sqrt(torch.clamp(r2 / (1.0 + xy), min=0.0))
        phi_small = torch.atan2(xr, -yr) * R2D
        theta = torch.where(r2 < 1e-10, theta_small, theta_val)
        phi_out = torch.where(r2 < 1e-10, phi_small, phi_val)
        theta = torch.where(
            valid | (r2 < 1e-10), theta, torch.full_like(theta, float("nan"))
        )
        phi_out = torch.where(
            valid | (r2 < 1e-10), phi_out, torch.full_like(phi_out, float("nan"))
        )
        phi = phi_out

    elif projection_code == "AIR":
        # Airy's zenithal (WCSLIB airx2s).
        # Parameter: theta_b = PV2_1 (default 90.0).
        theta_b = _pv_param(pv2, 1, 90.0, like=xi)
        r0 = R2D
        w0 = 2.0 * r0
        if float(theta_b.item()) == 90.0:
            w1 = -0.5
            w2 = 1.0
        else:
            cosxi_b = math.cos((90.0 - float(theta_b.item())) * 0.5 * D2R)
            cosxi_b_sq = cosxi_b * cosxi_b
            w1 = math.log(cosxi_b) * cosxi_b_sq / (1.0 - cosxi_b_sq)
            w2 = 0.5 - w1
        w5 = w2 * 1e-4  # tolerance-scaled small-r threshold
        w6 = R2D / w2
        r = torch.hypot(xi, eta) / w0
        # Torchsky zenithal convention: phi = atan2(-xi, -eta)
        phi = torch.where(r == 0.0, torch.zeros_like(r), torch.atan2(-xi, -eta) * R2D)
        # Small-r linear approximation: r ≈ xi when xi is small,
        # so for r < w5 the error is negligible.
        xi_small = r * w6
        # Solve r = 0.5*log1p(t^2)/t - w1*t with t = tan(xi) by Newton.
        # The direct r(xi) relation has no closed form; t is well conditioned
        # near xi=0 and the derivative is analytic:
        #   d/dt [0.5*log1p(t^2)/t - w1*t] = (t^2/(1+t^2) - 0.5*log1p(t^2))/t^2 - w1
        # Start from the small-angle root t ~ r/w2 and use a fixed iteration
        # count (no host-side convergence syncs).
        t_val = torch.clamp(r / w2, min=1.0e-300)
        for _ in range(8):
            t2 = t_val * t_val
            log1p_t = torch.log1p(t2)
            value = 0.5 * log1p_t / t_val - w1 * t_val - r
            derivative = (t2 / (1.0 + t2) - 0.5 * log1p_t) / t2 - w1
            t_val = t_val - value / derivative
        # Residual from the final iterate drives the validity mask.
        t2_final = t_val * t_val
        log1p_final = torch.log1p(t2_final)
        rt_final = 0.5 * log1p_final / t_val - w1 * t_val - r
        xi_val = torch.atan(t_val) * R2D

        xi_rad = torch.where(r < w5, xi_small, xi_val)
        theta = 90.0 - 2.0 * xi_rad
        # Valid: small-r path, or iterative path converged (residual small)
        converged = torch.abs(rt_final) < 1e-10
        valid = (r < w5) | converged
        theta = torch.where(valid, theta, torch.full_like(theta, float("nan")))
        phi = torch.where(valid, phi, torch.full_like(phi, float("nan")))

    else:
        raise NotImplementedError(f"Zenithal code {projection_code} not implemented.")

    return phi, theta


def deproject_zenithal(
    phi: Tensor,
    theta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Inverse: Spherical (phi, theta) -> Intermediate (xi, eta).
    """
    # R depends on theta
    r = torch.zeros_like(theta)

    # theta is native latitude [-90, 90]
    # colatitude w = 90 - theta?

    if projection_code == "TAN":
        # R = 180/pi * cot(theta)
        # Avoid theta=0 singularity
        # theta in degrees.
        # cot(theta) = 1/tan(theta)
        tan_theta = torch.tan(theta * D2R)
        valid = (
            torch.isfinite(theta)
            & (theta > 64.0 * torch.finfo(theta.dtype).eps * R2D)
            & (theta <= 90.0 + 1e-12)
        )
        tan_safe = torch.where(valid, tan_theta, torch.ones_like(tan_theta))
        r = torch.where(
            valid,
            R2D / tan_safe,
            torch.full_like(theta, float("nan")),
        )

    elif projection_code == "SIN":
        theta_rad = theta * D2R
        phi_rad = phi * D2R
        sin_phi, cos_phi = _sincos(phi_rad)
        cos_theta = torch.cos(theta_rad)
        z = torch.sin(theta_rad)
        a = pv2[1] if pv2 is not None and pv2.numel() > 1 else 0.0
        b = pv2[2] if pv2 is not None and pv2.numel() > 2 else 0.0
        xi = (-cos_theta * sin_phi + a * (1.0 - z)) * R2D
        eta = (-cos_theta * cos_phi + b * (1.0 - z)) * R2D
        return xi, eta

    elif projection_code == "ARC":
        valid = torch.isfinite(theta) & (theta >= -90.0) & (theta <= 90.0)
        r = torch.where(
            valid,
            90.0 - theta,
            torch.full_like(theta, float("nan")),
        )

    elif projection_code == "ZEA":
        r = 2.0 * torch.sin((90.0 - theta) * (0.5 * D2R)) * R2D

    elif projection_code == "STG":
        r = 2.0 * torch.tan((90.0 - theta) * (0.5 * D2R)) * R2D

    elif projection_code == "ZPN":
        if pv2 is not None:
            p_coeffs = pv2[:31]
            mask_valid = p_coeffs.abs() > 1e-15
            if not mask_valid.any():
                r = 90.0 - theta
            else:
                # Find last valid coefficient for order
                valid_indices = torch.nonzero(mask_valid, as_tuple=True)[0]
                max_m = valid_indices[-1].item()
                coeffs = p_coeffs[: max_m + 1]

                # Horner's method for P(w), where w and the result are radians.
                w = (90.0 - theta) * D2R
                rev_c = torch.flip(coeffs, dims=(0,))
                c0 = rev_c[0]
                val = torch.full_like(w, c0.item())
                for c in rev_c[1:]:
                    val.mul_(w).add_(c)
                r = val * R2D
        else:
            r = 90.0 - theta

    elif projection_code == "AZP":
        # Forward AZP (WCSLIB azps2x).
        # Parameters: mu = PV2_1, gamma = PV2_2.
        mu = _pv_param(pv2, 1, 0.0, like=phi)
        gamma = _pv_param(pv2, 2, 0.0, like=phi)
        r0 = R2D
        w0 = r0 * (mu + 1.0)
        cos_g = math.cos(float(gamma.item()) * D2R)
        sin_g = math.sin(float(gamma.item()) * D2R)
        w1 = sin_g / cos_g  # tan(gamma)
        w2 = 1.0 / cos_g  # sec(gamma)
        phi_rad = phi * D2R
        sin_phi, cos_phi = _sincos(phi_rad)
        theta_rad = theta * D2R
        sin_the = torch.sin(theta_rad)
        cos_the = torch.cos(theta_rad)
        s = w1 * cos_phi
        t = (mu + sin_the) + cos_the * s
        safe_t = torch.where(t == 0.0, torch.ones_like(t), t)
        r_val = w0 * cos_the / safe_t
        # Torchsky zenithal convention: xi = -x = -r*sin(phi)
        xi = -r_val * sin_phi
        eta = -r_val * cos_phi * w2
        xi = torch.where(t == 0.0, torch.full_like(xi, float("nan")), xi)
        eta = torch.where(t == 0.0, torch.full_like(eta, float("nan")), eta)
        return xi, eta

    elif projection_code == "SZP":
        # Forward SZP (WCSLIB szps2x).
        mu = _pv_param(pv2, 1, 0.0, like=phi)
        phi_c = _pv_param(pv2, 2, 0.0, like=phi)
        theta_c = _pv_param(pv2, 3, 90.0, like=phi)
        r0 = R2D
        zp = float(mu.item()) * math.sin(float(theta_c.item()) * D2R) + 1.0
        xp = (
            float(mu.item())
            * math.cos(float(theta_c.item()) * D2R)
            * math.sin(float(phi_c.item()) * D2R)
        )
        yp = (
            float(mu.item())
            * math.cos(float(theta_c.item()) * D2R)
            * math.cos(float(phi_c.item()) * D2R)
        )
        w4 = r0 * xp
        w5 = r0 * yp
        w6 = r0 * zp
        phi_rad = phi * D2R
        sin_phi, cos_phi = _sincos(phi_rad)
        s = 1.0 - torch.sin(theta * D2R)
        t = zp - s
        safe_t = torch.where(t == 0.0, torch.ones_like(t), t)
        r_val = w6 * torch.cos(theta * D2R) / safe_t
        u = w4 * s / safe_t
        v = w5 * s / safe_t
        # Torchsky zenithal convention: xi = -x = -(r*sin(phi) - u)
        xi = -r_val * sin_phi + u
        eta = -r_val * cos_phi - v
        xi = torch.where(t == 0.0, torch.full_like(xi, float("nan")), xi)
        eta = torch.where(t == 0.0, torch.full_like(eta, float("nan")), eta)
        return xi, eta

    elif projection_code == "AIR":
        # Forward AIR (WCSLIB airs2x).
        theta_b = _pv_param(pv2, 1, 90.0, like=phi)
        r0 = R2D
        w0 = 2.0 * r0
        tb = float(theta_b.item())
        if tb == 90.0:
            w1 = -0.5
            w2 = 1.0
        else:
            cosxi_b = math.cos((90.0 - tb) * 0.5 * D2R)
            cosxi_b_sq = cosxi_b * cosxi_b
            w1 = math.log(cosxi_b) * cosxi_b_sq / (1.0 - cosxi_b_sq)
            w2 = 0.5 - w1
        w3 = w0 * w2
        w4 = 1e-4  # tolerance
        phi_rad = phi * D2R
        sin_phi, cos_phi = _sincos(phi_rad)
        valid = torch.isfinite(theta) & (theta > -90.0) & (theta <= 90.0)
        safe_theta = torch.where(valid, theta, torch.zeros_like(theta))
        # Compute r for each theta: xi_half = (90 - theta)/2 in radians
        xi_half = (90.0 - safe_theta) * 0.5 * D2R
        cos_xi = torch.cos(xi_half)
        tan_xi = torch.tan(xi_half)
        # Small-r approximation: r = xi_half * w3 (linear)
        r_small = xi_half * w3
        # Full formula: r = -w0 * (log(cos(xi))/tan(xi) + w1*tan(xi))
        safe_tan = torch.where(tan_xi == 0.0, torch.ones_like(tan_xi), tan_xi)
        r_large = -(w0 * (torch.log(cos_xi) / safe_tan + w1 * tan_xi))
        r = torch.where(xi_half.abs() < w4, r_small, r_large)
        r = torch.where(safe_theta == 90.0, torch.zeros_like(theta), r)
        r = torch.where(valid, r, torch.full_like(r, float("nan")))
        # Torchsky zenithal convention: xi = -x = -r*sin(phi)
        xi = -r * sin_phi
        eta = -r * cos_phi
        return xi, eta

    # xi = -R * sin(phi)
    # eta = -R * cos(phi)  (torchsky zenithal convention)

    phi_rad = phi * D2R
    sin_phi, cos_phi = _sincos(phi_rad)
    xi = -r * sin_phi
    eta = -r * cos_phi

    return xi, eta
