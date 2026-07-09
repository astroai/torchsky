"""Conic and Bonne WCS projections.

Implements the FITS-WCS conic projection family (COP, COE, COD, COO) and the
Bonne polyconic projection (BON) using torch-native operations, matching
WCSLIB (Calabretta & Greisen 2002, WCS Paper II) semantics.

All conic projections share the structure:
  - PV2_1: standard parallel theta_1 (latitude of the reference)
  - PV2_2: second standard parallel theta_2 (delta from theta_1 for COE/COD/COO)
  - Native reference point: (phi_0, theta_0) = (0, theta_1)

Forward (s2x): native (phi, theta) -> plane (x, y)
Inverse (x2s): plane (x, y) -> native (phi, theta)
"""

import math

import torch
from torch import Tensor
from typing import Tuple, Optional

D2R = 0.017453292519943295
R2D = 57.29577951308232


def _pv_param(
    params: Tensor | None,
    idx: int,
    default: float,
    *,
    like: Tensor,
) -> float:
    """Return PV parameter *idx* as a Python float with a default."""
    if params is None or idx >= int(params.numel()):
        return default
    return float(params[idx].item())


# ---------------------------------------------------------------------------
# COP — Conic Perspective
# ---------------------------------------------------------------------------

def _cop_constants(
    pv1: Optional[Tensor], pv2: Optional[Tensor], like: Tensor
) -> tuple[float, float, float, float, float, float]:
    """Compute COP constants w[0..5] from PV parameters.

    WCSLIB convention (copset.c): PV2_1 = sigma (cone apex latitude),
    PV2_2 = delta (half-difference of standard parallels).
    When PV2_2 is 0, the single-standard-parallel case degenerates
    to standard parallels coincident with sigma.

    """
    sigma = _pv_param(pv2, 1, 0.0, like=like)
    delta = _pv_param(pv2, 2, 0.0, like=like)
    r0 = R2D
    w0 = math.sin(sigma * D2R)
    if w0 == 0.0:
        raise ValueError("COP: sin(PV2_1) must be non-zero")
    w1 = 1.0 / w0
    # w[3] = r0 * cos(delta) per WCSLIB copset.c.
    w3 = r0 * math.cos(delta * D2R)
    if w3 == 0.0:
        raise ValueError("COP: r0*cos(PV2_2) must be non-zero")
    w4 = 1.0 / w3
    w5 = 1.0 / math.tan(sigma * D2R)
    w2 = w3 * w5
    return w0, w1, w2, w3, w4, w5


def project_conic(
    xi: Tensor,
    eta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Inverse conic projection: plane (x, y) -> native (phi, theta)."""
    if projection_code == "BON":
        return _bon_x2s(xi, eta, pv1, pv2)

    theta1 = _pv_param(pv2, 1, 0.0, like=xi)

    if projection_code == "COP":
        w0, w1, w2, w3, w4, w5 = _cop_constants(pv1, pv2, like=xi)
        dy = w2 - eta
        dy2 = dy * dy
        r = torch.sqrt(xi * xi + dy2)
        if theta1 < 0.0:
            r = -r
        alpha = torch.where(
            r == 0.0,
            torch.zeros_like(r),
            torch.atan2(xi / r, dy / r) * R2D,
        )
        # Negate phi: the pole rotation (delta_p=90°) expects the opposite
        # sign convention from the natural atan2 output.
        phi = -alpha * w1
        theta = theta1 + torch.atan2(
            torch.tensor(w5, dtype=xi.dtype, device=xi.device)
            - r * w4,
            torch.ones_like(r),
        ) * R2D
        return phi, theta

    elif projection_code == "COE":
        w0, w1, w2, w3, w4, w5, w6, w7, w8 = _coe_constants(pv1, pv2, like=xi)
        dy = w2 - eta
        dy2 = dy * dy
        r = torch.sqrt(xi * xi + dy2)
        if theta1 < 0.0:
            r = -r
        alpha = torch.where(
            r == 0.0,
            torch.zeros_like(r),
            torch.atan2(xi / r, dy / r) * R2D,
        )
        w = (w6 - r * r) * w7
        tol = 1e-12
        # Handle edge cases near poles
        theta_pole = torch.where(
            (r - w8).abs() < tol,
            torch.tensor(-90.0, dtype=xi.dtype, device=xi.device),
            torch.where(
                (w - 1.0).abs() < tol,
                torch.tensor(90.0, dtype=xi.dtype, device=xi.device),
                torch.where(
                    (w + 1.0).abs() < tol,
                    torch.tensor(-90.0, dtype=xi.dtype, device=xi.device),
                    torch.asin(torch.clamp(w, -1.0, 1.0)) * R2D,
                ),
            ),
        )
        valid = (w.abs() <= 1.0 + tol) | ((r - w8).abs() < tol)
        phi = -alpha * w1
        theta = torch.where(valid, theta_pole, torch.full_like(theta_pole, float("nan")))
        return phi, theta

    elif projection_code == "COD":
        w0, w1, w2, w3 = _cod_constants(pv1, pv2, like=xi)
        dy = w2 - eta
        dy2 = dy * dy
        r = torch.sqrt(xi * xi + dy2)
        if theta1 < 0.0:
            r = -r
        alpha = torch.where(
            r == 0.0,
            torch.zeros_like(r),
            torch.atan2(xi / r, dy / r) * R2D,
        )
        phi = -alpha * w1
        theta = w3 - r
        return phi, theta

    elif projection_code == "COO":
        w0, w1, w2, w3, w4 = _coo_constants(pv1, pv2, like=xi)
        dy = w2 - eta
        dy2 = dy * dy
        r = torch.sqrt(xi * xi + dy2)
        if theta1 < 0.0:
            r = -r
        alpha = torch.where(
            r == 0.0,
            torch.zeros_like(r),
            torch.atan2(xi / r, dy / r) * R2D,
        )
        # theta = 90 - 2*atan((r*w4)^(1/w0))
        pow_arg = torch.clamp(r * w4, min=1e-30)
        theta = 90.0 - 2.0 * torch.atan(torch.pow(pow_arg, 1.0 / w0)) * R2D
        # Handle r=0 edge: w0 > 0 -> pole (theta=90), w0 < 0 -> antipole (theta=-90)
        pole_theta = -90.0 if w0 < 0.0 else 90.0
        theta = torch.where(
            r == 0.0,
            torch.tensor(pole_theta, dtype=xi.dtype, device=xi.device),
            theta,
        )
        phi = -alpha * w1
        return phi, theta

    raise NotImplementedError(f"Conic code {projection_code} not implemented.")


def deproject_conic(
    phi: Tensor,
    theta: Tensor,
    projection_code: str,
    pv1: Optional[Tensor] = None,
    pv2: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward conic projection: native (phi, theta) -> plane (x, y).

    The native longitude *phi* arrives from the inverse spherical rotation
    mapped to [0, 360); conic linear angular operations (alpha = -w0 * phi)
    expect [-180, 180), so we normalise here."""
    # Wrap phi to [-180, 180) for linear angular scaling.
    phi = torch.remainder(phi + 180.0, 360.0) - 180.0

    if projection_code == "BON":
        return _bon_s2x(phi, theta, pv1, pv2)

    theta1 = _pv_param(pv2, 1, 0.0, like=phi)

    if projection_code == "COP":
        w0, w1, w2, w3, w4, w5 = _cop_constants(pv1, pv2, like=phi)
        # Negate alpha: the pole rotation (delta_p=90°) uses the opposite
        # sign convention from the natural atan2 output.
        alpha = -w0 * phi
        sin_a = torch.sin(alpha * D2R)
        cos_a = torch.cos(alpha * D2R)
        t = theta - theta1
        s = torch.cos(t * D2R)
        safe_s = torch.where(s == 0.0, torch.ones_like(s), s)
        r = w2 - w3 * torch.sin(t * D2R) / safe_s
        y0 = -w2  # y0 offset (prj->y0 - prj->w[2] with prj->y0 = 0)
        xi = r * sin_a
        eta = -r * cos_a - y0
        xi = torch.where(s == 0.0, torch.full_like(xi, float("nan")), xi)
        eta = torch.where(s == 0.0, torch.full_like(eta, float("nan")), eta)
        return xi, eta

    elif projection_code == "COE":
        w0, w1, w2, w3, w4, w5, w6, w7, w8 = _coe_constants(pv1, pv2, like=phi)
        alpha = -w0 * phi
        sin_a = torch.sin(alpha * D2R)
        cos_a = torch.cos(alpha * D2R)
        r = torch.where(
            torch.isclose(theta, torch.tensor(-90.0, dtype=phi.dtype, device=phi.device), atol=1e-12),
            torch.tensor(w8, dtype=phi.dtype, device=phi.device),
            w3 * torch.sqrt(
                torch.clamp(w4 - w5 * torch.sin(theta * D2R), min=0.0)
            ),
        )
        y0 = -w2
        xi = r * sin_a
        eta = -r * cos_a - y0
        return xi, eta

    elif projection_code == "COD":
        w0, w1, w2, w3 = _cod_constants(pv1, pv2, like=phi)
        alpha = -w0 * phi
        sin_a = torch.sin(alpha * D2R)
        cos_a = torch.cos(alpha * D2R)
        r = w3 - theta
        y0 = -w2
        xi = r * sin_a
        eta = -r * cos_a - y0
        return xi, eta

    elif projection_code == "COO":
        w0, w1, w2, w3, w4 = _coo_constants(pv1, pv2, like=phi)
        alpha = -w0 * phi
        sin_a = torch.sin(alpha * D2R)
        cos_a = torch.cos(alpha * D2R)
        half_arg = torch.clamp((90.0 - theta) * 0.5 * D2R, min=1e-30)
        tan_half = torch.tan(half_arg)
        r = w3 * torch.pow(tan_half, w0)
        y0 = -w2
        xi = r * sin_a
        eta = -r * cos_a - y0
        return xi, eta

    raise NotImplementedError(
        f"Conic forward projection {projection_code} not implemented."
    )


# ---------------------------------------------------------------------------
# COE — Conic Equal Area
# ---------------------------------------------------------------------------

def _coe_constants(
    pv1: Optional[Tensor], pv2: Optional[Tensor], like: Tensor
) -> tuple[float, ...]:
    theta1_ref = _pv_param(pv2, 1, 0.0, like=like)
    delta = _pv_param(pv2, 2, 0.0, like=like)
    theta1 = theta1_ref - delta
    theta2 = theta1_ref + delta
    r0 = R2D
    w0 = (math.sin(theta1 * D2R) + math.sin(theta2 * D2R)) / 2.0
    if w0 == 0.0:
        raise ValueError("COE: (sin(theta1)+sin(theta2))/2 must be non-zero")
    w1 = 1.0 / w0
    w3 = r0 / w0
    w4 = 1.0 + math.sin(theta1 * D2R) * math.sin(theta2 * D2R)
    w5 = 2.0 * w0
    w6 = w3 * w3 * w4
    w7 = 1.0 / (2.0 * r0 * w3)
    w8 = w3 * math.sqrt(w4 + w5)
    w2 = w3 * math.sqrt(max(w4 - w5 * math.sin(theta1_ref * D2R), 0.0))
    return w0, w1, w2, w3, w4, w5, w6, w7, w8


# ---------------------------------------------------------------------------
# COD — Conic Equidistant
# ---------------------------------------------------------------------------

def _cod_constants(
    pv1: Optional[Tensor], pv2: Optional[Tensor], like: Tensor
) -> tuple[float, float, float, float]:
    """Compute COD constants w[0..3] from PV parameters.

    WCSLIB convention (codset.c): PV2_1 = sigma (reference latitude),
    PV2_2 = delta (half-difference of standard parallels).
    Standard parallels are sigma - delta and sigma + delta.

    C = w[0] = r0 * sin(sigma) * sin(delta) / delta  (two-standard)
             = sin(sigma)                             (single-standard)
    Y0 = w[2] = r0 * cos(delta) * cos(sigma) / C
    w[3] = Y0 + sigma  (places fiducial at projection plane origin)
    """
    sigma = _pv_param(pv2, 1, 0.0, like=like)
    delta = _pv_param(pv2, 2, 0.0, like=like)
    r0 = R2D
    if delta == 0.0:
        w0 = math.sin(sigma * D2R)
    else:
        w0 = r0 * math.sin(sigma * D2R) * math.sin(delta * D2R) / delta
    if w0 == 0.0:
        raise ValueError("COD: w[0] must be non-zero")
    w1 = 1.0 / w0
    # WCSLIB: Y0 = r0 * cos(delta) * cos(sigma) / C
    w2 = r0 * math.cos(delta * D2R) * math.cos(sigma * D2R) / w0
    w3 = w2 + sigma
    return w0, w1, w2, w3


# ---------------------------------------------------------------------------
# COO — Conic Orthomorphic
# ---------------------------------------------------------------------------

def _coo_constants(
    pv1: Optional[Tensor], pv2: Optional[Tensor], like: Tensor
) -> tuple[float, float, float, float, float]:
    theta1_ref = _pv_param(pv2, 1, 0.0, like=like)
    delta = _pv_param(pv2, 2, 0.0, like=like)
    theta1 = theta1_ref - delta
    theta2 = theta1_ref + delta
    r0 = R2D
    tan1 = math.tan((90.0 - theta1) * 0.5 * D2R)
    cos1 = math.cos(theta1 * D2R)
    if theta1 == theta2:
        w0 = math.sin(theta1 * D2R)
    else:
        tan2 = math.tan((90.0 - theta2) * 0.5 * D2R)
        cos2 = math.cos(theta2 * D2R)
        w0 = math.log(cos2 / cos1) / math.log(tan2 / tan1)
    if w0 == 0.0:
        raise ValueError("COO: w[0] must be non-zero")
    w1 = 1.0 / w0
    w3 = r0 * (cos1 / w0) / (tan1 ** w0)
    if w3 == 0.0:
        raise ValueError("COO: w[3] must be non-zero")
    w2 = w3 * (math.tan((90.0 - theta1_ref) * 0.5 * D2R) ** w0)
    w4 = 1.0 / w3
    return w0, w1, w2, w3, w4


# ---------------------------------------------------------------------------
# BON — Bonne's projection (polyconic, equal-area)
# ---------------------------------------------------------------------------

def _bon_constants(
    pv1: Optional[Tensor], pv2: Optional[Tensor], like: Tensor
) -> tuple[float, float, float]:
    theta1 = _pv_param(pv2, 1, 0.0, like=like)
    if theta1 == 0.0:
        # Sanson-Flamsteed (= SFL); caller should handle this
        return 0.0, 1.0, 0.0
    r0 = R2D
    w1 = r0 * D2R  # = 1.0
    w2 = r0 * (math.cos(theta1 * D2R) / math.sin(theta1 * D2R) + theta1 * D2R)
    return theta1, w1, w2


def _bon_x2s(
    xi: Tensor, eta: Tensor, pv1: Optional[Tensor], pv2: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    """Inverse Bonne: plane (x, y) -> native (phi, theta)."""
    theta1, w1, w2 = _bon_constants(pv1, pv2, like=xi)
    if theta1 == 0.0:
        # Sanson-Flamsteed: phi = x/cos(theta), theta = y
        # Inverse: theta = eta, phi = xi / cos(eta)
        theta = eta
        phi = xi / torch.clamp(torch.cos(theta * D2R), min=1e-15)
        return phi, theta

    dy = w2 - eta
    dy2 = dy * dy
    r = torch.sqrt(xi * xi + dy2)
    if theta1 < 0.0:
        r = -r
    alpha = torch.where(
        r == 0.0,
        torch.zeros_like(r),
        torch.atan2(xi / r, dy / r) * R2D,
    )
    t = (w2 - r) / w1
    cos_t = torch.cos(t * D2R)
    safe_cos = torch.where(cos_t == 0.0, torch.ones_like(cos_t), cos_t)
    s = alpha * (r / R2D) / safe_cos
    s = torch.where(cos_t == 0.0, torch.zeros_like(s), s)
    phi = s
    theta = t
    return phi, theta


def _bon_s2x(
    phi: Tensor, theta: Tensor, pv1: Optional[Tensor], pv2: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    """Forward Bonne: native (phi, theta) -> plane (x, y).

    WCSLIB convention: yp = -r*cos(alpha) - y0  with y0 = -w2.  The inverse
    (x2s) path uses dy = w2 - yp, so that dy = r*cos(alpha) at the reference
    point where r = w2.
    """
    theta1, w1, w2 = _bon_constants(pv1, pv2, like=phi)
    if theta1 == 0.0:
        # Sanson-Flamsteed: x = phi*cos(theta), y = theta
        phi_w = torch.remainder(phi + 180.0, 360.0) - 180.0
        xi = phi_w * torch.cos(theta * D2R)
        eta = theta
        return xi, eta

    r = w2 - w1 * theta
    cos_t = torch.cos(theta * D2R)
    safe_r = torch.where(r == 0.0, torch.ones_like(r), r)
    s = cos_t / safe_r
    alpha = s * R2D * phi
    sin_a = torch.sin(alpha * D2R)
    cos_a = torch.cos(alpha * D2R)
    xi = r * sin_a
    eta = -r * cos_a + w2
    return xi, eta
