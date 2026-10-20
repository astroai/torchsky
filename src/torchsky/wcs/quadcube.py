"""Quadrilateralized spherical cube WCS projections (TSC, CSC, QSC).

Paper II Sec.5.6 / WCSLIB face layout: six faces in a +-shaped cross,
each 90° on a side, origin at the centre of face 1.
  Face 0: +Z (north) at (0, 90)
  Faces 1..4: equatorial at (0,0), (90,0), (180,0), (270,0)
  Face 5: -Z (south) at (0, -90)

Native reference: (phi_0, theta_0) = (0, 0).
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Tuple

D2R = 0.017453292519943295
R2D = 57.29577951308232
_INV_SQRT2 = 0.7071067811865476

# Face centres (cx, cy) in degrees — Paper II / WCSLIB.
_FACE_CENTRES = (
    (0.0, 90.0),
    (0.0, 0.0),
    (90.0, 0.0),
    (180.0, 0.0),
    (270.0, 0.0),
    (0.0, -90.0),
)

# Chan & O'Neill (1975) CSC forward polynomial coefficients.
_CSC_GSTAR = 1.37484847732
_CSC_M = 0.004869491981
_CSC_GAMMA = -0.13161671474
_CSC_OMEGA1 = -0.159596235474
_CSC_D0 = 0.0759196200467
_CSC_D1 = -0.0217762490699
_CSC_C00 = 0.141189631152
_CSC_C10 = 0.0809701286525
_CSC_C01 = -0.281528535557
_CSC_C11 = 0.15384112876
_CSC_C20 = -0.178251207466
_CSC_C02 = 0.106959469314

# CSC inverse polynomial coefficients.
_CSC_P00 = -0.27292696
_CSC_P10 = -0.07629969
_CSC_P20 = -0.22797056
_CSC_P30 = 0.54852384
_CSC_P40 = -0.62930065
_CSC_P50 = 0.25795794
_CSC_P60 = 0.02584375
_CSC_P01 = -0.02819452
_CSC_P11 = -0.01471565
_CSC_P21 = 0.48051509
_CSC_P31 = -1.74114454
_CSC_P41 = 1.71547508
_CSC_P51 = -0.53022337
_CSC_P02 = 0.27058160
_CSC_P12 = -0.56800938
_CSC_P22 = 0.30803317
_CSC_P32 = 0.98938102
_CSC_P42 = -0.83180469
_CSC_P03 = -0.60441560
_CSC_P13 = 1.50880086
_CSC_P23 = -0.93678576
_CSC_P33 = 0.08693841
_CSC_P04 = 0.93412077
_CSC_P14 = -1.41601920
_CSC_P24 = 0.33887446
_CSC_P05 = -0.63915306
_CSC_P15 = 0.52032238
_CSC_P06 = 0.14381585

# Inverse Horner terms, innermost coefficient last (WCSLIB cscx2s).
_CSC_INV_Z0 = (_CSC_P00, _CSC_P10, _CSC_P20, _CSC_P30, _CSC_P40, _CSC_P50, _CSC_P60)
_CSC_INV_Z1 = (_CSC_P01, _CSC_P11, _CSC_P21, _CSC_P31, _CSC_P41, _CSC_P51)
_CSC_INV_Z2 = (_CSC_P02, _CSC_P12, _CSC_P22, _CSC_P32, _CSC_P42)
_CSC_INV_Z3 = (_CSC_P03, _CSC_P13, _CSC_P23, _CSC_P33)
_CSC_INV_Z4 = (_CSC_P04, _CSC_P14, _CSC_P24)
_CSC_INV_Z5 = (_CSC_P05, _CSC_P15)


def _horner_addcmul(x: Tensor, coeffs: tuple[float, ...]) -> Tensor:
    """Float32 Horner via addcmul so FMA matches WCSLIB/C ``a + x*b``."""
    acc = x.new_tensor(coeffs[-1])
    for coeff in reversed(coeffs[:-1]):
        acc = torch.addcmul(x.new_tensor(coeff), x, acc)
    return acc


def _horner_addcmul_terms(x: Tensor, terms: tuple[Tensor, ...]) -> Tensor:
    acc = terms[-1]
    for term in reversed(terms[:-1]):
        acc = torch.addcmul(term, x, acc)
    return acc


def project_quadcube(
    xi: Tensor,
    eta: Tensor,
    projection_code: str,
) -> Tuple[Tensor, Tensor]:
    """Inverse quad-cube: plane (xi, eta) -> native (phi, theta)."""
    code = projection_code.upper()
    if code == "TSC":
        return _tsc_x2s(xi, eta)
    if code == "CSC":
        return _csc_x2s(xi, eta)
    if code == "QSC":
        return _qsc_x2s(xi, eta)
    raise NotImplementedError(f"Quad-cube code {projection_code} not implemented.")


def deproject_quadcube(
    phi: Tensor,
    theta: Tensor,
    projection_code: str,
) -> Tuple[Tensor, Tensor]:
    """Forward quad-cube: native (phi, theta) -> plane (xi, eta)."""
    code = projection_code.upper()
    if code == "TSC":
        return _tsc_s2x(phi, theta)
    if code == "CSC":
        return _csc_s2x(phi, theta)
    if code == "QSC":
        return _qsc_s2x(phi, theta)
    raise NotImplementedError(
        f"Quad-cube forward projection {projection_code} not implemented."
    )


def _direction_cosines(phi: Tensor, theta: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    p = phi * D2R
    t = theta * D2R
    ct = torch.cos(t)
    return ct * torch.cos(p), ct * torch.sin(p), torch.sin(t)


def _face_from_lmn(
    lx: Tensor, m: Tensor, n: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pick dominating face; return face, xi, eta, zeta (face-local)."""
    face = torch.zeros_like(lx, dtype=torch.long)
    zeta = n.clone()
    # Face 1: +lx
    better = lx > zeta
    face = torch.where(better, torch.ones_like(face), face)
    zeta = torch.where(better, lx, zeta)
    # Face 2: +m
    better = m > zeta
    face = torch.where(better, torch.full_like(face, 2), face)
    zeta = torch.where(better, m, zeta)
    # Face 3: -lx
    better = -lx > zeta
    face = torch.where(better, torch.full_like(face, 3), face)
    zeta = torch.where(better, -lx, zeta)
    # Face 4: -m
    better = -m > zeta
    face = torch.where(better, torch.full_like(face, 4), face)
    zeta = torch.where(better, -m, zeta)
    # Face 5: -n
    better = -n > zeta
    face = torch.where(better, torch.full_like(face, 5), face)
    zeta = torch.where(better, -n, zeta)

    xi = torch.where(
        face == 0,
        m,
        torch.where(
            face == 1,
            m,
            torch.where(
                face == 2,
                -lx,
                torch.where(
                    face == 3,
                    -m,
                    torch.where(face == 4, lx, m),
                ),
            ),
        ),
    )
    eta = torch.where(
        face == 0,
        -lx,
        torch.where(
            face == 1,
            n,
            torch.where(
                face == 2,
                n,
                torch.where(
                    face == 3,
                    n,
                    torch.where(face == 4, n, lx),
                ),
            ),
        ),
    )
    return face, xi / zeta, eta / zeta, zeta


def _face_to_xy(face: Tensor, xf: Tensor, yf: Tensor) -> Tuple[Tensor, Tensor]:
    cx = torch.zeros_like(xf)
    cy = torch.zeros_like(yf)
    for f, (fcx, fcy) in enumerate(_FACE_CENTRES):
        m = face == f
        cx = torch.where(m, torch.full_like(cx, fcx), cx)
        cy = torch.where(m, torch.full_like(cy, fcy), cy)
    return cx + 45.0 * xf, cy + 45.0 * yf


def _xy_to_face(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return face, xf, yf, valid for plane points on the cross layout."""
    face = torch.full_like(x, -1, dtype=torch.long)
    xf = torch.zeros_like(x)
    yf = torch.zeros_like(y)
    valid = torch.zeros_like(x, dtype=torch.bool)
    for f, (cx, cy) in enumerate(_FACE_CENTRES):
        fx = (x - cx) / 45.0
        fy = (y - cy) / 45.0
        inside = (fx.abs() <= 1.0 + 1e-9) & (fy.abs() <= 1.0 + 1e-9) & (~valid)
        face = torch.where(inside, torch.full_like(face, f), face)
        xf = torch.where(inside, torch.clamp(fx, -1.0, 1.0), xf)
        yf = torch.where(inside, torch.clamp(fy, -1.0, 1.0), yf)
        valid = valid | inside
    return face, xf, yf, valid


def _lmn_to_native(lx: Tensor, m: Tensor, n: Tensor) -> Tuple[Tensor, Tensor]:
    theta = torch.asin(torch.clamp(n, -1.0, 1.0)) * R2D
    pole = (lx == 0.0) & (m == 0.0)
    phi = torch.where(pole, torch.zeros_like(lx), torch.atan2(m, lx) * R2D)
    return phi, theta


# ---------------------------------------------------------------------------
# TSC
# ---------------------------------------------------------------------------

def _tsc_s2x(phi: Tensor, theta: Tensor) -> Tuple[Tensor, Tensor]:
    lx, m, n = _direction_cosines(phi, theta)
    # Face selection matching fitsy/WCSLIB (dominating absolute component).
    abs_l, abs_m, abs_n = lx.abs(), m.abs(), n.abs()
    face = torch.zeros_like(lx, dtype=torch.long)
    # Prefer polar when |n| dominates
    north = (n >= abs_l) & (n >= abs_m)
    south = (~north) & (-n >= abs_l) & (-n >= abs_m)
    face = torch.where(south, torch.full_like(face, 5), face)
    eq = ~(north | south)
    f1 = eq & (lx >= abs_m) & (lx >= abs_n)
    f2 = eq & (~f1) & (m >= abs_l) & (m >= abs_n)
    f3 = eq & (~f1) & (~f2) & (-lx >= abs_m) & (-lx >= abs_n)
    f4 = eq & (~f1) & (~f2) & (~f3)
    face = torch.where(f1, torch.ones_like(face), face)
    face = torch.where(f2, torch.full_like(face, 2), face)
    face = torch.where(f3, torch.full_like(face, 3), face)
    face = torch.where(f4, torch.full_like(face, 4), face)
    # Keep face 0 where north
    face = torch.where(north, torch.zeros_like(face), face)

    xi = torch.where(
        face == 0,
        m / n,
        torch.where(
            face == 5,
            m / (-n),
            torch.where(
                face == 1,
                m / lx,
                torch.where(
                    face == 2,
                    -lx / m,
                    torch.where(face == 3, -m / (-lx), lx / (-m)),
                ),
            ),
        ),
    )
    eta = torch.where(
        face == 0,
        -lx / n,
        torch.where(
            face == 5,
            lx / (-n),
            torch.where(
                face == 1,
                n / lx,
                torch.where(
                    face == 2,
                    n / m,
                    torch.where(face == 3, n / (-lx), n / (-m)),
                ),
            ),
        ),
    )
    return _face_to_xy(face, xi, eta)


def _tsc_x2s(xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
    face, xf, yf, valid = _xy_to_face(xi, eta)
    norm = torch.sqrt(1.0 + xf * xf + yf * yf)
    lx = torch.where(
        face == 0,
        -yf / norm,
        torch.where(
            face == 1,
            1.0 / norm,
            torch.where(
                face == 2,
                -xf / norm,
                torch.where(
                    face == 3,
                    -1.0 / norm,
                    torch.where(face == 4, xf / norm, yf / norm),
                ),
            ),
        ),
    )
    m = torch.where(
        face == 0,
        xf / norm,
        torch.where(
            face == 1,
            xf / norm,
            torch.where(
                face == 2,
                1.0 / norm,
                torch.where(
                    face == 3,
                    -xf / norm,
                    torch.where(face == 4, -1.0 / norm, xf / norm),
                ),
            ),
        ),
    )
    n = torch.where(
        face == 0,
        1.0 / norm,
        torch.where(
            face == 5,
            -1.0 / norm,
            yf / norm,
        ),
    )
    phi, theta = _lmn_to_native(lx, m, n)
    nan = torch.full_like(phi, float("nan"))
    return torch.where(valid, phi, nan), torch.where(valid, theta, nan)


# ---------------------------------------------------------------------------
# CSC
# ---------------------------------------------------------------------------

def _csc_forward_poly(chi: Tensor, psi: Tensor) -> Tensor:
    # WCSLIB cscs2x evaluates this polynomial in float32 with FMA Horner.
    chi_f = chi.to(dtype=torch.float32)
    psi_f = psi.to(dtype=torch.float32)
    chi2 = chi_f * chi_f
    psi2 = psi_f * psi_f
    one = chi2.new_tensor(1.0)
    chi2co = one - chi2
    psi2co = one - psi2
    chi4 = torch.where(chi2 > 1.0e-16, chi2 * chi2, torch.zeros_like(chi2))
    psi4 = torch.where(psi2 > 1.0e-16, psi2 * psi2, torch.zeros_like(psi2))
    chipsi = (chi_f * psi_f).abs()
    chi2psi2 = torch.where(chipsi > 1.0e-16, chi2 * psi2, torch.zeros_like(chi2))
    inner = torch.addcmul(
        torch.addcmul(chi2.new_tensor(_CSC_C00), chi2.new_tensor(_CSC_C10), chi2),
        chi2.new_tensor(_CSC_C01),
        psi2,
    )
    inner = torch.addcmul(inner, chi2.new_tensor(_CSC_C11), chi2psi2)
    inner = torch.addcmul(inner, chi2.new_tensor(_CSC_C20), chi4)
    inner = torch.addcmul(inner, chi2.new_tensor(_CSC_C02), psi4)
    mid = torch.addcmul(
        torch.addcmul(
            chi2.new_tensor(_CSC_GAMMA) * chi2co, chi2.new_tensor(_CSC_M), chi2
        ),
        psi2co,
        inner,
    )
    omega = chi2.new_tensor(_CSC_OMEGA1) - chi2co * torch.addcmul(
        chi2.new_tensor(_CSC_D0), chi2.new_tensor(_CSC_D1), chi2
    )
    gterm = torch.addcmul(
        torch.addcmul(chi2.new_tensor(_CSC_GSTAR), psi2, mid),
        chi2,
        omega,
    )
    out = chi_f * torch.addcmul(chi2, chi2co, gterm)
    return out.to(dtype=chi.dtype)


def _csc_inverse_poly(xf: Tensor, yf: Tensor) -> Tensor:
    xf_f = xf.to(dtype=torch.float32)
    yf_f = yf.to(dtype=torch.float32)
    xx = xf_f * xf_f
    yy = yf_f * yf_f
    z0 = _horner_addcmul(xx, _CSC_INV_Z0)
    z1 = _horner_addcmul(xx, _CSC_INV_Z1)
    z2 = _horner_addcmul(xx, _CSC_INV_Z2)
    z3 = _horner_addcmul(xx, _CSC_INV_Z3)
    z4 = _horner_addcmul(xx, _CSC_INV_Z4)
    z5 = _horner_addcmul(xx, _CSC_INV_Z5)
    chi = _horner_addcmul_terms(yy, (z0, z1, z2, z3, z4, z5, xx.new_tensor(_CSC_P06)))
    one_minus_xx = xx.new_tensor(1.0) - xx
    return torch.addcmul(xf_f, xf_f * one_minus_xx, chi)


def _csc_cpp_pair(
    fn_name: str, a: Tensor, b: Tensor
) -> Tuple[Tensor, Tensor] | None:
    del fn_name, a, b
    return None


def _csc_s2x(phi: Tensor, theta: Tensor) -> Tuple[Tensor, Tensor]:
    cpp_out = _csc_cpp_pair("wcs_csc_s2x", phi, theta)
    if cpp_out is not None:
        return cpp_out
    lx, m, n = _direction_cosines(phi, theta)
    face, chi, psi, _ = _face_from_lmn(lx, m, n)
    xf = torch.clamp(_csc_forward_poly(chi, psi), -1.0, 1.0)
    yf = torch.clamp(_csc_forward_poly(psi, chi), -1.0, 1.0)
    return _face_to_xy(face, xf, yf)


def _csc_x2s(xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
    cpp_out = _csc_cpp_pair("wcs_csc_x2s", xi, eta)
    if cpp_out is not None:
        return cpp_out
    face, xf, yf, valid = _xy_to_face(xi, eta)
    chi = _csc_inverse_poly(xf, yf)
    psi = _csc_inverse_poly(yf, xf)
    # WCSLIB: float32 χ²+ψ², then double sqrt. Promoting first is the 8 mas gap.
    sumsq = (chi * chi + psi * psi).to(dtype=xi.dtype)
    t = 1.0 / torch.sqrt(sumsq + 1.0)
    chi = chi.to(dtype=xi.dtype)
    psi = psi.to(dtype=xi.dtype)
    lx = torch.where(
        face == 0,
        -psi * t,
        torch.where(
            face == 1,
            t,
            torch.where(
                face == 2,
                -chi * t,
                torch.where(
                    face == 3,
                    -t,
                    torch.where(face == 4, chi * t, psi * t),
                ),
            ),
        ),
    )
    m = torch.where(
        face == 0,
        chi * t,
        torch.where(
            face == 1,
            chi * t,
            torch.where(
                face == 2,
                t,
                torch.where(
                    face == 3,
                    -chi * t,
                    torch.where(face == 4, -t, chi * t),
                ),
            ),
        ),
    )
    n = torch.where(
        face == 0,
        t,
        torch.where(face == 5, -t, psi * t),
    )
    phi, theta = _lmn_to_native(lx, m, n)
    nan = torch.full_like(phi, float("nan"))
    return torch.where(valid, phi, nan), torch.where(valid, theta, nan)


# ---------------------------------------------------------------------------
# QSC
# ---------------------------------------------------------------------------

def _qsc_s2x(phi: Tensor, theta: Tensor) -> Tuple[Tensor, Tensor]:
    north_pole = (theta - 90.0).abs() < 1e-12
    south_pole = (theta + 90.0).abs() < 1e-12
    lx, m, n = _direction_cosines(phi, theta)
    face, _, _, _ = _face_from_lmn(lx, m, n)

    # Remap (xi, eta, zeta) per face
    xi = torch.where(
        face == 1,
        m,
        torch.where(
            face == 2,
            -lx,
            torch.where(
                face == 3,
                -m,
                torch.where(face == 4, lx, torch.where(face == 5, m, m)),
            ),
        ),
    )
    eta = torch.where(
        face == 1,
        n,
        torch.where(
            face == 2,
            n,
            torch.where(
                face == 3,
                n,
                torch.where(face == 4, n, torch.where(face == 5, lx, -lx)),
            ),
        ),
    )
    zeta = torch.where(
        face == 1,
        lx,
        torch.where(
            face == 2,
            m,
            torch.where(
                face == 3,
                -lx,
                torch.where(face == 4, -m, torch.where(face == 5, -n, n)),
            ),
        ),
    )
    zeco = 1.0 - zeta
    origin = (xi == 0.0) & (eta == 0.0)
    direct = xi.abs() >= eta.abs()

    omega_d = torch.where(xi == 0.0, torch.zeros_like(xi), eta / xi)
    tau_d = 1.0 + omega_d * omega_d
    mag_d = torch.sqrt(
        torch.clamp(zeco / (1.0 - 1.0 / torch.sqrt(1.0 + tau_d)), min=0.0)
    )
    xf_d = torch.sign(xi) * mag_d
    # Avoid sign(0)=0 leaving xf=0 incorrectly for torch.sign
    xf_d = torch.where(xi == 0.0, torch.zeros_like(xf_d), xf_d)
    inner_d = (
        torch.atan(omega_d) - torch.asin(omega_d / torch.sqrt(tau_d + tau_d))
    ) * R2D
    yf_d = (xf_d / 15.0) * inner_d

    omega_i = torch.where(eta == 0.0, torch.zeros_like(eta), xi / eta)
    tau_i = 1.0 + omega_i * omega_i
    mag_i = torch.sqrt(
        torch.clamp(zeco / (1.0 - 1.0 / torch.sqrt(1.0 + tau_i)), min=0.0)
    )
    yf_i = torch.sign(eta) * mag_i
    yf_i = torch.where(eta == 0.0, torch.zeros_like(yf_i), yf_i)
    inner_i = (
        torch.atan(omega_i) - torch.asin(omega_i / torch.sqrt(tau_i + tau_i))
    ) * R2D
    xf_i = (yf_i / 15.0) * inner_i

    xf = torch.where(origin, torch.zeros_like(xi), torch.where(direct, xf_d, xf_i))
    yf = torch.where(origin, torch.zeros_like(eta), torch.where(direct, yf_d, yf_i))
    xf = torch.clamp(xf, -1.0, 1.0)
    yf = torch.clamp(yf, -1.0, 1.0)
    x, y = _face_to_xy(face, xf, yf)
    x = torch.where(north_pole, torch.zeros_like(x), x)
    y = torch.where(north_pole, torch.full_like(y, 90.0), y)
    x = torch.where(south_pole, torch.zeros_like(x), x)
    y = torch.where(south_pole, torch.full_like(y, -90.0), y)
    return x, y


def _qsc_x2s(xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
    face, xf, yf, valid = _xy_to_face(xi, eta)
    direct = xf.abs() > yf.abs()

    def _omega_tau_zeta(u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # u is the dominating face coord, v the other
        w = (15.0 * v / torch.where(u == 0.0, torch.ones_like(u), u)) * D2R
        sw = torch.sin(w)
        cw = torch.cos(w)
        o = sw / (cw - _INV_SQRT2)
        t = 1.0 + o * o
        z = 1.0 - u * u * (1.0 - 1.0 / torch.sqrt(1.0 + t))
        return o, t, z

    omega_d, tau_d, zeta_d = _omega_tau_zeta(xf, yf)
    omega_i, tau_i, zeta_i = _omega_tau_zeta(yf, xf)
    zero = (xf == 0.0) & (yf == 0.0)
    omega = torch.where(
        zero,
        torch.zeros_like(xf),
        torch.where(direct & (xf != 0.0), omega_d, torch.where((~direct) & (yf != 0.0), omega_i, torch.zeros_like(xf))),
    )
    tau = torch.where(
        zero,
        torch.ones_like(xf),
        torch.where(direct & (xf != 0.0), tau_d, torch.where((~direct) & (yf != 0.0), tau_i, torch.ones_like(xf))),
    )
    zeta = torch.where(
        zero,
        torch.ones_like(xf),
        torch.where(direct & (xf != 0.0), zeta_d, torch.where((~direct) & (yf != 0.0), zeta_i, torch.ones_like(xf))),
    )
    zeta = torch.clamp(zeta, -1.0, 1.0)
    zeco = 1.0 - zeta
    w = torch.sqrt(torch.clamp(zeco * (2.0 - zeco) / tau, min=0.0))

    # Reconstruct (lx, m, n) per face — matches fitsy/WCSLIB qscx2s.
    # Face 0 (default): n=zeta; direct => m=±w, lx=-m*omega; else lx=±w, m=-lx*omega
    m_d = torch.where(xf < 0.0, -w, w)
    lx_d = -m_d * omega
    lx_i = torch.where(yf > 0.0, -w, w)
    m_i = -lx_i * omega
    lx = torch.where(direct, lx_d, lx_i)
    m = torch.where(direct, m_d, m_i)
    n = zeta

    # Face 1
    lx1 = zeta
    m1 = torch.where(direct, torch.where(xf < 0.0, -w, w), torch.where(yf < 0.0, -w, w) * omega)
    n1 = torch.where(direct, torch.where(xf < 0.0, -w, w) * omega, torch.where(yf < 0.0, -w, w))

    # Face 2
    m2 = zeta
    lx2 = torch.where(direct, torch.where(xf > 0.0, -w, w), -torch.where(yf < 0.0, -w, w) * omega)
    n2 = torch.where(direct, -torch.where(xf > 0.0, -w, w) * omega, torch.where(yf < 0.0, -w, w))

    # Face 3
    lx3 = -zeta
    m3 = torch.where(direct, torch.where(xf > 0.0, -w, w), -torch.where(yf < 0.0, -w, w) * omega)
    n3 = torch.where(direct, -torch.where(xf > 0.0, -w, w) * omega, torch.where(yf < 0.0, -w, w))

    # Face 4
    m4 = -zeta
    lx4 = torch.where(direct, torch.where(xf < 0.0, -w, w), torch.where(yf < 0.0, -w, w) * omega)
    n4 = torch.where(direct, torch.where(xf < 0.0, -w, w) * omega, torch.where(yf < 0.0, -w, w))

    # Face 5
    n5 = -zeta
    m5 = torch.where(direct, torch.where(xf < 0.0, -w, w), torch.where(yf < 0.0, -w, w) * omega)
    lx5 = torch.where(direct, torch.where(xf < 0.0, -w, w) * omega, torch.where(yf < 0.0, -w, w))

    lx = torch.where(
        face == 1,
        lx1,
        torch.where(
            face == 2,
            lx2,
            torch.where(
                face == 3,
                lx3,
                torch.where(face == 4, lx4, torch.where(face == 5, lx5, lx)),
            ),
        ),
    )
    m = torch.where(
        face == 1,
        m1,
        torch.where(
            face == 2,
            m2,
            torch.where(
                face == 3,
                m3,
                torch.where(face == 4, m4, torch.where(face == 5, m5, m)),
            ),
        ),
    )
    n = torch.where(
        face == 1,
        n1,
        torch.where(
            face == 2,
            n2,
            torch.where(
                face == 3,
                n3,
                torch.where(face == 4, n4, torch.where(face == 5, n5, n)),
            ),
        ),
    )
    phi, theta = _lmn_to_native(lx, m, n)
    nan = torch.full_like(phi, float("nan"))
    return torch.where(valid, phi, nan), torch.where(valid, theta, nan)
