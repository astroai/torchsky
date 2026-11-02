"""Celestial pole/Euler-angle setup for FITS WCS (Calabretta & Greisen 2002, Paper II).

Pure-Python scalar helpers (WCSLIB ``celset`` semantics). They return the exact
native-to-celestial rotation implied by CRVAL, the projection's native fiducial
point ``(phi_0, theta_0)``, and LONPOLE/LATPOLE.
"""

from __future__ import annotations

import math

_TOL = 1e-5


def default_lonpole(delta0: float, theta0: float, phi0: float = 0.0) -> float:
    """Paper II default LONPOLE: ``phi_0`` if ``delta_0 >= theta_0`` else ``phi_0 + 180``.

    Wrapped to ``[-180, 180]`` like WCSLIB ``celset``.
    """
    lonpole = phi0 if delta0 >= theta0 else phi0 + 180.0
    if lonpole > 180.0:
        lonpole -= 360.0
    elif lonpole < -180.0:
        lonpole += 360.0
    return lonpole


def celestial_euler(
    alpha0: float,
    delta0: float,
    phi0: float,
    theta0: float,
    lonpole: float,
    latpole: float,
) -> tuple[float, float, float]:
    """Return WCSLIB Euler angles ``(alpha_p, 90 - delta_p, phi_p)`` in degrees.

    Implements Paper II eqs. 8-10: the native pole latitude ``delta_p`` has two
    solutions; the one closest to ``latpole`` is selected.
    """
    d2r = math.pi / 180.0
    r2d = 180.0 / math.pi
    if theta0 == 90.0:
        return alpha0, 90.0 - delta0, lonpole
    clat0 = math.cos(delta0 * d2r)
    slat0 = math.sin(delta0 * d2r)
    cphip = math.cos((lonpole - phi0) * d2r)
    sphip = math.sin((lonpole - phi0) * d2r)
    cthe0 = math.cos(theta0 * d2r)
    sthe0 = math.sin(theta0 * d2r)
    x = cthe0 * cphip
    y = sthe0
    z = math.hypot(x, y)
    if z == 0.0:
        if slat0 != 0.0:
            raise ValueError(
                "invalid LONPOLE/LATPOLE for the reference point (Paper II eq. 8)"
            )
        latp = latpole
    else:
        slz = slat0 / z
        if abs(slz) > 1.0 + _TOL:
            raise ValueError(
                "invalid LONPOLE for the reference point: no native pole latitude "
                "satisfies Paper II eq. 8"
            )
        slz = max(-1.0, min(1.0, slz))
        u = math.atan2(y, x) * r2d
        v = math.acos(slz) * r2d
        lat1 = (u + v + 180.0) % 360.0 - 180.0
        lat2 = (u - v + 180.0) % 360.0 - 180.0
        if abs(latpole - lat1) < abs(latpole - lat2):
            latp = lat1 if abs(lat1) < 90.0 + _TOL else lat2
        else:
            latp = lat2 if abs(lat2) < 90.0 + _TOL else lat1
        if abs(latp) > 90.0 + _TOL:
            raise ValueError(
                "invalid LONPOLE/LATPOLE: no native pole latitude within [-90, 90]"
            )
        latp = max(-90.0, min(90.0, latp))
    e1 = 90.0 - latp
    z = math.cos(latp * d2r) * clat0
    if abs(z) < _TOL:
        if abs(clat0) < _TOL:
            e0 = alpha0
        elif latp > 0.0:
            e0 = alpha0 + (lonpole - phi0) - 180.0
        else:
            e0 = alpha0 - (lonpole - phi0)
    else:
        xx = (sthe0 - math.sin(latp * d2r) * slat0) / z
        yy = sphip * cthe0 / clat0
        if xx == 0.0 and yy == 0.0:
            raise ValueError("invalid LONPOLE/LATPOLE for the reference point")
        e0 = alpha0 - math.atan2(yy, xx) * r2d
    return e0, e1, lonpole


def native_to_celestial_matrix(
    euler: tuple[float, float, float],
) -> list[list[float]]:
    """3x3 rotation taking native unit vectors to celestial unit vectors.

    ``R = Rz(e0) @ A(e1) @ Rz(-e2)`` for native ``(cos t cos p, cos t sin p, sin t)``.
    """
    d2r = math.pi / 180.0
    e0, e1, e2 = (a * d2r for a in euler)
    ce0, se0 = math.cos(e0), math.sin(e0)
    ce2, se2 = math.cos(e2), math.sin(e2)
    c1, s1 = math.cos(e1), math.sin(e1)
    rz0 = [[ce0, -se0, 0.0], [se0, ce0, 0.0], [0.0, 0.0, 1.0]]
    a = [[-c1, 0.0, s1], [0.0, -1.0, 0.0], [s1, 0.0, c1]]
    rz2 = [[ce2, se2, 0.0], [-se2, ce2, 0.0], [0.0, 0.0, 1.0]]
    return _matmul(rz0, _matmul(a, rz2))


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)
    ]


def transpose(m: list[list[float]]) -> list[list[float]]:
    return [[m[j][i] for j in range(3)] for i in range(3)]


def celestial_correction(
    alpha0: float,
    delta0: float,
    theta0: float,
    lonpole: float,
    latpole: float,
    exist_lonpole: float,
    exist_matrix: list[list[float]] | None = None,
    *,
    phi0: float = 0.0,
    exist_theta0: float | None = None,
    force: bool = False,
) -> list[list[float]] | None:
    """Rotation ``M`` with ``R_true = M @ R_exist``; ``None`` when it is the identity.

    ``R_exist`` is the native pole rotation baked into the projection kernels
    (``exist_lonpole`` with ``LATPOLE = 90``); ``R_true`` honours the header's
    LONPOLE/LATPOLE exactly. ``force`` keeps an identity matrix so callers can
    use a non-None result to gate native fast paths off.
    """
    r_true = native_to_celestial_matrix(
        celestial_euler(alpha0, delta0, phi0, theta0, lonpole, latpole)
    )
    r_exist = exist_matrix or native_to_celestial_matrix(
        celestial_euler(
            alpha0,
            delta0,
            0.0,
            theta0 if exist_theta0 is None else exist_theta0,
            exist_lonpole,
            90.0,
        )
    )
    m = _matmul(r_true, transpose(r_exist))
    dev = max(
        abs(m[i][j] - (1.0 if i == j else 0.0)) for i in range(3) for j in range(3)
    )
    return None if (dev < 1e-13 and not force) else m
