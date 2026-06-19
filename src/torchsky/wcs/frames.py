from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

D2R = 0.017453292519943295
R2D = 57.29577951308232
ARCSEC_TO_RAD = D2R / 3600.0
JD_B1900 = 2415020.31352
JD_B1950 = 2433282.42345905
TROPICAL_YEAR_DAYS = 365.242198781

_ICRS_TO_GAL = torch.tensor(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
    ],
    dtype=torch.float64,
)

_B1950_TO_J2000_M = torch.tensor(
    [
        [0.9999256794956877, -0.0111814832204662, -0.0048590038153592],
        [0.0111814832391717, 0.9999374848933135, -0.0000271625947142],
        [0.0048590037723143, -0.0000271702937440, 0.9999881946023742],
    ],
    dtype=torch.float64,
)

_FK4_CORR = torch.tensor(
    [
        [-0.0026455262, -1.1539918689, 2.1111346190],
        [1.1540628161, -0.0129042997, 0.0236021478],
        [-2.1112979048, -0.0056024448, 0.0102587734],
    ],
    dtype=torch.float64,
) * 1.0e-6


def _as_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(dtype=torch.float64)
    return torch.as_tensor(value, dtype=torch.float64)


def _wrap360(angle: Tensor) -> Tensor:
    return torch.remainder(angle, 360.0)


def _besselian_epoch_to_jd(epoch: float) -> float:
    return JD_B1900 + (float(epoch) - 1900.0) * TROPICAL_YEAR_DAYS


def _besselian_epoch_to_julian_epoch(epoch: float) -> float:
    return 2000.0 + (_besselian_epoch_to_jd(epoch) - 2451545.0) / 365.25


def _rotation_matrix_z(angle_deg: float, *, device: torch.device) -> Tensor:
    a = torch.as_tensor(angle_deg * D2R, dtype=torch.float64, device=device)
    s = torch.sin(a)
    c = torch.cos(a)
    z = torch.zeros((), dtype=torch.float64, device=device)
    o = torch.ones((), dtype=torch.float64, device=device)
    return torch.stack(
        (
            torch.stack((c, s, z)),
            torch.stack((-s, c, z)),
            torch.stack((z, z, o)),
        )
    )


def _rotation_matrix_y(angle_deg: float, *, device: torch.device) -> Tensor:
    a = torch.as_tensor(angle_deg * D2R, dtype=torch.float64, device=device)
    s = torch.sin(a)
    c = torch.cos(a)
    z = torch.zeros((), dtype=torch.float64, device=device)
    o = torch.ones((), dtype=torch.float64, device=device)
    return torch.stack(
        (
            torch.stack((c, z, -s)),
            torch.stack((z, o, z)),
            torch.stack((s, z, c)),
        )
    )


def _fk4_precession_matrix(
    from_equinox: float, to_equinox: float, *, device: torch.device
) -> Tensor:
    t1 = (float(from_equinox) - 1850.0) / 1000.0
    dt = (float(to_equinox) - 1850.0) / 1000.0 - t1
    dt_over_3600 = dt / 3600.0

    z1 = zeta1 = (0.060 * t1 + 139.720) * t1 + 23035.545
    zeta2 = -0.27 * t1 + 30.240
    zeta = ((17.995 * dt + zeta2) * dt + zeta1) * dt_over_3600

    z2 = 109.480 + 0.39 * t1
    z = ((18.325 * dt + z2) * dt + z1) * dt_over_3600

    theta1 = (-0.37 * t1 - 85.29) * t1 + 20051.12
    theta2 = -0.37 * t1 - 42.65
    theta = ((-41.8 * dt + theta2) * dt + theta1) * dt_over_3600

    return (
        _rotation_matrix_z(-z, device=device)
        @ _rotation_matrix_y(theta, device=device)
        @ _rotation_matrix_z(-zeta, device=device)
    )


def _mean_obliquity_iau1980(jd: float) -> float:
    t = (float(jd) - 2451545.0) / 36525.0
    eps_arcsec = 84381.448 - 46.8150 * t - 0.00059 * t * t + 0.001813 * t * t * t
    return eps_arcsec * ARCSEC_TO_RAD


def _fk4_e_terms(equinox: float, *, device: torch.device) -> Tensor:
    k = 0.0056932 * D2R
    jd = _besselian_epoch_to_jd(equinox)
    t = (jd - JD_B1950) / 36525.0
    ek = k * ((-0.000000126 * t - 0.00004193) * t + 0.01673011)
    g = ((((0.012 * t + 1.65) * t + 6190.67) * t + 1015489.951) / 3600.0) * D2R
    eps = _mean_obliquity_iau1980(jd)
    minus_ek_cos_g = -ek * math.cos(g)
    return torch.tensor(
        [
            ek * math.sin(g),
            minus_ek_cos_g * math.cos(eps),
            minus_ek_cos_g * math.sin(eps),
        ],
        dtype=torch.float64,
        device=device,
    )


def _remove_fk4_e_terms(vec: Tensor, equinox: float) -> Tensor:
    eterms = _fk4_e_terms(equinox, device=vec.device).to(dtype=vec.dtype)
    return vec - eterms + torch.sum(eterms * vec, dim=-1, keepdim=True) * vec


def _add_fk4_e_terms(vec: Tensor, equinox: float) -> Tensor:
    eterms = _fk4_e_terms(equinox, device=vec.device).to(dtype=vec.dtype)
    rep = eterms + vec
    for _ in range(10):
        rep = (eterms + vec) / (1.0 + torch.sum(eterms * rep, dim=-1, keepdim=True))
    return rep / torch.linalg.norm(rep, dim=-1, keepdim=True).clamp_min(1e-30)


def _fk4_no_e_to_fk5_vec(
    vec: Tensor,
    *,
    from_equinox: float,
    to_equinox: float,
    obstime: float | None = None,
) -> Tensor:
    obstime = from_equinox if obstime is None else float(obstime)
    device = vec.device
    p1 = _fk4_precession_matrix(from_equinox, 1950.0, device=device)
    jyear = _besselian_epoch_to_julian_epoch(obstime)
    t = (jyear - 1950.0) / 100.0
    b = _B1950_TO_J2000_M.to(device=device, dtype=vec.dtype) + _FK4_CORR.to(
        device=device, dtype=vec.dtype
    ) * t
    out = torch.matmul(vec, p1.T.to(dtype=vec.dtype))
    out = torch.matmul(out, b.T)
    if float(to_equinox) != 2000.0:
        ra, dec = unit_to_radec(out)
        ra, dec = precess_fk5(ra, dec, from_equinox=2000.0, to_equinox=to_equinox)
        out = radec_to_unit(ra, dec)
    return out


def _fk5_to_fk4_no_e_vec(
    vec: Tensor,
    *,
    from_equinox: float,
    to_equinox: float,
    obstime: float | None = None,
) -> Tensor:
    obstime = to_equinox if obstime is None else float(obstime)
    device = vec.device
    if float(from_equinox) != 2000.0:
        ra, dec = unit_to_radec(vec)
        ra, dec = precess_fk5(ra, dec, from_equinox=from_equinox, to_equinox=2000.0)
        vec = radec_to_unit(ra, dec)
    jyear = _besselian_epoch_to_julian_epoch(obstime)
    t = (jyear - 1950.0) / 100.0
    b = _B1950_TO_J2000_M.to(device=device, dtype=vec.dtype) + _FK4_CORR.to(
        device=device, dtype=vec.dtype
    ) * t
    p2 = _fk4_precession_matrix(1950.0, to_equinox, device=device)
    out = torch.matmul(vec, b.to(dtype=vec.dtype))
    out = torch.matmul(out, p2.T.to(dtype=vec.dtype))
    return out


def radec_to_unit(lon_deg: Any, lat_deg: Any) -> Tensor:
    """Convert longitude/latitude in degrees to unit Cartesian vectors."""
    lon = _as_tensor(lon_deg)
    lat = _as_tensor(lat_deg).to(device=lon.device)
    lon_r = lon * D2R
    lat_r = lat * D2R
    cos_lat = torch.cos(lat_r)
    return torch.stack(
        (cos_lat * torch.cos(lon_r), cos_lat * torch.sin(lon_r), torch.sin(lat_r)),
        dim=-1,
    )


def unit_to_radec(vec: Any) -> tuple[Tensor, Tensor]:
    """Convert unit-like Cartesian vectors to longitude/latitude in degrees."""
    v = _as_tensor(vec)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    rxy = torch.hypot(x, y)
    lon = _wrap360(torch.atan2(y, x) * R2D)
    lat = torch.atan2(z, rxy) * R2D
    return lon, lat


def precess_fk5(
    ra_deg: Any,
    dec_deg: Any,
    *,
    from_equinox: float = 2000.0,
    to_equinox: float = 2000.0,
) -> tuple[Tensor, Tensor]:
    """Precess FK5 equatorial coordinates between Julian equinoxes.

    Uses the IAU 1976 precession angles for FK5-style mean places. This is a
    compact torch-native parity target for common WCS `RADESYS=FK5` / `EQUINOX`
    headers; higher-precision nutation and full FK4 E-term behavior are separate
    roadmap items.
    """
    ra = _as_tensor(ra_deg)
    dec = _as_tensor(dec_deg).to(device=ra.device)
    if float(from_equinox) == float(to_equinox):
        return ra.clone(), dec.clone()

    t0 = (float(from_equinox) - 2000.0) / 100.0
    t = (float(to_equinox) - float(from_equinox)) / 100.0
    zeta = (
        (2306.2181 + 1.39656 * t0 - 0.000139 * t0 * t0) * t
        + (0.30188 - 0.000344 * t0) * t * t
        + 0.017998 * t * t * t
    ) * ARCSEC_TO_RAD
    z = (
        (2306.2181 + 1.39656 * t0 - 0.000139 * t0 * t0) * t
        + (1.09468 + 0.000066 * t0) * t * t
        + 0.018203 * t * t * t
    ) * ARCSEC_TO_RAD
    theta = (
        (2004.3109 - 0.85330 * t0 - 0.000217 * t0 * t0) * t
        - (0.42665 + 0.000217 * t0) * t * t
        - 0.041833 * t * t * t
    ) * ARCSEC_TO_RAD

    ra1 = ra * D2R
    dec1 = dec * D2R
    sin_ra_zeta = torch.sin(ra1 + zeta)
    cos_ra_zeta = torch.cos(ra1 + zeta)
    cos_dec = torch.cos(dec1)
    sin_dec = torch.sin(dec1)

    theta_t = torch.as_tensor(theta, dtype=torch.float64, device=ra.device)
    cos_theta = torch.cos(theta_t)
    sin_theta = torch.sin(theta_t)

    a = cos_dec * sin_ra_zeta
    b = cos_theta * cos_dec * cos_ra_zeta - sin_theta * sin_dec
    c = sin_theta * cos_dec * cos_ra_zeta + cos_theta * sin_dec

    ra2 = _wrap360((torch.atan2(a, b) + z) * R2D)
    dec2 = torch.asin(torch.clamp(c, -1.0, 1.0)) * R2D
    return ra2, dec2


def icrs_to_galactic(ra_deg: Any, dec_deg: Any) -> tuple[Tensor, Tensor]:
    """Convert ICRS equatorial coordinates to Galactic lon/lat."""
    vec = radec_to_unit(ra_deg, dec_deg)
    mat = _ICRS_TO_GAL.to(device=vec.device, dtype=vec.dtype)
    gal_vec = torch.matmul(vec, mat.T)
    return unit_to_radec(gal_vec)


def galactic_to_icrs(l_deg: Any, b_deg: Any) -> tuple[Tensor, Tensor]:
    """Convert Galactic lon/lat to ICRS equatorial coordinates."""
    vec = radec_to_unit(l_deg, b_deg)
    mat = _ICRS_TO_GAL.to(device=vec.device, dtype=vec.dtype)
    icrs_vec = torch.matmul(vec, mat)
    return unit_to_radec(icrs_vec)


def canonical_celestial_frame(frame: str | None) -> str:
    """Normalize common FITS/AST celestial frame names."""
    if frame is None or str(frame).strip() == "":
        return "ICRS"
    name = str(frame).strip().upper()
    if name in {"RADECSYS", "EQUATORIAL"}:
        return "ICRS"
    if name in {"ICRS", "FK5", "FK4", "GALACTIC", "GAL"}:
        return "GALACTIC" if name == "GAL" else name
    raise ValueError(f"unsupported celestial frame {frame!r}")


def convert_celestial(
    lon_deg: Any,
    lat_deg: Any,
    *,
    from_frame: str = "ICRS",
    to_frame: str = "ICRS",
    from_equinox: float | None = None,
    to_equinox: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Convert between the currently supported celestial frames."""
    src = canonical_celestial_frame(from_frame)
    dst = canonical_celestial_frame(to_frame)
    lon = _as_tensor(lon_deg)
    lat = _as_tensor(lat_deg).to(device=lon.device)

    if src == "FK4":
        source_equinox = 1950.0 if from_equinox is None else float(from_equinox)
        vec = radec_to_unit(lon, lat)
        vec = _remove_fk4_e_terms(vec, source_equinox)
        vec = _fk4_no_e_to_fk5_vec(
            vec,
            from_equinox=source_equinox,
            to_equinox=2000.0,
            obstime=source_equinox,
        )
        lon, lat = unit_to_radec(vec)
        src = "ICRS"
    elif src == "GALACTIC":
        lon, lat = galactic_to_icrs(lon, lat)
        src = "ICRS"
    elif src == "FK5":
        lon, lat = precess_fk5(
            lon,
            lat,
            from_equinox=2000.0 if from_equinox is None else float(from_equinox),
            to_equinox=2000.0,
        )
        src = "ICRS"

    if dst == "ICRS":
        return lon, lat
    if dst == "FK4":
        target_equinox = 1950.0 if to_equinox is None else float(to_equinox)
        vec = radec_to_unit(lon, lat)
        vec = _fk5_to_fk4_no_e_vec(
            vec,
            from_equinox=2000.0,
            to_equinox=target_equinox,
            obstime=target_equinox,
        )
        vec = _add_fk4_e_terms(vec, target_equinox)
        return unit_to_radec(vec)
    if dst == "GALACTIC":
        return icrs_to_galactic(lon, lat)
    if dst == "FK5":
        return precess_fk5(
            lon,
            lat,
            from_equinox=2000.0,
            to_equinox=2000.0 if to_equinox is None else float(to_equinox),
        )
    raise ValueError(
        f"unsupported celestial frame conversion {from_frame!r}->{to_frame!r}"
    )
