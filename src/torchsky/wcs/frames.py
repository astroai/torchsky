from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .._dtype import as_work_tensor as _as_tensor
from .._dtype import work_dtype as _work_dtype

D2R = 0.017453292519943295
R2D = 57.29577951308232
ARCSEC_TO_RAD = D2R / 3600.0
MAS_TO_RAD = ARCSEC_TO_RAD / 1000.0
SECONDS_PER_JULIAN_YEAR = 365.25 * 86400.0
PARSEC_KM = 3.0856775814913674e13
KM_S_TO_PC_YR = SECONDS_PER_JULIAN_YEAR / PARSEC_KM
SPEED_OF_LIGHT_KM_S = 299792.458
SPEED_OF_LIGHT_PC_YR = SPEED_OF_LIGHT_KM_S * KM_S_TO_PC_YR
JD_B1900 = 2415020.31352
JD_B1950 = 2433282.42345905
TROPICAL_YEAR_DAYS = 365.242198781

# FK5 J2000 -> Galactic rotation from the IAU Galactic pole and node
# (Astropy's Galactic frame constants). ICRS inputs must first receive
# ``_ICRS_TO_FK5_J2000``.
_ICRS_TO_GAL = torch.tensor(
    [
        [-0.05487553939571455, -0.8734371047274331, -0.4838349917703224],
        [0.4941094536233634, -0.4448295942988166, 0.7469822487020499],
        [-0.8676661356858694, -0.1980763896109499, 0.4559837945175692],
    ],
    dtype=torch.float64,
)

# ICRS -> FK5 (J2000) frame-bias matrix, USNO Circular 179 / Astropy's
# _icrs_to_fk5_matrix(). The matrix maps column vectors; use ``vec @ B.T`` for
# ICRS -> FK5 and ``vec @ B`` for FK5 -> ICRS when vectors are rows.
_ICRS_TO_FK5_J2000 = torch.tensor(
    [
        [1.0, -1.11022333e-07, -4.41180450e-08],
        [1.11022337e-07, 1.0, 9.64779225e-08],
        [4.41180343e-08, -9.64779274e-08, 1.0],
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


def _to_device(value: Tensor, device: torch.device) -> Tensor:
    return value.to(device=device, dtype=_work_dtype(device))


def _wrap360(angle: Tensor) -> Tensor:
    return torch.remainder(angle, 360.0)


def _besselian_epoch_to_jd(epoch: float) -> float:
    return JD_B1900 + (float(epoch) - 1900.0) * TROPICAL_YEAR_DAYS


def _besselian_epoch_to_julian_epoch(epoch: float) -> float:
    return 2000.0 + (_besselian_epoch_to_jd(epoch) - 2451545.0) / 365.25


def _rotation_matrix_z(angle_deg: float, *, device: torch.device) -> Tensor:
    dtype = _work_dtype(device)
    a = torch.as_tensor(angle_deg * D2R, dtype=dtype, device=device)
    s = torch.sin(a)
    c = torch.cos(a)
    z = torch.zeros((), dtype=dtype, device=device)
    o = torch.ones((), dtype=dtype, device=device)
    return torch.stack(
        (
            torch.stack((c, s, z)),
            torch.stack((-s, c, z)),
            torch.stack((z, z, o)),
        )
    )


def _rotation_matrix_y(angle_deg: float, *, device: torch.device) -> Tensor:
    dtype = _work_dtype(device)
    a = torch.as_tensor(angle_deg * D2R, dtype=dtype, device=device)
    s = torch.sin(a)
    c = torch.cos(a)
    z = torch.zeros((), dtype=dtype, device=device)
    o = torch.ones((), dtype=dtype, device=device)
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
        dtype=_work_dtype(device),
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
    lat = _as_tensor(lat_deg).to(device=lon.device, dtype=_work_dtype(lon.device))
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
    dec = _as_tensor(dec_deg).to(device=ra.device, dtype=_work_dtype(ra.device))
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

    theta_t = torch.as_tensor(theta, dtype=_work_dtype(ra.device), device=ra.device)
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
    # Astropy/ERFA define the Galactic frame through FK5 J2000; apply the ICRS
    # frame bias before the FK5 -> Galactic rotation.
    bias = _ICRS_TO_FK5_J2000.to(device=vec.device, dtype=vec.dtype)
    mat = _ICRS_TO_GAL.to(device=vec.device, dtype=vec.dtype)
    gal_vec = torch.matmul(torch.matmul(vec, bias.T), mat.T)
    return unit_to_radec(gal_vec)


def galactic_to_icrs(l_deg: Any, b_deg: Any) -> tuple[Tensor, Tensor]:
    """Convert Galactic lon/lat to ICRS equatorial coordinates."""
    vec = radec_to_unit(l_deg, b_deg)
    bias = _ICRS_TO_FK5_J2000.to(device=vec.device, dtype=vec.dtype)
    mat = _ICRS_TO_GAL.to(device=vec.device, dtype=vec.dtype)
    icrs_vec = torch.matmul(torch.matmul(vec, mat), bias)
    return unit_to_radec(icrs_vec)


# IAU 2006 mean obliquity of the ecliptic at J2000, 84381.406 arcsec.
# ICRS -> mean ecliptic and equinox of J2000, from ERFA ``ecm06`` at
# JD 2451545.0 TT. This includes the frame bias and IAU 2006 precession, which
# the plain obliquity-only rotation misses by ~0.04 arcsec.
_ICRS_TO_ECLIPTIC_J2000 = torch.tensor(
    [
        [9.9999999999999412e-01, -7.0783689609715588e-08, 8.0562139776131861e-08],
        [3.2897004077419660e-08, 9.1748212991495837e-01, 3.9777699944404799e-01],
        [-1.0207044725484357e-07, -3.9777699944404304e-01, 9.1748212991495548e-01],
    ],
    dtype=torch.float64,
)


def _ecliptic_from_icrs_matrix(device: torch.device, dtype: torch.dtype) -> Tensor:
    return _ICRS_TO_ECLIPTIC_J2000.to(device=device, dtype=dtype)


def icrs_to_ecliptic(ra_deg: Any, dec_deg: Any) -> tuple[Tensor, Tensor]:
    """Rotate ICRS equatorial coordinates to mean ecliptic of J2000."""
    vec = radec_to_unit(ra_deg, dec_deg)
    mat = _ecliptic_from_icrs_matrix(vec.device, vec.dtype)
    return unit_to_radec(torch.matmul(vec, mat.T))


def ecliptic_to_icrs(lon_deg: Any, lat_deg: Any) -> tuple[Tensor, Tensor]:
    """Rotate mean ecliptic of J2000 coordinates to ICRS equatorial."""
    vec = radec_to_unit(lon_deg, lat_deg)
    mat = _ecliptic_from_icrs_matrix(vec.device, vec.dtype)
    return unit_to_radec(torch.matmul(vec, mat))


def icrs_to_ecliptic_of_date(
    ra_deg: Any, dec_deg: Any, tt_mjd: Any
) -> tuple[Tensor, Tensor]:
    """Rotate ICRS to the mean ecliptic of date via IAU 2006 NPB.

    ``tt_mjd`` is TT MJD. This is a geometric frame, not observed place:
    no aberration, refraction, or site.
    """
    vec = radec_to_unit(ra_deg, dec_deg)
    tt = _as_tensor(tt_mjd).to(device=vec.device, dtype=vec.dtype)
    if tt.numel() != 1:
        raise ValueError("ecliptic of date requires a scalar tt_mjd")
    # Circular: iau2006 imports D2R / ARCSEC_TO_RAD from this module.
    from .iau2006 import iau2006_fw_angles, npb_matrix

    n_matrix = npb_matrix(tt.reshape(()), nutation=False).to(
        device=vec.device, dtype=vec.dtype
    )
    _, _, _, epsa = iau2006_fw_angles(tt.reshape(()))
    v_date = torch.matmul(vec, n_matrix.T)
    return unit_to_radec(torch.matmul(v_date, _ecliptic_matrix_from_obliquity(epsa)))


def ecliptic_of_date_to_icrs(
    lon_deg: Any, lat_deg: Any, tt_mjd: Any
) -> tuple[Tensor, Tensor]:
    """Rotate mean ecliptic of date to ICRS via IAU 2006 NPB."""
    vec = radec_to_unit(lon_deg, lat_deg)
    tt = _as_tensor(tt_mjd).to(device=vec.device, dtype=vec.dtype)
    if tt.numel() != 1:
        raise ValueError("ecliptic of date requires a scalar tt_mjd")
    # Circular: iau2006 imports D2R / ARCSEC_TO_RAD from this module.
    from .iau2006 import iau2006_fw_angles, npb_matrix

    n_matrix = npb_matrix(tt.reshape(()), nutation=False).to(
        device=vec.device, dtype=vec.dtype
    )
    _, _, _, epsa = iau2006_fw_angles(tt.reshape(()))
    v_date = torch.matmul(vec, _ecliptic_matrix_from_obliquity(epsa).T)
    return unit_to_radec(torch.matmul(v_date, n_matrix))


def _ecliptic_matrix_from_obliquity(eps: Tensor) -> Tensor:
    c = torch.cos(eps)
    s = torch.sin(eps)
    z = torch.zeros((), dtype=eps.dtype, device=eps.device)
    o = torch.ones((), dtype=eps.dtype, device=eps.device)
    return torch.stack(
        (
            torch.stack((o, z, z)),
            torch.stack((z, c, -s)),
            torch.stack((z, s, c)),
        )
    )


# Reid & Brunthaler 2020 / Astropy v6 Galactocentric defaults.
_GALCEN_RA_DEG = 266.4051
_GALCEN_DEC_DEG = -28.936175
_GALCEN_DISTANCE_KPC = 8.122
_Z_SUN_PC = 20.8
# Schönrich, Binney & Dehnen 2010 solar peculiar velocity (km/s).
_SOLAR_U_KMS = 11.1
_SOLAR_V_KMS = 12.24
_SOLAR_W_KMS = 7.25


def icrs_to_galactocentric(
    ra_deg: Any,
    dec_deg: Any,
    distance_kpc: Any,
    *,
    galcen_distance_kpc: float = _GALCEN_DISTANCE_KPC,
    galcen_ra_deg: float = _GALCEN_RA_DEG,
    galcen_dec_deg: float = _GALCEN_DEC_DEG,
    z_sun_pc: float = _Z_SUN_PC,
) -> Tensor:
    """Heliocentric ICRS spherical → Galactocentric Cartesian (kpc).

    Origin at the Galactic Centre (``galcen_ra_deg`` / ``galcen_dec_deg``).
    +x toward the anticenter (Sun at ``x ≈ -R0``), +y in rotation, +z toward
    the NGP. ``z_sun_pc`` tilts the midplane so the Sun sits at ``z = z_sun``.
    Solar peculiar velocity is not applied.
    """
    vec = radec_to_unit(ra_deg, dec_deg)
    dist = _as_tensor(distance_kpc).to(device=vec.device, dtype=vec.dtype)
    r_helio = dist.unsqueeze(-1) * vec
    gc_hat = radec_to_unit(
        vec.new_tensor(float(galcen_ra_deg)),
        vec.new_tensor(float(galcen_dec_deg)),
    )
    r0 = dist.new_tensor(float(galcen_distance_kpc))
    gal = torch.matmul(
        r_helio - r0 * gc_hat,
        _ICRS_TO_GAL.to(device=vec.device, dtype=vec.dtype).T,
    )
    z_sun = dist.new_tensor(float(z_sun_pc) * 1.0e-3)
    theta = torch.atan2(z_sun, r0)
    c = torch.cos(theta)
    s = torch.sin(theta)
    x = gal[..., 0] * c + gal[..., 2] * s
    z = -gal[..., 0] * s + gal[..., 2] * c
    return torch.stack((x, gal[..., 1], z), dim=-1)


def galactocentric_to_icrs(
    xyz_kpc: Any,
    *,
    galcen_distance_kpc: float = _GALCEN_DISTANCE_KPC,
    galcen_ra_deg: float = _GALCEN_RA_DEG,
    galcen_dec_deg: float = _GALCEN_DEC_DEG,
    z_sun_pc: float = _Z_SUN_PC,
) -> tuple[Tensor, Tensor, Tensor]:
    """Galactocentric Cartesian (kpc) → ICRS spherical plus distance (kpc)."""
    xyz = _as_tensor(xyz_kpc)
    r0 = xyz.new_tensor(float(galcen_distance_kpc))
    z_sun = xyz.new_tensor(float(z_sun_pc) * 1.0e-3)
    theta = torch.atan2(z_sun, r0)
    c = torch.cos(theta)
    s = torch.sin(theta)
    x = xyz[..., 0] * c - xyz[..., 2] * s
    z = xyz[..., 0] * s + xyz[..., 2] * c
    gal = torch.stack((x, xyz[..., 1], z), dim=-1)
    gc_hat = radec_to_unit(
        xyz.new_tensor(float(galcen_ra_deg)),
        xyz.new_tensor(float(galcen_dec_deg)),
    )
    r_helio = torch.matmul(
        gal, _ICRS_TO_GAL.to(device=xyz.device, dtype=xyz.dtype)
    ) + r0 * gc_hat
    dist = torch.linalg.vector_norm(r_helio, dim=-1).clamp_min(1.0e-30)
    ra, dec = unit_to_radec(r_helio / dist.unsqueeze(-1))
    return ra, dec, dist


def lsr_line_of_sight_kms(
    ra_deg: Any,
    dec_deg: Any,
    *,
    solar_u_kms: float = _SOLAR_U_KMS,
    solar_v_kms: float = _SOLAR_V_KMS,
    solar_w_kms: float = _SOLAR_W_KMS,
) -> Tensor:
    """Solar peculiar velocity projected onto the ICRS line of sight (km/s).

    UVW are Galactic: U toward the GC, V in rotation, W toward the NGP.
    """
    lon, lat = icrs_to_galactic(ra_deg, dec_deg)
    lon_r = lon * D2R
    lat_r = lat * D2R
    cos_b = torch.cos(lat_r)
    u = lon.new_tensor(float(solar_u_kms))
    v = lon.new_tensor(float(solar_v_kms))
    w = lon.new_tensor(float(solar_w_kms))
    return (
        u * torch.cos(lon_r) * cos_b
        + v * torch.sin(lon_r) * cos_b
        + w * torch.sin(lat_r)
    )


def heliocentric_rv_to_lsr(
    ra_deg: Any,
    dec_deg: Any,
    rv_kms: Any,
    *,
    solar_u_kms: float = _SOLAR_U_KMS,
    solar_v_kms: float = _SOLAR_V_KMS,
    solar_w_kms: float = _SOLAR_W_KMS,
) -> Tensor:
    """Heliocentric radial velocity → LSR (km/s) with explicit solar UVW."""
    rv = _as_tensor(rv_kms)
    return rv + lsr_line_of_sight_kms(
        ra_deg,
        dec_deg,
        solar_u_kms=solar_u_kms,
        solar_v_kms=solar_v_kms,
        solar_w_kms=solar_w_kms,
    )


def propagate_proper_motion(
    lon_deg: Any,
    lat_deg: Any,
    pm_lon_coslat_mas_yr: Any,
    pm_lat_mas_yr: Any,
    *,
    source_epoch_jyear: Any,
    target_epoch_jyear: Any,
) -> tuple[Tensor, Tensor]:
    """Propagate angular coordinates with proper motion.

    ``pm_lon_coslat_mas_yr`` is the conventional longitude proper motion with
    the cosine-of-latitude factor already applied (for ICRS, ``pm_ra_cosdec``).
    Epochs are Julian years. Inputs broadcast and outputs are longitude and
    latitude in degrees, with longitude wrapped to ``[0, 360)``.

    This follows the same unit-distance Cartesian propagation used by
    ``SkyCoord.apply_space_motion`` when distance and radial velocity are not
    supplied. It stays well behaved at the poles, unlike dividing a longitude
    increment by ``cos(latitude)``.
    """
    lon = _as_tensor(lon_deg)
    device = lon.device
    lat = _as_tensor(lat_deg).to(device=device, dtype=_work_dtype(device))
    pm_lon = _as_tensor(pm_lon_coslat_mas_yr).to(device=device, dtype=_work_dtype(device))
    pm_lat = _as_tensor(pm_lat_mas_yr).to(device=device, dtype=_work_dtype(device))
    source_epoch = _as_tensor(source_epoch_jyear).to(device=device, dtype=_work_dtype(device))
    target_epoch = _as_tensor(target_epoch_jyear).to(device=device, dtype=_work_dtype(device))
    lon, lat, pm_lon, pm_lat, source_epoch, target_epoch = torch.broadcast_tensors(
        lon, lat, pm_lon, pm_lat, source_epoch, target_epoch
    )

    lon_rad = lon * D2R
    lat_rad = lat * D2R
    sin_lon, cos_lon = torch.sin(lon_rad), torch.cos(lon_rad)
    sin_lat, cos_lat = torch.sin(lat_rad), torch.cos(lat_rad)
    position = torch.stack(
        (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat), dim=-1
    )
    tangent_lon = torch.stack((-sin_lon, cos_lon, torch.zeros_like(lon)), dim=-1)
    tangent_lat = torch.stack(
        (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat), dim=-1
    )

    velocity = MAS_TO_RAD * (
        pm_lon.unsqueeze(-1) * tangent_lon + pm_lat.unsqueeze(-1) * tangent_lat
    )
    propagated = position + (target_epoch - source_epoch).unsqueeze(-1) * velocity
    # ponytail: unit-distance linear motion is the exact no-distance Astropy
    # contract; add parallax/radial velocity here when 6D phase space is needed.
    propagated = propagated / torch.linalg.norm(
        propagated, dim=-1, keepdim=True
    ).clamp_min(1e-30)
    return unit_to_radec(propagated)


def _proper_motion_position_delta(
    state: Tensor,
    perturbation: Tensor,
    propagated_position: Tensor,
    source_epoch: Tensor,
    target_epoch: Tensor,
) -> Tensor:
    """Map a local angular-state perturbation to propagated position."""
    lon_rad = state[0] * D2R
    lat_rad = state[1] * D2R
    sin_lon, cos_lon = torch.sin(lon_rad), torch.cos(lon_rad)
    sin_lat, cos_lat = torch.sin(lat_rad), torch.cos(lat_rad)
    direction = torch.stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat))
    tangent_lon = torch.stack((-sin_lon, cos_lon, torch.zeros_like(lon_rad)))
    tangent_lat = torch.stack(
        (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)
    )
    perturbed_direction = direction + MAS_TO_RAD * (
        perturbation[0] * tangent_lon + perturbation[1] * tangent_lat
    )
    perturbed_direction = perturbed_direction / torch.linalg.norm(
        perturbed_direction
    ).clamp_min(1e-30)
    perturbed_lon, perturbed_lat = unit_to_radec(perturbed_direction)
    moved_lon, moved_lat = propagate_proper_motion(
        perturbed_lon,
        perturbed_lat,
        state[2] + perturbation[2],
        state[3] + perturbation[3],
        source_epoch_jyear=source_epoch,
        target_epoch_jyear=target_epoch,
    )
    moved_direction = radec_to_unit(moved_lon, moved_lat)

    output_lon_rad = propagated_position[0] * D2R
    output_lat_rad = propagated_position[1] * D2R
    output_sin_lon, output_cos_lon = (
        torch.sin(output_lon_rad),
        torch.cos(output_lon_rad),
    )
    output_sin_lat, output_cos_lat = (
        torch.sin(output_lat_rad),
        torch.cos(output_lat_rad),
    )
    output_direction = torch.stack(
        (
            output_cos_lat * output_cos_lon,
            output_cos_lat * output_sin_lon,
            output_sin_lat,
        )
    )
    output_tangent_lon = torch.stack(
        (-output_sin_lon, output_cos_lon, torch.zeros_like(output_lon_rad))
    )
    output_tangent_lat = torch.stack(
        (
            -output_sin_lat * output_cos_lon,
            -output_sin_lat * output_sin_lon,
            output_cos_lat,
        )
    )
    direction_delta = moved_direction - output_direction
    return torch.stack(
        (
            torch.dot(direction_delta, output_tangent_lon) / MAS_TO_RAD,
            torch.dot(direction_delta, output_tangent_lat) / MAS_TO_RAD,
        )
    )


def propagate_proper_motion_with_jacobian(
    lon_deg: Any,
    lat_deg: Any,
    pm_lon_coslat_mas_yr: Any,
    pm_lat_mas_yr: Any,
    *,
    source_epoch_jyear: Any,
    target_epoch_jyear: Any,
) -> tuple[tuple[Tensor, Tensor], Tensor]:
    """Propagate angular motion and return its batched local Jacobian.

    The Jacobian has trailing dimensions ``(2, 4)``. Its input order is
    ``(position_east, position_north, pm_lon_coslat, pm_lat)`` and its output
    rows are local east/north position at the propagated coordinate. Position
    units are mas and proper-motion units are mas/yr.

    This is the unit-distance angular-motion model used by
    :func:`propagate_proper_motion`; it does not invent a distance or radial
    velocity and therefore does not model perspective acceleration.
    """
    propagated = propagate_proper_motion(
        lon_deg,
        lat_deg,
        pm_lon_coslat_mas_yr,
        pm_lat_mas_yr,
        source_epoch_jyear=source_epoch_jyear,
        target_epoch_jyear=target_epoch_jyear,
    )
    lon = _as_tensor(lon_deg)
    device = lon.device
    values = torch.broadcast_tensors(
        lon,
        _as_tensor(lat_deg).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(pm_lon_coslat_mas_yr).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(pm_lat_mas_yr).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(source_epoch_jyear).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(target_epoch_jyear).to(device=device, dtype=_work_dtype(device)),
    )
    state = torch.stack(values[:4], dim=-1)
    propagated_position = torch.stack(propagated, dim=-1)
    batch_shape = state.shape[:-1]
    flat_state = state.reshape(-1, 4)
    flat_propagated = propagated_position.reshape(-1, 2)
    perturbation = torch.zeros_like(flat_state)
    jacobian = torch.vmap(
        torch.func.jacrev(_proper_motion_position_delta, argnums=1),
        in_dims=(0, 0, 0, 0, 0),
    )(
        flat_state,
        perturbation,
        flat_propagated,
        values[4].reshape(-1),
        values[5].reshape(-1),
    )
    return propagated, jacobian.reshape(batch_shape + (2, 4))


def _propagate_space_motion(
    lon_deg: Any,
    lat_deg: Any,
    pm_lon_coslat_mas_yr: Any,
    pm_lat_mas_yr: Any,
    parallax_mas: Any,
    radial_velocity_km_s: Any,
    *,
    source_epoch_jyear: Any,
    target_epoch_jyear: Any,
    _validate: bool = True,
    _correction_iterations: int = 16,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Propagate a complete barycentric astrometric state in ICRS.

    Inputs and outputs are longitude/latitude in degrees, cosine-weighted
    longitude and latitude proper motions in mas/yr, parallax in mas, and
    radial velocity in km/s. Epochs are TCB Julian years. Inputs broadcast.

    The implementation follows the ERFA/SOFA ``starpv`` → rectilinear motion
    with changing light time → ``pvstar`` model, including the special-
    relativistic conversion between catalog observables and inertial velocity.
    A complete state is required: use :func:`propagate_proper_motion` when
    distance or radial velocity is unavailable.

    Returns:
        ``(lon, lat, pm_lon_coslat, pm_lat, parallax, radial_velocity)``.
    """
    lon = _as_tensor(lon_deg)
    device = lon.device
    values = [
        lon,
        _as_tensor(lat_deg).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(pm_lon_coslat_mas_yr).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(pm_lat_mas_yr).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(parallax_mas).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(radial_velocity_km_s).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(source_epoch_jyear).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(target_epoch_jyear).to(device=device, dtype=_work_dtype(device)),
    ]
    (
        lon,
        lat,
        pm_lon,
        pm_lat,
        parallax,
        radial_velocity,
        source_epoch,
        target_epoch,
    ) = torch.broadcast_tensors(*values)
    finite = torch.ones_like(lon, dtype=torch.bool)
    for value in (
        lon,
        lat,
        pm_lon,
        pm_lat,
        parallax,
        radial_velocity,
        source_epoch,
        target_epoch,
    ):
        finite &= torch.isfinite(value)
    if _validate and not bool(torch.all(finite)):
        raise ValueError("space-motion inputs must be finite")
    if _validate and not bool(torch.all((lat >= -90.0) & (lat <= 90.0))):
        raise ValueError("latitude must be within [-90, 90] degrees")
    if _validate and not bool(torch.all(parallax > 0.0)):
        raise ValueError("parallax_mas must be positive; use angular propagation otherwise")

    lon_rad = lon * D2R
    lat_rad = lat * D2R
    sin_lon, cos_lon = torch.sin(lon_rad), torch.cos(lon_rad)
    sin_lat, cos_lat = torch.sin(lat_rad), torch.cos(lat_rad)
    direction = torch.stack(
        (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat), dim=-1
    )
    tangent_lon = torch.stack((-sin_lon, cos_lon, torch.zeros_like(lon)), dim=-1)
    tangent_lat = torch.stack(
        (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat), dim=-1
    )

    distance_pc = 1000.0 / parallax
    position = distance_pc.unsqueeze(-1) * direction
    observed_tangent = distance_pc.unsqueeze(-1) * MAS_TO_RAD * (
        pm_lon.unsqueeze(-1) * tangent_lon + pm_lat.unsqueeze(-1) * tangent_lat
    )
    observed_radial = radial_velocity * KM_S_TO_PC_YR
    beta_radial = observed_radial / SPEED_OF_LIGHT_PC_YR
    beta_tangent = (
        torch.linalg.norm(observed_tangent, dim=-1) / SPEED_OF_LIGHT_PC_YR
    )
    if _validate and not bool(
        torch.all(beta_radial.square() + beta_tangent.square() < 0.25)
    ):
        raise ValueError("catalog state implies a space velocity of at least 0.5c")

    # Invert the observed-to-inertial catalog correction. ERFA stops once the
    # iteration reaches floating-point convergence; a fixed count keeps this
    # tensor path differentiable and converges to the same float64 solution.
    corrected_radial = beta_radial
    corrected_tangent = beta_tangent
    correction_scale = torch.ones_like(beta_radial)
    correction_delta = torch.zeros_like(beta_radial)
    for _ in range(_correction_iterations):
        correction_scale = 1.0 + corrected_radial
        beta_squared = corrected_radial.square() + corrected_tangent.square()
        correction_delta = -beta_squared / (
            torch.sqrt(torch.clamp_min(1.0 - beta_squared, 0.0)) + 1.0
        )
        corrected_radial = correction_scale * beta_radial + correction_delta
        corrected_tangent = correction_scale * beta_tangent

    if _validate:
        next_scale = 1.0 + corrected_radial
        next_beta_squared = corrected_radial.square() + corrected_tangent.square()
        next_delta = -next_beta_squared / (
            torch.sqrt(torch.clamp_min(1.0 - next_beta_squared, 0.0)) + 1.0
        )
        converged = torch.isclose(
            next_scale, correction_scale, rtol=1e-14, atol=1e-15
        ) & torch.isclose(next_delta, correction_delta, rtol=1e-14, atol=1e-15)
        if not bool(torch.all(converged)) and _correction_iterations < 100:
            # ponytail: real stellar catalogues converge in the fast loop;
            # retry SOFA's 100-iteration ceiling only for rare extreme states.
            return _propagate_space_motion(
                lon,
                lat,
                pm_lon,
                pm_lat,
                parallax,
                radial_velocity,
                source_epoch_jyear=source_epoch,
                target_epoch_jyear=target_epoch,
                _validate=True,
                _correction_iterations=100,
            )
        if not bool(torch.all(converged)):
            raise ValueError("relativistic space-motion correction did not converge")

    inertial_velocity = (
        correction_scale.unsqueeze(-1) * observed_tangent
        + (
            SPEED_OF_LIGHT_PC_YR
            * (correction_scale * beta_radial + correction_delta)
        ).unsqueeze(-1)
        * direction
    )

    light_time_before = distance_pc / SPEED_OF_LIGHT_PC_YR
    dt_years = target_epoch - source_epoch
    geometric_position = position + (
        dt_years + light_time_before
    ).unsqueeze(-1) * inertial_velocity
    radius_squared = (geometric_position * geometric_position).sum(dim=-1)
    radial_dot = (geometric_position * inertial_velocity).sum(dim=-1)
    speed_squared = (inertial_velocity * inertial_velocity).sum(dim=-1)
    c2_minus_v2 = SPEED_OF_LIGHT_PC_YR**2 - speed_squared
    if _validate and not bool(torch.all(c2_minus_v2 > 0.0)):
        raise ValueError("inertial space velocity must remain subluminal")
    light_time_after = (
        -radial_dot
        + torch.sqrt(
            torch.clamp_min(
                radial_dot.square() + c2_minus_v2 * radius_squared, 0.0
            )
        )
    ) / c2_minus_v2
    propagated_position = position + (
        dt_years + light_time_before - light_time_after
    ).unsqueeze(-1) * inertial_velocity

    distance_out = torch.linalg.norm(propagated_position, dim=-1)
    if _validate and not bool(torch.all(distance_out > 0.0)):
        raise ValueError("space-motion trajectory crosses the observer")
    direction_out = propagated_position / distance_out.unsqueeze(-1)
    inertial_radial = (direction_out * inertial_velocity).sum(dim=-1)
    inertial_tangent = inertial_velocity - inertial_radial.unsqueeze(-1) * direction_out
    beta_radial_out = inertial_radial / SPEED_OF_LIGHT_PC_YR
    beta_tangent_out = (
        torch.linalg.norm(inertial_tangent, dim=-1) / SPEED_OF_LIGHT_PC_YR
    )
    observed_scale = 1.0 + beta_radial_out
    beta_squared_out = beta_radial_out.square() + beta_tangent_out.square()
    if _validate and not bool(
        torch.all((observed_scale != 0.0) & (beta_squared_out < 1.0))
    ):
        raise ValueError("propagated state cannot be converted to catalog observables")
    observed_delta = -beta_squared_out / (
        torch.sqrt(torch.clamp_min(1.0 - beta_squared_out, 0.0)) + 1.0
    )
    observed_velocity = inertial_tangent / observed_scale.unsqueeze(-1) + (
        SPEED_OF_LIGHT_PC_YR
        * (beta_radial_out - observed_delta)
        / observed_scale
    ).unsqueeze(-1) * direction_out

    lon_out, lat_out = unit_to_radec(direction_out)
    lon_out_rad = lon_out * D2R
    lat_out_rad = lat_out * D2R
    sin_lon_out, cos_lon_out = torch.sin(lon_out_rad), torch.cos(lon_out_rad)
    sin_lat_out, cos_lat_out = torch.sin(lat_out_rad), torch.cos(lat_out_rad)
    tangent_lon_out = torch.stack(
        (-sin_lon_out, cos_lon_out, torch.zeros_like(lon_out)), dim=-1
    )
    tangent_lat_out = torch.stack(
        (
            -sin_lat_out * cos_lon_out,
            -sin_lat_out * sin_lon_out,
            cos_lat_out,
        ),
        dim=-1,
    )
    pm_lon_out = (observed_velocity * tangent_lon_out).sum(dim=-1) / (
        distance_out * MAS_TO_RAD
    )
    pm_lat_out = (observed_velocity * tangent_lat_out).sum(dim=-1) / (
        distance_out * MAS_TO_RAD
    )
    parallax_out = 1000.0 / distance_out
    radial_velocity_out = (
        (observed_velocity * direction_out).sum(dim=-1) / KM_S_TO_PC_YR
    )
    return (
        lon_out,
        lat_out,
        pm_lon_out,
        pm_lat_out,
        parallax_out,
        radial_velocity_out,
    )


def propagate_space_motion(
    lon_deg: Any,
    lat_deg: Any,
    pm_lon_coslat_mas_yr: Any,
    pm_lat_mas_yr: Any,
    parallax_mas: Any,
    radial_velocity_km_s: Any,
    *,
    source_epoch_jyear: Any,
    target_epoch_jyear: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Propagate a complete barycentric astrometric state in ICRS.

    Inputs and outputs are longitude/latitude in degrees, cosine-weighted
    longitude and latitude proper motions in mas/yr, parallax in mas, and
    radial velocity in km/s. Epochs are TCB Julian years. Inputs broadcast.

    The implementation follows the ERFA/SOFA ``starpv`` to rectilinear motion
    with changing light time to ``pvstar`` model, including the special-
    relativistic conversion between catalogue observables and inertial
    velocity. A complete state is required: use
    :func:`propagate_proper_motion` when distance or radial velocity is
    unavailable.

    Returns:
        ``(lon, lat, pm_lon_coslat, pm_lat, parallax, radial_velocity)``.
    """
    return _propagate_space_motion(
        lon_deg,
        lat_deg,
        pm_lon_coslat_mas_yr,
        pm_lat_mas_yr,
        parallax_mas,
        radial_velocity_km_s,
        source_epoch_jyear=source_epoch_jyear,
        target_epoch_jyear=target_epoch_jyear,
    )


def _space_motion_local_delta(
    state: Tensor,
    perturbation: Tensor,
    propagated_state: Tensor,
    source_epoch: Tensor,
    target_epoch: Tensor,
) -> Tensor:
    """Map a local catalogue perturbation to propagated local observables."""
    lon_rad = state[0] * D2R
    lat_rad = state[1] * D2R
    sin_lon, cos_lon = torch.sin(lon_rad), torch.cos(lon_rad)
    sin_lat, cos_lat = torch.sin(lat_rad), torch.cos(lat_rad)
    direction = torch.stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat))
    tangent_lon = torch.stack((-sin_lon, cos_lon, torch.zeros_like(lon_rad)))
    tangent_lat = torch.stack(
        (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)
    )
    perturbed_direction = direction + MAS_TO_RAD * (
        perturbation[0] * tangent_lon + perturbation[1] * tangent_lat
    )
    perturbed_direction = perturbed_direction / torch.linalg.norm(
        perturbed_direction
    ).clamp_min(1e-30)
    perturbed_lon, perturbed_lat = unit_to_radec(perturbed_direction)

    moved = _propagate_space_motion(
        perturbed_lon,
        perturbed_lat,
        state[2] + perturbation[2],
        state[3] + perturbation[3],
        state[4] + perturbation[4],
        state[5] + perturbation[5],
        source_epoch_jyear=source_epoch,
        target_epoch_jyear=target_epoch,
        _validate=False,
    )
    moved_direction = radec_to_unit(moved[0], moved[1])

    output_lon_rad = propagated_state[0] * D2R
    output_lat_rad = propagated_state[1] * D2R
    output_sin_lon, output_cos_lon = (
        torch.sin(output_lon_rad),
        torch.cos(output_lon_rad),
    )
    output_sin_lat, output_cos_lat = (
        torch.sin(output_lat_rad),
        torch.cos(output_lat_rad),
    )
    output_direction = torch.stack(
        (
            output_cos_lat * output_cos_lon,
            output_cos_lat * output_sin_lon,
            output_sin_lat,
        )
    )
    output_tangent_lon = torch.stack(
        (-output_sin_lon, output_cos_lon, torch.zeros_like(output_lon_rad))
    )
    output_tangent_lat = torch.stack(
        (
            -output_sin_lat * output_cos_lon,
            -output_sin_lat * output_sin_lon,
            output_cos_lat,
        )
    )
    direction_delta = moved_direction - output_direction
    return torch.stack(
        (
            torch.dot(direction_delta, output_tangent_lon) / MAS_TO_RAD,
            torch.dot(direction_delta, output_tangent_lat) / MAS_TO_RAD,
            moved[2] - propagated_state[2],
            moved[3] - propagated_state[3],
            moved[4] - propagated_state[4],
            moved[5] - propagated_state[5],
        )
    )


def propagate_space_motion_with_jacobian(
    lon_deg: Any,
    lat_deg: Any,
    pm_lon_coslat_mas_yr: Any,
    pm_lat_mas_yr: Any,
    parallax_mas: Any,
    radial_velocity_km_s: Any,
    *,
    source_epoch_jyear: Any,
    target_epoch_jyear: Any,
) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]:
    """Propagate space motion and return its batched local 6D Jacobian.

    The Jacobian's trailing dimensions are ``(6, 6)``. Its input and output
    order is ``(position_east, position_north, pm_lon_coslat, pm_lat,
    parallax, radial_velocity)``. Position perturbations are in mas, proper
    motions in mas/yr, parallax in mas, and radial velocity in km/s.

    The positional rows use local tangent bases at the source and propagated
    positions. This stays continuous across longitude wrap and avoids the
    singular raw-longitude derivative near the poles. A covariance in these
    local catalogue units can therefore be propagated with ``J @ C @ J.mT``.
    Inputs broadcast and are validated by :func:`propagate_space_motion`.

    Returns:
        ``(propagated_state, jacobian)``.
    """
    propagated = propagate_space_motion(
        lon_deg,
        lat_deg,
        pm_lon_coslat_mas_yr,
        pm_lat_mas_yr,
        parallax_mas,
        radial_velocity_km_s,
        source_epoch_jyear=source_epoch_jyear,
        target_epoch_jyear=target_epoch_jyear,
    )
    lon = _as_tensor(lon_deg)
    device = lon.device
    state_values = torch.broadcast_tensors(
        lon,
        _as_tensor(lat_deg).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(pm_lon_coslat_mas_yr).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(pm_lat_mas_yr).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(parallax_mas).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(radial_velocity_km_s).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(source_epoch_jyear).to(device=device, dtype=_work_dtype(device)),
        _as_tensor(target_epoch_jyear).to(device=device, dtype=_work_dtype(device)),
    )
    state = torch.stack(state_values[:6], dim=-1)
    propagated_state = torch.stack(propagated, dim=-1)
    batch_shape = state.shape[:-1]
    flat_state = state.reshape(-1, 6)
    flat_propagated = propagated_state.reshape(-1, 6)
    flat_source_epoch = state_values[6].reshape(-1)
    flat_target_epoch = state_values[7].reshape(-1)
    perturbation = torch.zeros_like(flat_state)
    jacobian = torch.vmap(
        torch.func.jacrev(_space_motion_local_delta, argnums=1),
        in_dims=(0, 0, 0, 0, 0),
    )(
        flat_state,
        perturbation,
        flat_propagated,
        flat_source_epoch,
        flat_target_epoch,
    )
    return propagated, jacobian.reshape(batch_shape + (6, 6))


def _icrs_to_fk5_radec(ra_deg: Any, dec_deg: Any) -> tuple[Tensor, Tensor]:
    """Apply the ICRS -> FK5 J2000 frame bias to equatorial coordinates."""
    vec = radec_to_unit(ra_deg, dec_deg)
    bias = _ICRS_TO_FK5_J2000.to(device=vec.device, dtype=vec.dtype)
    return unit_to_radec(torch.matmul(vec, bias.T))


def _fk5_to_icrs_radec(ra_deg: Any, dec_deg: Any) -> tuple[Tensor, Tensor]:
    """Apply the FK5 J2000 -> ICRS frame bias (inverse of the above)."""
    vec = radec_to_unit(ra_deg, dec_deg)
    bias = _ICRS_TO_FK5_J2000.to(device=vec.device, dtype=vec.dtype)
    return unit_to_radec(torch.matmul(vec, bias))


def canonical_celestial_frame(frame: str | None) -> str:
    """Normalize common FITS/AST celestial frame names."""
    if frame is None or str(frame).strip() == "":
        return "ICRS"
    name = str(frame).strip().upper()
    if name in {"RADECSYS", "EQUATORIAL"}:
        return "ICRS"
    if name in {
        "ICRS",
        "FK5",
        "FK4",
        "GALACTIC",
        "GAL",
        "ECLIPTIC",
        "ECL",
        "ECLIPTIC_OF_DATE",
        "GALACTOCENTRIC",
        "LSR",
        "TEME",
    }:
        if name == "GAL":
            return "GALACTIC"
        if name == "ECL":
            return "ECLIPTIC"
        return name
    raise ValueError(f"unsupported celestial frame {frame!r}")


def convert_celestial(
    lon_deg: Any,
    lat_deg: Any,
    *,
    from_frame: str = "ICRS",
    to_frame: str = "ICRS",
    from_equinox: float | None = None,
    to_equinox: float | None = None,
    tt_mjd: Any | None = None,
) -> tuple[Tensor, Tensor]:
    """Convert between the currently supported celestial frames."""
    src = canonical_celestial_frame(from_frame)
    dst = canonical_celestial_frame(to_frame)
    lon = _as_tensor(lon_deg)
    lat = _as_tensor(lat_deg).to(device=lon.device, dtype=_work_dtype(lon.device))
    if src == "LSR":
        src = "ICRS"
    if src == "TEME" and dst != "TEME":
        raise ValueError("TEME is a stored TLE frame; convert it explicitly")
    if src == "GALACTOCENTRIC":
        raise ValueError(
            "Galactocentric is Cartesian; use galactocentric_to_icrs with xyz_kpc"
        )

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
        lon, lat = _fk5_to_icrs_radec(*unit_to_radec(vec))
        src = "ICRS"
    elif src == "GALACTIC":
        lon, lat = galactic_to_icrs(lon, lat)
        src = "ICRS"
    elif src == "ECLIPTIC":
        lon, lat = ecliptic_to_icrs(lon, lat)
        src = "ICRS"
    elif src == "ECLIPTIC_OF_DATE":
        if tt_mjd is None:
            raise ValueError("ecliptic of date requires tt_mjd")
        lon, lat = ecliptic_of_date_to_icrs(lon, lat, tt_mjd)
        src = "ICRS"
    elif src == "FK5":
        lon, lat = precess_fk5(
            lon,
            lat,
            from_equinox=2000.0 if from_equinox is None else float(from_equinox),
            to_equinox=2000.0,
        )
        lon, lat = _fk5_to_icrs_radec(lon, lat)
        src = "ICRS"

    if dst == "ICRS":
        return lon, lat
    if dst == "LSR":
        return lon, lat
    if dst == "TEME":
        raise ValueError("TEME is a stored TLE frame; convert it explicitly")
    if dst == "GALACTOCENTRIC":
        raise ValueError(
            "Galactocentric is Cartesian; use icrs_to_galactocentric with distance_kpc"
        )
    if dst == "FK4":
        target_equinox = 1950.0 if to_equinox is None else float(to_equinox)
        lon, lat = _icrs_to_fk5_radec(lon, lat)
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
    if dst == "ECLIPTIC":
        return icrs_to_ecliptic(lon, lat)
    if dst == "ECLIPTIC_OF_DATE":
        if tt_mjd is None:
            raise ValueError("ecliptic of date requires tt_mjd")
        return icrs_to_ecliptic_of_date(lon, lat, tt_mjd)
    if dst == "FK5":
        lon, lat = _icrs_to_fk5_radec(lon, lat)
        return precess_fk5(
            lon,
            lat,
            from_equinox=2000.0,
            to_equinox=2000.0 if to_equinox is None else float(to_equinox),
        )
    raise ValueError(
        f"unsupported celestial frame conversion {from_frame!r}->{to_frame!r}"
    )
