from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

D2R = 0.017453292519943295
R2D = 57.29577951308232
ARCSEC_TO_RAD = D2R / 3600.0

_ICRS_TO_GAL = torch.tensor(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
    ],
    dtype=torch.float64,
)


def _as_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(dtype=torch.float64)
    return torch.as_tensor(value, dtype=torch.float64)


def _wrap360(angle: Tensor) -> Tensor:
    return torch.remainder(angle, 360.0)


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

    if src == "FK4" or dst == "FK4":
        raise NotImplementedError("FK4/Besselian E-term conversion is not implemented")

    if src == "GALACTIC":
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
    if dst == "GALACTIC":
        return icrs_to_galactic(lon, lat)
    if dst == "FK5":
        return precess_fk5(
            lon,
            lat,
            from_equinox=2000.0,
            to_equinox=2000.0 if to_equinox is None else float(to_equinox),
        )
    raise ValueError(f"unsupported celestial frame conversion {from_frame!r}->{to_frame!r}")
