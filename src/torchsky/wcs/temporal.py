from __future__ import annotations

import datetime
import math
from typing import Any

import torch
from torch import Tensor
from ..device import _to

SECONDS_PER_DAY = 86400.0
TT_MINUS_TAI_SECONDS = 32.184
TAI_MINUS_GPS_SECONDS = 19.0

# Effective MJD dates at 00:00:00 UTC and TAI-UTC offsets in seconds.
# IERS Bulletin C 72 / Leap_Second.dat give TAI-UTC = 37 s from 2017-01-01;
# that file expires on 2027-06-28 (no leap second at the end of 2026).
_LEAP_EFFECTIVE_MJD = torch.tensor(
    [
        41317.0,
        41499.0,
        41683.0,
        42048.0,
        42413.0,
        42778.0,
        43144.0,
        43509.0,
        43874.0,
        44239.0,
        44786.0,
        45151.0,
        45516.0,
        46247.0,
        47161.0,
        47892.0,
        48257.0,
        48804.0,
        49169.0,
        49534.0,
        50083.0,
        50630.0,
        51179.0,
        53736.0,
        54832.0,
        56109.0,
        57204.0,
        57754.0,
    ],
    dtype=torch.float64,
)
_TAI_MINUS_UTC = torch.tensor(
    [
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
        25.0,
        26.0,
        27.0,
        28.0,
        29.0,
        30.0,
        31.0,
        32.0,
        33.0,
        34.0,
        35.0,
        36.0,
        37.0,
    ],
    dtype=torch.float64,
)

_TIMEUNIT_TO_DAYS = {
    "d": 1.0,
    "day": 1.0,
    "days": 1.0,
    "s": 1.0 / SECONDS_PER_DAY,
    "sec": 1.0 / SECONDS_PER_DAY,
    "second": 1.0 / SECONDS_PER_DAY,
    "seconds": 1.0 / SECONDS_PER_DAY,
    "min": 60.0 / SECONDS_PER_DAY,
    "minute": 60.0 / SECONDS_PER_DAY,
    "minutes": 60.0 / SECONDS_PER_DAY,
    "h": 3600.0 / SECONDS_PER_DAY,
    "hr": 3600.0 / SECONDS_PER_DAY,
    "hour": 3600.0 / SECONDS_PER_DAY,
    "hours": 3600.0 / SECONDS_PER_DAY,
    "ms": 1.0e-3 / SECONDS_PER_DAY,
    "us": 1.0e-6 / SECONDS_PER_DAY,
    "µs": 1.0e-6 / SECONDS_PER_DAY,
    "ns": 1.0e-9 / SECONDS_PER_DAY,
    "a": 365.25,
    "yr": 365.25,
    "year": 365.25,
    "years": 365.25,
    "cy": 36525.0,
    "century": 36525.0,
    "centuries": 36525.0,
}

_TREFPOS_ALIASES = {
    "TOPO": "TOPOCENTER",
    "TOPOCENTER": "TOPOCENTER",
    "TOPOCENTRE": "TOPOCENTER",
    "GEO": "GEOCENTER",
    "GEOCENTER": "GEOCENTER",
    "GEOCENTRE": "GEOCENTER",
    "BARY": "BARYCENTER",
    "BARYCENTER": "BARYCENTER",
    "BARYCENTRE": "BARYCENTER",
    "SSB": "BARYCENTER",
    "SOLARSYSTEMBARYCENTER": "BARYCENTER",
    "HELIO": "HELIOCENTER",
    "HELIOCENTER": "HELIOCENTER",
    "HELIOCENTRE": "HELIOCENTER",
    "EMBARY": "EMBARYCENTER",
    "EMBARYCENTER": "EMBARYCENTER",
    "EARTHMOONBARYCENTER": "EMBARYCENTER",
    "MOON": "MOON",
    "CUSTOM": "CUSTOM",
    "BASE": "BASECENTER",
    "BASECENTER": "BASECENTER",
    "GALACTIC": "GALACTIC",
    "GALACTICENTER": "GALACTIC",
    "GALACTICCENTRE": "GALACTIC",
    "LOCALGROUP": "LOCALGROUP",
    "LOCALGROUPBARYCENTER": "LOCALGROUP",
    "MERCURY": "MERCURY",
    "VENUS": "VENUS",
    "MARS": "MARS",
    "JUPITER": "JUPITER",
    "SATURN": "SATURN",
    "URANUS": "URANUS",
    "NEPTUNE": "NEPTUNE",
}


def _as_float_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        dtype = torch.float32 if value.device.type == "mps" else torch.float64
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=torch.float64)


def tai_minus_utc_seconds(mjd_utc: Any) -> Tensor:
    """Return TAI-UTC in seconds for UTC MJD values covered by leap table."""
    mjd = _as_float_tensor(mjd_utc)
    table = _to(_LEAP_EFFECTIVE_MJD, device=mjd.device, dtype=mjd.dtype)
    offsets = _to(_TAI_MINUS_UTC, device=mjd.device, dtype=mjd.dtype)
    flat = mjd.reshape(-1)
    idx = torch.bucketize(flat, table, right=True) - 1
    idx = torch.clamp(idx, min=0, max=offsets.numel() - 1)
    values = offsets[idx]

    # UTC MJD days containing an inserted leap second have 86401 SI seconds.
    # Astropy/ERFA-compatible MJD coordinates therefore carry a linear
    # fractional-day offset ramp on [effective_mjd - 1, effective_mjd).
    for i in range(1, offsets.numel()):
        delta = offsets[i] - offsets[i - 1]
        if delta == 0:
            continue
        start = table[i] - 1.0
        mask = (flat >= start) & (flat < table[i])
        if torch.any(mask):
            values = torch.where(mask, offsets[i - 1] + (flat - start) * delta, values)
    return values.reshape(mjd.shape)


# UTC MJD bounds within which the leap table is authoritative: the first entry
# (1972-01-01; earlier UTC used drifting "rubber" seconds) and the expiry date
# of IERS Leap_Second.dat matching the table (2027-06-28).
_LEAP_TABLE_START_MJD = 41317.0
_LEAP_TABLE_EXPIRES_MJD = 61584.0


def _tt_minus_utc_seconds_checked(mjd_utc: Tensor) -> Tensor:
    """TT-UTC (s) from the leap table; ``ValueError`` outside its validity.

    Inside the table this is ``32.184 + (TAI-UTC)``, ERFA ``dat`` plus the
    TT-TAI constant, including the fractional ramp on leap-second days.
    Before 1972 or after the table expires TAI-UTC is unknown here, so the
    caller must supply it.
    """
    mjd = mjd_utc.detach().cpu().to(torch.float64)
    if not bool(torch.isfinite(mjd).all()):
        raise ValueError("UTC MJD must be finite to derive TT-UTC")
    lo = float(mjd.min()) if mjd.numel() else _LEAP_TABLE_START_MJD
    hi = float(mjd.max()) if mjd.numel() else _LEAP_TABLE_START_MJD
    if lo < _LEAP_TABLE_START_MJD or hi >= _LEAP_TABLE_EXPIRES_MJD:
        raise ValueError(
            "TT-UTC cannot be derived from torchsky's leap-second table outside "
            f"UTC MJD [{_LEAP_TABLE_START_MJD:g}, {_LEAP_TABLE_EXPIRES_MJD:g}) "
            "(1972-01-01 to the 2027-06-28 table expiry); pass "
            "EarthOrientation(tt_minus_utc_s=...) from IERS data"
        )
    return TT_MINUS_TAI_SECONDS + tai_minus_utc_seconds(mjd)


def convert_time_scale(mjd: Any, from_scale: str, to_scale: str) -> Tensor:
    """Convert MJD values between UTC, TAI, TT, and GPS time scales."""
    src = from_scale.strip().upper()
    dst = to_scale.strip().upper()
    value = _as_float_tensor(mjd)
    if src == dst:
        return value

    if src == "UTC":
        tai = value + tai_minus_utc_seconds(value) / SECONDS_PER_DAY
    elif src == "TAI":
        tai = value
    elif src == "TT":
        tai = value - TT_MINUS_TAI_SECONDS / SECONDS_PER_DAY
    elif src == "GPS":
        tai = value + TAI_MINUS_GPS_SECONDS / SECONDS_PER_DAY
    else:
        raise ValueError(f"unsupported input time scale {from_scale!r}")

    if dst == "TAI":
        return tai
    if dst == "TT":
        return tai + TT_MINUS_TAI_SECONDS / SECONDS_PER_DAY
    if dst == "GPS":
        return tai - TAI_MINUS_GPS_SECONDS / SECONDS_PER_DAY
    if dst == "UTC":
        utc = tai - tai_minus_utc_seconds(tai) / SECONDS_PER_DAY
        for _ in range(3):
            utc = tai - tai_minus_utc_seconds(utc) / SECONDS_PER_DAY
        return utc
    raise ValueError(f"unsupported output time scale {to_scale!r}")


def temporal_unit_to_days(unit: str | None) -> float:
    """Return days per temporal unit for common FITS/VO time units."""
    key = "d" if unit is None or str(unit).strip() == "" else str(unit).strip().lower()
    try:
        return _TIMEUNIT_TO_DAYS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported temporal unit {unit!r}") from exc


def canonical_time_reference_position(value: str | None) -> str:
    """Normalize common FITS temporal reference-position names."""
    key = "TOPOCENTER" if value is None else str(value).strip().upper()
    key = key.replace(" ", "").replace("_", "").replace("-", "")
    try:
        return _TREFPOS_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported temporal reference position {value!r}") from exc


def datetime_to_mjd(value: datetime.datetime) -> float:
    """Convert a timezone-aware or naive UTC datetime to MJD."""
    if value.tzinfo is not None:
        value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    epoch = datetime.datetime(1858, 11, 17)
    delta = value - epoch
    return delta.days + (delta.seconds + delta.microseconds * 1.0e-6) / SECONDS_PER_DAY


def iso_to_mjd(value: str) -> float:
    """Parse a common FITS ISO date/time string as UTC MJD."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "T" not in text and " " not in text:
        text = text + "T00:00:00"
    return datetime_to_mjd(datetime.datetime.fromisoformat(text))


SPEED_OF_LIGHT = 299792458.0  # m / s
AU_IN_METERS = 1.495978707e11

# JPL "Approximate Positions of the Planets" (Keplerian, 1800–2050).
# Tuple is a0, da, e0, de, I0, dI, L0, dL, varpi0, dvarpi, Omega0, dOmega
# with a in AU, angles in degrees, rates per Julian century from J2000.
_PLANET_KEPLER: dict[str, tuple[float, ...]] = {
    "MERCURY": (
        0.38709927,
        0.00000037,
        0.20563593,
        0.00001906,
        7.00497902,
        -0.00594749,
        252.25032350,
        149472.67411175,
        77.45779628,
        0.16047689,
        48.33076593,
        -0.12534081,
    ),
    "VENUS": (
        0.72333566,
        0.00000390,
        0.00677672,
        -0.00004107,
        3.39467605,
        -0.00078890,
        181.97909950,
        58517.81538729,
        131.60246718,
        0.00268329,
        76.67984255,
        -0.27769418,
    ),
    "EARTH": (
        1.00000011,
        -0.00000005,
        0.01671022,
        -0.00003804,
        0.00005,
        -0.01303889,
        100.46435,
        35999.37239722,
        102.94719,
        0.33285556,
        -11.26064,
        -5.06340278,
    ),
    "MARS": (
        1.52371034,
        0.00001847,
        0.09339410,
        0.00007882,
        1.84969142,
        -0.00813131,
        -4.55343205,
        19140.30268499,
        -23.94362959,
        0.44441088,
        49.55953891,
        -0.29257343,
    ),
    "JUPITER": (
        5.20288700,
        -0.00011607,
        0.04838624,
        -0.00013253,
        1.30439695,
        -0.00183714,
        34.39644051,
        3034.74612775,
        14.72847983,
        0.21252668,
        100.47390909,
        0.20469106,
    ),
    "SATURN": (
        9.53667594,
        -0.00125060,
        0.05386179,
        -0.00050991,
        2.48599187,
        0.00193609,
        49.95424423,
        1222.49362201,
        92.59887831,
        -0.41897216,
        113.66242448,
        -0.28867794,
    ),
    "URANUS": (
        19.18916464,
        -0.00196176,
        0.04725744,
        -0.00004397,
        0.77263783,
        -0.00242939,
        313.23810451,
        428.48202785,
        170.95427630,
        0.40805281,
        74.01692503,
        0.04240589,
    ),
    "NEPTUNE": (
        30.06992276,
        0.00026291,
        0.00859048,
        0.00005105,
        1.77004347,
        0.00035372,
        -55.12002969,
        218.45945325,
        44.96476227,
        -0.32241464,
        131.78422574,
        -0.00508664,
    ),
}

_IDENTITY_TREFPOS = {
    "TOPOCENTER",
    "GEOCENTER",
    "BASECENTER",
    "CUSTOM",
    "EMBARYCENTER",
    "MOON",
}
_BARY_LIKE_TREFPOS = {
    "BARYCENTER",
    "HELIOCENTER",
    "GALACTIC",
    "LOCALGROUP",
}


def _keplerian_heliocentric_equatorial(mjd: Tensor, body: str) -> Tensor:
    """Heliocentric J2000 equatorial position in meters (circular-Kepler bound).

    Not JPL DE. Light-time error vs DE440 is typically seconds to minutes at
    outer planets; use an ephemeris if that bound is too coarse.
    """
    elems = _PLANET_KEPLER[body]
    device = mjd.device
    dtype = mjd.dtype
    jd = mjd + 2400000.5
    T_flat = ((jd - 2451545.0) / 36525.0).to(dtype=dtype)
    d2r = math.pi / 180.0
    a0, da, e0, de, i0, di, L0, dL, vp0, dvp, om0, dom = elems
    a = _to(a0 + da * T_flat, device=device, dtype=dtype)
    e = _to(e0 + de * T_flat, device=device, dtype=dtype)
    i = _to((i0 + di * T_flat) * d2r, device=device, dtype=dtype)
    L = _to((L0 + dL * T_flat) * d2r, device=device, dtype=dtype)
    varpi = _to((vp0 + dvp * T_flat) * d2r, device=device, dtype=dtype)
    omega_node = _to((om0 + dom * T_flat) * d2r, device=device, dtype=dtype)

    omega_arg = varpi - omega_node
    M = L - varpi

    # Solve Kepler's equation M = E - e*sin(E)
    E = M.clone()
    for _ in range(3):
        E = E - (E - e * torch.sin(E) - M) / (1.0 - e * torch.cos(E))

    x_orb = a * (torch.cos(E) - e)
    y_orb = a * torch.sqrt(1.0 - e**2) * torch.sin(E)

    cos_o = torch.cos(omega_node)
    sin_o = torch.sin(omega_node)
    cos_w = torch.cos(omega_arg)
    sin_w = torch.sin(omega_arg)
    cos_i = torch.cos(i)
    sin_i = torch.sin(i)

    P_x = cos_o * cos_w - sin_o * sin_w * cos_i
    P_y = sin_o * cos_w + cos_o * sin_w * cos_i
    P_z = sin_w * sin_i

    Q_x = -cos_o * sin_w - sin_o * cos_w * cos_i
    Q_y = -sin_o * sin_w + cos_o * cos_w * cos_i
    Q_z = cos_w * sin_i

    r_x = x_orb * P_x + y_orb * Q_x
    r_y = x_orb * P_y + y_orb * Q_y
    r_z = x_orb * P_z + y_orb * Q_z

    obliq = 23.4392911 * d2r
    cos_ob = math.cos(obliq)
    sin_ob = math.sin(obliq)

    x_eq = r_x
    y_eq = r_y * cos_ob - r_z * sin_ob
    z_eq = r_y * sin_ob + r_z * cos_ob

    pos_au = torch.stack([x_eq, y_eq, z_eq], dim=-1)
    return pos_au * AU_IN_METERS


def _earth_position_equatorial(mjd: Tensor) -> Tensor:
    """Earth/Moon barycenter Keplerian position; same elements as before."""
    return _keplerian_heliocentric_equatorial(mjd, "EARTH")


def _gmst_rotation_matrix(mjd: Tensor) -> tuple[Tensor, Tensor]:
    """Calculate Greenwich Mean Sidereal Time (GMST) cos/sin at mjd."""
    jd = mjd + 2400000.5
    T = (jd - 2451545.0) / 36525.0

    T_flat = T.to(dtype=mjd.dtype)
    jd_flat = jd.to(dtype=mjd.dtype)

    gmst = (
        280.46061837
        + 360.98564736629 * (jd_flat - 2451545.0)
        + 0.000387933 * T_flat**2
        - T_flat**3 / 38710000.0
    )
    gmst_rad = _to(gmst * (math.pi / 180.0), device=mjd.device, dtype=mjd.dtype)

    return torch.cos(gmst_rad), torch.sin(gmst_rad)


class TemporalWCS:
    """
    FITS Temporal WCS implementation (Paper IV).

    Supports vectorized transformations between relative time and MJD plus
    leap-second-aware UTC/TAI/TT/GPS scale conversion.

    Main keywords:
    - MJDREF: Reference MJD
    - TIMESYS: Time scale (UTC, TT, TAI, TDB, GPS)
    - TIMEUNIT: 's', 'd', 'a', 'cy'
    - TREFPOS: TOPOCENTER, GEOCENTER, BARYCENTER, HELIOCENTER, EMBARYCENTER,
      MOON, CUSTOM, BASECENTER, GALACTIC, LOCALGROUP, and Keplerian planets.
    """

    def __init__(self, header: dict[str, Any]):
        self.mjd_ref = float(header.get("MJDREF", 0.0))
        # Handle MJDREFI and MJDREFF if present
        if "MJDREFI" in header and "MJDREFF" in header:
            self.mjd_ref = float(header["MJDREFI"]) + float(header["MJDREFF"])

        self.timesys = header.get("TIMESYS", "UTC").upper()
        self.timeunit = header.get("TIMEUNIT", "d").lower()
        self.trefpos = canonical_time_reference_position(header.get("TREFPOS"))
        self.trefdir = header.get("TREFDIR", None)
        self.plephem = header.get("PLEPHEM", None)
        self.timeoffs = float(header.get("TIMEOFFS", 0.0))

        self.obsgeo_xyz_m: tuple[float, float, float] | None = None
        if all(k in header for k in ("OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z")):
            self.obsgeo_xyz_m = (
                float(header["OBSGEO-X"]),
                float(header["OBSGEO-Y"]),
                float(header["OBSGEO-Z"]),
            )

        self.obsgeo_lon_lat_height: tuple[float, float, float] | None = None
        if all(k in header for k in ("OBSGEO-L", "OBSGEO-B", "OBSGEO-H")):
            self.obsgeo_lon_lat_height = (
                float(header["OBSGEO-L"]),
                float(header["OBSGEO-B"]),
                float(header["OBSGEO-H"]),
            )

        self.mjd_obs = self._mjd_metadata(header, "OBS")
        self.mjd_beg = self._mjd_metadata(header, "BEG")
        self.mjd_end = self._mjd_metadata(header, "END")
        self.exposure = self._optional_float(header, "EXPOSURE")
        self.telapse = self._optional_float(header, "TELAPSE")

        # Scaling factor to days
        self.to_days = temporal_unit_to_days(self.timeunit)

    def to_mjd(self, time_val: Tensor) -> Tensor:
        """Convert relative time to absolute MJD in the header ``TIMESYS``."""
        return self.mjd_ref + (time_val + self.timeoffs) * self.to_days

    def to_mjd_scale(self, time_val: Tensor, scale: str) -> Tensor:
        """Convert relative time to absolute MJD in another supported scale."""
        return convert_time_scale(self.to_mjd(time_val), self.timesys, scale)

    def from_mjd(self, mjd: Tensor) -> Tensor:
        """Convert absolute MJD in the header ``TIMESYS`` to relative time."""
        return (mjd - self.mjd_ref) / self.to_days - self.timeoffs

    def from_mjd_scale(self, mjd: Tensor, scale: str) -> Tensor:
        """Convert absolute MJD in ``scale`` to relative header time values."""
        local = convert_time_scale(mjd, scale, self.timesys)
        return self.from_mjd(local)

    @staticmethod
    def mjd_to_jd(mjd: float | Tensor) -> float | Tensor:
        """MJD to JD: JD = MJD + 2400000.5"""
        return mjd + 2400000.5

    @staticmethod
    def jd_to_mjd(jd: float | Tensor) -> float | Tensor:
        """JD to MJD: MJD = JD - 2400000.5"""
        return jd - 2400000.5

    def to_iso8601(self, mjd: Tensor) -> str | list[Any]:
        """
        Convert MJD to ISO-8601 strings (UTC).
        Note: This is not vectorized as strings are involved,
        but useful for parity.
        """
        utc = convert_time_scale(mjd, self.timesys, "UTC").detach().cpu()

        def format_values(values: Tensor) -> str | list[Any]:
            if values.ndim == 0:
                dt = datetime.datetime(1858, 11, 17) + datetime.timedelta(
                    days=float(values.item())
                )
                return dt.isoformat()
            return [format_values(item) for item in values]

        return format_values(utc)

    def apply_corrections(self, mjd: Tensor, ra: Tensor, dec: Tensor) -> Tensor:
        """Apply Roemer light-time to the header ``TREFPOS``.

        BARYCENTER/HELIOCENTER use a Keplerian Earth (Sun ≈ SSB). Planetary
        ``TREFPOS`` uses the same JPL approximate Keplerian elements, not DE.
        EMBARYCENTER and MOON are treated as GEOCENTER (~0.01 s and ~1.3 s).
        GALACTIC/LOCALGROUP use the barycentric delay (no Galactic motion).
        """
        if self.trefpos in _IDENTITY_TREFPOS:
            return mjd

        r_earth = _earth_position_equatorial(mjd)
        if self.trefpos in _BARY_LIKE_TREFPOS:
            r_ref = torch.zeros_like(r_earth)
        elif self.trefpos in _PLANET_KEPLER and self.trefpos != "EARTH":
            r_ref = _keplerian_heliocentric_equatorial(mjd, self.trefpos)
        else:
            raise NotImplementedError(
                f"{self.trefpos} temporal corrections are not supported"
            )

        obs_x, obs_y, obs_z = 0.0, 0.0, 0.0
        has_obs = False
        if self.obsgeo_xyz_m is not None:
            obs_x, obs_y, obs_z = self.obsgeo_xyz_m
            has_obs = True
        elif self.obsgeo_lon_lat_height is not None:
            lon, lat, height = self.obsgeo_lon_lat_height
            d2r = math.pi / 180.0
            a_wgs = 6378137.0
            f_wgs = 1.0 / 298.257223563
            e2 = 2 * f_wgs - f_wgs**2
            sin_lat = math.sin(lat * d2r)
            cos_lat = math.cos(lat * d2r)
            N = a_wgs / math.sqrt(1.0 - e2 * sin_lat**2)
            obs_x = (N + height) * cos_lat * math.cos(lon * d2r)
            obs_y = (N + height) * cos_lat * math.sin(lon * d2r)
            obs_z = (N * (1.0 - e2) + height) * sin_lat
            has_obs = True

        if has_obs:
            cos_g, sin_g = _gmst_rotation_matrix(mjd)
            obs_inertial_x = obs_x * cos_g - obs_y * sin_g
            obs_inertial_y = obs_x * sin_g + obs_y * cos_g
            obs_inertial_z = torch.full_like(obs_inertial_x, obs_z)
            r_obs = torch.stack(
                [obs_inertial_x, obs_inertial_y, obs_inertial_z], dim=-1
            )
            r = r_earth + r_obs - r_ref
        else:
            r = r_earth - r_ref

        d2r = math.pi / 180.0
        cos_dec = torch.cos(dec * d2r)
        s_x = cos_dec * torch.cos(ra * d2r)
        s_y = cos_dec * torch.sin(ra * d2r)
        s_z = torch.sin(dec * d2r)
        s = torch.stack([s_x, s_y, s_z], dim=-1)
        dt_days = torch.sum(r * s, dim=-1) / (SPEED_OF_LIGHT * SECONDS_PER_DAY)
        return mjd + dt_days

    @property
    def requires_ephemeris(self) -> bool:
        return self.trefpos not in _IDENTITY_TREFPOS

    @property
    def reference_position(self) -> str:
        return self.trefpos

    @property
    def reference_direction(self) -> Any:
        return self.trefdir

    @property
    def ephemeris(self) -> Any:
        return self.plephem

    @property
    def has_observatory_location(self) -> bool:
        return self.obsgeo_xyz_m is not None or self.obsgeo_lon_lat_height is not None

    @property
    def observatory_xyz_m(self) -> tuple[float, float, float] | None:
        return self.obsgeo_xyz_m

    @property
    def observatory_lon_lat_height_deg_m(self) -> tuple[float, float, float] | None:
        return self.obsgeo_lon_lat_height

    @property
    def duration_days(self) -> float | None:
        """Observation duration in days from MJD-BEG/END, TELAPSE, or EXPOSURE."""
        if self.mjd_beg is not None and self.mjd_end is not None:
            return self.mjd_end - self.mjd_beg
        if self.telapse is not None:
            return self.telapse * self.to_days
        if self.exposure is not None:
            return self.exposure / SECONDS_PER_DAY
        return None

    @property
    def midpoint_mjd(self) -> float | None:
        """Observation midpoint MJD when enough metadata is available."""
        if self.mjd_beg is not None and self.mjd_end is not None:
            return 0.5 * (self.mjd_beg + self.mjd_end)
        duration = self.duration_days
        if self.mjd_obs is not None and duration is not None:
            return self.mjd_obs + 0.5 * duration
        return self.mjd_obs

    @staticmethod
    def _mjd_metadata(header: dict[str, Any], suffix: str) -> float | None:
        mjd_key = f"MJD-{suffix}"
        date_key = f"DATE-{suffix}"
        if mjd_key in header:
            return float(header[mjd_key])
        if date_key in header:
            return iso_to_mjd(str(header[date_key]))
        return None

    @staticmethod
    def _optional_float(header: dict[str, Any], key: str) -> float | None:
        return float(header[key]) if key in header else None
