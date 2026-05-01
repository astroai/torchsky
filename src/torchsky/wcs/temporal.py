from __future__ import annotations

import datetime
from typing import Any

import torch
from torch import Tensor

SECONDS_PER_DAY = 86400.0
TT_MINUS_TAI_SECONDS = 32.184
TAI_MINUS_GPS_SECONDS = 19.0

# Effective MJD dates at 00:00:00 UTC and TAI-UTC offsets in seconds.
# IERS Bulletin C 71 / NIST still give TAI-UTC = 37 s from 2017-01-01
# through at least the 2026-06-30 no-leap announcement.
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


def _as_float_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(dtype=torch.float64)
    return torch.as_tensor(value, dtype=torch.float64)


def tai_minus_utc_seconds(mjd_utc: Any) -> Tensor:
    """Return TAI-UTC in seconds for UTC MJD values covered by leap table."""
    mjd = _as_float_tensor(mjd_utc)
    table = _LEAP_EFFECTIVE_MJD.to(device=mjd.device)
    offsets = _TAI_MINUS_UTC.to(device=mjd.device)
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


class TemporalWCS:
    """
    FITS Temporal WCS implementation (Paper IV).

    Supports vectorized transformations between relative time and MJD plus
    leap-second-aware UTC/TAI/TT/GPS scale conversion.

    Main keywords:
    - MJDREF: Reference MJD
    - TIMESYS: Time scale (UTC, TT, TAI, TDB, GPS)
    - TIMEUNIT: 's', 'd', 'a', 'cy'
    - TREFPOS: 'TOPOCENTER', 'GEOCENTER', 'BARYCENTER'
    """

    def __init__(self, header: dict[str, Any]):
        self.mjd_ref = float(header.get("MJDREF", 0.0))
        # Handle MJDREFI and MJDREFF if present
        if "MJDREFI" in header and "MJDREFF" in header:
            self.mjd_ref = float(header["MJDREFI"]) + float(header["MJDREFF"])

        self.timesys = header.get("TIMESYS", "UTC").upper()
        self.timeunit = header.get("TIMEUNIT", "d").lower()
        self.trefpos = header.get("TREFPOS", "TOPOCENTER").upper()
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
        return self.mjd_ref + time_val * self.to_days

    def to_mjd_scale(self, time_val: Tensor, scale: str) -> Tensor:
        """Convert relative time to absolute MJD in another supported scale."""
        return convert_time_scale(self.to_mjd(time_val), self.timesys, scale)

    def from_mjd(self, mjd: Tensor) -> Tensor:
        """Convert absolute MJD in the header ``TIMESYS`` to relative time."""
        return (mjd - self.mjd_ref) / self.to_days

    def from_mjd_scale(self, mjd: Tensor, scale: str) -> Tensor:
        """Convert absolute MJD in ``scale`` to relative header time values."""
        local = convert_time_scale(mjd, scale, self.timesys)
        return self.from_mjd(local)

    def convert_scale(self, mjd: Tensor, to_scale: str, from_scale: str | None = None) -> Tensor:
        """Convert MJD values from ``from_scale`` or header ``TIMESYS`` to ``to_scale``."""
        return convert_time_scale(mjd, from_scale or self.timesys, to_scale)

    @staticmethod
    def mjd_to_jd(mjd: float | Tensor) -> float | Tensor:
        """MJD to JD: JD = MJD + 2400000.5"""
        return mjd + 2400000.5

    @staticmethod
    def jd_to_mjd(jd: float | Tensor) -> float | Tensor:
        """JD to MJD: MJD = JD - 2400000.5"""
        return jd - 2400000.5

    def to_iso8601(self, mjd: Tensor) -> list[str]:
        """
        Convert MJD to ISO-8601 strings (UTC).
        Note: This is not vectorized as strings are involved,
        but useful for parity.
        """
        mjds = mjd.cpu().numpy()
        isos = []
        for m in mjds:
            # JD = MJD + 2400000.5
            # Simplified MJD to datetime
            dt = datetime.datetime(1858, 11, 17) + datetime.timedelta(days=float(m))
            isos.append(dt.isoformat())
        return isos

    def apply_corrections(self, mjd: Tensor, ra: Tensor, dec: Tensor) -> Tensor:
        """
        Apply reference-position light-time corrections when locally available.

        Full barycentric/heliocentric/geocentric corrections require observatory
        location plus solar-system ephemerides. torchsky does not currently ship
        an ephemeris backend, so unsupported corrections fail explicitly instead
        of returning a silently wrong timestamp.
        """
        if self.trefpos in {"TOPOCENTER", "TOPOCENTRE"}:
            return mjd
        raise NotImplementedError(
            f"{self.trefpos} temporal corrections require an ephemeris backend"
        )

    @property
    def requires_ephemeris(self) -> bool:
        return self.trefpos not in {"TOPOCENTER", "TOPOCENTRE"}

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
