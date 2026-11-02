"""Header parsing and axis identification for :class:`torchsky.wcs.WCS`."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
)
import math


from .tpv import TPV, PV1_KEYS_SET, PV2_KEYS_SET
from .sip import SIP
from .legacy import (
    extract_zpx_params,
)

if TYPE_CHECKING:
    pass

from ._core_support import (
    _celestial_axis_role,
    _ctype_projection_code,
)


class _HeaderParseMixin:
    if TYPE_CHECKING:
        # State and methods supplied by the host class.
        def __getattr__(self, name: str) -> Any: ...

    @staticmethod
    def _normalize_wcs_key(key: str | None) -> str:
        """Normalize a FITS alternate WCS version key."""
        if key is None or key == " ":
            return ""
        key_norm = str(key).strip().upper()
        if key_norm == "":
            return ""
        if len(key_norm) != 1 or not ("A" <= key_norm <= "Z"):
            raise ValueError("alternate WCS key must be '', ' ', or one letter A-Z")
        return key_norm

    @staticmethod
    def available_wcs_keys(header: Dict[str, Any]) -> tuple[str, ...]:
        """Return FITS-WCS version keys present in a header.

        The primary WCS is returned as ``""``; alternate WCS versions use ``"A"`` ..
        ``"Z"`` following the FITS standard keyword suffix convention.
        """
        keys: set[str] = set()
        for raw_key in header.keys():
            key = str(raw_key).strip().upper()
            if key in {"WCSAXES", "CTYPE1", "CTYPE2"}:
                keys.add("")
                continue
            if len(key) >= 2 and key[-1].isalpha():
                suffix = key[-1]
                base = key[:-1]
                if suffix == "A" or ("A" <= suffix <= "Z"):
                    if base == "WCSAXES" or base.startswith(
                        ("CTYPE", "CRPIX", "CRVAL", "CDELT", "CUNIT")
                    ):
                        keys.add(suffix)
        if not keys:
            keys.add("")
        return tuple(sorted(keys, key=lambda k: (k != "", k)))

    def _alt_key(self, key: str) -> str:
        return f"{key}{self.wcs_key}" if self.wcs_key else key

    def _header_get(self, header: Dict[str, Any], key: str, default: Any = None) -> Any:
        alt_key = self._alt_key(key)
        if alt_key in header:
            return header[alt_key]
        if self.wcs_key:
            return default
        return header.get(key, default)

    def _header_has(self, header: Dict[str, Any], key: str) -> bool:
        return self._alt_key(key) in header if self.wcs_key else key in header

    def _infer_wcs_naxis(self, header: Dict[str, Any]) -> int:
        key = self._alt_key("WCSAXES")
        if key in header:
            return int(header[key])
        if "WCSAXES" in header and not self.wcs_key:
            return int(header["WCSAXES"])
        return int(header.get("NAXIS", 2))

    def _infer_pixel_shape(self, header: Dict[str, Any]) -> tuple[int, ...] | None:
        naxis = int(header.get("NAXIS", self.naxis))
        if naxis <= 0:
            return None
        values: list[int] = []
        for i in range(1, naxis + 1):
            key = f"NAXIS{i}"
            if key not in header:
                return None
            values.append(int(header[key]))
        return tuple(values)

    def _parse_core(self, header: Dict[str, Any]) -> None:
        """Parse core WCS keywords (CTYPE, CRPIX, CRVAL, CDELT, CUNIT)."""
        for i in range(1, self.naxis + 1):
            self.wcs_params[f"CTYPE{i}"] = str(
                self._header_get(header, f"CTYPE{i}", "")
            )
            self.wcs_params[f"CRPIX{i}"] = float(
                self._header_get(header, f"CRPIX{i}", 0.0)
            )
            self.wcs_params[f"CRVAL{i}"] = float(
                self._header_get(header, f"CRVAL{i}", 0.0)
            )
            self.wcs_params[f"CDELT{i}"] = float(
                self._header_get(header, f"CDELT{i}", 1.0)
            )
            if self._header_has(header, f"CUNIT{i}"):
                self.wcs_params[f"CUNIT{i}"] = str(
                    self._header_get(header, f"CUNIT{i}", "")
                )

    def _supported_celestial_codes(self) -> frozenset[str]:
        # TAB: celestial axes backed by -TAB lookup tables.
        return frozenset(
            self.ZENITHAL_CODES
            + self.CYLINDRICAL_CODES
            + self.ALLSKY_CODES
            + self.CONIC_CODES
            + self.POLYCONIC_CODES
            + self.QUADCUBE_CODES
            + self.PSEUDOCYLINDRICAL_CODES
            + self.SPECIAL_CODES
            + ("TAB",)
        )

    def _identify_celestial_axes(self) -> None:
        longitude = []
        latitude = []
        for axis in range(self.naxis):
            ctype = str(self.wcs_params.get(f"CTYPE{axis + 1}", ""))
            role = _celestial_axis_role(ctype)
            if role == "longitude":
                longitude.append(axis)
            elif role == "latitude":
                latitude.append(axis)
        if not longitude and not latitude:
            self._has_celestial = False
            self._celestial_lon_axis = 0
            self._celestial_lat_axis = 1 if self.naxis >= 2 else 0
            self._celestial_axes = ()
            return
        if len(longitude) != 1 or len(latitude) != 1:
            raise ValueError(
                "celestial WCS requires exactly one longitude axis and one latitude axis"
            )
        lon_axis = longitude[0]
        lat_axis = latitude[0]
        lon_code = _ctype_projection_code(
            str(self.wcs_params.get(f"CTYPE{lon_axis + 1}", ""))
        )
        lat_code = _ctype_projection_code(
            str(self.wcs_params.get(f"CTYPE{lat_axis + 1}", ""))
        )
        if lon_code != lat_code:
            raise ValueError(
                "celestial longitude and latitude axes must use the same projection code"
            )
        supported = self._supported_celestial_codes()
        if lon_code not in supported:
            raise ValueError(
                "unsupported celestial projection "
                f"{lon_code!r} in CTYPE{lon_axis + 1}="
                f"{self.wcs_params.get(f'CTYPE{lon_axis + 1}', '')!r} and "
                f"CTYPE{lat_axis + 1}="
                f"{self.wcs_params.get(f'CTYPE{lat_axis + 1}', '')!r}; "
                f"supported projection codes: {', '.join(sorted(supported))}"
            )
        self._has_celestial = True
        self._celestial_lon_axis = lon_axis
        self._celestial_lat_axis = lat_axis
        self._celestial_axes = tuple(sorted((lon_axis, lat_axis)))

    def _setup_projection_params(self) -> None:
        params = dict(self.wcs_params)
        if self._has_celestial:
            for prefix in ("PV1_", "PV2_"):
                for key in [key for key in params if str(key).startswith(prefix)]:
                    params.pop(key, None)
            lon_number = self._celestial_lon_axis + 1
            lat_number = self._celestial_lat_axis + 1
            for index in range(40):
                lon_key = f"PV{lon_number}_{index}"
                lat_key = f"PV{lat_number}_{index}"
                if lon_key in self.wcs_params:
                    params[f"PV1_{index}"] = self.wcs_params[lon_key]
                if lat_key in self.wcs_params:
                    params[f"PV2_{index}"] = self.wcs_params[lat_key]
        self._projection_params = params

    def _celestial_unit_factor(self, axis: int) -> float:
        ctype = str(self.wcs_params.get(f"CTYPE{axis + 1}", "")).strip().upper()
        celestial = ctype.startswith(
            ("RA--", "DEC-", "GLON", "GLAT", "ELON", "ELAT", "HLON", "HLAT")
        )
        if not celestial:
            return 1.0
        unit = str(self.wcs_params.get(f"CUNIT{axis + 1}", "deg")).strip().lower()
        if unit in {"", "deg", "degree", "degrees"}:
            return 1.0
        if unit in {"rad", "radian", "radians"}:
            return 180.0 / math.pi
        if unit in {"arcmin", "arcminute", "arcminutes"}:
            return 1.0 / 60.0
        if unit in {"arcsec", "arcsecond", "arcseconds"}:
            return 1.0 / 3600.0
        raise ValueError(f"unsupported celestial CUNIT{axis + 1}: {unit!r}")

    def _parse_cd_pc(self, header: Dict[str, Any]) -> None:
        """Parse CD and PC matrix keywords."""
        for i in range(1, self.naxis + 1):
            for j in range(1, self.naxis + 1):
                cd_key = f"CD{i}_{j}"
                if self._header_has(header, cd_key):
                    self.wcs_params[cd_key] = float(self._header_get(header, cd_key))
                pc_key = f"PC{i}_{j}"
                if self._header_has(header, pc_key):
                    self.wcs_params[pc_key] = float(self._header_get(header, pc_key))
            crota_key = f"CROTA{i}"
            if self._header_has(header, crota_key):
                self.wcs_params[crota_key] = float(self._header_get(header, crota_key))

    def _parse_pole(self, header: Dict[str, Any]) -> None:
        """Parse LONPOLE and LATPOLE keywords."""
        if self._header_has(header, "LONPOLE"):
            self.wcs_params["LONPOLE"] = float(self._header_get(header, "LONPOLE"))
        if self._header_has(header, "LATPOLE"):
            self.wcs_params["LATPOLE"] = float(self._header_get(header, "LATPOLE"))

    def _parse_pv(self, header: Dict[str, Any]) -> None:
        """Parse PV distortion keywords."""
        axes = (
            (self._celestial_lon_axis, self._celestial_lat_axis)
            if self._has_celestial
            else tuple(range(min(self.naxis, 2)))
        )
        for axis in axes:
            for index in range(40):
                key = f"PV{axis + 1}_{index}"
                if self._header_has(header, key):
                    self.wcs_params[key] = float(self._header_get(header, key))

    def _parse_wat(self, header: Dict[str, Any]) -> None:
        """Parse WAT keywords for TNX/ZPX projections."""
        self.wat_data = {}
        has_wat = any(str(k).startswith("WAT") for k in header.keys())
        if has_wat:
            from .legacy import parse_wat_keywords

            lon_number = self._celestial_lon_axis + 1 if self._has_celestial else 1
            lat_number = self._celestial_lat_axis + 1 if self._has_celestial else 2
            self.wat_data[1] = parse_wat_keywords(header, prefix=f"WAT{lon_number}")
            self.wat_data[2] = parse_wat_keywords(header, prefix=f"WAT{lat_number}")
            lon_ctype = str(self.wcs_params.get(f"CTYPE{lon_number}", ""))
            if lon_ctype.strip().upper().endswith("ZPX"):
                for k, v in extract_zpx_params(self.wat_data).items():
                    self.wcs_params.setdefault(k, v)

    def _parse_sip(self, header: Dict[str, Any]) -> None:
        """Parse SIP distortion keywords."""
        if self._header_has(header, "A_ORDER") or self._header_has(header, "B_ORDER"):
            self.sip = SIP(header)

    def _parse_tpv(self) -> None:
        """Determine if TPV distortion is present."""
        has_pv = any(k in self.wcs_params for k in PV1_KEYS_SET) or any(
            k in self.wcs_params for k in PV2_KEYS_SET
        )
        is_tpv_ctype = self.wcs_params.get("CTYPE1", "").strip().upper().endswith("TPV")

        if is_tpv_ctype or (
            self.wcs_params.get("CTYPE1", "").strip().upper().endswith("TAN") and has_pv
        ):
            self.tpv = TPV(self.wcs_params)

    def _parse_header(self, header: Dict[str, Any]):
        """Parse standard FITS WCS keywords."""
        self.wcs_params["NAXIS"] = self.naxis

        self._parse_core(header)
        self._identify_celestial_axes()
        self._parse_cd_pc(header)
        self._parse_pole(header)
        self._parse_pv(header)
        self._parse_wat(header)
        self._parse_sip(header)
        self._parse_tpv()
        self._parse_spectral_metadata(header)
        self._parse_temporal_metadata(header)
        self._parse_celestial_metadata(header)
        for key in (
            "TSKYAF0",
            "TSKYAFA11",
            "TSKYAFA12",
            "TSKYAFA21",
            "TSKYAFA22",
            "TSKYAFB1",
            "TSKYAFB2",
        ):
            if key in header:
                self.wcs_params[key] = header[key]

    def _parse_spectral_metadata(self, header: Dict[str, Any]) -> None:
        for key in (
            "RESTFRQ",
            "RESTFREQ",
            "RESTWAV",
            "SPECSYS",
            "SSYSOBS",
            "SSYSSRC",
            "VELREF",
        ):
            if self._header_has(header, key):
                self.wcs_params[key] = self._header_get(header, key)

    def _parse_celestial_metadata(self, header: Dict[str, Any]) -> None:
        for key in ("RADESYS", "RADECSYS", "EQUINOX", "EPOCH"):
            if self._header_has(header, key):
                self.wcs_params[key] = self._header_get(header, key)

    def _parse_temporal_metadata(self, header: Dict[str, Any]) -> None:
        for key in (
            "MJDREF",
            "MJDREFI",
            "MJDREFF",
            "TIMESYS",
            "TIMEUNIT",
            "TREFPOS",
            "TREFDIR",
            "PLEPHEM",
            "TIMEOFFS",
            "DATE-OBS",
            "DATE-BEG",
            "DATE-END",
            "MJD-OBS",
            "MJD-BEG",
            "MJD-END",
            "EXPOSURE",
            "TELAPSE",
            "OBSGEO-X",
            "OBSGEO-Y",
            "OBSGEO-Z",
            "OBSGEO-L",
            "OBSGEO-B",
            "OBSGEO-H",
        ):
            if self._header_has(header, key):
                self.wcs_params[key] = self._header_get(header, key)
