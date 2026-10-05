from __future__ import annotations

import torch
from torch import Tensor
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import math
import os
import numpy as np

from torchsky._native import cpp as _cpp

from .tpv import TPV, PV1_KEYS, PV2_KEYS, PV1_KEYS_SET, PV2_KEYS_SET
from .sip import SIP
from .zenithal import project_zenithal, deproject_zenithal
from .cylindrical import project_cylindrical, deproject_cylindrical
from .allsky import project_allsky, deproject_allsky
from .conic import project_conic, deproject_conic
from .quadcube import project_quadcube, deproject_quadcube
from .legacy import (
    deproject_zpx,
    extract_zpx_params,
    invert_tnx_plane,
    project_tnx,
    project_zpx,
)
from .spectral import (
    canonical_spectral_type,
    frequency_to_spectral as _frequency_to_spectral,
    rest_frequency_from_header,
    spectral_to_frequency as _spectral_to_frequency,
)
from .tabular import CoupledTabularLookupWCS, TabularAxis, _solve_coupled_tabular_partial
from .temporal import TemporalWCS
from .frames import (
    canonical_celestial_frame,
    convert_celestial as _convert_celestial,
    precess_fk5 as _precess_fk5,
)

_HAS_TORCH_SINCOS = hasattr(torch, "sincos")


# Benchmark-backed small-batch CPU crossover points for the center-equator C++
# helpers. Above these sizes, the vectorized Torch path is typically as fast or
# faster, so we keep the native path for throughput-oriented workloads.
_CENTER_CPP_P2W_THRESHOLDS = {
    "AIT": 4096,
    "CAR": 1024,
    "CEA": 512,
    "HPX": 1024,
    "MER": 512,
    "SFL": 1024,
}

_CENTER_CPP_W2P_THRESHOLDS = {
    "AIT": 1024,
    "CAR": 512,
    "CEA": 2048,
    "HPX": 1024,
    "MER": 1024,
    "SFL": 2048,
}

# Fused-first only for zenithal pole two-arg p2w below this N. Vectorized
# cpp_direct stays primary above it so CEA/CAR and N>=10K are not regressable.
_ZENITHAL_FUSED_P2W_MAX = 1536


def _sincos(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Return sin(x), cos(x) with a torch.sincos fast path when available."""
    if _HAS_TORCH_SINCOS:
        return torch.sincos(x)
    return torch.sin(x), torch.cos(x)


def _wrap_lon360(angle: Tensor) -> Tensor:
    """Wrap longitudes to [0, 360)."""
    # Avoid torch.amin/amax bounds checks: they force GPU-CPU sync on large tensors.
    return torch.remainder(angle, 360.0)


def _wrap_lon360_checked(angle: Tensor) -> Tensor:
    """Compatibility alias; direct wrap is faster than range-check reductions."""
    return _wrap_lon360(angle)


def _wrap_lon180(angle: Tensor) -> Tensor:
    """Wrap longitudes to [-180, 180)."""
    return torch.remainder(angle + 180.0, 360.0) - 180.0


def _wrap_lon180_checked(angle: Tensor) -> Tensor:
    """Compatibility alias; direct wrap is faster than range-check reductions."""
    return _wrap_lon180(angle)


def _ctype_without_projection(ctype: str) -> str:
    """Strip FITS projection / SIP suffixes, leaving the coordinate type."""
    text = str(ctype).strip()
    upper = text.upper()
    if upper.endswith("-SIP"):
        text = text[:-4]
        upper = upper[:-4]
    if len(upper) >= 5 and upper[-4] == "-":
        return text[:-4].rstrip("-")
    return text


class DistortionLookupTable:
    """FITS WCS Paper IV 2D distortion lookup table (`CPDIS` / `DET2IM`).

    Matches `astropy.wcs.DistortionLookupTable` semantics:
    1-based FITS pixel coordinates `(x, y)` map to 0-based table indices
    `tx = (x - crval1) / cdelt1 + crpix1 - 1`, `ty = (y - crval2) / cdelt2 + crpix2 - 1`
    and are bilinearly interpolated on `data` (shape `(H, W)` = `(NAXIS2, NAXIS1)`),
    with clamping at table boundaries.
    """

    def __init__(
        self,
        data: Any,
        crpix: tuple[float, float] = (1.0, 1.0),
        crval: tuple[float, float] = (1.0, 1.0),
        cdelt: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        tensor = torch.as_tensor(data)
        if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
            raise ValueError("DistortionLookupTable data must be a non-empty 2D array")
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float64)
        self.data: Tensor = tensor
        self.crpix: tuple[float, float] = (float(crpix[0]), float(crpix[1]))
        self.crval: tuple[float, float] = (float(crval[0]), float(crval[1]))
        self.cdelt: tuple[float, float] = (float(cdelt[0]), float(cdelt[1]))
        if abs(self.cdelt[0]) <= 0.0 or abs(self.cdelt[1]) <= 0.0:
            raise ValueError("DistortionLookupTable cdelt must be non-zero")

    def to(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> "DistortionLookupTable":
        dev = torch.device(device)
        target_dtype = (
            dtype
            if dtype is not None
            else (torch.float32 if dev.type == "mps" else self.data.dtype)
        )
        self.data = self.data.to(device=dev, dtype=target_dtype)
        return self

    def eval_tensor(self, x1: Tensor, y1: Tensor) -> Tensor:
        """Evaluate bilinear lookup at 1-based FITS pixel coordinates `(x1, y1)`."""
        table = self.data.to(device=x1.device, dtype=x1.dtype)
        h, w = int(table.shape[0]), int(table.shape[1])
        tx = (x1 - self.crval[0]) / self.cdelt[0] + (self.crpix[0] - 1.0)
        ty = (y1 - self.crval[1]) / self.cdelt[1] + (self.crpix[1] - 1.0)
        tx_c = torch.clamp(tx, 0.0, float(max(w - 1, 0)))
        ty_c = torch.clamp(ty, 0.0, float(max(h - 1, 0)))
        x0 = torch.clamp(torch.floor(tx_c).to(torch.long), 0, max(w - 2, 0))
        y0 = torch.clamp(torch.floor(ty_c).to(torch.long), 0, max(h - 2, 0))
        x_hi = torch.clamp(x0 + 1, 0, max(w - 1, 0))
        y_hi = torch.clamp(y0 + 1, 0, max(h - 1, 0))
        fx = torch.clamp(tx_c - x0.to(tx_c.dtype), 0.0, 1.0)
        fy = torch.clamp(ty_c - y0.to(ty_c.dtype), 0.0, 1.0)
        v00 = table[y0, x0]
        v10 = table[y0, x_hi]
        v01 = table[y_hi, x0]
        v11 = table[y_hi, x_hi]
        return (
            v00 * (1.0 - fx) * (1.0 - fy)
            + v10 * fx * (1.0 - fy)
            + v01 * (1.0 - fx) * fy
            + v11 * fx * fy
        )

    def get_offset(self, x: Any, y: Any) -> Any:
        """Astropy-compatible offset lookup at 1-based FITS pixel coordinates `(x, y)`."""
        if isinstance(x, Tensor) or isinstance(y, Tensor):
            ref = x if isinstance(x, Tensor) else y
            assert isinstance(ref, Tensor)
            dt = torch.float32 if ref.device.type == "mps" else torch.float64
            xt = torch.as_tensor(x, dtype=dt, device=ref.device)
            yt = torch.as_tensor(y, dtype=dt, device=ref.device)
            return self.eval_tensor(xt, yt)
        xt = torch.as_tensor(x, dtype=torch.float64)
        yt = torch.as_tensor(y, dtype=torch.float64)
        out = self.eval_tensor(xt, yt)
        if out.ndim == 0:
            return float(out.item())
        return out.detach().cpu().numpy()


def _coerce_distortion_lookup_table(
    table: Any | None,
) -> DistortionLookupTable | None:
    if table is None:
        return None
    if isinstance(table, DistortionLookupTable):
        return table
    if all(hasattr(table, attr) for attr in ("data", "crpix", "crval", "cdelt")):
        return DistortionLookupTable(
            table.data,
            tuple(table.crpix),
            tuple(table.crval),
            tuple(table.cdelt),
        )
    raise TypeError(
        "distortion lookup table must be a DistortionLookupTable or compatible object"
    )


class WCS:
    """
    Base class for TorchWCS (PyTorch-native World Coordinate System).
    """

    ZENITHAL_CODES = ("TAN", "SIN", "ARC", "ZPN", "STG", "ZEA", "AZP", "SZP", "AIR")
    CYLINDRICAL_CODES = ("CEA", "MER", "CYP", "CAR")
    ALLSKY_CODES = ("AIT", "MOL", "HPX", "SFL", "XPH")
    CONIC_CODES = ("COP", "COE", "COD", "COO")
    POLYCONIC_CODES = ("BON", "PCO")
    QUADCUBE_CODES = ("TSC", "CSC", "QSC")
    PSEUDOCYLINDRICAL_CODES = ("PAR",)
    # NCP is rewritten to slant SIN in __init__ (Paper II §6.1.2); GLS is Global Sinusoidal (§6.1.1).
    SPECIAL_CODES = ("ZPX", "TNX", "TPV", "NCP", "GLS")

    def __init__(
        self,
        header: Optional[Dict[str, Any]] = None,
        key: str = "",
        tabular_axes: Mapping[int, TabularAxis] | None = None,
        coupled_tabular: CoupledTabularLookupWCS | None = None,
        coupled_tabular_axes: Sequence[int] | None = None,
        coupled_tabular_pixel_axes: Sequence[int] | None = None,
        tables: Any = None,
        cpdis1: DistortionLookupTable | Any | None = None,
        cpdis2: DistortionLookupTable | Any | None = None,
        det2im1: DistortionLookupTable | Any | None = None,
        det2im2: DistortionLookupTable | Any | None = None,
        **kwargs,
    ):
        """
        Initialize WCS object.

        Args:
            header: FITS header dictionary (or similar object behaving like dict)
            **kwargs: Manual override for WCS keywords (CRVAL, CRPIX, CD, etc.)
        """
        self.wcs_params = {}
        self.sip = None
        self.tpv = None
        self.cpdis1: DistortionLookupTable | None = _coerce_distortion_lookup_table(cpdis1)
        self.cpdis2: DistortionLookupTable | None = _coerce_distortion_lookup_table(cpdis2)
        self.det2im1: DistortionLookupTable | None = _coerce_distortion_lookup_table(det2im1)
        self.det2im2: DistortionLookupTable | None = _coerce_distortion_lookup_table(det2im2)
        self.wat_data = {}
        self.wcs_key = self._normalize_wcs_key(key)
        self._pixel_shape: tuple[int, ...] | None = None
        self.tabular_axes: dict[int, TabularAxis] = {}
        self.coupled_tabular: CoupledTabularLookupWCS | None = None
        self.coupled_tabular_axes: tuple[int, ...] = ()
        self.coupled_tabular_pixel_axes: tuple[int, ...] = ()
        # Optional pixel-space affine (corrected -> source) for distorted WCS.
        self._pixel_affine_matrix: Tensor | None = None
        self._pixel_affine_offset: Tensor | None = None
        self._pixel_affine_inv: Tensor | None = None
        self._pixel_affine_inv_offset: Tensor | None = None

        if header is not None:
            self.naxis = self._infer_wcs_naxis(header)
            self._pixel_shape = self._infer_pixel_shape(header)
            self._parse_header(header)
            # Check for SIP
            if self._header_has(header, "A_ORDER") or self._header_has(
                header, "B_ORDER"
            ):
                self.sip = SIP(header)

            # Check for TPV (Tangent with PV distortion)
            # SCAMP often uses CTYPE = RA---TAN with PV1_j keywords
            has_pv = bool(PV1_KEYS_SET.intersection(self.wcs_params)) or bool(
                PV2_KEYS_SET.intersection(self.wcs_params)
            )
            is_tpv_ctype = (
                self.wcs_params.get("CTYPE1", "").strip().upper().endswith("TPV")
            )

            if is_tpv_ctype or (
                self.wcs_params.get("CTYPE1", "").strip().upper().endswith("TAN")
                and has_pv
            ):
                self.tpv = TPV(self.wcs_params)
        else:
            self.naxis = int(
                kwargs.get("NAXIS", kwargs.get("WCSAXES", kwargs.get("naxis", 2)))
            )

        # Override with kwargs
        for k, v in kwargs.items():
            k_upper = k.upper()
            if k_upper.startswith("CRVAL"):
                self.wcs_params[k_upper] = float(v)
            elif k_upper.startswith("CRPIX"):
                self.wcs_params[k_upper] = float(v)
            elif k_upper.startswith("CD"):
                self.wcs_params[k_upper] = float(v)
            elif k_upper.startswith("CTYPE"):
                self.wcs_params[k_upper] = str(v)
            else:
                self.wcs_params[k_upper] = v

        # Default device
        self.device = torch.device("cpu")

        # Environment-controlled fused C++ path. Default is ON ("1").
        # Vectorized PyTorch can win at some sizes; set TORCHSKY_WCS_FUSED=0 to compare.
        self._enable_fused_wcs = os.environ.get("TORCHSKY_WCS_FUSED", "1") == "1"

        # Move parameters to appropriate buffers/tensors
        self._setup_tensors()

        if header is not None and tabular_axes is None and tables is not None:
            from .tabular import (
                coupled_tabular_wcs_and_axes_from_fits,
                tabular_axes_from_fits,
            )

            try:
                tabular_axes = tabular_axes_from_fits(header, tables, key=self.wcs_key)
            except ValueError as exc:
                if "coupled multidimensional coordinate array" not in str(exc):
                    raise
                coupled_tabular, coupled_tabular_axes, coupled_tabular_pixel_axes = (
                    coupled_tabular_wcs_and_axes_from_fits(
                        header, tables, key=self.wcs_key
                    )
                )

        self._setup_tabular_axes(tabular_axes)
        self._setup_coupled_tabular(
            coupled_tabular,
            coupled_tabular_axes,
            coupled_tabular_pixel_axes,
        )
        self._validate_required_tabular_axes()

        # Build spectral transforms
        from .spectral import SpectralTransform, canonical_spectral_type

        self.spectral_transforms = {}

        # We need restfrq/restwav if present
        restfrq = self.wcs_params.get("RESTFRQ", self.wcs_params.get("RESTFREQ"))
        restwav = self.wcs_params.get("RESTWAV")

        for axis_idx in range(self.naxis):
            axis_number = axis_idx + 1
            ctype = str(self.wcs_params.get(f"CTYPE{axis_number}", "")).strip().upper()
            try:
                canonical_spectral_type(ctype)
                is_spectral = True
            except ValueError:
                is_spectral = False

            if is_spectral:
                crval = float(self.wcs_params.get(f"CRVAL{axis_number}", 0.0))
                cdelt = float(self.wcs_params.get(f"CDELT{axis_number}", 1.0))
                cunit = self.wcs_params.get(f"CUNIT{axis_number}")

                # Extract PV keywords for this axis
                max_j = -1
                keys_to_search = set()
                if header is not None:
                    keys_to_search.update(header.keys())
                keys_to_search.update(self.wcs_params.keys())

                prefix = f"PV{axis_number}_"
                for k in keys_to_search:
                    if str(k).upper().startswith(prefix):
                        try:
                            j = int(str(k)[len(prefix) :])
                            if j > max_j:
                                max_j = j
                        except ValueError:
                            pass
                pv_list = None
                if max_j >= 0:
                    pv_list = [0.0] * (max_j + 1)
                    for j in range(max_j + 1):
                        pv_key = f"PV{axis_number}_{j}"
                        if pv_key in self.wcs_params:
                            pv_list[j] = float(self.wcs_params[pv_key])
                        elif header is not None and self._header_has(header, pv_key):
                            val = float(self._header_get(header, pv_key))
                            self.wcs_params[pv_key] = val
                            pv_list[j] = val

                self.spectral_transforms[axis_idx] = SpectralTransform(
                    ctype=ctype,
                    crval=crval,
                    cdelt=cdelt,
                    cunit=cunit,
                    restfrq=restfrq,
                    restwav=restwav,
                    pv=pv_list,
                )
        self._compiled_transform_fn = None
        self._compiled_pixel_to_world_2d_fn = None
        self._compiled_world_to_pixel_2d_fn = None
        # Origin -> compiled ``pixels_flat -> world_flat`` for 2D spatial (FITS origin 0/1).
        self._compiled_stack_pixel_to_world: (
            dict[float, Callable[[Tensor], Tensor]] | None
        ) = None
        self._compiled_stack_world_to_pixel: (
            dict[float, Callable[[Tensor], Tensor]] | None
        ) = None

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

    def _parse_pole(self, header: Dict[str, Any]) -> None:
        """Parse LONPOLE and LATPOLE keywords."""
        if self._header_has(header, "LONPOLE"):
            self.wcs_params["LONPOLE"] = float(self._header_get(header, "LONPOLE"))
        if self._header_has(header, "LATPOLE"):
            self.wcs_params["LATPOLE"] = float(self._header_get(header, "LATPOLE"))

    def _parse_pv(self, header: Dict[str, Any]) -> None:
        """Parse PV distortion keywords."""
        # Capture PV keywords (PVi_j)
        # SCAMP/PV distortions can have many terms (0..39 typically)
        for key in PV1_KEYS:
            if self._header_has(header, key):
                self.wcs_params[key] = float(self._header_get(header, key))
        for key in PV2_KEYS:
            if self._header_has(header, key):
                self.wcs_params[key] = float(self._header_get(header, key))

    def _parse_wat(self, header: Dict[str, Any]) -> None:
        """Parse WAT keywords for TNX/ZPX projections."""
        self.wat_data = {}
        has_wat = any(str(k).startswith("WAT") for k in header.keys())
        if has_wat:
            from .legacy import parse_wat_keywords

            self.wat_data[1] = parse_wat_keywords(header, prefix="WAT1")
            self.wat_data[2] = parse_wat_keywords(header, prefix="WAT2")
            if self.wcs_params.get("CTYPE1", "").strip().upper().endswith("ZPX"):
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
        self._parse_cd_pc(header)
        self._parse_pole(header)
        self._parse_pv(header)
        self._parse_wat(header)
        self._parse_sip(header)
        self._parse_tpv()
        self._parse_spectral_metadata(header)
        self._parse_temporal_metadata(header)
        self._parse_celestial_metadata(header)

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

    def _setup_tabular_axes(
        self, tabular_axes: Mapping[int, TabularAxis] | None
    ) -> None:
        if tabular_axes is None:
            self.tabular_axes = {}
            return
        axes: dict[int, TabularAxis] = {}
        for raw_axis, axis in tabular_axes.items():
            if not isinstance(axis, TabularAxis):
                raise TypeError("tabular_axes values must be TabularAxis instances")
            axis_number = int(raw_axis)
            if axis_number < 1 or axis_number > self.naxis:
                raise ValueError(
                    f"tabular axis {axis_number} outside WCS axes 1..{self.naxis}"
                )
            idx = axis_number - 1
            ctype = str(self.wcs_params.get(f"CTYPE{axis_number}", "")).strip().upper()
            if not ctype.endswith("-TAB"):
                raise ValueError(
                    f"tabular axis {axis_number} requires CTYPE{axis_number} ending in '-TAB'"
                )
            axes[idx] = axis.to(device=self.device, dtype=self._wcs_coord_dtype(self.device))
        self.tabular_axes = axes

    def _setup_coupled_tabular(
        self,
        coupled_tabular: CoupledTabularLookupWCS | None,
        coupled_tabular_axes: Sequence[int] | None,
        coupled_tabular_pixel_axes: Sequence[int] | None = None,
    ) -> None:
        if coupled_tabular is None:
            self.coupled_tabular = None
            self.coupled_tabular_axes = ()
            self.coupled_tabular_pixel_axes = ()
            return
        if not isinstance(coupled_tabular, CoupledTabularLookupWCS):
            raise TypeError("coupled_tabular must be a CoupledTabularLookupWCS")
        if coupled_tabular_axes is None:
            coupled_tabular_axes = tuple(range(1, coupled_tabular.world_n_dim + 1))
        world_axes = tuple(int(axis_number) for axis_number in coupled_tabular_axes)
        if coupled_tabular_pixel_axes is None:
            if coupled_tabular.pixel_n_dim == len(world_axes):
                coupled_tabular_pixel_axes = world_axes
            else:
                coupled_tabular_pixel_axes = world_axes[: coupled_tabular.pixel_n_dim]
        pixel_axes = tuple(
            int(axis_number) for axis_number in coupled_tabular_pixel_axes
        )
        if len(world_axes) == 0 or len(pixel_axes) == 0:
            raise ValueError("coupled tabular axis lists must not be empty")
        if len(set(world_axes)) != len(world_axes):
            raise ValueError("coupled_tabular_axes must be unique")
        if len(set(pixel_axes)) != len(pixel_axes):
            raise ValueError("coupled_tabular_pixel_axes must be unique")
        for axis_number in (*world_axes, *pixel_axes):
            if axis_number < 1 or axis_number > self.naxis:
                raise ValueError(
                    f"coupled_tabular axis {axis_number} outside WCS axes 1..{self.naxis}"
                )
            ctype = str(self.wcs_params.get(f"CTYPE{axis_number}", "")).strip().upper()
            if not ctype.endswith("-TAB"):
                raise ValueError(
                    "coupled_tabular axes require corresponding CTYPEi ending in '-TAB'"
                )
        if coupled_tabular.world_n_dim != len(world_axes):
            raise ValueError(
                "coupled_tabular world_n_dim must match coupled_tabular_axes "
                f"({coupled_tabular.world_n_dim} != {len(world_axes)})"
            )
        if coupled_tabular.pixel_n_dim != len(pixel_axes):
            raise ValueError(
                "coupled_tabular pixel_n_dim must match coupled_tabular_pixel_axes "
                f"({coupled_tabular.pixel_n_dim} != {len(pixel_axes)})"
            )
        self.coupled_tabular = coupled_tabular.to(
            device=self.device, dtype=self._wcs_coord_dtype(self.device)
        )
        self.coupled_tabular_axes = tuple(axis - 1 for axis in world_axes)
        self.coupled_tabular_pixel_axes = tuple(axis - 1 for axis in pixel_axes)

    def _required_tabular_axis_numbers(self) -> set[int]:
        axes: set[int] = set()
        for axis_number in range(1, self.naxis + 1):
            ctype = str(self.wcs_params.get(f"CTYPE{axis_number}", "")).strip().upper()
            if ctype.endswith("-TAB"):
                axes.add(axis_number)
        return axes

    def _validate_required_tabular_axes(self) -> None:
        required = self._required_tabular_axis_numbers()
        if not required:
            return
        supplied = {axis + 1 for axis in self.tabular_axes}
        if self.coupled_tabular is not None:
            supplied.update(axis + 1 for axis in self.coupled_tabular_axes)
            supplied.update(axis + 1 for axis in self.coupled_tabular_pixel_axes)
        missing = sorted(required.difference(supplied))
        if missing:
            axes = ", ".join(str(axis) for axis in missing)
            raise ValueError(
                f"FITS -TAB axis/axes {axes} require explicit tabular_axes or tables"
            )

    def _apply_tabular_pixel_to_world(self, linear_world: Tensor, axis: int) -> Tensor:
        tabular = self.tabular_axes.get(axis)
        if tabular is None:
            return self._apply_spectral_algorithm_pixel_to_world(linear_world, axis)
        return tabular.pixel_to_world(linear_world)

    def _apply_coupled_tabular_pixel_to_world(
        self, linear_world: Tensor
    ) -> dict[int, Tensor]:
        if self.coupled_tabular is None:
            return {}
        query = torch.stack(
            [linear_world[:, axis] for axis in self.coupled_tabular_pixel_axes],
            dim=-1,
        )
        coupled_world = self.coupled_tabular.pixel_to_world(query)
        return {
            axis: coupled_world[:, index]
            for index, axis in enumerate(self.coupled_tabular_axes)
        }

    def _apply_tabular_world_to_pixel(self, world: Tensor, axis: int) -> Tensor:
        tabular = self.tabular_axes.get(axis)
        if tabular is None:
            return self._apply_spectral_algorithm_world_to_pixel(world, axis)
        return tabular.world_to_pixel(world)

    def _wcs_coord_dtype(self, device: torch.device) -> torch.dtype:
        # MPS has no float64; CPU/CUDA keep the WCS compute dtype.
        return torch.float32 if device.type == "mps" else torch.float64

    def _as_wcs_coord(self, value: Any) -> Tensor:
        """Coerce pixel/world inputs to the WCS compute dtype.

        float64 on CPU/CUDA. MPS tensors stay float32 on MPS so map resampling
        keeps device-stable outputs; never ``.to(float64)`` from MPS.

        CUDA tensors keep their device even when the WCS object itself was not
        moved with ``.to("cuda")``, so SIP/index ops follow the pixel device.
        CPU tensors still follow ``self.device`` so ``wcs.to("cuda")`` lifts
        host arrays.
        """
        if not isinstance(value, Tensor):
            dtype = self._wcs_coord_dtype(self.device)
            return torch.as_tensor(value, device=self.device, dtype=dtype)
        if value.device.type == "mps":
            return value.to(dtype=torch.float32)
        dtype = self._wcs_coord_dtype(value.device)
        out = value if value.dtype == dtype else value.to(dtype=dtype)
        if out.device.type == "cpu" and self.device.type != "cpu":
            return out.to(device=self.device, dtype=self._wcs_coord_dtype(self.device))
        return out

    def _load_pixel_affine_from_params(self, *, prefix: str = "TSKYAF") -> None:
        """Load optional TSKYAF* pixel affine stored for distorted WCS."""
        marker = self.wcs_params.get(f"{prefix}0")
        if marker != "pixel_affine":
            return
        # Linear undistorted corrections bake into CRPIX/CD; only distorted WCS
        # keep a live pixel-space affine applied before SIP/TPV/TNX/ZPX.
        if (
            self.sip is None
            and not self._is_tpv
            and not self._is_tnx
            and not self._is_zpx
        ):
            return
        try:
            matrix = torch.tensor(
                [
                    [
                        float(self.wcs_params[f"{prefix}A11"]),
                        float(self.wcs_params[f"{prefix}A12"]),
                    ],
                    [
                        float(self.wcs_params[f"{prefix}A21"]),
                        float(self.wcs_params[f"{prefix}A22"]),
                    ],
                ],
                dtype=torch.float64,
                device=self.device,
            )
            offset = torch.tensor(
                [
                    float(self.wcs_params[f"{prefix}B1"]),
                    float(self.wcs_params[f"{prefix}B2"]),
                ],
                dtype=torch.float64,
                device=self.device,
            )
        except (KeyError, TypeError, ValueError):
            return
        self._set_pixel_affine(matrix, offset)

    def _set_pixel_affine(self, matrix: Tensor, offset: Tensor) -> None:
        affine = matrix.to(device=self.device, dtype=self._wcs_coord_dtype(self.device))
        shift = offset.to(device=self.device, dtype=self._wcs_coord_dtype(self.device))
        affine_inv = torch.linalg.inv(affine)
        self._pixel_affine_matrix = affine
        self._pixel_affine_offset = shift
        self._pixel_affine_inv = affine_inv
        self._pixel_affine_inv_offset = -(affine_inv @ shift)

    def _corrected_pixels_to_source(
        self, x: Tensor, y: Tensor, origin: float
    ) -> tuple[Tensor, Tensor]:
        """Map corrected pixels to source pixels: ``p_old = A_inv @ p_new + b``."""
        if self._pixel_affine_inv is None or self._pixel_affine_inv_offset is None:
            return x, y
        origin_f = float(origin)
        x0 = x - origin_f
        y0 = y - origin_f
        stacked = torch.stack((x0, y0), dim=-1)
        old = torch.matmul(stacked, self._pixel_affine_inv.T) + self._pixel_affine_inv_offset
        return old[..., 0] + origin_f, old[..., 1] + origin_f

    def _source_pixels_to_corrected(
        self, x: Tensor, y: Tensor, origin: float
    ) -> tuple[Tensor, Tensor]:
        """Map source pixels to corrected pixels: ``p_new = A @ p_old + offset``."""
        if self._pixel_affine_matrix is None or self._pixel_affine_offset is None:
            return x, y
        origin_f = float(origin)
        x0 = x - origin_f
        y0 = y - origin_f
        stacked = torch.stack((x0, y0), dim=-1)
        new = torch.matmul(stacked, self._pixel_affine_matrix.T) + self._pixel_affine_offset
        return new[..., 0] + origin_f, new[..., 1] + origin_f

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name in ("cpdis1", "cpdis2", "det2im1", "det2im2") and value is not None:
            object.__setattr__(self, "_use_fast_tan", False)
            object.__setattr__(self, "_use_center_equator_fast", False)
            object.__setattr__(self, "_cpp_direct_static_ok", False)
            object.__setattr__(self, "_cpp_p2w_direct_kind", "")
            object.__setattr__(self, "_cpp_w2p_direct_kind", "")
            object.__setattr__(self, "_has_cpp_direct_p2w", False)
            object.__setattr__(self, "_has_cpp_direct_w2p", False)
            object.__setattr__(self, "_can_use_fused_p2w_static", False)
            object.__setattr__(self, "_can_use_fused_w2p_static", False)
            object.__setattr__(self, "_compiled_pixel_to_world_2d_fn", None)

    def _has_paper4_lut(self) -> bool:
        return (
            self.cpdis1 is not None
            or self.cpdis2 is not None
            or self.det2im1 is not None
            or self.det2im2 is not None
        )

    def _apply_det2im_forward(
        self, x1: Tensor, y1: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.det2im1 is None and self.det2im2 is None:
            return x1, y1
        lut1 = _coerce_distortion_lookup_table(self.det2im1)
        lut2 = _coerce_distortion_lookup_table(self.det2im2)
        dx = lut1.eval_tensor(x1, y1) if lut1 is not None else torch.zeros_like(x1)
        dy = lut2.eval_tensor(x1, y1) if lut2 is not None else torch.zeros_like(y1)
        return x1 + dx, y1 + dy

    def _apply_det2im_inverse(
        self, x_d2im: Tensor, y_d2im: Tensor, *, max_iter: int = 16
    ) -> tuple[Tensor, Tensor]:
        if self.det2im1 is None and self.det2im2 is None:
            return x_d2im, y_d2im
        lut1 = _coerce_distortion_lookup_table(self.det2im1)
        lut2 = _coerce_distortion_lookup_table(self.det2im2)
        x_curr = x_d2im
        y_curr = y_d2im
        for _ in range(max_iter):
            dx = lut1.eval_tensor(x_curr, y_curr) if lut1 is not None else 0.0
            dy = lut2.eval_tensor(x_curr, y_curr) if lut2 is not None else 0.0
            x_curr = x_d2im - dx
            y_curr = y_d2im - dy
        return x_curr, y_curr

    def _apply_cpdis_forward(
        self, rel_x: Tensor, rel_y: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.cpdis1 is None and self.cpdis2 is None:
            return rel_x, rel_y
        lut1 = _coerce_distortion_lookup_table(self.cpdis1)
        lut2 = _coerce_distortion_lookup_table(self.cpdis2)
        x_sip = rel_x + self._crpix0
        y_sip = rel_y + self._crpix1
        dx = lut1.eval_tensor(x_sip, y_sip) if lut1 is not None else 0.0
        dy = lut2.eval_tensor(x_sip, y_sip) if lut2 is not None else 0.0
        return rel_x + dx, rel_y + dy

    def _apply_cpdis_inverse(
        self, rel_x: Tensor, rel_y: Tensor, *, max_iter: int = 16
    ) -> tuple[Tensor, Tensor]:
        if self.cpdis1 is None and self.cpdis2 is None:
            return rel_x, rel_y
        lut1 = _coerce_distortion_lookup_table(self.cpdis1)
        lut2 = _coerce_distortion_lookup_table(self.cpdis2)
        x_target = rel_x + self._crpix0
        y_target = rel_y + self._crpix1
        x_curr = x_target
        y_curr = y_target
        for _ in range(max_iter):
            dx = lut1.eval_tensor(x_curr, y_curr) if lut1 is not None else 0.0
            dy = lut2.eval_tensor(x_curr, y_curr) if lut2 is not None else 0.0
            x_curr = x_target - dx
            y_curr = y_target - dy
        return x_curr - self._crpix0, y_curr - self._crpix1

    def _gls_forward(self, xi: Tensor, eta: Tensor) -> tuple[Tensor, Tensor]:
        dec = eta + self._gls_delta0
        valid = torch.abs(dec) <= 90.0
        cos_dec = torch.cos(dec * self._d2r)
        cos_ok = torch.abs(cos_dec) > 1e-12
        cos_safe = torch.where(cos_ok, cos_dec, torch.ones_like(cos_dec))
        phi = xi / cos_safe
        valid = valid & cos_ok & (torch.abs(phi) <= 180.0 + 1e-9)
        ra = _wrap_lon360(phi + self.alpha0)
        if bool(valid.all()):
            return ra, dec
        nan = torch.full_like(ra, float("nan"))
        return torch.where(valid, ra, nan), torch.where(valid, dec, nan)

    def _gls_inverse(self, ra: Tensor, dec: Tensor) -> tuple[Tensor, Tensor]:
        eta = dec - self._gls_delta0
        xi = _wrap_lon180(ra - self.alpha0) * torch.cos(dec * self._d2r)
        return xi, eta

    def _invert_extra_axes(
        self, world_flat: Tensor, pix_internal_spatial: Tensor
    ) -> Tensor:
        """Internal extra-axis pixels from spatial internal pixels and world.

        Solves the extra CD block
        ``extra = crval_e + CD_e,s @ rel_s + CD_e,e @ rel_e``
        after inverting spectral/tabular algorithms on extra world axes.
        """
        coupled_pixel = self._apply_coupled_tabular_world_to_pixel(world_flat)
        extra_linear = torch.stack(
            [
                self._coupled_or_tabular_linear_world(coupled_pixel, world_flat, i)
                for i in range(2, self.naxis)
            ],
            dim=-1,
        )
        rel_spatial = pix_internal_spatial - self.crpix[:2]
        cd_extra_spatial = self.cd_full[2:, :2]
        cd_extra = self.cd_full[2:, 2:]
        rhs = (
            extra_linear
            - self.crval[2:]
            - torch.matmul(rel_spatial, cd_extra_spatial.T)
        )
        rel_extra = torch.linalg.solve(cd_extra, rhs.T).T
        return rel_extra + self.crpix[2:]

    def _coupled_or_tabular_linear_world(
        self,
        coupled_pixel: dict[int, Tensor],
        world: Tensor,
        axis: int,
    ) -> Tensor:
        if axis in coupled_pixel:
            return coupled_pixel[axis]
        if axis in self.coupled_tabular_axes:
            return (
                self.crval[axis]
                .to(device=world.device, dtype=world.dtype)
                .expand(world.shape[0])
            )
        return self._apply_tabular_world_to_pixel(world[:, axis], axis)

    def _apply_coupled_tabular_world_to_pixel(self, world: Tensor) -> dict[int, Tensor]:
        if self.coupled_tabular is None:
            return {}
        query = torch.stack(
            [world[:, axis] for axis in self.coupled_tabular_axes], dim=-1
        )
        coupled_pixel = self.coupled_tabular.world_to_pixel(query)
        return {
            axis: coupled_pixel[:, index]
            for index, axis in enumerate(self.coupled_tabular_pixel_axes)
        }

    def _axis_algorithm_code(self, axis: int) -> str:
        ctype = str(self.wcs_params.get(f"CTYPE{axis + 1}", "")).strip().upper()
        if "-" not in ctype:
            return ""
        return ctype.rsplit("-", 1)[1]

    def _apply_spectral_algorithm_pixel_to_world(
        self, linear_world: Tensor, axis: int
    ) -> Tensor:
        if hasattr(self, "spectral_transforms") and axis in self.spectral_transforms:
            crval = self.crval[axis].to(
                device=linear_world.device, dtype=linear_world.dtype
            )
            x = linear_world - crval
            return self.spectral_transforms[axis].pixel_to_world(x)

        algorithm = self._axis_algorithm_code(axis)
        if algorithm not in {"LOG", "F2W", "W2F"}:
            return linear_world
        crval = self.crval[axis].to(
            device=linear_world.device, dtype=linear_world.dtype
        )
        if torch.abs(crval) <= 0.0:
            return torch.full_like(linear_world, torch.nan)
        delta = linear_world - crval
        if algorithm == "LOG":
            return crval * torch.exp(delta / crval)
        denom = 1.0 - delta / crval
        safe_denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
        reciprocal = crval / safe_denom
        return torch.where(
            denom != 0.0, reciprocal, torch.full_like(linear_world, torch.nan)
        )

    def _apply_spectral_algorithm_world_to_pixel(
        self, world: Tensor, axis: int
    ) -> Tensor:
        if hasattr(self, "spectral_transforms") and axis in self.spectral_transforms:
            crval = self.crval[axis].to(device=world.device, dtype=world.dtype)
            x = self.spectral_transforms[axis].world_to_pixel(world)
            return crval + x

        algorithm = self._axis_algorithm_code(axis)
        if algorithm not in {"LOG", "F2W", "W2F"}:
            return world
        crval = self.crval[axis].to(device=world.device, dtype=world.dtype)
        if torch.abs(crval) <= 0.0:
            return torch.full_like(world, torch.nan)
        ratio = world / crval
        if algorithm == "LOG":
            linear = crval + crval * torch.log(ratio)
            return torch.where(ratio > 0.0, linear, torch.full_like(linear, torch.nan))
        valid = world != 0.0
        safe_world = torch.where(valid, world, torch.ones_like(world))
        linear = crval + crval * (1.0 - crval / safe_world)
        return torch.where(valid, linear, torch.full_like(linear, torch.nan))

    def _setup_tensors(self):
        """Convert scalar params to PyTorch tensors for computation."""
        self.crpix = torch.tensor(
            [self.wcs_params.get(f"CRPIX{i}", 0.0) for i in range(1, self.naxis + 1)],
            dtype=torch.float64,
            device=self.device,
        )

        self.crval = torch.tensor(
            [self.wcs_params.get(f"CRVAL{i}", 0.0) for i in range(1, self.naxis + 1)],
            dtype=torch.float64,
            device=self.device,
        )

        self.cdelt = torch.tensor(
            [self.wcs_params.get(f"CDELT{i}", 1.0) for i in range(1, self.naxis + 1)],
            dtype=torch.float64,
            device=self.device,
        )

        # 1b. CD matrix initialization (Paper I)
        # Check if CD keywords exist
        has_cd = any(
            f"CD{i}_{j}" in self.wcs_params
            for i in range(1, self.naxis + 1)
            for j in range(1, self.naxis + 1)
        )

        if has_cd:
            # Construct CD matrix
            cd_data = []
            for i in range(1, self.naxis + 1):
                row = []
                for j in range(1, self.naxis + 1):
                    row.append(
                        self.wcs_params.get(f"CD{i}_{j}", 1.0 if i == j else 0.0)
                    )
                cd_data.append(row)
            self.cd_full = torch.tensor(
                cd_data, dtype=self._wcs_coord_dtype(self.device), device=self.device
            )
        else:
            # Construct from PC and CDELT: CDi_j = CDELTi * PCi_j
            cd_data = []
            for i in range(1, self.naxis + 1):
                row = []
                for j in range(1, self.naxis + 1):
                    pc_val = self.wcs_params.get(f"PC{i}_{j}", 1.0 if i == j else 0.0)
                    row.append(self.cdelt[i - 1].item() * pc_val)
                cd_data.append(row)
            self.cd_full = torch.tensor(
                cd_data, dtype=self._wcs_coord_dtype(self.device), device=self.device
            )

        # Spatial shortcut for old code paths (2x2)
        if self.naxis >= 2:
            self.cd = self.cd_full[:2, :2]
        else:
            # 1x1 padded to 2x2 for safety in spatial paths
            self.cd = torch.eye(2, dtype=self._wcs_coord_dtype(self.device), device=self.device)
            self.cd[0, 0] = self.cd_full[0, 0]

        # Precompute inverse CD matrix for world->pixel
        self.cd_inv_full = torch.inverse(self.cd_full)
        self.cd_inv = torch.inverse(self.cd)
        self._crpix0 = float(self.crpix[0].item()) if self.crpix.numel() >= 1 else 0.0
        self._crpix1 = float(self.crpix[1].item()) if self.crpix.numel() >= 2 else 0.0
        self._cd00 = float(self.cd[0, 0].item())
        self._cd01 = float(self.cd[0, 1].item())
        self._cd10 = float(self.cd[1, 0].item())
        self._cd11 = float(self.cd[1, 1].item())
        self._cd_is_diag = abs(self._cd01) < 1e-15 and abs(self._cd10) < 1e-15
        self._cdi00 = float(self.cd_inv[0, 0].item())
        self._cdi01 = float(self.cd_inv[0, 1].item())
        self._cdi10 = float(self.cd_inv[1, 0].item())
        self._cdi11 = float(self.cd_inv[1, 1].item())

        # 2. Rotation Metadata (Calabretta & Greisen 2002)
        self.alpha0 = float(self.wcs_params.get("CRVAL1", 0.0))
        self.delta0 = float(self.wcs_params.get("CRVAL2", 0.0))

        # Precompute Projection Meta
        ctype1 = str(self.wcs_params.get("CTYPE1", "RA---TAN")).strip().upper()
        ctype1_base = ctype1[:-4] if ctype1.endswith("-SIP") else ctype1
        self._proj_code = ctype1_base[-3:]
        self._is_tnx = self._proj_code == "TNX"
        self._is_zpx = self._proj_code == "ZPX"
        self._is_tpv = self.tpv is not None or self._proj_code == "TPV"
        self._load_pixel_affine_from_params()

        self._theta0 = 90.0  # Zenithal
        if self._proj_code == "XPH":
            # HEALPix polar / butterfly: native pole at reference (Paper XPH).
            self._theta0 = 90.0
        elif self._proj_code in (
            "AIT",
            "MOL",
            "HPX",
            "CEA",
            "MER",
            "CYP",
            "CAR",
            "SFL",
            "PAR",
            "PCO",
        ) or self._proj_code in self.QUADCUBE_CODES:
            self._theta0 = 0.0
        elif (
            self._proj_code in self.CONIC_CODES
            or self._proj_code == "BON"
        ):
            # Conic and Bonne use PV2_1 (standard parallel) as theta_0.
            self._theta0 = float(self.wcs_params.get("PV2_1", 0.0))
            # Bonne with theta_1=0 degenerates to Sanson-Flamsteed (equatorial)
            if self._proj_code == "BON" and abs(self._theta0) < 1e-12:
                self._theta0 = 0.0

        # LONPOLE / LATPOLE
        self.phi_p = float(
            self.wcs_params.get("LONPOLE", 180.0 if self._theta0 == 90.0 else 0.0)
        )
        self.theta_p = float(self.wcs_params.get("LATPOLE", 90.0))

        d2r = math.pi / 180.0
        self._d2r = d2r
        self._r2d = 180.0 / math.pi
        self._sin_alpha0 = math.sin(self.alpha0 * d2r)
        self._cos_alpha0 = math.cos(self.alpha0 * d2r)
        self._sin_delta0 = math.sin(self.delta0 * d2r)
        self._cos_delta0 = math.cos(self.delta0 * d2r)
        self._east_x = -self._sin_alpha0
        self._east_y = self._cos_alpha0
        self._north_x = -self._sin_delta0 * self._cos_alpha0
        self._north_y = -self._sin_delta0 * self._sin_alpha0
        self._north_z = self._cos_delta0
        self._radial_x = self._cos_delta0 * self._cos_alpha0
        self._radial_y = self._cos_delta0 * self._sin_alpha0
        self._radial_z = self._sin_delta0

        # Eq 8 in Paper II: Calculate delta_p
        # Simplified for common cases.
        # For Zenithal, delta_p = delta0.
        # For Cylindrical, delta_p = 90.
        if self._theta0 == 90.0:
            self.delta_p = self.delta0
            self.alpha_p = self.alpha0
            self._native_type = "pole"
        elif self._proj_code in self.CONIC_CODES:
            # Conic: theta_0 = PV2_1, phi_0 = 0.
            # For pole-type rotation with phi_p=0, the reference (phi=0, theta=theta_0)
            # must map to (alpha_0, delta_0). Two solutions exist; the one with
            # correct d(dec)/d(theta) = +1 sign is:
            #   delta_p = theta_0 + 90 - delta_0,  alpha_p = alpha_0 + 180
            self.delta_p = self._theta0 + 90.0 - self.delta0
            self.alpha_p = self.alpha0 + 180.0
            self._native_type = "pole"
        elif self._proj_code == "BON":
            # Bonne: theta_0 = PV2_1, phi_0 = 0. Use "center" type since the
            # projection's y-offset places intermediate (0,0) at native (0,0),
            # matching the center-equator rotation when delta_0 ≈ 0.
            self.delta_p = 90.0
            self.alpha_p = self.alpha0 + 180.0 if self.phi_p == 0.0 else self.alpha0
            self._native_type = "center"
        elif self._proj_code in ("PCO",) or self._proj_code in self.QUADCUBE_CODES:
            self.delta_p = 90.0
            self.alpha_p = self.alpha0 + 180.0 if self.phi_p == 0.0 else self.alpha0
            self._native_type = "center"
        else:
            self.delta_p = 90.0
            self.alpha_p = self.alpha0 + 180.0 if self.phi_p == 0.0 else self.alpha0
            self._native_type = "center"

        # Precompute Tensors
        self._ra_p_rad = self.alpha_p * d2r
        self._dec_p_rad = self.delta_p * d2r
        self._phi_p_rad = self.phi_p * d2r
        dec_p_tensor = torch.as_tensor(
            self._dec_p_rad, dtype=self._wcs_coord_dtype(self.device), device=self.device
        )
        self._sin_dec_p = torch.sin(dec_p_tensor)
        self._cos_dec_p = torch.cos(dec_p_tensor)
        self._sin_alpha_p = math.sin(self._ra_p_rad)
        self._cos_alpha_p = math.cos(self._ra_p_rad)
        self._sin_delta_p = math.sin(self._dec_p_rad)
        self._cos_delta_p = math.cos(self._dec_p_rad)
        self._pole_east_x = -self._sin_alpha_p
        self._pole_east_y = self._cos_alpha_p
        self._pole_north_x = -self._sin_delta_p * self._cos_alpha_p
        self._pole_north_y = -self._sin_delta_p * self._sin_alpha_p
        self._pole_north_z = self._cos_delta_p
        self._pole_radial_x = self._cos_delta_p * self._cos_alpha_p
        self._pole_radial_y = self._cos_delta_p * self._sin_alpha_p
        self._pole_radial_z = self._sin_delta_p

        # Pre-calculate 3x3 rotation matrix from native to celestial
        # Columns are the celestial basis vectors in the native frame?
        # No, celestial = R * native.
        # If native basis is (e_nat, n_nat, r_nat), then V_cel = u*e_cel + v*n_cel + w*r_cel.
        # where (e_cel, n_cel, r_cel) are the basis vectors of celestial frame at CRVAL.

        # Basis vectors at alpha_p, delta_p
        ap_rad = self._ra_p_rad
        dp_rad = self._dec_p_rad
        sin_ap, cos_ap = math.sin(ap_rad), math.cos(ap_rad)
        sin_dp, cos_dp = math.sin(dp_rad), math.cos(dp_rad)

        # Celestial basis at (alpha_p, delta_p)
        # ex, ey, ez
        ex, ey, ez = -sin_ap, cos_ap, 0.0
        # nx, ny, nz
        nx, ny, nz = -sin_dp * cos_ap, -sin_dp * sin_ap, cos_dp
        # rx, ry, rz
        rx, ry, rz = cos_dp * cos_ap, cos_dp * sin_ap, sin_dp

        # The native basis at phi_p=180 matches the FITS standard rotation.
        # R_cel = R_z(-alpha_p) R_y(delta_p - 90) R_z(180 - phi_p) ?
        # Let's use the explicit basis construction from _spherical_rotation.
        # For 'pole' type: u->ex, v->nx, w->rx
        self._rot_matrix = torch.tensor(
            [[ex, nx, rx], [ey, ny, ry], [ez, nz, rz]],
            dtype=torch.float64,
            device=self.device,
        )
        self._rot_matrix_inv = self._rot_matrix.T  # Rotation matrix is orthogonal

        # Precompute center-native fast path
        self._use_center_fast = (
            self._native_type == "center"
            and abs(self.crval[0].item()) < 1e-9
            and abs(self.crval[1].item()) < 1e-9
        )

        # Precompute CEA/MER constants
        pv2_1 = self.wcs_params.get("PV2_1", 1.0)
        if pv2_1 == 0.0:
            pv2_1 = 1.0
        self._cea_eta_scale = 57.29577951308232 / pv2_1
        self._sqrt2 = math.sqrt(2.0)
        self._mol_abs_eta_max = self._sqrt2 * self._r2d
        # HPX parameters: only read PV1/PV2 when projection is HPX, so that
        # AZP/SZP/AIR/conic PV2_1 (mu, theta_b, standard parallel) doesn't
        # clobber the HPX H/K defaults.
        if self._proj_code == "HPX":
            self._hpx_h = float(
                self.wcs_params.get("PV1_1", self.wcs_params.get("PV2_1", 4.0))
            )
            self._hpx_k = float(
                self.wcs_params.get("PV1_2", self.wcs_params.get("PV2_2", 3.0))
            )
        else:
            self._hpx_h = 4.0
            self._hpx_k = 3.0
        if abs(self._hpx_h) < 1e-12:
            self._hpx_h = 4.0
        if abs(self._hpx_k) < 1e-12:
            self._hpx_k = 3.0
        self._hpx_eta_scale = 90.0 * (self._hpx_k / self._hpx_h)
        if abs(self._hpx_eta_scale) < 1e-12:
            self._hpx_eta_scale = 67.5
        self._hpx_inv_eta_scale = 1.0 / self._hpx_eta_scale
        self._hpx_eta_boundary = self._hpx_eta_scale * (2.0 / 3.0)
        self._hpx_eta_pole = 90.0
        self._hpx_polar_denom = self._hpx_eta_pole - self._hpx_eta_boundary
        self._hpx_inv_polar_denom = 1.0 / self._hpx_polar_denom

        # Precompute fused arguments to avoid overhead
        self._fused_proj_code = "TAN" if self._is_tpv else self._proj_code
        self._fused_pv2_1 = self.wcs_params.get("PV2_1", 1.0)
        self._fused_tpv_idx1 = (
            self.tpv.idx1
            if self._is_tpv
            else torch.empty((0, 3), dtype=torch.long, device="cpu")
        )
        self._fused_tpv_c1 = (
            self.tpv.c1
            if self._is_tpv
            else torch.empty(0, dtype=torch.float64, device="cpu")
        )
        self._fused_tpv_idx2 = (
            self.tpv.idx2
            if self._is_tpv
            else torch.empty((0, 3), dtype=torch.long, device="cpu")
        )
        self._fused_tpv_c2 = (
            self.tpv.c2
            if self._is_tpv
            else torch.empty(0, dtype=torch.float64, device="cpu")
        )

        # Apply projection-specific PV defaults before tensorization.
        # The pre-tensorized PV arrays fill missing keys with 0.0, but some
        # projections have non-zero defaults (e.g. SZP theta_c = 90°).
        if self._proj_code == "SZP" and "PV2_3" not in self.wcs_params:
            self.wcs_params["PV2_3"] = 90.0

        # NCP (obsolete AIPS) ≡ slant SIN with PV2_1=0, PV2_2=cot(δ₀)
        # (Paper II §6.1.2). Must set PV before project; δ₀ = CRVAL2.
        if self._proj_code == "NCP":
            d2r_ncp = math.pi / 180.0
            sin_d0 = math.sin(self.delta0 * d2r_ncp)
            cos_d0 = math.cos(self.delta0 * d2r_ncp)
            if abs(sin_d0) < 1e-15:
                raise ValueError(
                    "NCP requires non-zero CRVAL2 (δ₀); cot(δ₀) is undefined at the equator"
                )
            # Paper II: PV2_1 = 0, PV2_2 = cot(δ₀) = cos/sin
            if "PV2_1" not in self.wcs_params:
                self.wcs_params["PV2_1"] = 0.0
            if "PV2_2" not in self.wcs_params:
                self.wcs_params["PV2_2"] = cos_d0 / sin_d0
            self._proj_code = "SIN"
            self._fused_proj_code = "SIN"

        # Pre-tensorize PV parameters to avoid dict lookups in compiled hot paths.
        pv1_data = [self.wcs_params.get(k, 0.0) for k in PV1_KEYS]
        pv2_data = [self.wcs_params.get(k, 0.0) for k in PV2_KEYS]
        self._pv1_tensor = torch.tensor(
            pv1_data, dtype=self._wcs_coord_dtype(self.device), device=self.device
        )
        self._pv2_tensor = torch.tensor(
            pv2_data, dtype=self._wcs_coord_dtype(self.device), device=self.device
        )
        zpn_all = self._pv2_tensor[:31].detach().cpu()
        zpn_valid = torch.nonzero(zpn_all.abs() > 1e-15, as_tuple=False)
        if zpn_valid.numel() > 0:
            zpn_last = int(zpn_valid[-1].item())
            self._zpn_coeffs_cpu = zpn_all[: zpn_last + 1].contiguous()
        else:
            # Default ZPN behavior (no explicit polynomial terms): r = w.
            # Keep a linear coefficient vector so C++ direct ZPN path remains enabled.
            if self._proj_code == "ZPN":
                self._zpn_coeffs_cpu = torch.tensor([0.0, 1.0], dtype=torch.float64)
            else:
                self._zpn_coeffs_cpu = torch.empty((0,), dtype=torch.float64)

        # Precompute projection flags for compile-friendly dispatch.
        self._is_tan = self._proj_code == "TAN"
        self._is_sin = self._proj_code == "SIN"
        self._is_arc = self._proj_code == "ARC"
        self._is_zea = self._proj_code == "ZEA"
        self._is_stg = self._proj_code == "STG"
        self._is_zpn = self._proj_code == "ZPN"
        self._is_sin_slant = self._is_sin and (
            abs(float(self._pv2_tensor[1].detach().cpu().item())) > 1e-15
            or abs(float(self._pv2_tensor[2].detach().cpu().item())) > 1e-15
        )
        self._is_zpn_linear = False
        if self._is_zpn:
            if self._zpn_coeffs_cpu.numel() == 0:
                self._is_zpn_linear = True
            elif self._zpn_coeffs_cpu.numel() == 2:
                c0 = float(self._zpn_coeffs_cpu[0].item())
                c1 = float(self._zpn_coeffs_cpu[1].item())
                self._is_zpn_linear = abs(c0) <= 1e-15 and abs(c1 - 1.0) <= 1e-15
        self._is_ait = self._proj_code == "AIT"
        self._is_mol = self._proj_code == "MOL"
        self._is_hpx = self._proj_code == "HPX"
        self._is_cea = self._proj_code == "CEA"
        self._is_mer = self._proj_code == "MER"
        self._is_cyp = self._proj_code == "CYP"
        self._is_car = self._proj_code == "CAR"
        self._is_sfl = self._proj_code == "SFL"
        self._is_gls = self._proj_code == "GLS" or (
            self._proj_code == "SFL"
            and abs(float(self.wcs_params.get("PV1_0", 0.0)) - 1.0) <= 1e-12
            and "PV1_2" in self.wcs_params
        )
        self._gls_delta0 = (
            float(self.wcs_params["PV1_2"])
            if (self._proj_code == "SFL" and "PV1_2" in self.wcs_params)
            else self.delta0
        )

        # Detect if fast TAN is possible
        self._use_fast_tan = (
            self._proj_code == "TAN"
            and not self._is_tpv
            and not self._is_tnx
            and not self._is_zpx
            and self.sip is None
            and not self._has_paper4_lut()
            and self.alpha0 == self.alpha_p
            and self.delta0 == self.delta_p
        )
        self._is_spatial = self._proj_code in (
            self.ZENITHAL_CODES
            + self.CYLINDRICAL_CODES
            + self.ALLSKY_CODES
            + self.CONIC_CODES
            + self.POLYCONIC_CODES
            + self.QUADCUBE_CODES
            + self.PSEUDOCYLINDRICAL_CODES
            + self.SPECIAL_CODES
        )
        self._use_center_equator_fast = (
            self._native_type == "center"
            and not self._is_gls
            and not self._has_paper4_lut()
            and abs(self.delta0) < 1e-12
            and abs((self.phi_p % 360.0)) < 1e-12
        )
        self._cpp_zenithal_project = (
            getattr(_cpp, "wcs_zenithal_project", None) if _cpp is not None else None
        )
        self._cpp_zpn_project = (
            getattr(_cpp, "wcs_zpn_project", None) if _cpp is not None else None
        )
        self._cpp_zenithal_deproject = (
            getattr(_cpp, "wcs_zenithal_deproject", None) if _cpp is not None else None
        )
        self._cpp_cylindrical_project = (
            getattr(_cpp, "wcs_cylindrical_project", None) if _cpp is not None else None
        )
        self._cpp_cylindrical_deproject = (
            getattr(_cpp, "wcs_cylindrical_deproject", None)
            if _cpp is not None
            else None
        )
        self._cpp_center_pixel_to_world = (
            getattr(_cpp, "wcs_center_pixel_to_world", None)
            if _cpp is not None
            else None
        )
        self._cpp_center_world_to_pixel = (
            getattr(_cpp, "wcs_center_world_to_pixel", None)
            if _cpp is not None
            else None
        )
        self._cpp_ait_project = (
            getattr(_cpp, "wcs_ait_project", None) if _cpp is not None else None
        )
        self._cpp_ait_deproject = (
            getattr(_cpp, "wcs_ait_deproject", None) if _cpp is not None else None
        )
        self._cpp_mol_project = (
            getattr(_cpp, "wcs_mol_project", None) if _cpp is not None else None
        )
        self._cpp_mol_deproject = (
            getattr(_cpp, "wcs_mol_deproject", None) if _cpp is not None else None
        )
        self._cpp_hpx_project = (
            getattr(_cpp, "wcs_hpx_project", None) if _cpp is not None else None
        )
        self._cpp_hpx_deproject = (
            getattr(_cpp, "wcs_hpx_deproject", None) if _cpp is not None else None
        )
        self._cpp_inverse_spherical_rotation_pole = (
            getattr(_cpp, "wcs_inverse_spherical_rotation_pole", None)
            if _cpp is not None
            else None
        )
        self._cpp_spherical_rotation_pole = (
            getattr(_cpp, "wcs_spherical_rotation_pole", None)
            if _cpp is not None
            else None
        )
        self._cpp_tan_intermediate_from_radec = (
            getattr(_cpp, "wcs_tan_intermediate_from_radec", None)
            if _cpp is not None
            else None
        )
        self._cpp_direct_static_ok = (
            self.sip is None
            and not self._is_tpv
            and not self._is_tnx
            and not self._is_zpx
            and not self._is_gls
            and not self._has_paper4_lut()
        )
        self._cpp_p2w_direct_kind = ""
        if self._cpp_direct_static_ok:
            if self._is_ait and self._cpp_ait_project is not None:
                self._cpp_p2w_direct_kind = "ait"
            elif self._is_mol and self._cpp_mol_project is not None:
                self._cpp_p2w_direct_kind = "mol"
            elif self._is_hpx and self._cpp_hpx_project is not None:
                self._cpp_p2w_direct_kind = "hpx"
            elif (
                self._cpp_zenithal_project is not None
                and not self._is_tpv
                and not self._is_sin_slant
                and (
                    self._is_tan
                    or self._is_sin
                    or self._is_arc
                    or self._is_zea
                    or self._is_stg
                )
            ):
                self._cpp_p2w_direct_kind = "zenithal"
            elif (
                self._is_zpn
                and not self._is_zpn_linear
                and self._cpp_zpn_project is not None
                and self._zpn_coeffs_cpu.numel() > 0
            ):
                self._cpp_p2w_direct_kind = "zpn"
        self._cpp_w2p_direct_kind = ""
        if self._cpp_direct_static_ok:
            if self._is_ait and self._cpp_ait_deproject is not None:
                self._cpp_w2p_direct_kind = "ait"
            elif self._is_hpx and self._cpp_hpx_deproject is not None:
                self._cpp_w2p_direct_kind = "hpx"
            elif (
                self._cpp_zenithal_deproject is not None
                and not self._is_tpv
                and not self._is_sin_slant
                and (
                    self._is_tan
                    or self._is_sin
                    or self._is_arc
                    or self._is_zea
                    or self._is_stg
                )
            ):
                self._cpp_w2p_direct_kind = "zenithal"
        self._has_cpp_direct_p2w = bool(self._cpp_p2w_direct_kind)
        self._has_cpp_direct_w2p = bool(self._cpp_w2p_direct_kind)
        self._can_use_fused_p2w_static = (
            self._enable_fused_wcs
            and _cpp is not None
            and hasattr(_cpp, "wcs_pixel_to_world_fused_cpu")
            and self.sip is None
            and not self._is_tnx
            and not self._is_zpx
            and not self._is_gls
            and not self._has_paper4_lut()
            and not self._is_sin_slant
            and (
                self._proj_code
                in (
                    "TAN",
                    "SIN",
                    "ARC",
                    "CEA",
                    "AIT",
                    "HPX",
                    "CAR",
                    "SFL",
                    "MOL",
                    "PCO",
                    "XPH",
                    "TSC",
                    "CSC",
                    "QSC",
                    "TPV",
                )
                or (self._is_tpv and self._proj_code == "TAN")
            )
        )
        self._can_use_fused_w2p_static = (
            self._enable_fused_wcs
            and _cpp is not None
            and hasattr(_cpp, "wcs_world_to_pixel_fused_cpu")
            and self.sip is None
            and not self._is_tnx
            and not self._is_zpx
            and not self._is_gls
            and not self._has_paper4_lut()
            and not self._is_sin_slant
            and (not self._is_tpv or not self.tpv._invert_trace_enabled)
            and (
                self._proj_code
                in (
                    "TAN",
                    "SIN",
                    "ARC",
                    "CEA",
                    "AIT",
                    "HPX",
                    "CAR",
                    "SFL",
                    "MOL",
                    "PCO",
                    "XPH",
                    "TSC",
                    "CSC",
                    "QSC",
                    "TPV",
                )
                or (self._is_tpv and self._proj_code == "TAN")
            )
        )
        self._disable_cpp_pixel_to_world = False
        self._disable_cpp_world_to_pixel = False
        self._disable_fused_pixel_to_world = False
        self._disable_fused_world_to_pixel = False
        self._cpp_direct_p2w_verified = False
        self._cpp_direct_w2p_verified = False
        self._fused_p2w_verified = False
        self._fused_w2p_verified = False
        self._setup_projection_dispatch()

    def _setup_projection_dispatch(self) -> None:
        """Bind pixel->world projection dispatch once per WCS instance."""
        if self._is_zpx:
            self._project_native_fn = None
            self._project_native_kind = "zpx"
            self._project_native_type = "pole"
            return
        if self._is_tnx:
            self._project_native_fn = None
            self._project_native_kind = "tnx"
            self._project_native_type = "pole"
            return
        if self._is_tpv:
            self._project_native_fn = project_zenithal
            self._project_native_kind = "tpv_tan"
            self._project_native_type = "pole"
            return
        if self._proj_code in self.ZENITHAL_CODES:
            self._project_native_fn = project_zenithal
            self._project_native_kind = "zenithal"
            self._project_native_type = "pole"
            return
        if self._proj_code in self.CYLINDRICAL_CODES:
            self._project_native_fn = project_cylindrical
            self._project_native_kind = "cylindrical"
            self._project_native_type = "center"
            return
        if self._proj_code in self.ALLSKY_CODES or self._proj_code == "PAR":
            self._project_native_fn = project_allsky
            self._project_native_kind = "allsky"
            # XPH is polar (theta0=90); other all-sky codes are equatorial.
            self._project_native_type = "pole" if self._proj_code == "XPH" else "center"
            return
        if self._proj_code in self.CONIC_CODES:
            self._project_native_fn = project_conic
            self._project_native_kind = "conic"
            self._project_native_type = "pole"
            return
        if self._proj_code in self.POLYCONIC_CODES:
            self._project_native_fn = project_conic
            self._project_native_kind = "conic"
            self._project_native_type = "center"
            return
        if self._proj_code in self.QUADCUBE_CODES:
            self._project_native_fn = project_quadcube
            self._project_native_kind = "quadcube"
            self._project_native_type = "center"
            return
        self._project_native_fn = None
        self._project_native_kind = "linear"
        self._project_native_type = "pole"

    def _project_native_coords(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """Projection dispatch hot path with pre-bound family metadata."""
        kind = self._project_native_kind
        if kind == "zpx":
            return project_zpx(xi, eta, self.wcs_params, self.wat_data)
        if kind == "tnx":
            xi_t, eta_t = project_tnx(xi, eta, self.wcs_params, self.wat_data)
            return project_zenithal(xi_t, eta_t, "TAN")
        if kind == "tpv_tan":
            return project_zenithal(xi, eta, "TAN")
        fn = self._project_native_fn
        if fn is None:
            return xi, eta
        if kind == "quadcube":
            return fn(xi, eta, self._proj_code)
        # Use pre-tensorized PV parameters for modern projections
        return fn(xi, eta, self._proj_code, self._pv1_tensor, self._pv2_tensor)

    def _apply_cd_2d(self, rel_x: Tensor, rel_y: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply cached 2x2 CD transform with diagonal fast path."""
        if self._cd_is_diag:
            return self._cd00 * rel_x, self._cd11 * rel_y
        xi = self._cd00 * rel_x + self._cd01 * rel_y
        eta = self._cd10 * rel_x + self._cd11 * rel_y
        return xi, eta

    def _apply_cd_inv_2d(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply cached 2x2 inverse CD transform using scalar coefficients."""
        rel0 = self._cdi00 * xi + self._cdi01 * eta
        rel1 = self._cdi10 * xi + self._cdi11 * eta
        return rel0 + self._crpix0, rel1 + self._crpix1

    def _alloc_pix_internal_like(self, world: Tensor) -> Tensor:
        """Allocate output for analytic world->pixel without zeroing 2D hot path."""
        pix_internal = torch.empty_like(world)
        if world.shape[1] > 2:
            pix_internal[:, 2:] = 0
        return pix_internal

    @staticmethod
    def _contiguous_if_needed(t: Tensor) -> Tensor:
        """Avoid unnecessary contiguous copies on already contiguous benchmark inputs."""
        return t if t.is_contiguous() else t.contiguous()

    def _use_cpp_center_pixel_to_world_small_n(
        self, x: Tensor, y: Tensor, *, device_ok: bool, dtype_ok: bool
    ) -> bool:
        """Use the center-equator C++ helper only where it is a measured win."""
        if (
            not self._use_center_equator_fast
            or self._cpp_center_pixel_to_world is None
            or not device_ok
            or not dtype_ok
            or x.requires_grad
            or y.requires_grad
        ):
            return False
        limit = _CENTER_CPP_P2W_THRESHOLDS.get(self._proj_code)
        return limit is not None and x.numel() <= limit

    def _use_cpp_center_world_to_pixel_small_n(
        self, ra: Tensor, dec: Tensor, *, device_ok: bool, dtype_ok: bool
    ) -> bool:
        """Use the center-equator C++ helper only where it is a measured win."""
        if (
            not self._use_center_equator_fast
            or self._cpp_center_world_to_pixel is None
            or not device_ok
            or not dtype_ok
            or ra.requires_grad
            or dec.requires_grad
        ):
            return False
        limit = _CENTER_CPP_W2P_THRESHOLDS.get(self._proj_code)
        return limit is not None and ra.numel() <= limit

    def _cpp_project_direct_p2w(
        self, xi: Tensor, eta: Tensor, cpp_kind: str
    ) -> Tuple[Tensor, Tensor]:
        """Direct C++ projection dispatch used by pixel->world fast path."""
        if cpp_kind == "ait":
            return self._cpp_ait_project(xi, eta)
        if cpp_kind == "mol":
            return self._cpp_mol_project(xi, eta)
        if cpp_kind == "hpx":
            return self._cpp_hpx_project(xi, eta, self._hpx_h, self._hpx_k)
        if cpp_kind == "zenithal":
            phi, theta = self._cpp_zenithal_project(xi, eta, self._proj_code)
            return -phi, theta
        if cpp_kind == "zpn":
            return self._cpp_zpn_project(xi, eta, self._zpn_coeffs_cpu)
        raise RuntimeError("unsupported_cpp_direct_projection")

    def _cpp_deproject_direct_w2p(
        self, phi_w: Tensor, theta: Tensor, cpp_kind: str
    ) -> Tuple[Tensor, Tensor]:
        """Direct C++ deprojection dispatch used by world->pixel fast path."""
        if cpp_kind == "ait":
            return self._cpp_ait_deproject(phi_w, theta)
        if cpp_kind == "hpx":
            return self._cpp_hpx_deproject(phi_w, theta, self._hpx_h, self._hpx_k)
        if cpp_kind == "zenithal":
            return self._cpp_zenithal_deproject(-phi_w, theta, self._proj_code)
        raise RuntimeError("unsupported_cpp_direct_projection")

    def _local_uvw_from_radec(
        self, ra: Tensor, dec: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert celestial RA/Dec (deg) to local east/north/radial components."""
        ra_rad = ra * self._d2r
        dec_rad = dec * self._d2r

        sin_ra, cos_ra = _sincos(ra_rad)
        sin_dec, cos_dec = _sincos(dec_rad)
        X = cos_dec * cos_ra
        Y = cos_dec * sin_ra
        Z = sin_dec

        u = X * self._east_x + Y * self._east_y
        v = X * self._north_x + Y * self._north_y + Z * self._north_z
        w = X * self._radial_x + Y * self._radial_y + Z * self._radial_z
        return u, v, w

    def _tan_intermediate_from_radec(
        self, ra: Tensor, dec: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Exact TAN intermediate coordinates (xi, eta) from RA/Dec via cached local basis."""
        if (
            self._cpp_tan_intermediate_from_radec is not None
            and not ra.requires_grad
            and not dec.requires_grad
            and ra.device.type == "cpu"
            and dec.device.type == "cpu"
            and ra.dtype == torch.float64
            and dec.dtype == torch.float64
        ):
            ra_cpp = self._contiguous_if_needed(ra)
            dec_cpp = self._contiguous_if_needed(dec)
            return self._cpp_tan_intermediate_from_radec(
                ra_cpp,
                dec_cpp,
                self._east_x,
                self._east_y,
                self._north_x,
                self._north_y,
                self._north_z,
                self._radial_x,
                self._radial_y,
                self._radial_z,
            )
        u, v, w = self._local_uvw_from_radec(ra, dec)
        scale = self._r2d / torch.clamp(w, min=1e-12)
        return u * scale, v * scale

    def _spherical_rotation_finite_only(
        self, phi: Tensor, theta: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Rotate only finite native coordinates; keep NaNs from projection domain masks."""
        valid = torch.isfinite(phi) & torch.isfinite(theta)
        if bool(valid.all()):
            return self._spherical_rotation_optimized(phi, theta, self._native_type)
        if not bool(valid.any()):
            nan = torch.full_like(phi, float("nan"))
            return nan, nan.clone()
        ra = torch.full_like(phi, float("nan"))
        dec = torch.full_like(theta, float("nan"))
        ra_v, dec_v = self._spherical_rotation_optimized(
            phi[valid], theta[valid], self._native_type
        )
        ra[valid] = ra_v
        dec[valid] = dec_v
        return ra, dec

    def _pixel_to_world_2d_core(
        self, x: Tensor, y: Tensor, origin: float
    ) -> Tuple[Tensor, Tensor]:
        """
        Unified 2D spatial pixel-to-world core.
        """
        x, y = self._corrected_pixels_to_source(x, y, origin)
        x1 = x + (1.0 - origin)
        y1 = y + (1.0 - origin)
        if self.det2im1 is not None or self.det2im2 is not None:
            x1, y1 = self._apply_det2im_forward(x1, y1)
        # 1. Coordinate frame relative to CRPIX
        rel_x = x1 - self._crpix0
        rel_y = y1 - self._crpix1

        # 2. Distortion & Linear
        if self.sip is not None:
            rel_x, rel_y = self.sip.distort(rel_x, rel_y)
        if self.cpdis1 is not None or self.cpdis2 is not None:
            rel_x, rel_y = self._apply_cpdis_forward(rel_x, rel_y)

        # Apply CD matrix
        xi, eta = self._apply_cd_2d(rel_x, rel_y)

        if self._is_gls:
            return self._gls_forward(xi, eta)

        # Common center-equator specializations avoid full spherical rotation.
        if self._use_center_equator_fast and self._is_car:
            ra = _wrap_lon360(xi + self.alpha0)
            return ra, eta
        if self._use_center_equator_fast and self._is_sfl:
            dec = eta
            cos_dec = torch.cos(dec * self._d2r)
            cos_safe = torch.sign(cos_dec) * torch.clamp(torch.abs(cos_dec), min=1e-12)
            ra = _wrap_lon360((xi / cos_safe) + self.alpha0)
            return ra, dec
        if self._use_center_equator_fast and self._is_cea:
            theta = (
                torch.asin(torch.clamp(eta / self._cea_eta_scale, -1.0, 1.0))
                * self._r2d
            )
            ra = _wrap_lon360(xi + self.alpha0)
            return ra, theta
        if self._use_center_equator_fast and self._is_mer:
            arg = torch.clamp(eta * self._d2r, -20.0, 20.0)
            theta = 2.0 * (torch.atan(torch.exp(arg)) * self._r2d - 45.0)
            ra = _wrap_lon360(xi + self.alpha0)
            return ra, theta
        if self._use_center_equator_fast and self._is_cyp:
            phi, theta = project_cylindrical(
                xi, eta, "CYP", self._pv1_tensor, self._pv2_tensor
            )
            return _wrap_lon360(phi + self.alpha0), theta

        # 3. TPV / Special distortions
        zpx_native = False
        if self._is_tpv:
            xi, eta = self.tpv.distort(xi, eta)
        elif self._is_tnx:
            xi, eta = project_tnx(xi, eta, self.wcs_params, self.wat_data)
        elif self._is_zpx:
            # project_zpx returns native (phi, theta), not intermediate (xi, eta).
            phi, theta = project_zpx(xi, eta, self.wcs_params, self.wat_data)
            zpx_native = True

        # 4. Projection
        if zpx_native:
            pass
        elif (
            self._is_tan
            or self._is_tpv
            or self._is_tnx
            or self._is_arc
            or self._is_zea
            or self._is_stg
        ):
            r = torch.hypot(xi, eta)
            phi = torch.atan2(-xi, -eta) * self._r2d
            if self._is_tan or self._is_tpv or self._is_tnx:
                theta = torch.atan2(torch.ones_like(r), r * self._d2r) * self._r2d
            elif self._is_arc:
                theta = 90.0 - r
            elif self._is_zea:
                invalid_zea = (r * self._d2r) > 2.0
                theta = (
                    90.0
                    - 2.0
                    * torch.asin(torch.clamp((r * self._d2r) * 0.5, -1.0, 1.0))
                    * self._r2d
                )
                nan = torch.full_like(theta, float("nan"))
                phi = torch.where(invalid_zea, nan, phi)
                theta = torch.where(invalid_zea, nan, theta)
            else:
                theta = 90.0 - 2.0 * torch.atan((r * self._d2r) * 0.5) * self._r2d
        elif self._is_sin:
            phi, theta = project_zenithal(xi, eta, "SIN", None, self._pv2_tensor)
        elif self._is_ait:
            phi, theta = self._native_from_ait_fast(xi, eta)
        elif self._is_mol:
            phi, theta = project_allsky(xi, eta, "MOL")
        elif self._is_hpx:
            abs_eta = torch.abs(eta)
            invalid = abs_eta > self._hpx_eta_pole
            mask_eq = (~invalid) & (abs_eta <= self._hpx_eta_boundary)
            sigma = torch.clamp(
                (self._hpx_eta_pole - abs_eta) * self._hpx_inv_polar_denom, min=0.0
            )
            s_theta = torch.where(
                mask_eq,
                eta * self._hpx_inv_eta_scale,
                torch.sign(eta) * (1.0 - (sigma * sigma) * (1.0 / 3.0)),
            )
            theta = torch.asin(torch.clamp(s_theta, -1.0, 1.0)) * self._r2d
            xc = torch.round((xi - 45.0) / 90.0) * 90.0 + 45.0
            phi = torch.where(
                mask_eq, xi, xc + (xi - xc) / torch.clamp(sigma, min=1e-9)
            )
            invalid = invalid | (
                (~mask_eq) & (~invalid) & (torch.abs(xi - xc) > (45.0 * sigma + 1e-8))
            )
            nan = torch.full_like(phi, float("nan"))
            phi = torch.where(invalid, nan, phi)
            theta = torch.where(invalid, nan, theta)
        elif self._is_cea:
            phi = xi
            theta = (
                torch.asin(torch.clamp(eta / self._cea_eta_scale, -1.0, 1.0))
                * self._r2d
            )
        elif self._is_mer:
            phi = xi
            arg = torch.clamp(eta * self._d2r, -20.0, 20.0)
            theta = 2.0 * (torch.atan(torch.exp(arg)) * self._r2d - 45.0)
        elif self._is_cyp:
            phi, theta = project_cylindrical(
                xi, eta, "CYP", self._pv1_tensor, self._pv2_tensor
            )
        elif self._is_car:
            # Inline trivial CAR projection: phi = xi, theta = eta
            phi, theta = xi, eta
        elif self._is_sfl:
            theta = eta
            phi = xi / torch.clamp(torch.cos(theta * self._d2r), min=1e-12)
        elif self._is_zpn:
            if self._is_zpn_linear:
                r = torch.hypot(xi, eta)
                phi = torch.atan2(-xi, -eta) * self._r2d
                theta = 90.0 - r
            else:
                phi, theta = project_zenithal(xi, eta, "ZPN", None, self._pv2_tensor)
        elif (
            self._proj_code in self.CONIC_CODES
            or self._proj_code in self.POLYCONIC_CODES
        ):
            phi, theta = project_conic(
                xi, eta, self._proj_code, self._pv1_tensor, self._pv2_tensor
            )
        elif self._proj_code in self.QUADCUBE_CODES:
            phi, theta = project_quadcube(xi, eta, self._proj_code)
        elif self._proj_code == "XPH":
            phi, theta = project_allsky(
                xi, eta, "XPH", self._pv1_tensor, self._pv2_tensor
            )
        elif self._proj_code == "PAR":
            phi, theta = project_allsky(
                xi, eta, "PAR", self._pv1_tensor, self._pv2_tensor
            )
        elif self._proj_code in ("AZP", "SZP", "AIR"):
            phi, theta = project_zenithal(
                xi, eta, self._proj_code, self._pv1_tensor, self._pv2_tensor
            )
        else:
            # Linear fallback, or any projection bound via _project_native_fn.
            if self._project_native_fn is not None:
                phi, theta = self._project_native_coords(xi, eta)
            else:
                phi, theta = xi, eta

        if self._use_center_equator_fast:
            ra = _wrap_lon360(phi + self.alpha0)
            return ra, theta

        # 5. Celestial Rotation
        ra, dec = self._spherical_rotation_optimized(phi, theta, self._native_type)
        return ra, dec

    def _pixel_to_world_2d_fast(
        self, x: Tensor, y: Tensor, origin: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Fast path for common 2D separate-argument pixel->world calls."""
        if x.shape != y.shape:
            raise RuntimeError("pixel coordinate shapes must match")
        if (
            self._pixel_affine_matrix is not None
            or self._has_paper4_lut()
            or self._is_gls
        ):
            return self._pixel_to_world_2d_core(x, y, float(origin))

        device_ok = x.device.type == "cpu" and y.device.type == "cpu"
        dtype_ok = x.dtype == torch.float64 and y.dtype == torch.float64

        if self._use_cpp_center_pixel_to_world_small_n(
            x, y, device_ok=device_ok, dtype_ok=dtype_ok
        ):
            return self._cpp_center_pixel_to_world(
                x,
                y,
                float(origin),
                self._crpix0,
                self._crpix1,
                self._cd00,
                self._cd01,
                self._cd10,
                self._cd11,
                self._proj_code,
                self.alpha0,
                self._cea_eta_scale,
                self._hpx_h,
                self._hpx_k,
            )

        if self._use_center_equator_fast and (
            self._is_car or self._is_sfl or self._is_cea or self._is_mer
        ):
            rel_x = x + (1.0 - origin) - self._crpix0
            rel_y = y + (1.0 - origin) - self._crpix1
            xi, eta = self._apply_cd_2d(rel_x, rel_y)
            if self._is_car:
                return _wrap_lon360(xi + self.alpha0), eta
            if self._is_sfl:
                dec = eta
                cos_dec = torch.cos(dec * self._d2r)
                cos_safe = torch.sign(cos_dec) * torch.clamp(
                    torch.abs(cos_dec), min=1e-12
                )
                return _wrap_lon360((xi / cos_safe) + self.alpha0), dec
            if self._is_cea:
                theta = (
                    torch.asin(torch.clamp(eta / self._cea_eta_scale, -1.0, 1.0))
                    * self._r2d
                )
                return _wrap_lon360(xi + self.alpha0), theta
            arg = torch.clamp(eta * self._d2r, -20.0, 20.0)
            theta = 2.0 * (torch.atan(torch.exp(arg)) * self._r2d - 45.0)
            return _wrap_lon360(xi + self.alpha0), theta

        cpp_kind = self._cpp_p2w_direct_kind
        prefer_small_zenithal_fused = (
            self._can_use_fused_p2w_static
            and not self._disable_fused_pixel_to_world
            and device_ok
            and dtype_ok
            and not x.requires_grad
            and not y.requires_grad
            and self._native_type == "pole"
            and self._proj_code in {"TAN", "SIN", "ARC"}
            and not self._is_sin_slant
            and x.numel() <= _ZENITHAL_FUSED_P2W_MAX
        )
        use_cpp_direct = (
            self._has_cpp_direct_p2w
            and not self._disable_cpp_pixel_to_world
            and device_ok
            and dtype_ok
            and not x.requires_grad
            and not y.requires_grad
            and not prefer_small_zenithal_fused
        )
        if use_cpp_direct:
            rel_x = x + (1.0 - origin) - self._crpix0
            rel_y = y + (1.0 - origin) - self._crpix1
            xi, eta = self._apply_cd_2d(rel_x, rel_y)
            if self._cpp_direct_p2w_verified:
                phi, theta = self._cpp_project_direct_p2w(xi, eta, cpp_kind)
            else:
                try:
                    phi, theta = self._cpp_project_direct_p2w(xi, eta, cpp_kind)
                    self._cpp_direct_p2w_verified = True
                except (TypeError, RuntimeError):
                    self._disable_cpp_pixel_to_world = True
                else:
                    pass
            if not self._disable_cpp_pixel_to_world:
                if self._use_center_equator_fast:
                    return _wrap_lon360(phi + self.alpha0), theta
                if (
                    self._native_type == "pole"
                    and self._cpp_spherical_rotation_pole is not None
                ):
                    return self._cpp_spherical_rotation_pole(
                        phi,
                        theta,
                        self._phi_p_rad,
                        self._pole_east_x,
                        self._pole_east_y,
                        self._pole_north_x,
                        self._pole_north_y,
                        self._pole_north_z,
                        self._pole_radial_x,
                        self._pole_radial_y,
                        self._pole_radial_z,
                    )
                return self._spherical_rotation_optimized(phi, theta, self._native_type)

        # Fused C++ pixel->world when semantics allow (TORCHSKY_WCS_FUSED). Runs after
        # cpp_direct so vectorized projection paths stay primary when available.
        use_fused = (
            self._can_use_fused_p2w_static
            and not self._disable_fused_pixel_to_world
            and device_ok
            and dtype_ok
            and not x.requires_grad
            and not y.requires_grad
        )
        if use_fused:
            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)
            ra = torch.empty_like(x_flat)
            dec = torch.empty_like(y_flat)
            if self._native_type == "pole":
                ex, ey = self._pole_east_x, self._pole_east_y
                nx, ny, nz = self._pole_north_x, self._pole_north_y, self._pole_north_z
                rx, ry, rz = (
                    self._pole_radial_x,
                    self._pole_radial_y,
                    self._pole_radial_z,
                )
            else:
                ex, ey = self._east_x, self._east_y
                nx, ny, nz = self._north_x, self._north_y, self._north_z
                rx, ry, rz = self._radial_x, self._radial_y, self._radial_z

            try:
                _cpp.wcs_pixel_to_world_fused_cpu(
                    (x_flat + (1.0 - origin)).contiguous(),
                    (y_flat + (1.0 - origin)).contiguous(),
                    ra,
                    dec,
                    self._crpix0,
                    self._crpix1,
                    self._cd00,
                    self._cd01,
                    self._cd10,
                    self._cd11,
                    self._fused_proj_code,
                    self._fused_pv2_1,
                    self._hpx_h,
                    self._hpx_k,
                    ex,
                    ey,
                    nx,
                    ny,
                    nz,
                    rx,
                    ry,
                    rz,
                    self._ra_p_rad,
                    self._dec_p_rad,
                    self._phi_p_rad,
                    self._native_type,
                    self.alpha0,
                    self.delta0,
                    self._is_tpv,
                    self._fused_tpv_idx1,
                    self._fused_tpv_c1,
                    self._fused_tpv_idx2,
                    self._fused_tpv_c2,
                )
            except TypeError:
                # C++ extension can be stale relative to Python bindings in editable
                # workflows. Disable this fused path and fall back safely.
                self._disable_fused_pixel_to_world = True
                return self._pixel_to_world_2d_core(x, y, float(origin))
            return ra.reshape(x.shape), dec.reshape(x.shape)

        core_fn = self._compiled_pixel_to_world_2d_fn or self._pixel_to_world_2d_core
        return core_fn(x, y, float(origin))

    def _native_from_ait_fast(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """Fast AIT inverse projection: intermediate (deg) -> native (deg)."""
        X = xi * self._d2r
        Y = eta * self._d2r

        X25 = X * 0.25
        Y50 = Y * 0.5
        # Element-wise multiplication is faster than torch.square or **2
        r2 = X25 * X25 + Y50 * Y50
        valid = r2 <= 1.0
        z = torch.sqrt(torch.clamp(1.0 - r2, min=0.0))

        phi = 2.0 * torch.atan2(0.5 * z * X, 2.0 * z * z - 1.0) * self._r2d
        theta = torch.asin(torch.clamp(z * Y, -1.0, 1.0)) * self._r2d

        if valid.all():
            return phi, theta
        phi = phi.clone()
        theta = theta.clone()
        phi.masked_fill_(~valid, float("nan"))
        theta.masked_fill_(~valid, float("nan"))
        return phi, theta

    def _world_to_pixel_2d_core(
        self,
        ra: Tensor,
        dec: Tensor,
        origin: float,
        *,
        allow_cpp_inverse_sphere: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Unified 2D spatial world-to-pixel core.
        Designed to be graph-break free for torch.compile.
        """
        # Common center-equator specializations avoid full spherical rotation.
        if self._use_center_equator_fast and self.sip is None:
            if self._is_car:
                xi = _wrap_lon180(ra - self.alpha0)
                eta = dec
                rel_x = self._cdi00 * xi + self._cdi01 * eta
                rel_y = self._cdi10 * xi + self._cdi11 * eta
                px = rel_x + self._crpix0 - (1.0 - origin)
                py = rel_y + self._crpix1 - (1.0 - origin)
                return self._source_pixels_to_corrected(px, py, origin)
            if self._is_sfl:
                phi = _wrap_lon180(ra - self.alpha0)
                theta = dec
                xi = phi * torch.cos(theta * self._d2r)
                eta = theta
                rel_x = self._cdi00 * xi + self._cdi01 * eta
                rel_y = self._cdi10 * xi + self._cdi11 * eta
                px = rel_x + self._crpix0 - (1.0 - origin)
                py = rel_y + self._crpix1 - (1.0 - origin)
                return self._source_pixels_to_corrected(px, py, origin)
            if self._is_cea:
                phi = _wrap_lon180(ra - self.alpha0)
                theta = dec
                xi = phi
                eta = self._cea_eta_scale * torch.sin(theta * self._d2r)
                rel_x = self._cdi00 * xi + self._cdi01 * eta
                rel_y = self._cdi10 * xi + self._cdi11 * eta
                px = rel_x + self._crpix0 - (1.0 - origin)
                py = rel_y + self._crpix1 - (1.0 - origin)
                return self._source_pixels_to_corrected(px, py, origin)
            if self._is_mer:
                phi = _wrap_lon180(ra - self.alpha0)
                theta = dec
                xi = phi
                tan_arg = torch.tan((45.0 + theta * 0.5) * self._d2r)
                tan_arg = torch.clamp(tan_arg, min=1e-12)
                eta = self._r2d * torch.log(tan_arg)
                rel_x = self._cdi00 * xi + self._cdi01 * eta
                rel_y = self._cdi10 * xi + self._cdi11 * eta
                px = rel_x + self._crpix0 - (1.0 - origin)
                py = rel_y + self._crpix1 - (1.0 - origin)
                return self._source_pixels_to_corrected(px, py, origin)
            if self._is_cyp:
                phi = _wrap_lon180(ra - self.alpha0)
                theta = dec
                xi, eta = deproject_cylindrical(
                    phi, theta, "CYP", self._pv1_tensor, self._pv2_tensor
                )
                rel_x = self._cdi00 * xi + self._cdi01 * eta
                rel_y = self._cdi10 * xi + self._cdi11 * eta
                px = rel_x + self._crpix0 - (1.0 - origin)
                py = rel_y + self._crpix1 - (1.0 - origin)
                return self._source_pixels_to_corrected(px, py, origin)

        # 1. Inverse Spherical Rotation (Celestial -> Native)
        if self._is_gls:
            xi, eta = self._gls_inverse(ra, dec)
        elif (
            self._is_tan
            and not self._is_tpv
            and not self._is_tnx
            and not self._is_zpx
            and self.alpha0 == self.alpha_p
            and self.delta0 == self.delta_p
        ):
            xi, eta = self._tan_intermediate_from_radec(ra, dec)
        else:
            if self._use_center_equator_fast:
                phi, theta = _wrap_lon180(ra - self.alpha0), dec
            else:
                phi, theta = self._inverse_spherical_rotation(
                    ra,
                    dec,
                    self._native_type,
                    allow_cpp_inverse_sphere=allow_cpp_inverse_sphere,
                )

            # 2. Inverse Projection (Native -> Intermediate)
            if (
                self._is_tan
                or self._is_tpv
                or self._is_tnx
                or self._is_arc
                or self._is_zea
                or self._is_stg
            ):
                phi_rad = phi * self._d2r
                sin_phi, cos_phi = _sincos(phi_rad)
                if self._is_tan or self._is_tpv or self._is_tnx:
                    r = self._r2d * torch.tan((90.0 - theta) * self._d2r)
                elif self._is_arc:
                    r = 90.0 - theta
                elif self._is_zea:
                    w = (90.0 - theta) * self._d2r
                    r = 2.0 * self._r2d * torch.sin(0.5 * w)
                else:
                    w = (90.0 - theta) * self._d2r
                    r = 2.0 * self._r2d * torch.tan(0.5 * w)
                xi = -r * sin_phi
                eta = -r * cos_phi
            elif self._is_sin:
                xi, eta = deproject_zenithal(phi, theta, "SIN", None, self._pv2_tensor)
            elif self._is_ait:
                xi, eta = deproject_allsky(phi, theta, "AIT")
            elif self._is_mol:
                xi, eta = deproject_allsky(phi, theta, "MOL")
            elif self._is_hpx:
                phi_w = _wrap_lon180_checked(phi)
                s_theta = torch.sin(theta * self._d2r)
                abs_s = torch.abs(s_theta)
                mask_eq = abs_s <= (2.0 / 3.0)
                sigma = torch.sqrt(torch.clamp(3.0 * (1.0 - abs_s), min=0.0))
                eta_pol = torch.sign(s_theta) * (
                    self._hpx_eta_pole
                    - (self._hpx_eta_pole - self._hpx_eta_boundary) * sigma
                )
                xc = torch.round((phi_w - 45.0) / 90.0) * 90.0 + 45.0
                xi = torch.where(mask_eq, phi_w, xc + sigma * (phi_w - xc))
                eta = torch.where(mask_eq, self._hpx_eta_scale * s_theta, eta_pol)
            elif self._is_cea:
                xi = _wrap_lon180_checked(phi)
                eta = self._cea_eta_scale * torch.sin(theta * self._d2r)
            elif self._is_mer:
                xi = _wrap_lon180_checked(phi)
                tan_arg = torch.tan((45.0 + theta * 0.5) * self._d2r)
                tan_arg = torch.clamp(tan_arg, min=1e-12)
                eta = self._r2d * torch.log(tan_arg)
            elif self._is_cyp:
                xi, eta = deproject_cylindrical(
                    phi, theta, "CYP", self._pv1_tensor, self._pv2_tensor
                )
            elif self._is_car:
                # Inline trivial CAR projection: native RA -> intermediate xi, native Dec -> intermediate eta
                xi = _wrap_lon180_checked(phi)
                eta = theta
            elif self._is_sfl:
                xi = _wrap_lon180_checked(phi) * torch.cos(theta * self._d2r)
                eta = theta
            elif self._is_zpn:
                if self._is_zpn_linear:
                    phi_w = _wrap_lon180(phi)
                    r = 90.0 - theta
                    phi_rad = phi_w * self._d2r
                    sin_phi, cos_phi = _sincos(phi_rad)
                    xi = -r * sin_phi
                    eta = -r * cos_phi
                else:
                    xi, eta = deproject_zenithal(phi, theta, "ZPN", None, self._pv2_tensor)
            elif (
                self._proj_code in self.CONIC_CODES
                or self._proj_code in self.POLYCONIC_CODES
            ):
                xi, eta = deproject_conic(
                    phi, theta, self._proj_code, self._pv1_tensor, self._pv2_tensor
                )
            elif self._proj_code in self.QUADCUBE_CODES:
                xi, eta = deproject_quadcube(phi, theta, self._proj_code)
            elif self._proj_code == "XPH":
                xi, eta = deproject_allsky(
                    phi, theta, "XPH", self._pv1_tensor, self._pv2_tensor
                )
            elif self._proj_code == "PAR":
                xi, eta = deproject_allsky(
                    phi, theta, "PAR", self._pv1_tensor, self._pv2_tensor
                )
            elif self._proj_code in ("AZP", "SZP", "AIR"):
                xi, eta = deproject_zenithal(
                    phi, theta, self._proj_code, self._pv1_tensor, self._pv2_tensor
                )
            elif self._is_zpx:
                xi, eta = deproject_zpx(phi, theta, self.wcs_params, self.wat_data)
            else:
                # Fallback: dispatch to the appropriate deproject function.
                if self._project_native_kind == "zenithal":
                    xi, eta = deproject_zenithal(
                        phi, theta, self._proj_code, self._pv1_tensor, self._pv2_tensor
                    )
                elif self._project_native_kind == "cylindrical":
                    xi, eta = deproject_cylindrical(
                        phi, theta, self._proj_code, self._pv1_tensor, self._pv2_tensor
                    )
                elif self._project_native_kind == "allsky":
                    xi, eta = deproject_allsky(
                        phi, theta, self._proj_code, self._pv1_tensor, self._pv2_tensor
                    )
                elif self._project_native_kind == "conic":
                    xi, eta = deproject_conic(
                        phi, theta, self._proj_code, self._pv1_tensor, self._pv2_tensor
                    )
                elif self._project_native_kind == "quadcube":
                    xi, eta = deproject_quadcube(phi, theta, self._proj_code)
                else:
                    xi, eta = deproject_zenithal(
                        phi, theta, self._proj_code, self._pv1_tensor, self._pv2_tensor
                    )

        # 3. Inverse TPV / Special distortions
        if self._is_tpv:
            if self.tpv._affine_only:
                xi, eta = self.tpv.invert_affine_exact(xi, eta)
            else:
                xi, eta = self.tpv.invert(xi, eta)
        if self._is_tnx:
            xi, eta = invert_tnx_plane(xi, eta, self.wat_data)

        # 4. Inverse Linear (CD matrix)
        rel_x = self._cdi00 * xi + self._cdi01 * eta
        rel_y = self._cdi10 * xi + self._cdi11 * eta

        # 5. Inverse CPDIS / SIP / DET2IM (if any)
        if self.cpdis1 is not None or self.cpdis2 is not None:
            rel_x, rel_y = self._apply_cpdis_inverse(rel_x, rel_y)
        if self.sip is not None:
            rel_x, rel_y = self.sip.invert_distortion(rel_x, rel_y)

        # 6. Origin adjustment
        x1 = rel_x + self._crpix0
        y1 = rel_y + self._crpix1
        if self.det2im1 is not None or self.det2im2 is not None:
            x1, y1 = self._apply_det2im_inverse(x1, y1)
        px = x1 - (1.0 - origin)
        py = y1 - (1.0 - origin)
        return self._source_pixels_to_corrected(px, py, origin)

    def _stacked_world_to_pixel_flat(
        self,
        world_flat: Tensor,
        origin: float,
        *,
        allow_cpp_inverse_sphere: bool = True,
    ) -> Tensor:
        """Fused ``world -> (px, py)`` for stacked 2D spatial + ``torch.compile``."""
        ra = world_flat[:, 0]
        dec = world_flat[:, 1]
        px, py = self._world_to_pixel_2d_core(
            ra, dec, origin, allow_cpp_inverse_sphere=allow_cpp_inverse_sphere
        )
        return torch.stack((px, py), dim=-1)

    def _world_to_pixel_2d_fast(
        self, ra: Tensor, dec: Tensor, iterative: bool = True, origin: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Fast path for common 2D separate-argument world->pixel calls."""
        if ra.shape != dec.shape:
            raise RuntimeError("world coordinate shapes must match")
        if (
            self._pixel_affine_matrix is not None
            or self._has_paper4_lut()
            or self._is_gls
        ):
            return self._world_to_pixel_2d_core(ra, dec, float(origin))
        shape = ra.shape

        device_ok = ra.device.type == "cpu" and dec.device.type == "cpu"
        dtype_ok = ra.dtype == torch.float64 and dec.dtype == torch.float64

        # Exact TAN analytic inverse: use direct intermediate coordinates instead of
        # inverse-rotation + zenithal deprojection chain.
        if self._is_tan and not self._is_tpv and not self._is_tnx and not self._is_zpx:
            xi, eta = self._tan_intermediate_from_radec(ra, dec)
            if self.sip is None:
                px, py = self._apply_cd_inv_2d(xi, eta)
            else:
                rel_x = self._cdi00 * xi + self._cdi01 * eta
                rel_y = self._cdi10 * xi + self._cdi11 * eta
                rel_x, rel_y = self.sip.invert_distortion(rel_x, rel_y)
                px = rel_x + self._crpix0
                py = rel_y + self._crpix1
            if origin == 1:
                return px, py
            if origin == 0:
                return px.sub(1.0), py.sub(1.0)
            off = float(origin - 1)
            return px.add(off), py.add(off)

        if self._use_cpp_center_world_to_pixel_small_n(
            ra, dec, device_ok=device_ok, dtype_ok=dtype_ok
        ):
            try:
                return self._cpp_center_world_to_pixel(
                    ra,
                    dec,
                    self._crpix0,
                    self._crpix1,
                    self._cdi00,
                    self._cdi01,
                    self._cdi10,
                    self._cdi11,
                    self._proj_code,
                    self.alpha0,
                    self._cea_eta_scale,
                    float(origin),
                    self._hpx_h,
                    self._hpx_k,
                )
            except (TypeError, RuntimeError):
                pass

        if self._use_center_equator_fast and (
            self._is_car or self._is_sfl or self._is_cea or self._is_mer
        ):
            phi = _wrap_lon180(ra - self.alpha0)
            theta = dec
            if self._is_car:
                xi = phi
                eta = theta
            elif self._is_sfl:
                xi = phi * torch.cos(theta * self._d2r)
                eta = theta
            elif self._is_cea:
                xi = phi
                eta = self._cea_eta_scale * torch.sin(theta * self._d2r)
            else:
                xi = phi
                tan_arg = torch.tan((45.0 + theta * 0.5) * self._d2r)
                tan_arg = torch.clamp(tan_arg, min=1e-12)
                eta = self._r2d * torch.log(tan_arg)
            rel_x = self._cdi00 * xi + self._cdi01 * eta
            rel_y = self._cdi10 * xi + self._cdi11 * eta
            px = rel_x + self._crpix0 - (1.0 - origin)
            py = rel_y + self._crpix1 - (1.0 - origin)
            return px, py

        cpp_kind = self._cpp_w2p_direct_kind
        use_cpp_direct = (
            self._has_cpp_direct_w2p
            and not self._disable_cpp_world_to_pixel
            and device_ok
            and dtype_ok
            and not ra.requires_grad
            and not dec.requires_grad
        )
        if use_cpp_direct:
            if self._use_center_equator_fast:
                phi_w = _wrap_lon180(ra - self.alpha0)
                theta = dec
            elif (
                self._native_type == "pole"
                and self._cpp_inverse_spherical_rotation_pole is not None
            ):
                phi, theta = self._cpp_inverse_spherical_rotation_pole(
                    ra,
                    dec,
                    self._phi_p_rad,
                    self._pole_east_x,
                    self._pole_east_y,
                    self._pole_north_x,
                    self._pole_north_y,
                    self._pole_north_z,
                    self._pole_radial_x,
                    self._pole_radial_y,
                    self._pole_radial_z,
                )
                phi_w = _wrap_lon180(phi)
            else:
                phi, theta = self._inverse_spherical_rotation(
                    ra, dec, self._native_type
                )
                phi_w = _wrap_lon180(phi)
            if self._cpp_direct_w2p_verified:
                xi, eta = self._cpp_deproject_direct_w2p(phi_w, theta, cpp_kind)
            else:
                try:
                    xi, eta = self._cpp_deproject_direct_w2p(phi_w, theta, cpp_kind)
                    self._cpp_direct_w2p_verified = True
                except (TypeError, RuntimeError):
                    self._disable_cpp_world_to_pixel = True
            if not self._disable_cpp_world_to_pixel:
                rel_x = self._cdi00 * xi + self._cdi01 * eta
                rel_y = self._cdi10 * xi + self._cdi11 * eta
                px = rel_x + self._crpix0 - (1.0 - origin)
                py = rel_y + self._crpix1 - (1.0 - origin)
                return px, py

        use_fused = (
            self._can_use_fused_w2p_static
            and not self._disable_fused_world_to_pixel
            and device_ok
            and dtype_ok
            and not ra.requires_grad
            and not dec.requires_grad
            and (not self._is_tpv or not self.tpv._invert_trace_enabled)
        )
        if use_fused:
            ra_flat = ra.reshape(-1)
            dec_flat = dec.reshape(-1)
            px = torch.empty_like(ra_flat)
            py = torch.empty_like(dec_flat)

            if self._native_type == "pole":
                ex, ey = self._pole_east_x, self._pole_east_y
                nx, ny, nz = self._pole_north_x, self._pole_north_y, self._pole_north_z
                rx, ry, rz = (
                    self._pole_radial_x,
                    self._pole_radial_y,
                    self._pole_radial_z,
                )
            else:
                ex, ey = self._east_x, self._east_y
                nx, ny, nz = self._north_x, self._north_y, self._north_z
                rx, ry, rz = self._radial_x, self._radial_y, self._radial_z

            try:
                _cpp.wcs_world_to_pixel_fused_cpu(
                    ra_flat.contiguous(),
                    dec_flat.contiguous(),
                    px,
                    py,
                    self._crpix0,
                    self._crpix1,
                    self._cdi00,
                    self._cdi01,
                    self._cdi10,
                    self._cdi11,
                    self._fused_proj_code,
                    self._fused_pv2_1,
                    self._hpx_h,
                    self._hpx_k,
                    ex,
                    ey,
                    nx,
                    ny,
                    nz,
                    rx,
                    ry,
                    rz,
                    self._ra_p_rad,
                    self._dec_p_rad,
                    self._phi_p_rad,
                    self._native_type,
                    self.alpha0,
                    self.delta0,
                    self._is_tpv,
                    self._fused_tpv_idx1,
                    self._fused_tpv_c1,
                    self._fused_tpv_idx2,
                    self._fused_tpv_c2,
                )
            except TypeError:
                # C++ extension can be stale relative to Python bindings in editable
                # workflows. Disable this fused path and fall back safely.
                self._disable_fused_world_to_pixel = True
                return self._world_to_pixel_2d_core(ra, dec, float(origin))
            px = px.reshape(shape)
            py = py.reshape(shape)
            if origin == 1:
                return px, py
            if origin == 0:
                return px.sub_(1.0), py.sub_(1.0)
            off = float(origin - 1)
            return px.add_(off), py.add_(off)

        # Fallback to core (optionally per-instance compiled).
        core_fn = self._compiled_world_to_pixel_2d_fn or self._world_to_pixel_2d_core
        return core_fn(ra, dec, float(origin))

    def compile(self, **kwargs):
        """
        Wrap the pure-Python 2D spatial cores in ``torch.compile``.

        After ``compile()``, use a **stacked pixel tensor** ``pixel_to_world(pixels)``
        with ``pixels.shape == (..., 2)`` for 2D spatial WCS. That path prefers a
        dedicated compiled graph that fuses ``pixels + (1-origin) - CRPIX`` with the
        transform and uses **pure Torch** pole rotation (no C++ sphere op inside the
        graph) so ``torch.compile`` can optimize end-to-end. The inverse
        ``world_to_pixel(world)`` with ``world.shape == (..., 2)`` uses an analogous
        fused graph with **pure Torch** inverse pole rotation for ``origin`` in
        ``{0, 1}``. Other ``origin`` values and ``naxis > 2`` still use
        ``_compiled_transform_fn`` / the general inverse path on intermediate coords.

        The two-argument ``pixel_to_world(x, y)`` shortcut may still use native C++
        fast paths. Run ``benchmarks/bench_wcs_small_n_paths.py`` for timings vs Astropy.

        Args:
            **kwargs: Arguments passed to torch.compile (e.g., mode, options).
        """
        compile_kwargs = {"mode": "reduce-overhead"}
        compile_kwargs.update(kwargs)
        transform_core = self._transform_optimized
        pixel_core = self._pixel_to_world_2d_core
        world_core = self._world_to_pixel_2d_core
        self._compiled_transform_fn = torch.compile(
            lambda rel: transform_core(rel), **compile_kwargs
        )
        self._compiled_pixel_to_world_2d_fn = torch.compile(
            lambda x, y, origin: pixel_core(x, y, origin), **compile_kwargs
        )
        self._compiled_world_to_pixel_2d_fn = torch.compile(
            lambda ra, dec, origin: world_core(ra, dec, origin), **compile_kwargs
        )

        self._compiled_stack_pixel_to_world = {}
        if self.naxis == 2 and self._is_spatial:

            def _make_stack_fn(o: float) -> Callable[[Tensor], Tensor]:
                def _fn(pixels_flat: Tensor) -> Tensor:
                    return self._stacked_pixels_to_world_flat(
                        pixels_flat, o, allow_cpp_sphere=False
                    )

                return _fn

            for _origin_key in (0.0, 1.0):
                self._compiled_stack_pixel_to_world[_origin_key] = torch.compile(
                    _make_stack_fn(_origin_key), **compile_kwargs
                )
        else:
            self._compiled_stack_pixel_to_world = None

        self._compiled_stack_world_to_pixel = {}
        if self.naxis == 2 and self._is_spatial:

            def _make_w2p_stack_fn(o: float) -> Callable[[Tensor], Tensor]:
                def _fn(world_flat: Tensor) -> Tensor:
                    return self._stacked_world_to_pixel_flat(
                        world_flat, o, allow_cpp_inverse_sphere=False
                    )

                return _fn

            for _ow in (0.0, 1.0):
                self._compiled_stack_world_to_pixel[_ow] = torch.compile(
                    _make_w2p_stack_fn(_ow), **compile_kwargs
                )
        else:
            self._compiled_stack_world_to_pixel = None

        return self

    def to(self, device: Union[str, torch.device]):
        """Move WCS parameters to specified device.

        CUDA and CPU keep float64 cached parameters. MPS is float32-only, so
        cached tensors are cast to float32 there (arcsecond-class error vs
        float64 is expected). C++ fused kernels stay CPU.
        """
        self.device = torch.device(device)
        target_dtype = torch.float32 if self.device.type == "mps" else None
        # C++ ZPN reads a host coefficient vector; keep it on CPU.
        skip = {"_zpn_coeffs_cpu"}
        for name, value in vars(self).items():
            if name in skip:
                continue
            if isinstance(value, torch.Tensor):
                setattr(
                    self,
                    name,
                    value.to(device=self.device, dtype=target_dtype or value.dtype),
                )
        if self.tpv is not None:
            self.tpv.to(self.device)
        if self.sip is not None:
            self.sip.to(self.device)
        for lut_attr in ("cpdis1", "cpdis2", "det2im1", "det2im2"):
            lut_obj = getattr(self, lut_attr, None)
            if lut_obj is not None:
                setattr(
                    self,
                    lut_attr,
                    _coerce_distortion_lookup_table(lut_obj).to(self.device),
                )
        tabular_dtype = torch.float32 if self.device.type == "mps" else torch.float64
        self.tabular_axes = {
            axis: tabular.to(device=self.device, dtype=tabular_dtype)
            for axis, tabular in self.tabular_axes.items()
        }
        if self.coupled_tabular is not None:
            self.coupled_tabular = self.coupled_tabular.to(
                device=self.device, dtype=tabular_dtype
            )
        if self._pixel_affine_matrix is not None:
            self._set_pixel_affine(self._pixel_affine_matrix, self._pixel_affine_offset)
        return self

    @property
    def pixel_n_dim(self) -> int:
        """Number of pixel axes."""
        return self.naxis

    @property
    def world_n_dim(self) -> int:
        """Number of world axes."""
        return self.naxis

    @property
    def pixel_axis_names(self) -> tuple[str, ...]:
        """Pixel axis names, using FITS 1-based axis labels."""
        return tuple(f"PIXEL{i}" for i in range(1, self.naxis + 1))

    @property
    def world_axis_names(self) -> tuple[str, ...]:
        """World axis names from ``CTYPE`` with blanks filled by FITS axis labels."""
        names: list[str] = []
        for i in range(1, self.naxis + 1):
            ctype = str(self.wcs_params.get(f"CTYPE{i}", "")).strip()
            names.append(ctype or f"WORLD{i}")
        return tuple(names)

    @property
    def world_axis_units(self) -> tuple[str, ...]:
        """World axis units from ``CUNIT`` keywords."""
        return tuple(
            str(self.wcs_params.get(f"CUNIT{i}", "")) for i in range(1, self.naxis + 1)
        )

    @property
    def world_axis_physical_types(self) -> tuple[str | None, ...]:
        """Best-effort FITS CTYPE to Astropy/GWCS-style physical type mapping."""
        result: list[str | None] = []
        for i in range(1, self.naxis + 1):
            ctype = str(self.wcs_params.get(f"CTYPE{i}", "")).strip().upper()
            if i == 1 and ctype.startswith(("RA", "GLON", "ELON", "HLON", "SLON")):
                result.append(
                    "pos.eq.ra" if ctype.startswith("RA") else "pos.galactic.lon"
                )
            elif i == 2 and ctype.startswith(("DEC", "GLAT", "ELAT", "HLAT", "SLAT")):
                result.append(
                    "pos.eq.dec" if ctype.startswith("DEC") else "pos.galactic.lat"
                )
            elif ctype.startswith(
                ("FREQ", "ENER", "WAVN", "WAVE", "AWAV", "VELO", "VRAD", "VOPT", "ZOPT")
            ):
                result.append("em." + ctype.split("-")[0].lower())
            elif ctype.startswith(("TIME", "MJD", "JD")):
                result.append("time")
            else:
                result.append(None)
        return tuple(result)

    @property
    def spectral_axis_indices(self) -> tuple[int, ...]:
        """0-based indices of spectral-like world axes."""
        axes: list[int] = []
        for i in range(1, self.naxis + 1):
            ctype = str(self.wcs_params.get(f"CTYPE{i}", "")).strip()
            try:
                canonical_spectral_type(ctype)
            except ValueError:
                continue
            axes.append(i - 1)
        return tuple(axes)

    def _resolve_spectral_axis(self, axis: int | None = None) -> int:
        axes = self.spectral_axis_indices
        if axis is not None:
            idx = int(axis)
            if idx < 0:
                idx += self.naxis
            if idx < 0 or idx >= self.naxis:
                raise ValueError(f"spectral axis {axis!r} outside WCS axes")
            return idx
        if len(axes) != 1:
            raise ValueError(
                "axis must be specified when WCS has zero or multiple spectral axes"
            )
        return axes[0]

    def _rest_frequency_for_spectral(self) -> float | None:
        return rest_frequency_from_header(self.wcs_params)

    def spectral_to_frequency(
        self, value: Any, *, axis: int | None = None, unit: str | None = None
    ) -> Tensor:
        """Convert a WCS spectral world value to frequency in Hz."""
        idx = self._resolve_spectral_axis(axis)
        ctype = str(self.wcs_params.get(f"CTYPE{idx + 1}", ""))
        cunit = (
            unit
            if unit is not None
            else str(self.wcs_params.get(f"CUNIT{idx + 1}", ""))
        )
        return _spectral_to_frequency(
            value,
            ctype,
            unit=cunit or None,
            rest_frequency=self._rest_frequency_for_spectral(),
        )

    def frequency_to_spectral(
        self, frequency: Any, *, axis: int | None = None, unit: str | None = None
    ) -> Tensor:
        """Convert frequency in Hz to this WCS spectral world coordinate."""
        idx = self._resolve_spectral_axis(axis)
        ctype = str(self.wcs_params.get(f"CTYPE{idx + 1}", ""))
        cunit = (
            unit
            if unit is not None
            else str(self.wcs_params.get(f"CUNIT{idx + 1}", ""))
        )
        return _frequency_to_spectral(
            frequency,
            ctype,
            unit=cunit or None,
            rest_frequency=self._rest_frequency_for_spectral(),
        )

    @property
    def temporal_axis_indices(self) -> tuple[int, ...]:
        """0-based indices of temporal-like world axes."""
        axes: list[int] = []
        for i in range(1, self.naxis + 1):
            ctype = str(self.wcs_params.get(f"CTYPE{i}", "")).strip().upper()
            if ctype.startswith(("TIME", "MJD", "JD")):
                axes.append(i - 1)
        return tuple(axes)

    def _resolve_temporal_axis(self, axis: int | None = None) -> int:
        axes = self.temporal_axis_indices
        if axis is not None:
            idx = int(axis)
            if idx < 0:
                idx += self.naxis
            if idx < 0 or idx >= self.naxis:
                raise ValueError(f"temporal axis {axis!r} outside WCS axes")
            return idx
        if len(axes) != 1:
            raise ValueError(
                "axis must be specified when WCS has zero or multiple temporal axes"
            )
        return axes[0]

    def temporal_wcs(self, *, axis: int | None = None) -> TemporalWCS:
        """Return a ``TemporalWCS`` view using this WCS temporal metadata."""
        idx = self._resolve_temporal_axis(axis)
        header = dict(self.wcs_params)
        if "TIMEUNIT" not in header and f"CUNIT{idx + 1}" in header:
            header["TIMEUNIT"] = header[f"CUNIT{idx + 1}"]
        if "MJDREF" not in header and "MJDREFI" not in header:
            header["MJDREF"] = float(self.crval[idx].detach().cpu().item())
        return TemporalWCS(header)

    @property
    def celestial_frame(self) -> str:
        """Celestial reference frame from FITS metadata."""
        ctype1 = str(self.wcs_params.get("CTYPE1", "")).strip().upper()
        ctype2 = str(self.wcs_params.get("CTYPE2", "")).strip().upper()
        if ctype1.startswith("GLON") or ctype2.startswith("GLAT"):
            return "GALACTIC"
        if ctype1.startswith("ELON") or ctype2.startswith("ELAT"):
            return "ECLIPTIC"
        frame = self.wcs_params.get("RADESYS", self.wcs_params.get("RADECSYS", "ICRS"))
        return canonical_celestial_frame(str(frame))

    @property
    def celestial_equinox(self) -> float | None:
        """Julian equinox from FITS ``EQUINOX`` / ``EPOCH`` metadata."""
        raw = self.wcs_params.get("EQUINOX", self.wcs_params.get("EPOCH", None))
        return None if raw is None else float(raw)

    def precess_celestial(
        self, ra: Any, dec: Any, to_equinox: float, *, from_equinox: float | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Precess celestial coordinates using WCS frame/equinox metadata."""
        frame = self.celestial_frame
        if frame == "FK4":
            source = (
                float(from_equinox)
                if from_equinox is not None
                else float(self.celestial_equinox or 1950.0)
            )
            return _convert_celestial(
                ra,
                dec,
                from_frame="FK4",
                to_frame="FK4",
                from_equinox=source,
                to_equinox=float(to_equinox),
            )
        if frame in {"GALACTIC", "GAL", "ECLIPTIC", "ECL"}:
            # Mean Galactic and ecliptic of J2000 are ICRS-fixed rotations.
            return ra, dec
        if frame in {"GCRS", "CIRS", "TETE", "ALTAZ", "ITRS", "HORIZON"}:
            raise NotImplementedError(
                f"{frame} needs obstime and EarthOrientation via wcs.transform; "
                "it is not a geometric equinox precession"
            )
        if frame not in {"ICRS", "FK5"}:
            raise NotImplementedError(f"precession is not implemented for {frame}")
        source = (
            float(from_equinox)
            if from_equinox is not None
            else float(self.celestial_equinox or 2000.0)
        )
        return _precess_fk5(ra, dec, from_equinox=source, to_equinox=float(to_equinox))

    def convert_celestial(
        self,
        lon: Any,
        lat: Any,
        to_frame: str,
        *,
        to_equinox: float | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Convert WCS celestial coordinates to a supported target frame."""
        return _convert_celestial(
            lon,
            lat,
            from_frame=self.celestial_frame,
            to_frame=to_frame,
            from_equinox=self.celestial_equinox,
            to_equinox=to_equinox,
        )

    @property
    def pixel_shape(self) -> tuple[int, ...] | None:
        """Pixel shape in FITS pixel-axis order, when available from ``NAXISi``."""
        return self._pixel_shape

    @property
    def array_shape(self) -> tuple[int, ...] | None:
        """Array shape in NumPy order, when available from ``NAXISi``."""
        if self._pixel_shape is None:
            return None
        return tuple(reversed(self._pixel_shape))

    @property
    def pixel_bounds(self) -> list[tuple[float, float]] | None:
        """Inclusive 0-based pixel bounds from ``NAXISi`` keywords."""
        if self._pixel_shape is None:
            return None
        return [(0.0, float(n - 1)) for n in self._pixel_shape]

    @property
    def axis_correlation_matrix(self) -> np.ndarray:
        """Boolean matrix showing which world axes depend on which pixel axes."""
        matrix = np.zeros((self.world_n_dim, self.pixel_n_dim), dtype=bool)
        cd_deps = torch.abs(self.cd_full).detach().cpu().numpy() > 1.0e-15
        if self._is_spatial and self.naxis >= 2:
            matrix[:2, :2] = True
            for i in range(2, self.naxis):
                matrix[i, :] = cd_deps[i, :]
        else:
            matrix[:, :] = cd_deps
        for axis in self.tabular_axes:
            if 0 <= axis < self.naxis:
                matrix[axis, axis] = True
        for axis in self.coupled_tabular_axes:
            matrix[axis, :] = False
            matrix[axis, list(self.coupled_tabular_pixel_axes)] = True
        return matrix

    def footprint(self, *, origin: int = 0) -> Tensor:
        """Return the celestial footprint corners for a 2D image WCS."""
        if self._pixel_shape is None or self.naxis < 2:
            raise ValueError("footprint requires NAXIS1/NAXIS2 image bounds")
        nx, ny = self._pixel_shape[0], self._pixel_shape[1]
        corners_2d = torch.tensor(
            [
                [0.0, 0.0],
                [float(nx - 1), 0.0],
                [float(nx - 1), float(ny - 1)],
                [0.0, float(ny - 1)],
            ],
            dtype=self._wcs_coord_dtype(self.device),
            device=self.device,
        )
        if self.naxis == 2:
            corners = corners_2d
        else:
            corners = torch.empty(
                (4, self.naxis), dtype=self._wcs_coord_dtype(self.device), device=self.device
            )
            corners[:, 0:2] = corners_2d
            for axis in range(2, self.naxis):
                corners[:, axis] = self.crpix[axis] - (1.0 - origin)
        world = self.pixel_to_world(corners, origin=origin)
        if isinstance(world, tuple):
            return torch.stack(world[:2], dim=-1)
        return world[..., :2]

    def in_image(self, pixels: Any, *, origin: int = 0) -> Tensor:
        """Return a boolean mask for pixels inside ``pixel_bounds``."""
        if self._pixel_shape is None:
            raise ValueError("in_image requires NAXISi image bounds")
        pix = torch.as_tensor(pixels, dtype=self._wcs_coord_dtype(self.device), device=self.device)
        if pix.shape[-1] != self.naxis:
            raise ValueError(f"pixels must have trailing dimension {self.naxis}")
        lower = torch.full(
            (self.naxis,), float(origin), dtype=self._wcs_coord_dtype(self.device), device=self.device
        )
        upper = torch.tensor(
            [float(n - 1 + origin) for n in self._pixel_shape[: self.naxis]],
            dtype=self._wcs_coord_dtype(self.device),
            device=self.device,
        )
        return torch.all((pix >= lower) & (pix <= upper), dim=-1)

    def mixed_pixel_world(
        self,
        *,
        pixel: Any | None = None,
        world: Any | None = None,
        origin: int = 0,
        max_iter: int = 20,
        tol: float = 1e-10,
    ) -> tuple[Tensor, Tensor]:
        """Resolve a Kapteyn-style mixture of pixel and world coordinates.

        ``pixel`` and ``world`` are arrays with trailing dimension ``naxis``. Unknown
        entries are marked with NaN. Fully known pixel or world coordinates dispatch
        to the normal transforms. For 2D celestial axes, one known pixel axis plus
        one known world axis is solved by a vectorized one-dimensional Newton step,
        then the remaining coordinate is filled from the exact forward transform.
        """
        if pixel is None and world is None:
            raise ValueError("pixel or world must be provided")

        pixel_t = self._coerce_optional_coord(pixel, fill=float("nan"))
        world_t = self._coerce_optional_coord(world, fill=float("nan"))
        if pixel_t.shape != world_t.shape:
            pixel_t, world_t = torch.broadcast_tensors(pixel_t, world_t)
            pixel_t = pixel_t.clone()
            world_t = world_t.clone()

        pixel_known = torch.isfinite(pixel_t)
        world_known = torch.isfinite(world_t)

        if bool(pixel_known.all()):
            world_full = self.pixel_to_world(pixel_t, origin=origin)
            if isinstance(world_full, tuple):
                world_full = torch.stack(world_full, dim=-1)
            return pixel_t, world_full
        if bool(world_known.all()):
            pixel_full = self.world_to_pixel(world_t, origin=origin)
            if isinstance(pixel_full, tuple):
                pixel_full = torch.stack(pixel_full, dim=-1)
            return pixel_full, world_t

        out_pixel = pixel_t.clone()
        out_world = world_t.clone()

        if not self._is_spatial:
            if self.coupled_tabular is not None:
                return self._mixed_coupled_tabular_pixel_world(
                    out_pixel,
                    out_world,
                    pixel_known=pixel_known,
                    world_known=world_known,
                    origin=origin,
                )
            if self.tabular_axes:
                if not self._tabular_axes_uncoupled_separable():
                    raise NotImplementedError(
                        "mixed tabular WCS transforms require explicit pixel or world coordinates"
                    )
                return self._mixed_separable_tabular_pixel_world(
                    out_pixel,
                    out_world,
                    pixel_known=pixel_known,
                    world_known=world_known,
                    origin=origin,
                )
            return self._mixed_linear_pixel_world(
                out_pixel,
                out_world,
                pixel_known=pixel_known,
                world_known=world_known,
                origin=origin,
            )

        if self.naxis >= 2 and self._is_spatial:
            spatial_pix_known = [bool(pixel_known[..., i].all()) for i in range(2)]
            spatial_world_known = [bool(world_known[..., i].all()) for i in range(2)]
            if all(spatial_pix_known):
                ra, dec = self._pixel_to_world_2d_fast(
                    out_pixel[..., 0], out_pixel[..., 1], origin=origin
                )
                out_world[..., 0] = ra
                out_world[..., 1] = dec
            elif all(spatial_world_known):
                x, y = self._world_to_pixel_2d_fast(
                    out_world[..., 0], out_world[..., 1], origin=origin
                )
                out_pixel[..., 0] = x
                out_pixel[..., 1] = y
            elif sum(spatial_pix_known) == 1 and sum(spatial_world_known) == 1:
                known_pixel_axis = 0 if spatial_pix_known[0] else 1
                solve_pixel_axis = 1 - known_pixel_axis
                known_world_axis = 0 if spatial_world_known[0] else 1
                solved = self._solve_mixed_spatial_axis(
                    known_pixel=out_pixel[..., known_pixel_axis],
                    target_world=out_world[..., known_world_axis],
                    known_pixel_axis=known_pixel_axis,
                    target_world_axis=known_world_axis,
                    origin=origin,
                    max_iter=max_iter,
                    tol=tol,
                )
                out_pixel[..., solve_pixel_axis] = solved
                ra, dec = self._pixel_to_world_2d_fast(
                    out_pixel[..., 0], out_pixel[..., 1], origin=origin
                )
                out_world[..., 0] = ra
                out_world[..., 1] = dec
            else:
                raise NotImplementedError(
                    "mixed spatial transforms require x/y, ra/dec, or one pixel "
                    "axis plus one world axis"
                )

        if self.naxis >= 3 and self._is_spatial:
            if self.coupled_tabular is not None:
                out_pixel, out_world = self._mixed_coupled_tabular_pixel_world(
                    out_pixel,
                    out_world,
                    pixel_known=pixel_known,
                    world_known=world_known,
                    origin=origin,
                )
            else:
                out_pixel, out_world = self._mixed_spatial_extra_pixel_world(
                    out_pixel,
                    out_world,
                    pixel_known=pixel_known,
                    world_known=world_known,
                    origin=origin,
                )

        if torch.isnan(out_pixel).any() or torch.isnan(out_world).any():
            raise NotImplementedError("mixed transform left unresolved axes")
        return out_pixel, out_world

    def _tabular_axes_uncoupled_separable(self) -> bool:
        """True when each tabular world axis depends only on its own pixel axis."""
        if self.coupled_tabular is not None:
            return False
        off = self.cd_full.clone()
        off.diagonal().zero_()
        return not bool((off.abs() > 1e-15).any().detach().cpu().item())

    def _mixed_coupled_tabular_pixel_world(
        self,
        out_pixel: Tensor,
        out_world: Tensor,
        *,
        pixel_known: Tensor,
        world_known: Tensor,
        origin: int,
    ) -> tuple[Tensor, Tensor]:
        """Fill coupled -TAB when every coupled pixel or every coupled world is known.

        Per-row masks group complete-pixel rows (forward) and complete-world rows
        (inverse). Underdetermined mixed coupled lookups stay a documented
        particular solution only when the full inverse is well-posed.
        """
        original_shape = out_pixel.shape
        pixel_flat = out_pixel.reshape(-1, self.naxis).clone()
        world_flat = out_world.reshape(-1, self.naxis).clone()
        pk = pixel_known.reshape(-1, self.naxis)
        wk = world_known.reshape(-1, self.naxis)
        pix_axes = list(self.coupled_tabular_pixel_axes)
        world_axes = list(self.coupled_tabular_axes)
        all_pix = pk[:, pix_axes].all(dim=-1)
        all_world = wk[:, world_axes].all(dim=-1)
        # Prefer forward when a row has a complete pixel sample.
        if bool(all_pix.any()):
            rows = pixel_flat[all_pix]
            world_fill = self.pixel_to_world(rows, origin=origin)
            if isinstance(world_fill, tuple):
                world_fill = torch.stack(world_fill, dim=-1)
            world_flat[all_pix] = world_fill
            pixel_flat[all_pix] = rows
        remaining_world = all_world & ~all_pix
        if bool(remaining_world.any()):
            rows_w = world_flat[remaining_world]
            pix_fill = self.world_to_pixel(rows_w, origin=origin)
            if isinstance(pix_fill, tuple):
                pix_fill = torch.stack(pix_fill, dim=-1)
            pixel_flat[remaining_world] = pix_fill
            world_flat[remaining_world] = rows_w
        leftover = ~(all_pix | remaining_world)
        if bool(leftover.any()):
            lookup = self.coupled_tabular
            if lookup is None:
                raise NotImplementedError(
                    "mixed coupled tabular transforms require a complete pixel sample "
                    "or a complete world sample for the coupled axes"
                )
            pix_t = torch.as_tensor(pix_axes, dtype=torch.long, device=pixel_flat.device)
            world_t = torch.as_tensor(
                world_axes, dtype=torch.long, device=pixel_flat.device
            )
            for row_index in torch.nonzero(leftover, as_tuple=False).reshape(-1).tolist():
                if not bool(pk[row_index, pix_t].any()) and not bool(
                    wk[row_index, world_t].any()
                ):
                    continue
                solved_pix, solved_world = _solve_coupled_tabular_partial(
                    lookup,
                    pixel_flat[row_index, pix_t],
                    world_flat[row_index, world_t],
                    pk[row_index, pix_t],
                    wk[row_index, world_t],
                )
                pixel_flat[row_index, pix_t] = solved_pix
                world_flat[row_index, world_t] = solved_world
        extra = [
            axis
            for axis in range(self.naxis)
            if axis not in pix_axes and axis not in world_axes
        ]
        if extra:
            for axis in extra:
                p_mask = pk[:, axis]
                w_mask = wk[:, axis]
                if bool(p_mask.any()):
                    world_axis = self._linear_axis_pixel_to_world(
                        pixel_flat[:, axis], axis, origin
                    )
                    world_flat[:, axis] = torch.where(
                        p_mask, world_axis, world_flat[:, axis]
                    )
                if bool(w_mask.any()):
                    pixel_axis = self._linear_axis_world_to_pixel(
                        world_flat[:, axis], axis, origin
                    )
                    pixel_flat[:, axis] = torch.where(
                        w_mask, pixel_axis, pixel_flat[:, axis]
                    )
        if torch.isnan(pixel_flat).any() or torch.isnan(world_flat).any():
            raise NotImplementedError(
                "mixed coupled tabular transforms require a complete pixel sample "
                "or a complete world sample for the coupled axes"
            )
        return pixel_flat.view(original_shape), world_flat.view(original_shape)

    def _mixed_separable_tabular_pixel_world(
        self,
        out_pixel: Tensor,
        out_world: Tensor,
        *,
        pixel_known: Tensor,
        world_known: Tensor,
        origin: int,
    ) -> tuple[Tensor, Tensor]:
        for axis in range(self.naxis):
            p_mask = pixel_known[..., axis]
            w_mask = world_known[..., axis]
            if bool(p_mask.any()):
                world_axis = self._linear_axis_pixel_to_world(
                    out_pixel[..., axis], axis, origin
                )
                out_world[..., axis] = torch.where(
                    p_mask, world_axis, out_world[..., axis]
                )
            if bool(w_mask.any()):
                pixel_axis = self._linear_axis_world_to_pixel(
                    out_world[..., axis], axis, origin
                )
                out_pixel[..., axis] = torch.where(
                    w_mask, pixel_axis, out_pixel[..., axis]
                )
        if torch.isnan(out_pixel).any() or torch.isnan(out_world).any():
            raise NotImplementedError("mixed transform left unresolved axes")
        return out_pixel, out_world

    def _mixed_spatial_extra_pixel_world(
        self,
        out_pixel: Tensor,
        out_world: Tensor,
        *,
        pixel_known: Tensor,
        world_known: Tensor,
        origin: int,
    ) -> tuple[Tensor, Tensor]:
        extra_axes = list(range(2, self.naxis))
        if self.tabular_axes or self.coupled_tabular is not None:
            for axis in extra_axes:
                p_mask = pixel_known[..., axis]
                w_mask = world_known[..., axis]
                if bool(p_mask.any()):
                    world_axis = self._linear_axis_pixel_to_world(
                        out_pixel[..., axis], axis, origin
                    )
                    out_world[..., axis] = torch.where(
                        p_mask, world_axis, out_world[..., axis]
                    )
                if bool(w_mask.any()):
                    pixel_axis = self._linear_axis_world_to_pixel(
                        out_world[..., axis], axis, origin
                    )
                    out_pixel[..., axis] = torch.where(
                        w_mask, pixel_axis, out_pixel[..., axis]
                    )
            return out_pixel, out_world

        original_shape = out_pixel.shape
        pixel_flat = out_pixel.reshape(-1, self.naxis).clone()
        world_flat = out_world.reshape(-1, self.naxis).clone()
        pk = pixel_known.reshape(-1, self.naxis)
        wk = world_known.reshape(-1, self.naxis)
        # Spatial pixels are already filled; only extra-axis known bits form groups.
        n_extra = len(extra_axes)
        powers = torch.arange(n_extra, device=pk.device, dtype=torch.int64)
        pix_bits = (
            pk[:, extra_axes].to(torch.int64) * (2**powers)
        ).sum(dim=-1)
        world_bits = (
            wk[:, extra_axes].to(torch.int64) * (2**powers)
        ).sum(dim=-1)
        group_key = pix_bits + (world_bits << n_extra)

        for key in torch.unique(group_key):
            row_mask = group_key == key
            key_i = int(key.item())
            pixel_axes_known = [0, 1] + [
                extra_axes[i] for i in range(n_extra) if key_i & (1 << i)
            ]
            world_axes_known = [
                extra_axes[i] for i in range(n_extra) if key_i & (1 << (n_extra + i))
            ]
            unknown_pixel_axes = [
                axis for axis in extra_axes if axis not in pixel_axes_known
            ]
            rows_p = pixel_flat[row_mask]
            rows_w = world_flat[row_mask]
            rel = rows_p + (1.0 - float(origin)) - self.crpix
            if unknown_pixel_axes and world_axes_known:
                known_pix = torch.as_tensor(
                    pixel_axes_known, dtype=torch.long, device=self.device
                )
                unknown_pix = torch.as_tensor(
                    unknown_pixel_axes, dtype=torch.long, device=self.device
                )
                known_world = torch.as_tensor(
                    world_axes_known, dtype=torch.long, device=self.device
                )
                rhs = rows_w[:, known_world] - self.crval[known_world]
                rhs = rhs - torch.matmul(
                    rel[:, known_pix], self.cd_full[known_world, :][:, known_pix].T
                )
                design = self.cd_full[known_world, :][:, unknown_pix]
                if design.shape[0] == design.shape[1]:
                    solved_rel = torch.linalg.solve(design, rhs.T).T
                else:
                    solved_rel = torch.linalg.lstsq(design, rhs.T).solution.T
                rel[:, unknown_pix] = solved_rel
                rows_p[:, unknown_pix] = (
                    solved_rel + self.crpix[unknown_pix] + (float(origin) - 1.0)
                )
            world_full = self.crval + torch.matmul(rel, self.cd_full.T)
            rows_w[:, 2:] = world_full[:, 2:]
            pixel_flat[row_mask] = rows_p
            world_flat[row_mask] = rows_w

        return pixel_flat.view(original_shape), world_flat.view(original_shape)

    def _mixed_linear_pixel_world(
        self,
        out_pixel: Tensor,
        out_world: Tensor,
        *,
        pixel_known: Tensor,
        world_known: Tensor,
        origin: int,
    ) -> tuple[Tensor, Tensor]:
        original_shape = out_pixel.shape
        pixel_flat = out_pixel.reshape(-1, self.naxis).clone()
        world_flat = out_world.reshape(-1, self.naxis).clone()
        pk = pixel_known.reshape(-1, self.naxis)
        wk = world_known.reshape(-1, self.naxis)
        n = self.naxis
        powers = torch.arange(n, device=pk.device, dtype=torch.int64)
        pix_bits = (pk.to(torch.int64) * (2**powers)).sum(dim=-1)
        world_bits = (wk.to(torch.int64) * (2**powers)).sum(dim=-1)
        group_key = pix_bits + (world_bits << n)

        for key in torch.unique(group_key):
            row_mask = group_key == key
            key_i = int(key.item())
            pixel_axes_known = [i for i in range(n) if key_i & (1 << i)]
            world_axes_known = [i for i in range(n) if key_i & (1 << (n + i))]
            unknown_pixel_axes = [i for i in range(n) if i not in pixel_axes_known]
            rows_p = pixel_flat[row_mask]
            rows_w = world_flat[row_mask]
            if len(unknown_pixel_axes) == 0:
                world_full = self.pixel_to_world(rows_p, origin=origin)
                if isinstance(world_full, tuple):
                    world_full = torch.stack(world_full, dim=-1)
                pixel_flat[row_mask] = rows_p
                world_flat[row_mask] = world_full
                continue
            if len(world_axes_known) == 0:
                continue
            rel = rows_p + (1.0 - float(origin)) - self.crpix
            known_pix = torch.as_tensor(
                pixel_axes_known, dtype=torch.long, device=self.device
            )
            unknown_pix = torch.as_tensor(
                unknown_pixel_axes, dtype=torch.long, device=self.device
            )
            known_world = torch.as_tensor(
                world_axes_known, dtype=torch.long, device=self.device
            )
            rhs = rows_w[:, known_world] - self.crval[known_world]
            if len(pixel_axes_known) > 0:
                rhs = rhs - torch.matmul(
                    rel[:, known_pix], self.cd_full[known_world, :][:, known_pix].T
                )
            design = self.cd_full[known_world, :][:, unknown_pix]
            if design.shape[0] == design.shape[1]:
                solved_rel = torch.linalg.solve(design, rhs.T).T
            else:
                solved_rel = torch.linalg.lstsq(design, rhs.T).solution.T
            rel[:, unknown_pix] = solved_rel
            rows_p[:, unknown_pix] = (
                solved_rel + self.crpix[unknown_pix] + (float(origin) - 1.0)
            )
            world_full = self.crval + torch.matmul(rel, self.cd_full.T)
            pixel_flat[row_mask] = rows_p
            world_flat[row_mask] = world_full

        return pixel_flat.view(original_shape), world_flat.view(original_shape)

    def _coerce_optional_coord(self, value: Any | None, *, fill: float) -> Tensor:
        if value is None:
            return torch.full(
                (self.naxis,), fill, dtype=self._wcs_coord_dtype(self.device), device=self.device
            )
        tensor = torch.as_tensor(value, dtype=self._wcs_coord_dtype(self.device), device=self.device)
        if tensor.shape == ():
            tensor = tensor.reshape(1)
        if tensor.shape[-1] != self.naxis:
            raise ValueError(
                f"coordinate arrays must have trailing dimension {self.naxis}"
            )
        return tensor

    def _linear_axis_pixel_to_world(
        self, pixel: Tensor, axis: int, origin: int
    ) -> Tensor:
        rel = pixel + (1.0 - origin) - self.crpix[axis]
        linear_world = self.crval[axis] + rel * self.cdelt[axis]
        return self._apply_tabular_pixel_to_world(linear_world, axis)

    def _linear_axis_world_to_pixel(
        self, world: Tensor, axis: int, origin: int
    ) -> Tensor:
        linear_world = self._apply_tabular_world_to_pixel(world, axis)
        rel = (linear_world - self.crval[axis]) / self.cdelt[axis]
        return rel + self.crpix[axis] - (1.0 - origin)

    def _solve_mixed_spatial_axis(
        self,
        *,
        known_pixel: Tensor,
        target_world: Tensor,
        known_pixel_axis: int,
        target_world_axis: int,
        origin: int,
        max_iter: int,
        tol: float,
    ) -> Tensor:
        solve_axis = 1 - known_pixel_axis
        guess = torch.full_like(
            target_world,
            float(self.crpix[solve_axis].item() - (1.0 - origin)),
        )
        eps = torch.as_tensor(1.0e-4, dtype=self._wcs_coord_dtype(self.device), device=self.device)
        for _ in range(max_iter):
            x = known_pixel if known_pixel_axis == 0 else guess
            y = guess if known_pixel_axis == 0 else known_pixel
            ra, dec = self._pixel_to_world_2d_fast(x, y, origin=origin)
            value = ra if target_world_axis == 0 else dec
            resid = (
                _wrap_lon180(value - target_world)
                if target_world_axis == 0
                else value - target_world
            )
            if float(torch.max(torch.abs(resid)).detach().cpu().item()) <= tol:
                break
            x_eps = known_pixel if known_pixel_axis == 0 else guess + eps
            y_eps = guess + eps if known_pixel_axis == 0 else known_pixel
            ra_eps, dec_eps = self._pixel_to_world_2d_fast(x_eps, y_eps, origin=origin)
            value_eps = ra_eps if target_world_axis == 0 else dec_eps
            diff = (
                _wrap_lon180(value_eps - value)
                if target_world_axis == 0
                else value_eps - value
            )
            deriv = diff / eps
            valid = torch.abs(deriv) > 1e-14
            step = torch.where(
                valid,
                resid / torch.where(valid, deriv, torch.ones_like(deriv)),
                torch.zeros_like(resid),
            )
            guess = guess - step
        return guess

    def pixel_to_world_differentiable(self, pixels: Any, *, origin: int = 0) -> Tensor:
        """Pure-Torch pixel-to-world transform intended for optimization loops.

        This path deliberately bypasses C++ spherical-rotation shortcuts so gradients
        can flow through WCS evaluation when fitting fields, INRs, or observation
        geometry. For now the guaranteed nonlinear spatial path is 2D celestial WCS;
        higher-dimensional non-spatial axes use the existing linear FITS terms.
        """
        pixels_t = self._coerce_stacked_coord(pixels, "pixels")
        original_shape = pixels_t.shape
        pixels_flat = self._contiguous_if_needed(pixels_t).reshape(-1, self.naxis)
        if self.naxis == 2 and self._is_spatial:
            world_flat = self._stacked_pixels_to_world_flat(
                pixels_flat, float(origin), allow_cpp_sphere=False
            )
        else:
            rel_flat = pixels_flat + (1.0 - float(origin)) - self.crpix
            world_flat = self._transform_optimized(rel_flat, allow_cpp_sphere=False)
        return world_flat.view(original_shape)

    def world_to_pixel_differentiable(self, world: Any, *, origin: int = 0) -> Tensor:
        """Pure-Torch world-to-pixel transform intended for optimization loops."""
        world_t = self._coerce_stacked_coord(world, "world")
        original_shape = world_t.shape
        world_flat = self._contiguous_if_needed(world_t).reshape(-1, self.naxis)
        if self.naxis == 2 and self._is_spatial:
            pix_flat = self._stacked_world_to_pixel_flat(
                world_flat, float(origin), allow_cpp_inverse_sphere=False
            )
        elif self._is_spatial:
            pix_flat = torch.empty_like(world_flat)
            spatial_pix = self._stacked_world_to_pixel_flat(
                world_flat[:, :2], float(origin), allow_cpp_inverse_sphere=False
            )
            pix_flat[:, :2] = spatial_pix
            if self.naxis >= 3:
                pix_internal_spatial = spatial_pix + (1.0 - float(origin))
                pix_internal_extra = self._invert_extra_axes(
                    world_flat, pix_internal_spatial
                )
                pix_flat[:, 2:] = pix_internal_extra + (float(origin) - 1.0)
        else:
            # Non-spatial (including nonlinear -TAB): same inverse as world_to_pixel.
            linear_world = world_flat.clone()
            coupled_pixel = self._apply_coupled_tabular_world_to_pixel(world_flat)
            for i in range(self.naxis):
                linear_world[:, i] = self._coupled_or_tabular_linear_world(
                    coupled_pixel, world_flat, i
                )
            rel_flat = torch.matmul(self.cd_inv_full, (linear_world - self.crval).T).T
            pix_flat = rel_flat + self.crpix + (float(origin) - 1.0)
        return pix_flat.view(original_shape)

    def pixel_to_world_jacobian(
        self,
        pixels: Any,
        *,
        origin: int = 0,
        create_graph: bool = False,
    ) -> Tensor:
        """Return local ``d(world)/d(pixel)`` Jacobians.

        The output shape is ``pixels.shape[:-1] + (world_n_dim, pixel_n_dim)``.
        """
        pixels_t = self._coerce_stacked_coord(pixels, "pixels")
        coords = pixels_t.clone().detach().requires_grad_(True)
        world = self.pixel_to_world_differentiable(coords, origin=origin)
        grads = []
        for axis in range(self.world_n_dim):
            grad = torch.autograd.grad(
                world[..., axis].sum(),
                coords,
                retain_graph=True,
                create_graph=create_graph,
            )[0]
            grads.append(grad)
        return torch.stack(grads, dim=-2)

    def world_to_pixel_jacobian(
        self,
        world: Any,
        *,
        origin: int = 0,
        create_graph: bool = False,
    ) -> Tensor:
        """Return local ``d(pixel)/d(world)`` Jacobians."""
        world_t = self._coerce_stacked_coord(world, "world")
        if self.coupled_tabular is not None and self.pixel_n_dim == self.world_n_dim:
            pixels = self.world_to_pixel(world_t, origin=origin)
            forward_jacobian = self.pixel_to_world_jacobian(
                pixels,
                origin=origin,
                create_graph=create_graph,
            )
            return torch.linalg.inv(forward_jacobian)
        coords = world_t.clone().detach().requires_grad_(True)
        pixels = self.world_to_pixel_differentiable(coords, origin=origin)
        grads = []
        for axis in range(self.pixel_n_dim):
            grad = torch.autograd.grad(
                pixels[..., axis].sum(),
                coords,
                retain_graph=True,
                create_graph=create_graph,
            )[0]
            grads.append(grad)
        return torch.stack(grads, dim=-2)

    def pixel_to_world_values(
        self, *pixel_arrays: Any, origin: int = 0
    ) -> tuple[Tensor, ...]:
        """Astropy-compatible value alias for pixel-to-world coordinates."""
        result = self.pixel_to_world(*pixel_arrays, origin=origin)
        if isinstance(result, tuple):
            return result
        return tuple(result[..., i] for i in range(self.world_n_dim))

    def world_to_pixel_values(
        self, *world_arrays: Any, origin: int = 0
    ) -> tuple[Tensor, ...]:
        """Astropy-compatible value alias for world-to-pixel coordinates."""
        result = self.world_to_pixel(*world_arrays, origin=origin)
        if isinstance(result, tuple):
            return result
        return tuple(result[..., i] for i in range(self.pixel_n_dim))

    def all_pix2world(
        self, *args: Any, origin: int | None = None
    ) -> Tensor | tuple[Tensor, ...]:
        """Astropy-compatible alias for the full pixel-to-world transform.

        Supports both ``all_pix2world(pixels, origin)`` and
        ``all_pix2world(x, y, ..., origin)`` call styles.
        """
        pixel_args, origin_value = self._args_with_optional_origin(args, origin=origin)
        return self.pixel_to_world(*pixel_args, origin=origin_value)

    def all_world2pix(
        self, *args: Any, origin: int | None = None
    ) -> Tensor | tuple[Tensor, ...]:
        """Astropy-compatible alias for the full world-to-pixel transform."""
        world_args, origin_value = self._args_with_optional_origin(args, origin=origin)
        return self.world_to_pixel(*world_args, origin=origin_value)

    def _args_with_optional_origin(
        self, args: tuple[Any, ...], *, origin: int | None
    ) -> tuple[tuple[Any, ...], int]:
        if origin is not None:
            if len(args) == 0:
                raise ValueError("coordinate arguments are required")
            return args, int(origin)
        if len(args) >= 2 and self._is_scalar_origin_arg(args[-1]):
            coord_args = args[:-1]
            if len(coord_args) in {1, self.naxis}:
                return coord_args, int(args[-1])
        if len(args) == 0:
            raise ValueError("coordinate arguments are required")
        return args, 0

    @staticmethod
    def _is_scalar_origin_arg(value: Any) -> bool:
        if isinstance(value, (int, float, np.integer, np.floating)):
            return True
        if isinstance(value, Tensor):
            return value.ndim == 0
        return False

    def array_index_to_world_values(
        self, *index_arrays: Any, origin: int = 0
    ) -> tuple[Tensor, ...]:
        """Convert NumPy-order array indices to world coordinate values.

        Array index order is the reverse of FITS pixel-axis order, so a 2D image
        accepts ``(row, column)`` and maps that to pixel ``(x, y)``.
        """
        if len(index_arrays) != self.pixel_n_dim:
            raise ValueError(f"expected {self.pixel_n_dim} array index arrays")
        pixels = [
            torch.as_tensor(
                index_arrays[self.pixel_n_dim - 1 - i],
                device=self.device,
                dtype=self._wcs_coord_dtype(self.device),
            )
            for i in range(self.pixel_n_dim)
        ]
        return self.pixel_to_world_values(*pixels, origin=origin)

    def array_index_to_world(
        self, *index_arrays: Any, origin: int = 0
    ) -> tuple[Tensor, ...]:
        """Alias for ``array_index_to_world_values``."""
        return self.array_index_to_world_values(*index_arrays, origin=origin)

    def world_to_array_index_values(
        self, *world_arrays: Any, origin: int = 0
    ) -> tuple[Tensor, ...]:
        """Convert world coordinates to nearest integer NumPy-order indices."""
        pixels = self.world_to_pixel_values(*world_arrays, origin=origin)
        return tuple(
            torch.round(pixels[self.pixel_n_dim - 1 - i]).to(dtype=torch.long)
            for i in range(self.pixel_n_dim)
        )

    def world_to_array_index(
        self, *world_arrays: Any, origin: int = 0
    ) -> tuple[Tensor, ...]:
        """Alias for ``world_to_array_index_values``."""
        return self.world_to_array_index_values(*world_arrays, origin=origin)

    def slice(
        self,
        view: "slice | int | tuple[slice | int | type(Ellipsis), ...]",
        *,
        numpy_order: bool = True,
    ) -> "WCS":
        """Return a WCS adjusted for an array slice/cutout.

        By default ``view`` is in NumPy array order, so a 2D image slice is
        ``(rows, columns)`` while FITS pixel axes are updated as ``(x, y)``.
        Positive-step and negative-step slice objects preserve dimensionality. Integer indices
        fix one pixel coordinate and drop the corresponding uncoupled WCS axis.
        """
        pixel_items = self._normalize_wcs_slice(view, numpy_order=numpy_order)
        if all(isinstance(item, slice) for item in pixel_items):
            return self._slice_preserve_axes(pixel_items)

        fixed_axes: list[int] = []
        slice_items: list[slice] = []
        for axis, item in enumerate(pixel_items):
            axis_len = None if self._pixel_shape is None else self._pixel_shape[axis]
            if isinstance(item, slice):
                slice_items.append(item)
                continue
            fixed_axes.append(axis)
            index = self._normalize_axis_index(item, axis_len)
            slice_items.append(slice(index, index + 1, 1))

        sliced = self._slice_preserve_axes(tuple(slice_items))
        for axis in sorted(fixed_axes, reverse=True):
            sliced = sliced._dropaxis_fixed_at_zero(axis)
        return sliced

    def _slice_preserve_axes(self, pixel_slices: tuple[slice, ...]) -> "WCS":
        starts: list[int] = []
        steps: list[int] = []
        new_shape: list[int] | None = [] if self._pixel_shape is not None else None
        for axis, axis_slice in enumerate(pixel_slices):
            axis_len = None if self._pixel_shape is None else self._pixel_shape[axis]
            start, stop, step = self._normalize_axis_slice(axis_slice, axis_len)
            starts.append(start)
            steps.append(step)
            if new_shape is not None:
                if stop is None:
                    stop = axis_len
                assert stop is not None
                new_shape.append(len(range(start, stop, step)))

        step_t = torch.as_tensor(steps, dtype=self._wcs_coord_dtype(self.device), device=self.device)
        start_t = torch.as_tensor(starts, dtype=self._wcs_coord_dtype(self.device), device=self.device)
        new_crpix = 1.0 - (start_t + 1.0 - self.crpix) / step_t
        new_cd = self.cd_full @ torch.diag(step_t)

        header = self._copy_wcs_header()
        header["WCSAXES"] = self.naxis
        header["NAXIS"] = self.naxis
        for i in range(1, self.naxis + 1):
            header[f"CRPIX{i}"] = float(new_crpix[i - 1].detach().cpu().item())
            for j in range(1, self.naxis + 1):
                header[f"CD{i}_{j}"] = float(new_cd[i - 1, j - 1].detach().cpu().item())
            header.pop(f"CDELT{i}", None)
            for j in range(1, self.naxis + 1):
                header.pop(f"PC{i}_{j}", None)
            if new_shape is not None:
                header[f"NAXIS{i}"] = int(new_shape[i - 1])

        tabular_axes = {
            axis + 1: tabular for axis, tabular in self.tabular_axes.items()
        }
        return WCS(header, key=self.wcs_key, tabular_axes=tabular_axes)

    def _copy_wcs_header(self) -> dict[str, Any]:
        header: dict[str, Any] = {}
        for key, value in self.wcs_params.items():
            if isinstance(value, (str, int, float, bool)):
                header[str(key)] = value
        if self._pixel_shape is not None:
            header["NAXIS"] = self.naxis
            for i, size in enumerate(self._pixel_shape, start=1):
                header[f"NAXIS{i}"] = int(size)
        return header

    def to_header(self) -> dict[str, Any]:
        """Return a plain FITS-WCS header dictionary for this WCS."""
        return self._copy_wcs_header()

    def with_pixel_affine_correction(
        self,
        matrix: Any,
        offset: Any,
        *,
        metadata_prefix: str = "TSKYAF",
    ) -> "WCS":
        """Return a WCS corrected by a fitted 2D pixel affine transform.

        ``matrix`` and ``offset`` map old/source zero-based pixel coordinates to
        corrected zero-based pixel coordinates: ``p_new = matrix @ p_old +
        offset``. The returned WCS maps corrected pixels to the same world
        coordinates as the original WCS at the corresponding source pixels.

        The update is exact for distortion-free WCS with uncoupled axes beyond
        the first two pixel axes. Distorted WCS store the affine and apply it in
        pixel space before SIP/TPV/TNX/ZPX distortion.
        """
        if self.naxis < 2:
            raise ValueError("pixel affine correction requires at least two WCS axes")
        affine = torch.as_tensor(matrix, dtype=self._wcs_coord_dtype(self.device), device=self.device)
        shift = torch.as_tensor(offset, dtype=self._wcs_coord_dtype(self.device), device=self.device)
        if affine.shape != (2, 2):
            raise ValueError("matrix must have shape (2, 2)")
        if shift.shape != (2,):
            raise ValueError("offset must have shape (2,)")
        affine_inv = torch.linalg.inv(affine)
        extra_coupled = False
        for axis in range(2, self.naxis):
            row = self.cd_full[axis, :2]
            col = self.cd_full[:2, axis]
            extra_coupled = extra_coupled or bool(
                torch.any(torch.abs(row) > 1e-15)
                or torch.any(torch.abs(col) > 1e-15)
            )
        distorted = (
            self.sip is not None or self._is_tpv or self._is_tnx or self._is_zpx
        )

        header = self._copy_wcs_header()
        for i in range(1, self.naxis + 1):
            for j in range(1, self.naxis + 1):
                header[f"CD{i}_{j}"] = float(
                    self.cd_full[i - 1, j - 1].detach().cpu().item()
                )
            header.pop(f"CDELT{i}", None)
            for j in range(1, self.naxis + 1):
                header.pop(f"PC{i}_{j}", None)

            if not distorted:
                old_crpix = self.crpix[:2]
                new_crpix = 1.0 + shift + affine @ (old_crpix - 1.0)
                new_cd_2d = self.cd_full[:2, :2] @ affine_inv
                for i in range(2):
                    header[f"CRPIX{i + 1}"] = float(new_crpix[i].detach().cpu().item())
                    for j in range(2):
                        header[f"CD{i + 1}_{j + 1}"] = float(
                            new_cd_2d[i, j].detach().cpu().item()
                        )
            if extra_coupled:
                extra_spatial = self.cd_full[2:, :2] @ affine_inv
                for i in range(2, self.naxis):
                    for j in range(2):
                        header[f"CD{i + 1}_{j + 1}"] = float(
                            extra_spatial[i - 2, j].detach().cpu().item()
                        )

        prefix = str(metadata_prefix).strip().upper()
        if prefix:
            header[f"{prefix}0"] = "pixel_affine"
            for i in range(2):
                header[f"{prefix}B{i + 1}"] = float(shift[i].detach().cpu().item())
                for j in range(2):
                    header[f"{prefix}A{i + 1}{j + 1}"] = float(
                        affine[i, j].detach().cpu().item()
                    )
        tabular_axes = {
            axis + 1: tabular for axis, tabular in self.tabular_axes.items()
        }
        corrected = WCS(header, key=self.wcs_key, tabular_axes=tabular_axes)
        if prefix:
            for key, value in header.items():
                if str(key).startswith(prefix):
                    corrected.wcs_params[str(key)] = value
        if distorted:
            corrected._set_pixel_affine(affine, shift)
        return corrected

    def with_affine_fit_correction(
        self, fit: Any, *, metadata_prefix: str = "TSKYAF"
    ) -> "WCS":
        """Return a WCS corrected by an ``AffineFitResult`` pixel fit."""
        return self.with_pixel_affine_correction(
            fit.matrix,
            fit.offset,
            metadata_prefix=metadata_prefix,
        )

    def _normalize_wcs_slice(
        self,
        view: "slice | int | tuple[slice | int | type(Ellipsis), ...]",
        *,
        numpy_order: bool,
    ) -> "tuple[slice | int, ...]":
        if isinstance(view, (slice, int)) and not isinstance(view, bool):
            items: tuple[slice | int | type(Ellipsis), ...] = (view,)
        else:
            items = tuple(view)
        if items.count(Ellipsis) > 1:
            raise ValueError("slice view may contain at most one ellipsis")
        if Ellipsis in items:
            ellipsis_index = items.index(Ellipsis)
            missing = self.naxis - (len(items) - 1)
            items = (
                items[:ellipsis_index]
                + (slice(None),) * missing
                + items[ellipsis_index + 1 :]
            )
        if len(items) > self.naxis:
            raise ValueError(f"slice view has {len(items)} axes for {self.naxis}D WCS")
        items = items + (slice(None),) * (self.naxis - len(items))
        if not all(
            isinstance(item, slice)
            or (isinstance(item, int) and not isinstance(item, bool))
            for item in items
        ):
            raise TypeError("WCS.slice items must be slices, integers, or ellipsis")
        pixel_items = tuple(reversed(items)) if numpy_order else items
        return tuple(pixel_items)

    @staticmethod
    def _normalize_axis_slice(
        axis_slice: slice, axis_len: int | None
    ) -> tuple[int, int | None, int]:
        step = 1 if axis_slice.step is None else int(axis_slice.step)
        if step == 0:
            raise ValueError("WCS.slice step cannot be zero")
        if axis_len is not None:
            start, stop, step = axis_slice.indices(int(axis_len))
            return start, stop, step
        if step < 0:
            raise ValueError("negative slice steps require known NAXISi shape")
        start = 0 if axis_slice.start is None else int(axis_slice.start)
        stop = None if axis_slice.stop is None else int(axis_slice.stop)
        if start < 0 or (stop is not None and stop < 0):
            raise ValueError("negative slice starts require known NAXISi shape")
        return start, stop, step

    @staticmethod
    def _normalize_axis_index(axis_index: int, axis_len: int | None) -> int:
        idx = int(axis_index)
        if idx < 0:
            if axis_len is None:
                raise ValueError("negative integer indices require known NAXISi shape")
            idx += axis_len
        if axis_len is not None and (idx < 0 or idx >= axis_len):
            raise IndexError(f"WCS.slice integer index {axis_index!r} is out of bounds")
        return idx

    def dropaxis(self, axis: int) -> "WCS":
        """Return a WCS with one pixel/world axis removed.

        The axis is 0-based in FITS pixel-axis order. Coupled linear and celestial
        axes are baked at CRPIX (``fixed_rel=0``) so the retained transform matches
        the parent WCS on that slice.
        """
        idx = int(axis)
        if idx < 0:
            idx += self.naxis
        if idx < 0 or idx >= self.naxis:
            raise ValueError(f"axis {axis!r} outside WCS axes")
        if self.naxis <= 1:
            raise ValueError("cannot drop the only WCS axis")
        keep = [i for i in range(self.naxis) if i != idx]
        # Bake at CRPIX: relative pixel coordinate of the dropped axis is 0.
        return self._sub_wcs(keep, fixed_rel_by_axis={idx: 0.0})

    def sub(self, axes: Sequence[int]) -> "WCS":
        """Return a WCS containing selected pixel/world axes in the requested order.

        Axes are 0-based in FITS pixel-axis order. Dropped axes must be linearly
        uncoupled from retained axes. Spatial WCS instances keep the celestial
        projection axes as the leading pair; extra uncoupled axes may be dropped
        or reordered after them.
        """
        keep: list[int] = []
        for axis in axes:
            idx = int(axis)
            if idx < 0:
                idx += self.naxis
            if idx < 0 or idx >= self.naxis:
                raise ValueError(f"axis {axis!r} outside WCS axes")
            keep.append(idx)
        return self._sub_wcs(keep)

    @property
    def celestial(self) -> "WCS":
        """Return the 2D celestial component of a spatial WCS."""
        if not self._is_spatial or self.naxis < 2:
            raise ValueError("celestial WCS requires spatial axes")
        for axis in range(2, self.naxis):
            self._validate_axis_uncoupled_for_drop(axis)
        return self._sub_wcs([0, 1])

    def _validate_axis_uncoupled_for_drop(self, axis: int) -> None:
        row = self.cd_full[axis, :].clone()
        col = self.cd_full[:, axis].clone()
        row[axis] = 0.0
        col[axis] = 0.0
        coupled = torch.any(torch.abs(row) > 1e-15) or torch.any(torch.abs(col) > 1e-15)
        if bool(coupled.detach().cpu().item()):
            raise NotImplementedError("cannot drop a linearly coupled WCS axis")

    def _dropaxis_fixed_at_zero(self, axis: int) -> "WCS":
        """Drop an axis fixed by slicing at zero-based pixel coordinate 0."""
        idx = int(axis)
        if idx < 0:
            idx += self.naxis
        if idx < 0 or idx >= self.naxis:
            raise ValueError(f"axis {axis!r} outside WCS axes")
        if self.naxis <= 1:
            raise ValueError("cannot drop the only WCS axis")
        fixed_rel = 1.0 - self.crpix[idx]
        keep = [i for i in range(self.naxis) if i != idx]
        return self._sub_wcs(keep, fixed_rel_by_axis={idx: fixed_rel})

    def _sub_wcs(
        self,
        keep_axes: list[int],
        *,
        fixed_rel_by_axis: dict[int, Tensor] | None = None,
    ) -> "WCS":
        if len(keep_axes) == 0:
            raise ValueError("at least one WCS axis must be kept")
        if len(set(keep_axes)) != len(keep_axes):
            raise ValueError("kept axes must be unique")
        fixed_rel_by_axis = {} if fixed_rel_by_axis is None else fixed_rel_by_axis
        if self._is_spatial:
            celestial = [a for a in (0, 1) if a < self.naxis]
            dropped_celestial = [a for a in celestial if a not in keep_axes]
            if dropped_celestial:
                if any(a not in fixed_rel_by_axis for a in dropped_celestial):
                    raise NotImplementedError(
                        "spatial WCS subsetting must keep the first two celestial axes"
                    )
            elif len(keep_axes) >= 2 and keep_axes[:2] != [0, 1]:
                raise NotImplementedError(
                    "spatial WCS subsetting must keep the first two celestial axes"
                )
        dropped_axes = [axis for axis in range(self.naxis) if axis not in keep_axes]
        for axis in dropped_axes:
            if axis not in fixed_rel_by_axis:
                self._validate_axis_uncoupled_for_drop(axis)

        header: dict[str, Any] = {"NAXIS": len(keep_axes), "WCSAXES": len(keep_axes)}
        global_keys = (
            "LONPOLE",
            "LATPOLE",
            "RADESYS",
            "RADECSYS",
            "EQUINOX",
            "EPOCH",
            "RESTFRQ",
            "RESTFREQ",
            "RESTWAV",
            "SPECSYS",
            "SSYSOBS",
            "SSYSSRC",
            "VELREF",
        )
        for key in global_keys:
            if key in self.wcs_params:
                header[key] = self.wcs_params[key]

        cd_sub = self.cd_full[keep_axes, :][:, keep_axes]
        dropped_celestial = [
            a for a in (0, 1) if a < self.naxis and a not in keep_axes
        ]
        world_ref: Tensor | None = None
        if self._is_spatial and dropped_celestial:
            # Bake celestial drops at the fixed pixel via a real forward sample so
            # CRVAL matches the parent slice; strip projection codes so the child
            # is non-spatial (remaining axes are no longer a celestial pair).
            ref = (self.crpix - 1.0).clone()
            for fixed_axis, fixed_rel in fixed_rel_by_axis.items():
                ref[fixed_axis] = (
                    torch.as_tensor(
                        fixed_rel, dtype=self.crpix.dtype, device=self.device
                    )
                    + self.crpix[fixed_axis]
                    - 1.0
                )
            world_ref = self.pixel_to_world(ref.unsqueeze(0), origin=0)
            if isinstance(world_ref, tuple):
                world_ref = torch.stack(world_ref, dim=-1)
            world_ref = world_ref.reshape(-1, self.naxis)[0]
            # Local d(world)/d(pixel) on the kept axes at the bake-in point.
            eps = 1.0e-3
            jac_cols = []
            for keep_axis in keep_axes:
                pert = ref.clone()
                pert[keep_axis] = pert[keep_axis] + eps
                world_pert = self.pixel_to_world(pert.unsqueeze(0), origin=0)
                if isinstance(world_pert, tuple):
                    world_pert = torch.stack(world_pert, dim=-1)
                world_pert = world_pert.reshape(-1, self.naxis)[0]
                jac_cols.append((world_pert - world_ref) / eps)
            jac = torch.stack(jac_cols, dim=-1)
            cd_sub = jac[keep_axes, :]

        for new_axis, old_axis in enumerate(keep_axes, start=1):
            old_num = old_axis + 1
            ctype = str(self.wcs_params.get(f"CTYPE{old_num}", ""))
            if self._is_spatial and dropped_celestial and old_axis < 2:
                ctype = _ctype_without_projection(ctype)
            header[f"CTYPE{new_axis}"] = ctype
            header[f"CRPIX{new_axis}"] = float(
                self.crpix[old_axis].detach().cpu().item()
            )
            if world_ref is not None:
                crval = world_ref[old_axis]
            else:
                crval = self.crval[old_axis]
                for fixed_axis, fixed_rel in fixed_rel_by_axis.items():
                    crval = crval + self.cd_full[old_axis, fixed_axis] * fixed_rel
            header[f"CRVAL{new_axis}"] = float(crval.detach().cpu().item())
            if f"CUNIT{old_num}" in self.wcs_params:
                header[f"CUNIT{new_axis}"] = self.wcs_params[f"CUNIT{old_num}"]
            if self._pixel_shape is not None:
                header[f"NAXIS{new_axis}"] = int(self._pixel_shape[old_axis])
            for out_axis, _old_out in enumerate(keep_axes, start=1):
                header[f"CD{new_axis}_{out_axis}"] = float(
                    cd_sub[new_axis - 1, out_axis - 1].detach().cpu().item()
                )

        if self._is_spatial and keep_axes[:2] == [0, 1] and not dropped_celestial:
            for key, value in self.wcs_params.items():
                key_s = str(key)
                if key_s.startswith(("PV1_", "PV2_")):
                    header[key_s] = value

        tabular_axes = {
            keep_axes.index(axis) + 1: tabular
            for axis, tabular in self.tabular_axes.items()
            if axis in keep_axes
        }
        return WCS(header, key=self.wcs_key, tabular_axes=tabular_axes)

    def _coerce_stacked_coord(self, value: Any, name: str) -> Tensor:
        tensor = self._as_wcs_coord(value)
        if tensor.shape == ():
            raise ValueError(f"{name} must have trailing dimension {self.naxis}")
        if tensor.shape[-1] != self.naxis:
            raise ValueError(f"{name} must have trailing dimension {self.naxis}")
        return tensor

    def pixel_to_world(self, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Convert pixel coordinates to world coordinates.
        Args:
            origin: FITS origin (default 0). Set to 1 for FITS 1-based indexing.
        """
        origin = float(kwargs.get("origin", 0))
        if len(args) == 2 and self.naxis == 2:
            x_t = self._as_wcs_coord(args[0])
            y_t = self._as_wcs_coord(args[1])
            ra, dec = self._pixel_to_world_2d_fast(x_t, y_t, origin)
            return ra, dec

        if len(args) == 1:
            pixels = self._as_wcs_coord(args[0])
        else:
            pixels = torch.stack(
                [self._as_wcs_coord(a) for a in args],
                dim=-1,
            )

        original_shape = pixels.shape
        pixels_flat = self._contiguous_if_needed(pixels).reshape(-1, self.naxis)

        stack_compiled = self._compiled_stack_pixel_to_world
        if (
            stack_compiled is not None
            and self.naxis == 2
            and self._is_spatial
            and origin in stack_compiled
        ):
            # Fused ``pixels -> rel -> world`` + Torch-only pole rotation (see ``compile``).
            world_flat = stack_compiled[origin](pixels_flat)
        else:
            # If origin=0, rel = pix + 1 - crpix; if origin=1, rel = pix - crpix
            rel_flat = pixels_flat + (1 - origin) - self.crpix
            transform_fn = self._compiled_transform_fn or self._transform_optimized
            world_flat = transform_fn(rel_flat)

        world = world_flat.view(original_shape)

        # If input was separate args, return separate results if requested or standard
        if len(args) > 1:
            return tuple(world[..., i] for i in range(self.naxis))
        return world

    def _transform_optimized(
        self, rel: Tensor, *, allow_cpp_sphere: bool = True
    ) -> Tensor:
        """
        The hot path for coordinate transformation.
        Input: (N, NAXIS) relative pixels (pix - crpix [internal])
        Output: (N, NAXIS) world

        When ``allow_cpp_sphere`` is False, pole spherical rotation stays in Torch so
        ``torch.compile`` can fuse the full forward graph (stacked pixel API).
        """
        # 1. Linear Transform (CD @ rel) for first 2 axes
        rel_x = rel[:, 0]
        rel_y = rel[:, 1] if self.naxis >= 2 else torch.zeros_like(rel_x)

        if self._pixel_affine_matrix is not None and self.naxis >= 2:
            # Affine is on 0-based pixels; rel = pix_internal - crpix, pix_0 = pix_internal - 1.
            p0 = rel_x + self.crpix[0] - 1.0
            p1 = rel_y + self.crpix[1] - 1.0
            src_x, src_y = self._corrected_pixels_to_source(p0, p1, origin=0.0)
            rel_x = src_x - (self.crpix[0] - 1.0)
            rel_y = src_y - (self.crpix[1] - 1.0)
            if self.naxis == 2:
                rel = torch.stack((rel_x, rel_y), dim=-1)
            else:
                rel = rel.clone()
                rel[:, 0] = rel_x
                rel[:, 1] = rel_y

        if (self.det2im1 is not None or self.det2im2 is not None) and self.naxis >= 2:
            x1 = rel_x + self.crpix[0]
            y1 = rel_y + self.crpix[1]
            x1, y1 = self._apply_det2im_forward(x1, y1)
            rel_x = x1 - self.crpix[0]
            rel_y = y1 - self.crpix[1]

        # Check if first 2 axes are spatial (have projection code)
        is_spatial = self._is_spatial

        if not is_spatial:
            linear_world = self.crval + torch.matmul(rel, self.cd_full.T)
            if (
                self.coupled_tabular is not None
                and self.coupled_tabular_axes == tuple(range(self.naxis))
                and self.coupled_tabular_pixel_axes == tuple(range(self.naxis))
            ):
                return self.coupled_tabular.pixel_to_world(linear_world)
            coupled_world = self._apply_coupled_tabular_pixel_to_world(linear_world)
            results = [
                coupled_world.get(
                    i, self._apply_tabular_pixel_to_world(linear_world[:, i], i)
                )
                for i in range(self.naxis)
            ]
        else:
            # Distortion & Linear
            if self._is_tpv:
                u = self.cd[0, 0] * rel_x + self.cd[0, 1] * rel_y
                v = self.cd[1, 0] * rel_x + self.cd[1, 1] * rel_y
                xi, eta = self.tpv.distort(u, v)
            else:
                if self.sip is not None:
                    rel_x, rel_y = self.sip.distort(rel_x, rel_y)
                if self.cpdis1 is not None or self.cpdis2 is not None:
                    rel_x, rel_y = self._apply_cpdis_forward(rel_x, rel_y)

                # Linear Transform (CD @ coords)
                xi = self.cd[0, 0] * rel_x + self.cd[0, 1] * rel_y
                eta = self.cd[1, 0] * rel_x + self.cd[1, 1] * rel_y

            if self._is_gls:
                ra, dec = self._gls_forward(xi, eta)
            else:
                # Projection
                phi, theta = self._project_native_coords(xi, eta)

                # Spherical Rotation (result is in degrees)
                ra, dec = self._spherical_rotation_optimized(
                    phi, theta, self._native_type, allow_cpp_sphere=allow_cpp_sphere
                )
            results = [ra, dec]

        # Handle NAXIS > 2 (e.g. Spectral axis 3)
        if self.naxis >= 3:
            linear_extra = self.crval + torch.matmul(rel, self.cd_full.T)
            coupled_world = self._apply_coupled_tabular_pixel_to_world(linear_extra)
            for i in range(2, self.naxis):
                # Basic linear transformation for non-spatial axes
                val = linear_extra[:, i]
                results.append(
                    coupled_world.get(i, self._apply_tabular_pixel_to_world(val, i))
                )

        return torch.stack(results[: self.naxis], dim=-1)

    def _stacked_pixels_to_world_flat(
        self,
        pixels_flat: Tensor,
        origin: float,
        *,
        allow_cpp_sphere: bool = True,
    ) -> Tensor:
        """Fused ``pixels (internal) -> rel -> world`` for ``torch.compile`` (2D spatial)."""
        rel = pixels_flat + (1.0 - origin) - self.crpix
        if (
            self.naxis == 2
            and self._is_tan
            and not self._is_tpv
            and not self._is_tnx
            and not self._is_zpx
            and self.sip is None
            and not self._has_paper4_lut()
        ):
            xi, eta = self._apply_cd_2d(rel[:, 0], rel[:, 1])
            ra, dec = self._project_tan(xi, eta)
            return torch.stack((ra, dec), dim=-1)
        return self._transform_optimized(rel, allow_cpp_sphere=allow_cpp_sphere)

    def _spherical_rotation_optimized(
        self,
        phi: Tensor,
        theta: Tensor,
        native_type: str = "pole",
        *,
        allow_cpp_sphere: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Optimized rotation using vector rotation matrix.
        """
        if native_type != "pole":
            if self._use_center_equator_fast:
                ra = _wrap_lon360(phi + self.alpha0)
                return ra, theta
            return self._spherical_rotation(phi, theta, native_type)

        if (
            allow_cpp_sphere
            and self._cpp_spherical_rotation_pole is not None
            and phi.device.type == "cpu"
            and theta.device.type == "cpu"
            and phi.dtype == torch.float64
            and theta.dtype == torch.float64
            and not phi.requires_grad
            and not theta.requires_grad
        ):
            return self._cpp_spherical_rotation_pole(
                phi,
                theta,
                self._phi_p_rad,
                self._pole_east_x,
                self._pole_east_y,
                self._pole_north_x,
                self._pole_north_y,
                self._pole_north_z,
                self._pole_radial_x,
                self._pole_radial_y,
                self._pole_radial_z,
            )

        d2r = 0.017453292519943295
        r2d = 57.29577951308232

        phi_rad = (phi - self.phi_p) * d2r
        theta_rad = theta * d2r

        sin_theta, cos_theta = _sincos(theta_rad)
        sin_phi, cos_phi = _sincos(phi_rad)

        # Native Cartesian (u, v, w) relative to local basis (East, North, Radial)
        u = cos_theta * sin_phi
        v = cos_theta * cos_phi
        w = sin_theta

        # v_cel = R @ v_nat using precomputed scalar basis at (alpha_p, delta_p).
        x = u * self._pole_east_x + v * self._pole_north_x + w * self._pole_radial_x
        y = u * self._pole_east_y + v * self._pole_north_y + w * self._pole_radial_y
        z = v * self._pole_north_z + w * self._pole_radial_z

        ra = torch.remainder(torch.atan2(y, x) * r2d, 360.0)
        dec = torch.asin(torch.clamp(z, -1.0, 1.0)) * r2d

        return ra, dec

    def _spherical_rotation(
        self, phi: Tensor, theta: Tensor, native_type: str = "pole"
    ) -> Tuple[Tensor, Tensor]:
        """
        Rotate native spherical coordinates (phi, theta) to celestial (alpha, delta)
        using vector rotation (robust at pole).
        native_type: 'pole' (Ref is 0,90) or 'center' (Ref is 0,0).
        """
        # Constants
        d2r = math.pi / 180.0
        r2d = 180.0 / math.pi

        # Convert to radians
        phi_rad = phi * d2r
        theta_rad = theta * d2r

        # 1. Convert Native (phi, theta) to Cartesian (u, v, w)
        # u = cos(theta) * sin(phi)
        # v = -cos(theta) * cos(phi)
        # w = sin(theta)

        sin_theta = torch.sin(theta_rad)
        cos_theta = torch.cos(theta_rad)
        sin_phi = torch.sin(phi_rad)
        cos_phi = torch.cos(phi_rad)

        u = cos_theta * sin_phi
        v = -cos_theta * cos_phi
        w = sin_theta

        # 2. Construct Basis Vectors for Celestial Frame at CRVAL
        a0 = self.crval[0] * d2r
        d0 = self.crval[1] * d2r

        sin_a0, cos_a0 = torch.sin(a0), torch.cos(a0)
        sin_d0, cos_d0 = torch.sin(d0), torch.cos(d0)

        # Basis vectors definition:
        # r (radial, pointing to CRVAL) = [cos d0 cos a0, cos d0 sin a0, sin d0]
        # n (north) = [-sin d0 cos a0, -sin d0 sin a0, cos d0]
        # e (east)  = [-sin a0, cos a0, 0]

        # r components
        rx = cos_d0 * cos_a0
        ry = cos_d0 * sin_a0
        rz = sin_d0

        # n components
        nx = -sin_d0 * cos_a0
        ny = -sin_d0 * sin_a0
        nz = cos_d0

        # e components
        ex = -sin_a0
        ey = cos_a0
        ez = torch.zeros_like(a0)

        # 3. Rotate Vector
        if native_type == "pole":
            # Ref is (0, 90) -> w=1
            # Map w -> r (Pole to CRVAL)
            # Map v -> n (North-ish to North)
            # Map u -> e (East to East)
            X = u * ex + v * nx + w * rx
            Y = u * ey + v * ny + w * ry
            Z = u * ez + v * nz + w * rz
        else:  # 'center'
            # Ref is (0, 0) -> intersection of Eq/Meridian
            # At (0,0), w=0, u=0, v=-1 (cos(0)=1, v=-1).
            # We want Ref (v=-1) to map to CRVAL (r).
            # So map -v -> r => v -> -r.
            # We want North Pole (w=1) to map to North (n).
            # Map w -> n.
            # Map u -> e (East).
            # Check u x v = e x -r?
            # e x -r = n. Matched.

            X = u * ex + v * (-rx) + w * nx
            Y = u * ey + v * (-ry) + w * ny
            Z = u * ez + v * (-rz) + w * nz

        # 4. Convert Cartesian to RA/Dec
        # Dec = asin(Z)
        Z = torch.clamp(Z, -1.0, 1.0)
        dec_rad = torch.asin(Z)

        # alpha - alpha_p = atan2(y, x) -> NO.
        # X, Y, Z are absolute celestial coordinates.
        # alpha = atan2(Y, X) directly.
        alpha_rad = torch.atan2(Y, X)

        # Convert to degrees
        alpha = alpha_rad * r2d
        delta = dec_rad * r2d

        # Normalize Alpha
        alpha = torch.remainder(alpha, 360.0)

        return alpha, delta

    def _inverse_spherical_rotation(
        self,
        ra: Tensor,
        dec: Tensor,
        native_type: str = "pole",
        *,
        allow_cpp_inverse_sphere: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Inverse spherical rotation: celestial (ra, dec) -> native (phi, theta).
        """
        if native_type != "pole":
            if self._use_center_equator_fast:
                phi = _wrap_lon180_checked(ra - self.alpha0)
                return phi, dec
            d2r = 0.017453292519943295
            r2d = 57.29577951308232
            ra_rad = ra * d2r
            dec_rad = dec * d2r
            sin_dec, cos_dec = _sincos(dec_rad)
            sin_ra, cos_ra = _sincos(ra_rad)
            x_cel = cos_dec * cos_ra
            y_cel = cos_dec * sin_ra
            z_cel = sin_dec
            u = x_cel * self._pole_east_x + y_cel * self._pole_east_y
            w = (
                x_cel * self._pole_north_x
                + y_cel * self._pole_north_y
                + z_cel * self._pole_north_z
            )
            v = -(
                x_cel * self._pole_radial_x
                + y_cel * self._pole_radial_y
                + z_cel * self._pole_radial_z
            )
            theta = torch.asin(torch.clamp(w, -1.0, 1.0)) * r2d
            phi = torch.remainder(torch.atan2(u, -v) * r2d + self.phi_p, 360.0)
            return phi, theta

        if (
            allow_cpp_inverse_sphere
            and self._cpp_inverse_spherical_rotation_pole is not None
            and ra.device.type == "cpu"
            and dec.device.type == "cpu"
            and ra.dtype == torch.float64
            and dec.dtype == torch.float64
            and not ra.requires_grad
            and not dec.requires_grad
        ):
            return self._cpp_inverse_spherical_rotation_pole(
                ra,
                dec,
                self._phi_p_rad,
                self._pole_east_x,
                self._pole_east_y,
                self._pole_north_x,
                self._pole_north_y,
                self._pole_north_z,
                self._pole_radial_x,
                self._pole_radial_y,
                self._pole_radial_z,
            )

        d2r = 0.017453292519943295
        r2d = 57.29577951308232

        ra_rad = ra * d2r
        dec_rad = dec * d2r

        sin_dec, cos_dec = _sincos(dec_rad)
        sin_ra, cos_ra = _sincos(ra_rad)

        # Celestial Cartesian (X, Y, Z)
        x_cel = cos_dec * cos_ra
        y_cel = cos_dec * sin_ra
        z_cel = sin_dec

        # v_nat = R^T @ v_cel using precomputed scalar basis at (alpha_p, delta_p).
        u = x_cel * self._pole_east_x + y_cel * self._pole_east_y
        v = (
            x_cel * self._pole_north_x
            + y_cel * self._pole_north_y
            + z_cel * self._pole_north_z
        )
        w = (
            x_cel * self._pole_radial_x
            + y_cel * self._pole_radial_y
            + z_cel * self._pole_radial_z
        )

        theta = torch.asin(torch.clamp(w, -1.0, 1.0)) * r2d
        phi = torch.remainder(torch.atan2(u, v) * r2d + self.phi_p, 360.0)

        return phi, theta

    def _project_tan(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Gnomonic (TAN) projection: intermediate (deg) -> spherical (deg).

        xi, eta: Standard coordinates in degrees.
        Returns: ra, dec in degrees.
        """
        # Convert degress to radians
        rad = math.pi / 180.0
        deg = 180.0 / math.pi

        xi_rad = xi * rad
        eta_rad = eta * rad
        crval1_rad = self.crval[0] * rad
        crval2_rad = self.crval[1] * rad

        # Use the explicit rotation matrix method which is vectorizable.
        # 1. Construct vector in tangent plane (3D).
        #    Plane is z=1 (tangent at North Pole of unit sphere).
        #    V = [xi, eta, 1] is the correct direction.
        #    Normalize to get unit vector on sphere.

        x = xi_rad
        y = eta_rad

        # 2. Rotate this vector from North Pole (0, 90) to (CRVAL1, CRVAL2).
        # Vector in u,v,w space where w points to CRVAL.
        # u = x (East)
        # v = y (North)
        # w = 1.0

        u = x
        v = y
        w = torch.ones_like(x)

        # Normalize
        norm = torch.sqrt(u * u + v * v + w * w)
        u = u / norm
        v = v / norm
        w = w / norm

        # This vector (u,v,w) is in a frame where Z-axis points to (CRVAL1, CRVAL2).
        # Y-axis points North (towards NCP).
        # X-axis points East.

        # We want to rotate this frame to the celestial frame (X, Y, Z).
        # Rotation depends on (alpha0, delta0).

        a0 = crval1_rad
        d0 = crval2_rad

        # Construct the rotation matrix explicitly.

        # Local basis at (a0, d0):
        # r (radial) = [cos(d0)cos(a0), cos(d0)sin(a0), sin(d0)]
        # n (north)  = [-sin(d0)cos(a0), -sin(d0)sin(a0), cos(d0)] (partial d/delta)
        # e (east)   = [-sin(a0), cos(a0), 0] (partial d/alpha / cos(delta))

        # Our vector is V = u*e + v*n + w*r

        sin_a0, cos_a0 = torch.sin(a0), torch.cos(a0)
        sin_d0, cos_d0 = torch.sin(d0), torch.cos(d0)

        # e vector components
        ex = -sin_a0
        ey = cos_a0
        ez = torch.zeros_like(a0)

        # n vector components
        nx = -sin_d0 * cos_a0
        ny = -sin_d0 * sin_a0
        nz = cos_d0

        # r vector components
        rx = cos_d0 * cos_a0
        ry = cos_d0 * sin_a0
        rz = sin_d0

        # Final vector components in Celestial Frame
        X = u * ex + v * nx + w * rx
        Y = u * ey + v * ny + w * ry
        Z = u * ez + v * nz + w * rz

        # Convert X,Y,Z back to RA, Dec
        # Dec = asin(Z)
        dec_out_rad = torch.asin(Z)

        # RA = atan2(Y, X)
        ra_out_rad = torch.atan2(Y, X)

        # Convert to degrees
        ra_out = ra_out_rad * deg
        dec_out = dec_out_rad * deg

        # Normalize RA to [0, 360]
        ra_out = torch.remainder(ra_out, 360.0)

        return ra_out, dec_out

    def world_to_pixel(self, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        World -> Pixel.
        """
        iterative = kwargs.get("iterative", True)
        origin = float(kwargs.get("origin", 0))

        if len(args) == 2 and self.naxis == 2:
            ra_t = self._as_wcs_coord(args[0])
            dec_t = self._as_wcs_coord(args[1])
            return self._world_to_pixel_2d_fast(
                ra_t, dec_t, iterative=iterative, origin=origin
            )

        if len(args) == 1:
            world = self._as_wcs_coord(args[0])

            # Fast path for 2D spatial with (..., 2) tensor input
            if self.naxis == 2 and self._is_spatial:
                stack_inv = self._compiled_stack_world_to_pixel
                if stack_inv is not None and origin in stack_inv:
                    world_c = self._contiguous_if_needed(world)
                    wf = world_c.reshape(-1, 2)
                    pix_flat = stack_inv[origin](wf)
                    return pix_flat.view(world.shape)
                px, py = self._world_to_pixel_2d_fast(
                    world[..., 0], world[..., 1], iterative=iterative, origin=origin
                )
                return torch.stack([px, py], dim=-1)
        else:
            world = torch.stack(
                [self._as_wcs_coord(a) for a in args],
                dim=-1,
            )

        original_shape = world.shape
        world_flat = world.view(-1, self.naxis)

        # 1. Check if first 2 axes are spatial
        is_spatial = self._is_spatial

        if not is_spatial:
            # Entirely linear transformation
            # pix_rel = cd_inv @ (world - crval)
            linear_world = world_flat.clone()
            coupled_pixel = self._apply_coupled_tabular_world_to_pixel(world_flat)
            for i in range(self.naxis):
                linear_world[:, i] = self._coupled_or_tabular_linear_world(
                    coupled_pixel, world_flat, i
                )
            rel_flat = torch.matmul(self.cd_inv_full, (linear_world - self.crval).T).T
            pix_internal_flat = rel_flat + self.crpix
        else:
            pix_internal_flat = self._alloc_pix_internal_like(world_flat)
            px_1, py_1 = self._world_to_pixel_2d_fast(
                world_flat[:, 0],
                world_flat[:, 1],
                iterative=iterative,
                origin=1.0,
            )
            if self._pixel_affine_matrix is not None:
                src_x, src_y = self._corrected_pixels_to_source(px_1, py_1, origin=1.0)
            else:
                src_x, src_y = px_1, py_1
            pix_internal_flat[:, 0] = src_x
            pix_internal_flat[:, 1] = src_y

        # 3. Handle extra axes (NAXIS > 2) for non-spatial WCS
        if not is_spatial and self.naxis >= 3:
            # Already handled by cd_inv_full above
            pass
        elif is_spatial and self.naxis >= 3:
            pix_internal_flat[:, 2:] = self._invert_extra_axes(
                world_flat, pix_internal_flat[:, :2]
            )

        # Adjust for origin: pix = pix_internal + (origin - 1)
        pix_flat = pix_internal_flat + (origin - 1)
        if self._pixel_affine_matrix is not None and is_spatial and self.naxis >= 2:
            cx, cy = self._source_pixels_to_corrected(
                pix_flat[:, 0], pix_flat[:, 1], origin
            )
            pix_flat = pix_flat.clone()
            pix_flat[:, 0] = cx
            pix_flat[:, 1] = cy

        pix = pix_flat.view(original_shape)
        if len(args) > 1:
            return tuple(pix[..., i] for i in range(self.naxis))
        return pix

