"""Explicit spherical-harmonic transform semantics and regular CAR grids."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import Tensor

from . import spectral as _spectral


class SpinConvention(StrEnum):
    IAU = "iau"
    COSMO = "cosmo"


class CARSampling(StrEnum):
    CLENSHAW_CURTIS = "clenshaw-curtis"
    FEJER1 = "fejer-1"


class SHTOperation(StrEnum):
    SYNTHESIZE = "synthesize"
    ADJOINT_SYNTHESIZE = "adjoint_synthesize"
    ANALYZE = "analyze"
    PSEUDO_ANALYZE = "pseudo_analyze"


class SHTAnalysisKind(StrEnum):
    QUADRATURE = "quadrature"
    LEAST_SQUARES = "least-squares"


class SHTCapabilityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SHTGeometry:
    spin_convention: SpinConvention = field(kw_only=True)

    @property
    def kind(self) -> str:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class HEALPixGeometry(SHTGeometry):
    nside: int
    nest: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "spin_convention", SpinConvention(self.spin_convention))
        if int(self.nside) != self.nside or int(self.nside) <= 0:
            raise ValueError("nside must be a positive integer")

    @property
    def kind(self) -> str:
        return "healpix"

    @property
    def shape(self) -> tuple[int, ...]:
        return (12 * int(self.nside) * int(self.nside),)


def _cc_weights(size: int) -> tuple[float, ...]:
    if size < 2:
        raise ValueError("Clenshaw-Curtis sampling requires at least two latitude rings")
    n = size - 1
    theta = torch.arange(size, dtype=torch.float64) * (math.pi / n)
    weights = torch.empty(size, dtype=torch.float64)
    interior = theta[1:n]
    values = torch.ones(n - 1, dtype=torch.float64)
    if n % 2 == 0:
        endpoint = 1.0 / (n * n - 1.0)
        weights[0] = endpoint
        weights[n] = endpoint
        for k in range(1, n // 2):
            values = values - 2.0 * torch.cos(2.0 * k * interior) / (4.0 * k * k - 1.0)
        values = values - torch.cos(n * interior) / (n * n - 1.0)
    else:
        endpoint = 1.0 / (n * n)
        weights[0] = endpoint
        weights[n] = endpoint
        for k in range(1, (n + 1) // 2):
            values = values - 2.0 * torch.cos(2.0 * k * interior) / (4.0 * k * k - 1.0)
    weights[1:n] = 2.0 * values / n
    return tuple(float(value) for value in weights)


def _fejer1_weights(size: int) -> tuple[float, ...]:
    if size < 1:
        raise ValueError("Fejer-1 sampling requires at least one latitude ring")
    theta = (torch.arange(size, dtype=torch.float64) + 0.5) * (math.pi / size)
    values = torch.ones(size, dtype=torch.float64)
    for k in range(1, size // 2 + 1):
        values = values - 2.0 * torch.cos(2.0 * k * theta) / (4.0 * k * k - 1.0)
    return tuple(float(value) for value in (2.0 * values / size))


def _header_float(header: Mapping[str, Any], key: str) -> float:
    if key not in header:
        raise ValueError(f"CAR header requires {key}")
    try:
        value = float(header[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CAR header {key} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"CAR header {key} must be finite")
    return value


def _axis_delta(header: Mapping[str, Any], axis: int) -> float:
    cd_keys = {f"CD{i}_{j}" for i in (1, 2) for j in (1, 2)}
    has_cd = any(key in header for key in cd_keys)
    if has_cd:
        if any(key in header for key in {f"PC{i}_{j}" for i in (1, 2) for j in (1, 2)}):
            raise ValueError("CAR header cannot mix CD and PC matrices")
        offdiag = _header_float(header, "CD1_2") if "CD1_2" in header else 0.0
        offdiag_other = _header_float(header, "CD2_1") if "CD2_1" in header else 0.0
        if offdiag != 0.0 or offdiag_other != 0.0:
            raise ValueError("CAR SHT requires a separable unrotated CD matrix")
        return _header_float(header, f"CD{axis}_{axis}")
    offdiag = _header_float(header, "PC1_2") if "PC1_2" in header else 0.0
    offdiag_other = _header_float(header, "PC2_1") if "PC2_1" in header else 0.0
    if offdiag != 0.0 or offdiag_other != 0.0:
        raise ValueError("CAR SHT requires a separable unrotated PC matrix")
    diagonal = _header_float(header, f"PC{axis}_{axis}") if f"PC{axis}_{axis}" in header else 1.0
    return _header_float(header, f"CDELT{axis}") * diagonal


def _ctype_role(value: Any) -> str:
    ctype = str(value).strip().upper()
    if not ctype.endswith("-CAR") or len(ctype) < 8:
        raise ValueError("CAR SHT requires two undistorted celestial CTYPEs ending in -CAR")
    prefix = ctype[:-4].replace("-", "")
    if prefix in {"RA", "GLON", "ELON", "HLON", "SLON", "TLON"}:
        return "longitude"
    if prefix in {"DEC", "GLAT", "ELAT", "HLAT", "SLAT", "TLAT"}:
        return "latitude"
    raise ValueError(f"unrecognized CAR celestial axis {ctype!r}")


def _reject_distortion(header: Mapping[str, Any]) -> None:
    prefixes = ("A_", "B_", "AP_", "BP_", "CPDIS", "DET2IM", "DP", "DQ", "PV", "PS")
    for raw_key in header:
        key = str(raw_key).strip().upper()
        if key.startswith(prefixes):
            raise ValueError(f"CAR SHT does not accept distortion keyword {raw_key!r}")
    for key in ("CROTA1", "CROTA2"):
        if key in header and _header_float(header, key) != 0.0:
            raise ValueError("CAR SHT requires zero CROTA")


def _axis_unit(header: Mapping[str, Any], axis: int) -> None:
    unit = str(header.get(f"CUNIT{axis}", "deg")).strip().lower()
    if unit not in {"", "deg", "degree", "degrees"}:
        raise ValueError("CAR SHT celestial axes must use degrees")


@dataclass(frozen=True, slots=True)
class CARGeometry(SHTGeometry):
    array_shape: tuple[int, int]
    latitude_axis: int
    longitude_axis: int
    sampling: CARSampling
    latitude_radians: tuple[float, ...]
    quadrature_weights: tuple[float, ...]
    longitude_origin_radians: float
    latitude_reversed: bool
    longitude_reversed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "spin_convention", SpinConvention(self.spin_convention))
        object.__setattr__(self, "sampling", CARSampling(self.sampling))
        if len(self.array_shape) != 2 or min(self.array_shape) < 1:
            raise ValueError("array_shape must contain two positive dimensions")
        if {int(self.latitude_axis), int(self.longitude_axis)} != {0, 1}:
            raise ValueError("latitude_axis and longitude_axis must be distinct array axes")
        if len(self.latitude_radians) != self.nlat:
            raise ValueError("latitude_radians length does not match geometry")
        if len(self.quadrature_weights) != self.nlat:
            raise ValueError("quadrature_weights length does not match geometry")

    @property
    def kind(self) -> str:
        return "car"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array_shape

    @property
    def nlat(self) -> int:
        return int(self.array_shape[self.latitude_axis])

    @property
    def nlon(self) -> int:
        return int(self.array_shape[self.longitude_axis])

    @property
    def pixel_count(self) -> int:
        return self.nlat * self.nlon

    def latitudes(self, *, dtype: torch.dtype, device: torch.device | str) -> Tensor:
        return torch.tensor(self.latitude_radians, dtype=dtype, device=device)

    def ring_weights(self, *, dtype: torch.dtype, device: torch.device | str) -> Tensor:
        return torch.tensor(self.quadrature_weights, dtype=dtype, device=device)

    @classmethod
    def from_header(
        cls,
        header: Mapping[str, Any],
        shape: tuple[int, int] | None = None,
        *,
        spin_convention: SpinConvention | str,
    ) -> CARGeometry:
        normalized = {str(key).strip().upper(): value for key, value in header.items()}
        if "NAXIS" in normalized and int(normalized["NAXIS"]) != 2:
            raise ValueError("CAR SHT requires exactly two array axes")
        if "WCSAXES" in normalized and int(normalized["WCSAXES"]) != 2:
            raise ValueError("CAR SHT requires exactly two WCS axes")
        _reject_distortion(normalized)
        roles = (_ctype_role(normalized.get("CTYPE1", "")), _ctype_role(normalized.get("CTYPE2", "")))
        if roles[0] == roles[1]:
            raise ValueError("CAR SHT requires one longitude and one latitude axis")
        for axis in (1, 2):
            _axis_unit(normalized, axis)
        if shape is None:
            if "NAXIS1" not in normalized or "NAXIS2" not in normalized:
                raise ValueError("CAR header requires NAXIS1/NAXIS2 or an explicit shape")
            array_shape = (int(normalized["NAXIS2"]), int(normalized["NAXIS1"]))
        else:
            array_shape = tuple(int(value) for value in shape)
            if len(array_shape) != 2:
                raise ValueError("shape must be the two-dimensional array shape")
            if "NAXIS1" in normalized and int(normalized["NAXIS1"]) != array_shape[1]:
                raise ValueError("shape disagrees with NAXIS1")
            if "NAXIS2" in normalized and int(normalized["NAXIS2"]) != array_shape[0]:
                raise ValueError("shape disagrees with NAXIS2")
        if min(array_shape) < 1:
            raise ValueError("CAR dimensions must be positive")
        lon_fits_axis = roles.index("longitude") + 1
        lat_fits_axis = roles.index("latitude") + 1
        lon_array_axis = 2 - lon_fits_axis
        lat_array_axis = 2 - lat_fits_axis
        nlon = array_shape[lon_array_axis]
        nlat = array_shape[lat_array_axis]
        lon_delta = _axis_delta(normalized, lon_fits_axis)
        lat_delta = _axis_delta(normalized, lat_fits_axis)
        if lon_delta == 0.0 or lat_delta == 0.0:
            raise ValueError("CAR pixel increments must be non-zero")
        tolerance = 2.0e-9
        if not math.isclose(abs(lon_delta) * nlon, 360.0, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError("CAR longitude axis must be exactly full-360 and periodic")
        lon_crpix = _header_float(normalized, f"CRPIX{lon_fits_axis}")
        lon_crval = _header_float(normalized, f"CRVAL{lon_fits_axis}")
        lat_crpix = _header_float(normalized, f"CRPIX{lat_fits_axis}")
        lat_crval = _header_float(normalized, f"CRVAL{lat_fits_axis}")
        if not math.isclose(lat_crval, 0.0, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError("CAR SHT requires an unrotated latitude reference at zero degrees")
        if "LONPOLE" in normalized and not math.isclose(
            _header_float(normalized, "LONPOLE"), 0.0, rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError("CAR SHT requires an unrotated native longitude pole")
        if "LATPOLE" in normalized and not math.isclose(
            abs(_header_float(normalized, "LATPOLE")), 90.0, rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError("CAR SHT requires an unrotated native latitude pole")
        lon_values = tuple(lon_crval + ((index + 1.0) - lon_crpix) * lon_delta for index in range(nlon))
        lat_values = tuple(lat_crval + ((index + 1.0) - lat_crpix) * lat_delta for index in range(nlat))
        lon_reversed = lon_delta < 0.0
        lat_reversed = lat_delta > 0.0
        canonical_lon = tuple(reversed(lon_values)) if lon_reversed else lon_values
        canonical_lat = tuple(reversed(lat_values)) if lat_reversed else lat_values
        cc = tuple(90.0 - index * 180.0 / (nlat - 1) for index in range(nlat)) if nlat > 1 else ()
        f1 = tuple(90.0 - (index + 0.5) * 180.0 / nlat for index in range(nlat))
        if nlat > 1 and all(
            math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance)
            for value, expected in zip(canonical_lat, cc, strict=True)
        ):
            sampling = CARSampling.CLENSHAW_CURTIS
            weights = _cc_weights(nlat)
        elif all(
            math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance)
            for value, expected in zip(canonical_lat, f1, strict=True)
        ):
            sampling = CARSampling.FEJER1
            weights = _fejer1_weights(nlat)
        else:
            raise ValueError("CAR latitude centers are not Clenshaw-Curtis or Fejer-1")
        latitude_radians = tuple(math.radians(value) for value in canonical_lat)
        longitude_origin = math.remainder(math.radians(canonical_lon[0]), 2.0 * math.pi)
        return cls(
            array_shape=array_shape,
            latitude_axis=lat_array_axis,
            longitude_axis=lon_array_axis,
            sampling=sampling,
            latitude_radians=latitude_radians,
            quadrature_weights=weights,
            longitude_origin_radians=longitude_origin,
            latitude_reversed=lat_reversed,
            longitude_reversed=lon_reversed,
            spin_convention=SpinConvention(spin_convention),
        )


@dataclass(frozen=True, slots=True)
class SHTRequest:
    geometry: SHTGeometry
    lmax: int
    operation: SHTOperation
    mmax: int | None = None
    spin: int = 0
    latitude_block_size: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", SHTOperation(self.operation))
        if int(self.lmax) != self.lmax or int(self.lmax) < 0:
            raise ValueError("lmax must be a non-negative integer")
        mm = int(self.lmax if self.mmax is None else self.mmax)
        if mm < 0 or mm > int(self.lmax):
            raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
        object.__setattr__(self, "mmax", mm)
        if int(self.spin) != self.spin or int(self.spin) < 0:
            raise ValueError("spin must be a non-negative integer")
        if int(self.latitude_block_size) < 1:
            raise ValueError("latitude_block_size must be positive")
        if isinstance(self.geometry, CARGeometry) and self.geometry.nlon < 2 * mm + 1:
            raise ValueError("CAR longitude sampling requires nlon >= 2*mmax + 1")


@dataclass(frozen=True, slots=True)
class SHTBackendCapabilities:
    backend: str
    operations: frozenset[SHTOperation]
    geometry_kinds: frozenset[str]
    spins: frozenset[int]
    devices: frozenset[str]
    dtypes: frozenset[torch.dtype]
    batched: bool
    autograd: bool
    analysis_kinds: frozenset[SHTAnalysisKind] = frozenset()

    def supports(self, request: SHTRequest, value: Tensor | None = None) -> bool:
        if request.operation not in self.operations:
            return False
        if request.geometry.kind not in self.geometry_kinds or request.spin not in self.spins:
            return False
        if request.operation is SHTOperation.ANALYZE and SHTAnalysisKind.QUADRATURE not in self.analysis_kinds:
            return False
        if request.operation is SHTOperation.PSEUDO_ANALYZE and SHTAnalysisKind.LEAST_SQUARES not in self.analysis_kinds:
            return False
        if value is not None and (value.device.type not in self.devices or value.dtype not in self.dtypes):
            return False
        return True

    def require(self, request: SHTRequest, value: Tensor) -> None:
        if not self.supports(request, value):
            raise SHTCapabilityError(
                f"backend {self.backend!r} cannot provide {request.operation.value} "
                f"for {request.geometry.kind}, spin={request.spin}, {value.dtype} on {value.device.type}"
            )


@dataclass(frozen=True, slots=True)
class SHTResult:
    values: Tensor
    request: SHTRequest
    backend: str
    analysis_kind: SHTAnalysisKind | None = None
    converged: bool | None = None
    iterations: int | None = None
    relative_residual: float | None = None


SHTAnalysisResult = SHTResult


@runtime_checkable
class SHTBackend(Protocol):
    name: str
    capabilities: SHTBackendCapabilities

    def execute(
        self,
        values: Tensor,
        request: SHTRequest,
        *,
        tolerance: float,
        max_iterations: int,
    ) -> SHTResult: ...


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in {torch.float64, torch.complex128}:
        return torch.float64
    if dtype in {torch.float32, torch.complex64}:
        return torch.float32
    raise TypeError("SHT values must use float32, float64, complex64, or complex128")


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if _real_dtype(dtype) is torch.float64 else torch.complex64


def _resolved_mmax(request: SHTRequest) -> int:
    if request.mmax is None:
        raise RuntimeError("validated SHT request has no mmax")
    return request.mmax


def _validate_real_alm(alm: Tensor, request: SHTRequest) -> None:
    if alm.ndim < 1 or int(alm.shape[-1]) != _spectral.alm_size(
        int(request.lmax), _resolved_mmax(request)
    ):
        return
    if torch.is_complex(alm) and bool(torch.any(alm[..., : int(request.lmax) + 1].imag != 0).item()):
        raise ValueError("real-field packed alms require real m=0 coefficients")


def _validate_real_map(map_values: Tensor) -> None:
    if torch.is_complex(map_values):
        raise ValueError("real scalar and spin SHT maps must have a real dtype")


def _canonical_scalar_map(values: Tensor, geometry: CARGeometry) -> Tensor:
    if tuple(values.shape[-2:]) != geometry.array_shape:
        raise ValueError(f"map trailing shape must be {geometry.array_shape}")
    out = values.transpose(-2, -1) if geometry.latitude_axis == 1 else values
    if geometry.latitude_reversed:
        out = torch.flip(out, (-2,))
    if geometry.longitude_reversed:
        out = torch.flip(out, (-1,))
    return out


def _storage_scalar_map(values: Tensor, geometry: CARGeometry) -> Tensor:
    out = values
    if geometry.longitude_reversed:
        out = torch.flip(out, (-1,))
    if geometry.latitude_reversed:
        out = torch.flip(out, (-2,))
    return out.transpose(-2, -1) if geometry.latitude_axis == 1 else out


def _canonical_spin_map(values: Tensor, geometry: CARGeometry) -> Tensor:
    if values.ndim < 3 or int(values.shape[-3]) != 2:
        raise ValueError("spin maps must have shape (..., 2, axis2, axis1)")
    if tuple(values.shape[-2:]) != geometry.array_shape:
        raise ValueError(f"spin map trailing shape must be {geometry.array_shape}")
    out = values.transpose(-2, -1) if geometry.latitude_axis == 1 else values
    if geometry.latitude_reversed:
        out = torch.flip(out, (-2,))
    if geometry.longitude_reversed:
        out = torch.flip(out, (-1,))
    if geometry.spin_convention is SpinConvention.COSMO:
        out = torch.stack((out[..., 0, :, :], -out[..., 1, :, :]), dim=-3)
    return out


def _storage_spin_map(values: Tensor, geometry: CARGeometry) -> Tensor:
    out = values
    if geometry.spin_convention is SpinConvention.COSMO:
        out = torch.stack((out[..., 0, :, :], -out[..., 1, :, :]), dim=-3)
    if geometry.longitude_reversed:
        out = torch.flip(out, (-1,))
    if geometry.latitude_reversed:
        out = torch.flip(out, (-2,))
    return out.transpose(-2, -1) if geometry.latitude_axis == 1 else out


def _phase(request: SHTRequest, dtype: torch.dtype, device: torch.device) -> Tensor:
    if not isinstance(request.geometry, CARGeometry):
        raise TypeError("CAR geometry required")
    real_dtype = _real_dtype(dtype)
    m = torch.arange(_resolved_mmax(request) + 1, dtype=real_dtype, device=device)
    angle = m * float(request.geometry.longitude_origin_radians)
    return torch.polar(torch.ones_like(angle), angle).to(_complex_dtype(dtype))


def _packed_phase(request: SHTRequest, dtype: torch.dtype, device: torch.device) -> Tensor:
    m = _spectral._alm_m_array(int(request.lmax), _resolved_mmax(request)).to(device)
    return torch.index_select(_phase(request, dtype, device), 0, m)


def _latitude_blocks(geometry: CARGeometry, size: int) -> tuple[tuple[int, int], ...]:
    return tuple((start, min(start + size, geometry.nlat)) for start in range(0, geometry.nlat, size))


def _legendre_sequence(m: int, x: Tensor, lmax: int) -> Tensor:
    sin2theta = torch.clamp(1.0 - x * x, min=1.0e-30)
    log_pmm = 0.5 * (
        math.log(2 * m + 1)
        - math.log(4.0 * math.pi)
        + math.lgamma(2 * m + 1)
        - 2.0 * (m * math.log(2.0) + math.lgamma(m + 1))
    )
    pmm = torch.exp(log_pmm + 0.5 * m * torch.log(sin2theta))
    if m % 2:
        pmm = -pmm
    if lmax == m:
        return pmm.unsqueeze(0)
    pm1 = x * math.sqrt(2 * m + 3) * pmm
    sequence = [pmm, pm1]
    previous2 = pmm
    previous1 = pm1
    for ell in range(m + 2, lmax + 1):
        coefficient_a = math.sqrt((4.0 * ell * ell - 1.0) / (ell * ell - m * m))
        coefficient_b = math.sqrt(
            (2.0 * ell + 1.0)
            / (2.0 * ell - 3.0)
            * (((ell - 1.0) * (ell - 1.0)) - m * m)
            / (ell * ell - m * m)
        )
        current = coefficient_a * x * previous1 - coefficient_b * previous2
        sequence.append(current)
        previous2 = previous1
        previous1 = current
    return torch.stack(sequence, dim=0)


def _wigner_element(ell: int, m: int, mp: int, theta: Tensor) -> Tensor:
    if abs(m) > ell or abs(mp) > ell:
        return torch.zeros_like(theta)
    cosine = torch.cos(theta * 0.5)
    sine = torch.sin(theta * 0.5)
    output = torch.zeros_like(theta)
    log_prefactor = 0.5 * (
        math.lgamma(ell + m + 1)
        + math.lgamma(ell - m + 1)
        + math.lgamma(ell + mp + 1)
        + math.lgamma(ell - mp + 1)
    )
    for k in range(max(0, m - mp), min(ell + m, ell - mp) + 1):
        a = ell + m - k
        b = k
        c = mp - m + k
        d = ell - mp - k
        sign = -1.0 if (k + mp - m) & 1 else 1.0
        log_denominator = (
            math.lgamma(a + 1)
            + math.lgamma(b + 1)
            + math.lgamma(c + 1)
            + math.lgamma(d + 1)
        )
        output = output + sign * math.exp(log_prefactor - log_denominator) * (
            cosine ** (2 * ell + m - mp - 2 * k)
        ) * (sine ** (mp - m + 2 * k))
    return output


def _wigner_sequence(ell_min: int, ell_max: int, m: int, mp: int, theta: Tensor) -> Tensor:
    first = _wigner_element(ell_min, m, mp, theta)
    if ell_max == ell_min:
        return first.unsqueeze(0)
    second = _wigner_element(ell_min + 1, m, mp, theta)
    sequence = [first, second]
    previous2 = first
    previous1 = second
    cosine = torch.cos(theta)
    for ell in range(ell_min + 2, ell_max + 1):
        current_scale = math.sqrt((ell * ell - m * m) * (ell * ell - mp * mp))
        previous_scale = math.sqrt(
            (((ell - 1) * (ell - 1)) - m * m)
            * (((ell - 1) * (ell - 1)) - mp * mp)
        )
        current = (ell / current_scale) * (
            (2 * ell - 1)
            * (cosine - (m * mp) / (ell * (ell - 1)))
            * previous1
            - (previous_scale / (ell - 1)) * previous2
        )
        sequence.append(current)
        previous2 = previous1
        previous1 = current
    return torch.stack(sequence, dim=0)


def _scalar_modes_from_alm(alm: Tensor, request: SHTRequest) -> Tensor:
    geometry = request.geometry
    if not isinstance(geometry, CARGeometry):
        raise TypeError("CAR geometry required")
    ll = int(request.lmax)
    mm = _resolved_mmax(request)
    expected = _spectral.alm_size(ll, mm)
    if alm.ndim < 1 or int(alm.shape[-1]) != expected:
        raise ValueError(f"alm trailing dimension must be {expected}")
    complex_dtype = _complex_dtype(alm.dtype)
    real_dtype = _real_dtype(alm.dtype)
    coeff = alm.to(complex_dtype)
    modes = torch.zeros((*coeff.shape[:-1], geometry.nlat, mm + 1), dtype=complex_dtype, device=alm.device)
    latitudes = geometry.latitudes(dtype=real_dtype, device=alm.device)
    x = torch.sin(latitudes)
    for m in range(mm + 1):
        start = _spectral.alm_index(m, m, ll, mm)
        count = ll - m + 1
        coeff_m = coeff[..., start : start + count]
        for ring_start, ring_end in _latitude_blocks(geometry, int(request.latitude_block_size)):
            basis = _legendre_sequence(m, x[ring_start:ring_end], ll)
            modes[..., ring_start:ring_end, m] = torch.einsum("...l,lr->...r", coeff_m, basis.to(complex_dtype))
    return modes * _phase(request, alm.dtype, alm.device)


def _scalar_alm_from_modes(modes: Tensor, request: SHTRequest) -> Tensor:
    geometry = request.geometry
    if not isinstance(geometry, CARGeometry):
        raise TypeError("CAR geometry required")
    ll = int(request.lmax)
    mm = _resolved_mmax(request)
    complex_dtype = _complex_dtype(modes.dtype)
    real_dtype = _real_dtype(modes.dtype)
    modes = modes.to(complex_dtype) * torch.conj(_phase(request, modes.dtype, modes.device))
    out = torch.zeros((*modes.shape[:-2], _spectral.alm_size(ll, mm)), dtype=complex_dtype, device=modes.device)
    x = torch.sin(geometry.latitudes(dtype=real_dtype, device=modes.device))
    for m in range(mm + 1):
        start = _spectral.alm_index(m, m, ll, mm)
        count = ll - m + 1
        projected = torch.zeros((*modes.shape[:-2], count), dtype=complex_dtype, device=modes.device)
        for ring_start, ring_end in _latitude_blocks(geometry, int(request.latitude_block_size)):
            basis = _legendre_sequence(m, x[ring_start:ring_end], ll)
            projected = projected + torch.einsum(
                "...r,lr->...l", modes[..., ring_start:ring_end, m], basis.to(complex_dtype)
            )
        out[..., start : start + count] = projected
    return out


def _spin_bases(
    m: int,
    spin: int,
    theta: Tensor,
    lmax: int,
) -> tuple[Tensor, Tensor, int]:
    ell_start = max(m, abs(spin))
    if ell_start > lmax:
        empty = torch.empty((0, theta.numel()), dtype=theta.dtype, device=theta.device)
        return empty, empty, ell_start
    d_plus = _wigner_sequence(ell_start, lmax, m, -spin, theta)
    d_minus = _wigner_sequence(ell_start, lmax, m, spin, theta)
    ell = torch.arange(ell_start, lmax + 1, dtype=theta.dtype, device=theta.device)
    norms = torch.sqrt((2.0 * ell + 1.0) / (4.0 * math.pi))
    sign = (-1.0 if spin & 1 else 1.0) * (-1.0 if m & 1 else 1.0)
    scale = sign * norms.unsqueeze(1)
    return scale * d_plus, scale * d_minus, ell_start


def _spin_modes_from_alm(alm: Tensor, request: SHTRequest) -> tuple[Tensor, Tensor]:
    geometry = request.geometry
    if not isinstance(geometry, CARGeometry):
        raise TypeError("CAR geometry required")
    ll = int(request.lmax)
    mm = _resolved_mmax(request)
    expected = _spectral.alm_size(ll, mm)
    if alm.ndim < 2 or int(alm.shape[-2]) != 2 or int(alm.shape[-1]) != expected:
        raise ValueError(f"spin alm must have shape (..., 2, {expected})")
    complex_dtype = _complex_dtype(alm.dtype)
    real_dtype = _real_dtype(alm.dtype)
    coeff = alm.to(complex_dtype)
    coeff_plus = coeff[..., 0, :] + 1j * coeff[..., 1, :]
    coeff_minus = coeff[..., 0, :] - 1j * coeff[..., 1, :]
    plus = torch.zeros((*coeff.shape[:-2], geometry.nlat, mm + 1), dtype=complex_dtype, device=alm.device)
    minus = torch.zeros_like(plus)
    theta = (math.pi * 0.5) - geometry.latitudes(dtype=real_dtype, device=alm.device)
    for m in range(mm + 1):
        for ring_start, ring_end in _latitude_blocks(geometry, int(request.latitude_block_size)):
            basis_plus, basis_minus, ell_start = _spin_bases(
                m, int(request.spin), theta[ring_start:ring_end], ll
            )
            if ell_start > ll:
                continue
            start = _spectral.alm_index(ell_start, m, ll, mm)
            count = ll - ell_start + 1
            plus[..., ring_start:ring_end, m] = torch.einsum(
                "...l,lr->...r", coeff_plus[..., start : start + count], basis_plus.to(complex_dtype)
            )
            minus[..., ring_start:ring_end, m] = torch.einsum(
                "...l,lr->...r", coeff_minus[..., start : start + count], basis_minus.to(complex_dtype)
            )
    phase = _phase(request, alm.dtype, alm.device)
    return plus * phase, minus * phase


def _spin_alm_from_mode_gradients(plus: Tensor, minus: Tensor, request: SHTRequest) -> Tensor:
    geometry = request.geometry
    if not isinstance(geometry, CARGeometry):
        raise TypeError("CAR geometry required")
    ll = int(request.lmax)
    mm = _resolved_mmax(request)
    complex_dtype = _complex_dtype(plus.dtype)
    real_dtype = _real_dtype(plus.dtype)
    phase_conj = torch.conj(_phase(request, plus.dtype, plus.device))
    plus = plus.to(complex_dtype) * phase_conj
    minus = minus.to(complex_dtype) * phase_conj
    nalm = _spectral.alm_size(ll, mm)
    grad_plus = torch.zeros((*plus.shape[:-2], nalm), dtype=complex_dtype, device=plus.device)
    grad_minus = torch.zeros_like(grad_plus)
    theta = (math.pi * 0.5) - geometry.latitudes(dtype=real_dtype, device=plus.device)
    for m in range(mm + 1):
        ell_start = max(m, int(request.spin))
        if ell_start > ll:
            continue
        start = _spectral.alm_index(ell_start, m, ll, mm)
        count = ll - ell_start + 1
        projected_plus = torch.zeros((*plus.shape[:-2], count), dtype=complex_dtype, device=plus.device)
        projected_minus = torch.zeros_like(projected_plus)
        for ring_start, ring_end in _latitude_blocks(geometry, int(request.latitude_block_size)):
            basis_plus, basis_minus, _ = _spin_bases(
                m, int(request.spin), theta[ring_start:ring_end], ll
            )
            projected_plus = projected_plus + torch.einsum(
                "...r,lr->...l", plus[..., ring_start:ring_end, m], basis_plus.to(complex_dtype)
            )
            projected_minus = projected_minus + torch.einsum(
                "...r,lr->...l", minus[..., ring_start:ring_end, m], basis_minus.to(complex_dtype)
            )
        gain = 1.0 if m == 0 else 2.0
        grad_plus[..., start : start + count] = projected_plus / gain
        grad_minus[..., start : start + count] = projected_minus / gain
    grad_e = grad_plus + grad_minus
    grad_b = 1j * (grad_minus - grad_plus)
    return torch.stack((grad_e, grad_b), dim=-2)


def car_synthesize(alm: Tensor, request: SHTRequest) -> Tensor:
    if request.operation is not SHTOperation.SYNTHESIZE:
        raise ValueError("car_synthesize requires a synthesize request")
    geometry = request.geometry
    if not isinstance(geometry, CARGeometry):
        raise TypeError("car_synthesize requires CARGeometry")
    if request.spin == 0:
        modes = _scalar_modes_from_alm(alm, request)
        spectrum = torch.zeros(
            (*modes.shape[:-1], geometry.nlon // 2 + 1),
            dtype=modes.dtype,
            device=modes.device,
        )
        spectrum[..., : _resolved_mmax(request) + 1] = modes * geometry.nlon
        result = torch.fft.irfft(spectrum, n=geometry.nlon, dim=-1)
        return _storage_scalar_map(result, geometry)
    if request.spin not in {1, 2, 3}:
        raise SHTCapabilityError("native CAR synthesis currently supports spins 0 through 3")
    plus, minus = _spin_modes_from_alm(alm, request)
    spectrum_plus = torch.zeros((*plus.shape[:-1], geometry.nlon), dtype=plus.dtype, device=plus.device)
    spectrum_minus = torch.zeros_like(spectrum_plus)
    spectrum_plus[..., : _resolved_mmax(request) + 1] = plus * geometry.nlon
    spectrum_minus[..., : _resolved_mmax(request) + 1] = minus * geometry.nlon
    for m in range(1, _resolved_mmax(request) + 1):
        spectrum_plus[..., -m] = torch.conj(minus[..., m]) * geometry.nlon
        spectrum_minus[..., -m] = torch.conj(plus[..., m]) * geometry.nlon
    field_plus = torch.fft.ifft(spectrum_plus, dim=-1)
    field_minus = torch.fft.ifft(spectrum_minus, dim=-1)
    q = -0.5 * torch.real(field_plus + field_minus)
    u = torch.real(0.5j * (field_plus - field_minus))
    return _storage_spin_map(torch.stack((q, u), dim=-3), geometry)


def car_adjoint_synthesize(map_values: Tensor, request: SHTRequest) -> Tensor:
    if request.operation is not SHTOperation.ADJOINT_SYNTHESIZE:
        raise ValueError("car_adjoint_synthesize requires an adjoint_synthesize request")
    geometry = request.geometry
    if not isinstance(geometry, CARGeometry):
        raise TypeError("car_adjoint_synthesize requires CARGeometry")
    if request.spin == 0:
        canonical = _canonical_scalar_map(map_values, geometry).to(_real_dtype(map_values.dtype))
        modes = torch.fft.rfft(canonical, dim=-1)[..., : _resolved_mmax(request) + 1]
        return _scalar_alm_from_modes(modes, request)
    if request.spin not in {1, 2, 3}:
        raise SHTCapabilityError("native CAR adjoint currently supports spins 0 through 3")
    canonical = _canonical_spin_map(map_values, geometry).to(_real_dtype(map_values.dtype))
    q = canonical[..., 0, :, :]
    u = canonical[..., 1, :, :]
    grad_field_plus = torch.complex(-0.5 * q, -0.5 * u)
    grad_field_minus = torch.complex(-0.5 * q, 0.5 * u)
    fft_plus = torch.fft.fft(grad_field_plus, dim=-1)
    fft_minus = torch.fft.fft(grad_field_minus, dim=-1)
    plus = torch.empty((*q.shape[:-1], _resolved_mmax(request) + 1), dtype=fft_plus.dtype, device=q.device)
    minus = torch.empty_like(plus)
    plus[..., 0] = fft_plus[..., 0]
    minus[..., 0] = fft_minus[..., 0]
    for m in range(1, _resolved_mmax(request) + 1):
        plus[..., m] = fft_plus[..., m] + torch.conj(fft_minus[..., -m])
        minus[..., m] = fft_minus[..., m] + torch.conj(fft_plus[..., -m])
    return _spin_alm_from_mode_gradients(plus, minus, request)


def car_analyze(map_values: Tensor, request: SHTRequest) -> Tensor:
    if request.operation is not SHTOperation.ANALYZE:
        raise ValueError("car_analyze requires an analyze request")
    geometry = request.geometry
    if not isinstance(geometry, CARGeometry):
        raise TypeError("car_analyze requires CARGeometry")
    real_dtype = _real_dtype(map_values.dtype)
    weights = geometry.ring_weights(dtype=real_dtype, device=map_values.device) * (2.0 * math.pi / geometry.nlon)
    adjoint_request = SHTRequest(
        geometry=geometry,
        lmax=request.lmax,
        mmax=request.mmax,
        spin=request.spin,
        operation=SHTOperation.ADJOINT_SYNTHESIZE,
        latitude_block_size=request.latitude_block_size,
    )
    if request.spin == 0:
        canonical = _canonical_scalar_map(map_values, geometry) * weights.view(*((1,) * (map_values.ndim - 2)), -1, 1)
        weighted = _storage_scalar_map(canonical, geometry)
    else:
        canonical = _canonical_spin_map(map_values, geometry) * weights.view(*((1,) * (map_values.ndim - 3)), 1, -1, 1)
        weighted = _storage_spin_map(canonical, geometry)
    return car_adjoint_synthesize(weighted, adjoint_request)


def _packed_inner(left: Tensor, right: Tensor, request: SHTRequest) -> Tensor:
    m = _spectral._alm_m_array(int(request.lmax), _resolved_mmax(request)).to(left.device)
    gain = torch.where(m == 0, 1.0, 2.0).to(dtype=_real_dtype(left.dtype), device=left.device)
    value = gain * torch.real(torch.conj(left) * right)
    return value.sum(dim=(-2, -1)) if request.spin else value.sum(dim=-1)


def _pseudo_analyze_native(
    map_values: Tensor,
    request: SHTRequest,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[Tensor, bool, int, float]:
    adjoint_request = SHTRequest(
        geometry=request.geometry,
        lmax=request.lmax,
        mmax=request.mmax,
        spin=request.spin,
        operation=SHTOperation.ADJOINT_SYNTHESIZE,
        latitude_block_size=request.latitude_block_size,
    )
    synthesis_request = SHTRequest(
        geometry=request.geometry,
        lmax=request.lmax,
        mmax=request.mmax,
        spin=request.spin,
        operation=SHTOperation.SYNTHESIZE,
        latitude_block_size=request.latitude_block_size,
    )
    backend = NativeSHTBackend()
    rhs = backend.execute(map_values, adjoint_request, tolerance=tolerance, max_iterations=max_iterations).values
    estimate = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    residual_norm = _packed_inner(residual, residual, request)
    initial = torch.sqrt(residual_norm).clamp_min(torch.finfo(_real_dtype(rhs.dtype)).tiny)
    converged = bool(torch.all(initial == 0).item())
    iterations = 0
    for iteration in range(max_iterations):
        if converged:
            break
        projected_map = backend.execute(direction, synthesis_request, tolerance=tolerance, max_iterations=max_iterations).values
        applied = backend.execute(projected_map, adjoint_request, tolerance=tolerance, max_iterations=max_iterations).values
        denominator = _packed_inner(direction, applied, request)
        if bool(torch.any(denominator <= 0).item()):
            break
        alpha = residual_norm / denominator
        while alpha.ndim < estimate.ndim:
            alpha = alpha.unsqueeze(-1)
        estimate = estimate + alpha * direction
        residual = residual - alpha * applied
        next_norm = _packed_inner(residual, residual, request)
        relative = torch.sqrt(next_norm) / initial
        iterations = iteration + 1
        converged = bool(torch.all(relative <= tolerance).item())
        beta = next_norm / residual_norm.clamp_min(torch.finfo(next_norm.dtype).tiny)
        while beta.ndim < direction.ndim:
            beta = beta.unsqueeze(-1)
        direction = residual + beta * direction
        residual_norm = next_norm
    final_relative = float(torch.max(torch.sqrt(residual_norm) / initial).detach().cpu().item())
    return estimate, converged, iterations, final_relative


class NativeSHTBackend:
    name = "native"
    capabilities = SHTBackendCapabilities(
        backend=name,
        operations=frozenset(SHTOperation),
        geometry_kinds=frozenset({"healpix", "car"}),
        spins=frozenset({0, 1, 2, 3}),
        devices=frozenset({"cpu", "cuda", "mps"}),
        dtypes=frozenset({torch.float32, torch.float64, torch.complex64, torch.complex128}),
        batched=True,
        autograd=True,
        analysis_kinds=frozenset({SHTAnalysisKind.QUADRATURE, SHTAnalysisKind.LEAST_SQUARES}),
    )

    def execute(
        self,
        values: Tensor,
        request: SHTRequest,
        *,
        tolerance: float,
        max_iterations: int,
    ) -> SHTResult:
        values = torch.as_tensor(values)
        self.capabilities.require(request, values)
        if isinstance(request.geometry, CARGeometry):
            if request.operation is SHTOperation.SYNTHESIZE:
                output = car_synthesize(values, request)
                return SHTResult(output, request, self.name)
            if request.operation is SHTOperation.ADJOINT_SYNTHESIZE:
                output = car_adjoint_synthesize(values, request)
                return SHTResult(output, request, self.name)
            if request.operation is SHTOperation.ANALYZE:
                output = car_analyze(values, request)
                return SHTResult(output, request, self.name, SHTAnalysisKind.QUADRATURE)
            output, converged, iterations, residual = _pseudo_analyze_native(
                values, request, tolerance=tolerance, max_iterations=max_iterations
            )
            return SHTResult(
                output,
                request,
                self.name,
                SHTAnalysisKind.LEAST_SQUARES,
                converged,
                iterations,
                residual,
            )
        geometry = request.geometry
        if not isinstance(geometry, HEALPixGeometry):
            raise SHTCapabilityError(f"native backend does not recognize {type(geometry).__name__}")
        output = self._healpix(values, request, geometry, tolerance, max_iterations)
        kind = None
        if request.operation is SHTOperation.ANALYZE:
            kind = SHTAnalysisKind.QUADRATURE
        if request.operation is SHTOperation.PSEUDO_ANALYZE:
            kind = SHTAnalysisKind.LEAST_SQUARES
        if isinstance(output, tuple):
            tensor, residual, iterations = output
            return SHTResult(tensor, request, self.name, kind, residual <= tolerance, iterations, residual)
        return SHTResult(output, request, self.name, kind)

    def _healpix(
        self,
        values: Tensor,
        request: SHTRequest,
        geometry: HEALPixGeometry,
        tolerance: float,
        max_iterations: int,
    ) -> Tensor | tuple[Tensor, float, int]:
        ll = int(request.lmax)
        mm = _resolved_mmax(request)
        if request.operation is SHTOperation.SYNTHESIZE:
            if request.spin == 0:
                flat = values.reshape(-1, values.shape[-1])
                maps = _spectral.alm2map(flat, geometry.nside, lmax=ll, mmax=mm, nest=geometry.nest)
                return maps.reshape(*values.shape[:-1], geometry.shape[0])
            flat = values.reshape(-1, 2, values.shape[-1])
            maps = torch.stack(
                [
                    _spectral.alm2map_spin((row[0], row[1]), geometry.nside, 2, lmax=ll, mmax=mm, nest=geometry.nest)
                    for row in flat
                ],
                dim=0,
            )
            if geometry.spin_convention is SpinConvention.COSMO:
                maps = torch.stack((maps[:, 0], -maps[:, 1]), dim=1)
            return maps.reshape(*values.shape[:-2], 2, geometry.shape[0])
        if request.operation in {SHTOperation.ADJOINT_SYNTHESIZE, SHTOperation.ANALYZE}:
            scale = geometry.shape[0] / (4.0 * math.pi) if request.operation is SHTOperation.ADJOINT_SYNTHESIZE else 1.0
            if request.spin == 0:
                flat = values.reshape(-1, values.shape[-1])
                alms = _spectral.map2alm(flat, nside=geometry.nside, lmax=ll, mmax=mm, nest=geometry.nest)
                return (alms * scale).reshape(*values.shape[:-1], -1)
            maps = values
            if geometry.spin_convention is SpinConvention.COSMO:
                maps = values.clone()
                maps[..., 1, :] = -maps[..., 1, :]
            flat = maps.reshape(-1, 2, maps.shape[-1])
            alms = torch.stack(
                [
                    torch.stack(_spectral.map2alm_spin(row, 2, nside=geometry.nside, lmax=ll, mmax=mm, nest=geometry.nest))
                    for row in flat
                ],
                dim=0,
            )
            return (alms * scale).reshape(*values.shape[:-2], 2, -1)
        if request.spin == 0 and values.ndim == 1:
            return _spectral.map2alm_lsq(
                values,
                ll,
                mm,
                nside=geometry.nside,
                pol=False,
                tol=tolerance,
                maxiter=max_iterations,
                nest=geometry.nest,
            )
        estimate, _, iterations, residual = _pseudo_analyze_native(
            values, request, tolerance=tolerance, max_iterations=max_iterations
        )
        return estimate, residual, iterations


def _rectangular_alm(alm: Tensor, lmax: int, mmax: int) -> Tensor:
    out = torch.zeros((*alm.shape[:-1], lmax + 1, mmax + 1), dtype=alm.dtype, device=alm.device)
    for m in range(mmax + 1):
        start = _spectral.alm_index(m, m, lmax, mmax)
        out[..., m:, m] = alm[..., start : start + lmax - m + 1]
    return out


def _packed_alm(alm: Tensor, lmax: int, mmax: int) -> Tensor:
    parts = [alm[..., m:, m] for m in range(mmax + 1)]
    return torch.cat(parts, dim=-1)


class TorchHarmonicsBackend:
    name = "torch-harmonics"
    capabilities = SHTBackendCapabilities(
        backend=name,
        operations=frozenset({SHTOperation.SYNTHESIZE, SHTOperation.ANALYZE}),
        geometry_kinds=frozenset({"car"}),
        spins=frozenset({0}),
        devices=frozenset({"cpu", "cuda"}),
        dtypes=frozenset({torch.float32, torch.float64, torch.complex64, torch.complex128}),
        batched=True,
        autograd=True,
        analysis_kinds=frozenset({SHTAnalysisKind.QUADRATURE}),
    )

    def _module(self) -> Any:
        try:
            return importlib.import_module("torch_harmonics")
        except ImportError as exc:
            raise SHTCapabilityError(
                "torch-harmonics backend requested but torch_harmonics is not installed"
            ) from exc

    def _transform(
        self,
        module: Any,
        name: str,
        geometry: CARGeometry,
        lmax: int,
        mmax: int,
    ) -> Any:
        transform_type = getattr(module, name)
        if hasattr(module, "EquiangularGrid"):
            grid = module.EquiangularGrid(nlat=geometry.nlat, nlon=geometry.nlon)
            return transform_type(grid, lmax=lmax + 1, mmax=mmax + 1)
        return transform_type(
            geometry.nlat,
            geometry.nlon,
            lmax=lmax + 1,
            mmax=mmax + 1,
            grid="equiangular",
        )

    def execute(
        self,
        values: Tensor,
        request: SHTRequest,
        *,
        tolerance: float,
        max_iterations: int,
    ) -> SHTResult:
        del tolerance, max_iterations
        values = torch.as_tensor(values)
        self.capabilities.require(request, values)
        geometry = request.geometry
        if not isinstance(geometry, CARGeometry):
            raise SHTCapabilityError("torch-harmonics supports only CARGeometry")
        if geometry.sampling is not CARSampling.CLENSHAW_CURTIS:
            raise SHTCapabilityError("torch-harmonics equiangular adapter requires Clenshaw-Curtis rings")
        module = self._module()
        ll = int(request.lmax)
        mm = _resolved_mmax(request)
        real_dtype = _real_dtype(values.dtype)
        if request.operation is SHTOperation.SYNTHESIZE:
            rectangular = _rectangular_alm(values.to(_complex_dtype(values.dtype)), ll, mm)
            phase = _phase(request, values.dtype, values.device)
            rectangular = rectangular * phase.view(*((1,) * (rectangular.ndim - 1)), -1)
            transform = self._transform(
                module, "InverseRealSHT", geometry, ll, mm
            ).to(device=values.device, dtype=real_dtype)
            output = transform(rectangular)
            return SHTResult(_storage_scalar_map(output, geometry), request, self.name)
        canonical = _canonical_scalar_map(values, geometry).to(real_dtype)
        transform = self._transform(module, "RealSHT", geometry, ll, mm).to(
            device=values.device, dtype=real_dtype
        )
        rectangular = transform(canonical)
        packed = _packed_alm(rectangular, ll, mm)
        packed = packed * torch.conj(_packed_phase(request, values.dtype, values.device))
        return SHTResult(packed, request, self.name, SHTAnalysisKind.QUADRATURE)


_NATIVE_BACKEND = NativeSHTBackend()


def _resolve_backend(backend: str | SHTBackend | None) -> SHTBackend:
    if backend is None or backend == "native":
        return _NATIVE_BACKEND
    if backend in {"torch-harmonics", "torch_harmonics"}:
        return TorchHarmonicsBackend()
    if isinstance(backend, str):
        raise ValueError(f"unknown SHT backend {backend!r}")
    if not hasattr(backend, "execute") or not hasattr(backend, "capabilities"):
        raise TypeError("backend must be a registered name or SHTBackend")
    return backend


def _request(
    geometry_or_request: SHTGeometry | SHTRequest,
    operation: SHTOperation,
    *,
    lmax: int | None,
    mmax: int | None,
    spin: int,
    latitude_block_size: int,
) -> SHTRequest:
    if isinstance(geometry_or_request, SHTRequest):
        if geometry_or_request.operation is not operation:
            raise ValueError(f"request operation must be {operation.value}")
        if lmax is not None or mmax is not None or spin != 0 or latitude_block_size != 64:
            raise ValueError("request and transform keywords cannot be supplied together")
        return geometry_or_request
    if lmax is None:
        raise ValueError("lmax is required when passing a geometry")
    return SHTRequest(
        geometry=geometry_or_request,
        lmax=lmax,
        mmax=mmax,
        spin=spin,
        operation=operation,
        latitude_block_size=latitude_block_size,
    )


def synthesize(
    alm: Tensor,
    geometry_or_request: SHTGeometry | SHTRequest,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    spin: int = 0,
    backend: str | SHTBackend | None = None,
    latitude_block_size: int = 64,
) -> SHTResult:
    request = _request(
        geometry_or_request,
        SHTOperation.SYNTHESIZE,
        lmax=lmax,
        mmax=mmax,
        spin=spin,
        latitude_block_size=latitude_block_size,
    )
    alm_tensor = torch.as_tensor(alm)
    _validate_real_alm(alm_tensor, request)
    return _resolve_backend(backend).execute(
        alm_tensor, request, tolerance=0.0, max_iterations=0
    )


def adjoint_synthesize(
    map_values: Tensor,
    geometry_or_request: SHTGeometry | SHTRequest,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    spin: int = 0,
    backend: str | SHTBackend | None = None,
    latitude_block_size: int = 64,
) -> SHTResult:
    request = _request(
        geometry_or_request,
        SHTOperation.ADJOINT_SYNTHESIZE,
        lmax=lmax,
        mmax=mmax,
        spin=spin,
        latitude_block_size=latitude_block_size,
    )
    map_tensor = torch.as_tensor(map_values)
    _validate_real_map(map_tensor)
    return _resolve_backend(backend).execute(
        map_tensor, request, tolerance=0.0, max_iterations=0
    )


def analyze(
    map_values: Tensor,
    geometry_or_request: SHTGeometry | SHTRequest,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    spin: int = 0,
    backend: str | SHTBackend | None = None,
    latitude_block_size: int = 64,
) -> SHTResult:
    request = _request(
        geometry_or_request,
        SHTOperation.ANALYZE,
        lmax=lmax,
        mmax=mmax,
        spin=spin,
        latitude_block_size=latitude_block_size,
    )
    map_tensor = torch.as_tensor(map_values)
    _validate_real_map(map_tensor)
    return _resolve_backend(backend).execute(
        map_tensor, request, tolerance=0.0, max_iterations=0
    )


def pseudo_analyze(
    map_values: Tensor,
    geometry_or_request: SHTGeometry | SHTRequest,
    *,
    lmax: int | None = None,
    mmax: int | None = None,
    spin: int = 0,
    backend: str | SHTBackend | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 50,
    latitude_block_size: int = 64,
) -> SHTResult:
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    request = _request(
        geometry_or_request,
        SHTOperation.PSEUDO_ANALYZE,
        lmax=lmax,
        mmax=mmax,
        spin=spin,
        latitude_block_size=latitude_block_size,
    )
    map_tensor = torch.as_tensor(map_values)
    _validate_real_map(map_tensor)
    return _resolve_backend(backend).execute(
        map_tensor,
        request,
        tolerance=float(tolerance),
        max_iterations=int(max_iterations),
    )


__all__ = [
    "CARGeometry",
    "CARSampling",
    "HEALPixGeometry",
    "NativeSHTBackend",
    "SHTAnalysisKind",
    "SHTAnalysisResult",
    "SHTBackend",
    "SHTBackendCapabilities",
    "SHTCapabilityError",
    "SHTGeometry",
    "SHTOperation",
    "SHTRequest",
    "SHTResult",
    "SpinConvention",
    "TorchHarmonicsBackend",
    "adjoint_synthesize",
    "analyze",
    "car_adjoint_synthesize",
    "car_analyze",
    "car_synthesize",
    "pseudo_analyze",
    "synthesize",
]
