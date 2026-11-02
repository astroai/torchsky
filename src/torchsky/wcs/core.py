from __future__ import annotations

import torch
from torch import Tensor
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    cast,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import math
import os
import warnings

from torchsky import _ops
from ._geometry import _GeometryMixin, _HeaderIOMixin
from ._math import _wrap_lon180, _wrap_lon360
from ._rotation import _RotationMixin

from .tpv import TPV, PV1_KEYS_SET, PV2_KEYS_SET
from .sip import SIP
from .tabular import (
    CoupledTabularLookupWCS,
    TabularAxis,
)
from .functional import _pixel_to_world_fused, _world_to_pixel_fused

if TYPE_CHECKING:
    pass

from ._core_support import (  # noqa: F401
    _CELESTIAL_LATITUDE_PREFIXES,
    _CELESTIAL_LONGITUDE_PREFIXES,
    _CENTER_CPP_P2W_THRESHOLDS,
    _CENTER_CPP_W2P_THRESHOLDS,
    _HAS_TORCH_SINCOS,
    _MPS_PIXEL_IDENTIFICATION_LIMIT_PX,
    _ZENITHAL_FUSED_P2W_MAX,
    _bound_cpp,
    _celestial_axis_role,
    _ctype_projection_code,
    _ctype_without_projection,
    _float32_ulp,
)
from ._header_parse import (  # noqa: F401
    _HeaderParseMixin,
)
from ._wcs_state import (  # noqa: F401
    _WCSStateMixin,
)
from ._setup_tensors import (  # noqa: F401
    _SetupTensorsMixin,
)
from ._pixel_world_core import (  # noqa: F401
    _PixelWorldCoreMixin,
)
from ._axes_frames import (  # noqa: F401
    _AxesFramesMixin,
)
from ._mixed import (  # noqa: F401
    _MixedMixin,
)
from ._differentiable import (  # noqa: F401
    _DifferentiableMixin,
)


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

    def rebased_for_diagonal_pixel_map(
        self,
        starts: tuple[int, int],
        steps: tuple[int, int],
        *,
        component_axis: int,
    ) -> "DistortionLookupTable":
        if component_axis not in (0, 1):
            raise ValueError("component_axis must be 0 or 1")
        rebased_crval = (
            1.0 + (self.crval[0] - 1.0 - starts[0]) / steps[0],
            1.0 + (self.crval[1] - 1.0 - starts[1]) / steps[1],
        )
        rebased_cdelt = (self.cdelt[0] / steps[0], self.cdelt[1] / steps[1])
        return DistortionLookupTable(
            self.data / steps[component_axis],
            self.crpix,
            rebased_crval,
            rebased_cdelt,
        )


def _require_distortion_lookup_table(table: Any) -> DistortionLookupTable:
    coerced = _coerce_distortion_lookup_table(table)
    assert coerced is not None  # ``table`` is not None
    return coerced


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


class PrecisionWarning(UserWarning):
    """A WCS runs in float32 (MPS) with an expected error of half a pixel or more.

    MPS has no float64. The warning is emitted once per WCS when the expected
    float32 position error, ``(ulp32(world) / pixel scale) + ulp32(pixel extent)``
    in pixels, reaches 0.5 pixel: beyond that the nearest pixel can no longer be
    identified reliably. Use CPU or CUDA (float64) for such grids.
    """


class WCS(
    _RotationMixin,
    _GeometryMixin,
    _HeaderIOMixin,
    _HeaderParseMixin,
    _WCSStateMixin,
    _SetupTensorsMixin,
    _PixelWorldCoreMixin,
    _AxesFramesMixin,
    _MixedMixin,
    _DifferentiableMixin,
):
    """PyTorch-native world coordinate system parsed from FITS-style metadata.

    Args:
        header: FITS-style header mapping. An empty mapping creates a default WCS.
        key: Alternate-WCS suffix without the leading keyword letter.
        tabular_axes: Explicit one-dimensional lookup axes keyed by world axis.
        coupled_tabular: Optional coupled multi-axis lookup transform.
        tables: Table resolver used by ``-TAB`` metadata.
        cpdis1: First prior-distortion lookup table.
        cpdis2: Second prior-distortion lookup table.
        det2im1: First detector-to-image lookup table.
        det2im2: Second detector-to-image lookup table.

    Examples:
        >>> from torchsky.wcs import WCS
        >>> wcs = WCS({"NAXIS": 2, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"})
        >>> wcs.pixel_n_dim
        2
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
        **kwargs: Any,
    ):
        """
        Initialize WCS object.

        Args:
            header: FITS header dictionary (or similar object behaving like dict)
            **kwargs: Manual override for WCS keywords (CRVAL, CRPIX, CD, etc.)
        """
        self.wcs_params: dict[str, Any] = {}
        self.sip: SIP | None = None
        self.tpv: TPV | None = None
        self.cpdis1: DistortionLookupTable | None = _coerce_distortion_lookup_table(
            cpdis1
        )
        self.cpdis2: DistortionLookupTable | None = _coerce_distortion_lookup_table(
            cpdis2
        )
        self.det2im1: DistortionLookupTable | None = _coerce_distortion_lookup_table(
            det2im1
        )
        self.det2im2: DistortionLookupTable | None = _coerce_distortion_lookup_table(
            det2im2
        )
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

        self._identify_celestial_axes()
        self._setup_projection_params()
        if self._has_celestial:
            lon_ctype = (
                str(self.wcs_params.get(f"CTYPE{self._celestial_lon_axis + 1}", ""))
                .strip()
                .upper()
            )
            has_pv = any(
                key.startswith(("PV1_", "PV2_")) for key in self._projection_params
            )
            if lon_ctype.endswith("TPV") or (lon_ctype.endswith("TAN") and has_pv):
                self.tpv = TPV(self._projection_params)
            else:
                self.tpv = None

        if self.tpv is not None and (
            self._celestial_lon_axis != 0 or self._celestial_lat_axis != 1
        ):
            raise ValueError(
                "TPV distortion requires longitude/latitude to be FITS axes 1/2 in that order"
            )

        if self.naxis != 2 and (
            self.sip is not None
            or self.cpdis1 is not None
            or self.cpdis2 is not None
            or self.det2im1 is not None
            or self.det2im2 is not None
        ):
            raise ValueError(
                "FITS SIP, CPDIS, and DET2IM distortions require a two-dimensional WCS"
            )

        # Default device
        self.device = torch.device("cpu")

        # Environment-controlled fused C++ path. Default is ON ("1").
        # Vectorized PyTorch can win at some sizes; set TORCHSKY_WCS_FUSED=0 to compare.
        self._enable_fused_wcs = os.environ.get("TORCHSKY_WCS_FUSED", "1") == "1"
        self._enable_cpp_dispatcher = (
            os.environ.get("TORCHSKY_CPP_DISPATCHER", "1") == "1"
        )

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

    def _pixel_to_world_2d_fast(
        self, x: Tensor, y: Tensor, origin: float = 0
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

        finite_center_input = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        if finite_center_input and self._use_cpp_center_pixel_to_world_small_n(
            x, y, device_ok=device_ok, dtype_ok=dtype_ok
        ):
            return _bound_cpp(self._cpp_center_pixel_to_world)(
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

        if self._is_center_cylindrical_shortcut():
            rel_x, rel_y = self._detector_relative_pixels(x, y, float(origin))
            xi, eta = self._apply_cd_2d(rel_x, rel_y)
            return self._center_equator_project(xi, eta)

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
            phi, theta = xi, eta  # rebound by the direct projection below
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
            and (
                (not x.requires_grad and not y.requires_grad)
                or (self._enable_cpp_dispatcher and _ops.has_wcs_dispatcher())
            )
        )
        if use_fused:
            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)
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

            x_native = (x_flat + (1.0 - origin)).contiguous()
            y_native = (y_flat + (1.0 - origin)).contiguous()
            ra, dec = _pixel_to_world_fused(
                x_native,
                y_native,
                crpix0=self._crpix0,
                crpix1=self._crpix1,
                cd00=self._cd00,
                cd01=self._cd01,
                cd10=self._cd10,
                cd11=self._cd11,
                proj_code=self._fused_proj_code,
                lambda_param=self._fused_pv2_1,
                hpx_H=self._hpx_h,
                hpx_K=self._hpx_k,
                east_x=ex,
                east_y=ey,
                north_x=nx,
                north_y=ny,
                north_z=nz,
                radial_x=rx,
                radial_y=ry,
                radial_z=rz,
                ra_p_rad=self._ra_p_rad,
                dec_p_rad=self._dec_p_rad,
                phi_p_rad=self._phi_p_rad,
                native_type=self._native_type,
                alpha0=self.alpha0,
                delta0=self.delta0,
                is_tpv=self._is_tpv,
                tpv_idx1=self._fused_tpv_idx1,
                tpv_c1=self._fused_tpv_c1,
                tpv_idx2=self._fused_tpv_idx2,
                tpv_c2=self._fused_tpv_c2,
            )
            return ra.reshape(x.shape), dec.reshape(x.shape)

        core_fn = self._compiled_pixel_to_world_2d_fn or self._pixel_to_world_2d_core
        return core_fn(x, y, float(origin))

    def _world_to_pixel_2d_fast(
        self, ra: Tensor, dec: Tensor, iterative: bool = True, origin: float = 0
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
        if (
            self._is_tan
            and not self._is_tpv
            and not self._is_tnx
            and not self._is_zpx
            and self._celestial_postrot is None
        ):
            xi, eta = self._tan_intermediate_from_radec(ra, dec)
            if self.sip is None:
                px, py = self._apply_cd_inv_2d(xi, eta)
            else:
                rel_x, rel_y = self._apply_cd_inv_rel_2d(xi, eta)
                rel_x, rel_y = self.sip.invert_distortion(rel_x, rel_y)
                px = rel_x + self._crpix0
                py = rel_y + self._crpix1
            if origin == 1:
                return px, py
            if origin == 0:
                return px.sub(1.0), py.sub(1.0)
            off = float(origin - 1)
            return px.add(off), py.add(off)

        valid_center_input = bool(
            torch.isfinite(ra).all() and torch.isfinite(dec).all()
        )
        if self._is_mer:
            valid_center_input = valid_center_input and bool((dec.abs() < 90.0).all())
        if valid_center_input and self._use_cpp_center_world_to_pixel_small_n(
            ra, dec, device_ok=device_ok, dtype_ok=dtype_ok
        ):
            # Gated by _use_cpp_center_world_to_pixel_small_n (binding present,
            # supported projection, CPU float64, no autograd); errors propagate.
            return _bound_cpp(self._cpp_center_world_to_pixel)(
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

        if self._is_center_cylindrical_shortcut():
            xi, eta = self._center_equator_deproject(ra, dec)
            return self._pixels_from_intermediate(xi, eta, float(origin))

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
                rel_x, rel_y = self._apply_cd_inv_rel_2d(xi, eta)
                px = rel_x + self._crpix0 - (1.0 - origin)
                py = rel_y + self._crpix1 - (1.0 - origin)
                return px, py

        use_fused = (
            self._can_use_fused_w2p_static
            and not self._disable_fused_world_to_pixel
            and device_ok
            and dtype_ok
            and (
                (not ra.requires_grad and not dec.requires_grad)
                or (self._enable_cpp_dispatcher and _ops.has_wcs_dispatcher())
            )
            and (not self._is_tpv or not self._require_tpv()._invert_trace_enabled)
        )
        if use_fused:
            ra_flat = ra.reshape(-1)
            dec_flat = dec.reshape(-1)

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

            ra_native = ra_flat.contiguous()
            dec_native = dec_flat.contiguous()
            px, py = _world_to_pixel_fused(
                ra_native,
                dec_native,
                crpix0=self._crpix0,
                crpix1=self._crpix1,
                cdi00=self._cdi00,
                cdi01=self._cdi01,
                cdi10=self._cdi10,
                cdi11=self._cdi11,
                proj_code=self._fused_proj_code,
                lambda_param=self._fused_pv2_1,
                hpx_H=self._hpx_h,
                hpx_K=self._hpx_k,
                east_x=ex,
                east_y=ey,
                north_x=nx,
                north_y=ny,
                north_z=nz,
                radial_x=rx,
                radial_y=ry,
                radial_z=rz,
                ra_p_rad=self._ra_p_rad,
                dec_p_rad=self._dec_p_rad,
                phi_p_rad=self._phi_p_rad,
                native_type=self._native_type,
                alpha0=self.alpha0,
                delta0=self.delta0,
                is_tpv=self._is_tpv,
                tpv_idx1=self._fused_tpv_idx1,
                tpv_c1=self._fused_tpv_c1,
                tpv_idx2=self._fused_tpv_idx2,
                tpv_c2=self._fused_tpv_c2,
            )
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

    def compile(self, **kwargs: Any) -> "WCS":
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
        fast paths. Run ``benchmarks/bench_wcs_small_n_paths.py`` from the
        torchsky-harness checkout for timings against Astropy.

        Args:
            **kwargs: Arguments passed to torch.compile (e.g., mode, options).
        """
        compile_kwargs: dict[str, Any] = {"mode": "reduce-overhead"}
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

    def _mps_float32_error_pixels(self) -> float:
        """Expected float32 (MPS) round-trip position error in pixels.

        One float32 ulp of the largest world value on each celestial axis
        (longitude up to 360 degrees, scaled by ``cos(dec_0)``; latitude up to 90)
        divided by the smallest pixel scale (norm of a CD column, degrees per
        pixel), plus one ulp of the largest pixel coordinate (``NAXISn`` or
        ``2 CRPIXn``). Zero without a celestial pair.
        """
        if not self._has_celestial:
            return 0.0
        cd = self.cd.detach().cpu().to(torch.float64)
        scale = float(torch.linalg.vector_norm(cd, dim=0).min())
        if not scale > 0.0:
            return float("inf")
        world = max(
            _float32_ulp(360.0) * math.cos(math.radians(min(abs(self.delta0), 89.0))),
            _float32_ulp(90.0),
        )
        extent = 1.0
        for i in range(self.naxis):
            size = float(self.wcs_params.get(f"NAXIS{i + 1}", 0.0) or 0.0)
            extent = max(extent, size, 2.0 * abs(float(self.crpix[i])))
        return world / scale + _float32_ulp(extent)

    def _warn_mps_precision(self) -> None:
        """Emit :class:`PrecisionWarning` once per WCS when MPS float32 is too coarse."""
        if getattr(self, "_mps_precision_warned", False):
            return
        error = self._mps_float32_error_pixels()
        if error >= _MPS_PIXEL_IDENTIFICATION_LIMIT_PX:
            self._mps_precision_warned = True
            warnings.warn(
                f"WCS on MPS is float32: expected position error {error:.2f} pixel "
                "(>= 0.5 pixel) at this pixel scale; the nearest pixel is not "
                "reliable. Use CPU or CUDA (float64) for this grid.",
                PrecisionWarning,
                stacklevel=3,
            )

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
                    _require_distortion_lookup_table(lut_obj).to(self.device),
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
            assert self._pixel_affine_offset is not None
            self._set_pixel_affine(self._pixel_affine_matrix, self._pixel_affine_offset)
        if self.device.type == "mps":
            self._warn_mps_precision()
        return self

    # Rows within ``_REF_TOL`` of CRPIX (pixel) / of the CRPIX world position
    # (degrees) are the only ones where the native-longitude chain rule is 0 * inf.
    _REF_TOL = 1e-7
    _KINKED_CODES = ("XPH", "QSC")
    _KINK_STEP = 1e-4

    def slice(
        self,
        view: "slice | int | tuple[slice | int | EllipsisType, ...]",
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
        only_slices = tuple(item for item in pixel_items if isinstance(item, slice))
        if len(only_slices) == len(pixel_items):
            return self._slice_preserve_axes(only_slices)

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

    def to_header(
        self, *, relax: bool = True, include_naxis: bool = False
    ) -> dict[str, Any]:
        header = super().to_header(relax=relax, include_naxis=include_naxis)
        suffix = self.wcs_key
        for axis, factor_t in enumerate(self._celestial_unit_factors, start=1):
            factor = float(factor_t.detach().cpu().item())
            if factor == 1.0:
                continue
            crval_key = f"CRVAL{axis}{suffix}"
            if crval_key in header:
                header[crval_key] = float(header[crval_key]) / factor
            cdelt_key = f"CDELT{axis}{suffix}"
            if cdelt_key in header:
                header[cdelt_key] = float(header[cdelt_key]) / factor
            for pixel_axis in range(1, self.naxis + 1):
                cd_key = f"CD{axis}_{pixel_axis}{suffix}"
                if cd_key in header:
                    header[cd_key] = float(header[cd_key]) / factor
        return header

    def _clone_preserving_wcs_state(self) -> "WCS":
        tabular_axes = {
            axis + 1: tabular for axis, tabular in self.tabular_axes.items()
        }
        clone = WCS(
            self.to_header(relax=True, include_naxis=True),
            key=self.wcs_key,
            tabular_axes=tabular_axes,
            coupled_tabular=self.coupled_tabular,
            coupled_tabular_axes=tuple(axis + 1 for axis in self.coupled_tabular_axes),
            coupled_tabular_pixel_axes=tuple(
                axis + 1 for axis in self.coupled_tabular_pixel_axes
            ),
            cpdis1=self.cpdis1,
            cpdis2=self.cpdis2,
            det2im1=self.det2im1,
            det2im2=self.det2im2,
        ).to(self.device)
        if self._pixel_affine_matrix is not None:
            assert self._pixel_affine_offset is not None
            clone._set_pixel_affine(
                self._pixel_affine_matrix, self._pixel_affine_offset
            )
        return clone

    def _rebase_sip_header_for_diagonal_pixel_map(
        self, header: dict[str, Any], steps: tuple[int, int]
    ) -> None:
        if self.sip is None:
            return
        sx, sy = steps
        coefficient_sets = (
            ("A", self.sip.a_coeffs, sx),
            ("B", self.sip.b_coeffs, sy),
            ("AP", self.sip.ap_coeffs, sx),
            ("BP", self.sip.bp_coeffs, sy),
        )
        for prefix, coefficients, output_scale in coefficient_sets:
            for (p, q), coefficient in coefficients.items():
                header[f"{prefix}_{p}_{q}"] = float(
                    coefficient * (sx**p) * (sy**q) / output_scale
                )

    def _wcs_state_constructor_kwargs(self) -> dict[str, Any]:
        return {
            "tabular_axes": {
                axis + 1: tabular for axis, tabular in self.tabular_axes.items()
            },
            "coupled_tabular": self.coupled_tabular,
            "coupled_tabular_axes": tuple(
                axis + 1 for axis in self.coupled_tabular_axes
            ),
            "coupled_tabular_pixel_axes": tuple(
                axis + 1 for axis in self.coupled_tabular_pixel_axes
            ),
        }

    def _slice_preserve_axes(self, pixel_slices: tuple[slice, ...]) -> "WCS":
        starts: list[int] = []
        stops: list[int | None] = []
        steps: list[int] = []
        new_shape: list[int] | None = [] if self._pixel_shape is not None else None
        for axis, axis_slice in enumerate(pixel_slices):
            axis_len = None if self._pixel_shape is None else self._pixel_shape[axis]
            start, stop, step = self._normalize_axis_slice(axis_slice, axis_len)
            starts.append(start)
            stops.append(stop)
            steps.append(step)
            if new_shape is not None:
                if stop is None:
                    stop = axis_len
                assert stop is not None
                new_shape.append(len(range(start, stop, step)))

        identity = all(
            start == 0
            and step == 1
            and (
                stop is None
                or (self._pixel_shape is not None and stop == self._pixel_shape[axis])
            )
            for axis, (start, stop, step) in enumerate(zip(starts, stops, steps))
        )
        if identity:
            return self._clone_preserving_wcs_state()

        step_t = torch.as_tensor(
            steps, dtype=self._wcs_coord_dtype(self.device), device=self.device
        )
        start_t = torch.as_tensor(
            starts, dtype=self._wcs_coord_dtype(self.device), device=self.device
        )

        if self._pixel_affine_matrix is not None:
            if self.naxis < 2:
                raise ValueError("pixel-affine WCS must have at least two axes")
            header = self._copy_wcs_header()
            header["WCSAXES"] = self.naxis
            header["NAXIS"] = self.naxis
            new_crpix = self.crpix.clone()
            new_cd = self.cd_full.clone()
            new_pc = self.pc_full.clone()
            affine_axes = list(self._pixel_affine_axes())
            other_axes = [axis for axis in range(self.naxis) if axis not in affine_axes]
            if other_axes:
                new_crpix[other_axes] = (
                    1.0
                    - (start_t[other_axes] + 1.0 - self.crpix[other_axes])
                    / step_t[other_axes]
                )
                new_cd[:, other_axes] = new_cd[:, other_axes] * step_t[other_axes]
                new_pc[:, other_axes] = new_pc[:, other_axes] * step_t[other_axes]
            for i in range(1, self.naxis + 1):
                header[f"CRPIX{i}"] = float(new_crpix[i - 1].detach().cpu().item())
                unit_factor = float(self._celestial_unit_factors[i - 1].item())
                if self._is_tpv and self._tpv_uses_pixel_intermediate:
                    header[f"CDELT{i}"] = float(self.cdelt[i - 1].item()) / unit_factor
                    for j in range(1, self.naxis + 1):
                        header[f"PC{i}_{j}"] = float(
                            new_pc[i - 1, j - 1].detach().cpu().item()
                        )
                        header.pop(f"CD{i}_{j}", None)
                else:
                    for j in range(1, self.naxis + 1):
                        header[f"CD{i}_{j}"] = (
                            float(new_cd[i - 1, j - 1].detach().cpu().item())
                            / unit_factor
                        )
                    header.pop(f"CDELT{i}", None)
                    for j in range(1, self.naxis + 1):
                        header.pop(f"PC{i}_{j}", None)
                if new_shape is not None:
                    header[f"NAXIS{i}"] = int(new_shape[i - 1])
            sliced = WCS(
                header,
                key=self.wcs_key,
                cpdis1=self.cpdis1,
                cpdis2=self.cpdis2,
                det2im1=self.det2im1,
                det2im2=self.det2im2,
                **self._wcs_state_constructor_kwargs(),
            ).to(self.device)
            spatial_start = start_t[affine_axes]
            assert self._pixel_affine_offset is not None
            step_inv = torch.diag(1.0 / step_t[affine_axes])
            sliced._set_pixel_affine(
                step_inv @ self._pixel_affine_matrix,
                step_inv @ (self._pixel_affine_offset - spatial_start),
            )
            return sliced

        new_crpix = 1.0 - (start_t + 1.0 - self.crpix) / step_t
        new_cd = self.cd_full @ torch.diag(step_t)
        new_pc = self.pc_full @ torch.diag(step_t)

        header = self._copy_wcs_header()
        header["WCSAXES"] = self.naxis
        header["NAXIS"] = self.naxis
        for i in range(1, self.naxis + 1):
            header[f"CRPIX{i}"] = float(new_crpix[i - 1].detach().cpu().item())
            unit_factor = float(self._celestial_unit_factors[i - 1].item())
            if self._is_tpv and self._tpv_uses_pixel_intermediate:
                header[f"CDELT{i}"] = float(self.cdelt[i - 1].item()) / unit_factor
                for j in range(1, self.naxis + 1):
                    header[f"PC{i}_{j}"] = float(
                        new_pc[i - 1, j - 1].detach().cpu().item()
                    )
                    header.pop(f"CD{i}_{j}", None)
            else:
                for j in range(1, self.naxis + 1):
                    header[f"CD{i}_{j}"] = (
                        float(new_cd[i - 1, j - 1].detach().cpu().item()) / unit_factor
                    )
                header.pop(f"CDELT{i}", None)
                for j in range(1, self.naxis + 1):
                    header.pop(f"PC{i}_{j}", None)
            if new_shape is not None:
                header[f"NAXIS{i}"] = int(new_shape[i - 1])

        cpdis1 = self.cpdis1
        cpdis2 = self.cpdis2
        det2im1 = self.det2im1
        det2im2 = self.det2im2
        if self.naxis >= 2:
            spatial_starts = (starts[0], starts[1])
            spatial_steps = (steps[0], steps[1])
            self._rebase_sip_header_for_diagonal_pixel_map(header, spatial_steps)
            if cpdis1 is not None:
                cpdis1 = cpdis1.rebased_for_diagonal_pixel_map(
                    spatial_starts, spatial_steps, component_axis=0
                )
            if cpdis2 is not None:
                cpdis2 = cpdis2.rebased_for_diagonal_pixel_map(
                    spatial_starts, spatial_steps, component_axis=1
                )
            if det2im1 is not None:
                det2im1 = det2im1.rebased_for_diagonal_pixel_map(
                    spatial_starts, spatial_steps, component_axis=0
                )
            if det2im2 is not None:
                det2im2 = det2im2.rebased_for_diagonal_pixel_map(
                    spatial_starts, spatial_steps, component_axis=1
                )
        return WCS(
            header,
            key=self.wcs_key,
            cpdis1=cpdis1,
            cpdis2=cpdis2,
            det2im1=det2im1,
            det2im2=det2im2,
            **self._wcs_state_constructor_kwargs(),
        ).to(self.device)

    def _copy_wcs_header(self) -> dict[str, Any]:
        header: dict[str, Any] = {}
        for key, value in self.wcs_params.items():
            if isinstance(value, (str, int, float, bool)):
                header[str(key)] = value
        if self._pixel_shape is not None:
            header["NAXIS"] = self.naxis
            for i, size in enumerate(self._pixel_shape, start=1):
                header[f"NAXIS{i}"] = int(size)
        if self.sip is not None:
            coefficient_sets = (
                ("A", self.sip.a_order, self.sip.a_coeffs),
                ("B", self.sip.b_order, self.sip.b_coeffs),
                ("AP", self.sip.ap_order, self.sip.ap_coeffs),
                ("BP", self.sip.bp_order, self.sip.bp_coeffs),
            )
            for prefix, order, coefficients in coefficient_sets:
                if order > 0 or coefficients:
                    header[f"{prefix}_ORDER"] = int(order)
                for (p, q), coefficient in coefficients.items():
                    header[f"{prefix}_{p}_{q}"] = float(coefficient)
        return header

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

        The transform acts on the two pixel axes associated with the celestial
        CTYPE pair. Distorted WCS store the affine and apply it in pixel space
        before SIP/TPV/TNX/ZPX distortion.
        """
        if self.naxis < 2:
            raise ValueError("pixel affine correction requires at least two WCS axes")
        affine = torch.as_tensor(
            matrix, dtype=self._wcs_coord_dtype(self.device), device=self.device
        )
        shift = torch.as_tensor(
            offset, dtype=self._wcs_coord_dtype(self.device), device=self.device
        )
        if affine.shape != (2, 2):
            raise ValueError("matrix must have shape (2, 2)")
        if shift.shape != (2,):
            raise ValueError("offset must have shape (2,)")
        if torch.equal(
            affine, torch.eye(2, dtype=affine.dtype, device=affine.device)
        ) and not bool(torch.count_nonzero(shift)):
            return self._clone_preserving_wcs_state()
        if self._pixel_affine_matrix is not None:
            assert self._pixel_affine_offset is not None
            corrected = self._clone_preserving_wcs_state()
            corrected._set_pixel_affine(
                affine @ self._pixel_affine_matrix,
                affine @ self._pixel_affine_offset + shift,
            )
            return corrected
        affine_inv = torch.linalg.inv(affine)
        distorted = (
            self.sip is not None
            or self._is_tpv
            or self._is_tnx
            or self._is_zpx
            or self._has_paper4_lut()
        )

        header = self._copy_wcs_header()
        if not distorted:
            for i in range(1, self.naxis + 1):
                unit_factor = float(self._celestial_unit_factors[i - 1].item())
                for j in range(1, self.naxis + 1):
                    header[f"CD{i}_{j}"] = (
                        float(self.cd_full[i - 1, j - 1].detach().cpu().item())
                        / unit_factor
                    )
                header.pop(f"CDELT{i}", None)
                for j in range(1, self.naxis + 1):
                    header.pop(f"PC{i}_{j}", None)
            affine_axes = list(self._pixel_affine_axes())
            old_crpix = self.crpix[affine_axes]
            new_crpix = 1.0 + shift + affine @ (old_crpix - 1.0)
            new_cd_columns = self.cd_full[:, affine_axes] @ affine_inv
            for pair_index, pixel_axis in enumerate(affine_axes):
                header[f"CRPIX{pixel_axis + 1}"] = float(
                    new_crpix[pair_index].detach().cpu().item()
                )
                for world_axis in range(self.naxis):
                    header[f"CD{world_axis + 1}_{pixel_axis + 1}"] = float(
                        new_cd_columns[world_axis, pair_index].detach().cpu().item()
                    ) / float(self._celestial_unit_factors[world_axis].item())

        prefix = str(metadata_prefix).strip().upper()
        if prefix:
            header[f"{prefix}0"] = "pixel_affine"
            for i in range(2):
                header[f"{prefix}B{i + 1}"] = float(shift[i].detach().cpu().item())
                for j in range(2):
                    header[f"{prefix}A{i + 1}{j + 1}"] = float(
                        affine[i, j].detach().cpu().item()
                    )
        corrected = WCS(
            header,
            key=self.wcs_key,
            cpdis1=self.cpdis1,
            cpdis2=self.cpdis2,
            det2im1=self.det2im1,
            det2im2=self.det2im2,
            **self._wcs_state_constructor_kwargs(),
        )
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

    def with_affine(
        self,
        matrix_or_fit: Any,
        offset: Any | None = None,
        *,
        metadata_prefix: str = "TSKYAF",
    ) -> "WCS":
        """Return a WCS corrected by an ``AffineFitResult`` or ``(matrix, offset)``."""
        if (
            offset is None
            and hasattr(matrix_or_fit, "matrix")
            and hasattr(matrix_or_fit, "offset")
        ):
            return self.with_affine_fit_correction(
                matrix_or_fit, metadata_prefix=metadata_prefix
            )
        if offset is None:
            raise ValueError(
                "offset must be provided when matrix_or_fit is not an AffineFitResult"
            )
        return self.with_pixel_affine_correction(
            matrix_or_fit, offset, metadata_prefix=metadata_prefix
        )

    def _normalize_wcs_slice(
        self,
        view: "slice | int | tuple[slice | int | EllipsisType, ...]",
        *,
        numpy_order: bool,
    ) -> "tuple[slice | int, ...]":
        if isinstance(view, (slice, int)) and not isinstance(view, bool):
            items: tuple[slice | int | EllipsisType, ...] = (view,)
        else:
            items = tuple(cast("tuple[slice | int | EllipsisType, ...]", view))
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
        # Ellipsis entries were expanded above, so only slices and ints remain.
        return cast("tuple[slice | int, ...]", tuple(pixel_items))

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
        uncoupled from retained axes. Spatial WCS instances keep both celestial
        projection axes; retained axes may be reordered in FITS/API order.
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
        celestial_axes = list(self._celestial_axes)
        for axis in range(self.naxis):
            if axis in celestial_axes:
                continue
            self._validate_axis_uncoupled_for_drop(axis)
        return self._sub_wcs(celestial_axes)

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
        fixed_rel_by_axis: dict[int, Tensor | float] | None = None,
    ) -> "WCS":
        if len(keep_axes) == 0:
            raise ValueError("at least one WCS axis must be kept")
        if len(set(keep_axes)) != len(keep_axes):
            raise ValueError("kept axes must be unique")
        fixed_rel_by_axis = {} if fixed_rel_by_axis is None else fixed_rel_by_axis
        if self._is_spatial:
            celestial = list(self._celestial_axes)
            dropped_celestial = [a for a in celestial if a not in keep_axes]
            if dropped_celestial:
                if any(a not in fixed_rel_by_axis for a in dropped_celestial):
                    raise NotImplementedError(
                        "spatial WCS subsetting must keep both celestial axes"
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
        dropped_celestial = [a for a in self._celestial_axes if a not in keep_axes]
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
            world_ref_raw = self.pixel_to_world(ref.unsqueeze(0), origin=0)
            if isinstance(world_ref_raw, tuple):
                world_ref_raw = torch.stack(world_ref_raw, dim=-1)
            world_ref = world_ref_raw.reshape(-1, self.naxis)[0]
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
            if (
                self._is_spatial
                and dropped_celestial
                and old_axis in self._celestial_axes
            ):
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
            unit_factor = float(self._celestial_unit_factors[old_axis].item())
            header[f"CRVAL{new_axis}"] = (
                float(crval.detach().cpu().item()) / unit_factor
            )
            if f"CUNIT{old_num}" in self.wcs_params:
                header[f"CUNIT{new_axis}"] = self.wcs_params[f"CUNIT{old_num}"]
            if self._pixel_shape is not None:
                header[f"NAXIS{new_axis}"] = int(self._pixel_shape[old_axis])
            for out_axis, _old_out in enumerate(keep_axes, start=1):
                header[f"CD{new_axis}_{out_axis}"] = (
                    float(cd_sub[new_axis - 1, out_axis - 1].detach().cpu().item())
                    / unit_factor
                )

        if self._is_spatial and not dropped_celestial:
            old_to_new = {
                old_axis + 1: new_axis
                for new_axis, old_axis in enumerate(keep_axes, start=1)
            }
            for key, value in self.wcs_params.items():
                key_s = str(key)
                if not key_s.startswith("PV") or "_" not in key_s:
                    continue
                axis_text, suffix = key_s[2:].split("_", 1)
                if axis_text.isdigit() and int(axis_text) in old_to_new:
                    header[f"PV{old_to_new[int(axis_text)]}_{suffix}"] = value

        tabular_axes = {
            keep_axes.index(axis) + 1: tabular
            for axis, tabular in self.tabular_axes.items()
            if axis in keep_axes
        }
        has_pixel_distortion = self.sip is not None or self._has_paper4_lut()
        identity_axes = keep_axes == list(range(self.naxis))
        if has_pixel_distortion and not identity_axes:
            raise NotImplementedError(
                "SIP/CPDIS/DET2IM WCS axis reordering is not supported"
            )
        if identity_axes and self.sip is not None:
            for key, value in self._copy_wcs_header().items():
                if str(key).startswith(("A_", "B_", "AP_", "BP_")):
                    header[str(key)] = value
        return WCS(
            header,
            key=self.wcs_key,
            tabular_axes=tabular_axes,
            cpdis1=self.cpdis1 if identity_axes else None,
            cpdis2=self.cpdis2 if identity_axes else None,
            det2im1=self.det2im1 if identity_axes else None,
            det2im2=self.det2im2 if identity_axes else None,
        )

    def _require_wcs_device(self, value: Tensor, name: str) -> Tensor:
        """Differentiable forms need the WCS on the coordinates' device.

        The plain forms follow accelerator inputs; the differentiable forms keep
        WCS parameters in the graph (WCS fitting), so a device mismatch is
        reported here instead of moving parameters or failing inside torch.
        """
        if (
            isinstance(self.device, torch.device)
            and value.device.type != self.device.type
        ):
            raise ValueError(
                f"{name} are on {value.device} but this WCS is on {self.device}; "
                f"call wcs.to({str(value.device)!r}) first (the non-differentiable "
                "pixel_to_world/world_to_pixel follow the input device)"
            )
        return value

    def _coerce_stacked_coord(self, value: Any, name: str) -> Tensor:
        tensor = self._as_wcs_coord(value)
        self._require_wcs_device(tensor, name)
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
            x_t, y_t = torch.broadcast_tensors(x_t, y_t)
            ra, dec = self._pixel_to_world_2d_fast(x_t, y_t, origin)
            world = self._world_from_celestial_pair(ra, dec)
            return world[..., 0], world[..., 1]

        if len(args) == 1:
            pixels = self._as_wcs_coord(args[0])
            if (
                self.naxis == 2
                and isinstance(self.device, torch.device)
                and pixels.device.type != self.device.type
                and pixels.shape[-1:] == (2,)
            ):
                # Follow the coordinates' device like the two-argument form does.
                lon, lat = self.pixel_to_world(
                    pixels[..., 0], pixels[..., 1], origin=origin
                )
                return torch.stack([lon, lat], dim=-1)
        else:
            coordinates = torch.broadcast_tensors(
                *(self._as_wcs_coord(a) for a in args)
            )
            pixels = torch.stack(
                coordinates,
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
        # Pixel-space distortions operate on FITS pixel axes 1/2. Celestial
        # projection rows are gathered independently below.
        if self._pixel_affine_matrix is not None and self.naxis >= 2:
            axis0, axis1 = self._pixel_affine_axes()
            p0 = rel[:, axis0] + self.crpix[axis0] - 1.0
            p1 = rel[:, axis1] + self.crpix[axis1] - 1.0
            src_x, src_y = self._corrected_pixels_to_source(p0, p1, origin=0.0)
            rel = rel.clone()
            rel[:, axis0] = src_x - (self.crpix[axis0] - 1.0)
            rel[:, axis1] = src_y - (self.crpix[axis1] - 1.0)

        rel_x = rel[:, 0]
        rel_y = rel[:, 1] if self.naxis >= 2 else torch.zeros_like(rel_x)

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
            results = []
            for axis in range(self.naxis):
                if axis in coupled_world:
                    results.append(coupled_world[axis])
                else:
                    results.append(
                        self._apply_tabular_pixel_to_world(linear_world[:, axis], axis)
                    )
        else:
            if self.sip is not None:
                rel_x, rel_y = self.sip.distort(rel_x, rel_y)
            if self.cpdis1 is not None or self.cpdis2 is not None:
                rel_x, rel_y = self._apply_cpdis_forward(rel_x, rel_y)
            if self.sip is not None or self._has_paper4_lut():
                rel = rel.clone()
                rel[:, 0] = rel_x
                rel[:, 1] = rel_y

            intermediate = torch.matmul(rel, self.cd_full.T)
            xi = intermediate[:, self._celestial_lon_axis]
            eta = intermediate[:, self._celestial_lat_axis]
            if self._is_tpv:
                if self._tpv_uses_pixel_intermediate:
                    pc_intermediate = torch.matmul(rel, self.pc_full.T)
                    xi, eta = self._require_tpv().distort(
                        pc_intermediate[:, self._celestial_lon_axis],
                        pc_intermediate[:, self._celestial_lat_axis],
                    )
                    xi = xi * self.cdelt[self._celestial_lon_axis]
                    eta = eta * self.cdelt[self._celestial_lat_axis]
                else:
                    xi, eta = self._require_tpv().distort(xi, eta)
            if self._is_gls:
                lon, lat = self._gls_forward(xi, eta)
            elif (
                (self._is_tan or self._is_tpv)
                and not self._is_tnx
                and self._celestial_postrot is None
                and self.alpha0 == self.alpha_p
                and self.delta0 == self.delta_p
            ):
                lon, lat = self._project_tan(xi, eta)
            else:
                phi, theta = self._project_native_coords(xi, eta)
                lon, lat = self._spherical_rotation_optimized(
                    phi, theta, self._native_type, allow_cpp_sphere=allow_cpp_sphere
                )

            linear_world = self.crval + intermediate
            coupled_world = self._apply_coupled_tabular_pixel_to_world(linear_world)
            results = []
            for axis in range(self.naxis):
                if axis == self._celestial_lon_axis:
                    results.append(lon)
                elif axis == self._celestial_lat_axis:
                    results.append(lat)
                elif axis in coupled_world:
                    results.append(coupled_world[axis])
                else:
                    results.append(
                        self._apply_tabular_pixel_to_world(linear_world[:, axis], axis)
                    )

        return torch.stack(results, dim=-1)

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
            and self._celestial_postrot is None
            and not self._is_tpv
            and not self._is_tnx
            and not self._is_zpx
            and self.sip is None
            and not self._has_paper4_lut()
        ):
            xi, eta = self._apply_cd_2d(rel[:, 0], rel[:, 1])
            ra, dec = self._project_tan(xi, eta)
            return self._world_from_celestial_pair(ra, dec)
        return self._transform_optimized(rel, allow_cpp_sphere=allow_cpp_sphere)

    def world_to_pixel(self, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        World -> Pixel.
        """
        iterative = kwargs.get("iterative", True)
        origin = float(kwargs.get("origin", 0))

        if len(args) == 2 and self.naxis == 2:
            world_args = [self._as_wcs_coord(args[0]), self._as_wcs_coord(args[1])]
            ra_t = world_args[self._celestial_lon_axis]
            dec_t = world_args[self._celestial_lat_axis]
            ra_t, dec_t = torch.broadcast_tensors(ra_t, dec_t)
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
                    world[..., self._celestial_lon_axis],
                    world[..., self._celestial_lat_axis],
                    iterative=iterative,
                    origin=origin,
                )
                return torch.stack([px, py], dim=-1)
        else:
            coordinates = torch.broadcast_tensors(
                *(self._as_wcs_coord(a) for a in args)
            )
            world = torch.stack(
                coordinates,
                dim=-1,
            )

        original_shape = world.shape
        world_flat = world.reshape(-1, self.naxis)

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
            lon = world_flat[:, self._celestial_lon_axis]
            lat = world_flat[:, self._celestial_lat_axis]
            xi, eta = self._world_to_pixel_2d_core(
                lon,
                lat,
                1.0,
                intermediate_only=True,
            )
            linear_world = world_flat.clone()
            coupled_pixel = self._apply_coupled_tabular_world_to_pixel(world_flat)
            for axis in range(self.naxis):
                if axis not in self._celestial_axes:
                    linear_world[:, axis] = self._coupled_or_tabular_linear_world(
                        coupled_pixel, world_flat, axis
                    )
            intermediate = linear_world - self.crval
            intermediate[:, self._celestial_lon_axis] = xi
            intermediate[:, self._celestial_lat_axis] = eta
            rel_flat = torch.matmul(intermediate, self.cd_inv_full.T)
            if self.cpdis1 is not None or self.cpdis2 is not None:
                rel0, rel1 = self._apply_cpdis_inverse(rel_flat[:, 0], rel_flat[:, 1])
                rel_flat = rel_flat.clone()
                rel_flat[:, 0] = rel0
                rel_flat[:, 1] = rel1
            if self.sip is not None:
                rel0, rel1 = self.sip.invert_distortion(rel_flat[:, 0], rel_flat[:, 1])
                rel_flat = rel_flat.clone()
                rel_flat[:, 0] = rel0
                rel_flat[:, 1] = rel1
            pix_internal_flat = rel_flat + self.crpix
            if self.det2im1 is not None or self.det2im2 is not None:
                p0, p1 = self._apply_det2im_inverse(
                    pix_internal_flat[:, 0], pix_internal_flat[:, 1]
                )
                pix_internal_flat = pix_internal_flat.clone()
                pix_internal_flat[:, 0] = p0
                pix_internal_flat[:, 1] = p1

        # 3. Handle extra axes (NAXIS > 2) for non-spatial WCS
        if not is_spatial and self.naxis >= 3:
            # Already handled by cd_inv_full above
            pass
        elif is_spatial and self.naxis >= 3:
            pass

        # Adjust for origin: pix = pix_internal + (origin - 1)
        pix_flat = pix_internal_flat + (origin - 1)
        if self._pixel_affine_matrix is not None and is_spatial and self.naxis >= 2:
            axis0, axis1 = self._pixel_affine_axes()
            cx, cy = self._source_pixels_to_corrected(
                pix_flat[:, axis0], pix_flat[:, axis1], origin
            )
            pix_flat = pix_flat.clone()
            pix_flat[:, axis0] = cx
            pix_flat[:, axis1] = cy

        pix = pix_flat.view(original_shape)
        if len(args) > 1:
            return tuple(pix[..., i] for i in range(self.naxis))
        return pix
