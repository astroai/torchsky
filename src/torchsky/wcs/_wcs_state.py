"""Tabular-axis setup, coordinate dtype handling, and distortion helpers for :class:`torchsky.wcs.WCS`."""

from __future__ import annotations

import torch
from torch import Tensor
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Sequence,
)
import math

from ._math import _wrap_lon180, _wrap_lon360

from .tpv import TPV
from .tabular import (
    CoupledTabularLookupWCS,
    TabularAxis,
)

if TYPE_CHECKING:
    pass


class _WCSStateMixin:
    if TYPE_CHECKING:
        # State and methods supplied by the host class.
        def __getattr__(self, name: str) -> Any: ...

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
            axes[idx] = axis.to(
                device=self.device, dtype=self._wcs_coord_dtype(self.device)
            )
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

    def _require_tpv(self) -> TPV:
        tpv = self.tpv
        assert tpv is not None  # callers check ``_is_tpv`` first
        return tpv

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
            self._warn_mps_precision()
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
            and not self._has_paper4_lut()
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
        old = (
            torch.matmul(stacked, self._pixel_affine_inv.T)
            + self._pixel_affine_inv_offset
        )
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
        new = (
            torch.matmul(stacked, self._pixel_affine_matrix.T)
            + self._pixel_affine_offset
        )
        return new[..., 0] + origin_f, new[..., 1] + origin_f

    def _has_paper4_lut(self) -> bool:
        return (
            self.cpdis1 is not None
            or self.cpdis2 is not None
            or self.det2im1 is not None
            or self.det2im2 is not None
        )

    def _apply_det2im_forward(self, x1: Tensor, y1: Tensor) -> tuple[Tensor, Tensor]:
        if self.det2im1 is None and self.det2im2 is None:
            return x1, y1
        lut1 = _core_mod._coerce_distortion_lookup_table(self.det2im1)
        lut2 = _core_mod._coerce_distortion_lookup_table(self.det2im2)
        dx = lut1.eval_tensor(x1, y1) if lut1 is not None else torch.zeros_like(x1)
        dy = lut2.eval_tensor(x1, y1) if lut2 is not None else torch.zeros_like(y1)
        return x1 + dx, y1 + dy

    def _apply_det2im_inverse(
        self, x_d2im: Tensor, y_d2im: Tensor, *, max_iter: int = 16
    ) -> tuple[Tensor, Tensor]:
        if self.det2im1 is None and self.det2im2 is None:
            return x_d2im, y_d2im
        lut1 = _core_mod._coerce_distortion_lookup_table(self.det2im1)
        lut2 = _core_mod._coerce_distortion_lookup_table(self.det2im2)
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
        lut1 = _core_mod._coerce_distortion_lookup_table(self.cpdis1)
        lut2 = _core_mod._coerce_distortion_lookup_table(self.cpdis2)
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
        lut1 = _core_mod._coerce_distortion_lookup_table(self.cpdis1)
        lut2 = _core_mod._coerce_distortion_lookup_table(self.cpdis2)
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

    def _crota_pc_overrides(self) -> dict[tuple[int, int], float]:
        """PCi_j implied by the legacy CROTAi keyword (Paper I, sec. 6.1 / WCSLIB).

        Only used when no CD/PC keywords exist. The rotation is the CROTA of the
        latitude axis: PC_lon,lon = PC_lat,lat = cos(rho), PC_lon,lat =
        -sin(rho) * CDELT_lat / CDELT_lon, PC_lat,lon = sin(rho) * CDELT_lon / CDELT_lat.
        """
        if not self._has_celestial or self.naxis < 2:
            return {}
        if any(key.startswith("PC") for key in self.wcs_params):
            return {}
        lon, lat = self._celestial_lon_axis, self._celestial_lat_axis
        rho = self.wcs_params.get(f"CROTA{lat + 1}")
        if rho is None:
            return {}
        rho = math.radians(float(rho))
        c_lon = float(self.wcs_params.get(f"CDELT{lon + 1}", 1.0))
        c_lat = float(self.wcs_params.get(f"CDELT{lat + 1}", 1.0))
        if c_lon == 0.0 or c_lat == 0.0:
            return {}
        return {
            (lon + 1, lon + 1): math.cos(rho),
            (lat + 1, lat + 1): math.cos(rho),
            (lon + 1, lat + 1): -math.sin(rho) * c_lat / c_lon,
            (lat + 1, lon + 1): math.sin(rho) * c_lon / c_lat,
        }


# Late-bound so monkeypatching ``core`` attributes still reaches this code.
from . import core as _core_mod  # noqa: E402
