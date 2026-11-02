"""Core 2-D pixel/world kernels for :class:`torchsky.wcs.WCS`."""

from __future__ import annotations

import torch
from torch import Tensor
from typing import (
    TYPE_CHECKING,
    Any,
    Tuple,
)
import math

from ._math import _sincos, _wrap_lon180, _wrap_lon360

from .zenithal import project_zenithal, deproject_zenithal
from .cylindrical import project_cylindrical, deproject_cylindrical
from .allsky import project_allsky, deproject_allsky
from .conic import project_conic, deproject_conic
from .quadcube import project_quadcube, deproject_quadcube
from .legacy import (
    deproject_zpx,
    invert_tnx_plane,
    project_tnx,
    project_zpx,
)

if TYPE_CHECKING:
    pass

from ._core_support import (
    _CENTER_CPP_P2W_THRESHOLDS,
    _CENTER_CPP_W2P_THRESHOLDS,
    _bound_cpp,
)


class _PixelWorldCoreMixin:
    if TYPE_CHECKING:
        # State and methods supplied by the host class.
        def __getattr__(self, name: str) -> Any: ...

    def _project_native_coords(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """Projection dispatch hot path with pre-bound family metadata."""
        kind = self._project_native_kind
        if kind == "zpx":
            return project_zpx(xi, eta, self._projection_params, self.wat_data)
        if kind == "tnx":
            xi_t, eta_t = project_tnx(xi, eta, self._projection_params, self.wat_data)
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
        xi = self._scaled_axis(self._cd00, rel_x) + self._scaled_axis(self._cd01, rel_y)
        eta = self._scaled_axis(self._cd10, rel_x) + self._scaled_axis(
            self._cd11, rel_y
        )
        return xi, eta

    @staticmethod
    def _scaled_axis(coefficient: float, value: Tensor) -> Tensor:
        if coefficient == 0.0:
            return torch.zeros_like(value)
        return coefficient * value

    def _apply_cd_inv_rel_2d(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        rel0 = self._scaled_axis(self._cdi00, xi) + self._scaled_axis(self._cdi01, eta)
        rel1 = self._scaled_axis(self._cdi10, xi) + self._scaled_axis(self._cdi11, eta)
        return rel0, rel1

    def _apply_cd_inv_2d(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply cached 2x2 inverse CD transform using scalar coefficients."""
        rel0, rel1 = self._apply_cd_inv_rel_2d(xi, eta)
        return rel0 + self._crpix0, rel1 + self._crpix1

    def _is_center_cylindrical_shortcut(self) -> bool:
        return self._use_center_equator_fast and (
            self._is_car or self._is_sfl or self._is_cea or self._is_mer or self._is_cyp
        )

    def _cylindrical_native_to_intermediate(
        self, phi: Tensor, theta: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Native spherical degrees to intermediate degrees for CAR/SFL/CEA/MER/CYP."""
        if self._is_car:
            return _wrap_lon180(phi), theta
        if self._is_sfl:
            return _wrap_lon180(phi) * torch.cos(theta * self._d2r), theta
        if self._is_cea:
            return (
                _wrap_lon180(phi),
                self._cea_eta_scale * torch.sin(theta * self._d2r),
            )
        if self._is_cyp:
            return deproject_cylindrical(
                phi, theta, "CYP", self._pv1_tensor, self._pv2_tensor
            )
        return deproject_cylindrical(phi, theta, "MER")

    def _center_equator_deproject(
        self, ra: Tensor, dec: Tensor
    ) -> Tuple[Tensor, Tensor]:
        phi = _wrap_lon180(ra - self.alpha0)
        return self._cylindrical_native_to_intermediate(phi, dec)

    def _center_equator_project(self, xi: Tensor, eta: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse of :meth:`_center_equator_deproject` for the same projections."""
        if self._is_car:
            return _wrap_lon360(xi + self.alpha0), eta
        if self._is_sfl:
            dec = eta
            cos_dec = torch.cos(dec * self._d2r)
            cos_safe = torch.sign(cos_dec) * torch.clamp(torch.abs(cos_dec), min=1e-12)
            return _wrap_lon360((xi / cos_safe) + self.alpha0), dec
        if self._is_cea:
            value = eta / self._cea_eta_scale
            valid = torch.isfinite(value) & (value.abs() <= 1.0 + 1e-12)
            theta = torch.asin(torch.clamp(value, -1.0, 1.0)) * self._r2d
            ra = _wrap_lon360(xi + self.alpha0)
            nan = torch.full_like(theta, float("nan"))
            return torch.where(valid, ra, nan), torch.where(valid, theta, nan)
        if self._is_mer:
            arg = torch.clamp(eta * self._d2r, -20.0, 20.0)
            theta = 2.0 * (torch.atan(torch.exp(arg)) * self._r2d - 45.0)
            return _wrap_lon360(xi + self.alpha0), theta
        phi, theta = project_cylindrical(
            xi, eta, "CYP", self._pv1_tensor, self._pv2_tensor
        )
        return _wrap_lon360(phi + self.alpha0), theta

    def _detector_relative_pixels(
        self, x: Tensor, y: Tensor, origin: float
    ) -> Tuple[Tensor, Tensor]:
        """Corrected pixels to CD-input offsets, including SIP and lookup distortions."""
        x, y = self._corrected_pixels_to_source(x, y, origin)
        x1 = x + (1.0 - origin)
        y1 = y + (1.0 - origin)
        if self.det2im1 is not None or self.det2im2 is not None:
            x1, y1 = self._apply_det2im_forward(x1, y1)
        rel_x = x1 - self._crpix0
        rel_y = y1 - self._crpix1
        if self.sip is not None:
            rel_x, rel_y = self.sip.distort(rel_x, rel_y)
        if self.cpdis1 is not None or self.cpdis2 is not None:
            rel_x, rel_y = self._apply_cpdis_forward(rel_x, rel_y)
        return rel_x, rel_y

    def _pixels_from_intermediate(
        self, xi: Tensor, eta: Tensor, origin: float
    ) -> Tuple[Tensor, Tensor]:
        """CD inverse, SIP, lookup distortions, CRPIX, and pixel-affine correction."""
        rel_x, rel_y = self._apply_cd_inv_rel_2d(xi, eta)
        if self.cpdis1 is not None or self.cpdis2 is not None:
            rel_x, rel_y = self._apply_cpdis_inverse(rel_x, rel_y)
        if self.sip is not None:
            rel_x, rel_y = self.sip.invert_distortion(rel_x, rel_y)
        x1 = rel_x + self._crpix0
        y1 = rel_y + self._crpix1
        if self.det2im1 is not None or self.det2im2 is not None:
            x1, y1 = self._apply_det2im_inverse(x1, y1)
        px = x1 - (1.0 - origin)
        py = y1 - (1.0 - origin)
        return self._source_pixels_to_corrected(px, py, origin)

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
            or self.sip is not None
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
            or self.sip is not None
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
            return _bound_cpp(self._cpp_ait_project)(xi, eta)
        if cpp_kind == "mol":
            return _bound_cpp(self._cpp_mol_project)(xi, eta)
        if cpp_kind == "hpx":
            return _bound_cpp(self._cpp_hpx_project)(xi, eta, self._hpx_h, self._hpx_k)
        if cpp_kind == "zenithal":
            return _bound_cpp(self._cpp_zenithal_project)(xi, eta, self._proj_code)
        if cpp_kind == "zpn":
            return _bound_cpp(self._cpp_zpn_project)(xi, eta, self._zpn_coeffs_cpu)
        raise RuntimeError("unsupported_cpp_direct_projection")

    def _cpp_deproject_direct_w2p(
        self, phi_w: Tensor, theta: Tensor, cpp_kind: str
    ) -> Tuple[Tensor, Tensor]:
        """Direct C++ deprojection dispatch used by world->pixel fast path."""
        if cpp_kind == "ait":
            return _bound_cpp(self._cpp_ait_deproject)(phi_w, theta)
        if cpp_kind == "hpx":
            return _bound_cpp(self._cpp_hpx_deproject)(
                phi_w, theta, self._hpx_h, self._hpx_k
            )
        if cpp_kind == "zenithal":
            return _bound_cpp(self._cpp_zenithal_deproject)(
                phi_w, theta, self._proj_code
            )
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
        valid = torch.isfinite(w) & (w > 64.0 * torch.finfo(w.dtype).eps)
        w_safe = torch.where(valid, w, torch.ones_like(w))
        scale = self._r2d / w_safe
        nan = torch.full_like(w, float("nan"))
        return torch.where(valid, u * scale, nan), torch.where(valid, v * scale, nan)

    def _pixel_to_world_2d_core(
        self, x: Tensor, y: Tensor, origin: float
    ) -> Tuple[Tensor, Tensor]:
        """
        Unified 2D spatial pixel-to-world core.
        """
        rel_x, rel_y = self._detector_relative_pixels(x, y, origin)
        xi, eta = self._apply_cd_2d(rel_x, rel_y)

        if self._is_gls:
            return self._gls_forward(xi, eta)

        # Center-equator cylindrical projections skip spherical rotation.
        if self._is_center_cylindrical_shortcut():
            return self._center_equator_project(xi, eta)

        # 3. TPV / Special distortions
        zpx_native = False
        # Every projection branch below rebinds phi/theta; this keeps them bound for the checker.
        phi, theta = xi, eta
        if self._is_tpv:
            if self._tpv_uses_pixel_intermediate:
                lon_scale = self.cdelt[self._celestial_lon_axis]
                lat_scale = self.cdelt[self._celestial_lat_axis]
                xi, eta = self._require_tpv().distort(xi / lon_scale, eta / lat_scale)
                xi, eta = xi * lon_scale, eta * lat_scale
            else:
                xi, eta = self._require_tpv().distort(xi, eta)
        elif self._is_tnx:
            xi, eta = project_tnx(xi, eta, self._projection_params, self.wat_data)
        elif self._is_zpx:
            # project_zpx returns native (phi, theta), not intermediate (xi, eta).
            phi, theta = project_zpx(xi, eta, self._projection_params, self.wat_data)
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
                valid_arc = torch.isfinite(r) & (r <= math.nextafter(180.0, math.inf))
                theta = 90.0 - torch.clamp(r, max=180.0)
                nan = torch.full_like(theta, float("nan"))
                phi = torch.where(valid_arc, phi, nan)
                theta = torch.where(valid_arc, theta, nan)
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
            invalid = (
                (~torch.isfinite(xi))
                | (~torch.isfinite(eta))
                | (abs_eta > self._hpx_eta_pole)
            )
            mask_eq = (~invalid) & (abs_eta <= self._hpx_eta_boundary)
            sigma = torch.clamp(
                (self._hpx_eta_pole - abs_eta) * self._hpx_inv_polar_denom, min=0.0
            )
            s_theta = torch.where(
                mask_eq,
                eta * self._hpx_inv_eta_scale,
                torch.sign(eta) * (1.0 - (sigma * sigma) / self._hpx_k),
            )
            theta = torch.asin(torch.clamp(s_theta, -1.0, 1.0)) * self._r2d
            facet_offset = torch.full_like(xi, self._hpx_facet_half_width)
            if self._hpx_k % 2.0 == 0.0:
                facet_offset = torch.where(
                    eta < 0.0, torch.zeros_like(xi), facet_offset
                )
            xc = (
                torch.floor(
                    (xi + 180.0 - facet_offset + self._hpx_facet_half_width)
                    / self._hpx_facet_spacing
                )
                * self._hpx_facet_spacing
                - 180.0
                + facet_offset
            )
            phi = torch.where(
                mask_eq, xi, xc + (xi - xc) / torch.clamp(sigma, min=1e-9)
            )
            invalid = invalid | (
                (~mask_eq)
                & (~invalid)
                & (torch.abs(xi - xc) > (self._hpx_facet_half_width * sigma + 1e-8))
            )
            nan = torch.full_like(phi, float("nan"))
            phi = torch.where(invalid, nan, phi)
            theta = torch.where(invalid, nan, theta)
        elif self._is_cea:
            phi = xi
            value = eta / self._cea_eta_scale
            valid = torch.isfinite(value) & (value.abs() <= 1.0 + 1e-12)
            theta = torch.asin(torch.clamp(value, -1.0, 1.0)) * self._r2d
            nan = torch.full_like(theta, float("nan"))
            phi = torch.where(valid, phi, nan)
            theta = torch.where(valid, theta, nan)
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

    def _native_to_intermediate(
        self, phi: Tensor, theta: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Native spherical (phi, theta) to intermediate (xi, eta), default fiducial."""
        xi, eta = phi, theta
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
            if self._is_tan or self._is_tpv or self._is_tnx:
                valid = (
                    torch.isfinite(theta)
                    & (theta > 64.0 * torch.finfo(theta.dtype).eps * self._r2d)
                    & (theta <= 90.0 + 1e-12)
                )
            elif self._is_arc:
                valid = torch.isfinite(theta) & (theta >= -90.0) & (theta <= 90.0)
            else:
                valid = torch.ones_like(theta, dtype=torch.bool)
            nan = torch.full_like(theta, float("nan"))
            xi = torch.where(valid, xi, nan)
            eta = torch.where(valid, eta, nan)
        elif self._is_sin:
            xi, eta = deproject_zenithal(phi, theta, "SIN", None, self._pv2_tensor)
        elif self._is_ait:
            xi, eta = deproject_allsky(phi, theta, "AIT")
        elif self._is_mol:
            xi, eta = deproject_allsky(phi, theta, "MOL")
        elif self._is_hpx:
            phi_w = _wrap_lon180(phi)
            s_theta = torch.sin(theta * self._d2r)
            abs_s = torch.abs(s_theta)
            mask_eq = abs_s <= self._hpx_transition_sin
            sigma = torch.sqrt(torch.clamp(self._hpx_k * (1.0 - abs_s), min=0.0))
            eta_pol = torch.sign(s_theta) * (
                self._hpx_eta_pole - self._hpx_facet_half_width * sigma
            )
            facet_offset = torch.full_like(phi_w, self._hpx_facet_half_width)
            if self._hpx_k % 2.0 == 0.0:
                facet_offset = torch.where(
                    s_theta < 0.0, torch.zeros_like(phi_w), facet_offset
                )
            xc = (
                torch.floor(
                    (phi_w + 180.0 - facet_offset + self._hpx_facet_half_width)
                    / self._hpx_facet_spacing
                )
                * self._hpx_facet_spacing
                - 180.0
                + facet_offset
            )
            xi = torch.where(mask_eq, phi_w, xc + sigma * (phi_w - xc))
            eta = torch.where(mask_eq, self._hpx_eta_scale * s_theta, eta_pol)
        elif (
            self._is_cea or self._is_mer or self._is_cyp or self._is_car or self._is_sfl
        ):
            xi, eta = self._cylindrical_native_to_intermediate(phi, theta)
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
            xi, eta = deproject_zpx(phi, theta, self._projection_params, self.wat_data)
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
        return xi, eta

    def _world_to_pixel_2d_core(
        self,
        ra: Tensor,
        dec: Tensor,
        origin: float,
        *,
        allow_cpp_inverse_sphere: bool = True,
        intermediate_only: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Unified 2D spatial world-to-pixel core.
        Designed to be graph-break free for torch.compile.
        """
        # Center-equator cylindrical projections skip spherical rotation.
        # SIP and lookup distortions are applied by the shared pixel tail.
        if self._is_center_cylindrical_shortcut():
            xi, eta = self._center_equator_deproject(ra, dec)
            if intermediate_only:
                return xi, eta
            return self._pixels_from_intermediate(xi, eta, origin)

        # 1. Inverse Spherical Rotation (Celestial -> Native)
        if self._is_gls:
            xi, eta = self._gls_inverse(ra, dec)
        elif (
            self._is_tan
            and not self._is_tpv
            and not self._is_tnx
            and not self._is_zpx
            and self._celestial_postrot is None
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
            xi, eta = self._native_to_intermediate(phi, theta)

        # 3. Inverse TPV / Special distortions
        if self._is_tpv:
            lon_scale = self.cdelt[self._celestial_lon_axis]
            lat_scale = self.cdelt[self._celestial_lat_axis]
            if self._tpv_uses_pixel_intermediate:
                xi, eta = xi / lon_scale, eta / lat_scale
            if self._require_tpv()._affine_only:
                xi, eta = self._require_tpv().invert_affine_exact(xi, eta)
            else:
                xi, eta = self._require_tpv().invert(xi, eta)
            if self._tpv_uses_pixel_intermediate:
                xi, eta = xi * lon_scale, eta * lat_scale
        if self._is_tnx:
            xi, eta = invert_tnx_plane(xi, eta, self.wat_data)

        if intermediate_only:
            return xi, eta
        return self._pixels_from_intermediate(xi, eta, origin)

    def _stacked_world_to_pixel_flat(
        self,
        world_flat: Tensor,
        origin: float,
        *,
        allow_cpp_inverse_sphere: bool = True,
    ) -> Tensor:
        """Fused ``world -> (px, py)`` for stacked 2D spatial + ``torch.compile``."""
        ra = world_flat[:, self._celestial_lon_axis]
        dec = world_flat[:, self._celestial_lat_axis]
        px, py = self._world_to_pixel_2d_core(
            ra, dec, origin, allow_cpp_inverse_sphere=allow_cpp_inverse_sphere
        )
        return torch.stack((px, py), dim=-1)
