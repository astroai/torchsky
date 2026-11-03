"""Tensor and projection setup for :class:`torchsky.wcs.WCS`."""

from __future__ import annotations

import torch
from torch import Tensor
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Tuple,
)
import math

from torchsky import _ops
from . import _celprm

from .tpv import PV1_KEYS, PV2_KEYS
from .zenithal import project_zenithal
from .cylindrical import project_cylindrical
from .allsky import project_allsky
from .conic import project_conic
from .quadcube import project_quadcube

if TYPE_CHECKING:
    pass


class _SetupTensorsMixin:
    if TYPE_CHECKING:
        # State and methods supplied by the host class.
        def __getattr__(self, name: str) -> Any: ...

    def _setup_tensors(self):
        """Convert scalar params to PyTorch tensors for computation."""
        unit_factors = torch.tensor(
            [self._celestial_unit_factor(i) for i in range(self.naxis)],
            dtype=torch.float64,
            device=self.device,
        )
        self._celestial_unit_factors = unit_factors
        self.crpix = torch.tensor(
            [self.wcs_params.get(f"CRPIX{i}", 0.0) for i in range(1, self.naxis + 1)],
            dtype=torch.float64,
            device=self.device,
        )

        self.crval = (
            torch.tensor(
                [
                    self.wcs_params.get(f"CRVAL{i}", 0.0)
                    for i in range(1, self.naxis + 1)
                ],
                dtype=torch.float64,
                device=self.device,
            )
            * unit_factors
        )

        self.cdelt = (
            torch.tensor(
                [
                    self.wcs_params.get(f"CDELT{i}", 1.0)
                    for i in range(1, self.naxis + 1)
                ],
                dtype=torch.float64,
                device=self.device,
            )
            * unit_factors
        )

        # 1b. CD matrix initialization (Paper I)
        # Check if CD keywords exist
        has_cd = any(
            f"CD{i}_{j}" in self.wcs_params
            for i in range(1, self.naxis + 1)
            for j in range(1, self.naxis + 1)
        )
        self._has_cd_matrix = has_cd

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
            self.cd_full = self.cd_full * unit_factors[:, None]
        else:
            # Construct from PC and CDELT: CDi_j = CDELTi * PCi_j
            cd_data = []
            pc_data = []
            crota_pc = self._crota_pc_overrides()
            for i in range(1, self.naxis + 1):
                row = []
                pc_row = []
                for j in range(1, self.naxis + 1):
                    pc_val = self.wcs_params.get(f"PC{i}_{j}", 1.0 if i == j else 0.0)
                    if (i, j) in crota_pc:
                        pc_val = crota_pc[(i, j)]
                    row.append(self.cdelt[i - 1].item() * pc_val)
                    pc_row.append(pc_val)
                cd_data.append(row)
                pc_data.append(pc_row)
            self.cd_full = torch.tensor(
                cd_data, dtype=self._wcs_coord_dtype(self.device), device=self.device
            )
            self.pc_full = torch.tensor(
                pc_data, dtype=self._wcs_coord_dtype(self.device), device=self.device
            )

        if has_cd:
            self.pc_full = self.cd_full / self.cdelt[:, None]

        # Canonical projection-plane rows are always longitude, latitude while
        # pixel columns retain FITS axis order.
        if self.naxis >= 2 and self._has_celestial:
            celestial_rows = torch.as_tensor(
                [self._celestial_lon_axis, self._celestial_lat_axis],
                dtype=torch.long,
                device=self.device,
            )
            self.cd = self.cd_full.index_select(0, celestial_rows)[:, :2]
        elif self.naxis >= 2:
            self.cd = self.cd_full[:2, :2]
        else:
            # 1x1 padded to 2x2 for safety in spatial paths
            self.cd = torch.eye(
                2, dtype=self._wcs_coord_dtype(self.device), device=self.device
            )
            self.cd[0, 0] = self.cd_full[0, 0]

        # Precompute inverse CD matrix for world->pixel
        self.cd_inv_full = torch.inverse(self.cd_full)
        if self.naxis == 2:
            self.cd_inv = torch.inverse(self.cd)
        else:
            self.cd_inv = torch.linalg.pinv(self.cd)
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
        self.alpha0 = float(self.crval[self._celestial_lon_axis].item())
        self.delta0 = (
            float(self.crval[self._celestial_lat_axis].item())
            if self.naxis >= 2
            else 0.0
        )

        # Precompute Projection Meta
        projection_axis = self._celestial_lon_axis + 1 if self._has_celestial else 1
        ctype1 = (
            str(self.wcs_params.get(f"CTYPE{projection_axis}", "RA---TAN"))
            .strip()
            .upper()
        )
        ctype1_base = ctype1[:-4] if ctype1.endswith("-SIP") else ctype1
        self._proj_code = ctype1_base[-3:]
        self._is_tnx = self._proj_code == "TNX"
        self._is_zpx = self._proj_code == "ZPX"
        self._is_tpv = self.tpv is not None or self._proj_code == "TPV"
        self._tpv_uses_pixel_intermediate = self._is_tpv and not self._has_cd_matrix
        self._load_pixel_affine_from_params()

        self._theta0 = 90.0  # Zenithal
        if self._proj_code == "XPH":
            # HEALPix polar / butterfly: native pole at reference (Paper XPH).
            self._theta0 = 90.0
        elif (
            self._proj_code
            in (
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
            )
            or self._proj_code in self.QUADCUBE_CODES
        ):
            self._theta0 = 0.0
        elif self._proj_code in self.CONIC_CODES or self._proj_code == "BON":
            # Conic and Bonne use PV2_1 (standard parallel) as theta_0.
            self._theta0 = float(self._projection_params.get("PV2_1", 0.0))
            # Bonne with theta_1=0 degenerates to Sanson-Flamsteed (equatorial)
            if self._proj_code == "BON" and abs(self._theta0) < 1e-12:
                self._theta0 = 0.0

        # LONPOLE / LATPOLE
        # Paper II: LONPOLE defaults to 0 if delta_0 >= theta_0, else 180.
        # PV{lon}_3 / PV{lon}_4 are LONPOLE / LATPOLE; PV{lon}_1 / PV{lon}_2 are the
        # native fiducial point (phi_0, theta_0) handled below.
        pj = self._projection_params
        # Bonne's native fiducial point is (0, 0): its y-offset puts the origin there.
        cel_theta0 = 0.0 if self._proj_code == "BON" else self._theta0
        # Native fiducial point (phi_0, theta_0) = (PV1_1, PV1_2) (Paper II 2.5,
        # WCSLIB celset): it is the native point mapped to CRVAL and fixes the
        # Euler angles / default LONPOLE. Like WCSLIB with ``cel.offset = 0`` the
        # projection origin (x, y) = (0, 0) is not shifted onto it.
        self._fiducial_override = False
        phi0_cel = 0.0
        theta0_cel = cel_theta0
        if self._has_celestial and not (
            self._is_tpv or self._proj_code in ("HPX", "SFL", "GLS", "TPV")
        ):
            if "PV1_1" in pj:
                phi0_cel = float(pj["PV1_1"])
            if "PV1_2" in pj:
                theta0_cel = float(pj["PV1_2"])
            self._fiducial_override = phi0_cel != 0.0 or theta0_cel != cel_theta0
        # Under TPV the longitude-axis PV1_m are distortion coefficients, not
        # Paper II (phi_0, theta_0, LONPOLE, LATPOLE); DECam writes PV1_3 = 0.
        pv_poles = not (self._is_tpv or self._proj_code == "TPV")
        if "LONPOLE" in self.wcs_params:
            lonpole = float(self.wcs_params["LONPOLE"])
        elif pv_poles and "PV1_3" in pj:
            lonpole = float(pj["PV1_3"])
        else:
            lonpole = _celprm.default_lonpole(self.delta0, theta0_cel, phi0_cel)
        if "LATPOLE" in self.wcs_params:
            latpole = float(self.wcs_params["LATPOLE"])
        elif pv_poles and "PV1_4" in pj:
            latpole = float(pj["PV1_4"])
        else:
            latpole = 90.0
        self.phi_p = lonpole
        self.theta_p = latpole
        # The projection/rotation kernels bake in LONPOLE=180 (zenithal) or the
        # Paper II default (others) with LATPOLE=90; ``_celestial_postrot`` is the
        # exact residual rotation, applied on every path (None when identity).
        # Kernel-internal phi_p (selects the pole basis conventions below).
        self._phi_p_native = 180.0 if self._theta0 == 90.0 else 0.0
        exist_lonpole = (
            180.0
            if self._theta0 == 90.0
            else _celprm.default_lonpole(self.delta0, cel_theta0)
        )
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
            self.alpha_p = (
                self.alpha0 + 180.0 if self._phi_p_native == 0.0 else self.alpha0
            )
            self._native_type = "center"
        elif self._proj_code in ("PCO",) or self._proj_code in self.QUADCUBE_CODES:
            self.delta_p = 90.0
            self.alpha_p = (
                self.alpha0 + 180.0 if self._phi_p_native == 0.0 else self.alpha0
            )
            self._native_type = "center"
        else:
            self.delta_p = 90.0
            self.alpha_p = (
                self.alpha0 + 180.0 if self._phi_p_native == 0.0 else self.alpha0
            )
            self._native_type = "center"

        exist_matrix = None
        if self._native_type == "center":
            # Realized by _spherical_rotation(center): (cos t cos p, cos t sin p, sin t)
            # -> cos t cos p * r + cos t sin p * e + sin t * n, with (e, n, r) at CRVAL.
            exist_matrix = [
                [self._radial_x, self._east_x, self._north_x],
                [self._radial_y, self._east_y, self._north_y],
                [self._radial_z, 0.0, self._north_z],
            ]
        elif self._proj_code in self.CONIC_CODES:
            # Pole-type basis at (alpha_p, delta_p) with the conic kernels' negated phi:
            # (u, v, w) = (-n_y, n_x, n_z) for std native n.
            sa, ca = math.sin(self.alpha_p * d2r), math.cos(self.alpha_p * d2r)
            sd, cd = math.sin(self.delta_p * d2r), math.cos(self.delta_p * d2r)
            ex, ey, ez = -sa, ca, 0.0
            nx_, ny_, nz_ = -sd * ca, -sd * sa, cd
            rx_, ry_, rz_ = cd * ca, cd * sa, sd
            exist_matrix = [
                [nx_, -ex, rx_],
                [ny_, -ey, ry_],
                [nz_, -ez, rz_],
            ]
        self._celestial_postrot = (
            _celprm.celestial_correction(
                self.alpha0,
                self.delta0,
                theta0_cel,
                lonpole,
                latpole,
                exist_lonpole,
                exist_matrix,
                phi0=phi0_cel,
                exist_theta0=cel_theta0,
                force=self._fiducial_override,
            )
            if self._has_celestial
            else None
        )

        # Precompute Tensors
        self._ra_p_rad = self.alpha_p * d2r
        self._dec_p_rad = self.delta_p * d2r
        self._phi_p_rad = self._phi_p_native * d2r
        dec_p_tensor = torch.as_tensor(
            self._dec_p_rad,
            dtype=self._wcs_coord_dtype(self.device),
            device=self.device,
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
            and abs(self.crval[self._celestial_lon_axis].item()) < 1e-9
            and abs(self.crval[self._celestial_lat_axis].item()) < 1e-9
        )

        # Precompute CEA/MER constants
        pv2_1 = self._projection_params.get("PV2_1", 1.0)
        if self._proj_code == "CEA" and (
            not math.isfinite(float(pv2_1)) or float(pv2_1) == 0.0
        ):
            raise ValueError("CEA PV2_1 (lambda) must be finite and non-zero")
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
                self._projection_params.get(
                    "PV2_1", self._projection_params.get("PV1_1", 4.0)
                )
            )
            self._hpx_k = float(
                self._projection_params.get(
                    "PV2_2", self._projection_params.get("PV1_2", 3.0)
                )
            )
        else:
            self._hpx_h = 4.0
            self._hpx_k = 3.0
        if (
            not math.isfinite(self._hpx_h)
            or self._hpx_h <= 0.0
            or not self._hpx_h.is_integer()
        ):
            raise ValueError("HPX H must be a finite positive integer")
        if (
            not math.isfinite(self._hpx_k)
            or self._hpx_k <= 0.0
            or not self._hpx_k.is_integer()
        ):
            raise ValueError("HPX K must be a finite positive integer")
        self._hpx_transition_sin = (self._hpx_k - 1.0) / self._hpx_k
        self._hpx_eta_scale = 90.0 * (self._hpx_k / self._hpx_h)
        self._hpx_inv_eta_scale = 1.0 / self._hpx_eta_scale
        self._hpx_eta_boundary = 90.0 * (self._hpx_k - 1.0) / self._hpx_h
        self._hpx_eta_pole = 90.0 * (self._hpx_k + 1.0) / self._hpx_h
        self._hpx_facet_spacing = 360.0 / self._hpx_h
        self._hpx_facet_half_width = 0.5 * self._hpx_facet_spacing
        self._hpx_polar_denom = self._hpx_facet_half_width
        self._hpx_inv_polar_denom = 1.0 / self._hpx_polar_denom

        # Precompute fused arguments to avoid overhead
        self._fused_proj_code = "TAN" if self._is_tpv else self._proj_code
        self._fused_pv2_1 = self._projection_params.get("PV2_1", 1.0)
        self._fused_tpv_idx1 = (
            self._require_tpv().idx1
            if self._is_tpv
            else torch.empty((0, 3), dtype=torch.long, device="cpu")
        )
        self._fused_tpv_c1 = (
            self._require_tpv().c1
            if self._is_tpv
            else torch.empty(0, dtype=torch.float64, device="cpu")
        )
        self._fused_tpv_idx2 = (
            self._require_tpv().idx2
            if self._is_tpv
            else torch.empty((0, 3), dtype=torch.long, device="cpu")
        )
        self._fused_tpv_c2 = (
            self._require_tpv().c2
            if self._is_tpv
            else torch.empty(0, dtype=torch.float64, device="cpu")
        )

        # Apply projection-specific PV defaults before tensorization.
        # The pre-tensorized PV arrays fill missing keys with 0.0, but some
        # projections have non-zero defaults (e.g. SZP theta_c = 90°).
        if self._proj_code == "SZP" and "PV2_3" not in self._projection_params:
            self._projection_params["PV2_3"] = 90.0
        # Defaults fill absent keywords only; an explicit PV value of 0 is honoured.
        for _code, _defaults in (
            ("CEA", {"PV2_1": 1.0}),
            ("CYP", {"PV2_1": 1.0, "PV2_2": 1.0}),
            ("AIR", {"PV2_1": 90.0}),
        ):
            if self._proj_code == _code:
                for _key, _value in _defaults.items():
                    self._projection_params.setdefault(_key, _value)

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
            if "PV2_1" not in self._projection_params:
                self._projection_params["PV2_1"] = 0.0
            if "PV2_2" not in self._projection_params:
                self._projection_params["PV2_2"] = cos_d0 / sin_d0
            self._proj_code = "SIN"
            self._fused_proj_code = "SIN"

        # Pre-tensorize PV parameters to avoid dict lookups in compiled hot paths.
        pv1_data = [self._projection_params.get(k, 0.0) for k in PV1_KEYS]
        pv2_data = [self._projection_params.get(k, 0.0) for k in PV2_KEYS]
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
            and abs(float(self._projection_params.get("PV1_0", 0.0)) - 1.0) <= 1e-12
            and "PV1_2" in self._projection_params
        )
        self._gls_delta0 = (
            float(self._projection_params["PV1_2"])
            if (self._proj_code == "SFL" and "PV1_2" in self._projection_params)
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
            and self._celestial_postrot is None
            and self.alpha0 == self.alpha_p
            and self.delta0 == self.delta_p
        )
        self._is_spatial = self._has_celestial and self._proj_code in (
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
            and abs((self._phi_p_native % 360.0)) < 1e-12
            and self._celestial_postrot is None
        )
        self._cpp_zenithal_project: Callable[..., Any] | None = None
        self._cpp_zpn_project: Callable[..., Any] | None = None
        self._cpp_zenithal_deproject: Callable[..., Any] | None = None
        self._cpp_cylindrical_project: Callable[..., Any] | None = None
        self._cpp_cylindrical_deproject: Callable[..., Any] | None = None
        self._cpp_center_pixel_to_world: Callable[..., Any] | None = None
        self._cpp_center_world_to_pixel: Callable[..., Any] | None = None
        self._cpp_ait_project: Callable[..., Any] | None = None
        self._cpp_ait_deproject: Callable[..., Any] | None = None
        self._cpp_mol_project: Callable[..., Any] | None = None
        self._cpp_mol_deproject: Callable[..., Any] | None = None
        self._cpp_hpx_project: Callable[..., Any] | None = None
        self._cpp_hpx_deproject: Callable[..., Any] | None = None
        self._cpp_inverse_spherical_rotation_pole: Callable[..., Any] | None = None
        self._cpp_spherical_rotation_pole: Callable[..., Any] | None = None
        self._cpp_tan_intermediate_from_radec: Callable[..., Any] | None = None
        self._cpp_direct_static_ok = (
            self._celestial_postrot is None
            and self.sip is None
            and not self._tpv_uses_pixel_intermediate
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
            self._celestial_postrot is None
            and self._enable_fused_wcs
            and self._enable_cpp_dispatcher
            and _ops.has_wcs_dispatcher()
            and self.sip is None
            and not self._tpv_uses_pixel_intermediate
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
            self._celestial_postrot is None
            and self._enable_fused_wcs
            and self._enable_cpp_dispatcher
            and _ops.has_wcs_dispatcher()
            and self.sip is None
            and not self._tpv_uses_pixel_intermediate
            and not self._is_tnx
            and not self._is_zpx
            and not self._is_gls
            and not self._has_paper4_lut()
            and not self._is_sin_slant
            and (not self._is_tpv or not self._require_tpv()._invert_trace_enabled)
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
            self._project_native_fn: Callable[..., Tuple[Tensor, Tensor]] | None = None
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
