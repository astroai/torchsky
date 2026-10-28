"""Geometry, footprint, and serialization mixins for :class:`torchsky.wcs.WCS`."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor


class _GeometryMixin:
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
            celestial_deps = np.logical_or(
                cd_deps[self._celestial_lon_axis, :],
                cd_deps[self._celestial_lat_axis, :],
            )
            matrix[:, :] = cd_deps
            matrix[self._celestial_lon_axis, :] = celestial_deps
            matrix[self._celestial_lat_axis, :] = celestial_deps
        else:
            matrix[:, :] = cd_deps
        for axis in self.tabular_axes:
            if 0 <= axis < self.naxis:
                matrix[axis, axis] = True
        for axis in self.coupled_tabular_axes:
            matrix[axis, :] = False
            matrix[axis, list(self.coupled_tabular_pixel_axes)] = True
        return matrix
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
    @property
    def pixel_shape(self) -> tuple[int, ...] | None:
        """Image dimensions in FITS axis order ``(NAXIS1, NAXIS2, ...)``."""
        return self._pixel_shape
    @pixel_shape.setter
    def pixel_shape(self, value: Sequence[int] | None) -> None:
        self._pixel_shape = None if value is None else tuple(int(v) for v in value)
    @property
    def array_shape(self) -> tuple[int, ...] | None:
        """Image dimensions in NumPy/PyTorch array order ``(..., NAXIS2, NAXIS1)``."""
        if self._pixel_shape is None:
            return None
        return tuple(reversed(self._pixel_shape))
    def pixel_scales(self) -> tuple[float, float]:
        """Return nominal pixel scales ``(scale_x_deg, scale_y_deg)`` in degrees/pixel at CRPIX."""
        rows = [self._celestial_lon_axis, self._celestial_lat_axis]
        cols = list(self._celestial_axes)
        cd2 = self.cd_full[rows, :][:, cols].detach().cpu().to(torch.float64)
        sx = float(torch.hypot(cd2[0, 0], cd2[1, 0]).item())
        sy = float(torch.hypot(cd2[0, 1], cd2[1, 1]).item())
        return (sx, sy)
    def pixel_area(self) -> float:
        """Return nominal projection-plane pixel solid angle in ``deg**2`` at CRPIX."""
        rows = [self._celestial_lon_axis, self._celestial_lat_axis]
        cols = list(self._celestial_axes)
        cd2 = self.cd_full[rows, :][:, cols].detach().cpu().to(torch.float64)
        return abs(float((cd2[0, 0] * cd2[1, 1] - cd2[0, 1] * cd2[1, 0]).item()))
    def position_angle(self) -> float:
        """Return position angle (degrees East of North) of the +y pixel axis at CRPIX."""
        rows = [self._celestial_lon_axis, self._celestial_lat_axis]
        cols = list(self._celestial_axes)
        cd2 = self.cd_full[rows, :][:, cols].detach().cpu().to(torch.float64)
        det = float((cd2[0, 0] * cd2[1, 1] - cd2[0, 1] * cd2[1, 0]).item())
        east_sign = -1.0 if det < 0.0 else 1.0
        pa_rad = math.atan2(east_sign * float(cd2[0, 1].item()), float(cd2[1, 1].item()))
        return math.degrees(pa_rad)
    def footprint(
        self,
        shape: tuple[int, int] | None = None,
        *,
        center: bool = True,
        origin: int = 0,
    ) -> Tensor:
        """Return ``(4, 2)`` tensor of ``(lon_deg, lat_deg)`` corner sky coordinates.

        Parameters
        ----------
        shape
            Image shape in PyTorch ``(height, width) = (NAXIS2, NAXIS1)`` order.
            If omitted, inferred from ``self.array_shape``.
        center
            If ``True`` (default), evaluate at pixel centers
            ``(0, 0), (0, H-1), (W-1, H-1), (W-1, 0)``. If ``False``, evaluate at
            outer pixel corners ``(-0.5, -0.5) .. (W-0.5, H-0.5)``.
        origin
            Pixel coordinate origin (``0`` for 0-based, ``1`` for FITS 1-based).
        """
        if shape is not None:
            height, width = int(shape[0]), int(shape[1])
        elif self._pixel_shape is not None and len(self._pixel_shape) >= 2:
            pixel_axes = list(self._celestial_axes)
            width, height = (
                int(self._pixel_shape[pixel_axes[0]]),
                int(self._pixel_shape[pixel_axes[1]]),
            )
        else:
            raise ValueError(
                "shape=(height, width) must be provided when WCS has no NAXIS1/NAXIS2"
            )
        dt = self._wcs_coord_dtype(self.device)
        off = float(origin)
        if center:
            xs = torch.tensor(
                [off, off, off + float(width - 1), off + float(width - 1)],
                dtype=dt,
                device=self.device,
            )
            ys = torch.tensor(
                [off, off + float(height - 1), off + float(height - 1), off],
                dtype=dt,
                device=self.device,
            )
        else:
            xs = torch.tensor(
                [off - 0.5, off - 0.5, off + float(width) - 0.5, off + float(width) - 0.5],
                dtype=dt,
                device=self.device,
            )
            ys = torch.tensor(
                [off - 0.5, off + float(height) - 0.5, off + float(height) - 0.5, off - 0.5],
                dtype=dt,
                device=self.device,
            )
        if self.naxis == 2:
            ra, dec = self.pixel_to_world(xs, ys, origin=origin)
            world = torch.stack([ra, dec], dim=-1)
            return world[:, [self._celestial_lon_axis, self._celestial_lat_axis]]
        corners = (self.crpix - (1.0 - off)).expand(4, -1).clone()
        pixel_axes = list(self._celestial_axes)
        corners[:, pixel_axes[0]] = xs
        corners[:, pixel_axes[1]] = ys
        world = self.pixel_to_world(corners, origin=origin)
        return world[:, [self._celestial_lon_axis, self._celestial_lat_axis]]
    def pixel_area_map(
        self,
        shape: tuple[int, int] | None = None,
        *,
        unit: str = "deg2",
    ) -> Tensor:
        """Compute exact 2D per-pixel spherical solid-angle map ``Omega(y, x)``.

        Parameters
        ----------
        shape
            Image shape ``(height, width)``. Inferred from ``self.array_shape`` if omitted.
        unit
            One of ``"deg2"`` (default), ``"arcsec2"``, or ``"sr"``.
        """
        if shape is not None:
            height, width = int(shape[0]), int(shape[1])
        elif self._pixel_shape is not None and len(self._pixel_shape) >= 2:
            width, height = int(self._pixel_shape[0]), int(self._pixel_shape[1])
        else:
            raise ValueError(
                "shape=(height, width) must be provided when WCS has no NAXIS1/NAXIS2"
            )
        dt = self._wcs_coord_dtype(self.device)
        y_1d = torch.arange(height, dtype=dt, device=self.device)
        x_1d = torch.arange(width, dtype=dt, device=self.device)
        yy, xx = torch.meshgrid(y_1d, x_1d, indexing="ij")
        ra_xp, dec_xp = self.pixel_to_world(xx + 0.5, yy)
        ra_xm, dec_xm = self.pixel_to_world(xx - 0.5, yy)
        ra_yp, dec_yp = self.pixel_to_world(xx, yy + 0.5)
        ra_ym, dec_ym = self.pixel_to_world(xx, yy - 0.5)
        _, dec_c = self.pixel_to_world(xx, yy)
        cos_dec = torch.cos(torch.deg2rad(dec_c)).clamp_min(1e-12)

        dra_dx = (torch.remainder(ra_xp - ra_xm + 180.0, 360.0) - 180.0) * cos_dec
        ddec_dx = dec_xp - dec_xm
        dra_dy = (torch.remainder(ra_yp - ra_ym + 180.0, 360.0) - 180.0) * cos_dec
        ddec_dy = dec_yp - dec_ym
        area_deg2 = torch.abs(dra_dx * ddec_dy - ddec_dx * dra_dy)
        unit_norm = unit.strip().lower()
        if unit_norm in {"deg2", "deg^2", "deg**2"}:
            return area_deg2
        if unit_norm in {"arcsec2", "arcsec^2", "arcsec**2"}:
            return area_deg2 * (3600.0 * 3600.0)
        if unit_norm in {"sr", "steradian", "steradians"}:
            return area_deg2 * ((math.pi / 180.0) ** 2)
        raise ValueError("unit must be one of 'deg2', 'arcsec2', or 'sr'")


class _HeaderIOMixin:
    def to_header(
        self,
        *,
        relax: bool = True,
        include_naxis: bool = False,
    ) -> dict[str, Any]:
        """Serialize this WCS into a standard FITS header dictionary.

        Parameters
        ----------
        relax
            If ``True`` (default), include SIP polynomial distortion keywords
            (``A_ORDER``, ``A_p_q``, ``B_ORDER``, ``B_p_q``, ``AP_*``, ``BP_*``)
            and TPV ``PVi_m`` keywords when present.
        include_naxis
            If ``True`` and ``pixel_shape`` is known, include ``NAXIS`` and
            ``NAXIS1..NAXISn`` cards.
        """
        hdr: dict[str, Any] = {}
        sfx = self.wcs_key
        if include_naxis and self._pixel_shape is not None:
            hdr["NAXIS"] = len(self._pixel_shape)
            for idx, dim in enumerate(self._pixel_shape, start=1):
                hdr[f"NAXIS{idx}"] = int(dim)
        hdr[f"WCSAXES{sfx}"] = int(self.naxis)

        crpix_cpu = self.crpix.detach().cpu().to(torch.float64).tolist()
        crval_cpu = self.crval.detach().cpu().to(torch.float64).tolist()
        cd_cpu = self.cd_full.detach().cpu().to(torch.float64).tolist()

        for i in range(1, self.naxis + 1):
            ctype = str(self.wcs_params.get(f"CTYPE{i}", "")).strip()
            if (
                relax
                and self.sip is not None
                and (i - 1) in self._celestial_axes
                and ctype
                and not ctype.upper().endswith("-SIP")
            ):
                ctype = f"{ctype}-SIP"
            if ctype:
                hdr[f"CTYPE{i}{sfx}"] = ctype
            hdr[f"CRPIX{i}{sfx}"] = float(crpix_cpu[i - 1])
            hdr[f"CRVAL{i}{sfx}"] = float(crval_cpu[i - 1])
            cunit = self.wcs_params.get(f"CUNIT{i}")
            if cunit:
                hdr[f"CUNIT{i}{sfx}"] = str(cunit)

        if self._is_tpv and self._tpv_uses_pixel_intermediate:
            pc_cpu = self.pc_full.detach().cpu().to(torch.float64).tolist()
            cdelt_cpu = self.cdelt.detach().cpu().to(torch.float64).tolist()
            for i in range(1, self.naxis + 1):
                hdr[f"CDELT{i}{sfx}"] = float(cdelt_cpu[i - 1])
                for j in range(1, self.naxis + 1):
                    val = float(pc_cpu[i - 1][j - 1])
                    if i == j or abs(val) > 0.0:
                        hdr[f"PC{i}_{j}{sfx}"] = val
        else:
            for i in range(1, self.naxis + 1):
                for j in range(1, self.naxis + 1):
                    val = float(cd_cpu[i - 1][j - 1])
                    if (
                        i == j
                        or abs(val) > 0.0
                        or (
                            (i - 1) in self._celestial_axes
                            and (j - 1) in self._celestial_axes
                        )
                    ):
                        hdr[f"CD{i}_{j}{sfx}"] = val

        for meta_key in (
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
            "MJD-OBS",
            "MJDREF",
            "MJDREFI",
            "MJDREFF",
            "DATE-OBS",
            "TIMESYS",
            "TIMEUNIT",
            "TIMEOFFS",
        ):
            if meta_key in self.wcs_params:
                hdr[f"{meta_key}{sfx}" if meta_key in {"LONPOLE", "LATPOLE", "RADESYS", "EQUINOX", "RESTFRQ", "RESTWAV", "SPECSYS"} else meta_key] = self.wcs_params[meta_key]

        for k, v in self.wcs_params.items():
            if k.startswith(("PV", "PS", "PROJP", "WAT")):
                if not relax and self._is_tpv and k.startswith("PV"):
                    continue
                hdr[f"{k}{sfx}" if not sfx or not k.endswith(sfx) else k] = v

        if relax and self.sip is not None:
            if self.sip.a_order > 0 or self.sip.a_coeffs:
                hdr["A_ORDER"] = int(self.sip.a_order)
                for (p, q), coeff in self.sip.a_coeffs.items():
                    hdr[f"A_{p}_{q}"] = float(coeff)
            if self.sip.b_order > 0 or self.sip.b_coeffs:
                hdr["B_ORDER"] = int(self.sip.b_order)
                for (p, q), coeff in self.sip.b_coeffs.items():
                    hdr[f"B_{p}_{q}"] = float(coeff)
            if self.sip.has_inverse:
                if self.sip.ap_order > 0 or self.sip.ap_coeffs:
                    hdr["AP_ORDER"] = int(self.sip.ap_order)
                    for (p, q), coeff in self.sip.ap_coeffs.items():
                        hdr[f"AP_{p}_{q}"] = float(coeff)
                if self.sip.bp_order > 0 or self.sip.bp_coeffs:
                    hdr["BP_ORDER"] = int(self.sip.bp_order)
                    for (p, q), coeff in self.sip.bp_coeffs.items():
                        hdr[f"BP_{p}_{q}"] = float(coeff)

        if self._pixel_affine_matrix is not None and self._pixel_affine_offset is not None:
            mat = self._pixel_affine_matrix.detach().cpu().to(torch.float64)
            off = self._pixel_affine_offset.detach().cpu().to(torch.float64)
            hdr["TSKYAF0"] = "pixel_affine"
            hdr["TSKYAFA11"] = float(mat[0, 0].item())
            hdr["TSKYAFA12"] = float(mat[0, 1].item())
            hdr["TSKYAFA21"] = float(mat[1, 0].item())
            hdr["TSKYAFA22"] = float(mat[1, 1].item())
            hdr["TSKYAFB1"] = float(off[0].item())
            hdr["TSKYAFB2"] = float(off[1].item())
        elif self.wcs_params.get("TSKYAF0") == "pixel_affine":
            # Undistorted corrections bake into CRPIX/CD and keep no live
            # pixel-space affine, but the provenance marker must survive a
            # header round-trip.
            for key, value in self.wcs_params.items():
                if str(key).startswith("TSKYAF"):
                    hdr[str(key)] = value

        return hdr
    def to_head_string(self, *, relax: bool = True) -> str:
        """Format this WCS as a SCAMP/SWarp-compatible ``.head`` ASCII string."""
        hdr = self.to_header(relax=relax, include_naxis=False)
        lines: list[str] = []
        for k, v in hdr.items():
            key_str = f"{str(k)[:8]:<8s}"
            if isinstance(v, bool):
                val_str = f"{'T' if v else 'F':>20s}"
            elif isinstance(v, int):
                val_str = f"{v:>20d}"
            elif isinstance(v, float):
                val_str = f"{v:>20.12G}"
            else:
                s_escaped = str(v).replace("'", "''")
                val_str = f"'{s_escaped:<8s}'"
            card = f"{key_str}= {val_str}"
            lines.append(f"{card:<80s}"[:80])
        lines.append(f"{'END':<80s}")
        return "\n".join(lines) + "\n"
