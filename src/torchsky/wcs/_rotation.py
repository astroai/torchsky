"""Spherical rotation and TAN projection mixin for :class:`torchsky.wcs.WCS`.

The methods here are pure functions of the WCS state (CRVAL/pole basis and the
native-type flag) and are shared by every projection path.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

from ._math import _sincos, _wrap_lon180, _wrap_lon360


class _RotationMixin:
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
        crval1_rad = torch.as_tensor(self.alpha0 * rad, dtype=xi.dtype, device=xi.device)
        crval2_rad = torch.as_tensor(self.delta0 * rad, dtype=xi.dtype, device=xi.device)

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
    def _spherical_rotation_optimized_base(
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

        phi_rad = (phi - self._phi_p_native) * d2r
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
    def _inverse_spherical_rotation_base(
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
                phi = _wrap_lon180(ra - self.alpha0)
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
            # Center-native rotation, inverted with the same basis as
            # _spherical_rotation: celestial = u*e + v*(-r) + w*n, with
            # (e, n, r) the east/north/radial basis at (alpha0, delta0).
            # The stored _pole_* basis belongs to pole-native projections and
            # is wrong here whenever delta0 != 0 (it would map the reference
            # point to native latitude 90 - |delta0| instead of 0).
            u = x_cel * self._east_x + y_cel * self._east_y
            w = (
                x_cel * self._north_x
                + y_cel * self._north_y
                + z_cel * self._north_z
            )
            v = -(
                x_cel * self._radial_x
                + y_cel * self._radial_y
                + z_cel * self._radial_z
            )
            theta = torch.atan2(w, torch.hypot(u, v)) * r2d
            phi = torch.remainder(torch.atan2(u, -v) * r2d, 360.0)
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

        theta = torch.atan2(w, torch.hypot(u, v)) * r2d
        phi = torch.remainder(torch.atan2(u, v) * r2d + self._phi_p_native, 360.0)

        return phi, theta


    def _apply_celestial_postrot(
        self, ra: Tensor, dec: Tensor, *, inverse: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Apply the exact residual LONPOLE/LATPOLE rotation (Paper II) to (ra, dec)."""
        m = self._celestial_postrot
        d2r = 0.017453292519943295
        r2d = 57.29577951308232
        sin_dec, cos_dec = _sincos(dec * d2r)
        sin_ra, cos_ra = _sincos(ra * d2r)
        v = (cos_dec * cos_ra, cos_dec * sin_ra, sin_dec)
        if inverse:
            c = [[m[j][i] for j in range(3)] for i in range(3)]
        else:
            c = m
        x = c[0][0] * v[0] + c[0][1] * v[1] + c[0][2] * v[2]
        y = c[1][0] * v[0] + c[1][1] * v[1] + c[1][2] * v[2]
        z = c[2][0] * v[0] + c[2][1] * v[1] + c[2][2] * v[2]
        lon = torch.remainder(torch.atan2(y, x) * r2d, 360.0)
        lat = torch.atan2(z, torch.hypot(x, y)) * r2d
        return lon, lat

    def _spherical_rotation_optimized(
        self,
        phi: Tensor,
        theta: Tensor,
        native_type: str = "pole",
        *,
        allow_cpp_sphere: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Native -> celestial, honouring LONPOLE/LATPOLE exactly."""
        if self._celestial_postrot is None:
            return self._spherical_rotation_optimized_base(
                phi, theta, native_type, allow_cpp_sphere=allow_cpp_sphere
            )
        ra, dec = self._spherical_rotation_optimized_base(
            phi, theta, native_type, allow_cpp_sphere=allow_cpp_sphere
        )
        return self._apply_celestial_postrot(ra, dec)

    def _inverse_spherical_rotation(
        self,
        ra: Tensor,
        dec: Tensor,
        native_type: str = "pole",
        *,
        allow_cpp_inverse_sphere: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Celestial -> native, honouring LONPOLE/LATPOLE exactly."""
        if self._celestial_postrot is not None:
            ra, dec = self._apply_celestial_postrot(ra, dec, inverse=True)
        return self._inverse_spherical_rotation_base(
            ra,
            dec,
            native_type,
            allow_cpp_inverse_sphere=allow_cpp_inverse_sphere,
        )
