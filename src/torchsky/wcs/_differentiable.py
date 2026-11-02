"""Differentiable paths, Jacobians, and astropy-shaped value APIs for :class:`torchsky.wcs.WCS`."""

from __future__ import annotations

import torch
from torch import Tensor
from typing import (
    TYPE_CHECKING,
    Any,
)
import numpy as np


if TYPE_CHECKING:
    pass


class _DifferentiableMixin:
    if TYPE_CHECKING:
        # State and methods supplied by the host class.
        def __getattr__(self, name: str) -> Any: ...

    def _reference_pixel(self, origin: float) -> Tensor:
        """CRPIX of the celestial pair in the caller's pixel convention (float64)."""
        return (self.crpix.detach().to(torch.float64).cpu() - (1.0 - origin))[:2]

    def _reference_jacobian_pixel_to_world(self, origin: float) -> Tensor:
        """Exact ``d(world_lon, world_lat)/d(pixel_x, pixel_y)`` at CRPIX.

        Returns a float64 CPU ``(2, 2)`` matrix in ``(lon, lat) x (x, y)`` order.

        * Zenithal projections (except TAN) have their native pole at CRPIX, where
          ``phi`` is undefined. Approaching the pole along a fixed native longitude
          the maps are smooth in ``t = 90 - theta``; two directions ``phi = 0, 90``
          give ``K = d(xi, eta)/dt`` (the projection's own ``R'(theta_0)`` and
          obliquity terms) and ``W = d(lon, lat)/dt`` (the celestial rotation), so
          ``J = W K^-1 CD`` exactly, to autograd precision.
        * PCO has native fiducial point (0, 0) where ``cot(theta)`` is singular; its
          native Jacobian there is the identity (the limit of the closed form).
        * XPH and QSC are piecewise smooth with a kink at CRPIX (the derivative is
          direction dependent and undefined there). The defined subgradient is the
          autograd Jacobian of the branch entered along ``(+1, +0.371)`` pixels,
          evaluated ``_KINK_STEP`` (1e-4) pixels from CRPIX, i.e. the branch derivative
          to ~1e-4 relative (QSC's forward map is noisy closer in);
          the inverse direction uses its matrix inverse, so ``J_w2p J_p2w = I``.
        * All other families evaluate the ordinary autograd Jacobian, which is exact
          at CRPIX.
        """
        cd = torch.tensor(
            [[self._cd00, self._cd01], [self._cd10, self._cd11]], dtype=torch.float64
        )
        code = self._proj_code
        dev = self.device
        dt = self._wcs_coord_dtype(dev)
        special_zen = (
            code in self.ZENITHAL_CODES
            and code != "TAN"
            and not (self._is_tpv or self._is_tnx or self._is_zpx)
        )
        if special_zen:
            phi = torch.tensor([0.0, 90.0], dtype=dt, device=dev)
            t = torch.zeros(2, dtype=dt, device=dev, requires_grad=True)
            theta = 90.0 - t
            xi, eta = self._native_to_intermediate(phi, theta)
            lon, lat = self._spherical_rotation_optimized(
                phi, theta, "pole", allow_cpp_sphere=False
            )
            kmat = torch.stack([self._row_grad(v, t) for v in (xi, eta)])
            wmat = torch.stack([self._row_grad(v, t) for v in (lon, lat)])
            return wmat @ torch.linalg.inv(kmat) @ cd
        if code == "PCO":
            nat = torch.zeros(2, dtype=dt, device=dev, requires_grad=True)
            lon, lat = self._spherical_rotation_optimized(
                nat[0:1], nat[1:2], self._native_type, allow_cpp_sphere=False
            )
            wmat = (
                torch.stack(
                    [
                        torch.autograd.grad(v.sum(), nat, retain_graph=True)[0]
                        for v in (lon, lat)
                    ]
                )
                .to(torch.float64)
                .cpu()
            )
            return wmat @ cd
        pix = self._reference_pixel(origin).to(dtype=dt, device=dev)
        if code in self._KINKED_CODES:
            pix = pix + self._KINK_STEP * torch.tensor(
                [1.0, 0.371], dtype=dt, device=dev
            )
        pix = pix.detach().clone().requires_grad_(True)
        world = self._stacked_pixels_to_world_flat(
            pix[None], origin, allow_cpp_sphere=False
        )
        lonlat = (
            world[0, self._celestial_lon_axis],
            world[0, self._celestial_lat_axis],
        )
        return (
            torch.stack(
                [torch.autograd.grad(v, pix, retain_graph=True)[0] for v in lonlat]
            )
            .to(torch.float64)
            .cpu()
        )

    @staticmethod
    def _row_grad(v: Tensor, t: Tensor) -> Tensor:
        """Per-row ``dv_i/dt_i`` when row ``i`` depends only on ``t_i``."""
        return torch.autograd.grad(v.sum(), t, retain_graph=True)[0]

    def _grad_safe_pixels_to_world(self, pixels_flat: Tensor, origin: float) -> Tensor:
        def fn(p):
            return self._stacked_pixels_to_world_flat(p, origin, allow_cpp_sphere=False)

        if not (torch.is_grad_enabled() and pixels_flat.requires_grad):
            return fn(pixels_flat)
        rel = (
            pixels_flat.detach()[:, :2]
            + (1.0 - origin)
            - self.crpix[:2].to(device=pixels_flat.device, dtype=pixels_flat.dtype)
        )
        at_ref = rel.abs().amax(dim=1) < self._REF_TOL
        if not bool(at_ref.any()):
            return fn(pixels_flat)
        code = self._proj_code
        needs_exact = code in ("PCO", *self._KINKED_CODES) or (
            code in self.ZENITHAL_CODES
            and code != "TAN"
            and not (self._is_tpv or self._is_tnx or self._is_zpx)
        )
        if not needs_exact:
            return fn(pixels_flat)
        jac2 = self._reference_jacobian_pixel_to_world(origin)
        jac = torch.zeros((self.naxis, 2), dtype=torch.float64)
        jac[self._celestial_lon_axis] = jac2[0]
        jac[self._celestial_lat_axis] = jac2[1]
        return self._first_order_rows(fn, pixels_flat, at_ref, jac, (0, 1))

    def _grad_safe_world_to_pixel(self, world_flat: Tensor, origin: float) -> Tensor:
        def fn(w):
            return self._stacked_world_to_pixel_flat(
                w, origin, allow_cpp_inverse_sphere=False
            )

        if not (torch.is_grad_enabled() and world_flat.requires_grad):
            return fn(world_flat)
        lon, lat = self._celestial_lon_axis, self._celestial_lat_axis
        with torch.no_grad():
            pix_ref = self._reference_pixel(origin).to(
                dtype=world_flat.dtype, device=world_flat.device
            )
            world_ref = self._stacked_pixels_to_world_flat(
                pix_ref[None], origin, allow_cpp_sphere=False
            )[0]
        w = world_flat.detach()
        dlon = torch.remainder(w[:, lon] - world_ref[lon] + 180.0, 360.0) - 180.0
        at_ref = (dlon.abs() < self._REF_TOL) & (
            (w[:, lat] - world_ref[lat]).abs() < self._REF_TOL
        )
        if not bool(at_ref.any()):
            return fn(world_flat)
        jac2 = self._reference_jacobian_pixel_to_world(origin)
        jac_w2p = torch.linalg.inv(jac2)
        return self._first_order_rows(fn, world_flat, at_ref, jac_w2p, (lon, lat))

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
            world_flat = self._grad_safe_pixels_to_world(pixels_flat, float(origin))
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
            pix_flat = self._grad_safe_world_to_pixel(world_flat, float(origin))
        elif self._is_spatial:
            pix_result = self.world_to_pixel(world_flat, origin=origin)
            assert isinstance(pix_result, Tensor)
            pix_flat = pix_result
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

    def jacobian(
        self,
        coords: Any,
        y: Any | None = None,
        *,
        inverse: bool = False,
        origin: int = 0,
        create_graph: bool = False,
    ) -> Tensor:
        """Return local ``d(world)/d(pixel)`` (or ``d(pixel)/d(world)`` when ``inverse=True``) Jacobians.

        Accepts either a stacked coordinate tensor ``(..., naxis)`` or separate
        ``(x, y)`` tensors for 2D WCS.
        """
        if y is not None:
            dt = self._wcs_coord_dtype(self.device)
            x_t = torch.as_tensor(coords, dtype=dt, device=self.device)
            y_t = torch.as_tensor(y, dtype=dt, device=self.device)
            x_t, y_t = torch.broadcast_tensors(x_t, y_t)
            stacked = torch.stack([x_t, y_t], dim=-1)
        else:
            stacked = coords
        if inverse:
            return self.world_to_pixel_jacobian(
                stacked, origin=origin, create_graph=create_graph
            )
        return self.pixel_to_world_jacobian(
            stacked, origin=origin, create_graph=create_graph
        )

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

    def world_to_array_index_values(
        self, *world_arrays: Any, origin: int = 0
    ) -> tuple[Tensor, ...]:
        """Convert world coordinates to nearest integer NumPy-order indices."""
        pixels = self.world_to_pixel_values(*world_arrays, origin=origin)
        return tuple(
            torch.round(pixels[self.pixel_n_dim - 1 - i]).to(dtype=torch.long)
            for i in range(self.pixel_n_dim)
        )
