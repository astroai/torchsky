"""Mixed pixel/world solving for :class:`torchsky.wcs.WCS`."""

from __future__ import annotations

import torch
from torch import Tensor
from typing import (
    TYPE_CHECKING,
    Any,
)

from ._math import _wrap_lon180

from .tabular import (
    _solve_coupled_tabular_partial,
)
from ..device import _as_tensor, _to

if TYPE_CHECKING:
    pass


class _MixedMixin:
    if TYPE_CHECKING:
        # State and methods supplied by the host class.
        def __getattr__(self, name: str) -> Any: ...

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
            pix_t = _as_tensor(pix_axes, dtype=torch.long, device=pixel_flat.device)
            world_t = _as_tensor(world_axes, dtype=torch.long, device=pixel_flat.device)
            for row_index in (
                torch.nonzero(leftover, as_tuple=False).reshape(-1).tolist()
            ):
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
        pix_bits = (pk[:, extra_axes].to(torch.int64) * (2**powers)).sum(dim=-1)
        world_bits = (wk[:, extra_axes].to(torch.int64) * (2**powers)).sum(dim=-1)
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
                known_pix = _as_tensor(
                    pixel_axes_known, dtype=torch.long, device=self.device
                )
                unknown_pix = _as_tensor(
                    unknown_pixel_axes, dtype=torch.long, device=self.device
                )
                known_world = _as_tensor(
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
            known_pix = _as_tensor(
                pixel_axes_known, dtype=torch.long, device=self.device
            )
            unknown_pix = _as_tensor(
                unknown_pixel_axes, dtype=torch.long, device=self.device
            )
            known_world = _as_tensor(
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
                (self.naxis,),
                fill,
                dtype=self._wcs_coord_dtype(self.device),
                device=self.device,
            )
        tensor = _as_tensor(
            value, dtype=self._wcs_coord_dtype(self.device), device=self.device
        )
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
        eps = torch.as_tensor(
            1.0e-4, dtype=self._wcs_coord_dtype(self.device), device=self.device
        )
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

    def _scatter_rows(self, full_shape, dtype, device, idx_a, vals_a, idx_b, vals_b):
        out = torch.zeros(full_shape, dtype=dtype, device=device)
        out = out.index_put((idx_a,), vals_a)
        return out.index_put((idx_b,), vals_b)

    def _first_order_rows(
        self, fn, x: Tensor, at_ref: Tensor, jac: Tensor, x_ref_axes: tuple[int, int]
    ) -> Tensor:
        """Evaluate ``fn`` row-wise with an exact analytic Jacobian on reference rows.

        Rows flagged by ``at_ref`` keep the exact forward value (computed without
        autograd) and carry the analytic first derivative ``jac`` (output-columns x
        input-columns over ``x_ref_axes``); all other rows use ordinary autograd.
        The Hessian on the flagged rows is not defined by this construction (zero).
        """
        idx_s = torch.nonzero(at_ref, as_tuple=True)[0]
        idx_o = torch.nonzero(~at_ref, as_tuple=True)[0]
        out_o = fn(x.index_select(0, idx_o)) if idx_o.numel() else None
        xs = x.index_select(0, idx_s)
        with torch.no_grad():
            val = fn(xs.detach())
        j = _to(jac, dtype=xs.dtype, device=xs.device)
        cols = list(x_ref_axes)
        dx = (xs - xs.detach())[:, cols]
        out_s = val + dx @ j.T
        if out_o is None:
            return self._scatter_rows(
                (x.shape[0], out_s.shape[1]),
                out_s.dtype,
                out_s.device,
                idx_s,
                out_s,
                idx_s[:0],
                out_s[:0],
            )
        return self._scatter_rows(
            (x.shape[0], out_s.shape[1]),
            out_s.dtype,
            out_s.device,
            idx_o,
            out_o,
            idx_s,
            out_s,
        )
