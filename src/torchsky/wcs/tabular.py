from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Literal

import torch
from torch import Tensor

TabularBoundsMode = Literal["nan", "clamp"]


def _as_1d_floating_tensor(value: Any, name: str) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional table")
    if tensor.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if tensor.is_complex():
        raise ValueError(f"{name} must be real-valued")
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all().detach().cpu().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _strict_direction(values: Tensor, name: str) -> int:
    delta = values[1:] - values[:-1]
    if bool(torch.all(delta > 0).detach().cpu().item()):
        return 1
    if bool(torch.all(delta < 0).detach().cpu().item()):
        return -1
    raise ValueError(f"{name} must be strictly monotonic")


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _interp1d(
    query: Tensor, xp: Tensor, fp: Tensor, *, bounds: TabularBoundsMode
) -> Tensor:
    if bounds not in {"nan", "clamp"}:
        raise ValueError("bounds must be 'nan' or 'clamp'")

    dtype = torch.promote_types(torch.promote_types(query.dtype, xp.dtype), fp.dtype)
    device = query.device
    x = query.to(dtype=dtype)
    grid_raw = xp.to(device=device, dtype=dtype)
    values_raw = fp.to(device=device, dtype=dtype)
    if bool((grid_raw[0] <= grid_raw[-1]).detach().cpu().item()):
        grid = grid_raw
        values = values_raw
    else:
        grid = torch.flip(grid_raw, dims=(0,))
        values = torch.flip(values_raw, dims=(0,))

    below = x < grid[0]
    above = x > grid[-1]
    if bounds == "clamp":
        x_eval = torch.clamp(x, min=grid[0], max=grid[-1])
    else:
        x_eval = x

    hi = torch.searchsorted(grid.contiguous(), x_eval.contiguous(), right=False)
    hi = torch.clamp(hi, min=1, max=grid.shape[0] - 1)
    lo = hi - 1

    x0 = grid[lo]
    x1 = grid[hi]
    y0 = values[lo]
    y1 = values[hi]
    t = (x_eval - x0) / (x1 - x0)
    out = y0 + t * (y1 - y0)
    if bounds == "nan":
        out = torch.where(below | above, torch.full_like(out, torch.nan), out)
    return out


@dataclass(frozen=True)
class TabularAxis:
    """One-dimensional table-driven coordinate transform.

    The forward direction maps pixel/index coordinates to world coordinates by
    piecewise-linear interpolation. The inverse maps monotonic world coordinates
    back to pixel/index coordinates with the same interpolation rule.
    """

    pixel: Any
    world: Any
    name: str | None = None
    unit: str | None = None
    bounds: TabularBoundsMode = "nan"

    def __post_init__(self) -> None:
        pixel = _as_1d_floating_tensor(self.pixel, "pixel")
        world = _as_1d_floating_tensor(self.world, "world").to(device=pixel.device)
        if pixel.shape != world.shape:
            raise ValueError("pixel and world tables must have the same shape")
        if self.bounds not in {"nan", "clamp"}:
            raise ValueError("bounds must be 'nan' or 'clamp'")
        _strict_direction(pixel, "pixel")
        object.__setattr__(self, "pixel", pixel)
        object.__setattr__(self, "world", world)
        object.__setattr__(self, "_world_direction", None)

    @property
    def world_direction(self) -> int:
        """Return ``1`` for increasing world values, ``-1`` for decreasing."""
        cached = getattr(self, "_world_direction", None)
        if cached is None:
            cached = _strict_direction(self.world, "world")
            object.__setattr__(self, "_world_direction", cached)
        return cached

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TabularAxis":
        """Return a copy with tables moved to ``device`` and/or ``dtype``."""
        pixel = self.pixel.to(device=device, dtype=dtype)
        world = self.world.to(device=device, dtype=dtype)
        return TabularAxis(
            pixel, world, name=self.name, unit=self.unit, bounds=self.bounds
        )

    def pixel_to_world(
        self, pixel: Any, *, bounds: TabularBoundsMode | None = None
    ) -> Tensor:
        """Interpolate world coordinates from pixel/index coordinates."""
        query = torch.as_tensor(pixel)
        if not query.is_floating_point():
            query = query.to(dtype=self.pixel.dtype)
        return _interp1d(query, self.pixel, self.world, bounds=bounds or self.bounds)

    def world_to_pixel(
        self, world: Any, *, bounds: TabularBoundsMode | None = None
    ) -> Tensor:
        """Interpolate pixel/index coordinates from monotonic world coordinates."""
        query = torch.as_tensor(world)
        if not query.is_floating_point():
            query = query.to(dtype=self.world.dtype)
        if self.world_direction > 0:
            xp = self.world
            fp = self.pixel
        else:
            xp = torch.flip(self.world, dims=(0,))
            fp = torch.flip(self.pixel, dims=(0,))
        return _interp1d(query, xp, fp, bounds=bounds or self.bounds)


@dataclass(frozen=True)
class TabularLookupWCS:
    """Separable multi-axis table-driven WCS transform.

    This is the torch-native execution primitive for independent lookup axes and
    separable multidimensional FITS ``-TAB`` coordinate arrays. Arbitrary coupled
    table-coordinate interpolation needs a separate multi-axis substrate.
    """

    axes: tuple[TabularAxis, ...]

    def __init__(self, axes: Any) -> None:
        axis_tuple = tuple(axes)
        if len(axis_tuple) == 0:
            raise ValueError("at least one tabular axis is required")
        if not all(isinstance(axis, TabularAxis) for axis in axis_tuple):
            raise TypeError("axes must contain only TabularAxis instances")
        object.__setattr__(self, "axes", axis_tuple)

    @property
    def pixel_n_dim(self) -> int:
        return len(self.axes)

    @property
    def world_n_dim(self) -> int:
        return len(self.axes)

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TabularLookupWCS":
        return TabularLookupWCS(
            axis.to(device=device, dtype=dtype) for axis in self.axes
        )

    def pixel_to_world(
        self, pixels: Any, *, bounds: TabularBoundsMode | None = None
    ) -> Tensor:
        pts = torch.as_tensor(pixels)
        if pts.shape[-1] != len(self.axes):
            raise ValueError(f"pixels last dimension must be {len(self.axes)}")
        values = [
            axis.pixel_to_world(pts[..., index], bounds=bounds)
            for index, axis in enumerate(self.axes)
        ]
        return torch.stack(values, dim=-1)

    def world_to_pixel(
        self, world: Any, *, bounds: TabularBoundsMode | None = None
    ) -> Tensor:
        pts = torch.as_tensor(world)
        if pts.shape[-1] != len(self.axes):
            raise ValueError(f"world last dimension must be {len(self.axes)}")
        values = [
            axis.world_to_pixel(pts[..., index], bounds=bounds)
            for index, axis in enumerate(self.axes)
        ]
        return torch.stack(values, dim=-1)


@dataclass(frozen=True)
class CoupledTabularLookupWCS:
    """Coupled multidimensional table lookup with multilinear interpolation.

    ``pixel_axes`` define the table-grid coordinates for each pixel dimension.
    ``world`` has shape ``(*grid_shape, world_n_dim)``. The forward transform is
    differentiable with respect to query coordinates inside each grid cell.
    Inverse lookup is unique only when ``world_n_dim >= pixel_n_dim`` and the
    query lies on the interpolated manifold. Underdetermined tables
    (``world_n_dim < pixel_n_dim``) return a nearest-seed Gauss–Newton
    particular solution (minimum-norm step); it is not the unique inverse.
    """

    pixel_axes: tuple[Tensor, ...]
    world: Tensor
    names: tuple[str | None, ...]
    units: tuple[str | None, ...]
    bounds: TabularBoundsMode = "nan"

    def __init__(
        self,
        pixel_axes: Any,
        world: Any,
        *,
        names: Any = None,
        units: Any = None,
        bounds: TabularBoundsMode = "nan",
    ) -> None:
        axis_tuple = tuple(
            _as_1d_floating_tensor(axis, f"pixel_axes[{index}]")
            for index, axis in enumerate(pixel_axes)
        )
        if len(axis_tuple) == 0:
            raise ValueError("at least one pixel axis is required")
        if bounds not in {"nan", "clamp"}:
            raise ValueError("bounds must be 'nan' or 'clamp'")
        for index, axis in enumerate(axis_tuple):
            _strict_direction(axis, f"pixel_axes[{index}]")

        device = axis_tuple[0].device
        world_raw = torch.as_tensor(world)
        if world_raw.is_complex():
            raise ValueError("world grid must be real-valued")
        dtype = torch.promote_types(
            torch.float64,
            world_raw.dtype if world_raw.is_floating_point() else torch.float64,
        )
        axes = [axis.to(device=device, dtype=dtype) for axis in axis_tuple]
        world_t = world_raw.to(device=device)
        if not world_t.is_floating_point():
            world_t = world_t.to(dtype=dtype)
        else:
            world_t = world_t.to(dtype=dtype)
        if world_t.ndim == len(axes):
            world_t = world_t.unsqueeze(-1)
        if world_t.ndim != len(axes) + 1:
            raise ValueError(
                "world must have shape (*grid_shape, world_n_dim) or grid_shape"
            )
        grid_shape = tuple(axis.shape[0] for axis in axes)
        if tuple(world_t.shape[:-1]) != grid_shape:
            raise ValueError(
                f"world grid shape {tuple(world_t.shape[:-1])} does not match "
                f"pixel axes {grid_shape}"
            )
        if world_t.shape[-1] < 1:
            raise ValueError("world_n_dim must be at least one")
        if not bool(torch.isfinite(world_t).all().detach().cpu().item()):
            raise ValueError("world grid must contain only finite values")

        for index, axis in enumerate(axes):
            if not bool((axis[0] <= axis[-1]).detach().cpu().item()):
                axes[index] = torch.flip(axis, dims=(0,))
                world_t = torch.flip(world_t, dims=(index,))

        world_n_dim = int(world_t.shape[-1])
        name_tuple = tuple(names) if names is not None else (None,) * world_n_dim
        unit_tuple = tuple(units) if units is not None else (None,) * world_n_dim
        if len(name_tuple) != world_n_dim:
            raise ValueError("names length must match world_n_dim")
        if len(unit_tuple) != world_n_dim:
            raise ValueError("units length must match world_n_dim")

        object.__setattr__(self, "pixel_axes", tuple(axes))
        object.__setattr__(self, "world", world_t)
        object.__setattr__(self, "names", name_tuple)
        object.__setattr__(self, "units", unit_tuple)
        object.__setattr__(self, "bounds", bounds)

    @property
    def pixel_n_dim(self) -> int:
        return len(self.pixel_axes)

    @property
    def world_n_dim(self) -> int:
        return int(self.world.shape[-1])

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(int(axis.shape[0]) for axis in self.pixel_axes)

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "CoupledTabularLookupWCS":
        return CoupledTabularLookupWCS(
            tuple(axis.to(device=device, dtype=dtype) for axis in self.pixel_axes),
            self.world.to(device=device, dtype=dtype),
            names=self.names,
            units=self.units,
            bounds=self.bounds,
        )

    def pixel_to_world(
        self, pixels: Any, *, bounds: TabularBoundsMode | None = None
    ) -> Tensor:
        pts = torch.as_tensor(pixels)
        if pts.shape[-1] != len(self.pixel_axes):
            raise ValueError(f"pixels last dimension must be {len(self.pixel_axes)}")
        bounds_mode = bounds or self.bounds
        if bounds_mode not in {"nan", "clamp"}:
            raise ValueError("bounds must be 'nan' or 'clamp'")

        dtype = self.world.dtype if pts.is_floating_point() else torch.float64
        if pts.is_floating_point():
            dtype = torch.promote_types(pts.dtype, self.world.dtype)
        device = pts.device
        query = pts.to(dtype=dtype)
        axes = tuple(axis.to(device=device, dtype=dtype) for axis in self.pixel_axes)
        values = self.world.to(device=device, dtype=dtype)
        return _interp_multilinear(query, axes, values, bounds=bounds_mode)

    def world_to_pixel(
        self,
        world: Any,
        *,
        bounds: TabularBoundsMode | None = None,
        max_iter: int = 20,
        tol: float = 1.0e-10,
    ) -> Tensor:
        pts = torch.as_tensor(world)
        if pts.shape[-1] != self.world_n_dim:
            raise ValueError(f"world last dimension must be {self.world_n_dim}")
        bounds_mode = bounds or self.bounds
        if bounds_mode not in {"nan", "clamp"}:
            raise ValueError("bounds must be 'nan' or 'clamp'")

        query_shape = tuple(pts.shape[:-1])
        dtype = self.world.dtype if pts.is_floating_point() else torch.float64
        if pts.is_floating_point():
            dtype = torch.promote_types(pts.dtype, self.world.dtype)
        device = pts.device
        target = pts.to(dtype=dtype).reshape(-1, self.world_n_dim)
        axes = tuple(axis.to(device=device, dtype=dtype) for axis in self.pixel_axes)
        world_grid = self.world.to(device=device, dtype=dtype)
        flat_world = world_grid.reshape(-1, self.world_n_dim)
        flat_axes = _flat_grid_pixel_coordinates(axes)

        distances = torch.sum(
            (target[:, None, :] - flat_world[None, :, :]) ** 2, dim=-1
        )
        nearest = torch.argmin(distances, dim=-1)
        initial = flat_axes.index_select(0, nearest)

        use_global_2d = self.pixel_n_dim == 2 and self.world_n_dim >= 2
        if use_global_2d:
            out = _solve_coupled_tabular_2d_global_batch(
                self,
                target,
                axes,
                max_iter=max_iter,
                tol=tol,
            )
        else:
            solved = [
                _solve_coupled_tabular_point(
                    self,
                    target[row_index],
                    initial[row_index],
                    axes,
                    max_iter=max_iter,
                    tol=tol,
                    bounds=bounds_mode,
                    lower_bounds=None,
                    upper_bounds=None,
                )
                for row_index in range(target.shape[0])
            ]
            out = torch.stack(solved, dim=0)
        return out.reshape(*query_shape, self.pixel_n_dim)


def _flat_grid_pixel_coordinates(axes: tuple[Tensor, ...]) -> Tensor:
    meshes = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([mesh.reshape(-1) for mesh in meshes], dim=-1)


def _solve_coupled_tabular_point(
    lookup: CoupledTabularLookupWCS,
    target: Tensor,
    initial: Tensor,
    axes: tuple[Tensor, ...],
    *,
    max_iter: int,
    tol: float,
    bounds: TabularBoundsMode,
    lower_bounds: Tensor | None,
    upper_bounds: Tensor | None,
) -> Tensor:
    current = initial.clone()
    if lower_bounds is None:
        lower = torch.stack([axis[0] for axis in axes]).to(
            device=target.device, dtype=target.dtype
        )
    else:
        lower = lower_bounds.to(device=target.device, dtype=target.dtype)
    if upper_bounds is None:
        upper = torch.stack([axis[-1] for axis in axes]).to(
            device=target.device, dtype=target.dtype
        )
    else:
        upper = upper_bounds.to(device=target.device, dtype=target.dtype)
    eps = torch.stack(
        [torch.clamp((axis[-1] - axis[0]).abs() * 1.0e-6, min=1.0e-8) for axis in axes]
    ).to(device=target.device, dtype=target.dtype)
    for _ in range(max_iter):
        value = lookup.pixel_to_world(current.unsqueeze(0), bounds="clamp")[0].to(
            dtype=target.dtype
        )
        residual = value - target
        if not bool(torch.isfinite(residual).all().detach().cpu().item()):
            break
        if float(torch.linalg.vector_norm(residual).detach().cpu().item()) <= tol:
            return current
        columns = []
        for axis_index in range(len(axes)):
            perturbed = current.clone()
            step = eps[axis_index]
            if bool(
                (current[axis_index] + step > axes[axis_index][-1])
                .detach()
                .cpu()
                .item()
            ):
                step = -step
            perturbed[axis_index] = perturbed[axis_index] + step
            pvalue = lookup.pixel_to_world(perturbed.unsqueeze(0), bounds="clamp")[
                0
            ].to(dtype=target.dtype)
            columns.append((pvalue - value) / step)
        jac = torch.stack(columns, dim=-1)
        try:
            if jac.shape[0] == jac.shape[1]:
                step = torch.linalg.solve(jac, residual)
            else:
                step = torch.linalg.lstsq(jac, residual.unsqueeze(-1)).solution.squeeze(
                    -1
                )
        except RuntimeError:
            break
        if not bool(torch.isfinite(step).all().detach().cpu().item()):
            break
        current = current - step
        current = torch.minimum(torch.maximum(current, lower), upper)
    return torch.full_like(current, torch.nan)


def _solve_coupled_tabular_point_2d_global(
    lookup: CoupledTabularLookupWCS,
    target: Tensor,
    axes: tuple[Tensor, ...],
    *,
    max_iter: int,
    tol: float,
) -> Tensor:
    x_axis, y_axis = axes
    world = lookup.world.to(device=target.device, dtype=target.dtype)
    axis_tol = torch.stack(
        [
            torch.clamp((x_axis[-1] - x_axis[0]).abs() * 1.0e-9, min=1.0e-10),
            torch.clamp((y_axis[-1] - y_axis[0]).abs() * 1.0e-9, min=1.0e-10),
        ]
    ).to(device=target.device, dtype=target.dtype)
    world_tol = torch.as_tensor(tol, device=target.device, dtype=target.dtype)
    solutions: list[Tensor] = []

    for x_index in range(x_axis.shape[0] - 1):
        for y_index in range(y_axis.shape[0] - 1):
            corners = world[
                x_index : x_index + 2,
                y_index : y_index + 2,
                :,
            ].reshape(-1, lookup.world_n_dim)
            if bool(
                torch.any(target < corners.amin(dim=0) - world_tol)
                .detach()
                .cpu()
                .item()
            ):
                continue
            if bool(
                torch.any(target > corners.amax(dim=0) + world_tol)
                .detach()
                .cpu()
                .item()
            ):
                continue

            lower = torch.stack((x_axis[x_index], y_axis[y_index])).to(
                device=target.device, dtype=target.dtype
            )
            upper = torch.stack((x_axis[x_index + 1], y_axis[y_index + 1])).to(
                device=target.device, dtype=target.dtype
            )
            initial = (lower + upper) * 0.5
            candidate = _solve_coupled_tabular_point(
                lookup,
                target,
                initial,
                axes,
                max_iter=max_iter,
                tol=tol,
                bounds="clamp",
                lower_bounds=lower,
                upper_bounds=upper,
            )
            if not bool(torch.isfinite(candidate).all().detach().cpu().item()):
                continue
            if bool(
                torch.any(candidate < lower - axis_tol).detach().cpu().item()
            ) or bool(torch.any(candidate > upper + axis_tol).detach().cpu().item()):
                continue
            residual = (
                lookup.pixel_to_world(candidate.unsqueeze(0), bounds="clamp")[0].to(
                    dtype=target.dtype
                )
                - target
            )
            if float(torch.linalg.vector_norm(residual).detach().cpu().item()) > tol:
                continue
            duplicate = False
            for solution in solutions:
                if bool(
                    torch.all(torch.abs(solution - candidate) <= axis_tol)
                    .detach()
                    .cpu()
                    .item()
                ):
                    duplicate = True
                    break
            if not duplicate:
                solutions.append(candidate)
                if len(solutions) > 1:
                    return torch.full(
                        (lookup.pixel_n_dim,),
                        torch.nan,
                        dtype=target.dtype,
                        device=target.device,
                    )

    if len(solutions) == 1:
        return solutions[0]
    return torch.full(
        (lookup.pixel_n_dim,),
        torch.nan,
        dtype=target.dtype,
        device=target.device,
    )


def _newton_coupled_tabular_batch(
    lookup: CoupledTabularLookupWCS,
    target: Tensor,
    initial: Tensor,
    axes: tuple[Tensor, ...],
    lower: Tensor,
    upper: Tensor,
    *,
    max_iter: int,
    tol: float,
) -> Tensor:
    """Gauss–Newton for N independent seeds. ``target`` is ``[N, W]``."""
    current = initial.clone()
    n_points = int(current.shape[0])
    if n_points == 0:
        return current
    eps = torch.stack(
        [torch.clamp((axis[-1] - axis[0]).abs() * 1.0e-6, min=1.0e-8) for axis in axes]
    ).to(device=target.device, dtype=target.dtype)
    converged = torch.zeros(n_points, dtype=torch.bool, device=target.device)
    for _ in range(max_iter):
        value = lookup.pixel_to_world(current, bounds="clamp").to(dtype=target.dtype)
        residual = value - target
        finite = torch.isfinite(residual).all(dim=-1)
        residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        converged = converged | (finite & (residual_norm <= tol))
        active = (~converged) & finite
        if not bool(active.any()):
            break
        columns = []
        for axis_index, axis in enumerate(axes):
            step = torch.where(
                current[:, axis_index] + eps[axis_index] > axis[-1],
                -eps[axis_index],
                eps[axis_index],
            )
            perturbed = current.clone()
            perturbed[:, axis_index] = perturbed[:, axis_index] + step
            pvalue = lookup.pixel_to_world(perturbed, bounds="clamp").to(
                dtype=target.dtype
            )
            columns.append((pvalue - value) / step.unsqueeze(-1))
        jac = torch.stack(columns, dim=-1)
        delta = torch.linalg.lstsq(jac, residual.unsqueeze(-1)).solution.squeeze(-1)
        delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
        updated = torch.minimum(torch.maximum(current - delta, lower), upper)
        current = torch.where(active.unsqueeze(-1), updated, current)
    value = lookup.pixel_to_world(current, bounds="clamp").to(dtype=target.dtype)
    residual = value - target
    ok = (
        torch.isfinite(current).all(dim=-1)
        & torch.isfinite(residual).all(dim=-1)
        & (torch.linalg.vector_norm(residual, dim=-1) <= tol)
    )
    return torch.where(ok.unsqueeze(-1), current, torch.full_like(current, torch.nan))


def _solve_coupled_tabular_partial(
    lookup: CoupledTabularLookupWCS,
    pixel: Tensor,
    world: Tensor,
    pixel_known: Tensor,
    world_known: Tensor,
    *,
    max_iter: int = 20,
    tol: float = 1.0e-10,
) -> tuple[Tensor, Tensor]:
    """Solve leftover mixed coupled rows; freeze known pixels, match known world."""
    axes = lookup.pixel_axes
    current = pixel.to(dtype=world.dtype, device=world.device).clone()
    freeze = pixel_known.to(dtype=torch.bool, device=world.device)
    known_w = world_known.to(dtype=torch.bool, device=world.device)
    for axis_index, axis in enumerate(axes):
        if not bool(freeze[axis_index]):
            current[axis_index] = 0.5 * (axis[0] + axis[-1])
    lower = torch.stack(
        [pixel[i] if bool(freeze[i]) else axis[0] for i, axis in enumerate(axes)]
    ).to(device=world.device, dtype=world.dtype)
    upper = torch.stack(
        [pixel[i] if bool(freeze[i]) else axis[-1] for i, axis in enumerate(axes)]
    ).to(device=world.device, dtype=world.dtype)
    if not bool(known_w.any()):
        got = lookup.pixel_to_world(current.unsqueeze(0), bounds="clamp")[0]
        return current, got
    eps = torch.stack(
        [torch.clamp((axis[-1] - axis[0]).abs() * 1.0e-6, min=1.0e-8) for axis in axes]
    ).to(device=world.device, dtype=world.dtype)
    for _ in range(max_iter):
        value = lookup.pixel_to_world(current.unsqueeze(0), bounds="clamp")[0]
        residual = torch.where(known_w, value - world, torch.zeros_like(value))
        if float(torch.linalg.vector_norm(residual)) <= tol:
            break
        columns = []
        for axis_index, axis in enumerate(axes):
            if bool(freeze[axis_index]):
                columns.append(torch.zeros_like(value))
                continue
            step = (
                -eps[axis_index]
                if current[axis_index] + eps[axis_index] > axis[-1]
                else eps[axis_index]
            )
            perturbed = current.clone()
            perturbed[axis_index] = perturbed[axis_index] + step
            pvalue = lookup.pixel_to_world(perturbed.unsqueeze(0), bounds="clamp")[0]
            columns.append((pvalue - value) / step)
        jac = torch.stack(columns, dim=-1)
        delta = torch.linalg.lstsq(jac, residual.unsqueeze(-1)).solution.reshape(-1)
        delta = torch.where(freeze, torch.zeros_like(delta), delta)
        delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
        current = torch.minimum(torch.maximum(current - delta, lower), upper)
    got = lookup.pixel_to_world(current.unsqueeze(0), bounds="clamp")[0]
    return current, got


def _point_in_convex_quad_2d(
    point: Tensor,
    c00: Tensor,
    c10: Tensor,
    c11: Tensor,
    c01: Tensor,
    *,
    tol: Tensor,
) -> Tensor:
    """True when ``point`` is inside the convex quad ``c00-c10-c11-c01``."""

    def _cross(origin: Tensor, dest: Tensor) -> Tensor:
        edge = dest - origin
        delta = point - origin
        return edge[..., 0] * delta[..., 1] - edge[..., 1] * delta[..., 0]

    s0 = _cross(c00, c10)
    s1 = _cross(c10, c11)
    s2 = _cross(c11, c01)
    s3 = _cross(c01, c00)
    nonneg = (s0 >= -tol) & (s1 >= -tol) & (s2 >= -tol) & (s3 >= -tol)
    nonpos = (s0 <= tol) & (s1 <= tol) & (s2 <= tol) & (s3 <= tol)
    return nonneg | nonpos


def _solve_coupled_tabular_2d_global_batch(
    lookup: CoupledTabularLookupWCS,
    target: Tensor,
    axes: tuple[Tensor, ...],
    *,
    max_iter: int,
    tol: float,
) -> Tensor:
    """Unique-cell 2-D inverse for a batch of world samples."""
    x_axis, y_axis = axes
    n_points = int(target.shape[0])
    out = torch.full(
        (n_points, lookup.pixel_n_dim),
        torch.nan,
        dtype=target.dtype,
        device=target.device,
    )
    if n_points == 0:
        return out

    world = lookup.world.to(device=target.device, dtype=target.dtype)
    world_tol = torch.as_tensor(tol, device=target.device, dtype=target.dtype)
    nx = int(x_axis.shape[0] - 1)
    ny = int(y_axis.shape[0] - 1)
    # ponytail: AABB mask is N×(nx-1)×(ny-1). Ceiling ~32 MiB of flags; chunk cells if a survey table exceeds it.
    if n_points * nx * ny > 4_000_000:
        for row_index in range(n_points):
            out[row_index] = _solve_coupled_tabular_point_2d_global(
                lookup,
                target[row_index],
                axes,
                max_iter=max_iter,
                tol=tol,
            )
        return out
    if lookup.world_n_dim == 2:
        c00 = world[:-1, :-1]
        c10 = world[1:, :-1]
        c11 = world[1:, 1:]
        c01 = world[:-1, 1:]
        point = target[:, None, None, :]
        in_cell = _point_in_convex_quad_2d(
            point, c00, c10, c11, c01, tol=world_tol
        )
    else:
        corners = torch.stack(
            (
                world[:-1, :-1],
                world[1:, :-1],
                world[:-1, 1:],
                world[1:, 1:],
            ),
            dim=0,
        )
        cell_lo = corners.amin(dim=0)
        cell_hi = corners.amax(dim=0)
        in_cell = (target[:, None, None, :] + world_tol >= cell_lo[None, :, :, :]) & (
            target[:, None, None, :] - world_tol <= cell_hi[None, :, :, :]
        )
        in_cell = in_cell.all(dim=-1)
    n_cells = in_cell.reshape(n_points, -1).sum(dim=-1)
    unique = n_cells == 1
    unique_idx = torch.nonzero(unique, as_tuple=False).reshape(-1)
    if unique_idx.numel() > 0:
        flat = in_cell.reshape(n_points, -1)
        cell_index = flat.to(dtype=torch.int64).argmax(dim=-1)[unique_idx]
        ix = torch.div(cell_index, ny, rounding_mode="floor")
        iy = cell_index - ix * ny
        lower = torch.stack((x_axis[ix], y_axis[iy]), dim=-1)
        upper = torch.stack((x_axis[ix + 1], y_axis[iy + 1]), dim=-1)
        unique_target = target.index_select(0, unique_idx)
        solved = _newton_coupled_tabular_batch(
            lookup,
            unique_target,
            (lower + upper) * 0.5,
            axes,
            lower,
            upper,
            max_iter=max_iter,
            tol=tol,
        )
        axis_tol = torch.stack(
            [
                torch.clamp((x_axis[-1] - x_axis[0]).abs() * 1.0e-9, min=1.0e-10),
                torch.clamp((y_axis[-1] - y_axis[0]).abs() * 1.0e-9, min=1.0e-10),
            ]
        ).to(device=target.device, dtype=target.dtype)
        in_bounds = torch.all(solved + axis_tol >= lower, dim=-1) & torch.all(
            solved - axis_tol <= upper, dim=-1
        )
        keep = torch.isfinite(solved).all(dim=-1) & in_bounds
        out[unique_idx] = torch.where(keep.unsqueeze(-1), solved, out[unique_idx])

    fallback = ~unique
    if bool(fallback.any()):
        rows = torch.nonzero(fallback, as_tuple=False).reshape(-1)
        for row_index in rows.tolist():
            out[row_index] = _solve_coupled_tabular_point_2d_global(
                lookup,
                target[row_index],
                axes,
                max_iter=max_iter,
                tol=tol,
            )
    return out


def _interp_multilinear(
    query: Tensor,
    axes: tuple[Tensor, ...],
    values: Tensor,
    *,
    bounds: TabularBoundsMode,
) -> Tensor:
    query_shape = tuple(query.shape[:-1])
    invalid = torch.zeros(query_shape, dtype=torch.bool, device=query.device)
    lows: list[Tensor] = []
    highs: list[Tensor] = []
    weights_hi: list[Tensor] = []

    for axis_index, axis in enumerate(axes):
        q = query[..., axis_index]
        below = q < axis[0]
        above = q > axis[-1]
        invalid = invalid | below | above
        if bounds == "clamp":
            q_eval = torch.clamp(q, min=axis[0], max=axis[-1])
        else:
            q_eval = q
        hi = torch.searchsorted(axis.contiguous(), q_eval.contiguous(), right=False)
        hi = torch.clamp(hi, min=1, max=axis.shape[0] - 1)
        lo = hi - 1
        x0 = axis[lo]
        x1 = axis[hi]
        t = (q_eval - x0) / (x1 - x0)
        lows.append(lo)
        highs.append(hi)
        weights_hi.append(t)

    grid_shape = tuple(int(axis.shape[0]) for axis in axes)
    strides: list[int] = []
    stride = 1
    for size in reversed(grid_shape[1:]):
        stride *= size
        strides.insert(0, stride)
    strides.append(1)

    flat_values = values.reshape(-1, values.shape[-1])
    out = torch.zeros(
        (*query_shape, values.shape[-1]), dtype=values.dtype, device=values.device
    )
    n_corners = 1 << len(axes)
    for corner in range(n_corners):
        flat_index = torch.zeros(query_shape, dtype=torch.int64, device=query.device)
        weight = torch.ones(query_shape, dtype=values.dtype, device=values.device)
        for axis_index, stride_value in enumerate(strides):
            use_hi = bool((corner >> axis_index) & 1)
            idx = highs[axis_index] if use_hi else lows[axis_index]
            weight_axis = (
                weights_hi[axis_index] if use_hi else (1.0 - weights_hi[axis_index])
            )
            flat_index = flat_index + idx * stride_value
            weight = weight * weight_axis.to(dtype=values.dtype, device=values.device)
        corner_values = flat_values.index_select(0, flat_index.reshape(-1)).reshape(
            *query_shape, values.shape[-1]
        )
        out = out + corner_values * weight.unsqueeze(-1)
    if bounds == "nan":
        out = torch.where(invalid.unsqueeze(-1), torch.full_like(out, torch.nan), out)
    return out


def _tab_alt_key(name: str, key: str) -> str:
    return f"{name}{key}" if key else name


def _tab_header_get(header: Mapping[str, Any], name: str, key: str) -> Any:
    alt_name = _tab_alt_key(name, key)
    if alt_name in header:
        return header[alt_name]
    if key:
        return None
    return header.get(name)


def _tab_header_naxis(header: Mapping[str, Any], key: str) -> int:
    wcsaxes = _tab_header_get(header, "WCSAXES", key)
    if wcsaxes is not None:
        return int(wcsaxes)
    return int(header.get("NAXIS", 0))


def _case_insensitive_column_key(keys: Any, column: str) -> Any | None:
    target = str(column).strip().upper()
    matches = [key for key in keys if str(key).strip().upper() == target]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"table column {column!r} is ambiguous by case")
    return None


def _read_mapping_column(mapping: Mapping[str, Any], column: str) -> Any:
    if column in mapping:
        return mapping[column]
    key = _case_insensitive_column_key(mapping.keys(), column)
    if key is not None:
        return mapping[key]
    raise KeyError(f"table column {column!r} not found")


def _read_table_column(table: Any, column: str) -> Any:
    if isinstance(table, Mapping):
        return _read_mapping_column(table, column)
    if hasattr(table, "data"):
        data = table.data
        if isinstance(data, Mapping):
            return _read_mapping_column(data, column)
        try:
            return data[column]
        except (KeyError, TypeError):
            pass
        names = getattr(getattr(data, "dtype", None), "names", None) or getattr(
            data, "names", None
        )
        if names:
            key = _case_insensitive_column_key(names, column)
            if key is not None:
                return data[key]
    if hasattr(table, "get_column"):
        try:
            return table.get_column(column)
        except (KeyError, TypeError, ValueError):
            pass
    if hasattr(table, "keys"):
        try:
            key = _case_insensitive_column_key(table.keys(), column)
            if key is not None:
                return table[key]
        except TypeError:
            pass
    try:
        return table[column]
    except Exception as exc:  # pragma: no cover - adapter fallback
        raise KeyError(f"table column {column!r} not found") from exc


def _header_get_int(header: Mapping[str, Any], name: str, default: int) -> int:
    value = header.get(name)
    if value is None:
        return default
    return int(value)


def _table_header(table: Any) -> Mapping[str, Any] | None:
    header = getattr(table, "header", None)
    if isinstance(header, Mapping):
        return header
    return None


def _table_matches_identity(
    table: Any,
    *,
    extname: str | None,
    extver: int | None,
    extlev: int | None,
) -> bool:
    header = _table_header(table)
    if header is None:
        return extver is None and extlev is None
    if extname is not None:
        table_extname = header.get("EXTNAME")
        if table_extname is None:
            return False
        if str(table_extname).strip().upper() != extname.strip().upper():
            return False
    if extver is not None and _header_get_int(header, "EXTVER", 1) != extver:
        return False
    if extlev is not None and _header_get_int(header, "EXTLEVEL", 1) != extlev:
        return False
    return True


def _mapping_key_matches_identity(
    key: Any,
    *,
    extname: str | None,
    extver: int | None,
    extlev: int | None,
) -> bool:
    if isinstance(key, tuple):
        if not key:
            return False
        key_name = str(key[0]).strip().upper()
        if extname is not None and key_name != extname.strip().upper():
            return False
        if len(key) >= 2 and extver is not None and int(key[1]) != extver:
            return False
        if len(key) >= 3 and extlev is not None and int(key[2]) != extlev:
            return False
        if extver is not None and len(key) < 2:
            return False
        if extlev is not None and len(key) < 3:
            return False
        return True
    if extname is None:
        return False
    return (
        extver is None
        and extlev is None
        and str(key).strip().upper() == extname.strip().upper()
    )


def _resolve_tab_table(
    tables: Any,
    extname: str | None,
    *,
    extver: int | None = None,
    extlev: int | None = None,
) -> Any:
    if extname is None or extname == "":
        if extver is not None or extlev is not None:
            raise KeyError("EXTVER/EXTLEVEL require PSi_0 table extension name")
        return tables
    target = str(extname).strip().upper()
    if isinstance(tables, Mapping):
        for name, table in tables.items():
            if not _mapping_key_matches_identity(
                name, extname=target, extver=extver, extlev=extlev
            ):
                continue
            if _table_header(table) is None or _table_matches_identity(
                table, extname=None, extver=extver, extlev=extlev
            ):
                return table
    # Check if tables supports string lookup (like HDUList)
    if hasattr(tables, "__getitem__"):
        lookup_keys: list[Any] = []
        if extver is not None and extlev is not None:
            lookup_keys.append((extname, extver, extlev))
            lookup_keys.append((target, extver, extlev))
        if extver is not None:
            lookup_keys.append((extname, extver))
            lookup_keys.append((target, extver))
        if extver is None and extlev is None:
            lookup_keys.extend([extname, target])
        for key in lookup_keys:
            try:
                table = tables[key]
                if _table_matches_identity(
                    table, extname=target, extver=extver, extlev=extlev
                ):
                    return table
            except (KeyError, TypeError, ValueError, IndexError):
                pass
    # If it is a sequence/list/iterable, search for matching EXTNAME
    if hasattr(tables, "__iter__") or isinstance(tables, (list, tuple)):
        if not isinstance(tables, (str, bytes, Mapping)):
            for hdu in tables:
                if _table_header(hdu) is not None and _table_matches_identity(
                    hdu, extname=target, extver=extver, extlev=extlev
                ):
                    return hdu
    detail = f"table extension {extname!r}"
    if extver is not None:
        detail += f" EXTVER={extver}"
    if extlev is not None:
        detail += f" EXTLEVEL={extlev}"
    raise KeyError(f"{detail} not found")


def _extract_separable_tab_axis(
    tensor: Tensor, name: str, tab_axis_number: int
) -> Tensor:
    if tab_axis_number < 1:
        raise ValueError(f"{name} PVi_3 tabular axis number must be >= 1")
    if tab_axis_number > tensor.ndim:
        raise ValueError(
            f"{name} PVi_3 selects tabular axis {tab_axis_number}, "
            f"but the table array has {tensor.ndim} dimension(s)"
        )
    moved = torch.movedim(tensor, tab_axis_number - 1, 0)
    flat = moved.reshape(moved.shape[0], -1)
    reference = flat[:, :1]
    if not torch.equal(flat, reference.expand_as(flat)):
        raise ValueError(
            f"{name} is a coupled multidimensional coordinate array; only "
            "separable -TAB coordinate arrays are supported by tabular_axes_from_fits"
        )
    return reference[:, 0]


def _as_independent_tab_array(
    value: Any, name: str, tab_axis_number: int | None = None
) -> Tensor:
    tensor = torch.as_tensor(value)
    squeezed = torch.squeeze(tensor)
    if squeezed.ndim == 1:
        return squeezed
    if tab_axis_number is None:
        raise ValueError(
            f"{name} is multidimensional and requires PVi_3 tabular-axis mapping"
        )
    return _extract_separable_tab_axis(tensor, name, tab_axis_number)


def _as_tabular_index_array(
    value: Any, name: str, tab_axis_number: int | None = None
) -> Tensor:
    tensor = torch.as_tensor(value)
    squeezed = torch.squeeze(tensor)
    if squeezed.ndim == 1:
        return squeezed
    if tab_axis_number is None:
        raise ValueError(
            f"{name} is multidimensional and requires PVi_3 tabular-axis mapping"
        )
    return _extract_separable_tab_axis(tensor, name, tab_axis_number)


def _default_tab_axis_number(value: int | None) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise ValueError("PVi_3 tabular axis number must be >= 1")
    return value


def tabular_axes_from_fits(
    header: Mapping[str, Any],
    tables: Any,
    *,
    key: str = "",
) -> dict[int, TabularAxis]:
    """Build explicit ``WCS(..., tabular_axes=...)`` axes from FITS ``-TAB`` metadata.

    This implements independent one-dimensional FITS table lookup WCS plus the
    exact separable subset of multidimensional coordinate arrays selected by
    ``PVi_3``. For each ``CTYPEi`` ending in ``-TAB``, ``PSi_0`` selects a table
    by ``EXTNAME`` when ``tables`` is a mapping, ``PSi_1`` names the
    coordinate-array column, and optional ``PSi_2`` names the indexing-vector
    column. If no indexing vector is present, samples are indexed by ``0..N-1``.
    Multidimensional coordinate arrays must be invariant along non-selected
    dimensions; truly coupled arrays fail explicitly.
    """
    alt_key = "" if key in {"", " "} else str(key).strip().upper()
    if alt_key and (len(alt_key) != 1 or not ("A" <= alt_key <= "Z")):
        raise ValueError("alternate WCS key must be '', ' ', or one letter A-Z")

    axes: dict[int, TabularAxis] = {}
    for axis_number in range(1, _tab_header_naxis(header, alt_key) + 1):
        ctype = (
            str(_tab_header_get(header, f"CTYPE{axis_number}", alt_key) or "")
            .strip()
            .upper()
        )
        if not ctype.endswith("-TAB"):
            continue
        extname_raw = _tab_header_get(header, f"PS{axis_number}_0", alt_key)
        coord_col_raw = _tab_header_get(header, f"PS{axis_number}_1", alt_key)
        index_col_raw = _tab_header_get(header, f"PS{axis_number}_2", alt_key)
        extver = _optional_int(
            _tab_header_get(header, f"PV{axis_number}_1", alt_key),
            f"PV{axis_number}_1",
        )
        extlev = _optional_int(
            _tab_header_get(header, f"PV{axis_number}_2", alt_key),
            f"PV{axis_number}_2",
        )
        tab_axis_number = _optional_int(
            _tab_header_get(header, f"PV{axis_number}_3", alt_key),
            f"PV{axis_number}_3",
        )
        tab_axis_number = _default_tab_axis_number(tab_axis_number)
        if coord_col_raw is None:
            raise ValueError(
                f"CTYPE{axis_number} is '-TAB' but PS{axis_number}_1 is missing"
            )

        table = _resolve_tab_table(
            tables,
            None if extname_raw is None else str(extname_raw).strip(),
            extver=extver,
            extlev=extlev,
        )
        coord_col = str(coord_col_raw).strip()
        coord = _as_independent_tab_array(
            _read_table_column(table, coord_col),
            f"PS{axis_number}_1 column {coord_col!r}",
            tab_axis_number,
        )
        if index_col_raw is None or str(index_col_raw).strip() == "":
            pixel = torch.arange(coord.shape[0], dtype=torch.float64)
        else:
            index_col = str(index_col_raw).strip()
            pixel = _as_tabular_index_array(
                _read_table_column(table, index_col),
                f"PS{axis_number}_2 column {index_col!r}",
                tab_axis_number,
            )
        axes[axis_number] = TabularAxis(
            pixel,
            coord,
            name=coord_col,
            unit=str(_tab_header_get(header, f"CUNIT{axis_number}", alt_key) or "")
            or None,
        )
    return axes


def coupled_tabular_wcs_and_axes_from_fits(
    header: Mapping[str, Any],
    tables: Any,
    *,
    key: str = "",
) -> tuple[CoupledTabularLookupWCS, tuple[int, ...], tuple[int, ...]]:
    """Build a coupled table-lookup WCS plus FITS world and table-index axes.

    World axes are the ``-TAB`` CTYPEs. Table-index (pixel) axes are the WCS
    axes that own each table dimension via ``PVi_3`` (lowest axis number per
    table dim). Inverse solving is on ``CoupledTabularLookupWCS``.
    """
    alt_key = "" if key in {"", " "} else str(key).strip().upper()
    if alt_key and (len(alt_key) != 1 or not ("A" <= alt_key <= "Z")):
        raise ValueError("alternate WCS key must be '', ' ', or one letter A-Z")

    coords: list[tuple[int, str, str | None, Tensor, int]] = []
    grid_shape: tuple[int, ...] | None = None
    pixel_axes_by_tab_axis: dict[int, Tensor] = {}

    for axis_number in range(1, _tab_header_naxis(header, alt_key) + 1):
        ctype = (
            str(_tab_header_get(header, f"CTYPE{axis_number}", alt_key) or "")
            .strip()
            .upper()
        )
        if not ctype.endswith("-TAB"):
            continue
        tab_axis_number = _optional_int(
            _tab_header_get(header, f"PV{axis_number}_3", alt_key),
            f"PV{axis_number}_3",
        )
        if tab_axis_number is None:
            raise ValueError(
                "coupled FITS -TAB lookup requires PVi_3 on every tabular axis"
            )
        tab_axis_number = _default_tab_axis_number(tab_axis_number)
        coord_col_raw = _tab_header_get(header, f"PS{axis_number}_1", alt_key)
        if coord_col_raw is None:
            raise ValueError(
                f"CTYPE{axis_number} is '-TAB' but PS{axis_number}_1 is missing"
            )
        extname_raw = _tab_header_get(header, f"PS{axis_number}_0", alt_key)
        index_col_raw = _tab_header_get(header, f"PS{axis_number}_2", alt_key)
        extver = _optional_int(
            _tab_header_get(header, f"PV{axis_number}_1", alt_key),
            f"PV{axis_number}_1",
        )
        extlev = _optional_int(
            _tab_header_get(header, f"PV{axis_number}_2", alt_key),
            f"PV{axis_number}_2",
        )
        table = _resolve_tab_table(
            tables,
            None if extname_raw is None else str(extname_raw).strip(),
            extver=extver,
            extlev=extlev,
        )

        coord_col = str(coord_col_raw).strip()
        coord = torch.as_tensor(_read_table_column(table, coord_col))
        coord = torch.squeeze(coord)
        if coord.ndim < 2:
            raise ValueError(
                "coupled FITS -TAB lookup requires multidimensional coordinate arrays"
            )
        if not coord.is_floating_point():
            coord = coord.to(dtype=torch.float64)
        if not bool(torch.isfinite(coord).all().detach().cpu().item()):
            raise ValueError(f"PS{axis_number}_1 column {coord_col!r} is not finite")
        shape = tuple(int(size) for size in coord.shape)
        if grid_shape is None:
            grid_shape = shape
        elif shape != grid_shape:
            raise ValueError(
                f"all coupled -TAB coordinate arrays must share shape {grid_shape}; "
                f"got {shape} for PS{axis_number}_1 column {coord_col!r}"
            )
        if tab_axis_number > len(shape):
            raise ValueError(
                f"PV{axis_number}_3 selects tabular axis {tab_axis_number}, "
                f"but coordinate array has {len(shape)} dimension(s)"
            )

        if index_col_raw is None or str(index_col_raw).strip() == "":
            pixel_axis = torch.arange(shape[tab_axis_number - 1], dtype=torch.float64)
        else:
            index_col = str(index_col_raw).strip()
            pixel_axis = _as_tabular_index_array(
                _read_table_column(table, index_col),
                f"PS{axis_number}_2 column {index_col!r}",
                tab_axis_number,
            )
        existing = pixel_axes_by_tab_axis.get(tab_axis_number)
        if existing is None:
            pixel_axes_by_tab_axis[tab_axis_number] = pixel_axis
        elif not torch.equal(existing, pixel_axis.to(dtype=existing.dtype)):
            raise ValueError(
                f"inconsistent indexing vectors for coupled -TAB table axis "
                f"{tab_axis_number}"
            )
        unit = (
            str(_tab_header_get(header, f"CUNIT{axis_number}", alt_key) or "") or None
        )
        coords.append((axis_number, coord_col, unit, coord, tab_axis_number))

    if not coords:
        raise ValueError("no FITS -TAB axes found")
    if grid_shape is None:
        raise ValueError("no multidimensional FITS -TAB coordinate arrays found")

    missing = [
        axis
        for axis in range(1, len(grid_shape) + 1)
        if axis not in pixel_axes_by_tab_axis
    ]
    if missing:
        for axis in missing:
            pixel_axes_by_tab_axis[axis] = torch.arange(
                grid_shape[axis - 1], dtype=torch.float64
            )

    coords.sort(key=lambda item: item[0])
    world = torch.stack([item[3] for item in coords], dim=-1)
    pixel_axes = tuple(
        pixel_axes_by_tab_axis[index] for index in range(1, len(grid_shape) + 1)
    )
    names = tuple(item[1] for item in coords)
    units = tuple(item[2] for item in coords)
    lookup = CoupledTabularLookupWCS(pixel_axes, world, names=names, units=units)
    world_axes = tuple(item[0] for item in coords)
    pixel_wcs_axes = []
    for tab_dim in range(1, len(grid_shape) + 1):
        owners = [item[0] for item in coords if item[4] == tab_dim]
        pixel_wcs_axes.append(min(owners) if owners else tab_dim)
    return lookup, world_axes, tuple(pixel_wcs_axes)


def coupled_tabular_wcs_from_fits(
    header: Mapping[str, Any],
    tables: Any,
    *,
    key: str = "",
) -> CoupledTabularLookupWCS:
    """Build a forward coupled table-lookup WCS from FITS ``-TAB`` metadata."""
    lookup, _world_axes, _pixel_axes = coupled_tabular_wcs_and_axes_from_fits(
        header, tables, key=key
    )
    return lookup
