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


def _interp1d(query: Tensor, xp: Tensor, fp: Tensor, *, bounds: TabularBoundsMode) -> Tensor:
    if bounds not in {"nan", "clamp"}:
        raise ValueError("bounds must be 'nan' or 'clamp'")

    dtype = torch.promote_types(torch.promote_types(query.dtype, xp.dtype), fp.dtype)
    device = query.device
    x = query.to(dtype=dtype)
    grid = xp.to(device=device, dtype=dtype)
    values = fp.to(device=device, dtype=dtype)

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

    @property
    def world_direction(self) -> int:
        """Return ``1`` for increasing world values, ``-1`` for decreasing."""
        return _strict_direction(self.world, "world")

    def to(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "TabularAxis":
        """Return a copy with tables moved to ``device`` and/or ``dtype``."""
        pixel = self.pixel.to(device=device, dtype=dtype)
        world = self.world.to(device=device, dtype=dtype)
        return TabularAxis(pixel, world, name=self.name, unit=self.unit, bounds=self.bounds)

    def pixel_to_world(self, pixel: Any, *, bounds: TabularBoundsMode | None = None) -> Tensor:
        """Interpolate world coordinates from pixel/index coordinates."""
        query = torch.as_tensor(pixel)
        if not query.is_floating_point():
            query = query.to(dtype=self.pixel.dtype)
        return _interp1d(query, self.pixel, self.world, bounds=bounds or self.bounds)

    def world_to_pixel(self, world: Any, *, bounds: TabularBoundsMode | None = None) -> Tensor:
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

    This is the torch-native execution primitive for independent lookup axes.
    Full FITS ``-TAB`` parsing and coupled multidimensional coordinate arrays
    can be layered on top without changing the interpolation substrate.
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

    def to(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "TabularLookupWCS":
        return TabularLookupWCS(axis.to(device=device, dtype=dtype) for axis in self.axes)

    def pixel_to_world(self, pixels: Any, *, bounds: TabularBoundsMode | None = None) -> Tensor:
        pts = torch.as_tensor(pixels)
        if pts.shape[-1] != len(self.axes):
            raise ValueError(f"pixels last dimension must be {len(self.axes)}")
        values = [
            axis.pixel_to_world(pts[..., index], bounds=bounds)
            for index, axis in enumerate(self.axes)
        ]
        return torch.stack(values, dim=-1)

    def world_to_pixel(self, world: Any, *, bounds: TabularBoundsMode | None = None) -> Tensor:
        pts = torch.as_tensor(world)
        if pts.shape[-1] != len(self.axes):
            raise ValueError(f"world last dimension must be {len(self.axes)}")
        values = [
            axis.world_to_pixel(pts[..., index], bounds=bounds)
            for index, axis in enumerate(self.axes)
        ]
        return torch.stack(values, dim=-1)


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


def _read_table_column(table: Any, column: str) -> Any:
    if isinstance(table, Mapping):
        if column in table:
            return table[column]
        raise KeyError(f"table column {column!r} not found")
    if hasattr(table, "data"):
        data = table.data
        try:
            return data[column]
        except (KeyError, TypeError):
            pass
    if hasattr(table, "get_column"):
        return table.get_column(column)
    try:
        return table[column]
    except Exception as exc:  # pragma: no cover - adapter fallback
        raise KeyError(f"table column {column!r} not found") from exc


def _resolve_tab_table(tables: Any, extname: str | None) -> Any:
    if extname is None or extname == "":
        return tables
    if isinstance(tables, Mapping) and extname in tables:
        return tables[extname]
    raise KeyError(f"table extension {extname!r} not found")


def _as_independent_tab_array(value: Any, name: str) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim > 1:
        tensor = torch.squeeze(tensor)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must describe one independent tabular axis")
    return tensor


def tabular_axes_from_fits(
    header: Mapping[str, Any],
    tables: Any,
    *,
    key: str = "",
) -> dict[int, TabularAxis]:
    """Build explicit ``WCS(..., tabular_axes=...)`` axes from FITS ``-TAB`` metadata.

    This implements the independent one-dimensional subset of FITS table lookup
    WCS. For each ``CTYPEi`` ending in ``-TAB``, ``PSi_0`` selects a table by
    ``EXTNAME`` when ``tables`` is a mapping, ``PSi_1`` names the coordinate-array
    column, and optional ``PSi_2`` names the indexing-vector column. If no
    indexing vector is present, samples are indexed by ``0..N-1``.
    """
    alt_key = "" if key in {"", " "} else str(key).strip().upper()
    if alt_key and (len(alt_key) != 1 or not ("A" <= alt_key <= "Z")):
        raise ValueError("alternate WCS key must be '', ' ', or one letter A-Z")

    axes: dict[int, TabularAxis] = {}
    for axis_number in range(1, _tab_header_naxis(header, alt_key) + 1):
        ctype = str(_tab_header_get(header, f"CTYPE{axis_number}", alt_key) or "").strip().upper()
        if not ctype.endswith("-TAB"):
            continue
        extname_raw = _tab_header_get(header, f"PS{axis_number}_0", alt_key)
        coord_col_raw = _tab_header_get(header, f"PS{axis_number}_1", alt_key)
        index_col_raw = _tab_header_get(header, f"PS{axis_number}_2", alt_key)
        if coord_col_raw is None:
            raise ValueError(f"CTYPE{axis_number} is '-TAB' but PS{axis_number}_1 is missing")

        table = _resolve_tab_table(tables, None if extname_raw is None else str(extname_raw).strip())
        coord_col = str(coord_col_raw).strip()
        coord = _as_independent_tab_array(
            _read_table_column(table, coord_col),
            f"PS{axis_number}_1 column {coord_col!r}",
        )
        if index_col_raw is None or str(index_col_raw).strip() == "":
            pixel = torch.arange(coord.shape[0], dtype=torch.float64)
        else:
            index_col = str(index_col_raw).strip()
            pixel = _as_independent_tab_array(
                _read_table_column(table, index_col),
                f"PS{axis_number}_2 column {index_col!r}",
            )
        axes[axis_number] = TabularAxis(
            pixel,
            coord,
            name=coord_col,
            unit=str(_tab_header_get(header, f"CUNIT{axis_number}", alt_key) or "") or None,
        )
    return axes
