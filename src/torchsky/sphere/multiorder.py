from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from .._dtype import work_dtype as _work_dtype
from . import healpix as _healpix
from .moc import MOC, _order_from_uniq, nest2uniq, uniq2nest
from .sparse import SparseHealpixMap


@runtime_checkable
class UniqValueMapProtocol(Protocol):
    @property
    def uniq(self) -> Any: ...

    @property
    def data(self) -> Any: ...


def _validate_disjoint_uniq(uniq: Tensor) -> None:
    if uniq.numel() < 2:
        return
    orders = _order_from_uniq(uniq)
    max_order = int(orders.max().item())
    _, pixels = uniq2nest(uniq)
    shifts = 2 * (max_order - orders)
    starts = pixels << shifts
    ends = (pixels + 1) << shifts
    order = torch.argsort(starts)
    starts = starts[order]
    ends = ends[order]
    if bool((starts[1:] < ends[:-1]).any()):
        raise ValueError("uniq cells must be unique and non-overlapping")


class MultiOrderHealpixMap:
    """Tensor values attached to sorted, disjoint IVOA NUNIQ cells."""

    def __init__(
        self,
        uniq: Tensor | Any,
        values: Tensor | Any,
        *,
        values_are_densities: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        values_t = torch.as_tensor(values)
        if values_t.ndim == 0:
            raise ValueError("values must have at least one dimension")
        uniq_t = torch.as_tensor(uniq)
        if uniq_t.dtype == torch.bool or uniq_t.dtype.is_floating_point or uniq_t.dtype.is_complex:
            raise ValueError("uniq must contain integers")
        uniq_t = uniq_t.to(device=values_t.device, dtype=torch.int64).reshape(-1)
        if values_t.shape[-1] != uniq_t.numel():
            raise ValueError("the last values dimension must match uniq")
        _order_from_uniq(uniq_t)
        if uniq_t.numel() > 0:
            order = torch.argsort(uniq_t, stable=True)
            uniq_t = uniq_t.index_select(0, order)
            values_t = values_t.index_select(-1, order)
            if bool((uniq_t[1:] == uniq_t[:-1]).any()):
                raise ValueError("uniq cells must be unique and non-overlapping")
        _validate_disjoint_uniq(uniq_t)
        self.uniq = uniq_t
        self.values = values_t
        self.values_are_densities = bool(values_are_densities)
        self.metadata = dict(metadata) if metadata is not None else None

    @property
    def max_order(self) -> int:
        if self.uniq.numel() == 0:
            return 0
        return int(_order_from_uniq(self.uniq).max().item())

    @property
    def coverage_moc(self) -> MOC:
        return MOC(self.uniq)

    def to_moc(self) -> MOC:
        return self.coverage_moc

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> MultiOrderHealpixMap:
        return type(self)(
            self.uniq.to(device=device),
            self.values.to(device=device, dtype=dtype),
            values_are_densities=self.values_are_densities,
            metadata=self.metadata,
        )

    def cell_area(self) -> Tensor:
        nsides, _ = uniq2nest(self.uniq)
        nsides_f = nsides.to(_work_dtype(self.uniq.device))
        return (4.0 * math.pi) / (12.0 * nsides_f * nsides_f)

    def overlap_fraction(self, footprint: MOC) -> Tensor:
        return footprint.overlap_fraction(self.uniq)

    def integral(self, footprint: MOC | None = None) -> Tensor:
        weights = torch.ones(
            self.uniq.shape,
            dtype=_work_dtype(self.uniq.device),
            device=self.uniq.device,
        )
        if footprint is not None:
            weights = footprint.overlap_fraction(self.uniq)
        if self.values_are_densities:
            weights = weights * self.cell_area()
        value_dtype = (
            self.values.dtype
            if self.values.dtype.is_floating_point or self.values.dtype.is_complex
            else weights.dtype
        )
        return (
            self.values.to(dtype=value_dtype) * weights.to(dtype=value_dtype)
        ).sum(dim=-1)

    def select(self, footprint: MOC) -> MultiOrderHealpixMap:
        keep = footprint.intersects_uniq(self.uniq)
        return type(self)(
            self.uniq[keep],
            self.values[..., keep],
            values_are_densities=self.values_are_densities,
            metadata=self.metadata,
        )

    def get_values_pos(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        fill_value: float | int = 0,
        return_valid: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        lon = torch.as_tensor(lon_deg, device=self.uniq.device)
        dtype = _work_dtype(self.uniq.device)
        lon = lon.to(dtype=dtype)
        lat = torch.as_tensor(lat_deg, device=self.uniq.device, dtype=dtype)
        lon, lat = torch.broadcast_tensors(lon, lat)
        shape = lon.shape
        count = lon.numel()
        out_shape = tuple(self.values.shape[:-1]) + shape
        if self.uniq.numel() == 0:
            output = torch.full(
                out_shape,
                fill_value,
                dtype=self.values.dtype,
                device=self.values.device,
            )
            valid = torch.zeros(shape, dtype=torch.bool, device=self.uniq.device)
            return (output, valid) if return_valid else output

        max_order = self.max_order
        pixels = _healpix.ang2pix(
            1 << max_order, lon, lat, nest=True, lonlat=True
        ).to(torch.int64).reshape(-1)
        matched = torch.zeros(count, dtype=torch.int64, device=self.uniq.device)
        valid = torch.zeros(count, dtype=torch.bool, device=self.uniq.device)
        n_cells = self.uniq.numel()
        for order in range(max_order + 1):
            parent = pixels >> (2 * (max_order - order))
            candidate = nest2uniq(1 << order, parent)
            index = torch.searchsorted(self.uniq, candidate)
            safe = index.clamp(max=n_cells - 1)
            found = (index < n_cells) & (self.uniq[safe] == candidate)
            matched = torch.where(found, safe, matched)
            valid |= found
        gathered = self.values.index_select(-1, matched)
        fill = torch.full(
            gathered.shape,
            fill_value,
            dtype=self.values.dtype,
            device=self.values.device,
        )
        output = torch.where(valid, gathered, fill).reshape(out_shape)
        valid = valid.reshape(shape)
        return (output, valid) if return_valid else output

    @classmethod
    def from_columns(
        cls,
        columns: Mapping[str, Any] | Any,
        *,
        value_column: str,
        uniq_column: str = "UNIQ",
        values_are_densities: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> MultiOrderHealpixMap:
        return cls(
            columns[uniq_column],
            columns[value_column],
            values_are_densities=values_are_densities,
            metadata=metadata,
        )

    def to_columns(self, *, value_column: str = "VALUE") -> dict[str, Tensor]:
        if self.values.ndim != 1:
            raise ValueError("column interchange requires one-dimensional values")
        return {"UNIQ": self.uniq, value_column: self.values}

    @classmethod
    def from_sparse_map(cls, sparse_map: SparseHealpixMap) -> MultiOrderHealpixMap:
        pixels = sparse_map.pixels
        if not sparse_map.nest and pixels.numel() > 0:
            pixels = _healpix.ring2nest(sparse_map.nside, pixels)
        return cls(
            nest2uniq(sparse_map.nside, pixels),
            sparse_map.values,
            metadata=sparse_map.metadata,
        )

    def to_sparse_map(
        self,
        *,
        nside_coverage: int | None = None,
        fill_value: float = float(_healpix.UNSEEN),
    ) -> SparseHealpixMap:
        nsides, pixels = uniq2nest(self.uniq)
        if nsides.numel() == 0:
            nside = 1
        else:
            nside = int(nsides[0].item())
            if bool((nsides != nside).any()):
                raise ValueError("fixed-resolution sparse interchange requires one order")
        return SparseHealpixMap.from_pixels(
            pixels,
            self.values,
            nside,
            nest=True,
            nside_coverage=nside_coverage,
            sentinel=fill_value,
            metadata=self.metadata,
        )

    @classmethod
    def from_mhealpy(cls, map_obj: UniqValueMapProtocol) -> MultiOrderHealpixMap:
        density = bool(getattr(map_obj, "density", getattr(map_obj, "_density", False)))
        metadata: dict[str, Any] = {}
        for name in ("coordsys", "unit"):
            value = getattr(map_obj, name, None)
            if value is not None:
                metadata[name] = str(value)
        return cls(
            map_obj.uniq,
            map_obj.data,
            values_are_densities=density,
            metadata=metadata or None,
        )

    def to_mhealpy(self) -> Any:
        try:
            from mhealpy import HealpixMap
        except ImportError as exc:
            raise ImportError("mhealpy is required for to_mhealpy()") from exc

        return HealpixMap(
            data=self.values.detach().cpu().numpy(),
            uniq=self.uniq.detach().cpu().numpy(),
            density=self.values_are_densities,
            coordsys=None if self.metadata is None else self.metadata.get("coordsys"),
            unit=None if self.metadata is None else self.metadata.get("unit"),
        )

    @classmethod
    def from_cdshealpix(cls, map_obj: Any) -> MultiOrderHealpixMap:
        order = int(map_obj.order)
        return cls(nest2uniq(1 << order, map_obj.keys), map_obj.values)

    def to_cdshealpix(self) -> Any:
        try:
            from cdshealpix.skymap import SkymapExplicit
        except ImportError as exc:
            raise ImportError("cdshealpix is required for to_cdshealpix()") from exc

        if self.values.ndim != 1:
            raise ValueError("cdshealpix interchange requires one-dimensional values")
        nsides, pixels = uniq2nest(self.uniq)
        if nsides.numel() == 0:
            order = 0
        else:
            nside = int(nsides[0].item())
            if bool((nsides != nside).any()):
                raise ValueError("cdshealpix explicit maps require one order")
            order = nside.bit_length() - 1
        return SkymapExplicit(
            pixels.detach().cpu().numpy(),
            self.values.detach().cpu().numpy(),
            order,
        )


__all__ = ["MultiOrderHealpixMap", "UniqValueMapProtocol"]
