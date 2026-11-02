"""Shared tensor-domain descriptions for transforms and observation operators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class DomainSpec:
    """Immutable logical tensor domain, independent of a tensor allocation."""

    shape: tuple[int, ...]
    dtype: torch.dtype | None = None
    axes: tuple[str, ...] = ()
    units: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        shape = tuple(int(size) for size in self.shape)
        if any(size < 0 for size in shape):
            raise ValueError("domain dimensions must be non-negative")
        if self.axes and len(self.axes) != len(shape):
            raise ValueError("domain axes must match the number of dimensions")
        if self.units and len(self.units) != len(shape):
            raise ValueError("domain units must match the number of dimensions")
        if any(not isinstance(axis, str) or not axis.strip() for axis in self.axes):
            raise ValueError("domain axis names must be nonempty strings")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "units", tuple(self.units))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    def validate(self, tensor: Tensor, *, name: str = "tensor") -> Tensor:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tuple(tensor.shape) != self.shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match {self.shape}"
            )
        if self.dtype is not None and tensor.dtype != self.dtype:
            raise TypeError(f"{name} dtype {tensor.dtype} does not match {self.dtype}")
        return tensor

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        *,
        axes: tuple[str, ...] = (),
        units: tuple[str | None, ...] = (),
    ) -> DomainSpec:
        if not isinstance(tensor, Tensor):
            raise TypeError("tensor must be a tensor")
        return cls(tuple(tensor.shape), tensor.dtype, axes, units)

    @classmethod
    def from_wcs(
        cls,
        wcs: object,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
    ) -> DomainSpec:
        pixel_shape = shape
        if pixel_shape is None:
            candidate = getattr(wcs, "pixel_shape", None)
            if candidate is None:
                candidate = getattr(wcs, "_pixel_shape", None)
            if candidate is not None:
                pixel_shape = tuple(reversed(tuple(int(size) for size in candidate)))
        if pixel_shape is None:
            raise ValueError("a WCS domain requires a declared pixel shape")
        ndim = len(pixel_shape)
        world_names = tuple(getattr(wcs, "world_axis_names", ()))
        world_units = tuple(getattr(wcs, "world_axis_units", ()))
        axes = tuple(reversed(world_names[:ndim])) if len(world_names) >= ndim else ()
        units = tuple(reversed(world_units[:ndim])) if len(world_units) >= ndim else ()
        return cls(tuple(pixel_shape), dtype, axes, units)


def as_domain_spec(
    value: DomainSpec | Tensor | object,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
) -> DomainSpec:
    if isinstance(value, DomainSpec):
        if shape is not None or dtype is not None:
            raise ValueError("shape and dtype cannot override an existing DomainSpec")
        return value
    if isinstance(value, Tensor):
        if shape is not None or dtype is not None:
            raise ValueError("shape and dtype cannot override a tensor domain")
        return DomainSpec.from_tensor(value)
    if isinstance(value, Mapping):
        mapping: Mapping[str, Any] = value
        mapped_shape = shape if shape is not None else mapping.get("shape")
        if mapped_shape is None:
            raise ValueError("domain mapping requires shape")
        return DomainSpec(
            tuple(mapped_shape),
            dtype if dtype is not None else mapping.get("dtype"),
            tuple(mapping.get("axes", ())),
            tuple(mapping.get("units", ())),
        )
    return DomainSpec.from_wcs(value, shape=shape, dtype=dtype)


__all__ = ["DomainSpec", "as_domain_spec"]
