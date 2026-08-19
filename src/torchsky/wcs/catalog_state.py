"""Batched catalog geometry. This is not a SkyCoord clone.

Geometric frames and astrometric propagation live here. Observed place
(AltAz / Earth orientation) is :mod:`torchsky.wcs.observed`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from torch import Tensor

from .frames import (
    _to_device,
    _work_dtype,
    canonical_celestial_frame,
    convert_celestial,
    propagate_proper_motion,
    propagate_space_motion,
)

Representation = Literal["lonlat_deg", "unit_xyz", "barycentric_au"]

# IAU: 1 pc is the distance at which 1 au subtends 1 arcsecond.
_AU_PER_PC = 648000.0 / math.pi


def _as_f64(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(dtype=_work_dtype(value.device))
    return torch.as_tensor(value, dtype=torch.float64)


@dataclass
class CatalogState:
    """Batched sky state with explicit missing components.

    Missing proper motion, parallax, or radial velocity stay ``None``. They
    are never filled from a default distance or a ``pmsafe`` override.
    ``valid`` is an optional boolean mask; it is never used to impute values.
    """

    lon: Tensor
    lat: Tensor
    frame: str = "ICRS"
    epoch: Tensor | float = 2000.0
    pm_lon_coslat_mas_yr: Tensor | None = None
    pm_lat_mas_yr: Tensor | None = None
    parallax_mas: Tensor | None = None
    radial_velocity_km_s: Tensor | None = None
    valid: Tensor | None = None

    def __post_init__(self) -> None:
        lon = _as_f64(self.lon)
        lat = _to_device(_as_f64(self.lat), lon.device)
        self.lon, self.lat = torch.broadcast_tensors(lon, lat)
        self.frame = canonical_celestial_frame(self.frame)
        self.epoch = _to_device(_as_f64(self.epoch), self.lon.device)
        if self.pm_lon_coslat_mas_yr is not None:
            self.pm_lon_coslat_mas_yr = _to_device(
                _as_f64(self.pm_lon_coslat_mas_yr), self.lon.device
            )
        if self.pm_lat_mas_yr is not None:
            self.pm_lat_mas_yr = _to_device(
                _as_f64(self.pm_lat_mas_yr), self.lon.device
            )
        if self.parallax_mas is not None:
            self.parallax_mas = _to_device(_as_f64(self.parallax_mas), self.lon.device)
        if self.radial_velocity_km_s is not None:
            self.radial_velocity_km_s = _to_device(
                _as_f64(self.radial_velocity_km_s), self.lon.device
            )
        if self.valid is not None:
            self.valid = torch.as_tensor(
                self.valid, dtype=torch.bool, device=self.lon.device
            )

    def to_representation(self, kind: Representation = "lonlat_deg") -> Tensor:
        if kind == "lonlat_deg":
            return torch.stack((self.lon, self.lat), dim=-1)
        if kind == "unit_xyz":
            lon_r = self.lon * (torch.pi / 180.0)
            lat_r = self.lat * (torch.pi / 180.0)
            cos_lat = torch.cos(lat_r)
            return torch.stack(
                (
                    cos_lat * torch.cos(lon_r),
                    cos_lat * torch.sin(lon_r),
                    torch.sin(lat_r),
                ),
                dim=-1,
            )
        if kind == "barycentric_au":
            if self.parallax_mas is None:
                raise ValueError(
                    "barycentric_au requires parallax_mas; missing parallax is not imputed"
                )
            if not bool(torch.all(self.parallax_mas > 0.0)):
                raise ValueError("parallax_mas must be positive")
            unit = self.to_representation("unit_xyz")
            distance_pc = 1000.0 / self.parallax_mas
            return unit * (distance_pc * _AU_PER_PC).unsqueeze(-1)
        raise ValueError(f"unsupported representation {kind!r}")

    def to_frame(self, frame: str) -> CatalogState:
        lon, lat = convert_celestial(
            self.lon, self.lat, from_frame=self.frame, to_frame=frame
        )
        return replace(self, lon=lon, lat=lat, frame=canonical_celestial_frame(frame))

    def propagate_angular(self, epoch: Tensor | float) -> CatalogState:
        if self.pm_lon_coslat_mas_yr is None or self.pm_lat_mas_yr is None:
            raise ValueError(
                "angular propagation requires pm_lon_coslat_mas_yr and pm_lat_mas_yr"
            )
        lon, lat = propagate_proper_motion(
            self.lon,
            self.lat,
            self.pm_lon_coslat_mas_yr,
            self.pm_lat_mas_yr,
            source_epoch_jyear=self.epoch,
            target_epoch_jyear=epoch,
        )
        return replace(self, lon=lon, lat=lat, epoch=_to_device(_as_f64(epoch), lon.device))

    def propagate_physical(self, epoch: Tensor | float) -> CatalogState:
        if (
            self.pm_lon_coslat_mas_yr is None
            or self.pm_lat_mas_yr is None
            or self.parallax_mas is None
            or self.radial_velocity_km_s is None
        ):
            raise ValueError(
                "physical space motion requires proper motion, parallax, and "
                "radial velocity; missing components are not imputed"
            )
        state = self if self.frame == "ICRS" else self.to_frame("ICRS")
        lon, lat, pm_lon, pm_lat, parallax, rv = propagate_space_motion(
            state.lon,
            state.lat,
            state.pm_lon_coslat_mas_yr,
            state.pm_lat_mas_yr,
            state.parallax_mas,
            state.radial_velocity_km_s,
            source_epoch_jyear=state.epoch,
            target_epoch_jyear=epoch,
        )
        out = replace(
            state,
            lon=lon,
            lat=lat,
            epoch=_to_device(_as_f64(epoch), lon.device),
            pm_lon_coslat_mas_yr=pm_lon,
            pm_lat_mas_yr=pm_lat,
            parallax_mas=parallax,
            radial_velocity_km_s=rv,
        )
        if self.frame == "ICRS":
            return out
        return out.to_frame(self.frame)
