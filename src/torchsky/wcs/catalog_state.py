"""Batched catalog geometry. This is not a SkyCoord clone.

Geometric frames and astrometric propagation live here. Observed place
(AltAz / Earth orientation) is :mod:`torchsky.wcs.observed`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor

from .._dtype import as_work_tensor as _as_f64
from .frames import (
    _linear_angular_motion,
    _to_device,
    canonical_celestial_frame,
    convert_celestial,
    propagate_space_motion,
    radec_to_unit,
)
from .semantics import Epoch, TimeFormat, TimeScale, TimeSpec
from ..device import _as_tensor

Representation = Literal["lonlat_deg", "unit_xyz", "barycentric_au"]

# IAU: 1 pc is the distance at which 1 au subtends 1 arcsecond.
_AU_PER_PC = 648000.0 / math.pi


def _tangent_basis(lon: Tensor, lat: Tensor) -> tuple[Tensor, Tensor]:
    lon, lat = torch.broadcast_tensors(
        lon * (torch.pi / 180.0), lat * (torch.pi / 180.0)
    )
    east = torch.stack((-torch.sin(lon), torch.cos(lon), torch.zeros_like(lon)), dim=-1)
    north = torch.stack(
        (
            -torch.sin(lat) * torch.cos(lon),
            -torch.sin(lat) * torch.sin(lon),
            torch.cos(lat),
        ),
        dim=-1,
    )
    return east, north


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
    epoch: Tensor | float | Epoch = 2000.0
    epoch_scale: TimeScale | None = None
    epoch_format: TimeFormat | None = None
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
        if isinstance(self.epoch, Epoch):
            epoch = self.epoch
        else:
            epoch = Epoch(
                self.epoch,
                TimeSpec(
                    format="jyear" if self.epoch_format is None else self.epoch_format,
                    scale="tcb" if self.epoch_scale is None else self.epoch_scale,
                ),
            )
        self.epoch = _to_device(_as_f64(epoch.value), self.lon.device)
        self.epoch_scale = epoch.spec.scale
        self.epoch_format = epoch.spec.format
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
            self.valid = _as_tensor(
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

    def to_frame(
        self, frame: str, *, tt_mjd: Tensor | float | None = None
    ) -> CatalogState:
        destination = canonical_celestial_frame(frame)
        if destination == self.frame:
            return replace(self)
        pm_lon, pm_lat = self.pm_lon_coslat_mas_yr, self.pm_lat_mas_yr
        if (pm_lon is None) != (pm_lat is None):
            raise ValueError("frame conversion requires both proper-motion components")
        if pm_lon is not None and "FK4" in (self.frame, destination):
            raise ValueError(
                "FK4 proper-motion conversion requires an unsupported time-dependent astrometric transform"
            )
        lon, lat = convert_celestial(
            self.lon,
            self.lat,
            from_frame=self.frame,
            to_frame=destination,
            tt_mjd=tt_mjd,
        )
        if pm_lon is not None and pm_lat is not None:
            # The supported non-FK4 conversions are rigid Cartesian rotations.
            axes_lon, axes_lat = convert_celestial(
                self.lon.new_tensor([0.0, 90.0, 0.0]),
                self.lon.new_tensor([0.0, 0.0, 90.0]),
                from_frame=self.frame,
                to_frame=destination,
                tt_mjd=tt_mjd,
            )
            rotation = radec_to_unit(axes_lon, axes_lat)
            east, north = _tangent_basis(self.lon, self.lat)
            velocity = pm_lon.unsqueeze(-1) * east + pm_lat.unsqueeze(-1) * north
            rotated = velocity @ rotation
            east_out, north_out = _tangent_basis(lon, lat)
            pm_lon = (rotated * east_out).sum(dim=-1)
            pm_lat = (rotated * north_out).sum(dim=-1)
        return replace(
            self,
            lon=lon,
            lat=lat,
            frame=destination,
            pm_lon_coslat_mas_yr=pm_lon,
            pm_lat_mas_yr=pm_lat,
        )

    def _target_epoch_value(self, epoch: Tensor | float | Epoch) -> Tensor:
        if isinstance(epoch, Epoch):
            if self.epoch_scale is None or self.epoch_format is None:
                raise RuntimeError("catalog epoch metadata was not initialized")
            epoch = epoch.to(
                TimeSpec(format=self.epoch_format, scale=self.epoch_scale)
            ).value
        return _to_device(_as_f64(epoch), self.lon.device)

    def propagate_angular(self, epoch: Tensor | float | Epoch) -> CatalogState:
        """Linear unit-distance motion; proper motion is carried to the new epoch.

        Rows whose epoch changes must have finite proper motion (missing motion
        is never treated as zero). Rows already at ``epoch`` are unchanged.
        """
        if self.pm_lon_coslat_mas_yr is None or self.pm_lat_mas_yr is None:
            raise ValueError(
                "angular propagation requires pm_lon_coslat_mas_yr and pm_lat_mas_yr"
            )
        if self.epoch_format != "jyear":
            raise ValueError("angular propagation requires jyear epochs")
        target_epoch = self._target_epoch_value(epoch)
        source = torch.as_tensor(self.epoch)
        dt = torch.broadcast_to(target_epoch - source, self.lon.shape)
        pm_lon, pm_lat = self.pm_lon_coslat_mas_yr, self.pm_lat_mas_yr
        finite = torch.isfinite(pm_lon) & torch.isfinite(pm_lat)
        moving = dt != 0
        missing = moving & ~finite
        if bool(missing.any()):
            raise ValueError(
                f"{int(missing.sum())} row(s) change epoch without finite proper "
                "motion; drop them, or use CatalogState.propagate with a motion prior"
            )
        lon, lat, pm_lon_out, pm_lat_out = _linear_angular_motion(
            self.lon,
            self.lat,
            torch.where(finite, pm_lon, torch.zeros_like(pm_lon)),
            torch.where(finite, pm_lat, torch.zeros_like(pm_lat)),
            dt,
        )
        return replace(
            self,
            lon=lon,
            lat=lat,
            epoch=target_epoch,
            pm_lon_coslat_mas_yr=torch.where(finite, pm_lon_out, pm_lon),
            pm_lat_mas_yr=torch.where(finite, pm_lat_out, pm_lat),
        )

    def motion_level(self) -> Tensor:
        """Per-row astrometric level: 6 (pm, parallax > 0, RV), 4 (pm), 2 (none).

        Level 6 requires finite proper motion, positive finite parallax, and a
        finite radial velocity. Level 4 only needs finite proper motion; its
        parallax is ignored, even when present but non-positive. Nothing is imputed.
        """
        n = self.lon.shape
        level = torch.full(n, 2, dtype=torch.int8, device=self.lon.device)
        if self.pm_lon_coslat_mas_yr is None or self.pm_lat_mas_yr is None:
            return level
        pm = torch.isfinite(self.pm_lon_coslat_mas_yr) & torch.isfinite(
            self.pm_lat_mas_yr
        )
        level = torch.where(pm, torch.full_like(level, 4), level)
        if self.parallax_mas is not None and self.radial_velocity_km_s is not None:
            full = (
                pm
                & torch.isfinite(self.parallax_mas)
                & (self.parallax_mas > 0.0)
                & torch.isfinite(self.radial_velocity_km_s)
            )
            level = torch.where(full, torch.full_like(level, 6), level)
        return level

    def propagate(
        self,
        epoch: Tensor | float | Epoch,
        *,
        prior_pm_mas_yr: tuple[Tensor, Tensor] | None = None,
    ) -> CatalogState:
        """Per-row propagation with the most complete motion each row has.

        - Level 6 rows (see :meth:`motion_level`) use full space motion
          (:meth:`propagate_physical`, ERFA ``starpv``/``pvstar``).
        - Level 4 rows use linear unit-distance proper motion
          (:meth:`propagate_angular`).
        - Level 2 rows (no proper motion) move with ``prior_pm_mas_yr``, a
          per-row (east, north) prior mean motion, for example from
          ``torchsky.catalogs.MotionPrior``. Without a prior they must not
          change epoch.

        Parallax is never imputed and missing motion is never treated as zero.
        Proper motion is carried to the new epoch; level-2 rows keep NaN proper
        motion, because the prior is not a measurement.
        """
        if self.epoch_format != "jyear":
            raise ValueError("propagation requires jyear epochs")
        target_epoch = self._target_epoch_value(epoch)
        shape = self.lon.shape
        dt = torch.broadcast_to(target_epoch - torch.as_tensor(self.epoch), shape)
        level = self.motion_level()
        nan = torch.full(shape, math.nan, dtype=self.lon.dtype, device=self.lon.device)
        pm_lon = (
            self.pm_lon_coslat_mas_yr if self.pm_lon_coslat_mas_yr is not None else nan
        )
        pm_lat = self.pm_lat_mas_yr if self.pm_lat_mas_yr is not None else nan
        no_pm = level == 2
        if bool((no_pm & (dt != 0)).any()):
            if prior_pm_mas_yr is None:
                raise ValueError(
                    f"{int((no_pm & (dt != 0)).sum())} row(s) without proper motion "
                    "change epoch; pass prior_pm_mas_yr (torchsky.catalogs.MotionPrior)"
                )
            prior_lon = torch.broadcast_to(
                _as_f64(prior_pm_mas_yr[0]).to(self.lon.device), shape
            )
            prior_lat = torch.broadcast_to(
                _as_f64(prior_pm_mas_yr[1]).to(self.lon.device), shape
            )
            if not bool(
                torch.isfinite(prior_lon[no_pm]).all()
                & torch.isfinite(prior_lat[no_pm]).all()
            ):
                raise ValueError(
                    "prior_pm_mas_yr must be finite for rows without proper motion"
                )
        else:
            prior_lon = prior_lat = torch.zeros_like(nan)
        # Level 2 and 4: linear angular motion (prior mean for level 2).
        use_lon = torch.where(no_pm, prior_lon, pm_lon)
        use_lat = torch.where(no_pm, prior_lat, pm_lat)
        safe = torch.isfinite(use_lon) & torch.isfinite(use_lat)
        lon, lat, pm_lon_out, pm_lat_out = _linear_angular_motion(
            self.lon,
            self.lat,
            torch.where(safe, use_lon, torch.zeros_like(use_lon)),
            torch.where(safe, use_lat, torch.zeros_like(use_lat)),
            dt,
        )
        pm_lon_out = torch.where(no_pm, pm_lon, pm_lon_out)
        pm_lat_out = torch.where(no_pm, pm_lat, pm_lat_out)
        plx, rv = self.parallax_mas, self.radial_velocity_km_s
        full = level == 6
        if bool(full.any()):
            rows = torch.nonzero(full, as_tuple=True)[0]
            sub = CatalogState(
                lon=self.lon[rows],
                lat=self.lat[rows],
                frame=self.frame,
                epoch=Epoch(
                    torch.broadcast_to(torch.as_tensor(self.epoch), shape)[rows],
                    TimeSpec(format="jyear", scale=self.epoch_scale or "tcb"),
                ),
                pm_lon_coslat_mas_yr=pm_lon[rows],
                pm_lat_mas_yr=pm_lat[rows],
                parallax_mas=plx[rows] if plx is not None else None,
                radial_velocity_km_s=rv[rows] if rv is not None else None,
            )
            target_rows = torch.broadcast_to(target_epoch, shape)[rows]
            moved = sub.propagate_physical(target_rows)
            lon, lat = lon.clone(), lat.clone()
            pm_lon_out, pm_lat_out = pm_lon_out.clone(), pm_lat_out.clone()
            lon[rows], lat[rows] = moved.lon, moved.lat
            assert moved.pm_lon_coslat_mas_yr is not None
            assert moved.pm_lat_mas_yr is not None
            assert moved.parallax_mas is not None
            assert moved.radial_velocity_km_s is not None
            pm_lon_out[rows] = moved.pm_lon_coslat_mas_yr
            pm_lat_out[rows] = moved.pm_lat_mas_yr
            if plx is not None and rv is not None:
                plx, rv = plx.clone(), rv.clone()
                plx[rows] = moved.parallax_mas
                rv[rows] = moved.radial_velocity_km_s
        keep = dt == 0
        return replace(
            self,
            lon=torch.where(keep, self.lon, lon),
            lat=torch.where(keep, self.lat, lat),
            epoch=target_epoch,
            pm_lon_coslat_mas_yr=None
            if self.pm_lon_coslat_mas_yr is None
            else torch.where(keep, pm_lon, pm_lon_out),
            pm_lat_mas_yr=None
            if self.pm_lat_mas_yr is None
            else torch.where(keep, pm_lat, pm_lat_out),
            parallax_mas=plx,
            radial_velocity_km_s=rv,
        )

    def propagate_physical(self, epoch: Tensor | float | Epoch) -> CatalogState:
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
        if self.epoch_scale != "tcb" or self.epoch_format != "jyear":
            raise ValueError("physical space motion requires TCB jyear epochs")
        target_epoch = self._target_epoch_value(epoch)
        state = self if self.frame == "ICRS" else self.to_frame("ICRS")
        lon, lat, pm_lon, pm_lat, parallax, rv = propagate_space_motion(
            state.lon,
            state.lat,
            state.pm_lon_coslat_mas_yr,
            state.pm_lat_mas_yr,
            state.parallax_mas,
            state.radial_velocity_km_s,
            source_epoch_jyear=state.epoch,
            target_epoch_jyear=target_epoch,
        )
        out = replace(
            state,
            lon=lon,
            lat=lat,
            epoch=target_epoch,
            pm_lon_coslat_mas_yr=pm_lon,
            pm_lat_mas_yr=pm_lat,
            parallax_mas=parallax,
            radial_velocity_km_s=rv,
        )
        if self.frame == "ICRS":
            return out
        return out.to_frame(self.frame)
