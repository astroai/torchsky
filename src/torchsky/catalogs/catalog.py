"""Batched sky catalog. This is not a SkyCoord clone."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..wcs.catalog_state import CatalogState
from ..wcs.coords import transform
from ..wcs.frames import _to_device, _work_dtype, canonical_celestial_frame
from ..wcs.observed import EarthOrientation, PNModel, Site
from .core import CrossmatchResult, crossmatch_sky
from .index import HealpixIndex

_OBSERVED_FRAMES = {"ALTAZ", "ITRS", "GCRS", "CIRS", "TETE"}


def _normalize_frame(frame: str) -> str:
    name = str(frame).strip().upper().replace(" ", "")
    if name in {"HORIZON"}:
        return "ALTAZ"
    if name in _OBSERVED_FRAMES:
        return name
    return canonical_celestial_frame(frame)


def _as_rows(value: Tensor | float, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return _to_device(value, device).reshape(-1)
    return torch.as_tensor(value, dtype=_work_dtype(device), device=device).reshape(-1)


def _move_column(col: Tensor, device: torch.device) -> Tensor:
    if device.type == "mps" and col.dtype == torch.float64:
        return col.to(device=device, dtype=torch.float32)
    return col.to(device=device)


@dataclass
class Catalog:
    """Batched table of sky rows on one device.

    Required sky columns are ``lon`` / ``lat`` in degrees. Missing kinematics
    stay ``None``. Extra payload lives in ``columns`` (same length, same device).

    Args:
        lon: Longitude tensor in degrees.
        lat: Latitude tensor in degrees, with the same row count as ``lon``.
        frame: Celestial or observed coordinate-frame name.
        epoch: Scalar or per-row Julian epoch.
        pm_lon_coslat_mas_yr: Cosine-weighted longitude proper motion in mas/yr.
        pm_lat_mas_yr: Latitude proper motion in mas/yr.
        parallax_mas: Parallax in milliarcseconds.
        radial_velocity_km_s: Radial velocity in km/s.
        valid: Optional per-row validity mask.
        columns: Additional tensors whose leading dimension matches the catalog.

    Examples:
        >>> import torch
        >>> from torchsky.catalogs import Catalog
        >>> catalog = Catalog(torch.tensor([10.0]), torch.tensor([-2.0]))
        >>> len(catalog)
        1
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
    columns: dict[str, Tensor] = field(default_factory=dict)
    _index: HealpixIndex | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        lon = torch.as_tensor(self.lon)
        device = lon.device
        self.lon = _as_rows(lon, device)
        self.lat = _as_rows(self.lat, device)
        if self.lon.numel() != self.lat.numel():
            raise ValueError("lon and lat must have the same number of rows")
        self.frame = _normalize_frame(self.frame)
        epoch_t = (
            _to_device(self.epoch, device)
            if isinstance(self.epoch, Tensor)
            else torch.as_tensor(self.epoch, dtype=_work_dtype(device), device=device)
        )
        if epoch_t.ndim == 0 or epoch_t.numel() == 1:
            self.epoch = epoch_t.reshape(())
        else:
            self.epoch = _as_rows(epoch_t, device)
            if int(self.epoch.numel()) != int(self.lon.numel()):
                raise ValueError("epoch must be scalar or match the number of rows")
        n = int(self.lon.numel())
        self.pm_lon_coslat_mas_yr = self._optional_rows(
            self.pm_lon_coslat_mas_yr, n, device
        )
        self.pm_lat_mas_yr = self._optional_rows(self.pm_lat_mas_yr, n, device)
        self.parallax_mas = self._optional_rows(self.parallax_mas, n, device)
        self.radial_velocity_km_s = self._optional_rows(
            self.radial_velocity_km_s, n, device
        )
        if self.valid is not None:
            valid = torch.as_tensor(
                self.valid, dtype=torch.bool, device=device
            ).reshape(-1)
            if int(valid.numel()) != n:
                raise ValueError("valid must match the number of rows")
            self.valid = valid
        extras: dict[str, Tensor] = {}
        for name, col in self.columns.items():
            tensor = _move_column(torch.as_tensor(col), device)
            if tensor.ndim == 0 or int(tensor.shape[0]) != n:
                raise ValueError(f"columns[{name!r}] must match the number of rows")
            extras[str(name)] = tensor
        self.columns = extras
        self._index = None

    @staticmethod
    def _optional_rows(
        value: Tensor | None, n: int, device: torch.device
    ) -> Tensor | None:
        if value is None:
            return None
        rows = _as_rows(value, device)
        if int(rows.numel()) != n:
            raise ValueError("optional sky columns must match the number of rows")
        return rows

    def __len__(self) -> int:
        return int(self.lon.numel())

    @property
    def device(self) -> torch.device:
        return self.lon.device

    def to(self, device: torch.device | str) -> Catalog:
        """Return a copy on ``device``; ``__post_init__`` re-normalises dtypes."""
        device_t = torch.device(device)

        def _move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(device=device_t)

        epoch = (
            self.epoch.to(device=device_t)
            if isinstance(self.epoch, Tensor)
            else self.epoch
        )
        return Catalog(
            lon=self.lon.to(device=device_t),
            lat=self.lat.to(device=device_t),
            frame=self.frame,
            epoch=epoch,
            pm_lon_coslat_mas_yr=_move(self.pm_lon_coslat_mas_yr),
            pm_lat_mas_yr=_move(self.pm_lat_mas_yr),
            parallax_mas=_move(self.parallax_mas),
            radial_velocity_km_s=_move(self.radial_velocity_km_s),
            valid=_move(self.valid),
            columns={
                name: col.to(device=device_t) for name, col in self.columns.items()
            },
        )

    @property
    def state(self) -> CatalogState:
        """Return this catalog's celestial kinematic state as a :class:`CatalogState`."""
        return self.to_state()

    def to_state(self) -> CatalogState:
        if self.frame in _OBSERVED_FRAMES:
            raise ValueError(
                "CatalogState is celestial; observed frames stay on Catalog"
            )
        return CatalogState(
            lon=self.lon,
            lat=self.lat,
            frame=self.frame,
            epoch=self.epoch,
            pm_lon_coslat_mas_yr=self.pm_lon_coslat_mas_yr,
            pm_lat_mas_yr=self.pm_lat_mas_yr,
            parallax_mas=self.parallax_mas,
            radial_velocity_km_s=self.radial_velocity_km_s,
            valid=self.valid,
        )

    @classmethod
    def from_state(
        cls, state: CatalogState, *, columns: dict[str, Tensor] | None = None
    ) -> Catalog:
        assert isinstance(
            state.epoch, Tensor
        )  # normalised by CatalogState.__post_init__
        return cls(
            lon=state.lon,
            lat=state.lat,
            frame=state.frame,
            epoch=state.epoch,
            pm_lon_coslat_mas_yr=state.pm_lon_coslat_mas_yr,
            pm_lat_mas_yr=state.pm_lat_mas_yr,
            parallax_mas=state.parallax_mas,
            radial_velocity_km_s=state.radial_velocity_km_s,
            valid=state.valid,
            columns={} if columns is None else columns,
        )

    def _with_state(self, state: CatalogState) -> Catalog:
        return self.from_state(state, columns=self.columns)

    def to_frame(self, name: str, *, tt_mjd: Tensor | float | None = None) -> Catalog:
        dest = _normalize_frame(name)
        if dest in _OBSERVED_FRAMES and tt_mjd is None:
            raise ValueError(
                "observed frames require Catalog.observe with site and EOP"
            )
        return self._with_state(self.to_state().to_frame(dest, tt_mjd=tt_mjd))

    def propagate_angular(self, epoch: Tensor | float) -> Catalog:
        return self._with_state(self.to_state().propagate_angular(epoch))

    def propagate_physical(self, epoch: Tensor | float) -> Catalog:
        return self._with_state(self.to_state().propagate_physical(epoch))

    def observe(
        self,
        to_frame: str = "ALTAZ",
        *,
        obstime_mjd_utc: Any,
        site: Site,
        eop: EarthOrientation,
        pn: PNModel | None = None,
        aberration: bool = False,
        refraction: bool = False,
    ) -> Catalog:
        lon, lat = transform(
            self.lon,
            self.lat,
            from_frame=self.frame,
            to_frame=to_frame,
            obstime_mjd_utc=obstime_mjd_utc,
            site=site,
            eop=eop,
            pn=pn,
            aberration=aberration,
            refraction=refraction,
        )
        dest = _normalize_frame(to_frame)
        return Catalog(
            lon=lon,
            lat=lat,
            frame=dest,
            epoch=self.epoch,
            pm_lon_coslat_mas_yr=self.pm_lon_coslat_mas_yr,
            pm_lat_mas_yr=self.pm_lat_mas_yr,
            parallax_mas=self.parallax_mas,
            radial_velocity_km_s=self.radial_velocity_km_s,
            valid=self.valid,
            columns=dict(self.columns),
        )

    def index(self, nside: int | None = None) -> HealpixIndex:
        cached = self._index
        if cached is not None and (nside is None or nside == cached.nside):
            return cached
        built = HealpixIndex.build(self.lon, self.lat, nside=nside, nest=True)
        self._index = built
        return built

    def cone(
        self,
        lon: Tensor | float,
        lat: Tensor | float,
        radius_deg: float,
        *,
        nside: int | None = None,
    ) -> Tensor:
        return self.index(nside=nside).query_disc(lon, lat, radius_deg)

    def inside(
        self, coverage: Any, *, time_mjd: Tensor | float | None = None
    ) -> Tensor:
        if time_mjd is not None:
            t_mjd = (
                _to_device(time_mjd, self.device)
                if isinstance(time_mjd, Tensor)
                else torch.as_tensor(
                    time_mjd, dtype=_work_dtype(self.device), device=self.device
                )
            )
            return coverage.contains(t_mjd, self.lon, self.lat)
        return coverage.contains(self.lon, self.lat)

    def take(self, mask_or_index: Tensor) -> Catalog:
        sel = torch.as_tensor(mask_or_index, device=self.lon.device)
        extras = {name: col[sel] for name, col in self.columns.items()}

        def _sel(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            if value.ndim == 0:
                return value
            return value[sel]

        epoch = self.epoch
        if isinstance(epoch, Tensor) and epoch.ndim > 0:
            epoch = epoch[sel]
        return Catalog(
            lon=self.lon[sel],
            lat=self.lat[sel],
            frame=self.frame,
            epoch=epoch,
            pm_lon_coslat_mas_yr=_sel(self.pm_lon_coslat_mas_yr),
            pm_lat_mas_yr=_sel(self.pm_lat_mas_yr),
            parallax_mas=_sel(self.parallax_mas),
            radial_velocity_km_s=_sel(self.radial_velocity_km_s),
            valid=_sel(self.valid),
            columns=extras,
        )

    def crossmatch(
        self,
        other: Catalog,
        radius_arcsec: float | Tensor | None = None,
        *,
        find: str = "best",
        right_index: HealpixIndex | None = None,
        target_epoch: float | Tensor | None = None,
        motion: str = "angular",
        left_err_arcsec: float | Tensor | str | None = None,
        right_err_arcsec: float | Tensor | str | None = None,
    ) -> CrossmatchResult:
        """Match stored positions, or explicitly propagate to a common Julian year.

        ``motion="angular"`` requires both PM components for rows whose epoch
        changes; ``"physical"`` additionally requires parallax and radial
        velocity. Missing motion is never filled with zero. A supplied index
        must be rebuilt by the caller after frame/epoch changes.

        ``find`` and the error options follow :func:`crossmatch_sky`;
        ``left_err_arcsec`` / ``right_err_arcsec`` may also name a column of
        ``self`` / ``other`` (values in arcsec). Epoch/frame alignment reorders
        nothing, so returned indices always refer to the original rows.
        """
        if motion not in ("angular", "physical"):
            raise ValueError("motion must be 'angular' or 'physical'")
        if right_index is not None and (
            target_epoch is not None or other.frame != self.frame
        ):
            raise ValueError(
                "right_index cannot be reused after epoch or frame conversion"
            )
        left = self
        right = other if other.device == self.device else other.to(self.device)
        if target_epoch is not None:
            target = _to_device(torch.as_tensor(target_epoch), self.device)
            if target.numel() != 1 or not bool(torch.isfinite(target).all()):
                raise ValueError("target_epoch must be a finite scalar Julian year")

            def align(catalog: Catalog) -> Catalog:
                epoch = torch.as_tensor(catalog.epoch)
                if bool(torch.all(epoch == target.to(epoch.device))):
                    return catalog
                if motion == "angular":
                    return catalog.propagate_angular(target.reshape(()))
                return catalog.propagate_physical(target.reshape(()))

            left, right = align(left), align(right)
        right = right if right.frame == left.frame else right.to_frame(left.frame)

        def _err(value: Any, cat: Catalog) -> Any:
            return cat.columns[value] if isinstance(value, str) else value

        left_err = _err(left_err_arcsec, self)
        right_err = _err(right_err_arcsec, other)
        if (
            left_err is not None
            or right_err is not None
            or find not in ("best", "best1", "all")
        ):
            # Error radii, best2 and symmetric use the strip index directly.
            return crossmatch_sky(
                left.lon,
                left.lat,
                right.lon,
                right.lat,
                radius_arcsec=radius_arcsec,
                find=find,
                left_err_arcsec=left_err,
                right_err_arcsec=right_err,
            )
        index = right_index if right_index is not None else right.index()
        return crossmatch_sky(
            left.lon,
            left.lat,
            right.lon,
            right.lat,
            radius_arcsec=radius_arcsec,
            find=find,
            right_index=index,
        )

    @classmethod
    def from_arrow(
        cls,
        source: Any,
        *,
        lon: str = "ra",
        lat: str = "dec",
        frame: str = "ICRS",
        epoch: float = 2000.0,
        columns: tuple[str, ...] = (),
    ) -> Catalog:
        import pyarrow as pa

        if isinstance(source, (str, Path)):
            source = pa.ipc.open_file(str(source)).read_all()
        if hasattr(source, "read_all") and not isinstance(source, pa.Table):
            source = source.read_all()
        if not isinstance(source, pa.Table):
            raise TypeError("from_arrow expects a pyarrow.Table, IPC reader, or path")

        def _col(name: str, *, sky: bool = False) -> Tensor:
            tensor = torch.from_numpy(source.column(name).to_numpy().copy())
            if sky:
                return tensor.to(dtype=torch.float64)
            return tensor

        extras = {name: _col(name) for name in columns}
        return cls(
            lon=_col(lon, sky=True),
            lat=_col(lat, sky=True),
            frame=frame,
            epoch=epoch,
            columns=extras,
        )


__all__ = ["Catalog"]
