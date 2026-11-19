"""Astrometric uncertainties, motion priors, and covariance through epoch changes.

Positions and motions live on :class:`~torchsky.catalogs.Catalog`. This module
holds their *uncertainties*: one ``(6, 6)`` covariance per row in the local
(east, north) tangent basis at the row's epoch. The order is

``(east, north, pm_east, pm_north, parallax, radial_velocity)``

in mas, mas, mas/yr, mas/yr, mas, km/s. Gaia's ``ra_error`` (mas, already
cos(dec)-scaled) is the east component. Blocks a catalogue does not measure
are NaN, never zero, so using them fails rather than silently claiming
perfect knowledge.

:func:`propagate_astrometry` moves positions to a target epoch with the
per-row router :meth:`~torchsky.coords.CatalogState.propagate`. It returns
the target-epoch position covariance ``J C J^T`` from the matching motion
Jacobian. Rows without proper motion take a :class:`MotionPrior` instead:
their position moves by ``mean * dt`` and their covariance grows by
``dt^2 * cov``. Parallax is never imputed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from ..device import _as_tensor

from ..sphere import healpix as _healpix
from ..wcs.catalog_state import CatalogState
from ..wcs.frames import (
    propagate_proper_motion_with_jacobian,
    propagate_space_motion_with_jacobian,
)

_F64 = torch.float64
_POS = slice(0, 2)


def _rows(value: Tensor | float, n: int, device: torch.device, name: str) -> Tensor:
    t = _as_tensor(value, dtype=_F64, device=device)
    t = t.reshape(1).expand(n) if t.ndim == 0 else t.reshape(-1)
    if int(t.numel()) != n:
        raise ValueError(f"{name} must be a scalar or have one entry per row")
    return t


@dataclass(frozen=True)
class Astrometry:
    """Per-row astrometric covariance, ``(N, 6, 6)`` float64 (see module docstring).

    ``floor_mas2`` is an optional per-catalogue ``(2, 2)`` systematic position
    covariance (mas^2), for example a survey's documented frame-tie error.
    It is added after propagation. Use the constructors; they validate.
    """

    cov: Tensor
    floor_mas2: Tensor | None = None

    def __post_init__(self) -> None:
        cov = torch.as_tensor(self.cov, dtype=_F64)
        if cov.ndim != 3 or cov.shape[1:] != (6, 6):
            raise ValueError("cov must have shape (N, 6, 6)")
        object.__setattr__(self, "cov", cov)
        if self.floor_mas2 is not None:
            floor = _as_tensor(self.floor_mas2, dtype=_F64, device=cov.device)
            if floor.shape != (2, 2) or not bool(torch.isfinite(floor).all()):
                raise ValueError("floor_mas2 must be a finite (2, 2) covariance")
            object.__setattr__(self, "floor_mas2", floor)
        _validate(cov)

    def __len__(self) -> int:
        return int(self.cov.shape[0])

    @property
    def level(self) -> Tensor:
        """Measured block per row: 2 (position), 4 (+pm), 5 (+parallax), 6 (+RV)."""
        diag = torch.diagonal(self.cov, dim1=-2, dim2=-1)
        known = torch.isfinite(diag)
        level = torch.full((len(self),), 2, dtype=torch.int8, device=self.cov.device)
        level = torch.where(known[:, 2:4].all(-1), torch.full_like(level, 4), level)
        level = torch.where(known[:, 2:5].all(-1), torch.full_like(level, 5), level)
        return torch.where(known[:, 2:6].all(-1), torch.full_like(level, 6), level)

    def position_cov(self) -> Tensor:
        """``(N, 2, 2)`` position covariance at the catalogue epoch (mas^2), floor added."""
        pos = self.cov[:, _POS, _POS]
        return pos if self.floor_mas2 is None else pos + self.floor_mas2

    def take(self, index: Tensor) -> Astrometry:
        return Astrometry(self.cov[index], self.floor_mas2)

    def to(self, device: torch.device | str) -> Astrometry:
        if torch.device(device).type == "mps":
            raise ValueError(
                "astrometric covariances are float64; keep them on CPU or CUDA"
            )
        floor = None if self.floor_mas2 is None else self.floor_mas2.to(device)
        return Astrometry(self.cov.to(device), floor)

    # ------------------------------------------------------------ builders

    @classmethod
    def from_errors(
        cls,
        sigma_east_mas: Tensor | float,
        sigma_north_mas: Tensor | float,
        corr_east_north: Tensor | float = 0.0,
        *,
        n: int | None = None,
        floor_mas2: Tensor | None = None,
    ) -> Astrometry:
        """Position-only covariance from 1-sigma errors and their correlation."""
        se = torch.as_tensor(sigma_east_mas, dtype=_F64)
        n = int(se.numel()) if n is None else n
        dev = se.device
        se = _rows(se, n, dev, "sigma_east_mas")
        sn = _rows(sigma_north_mas, n, dev, "sigma_north_mas")
        rho = _rows(corr_east_north, n, dev, "corr_east_north")
        cov = torch.full((n, 6, 6), math.nan, dtype=_F64, device=dev)
        cov[:, 0, 0] = se * se
        cov[:, 1, 1] = sn * sn
        cov[:, 0, 1] = cov[:, 1, 0] = rho * se * sn
        return cls(cov, floor_mas2)

    @classmethod
    def from_ellipse(
        cls,
        major_mas: Tensor | float,
        minor_mas: Tensor | float,
        pa_deg: Tensor | float,
        *,
        n: int | None = None,
        floor_mas2: Tensor | None = None,
    ) -> Astrometry:
        """Position-only covariance from a 1-sigma error ellipse.

        ``pa_deg`` is the position angle of the major axis, east of north, as in
        the 2MASS ``err_maj`` / ``err_min`` / ``err_ang`` columns.
        """
        a = torch.as_tensor(major_mas, dtype=_F64)
        n = int(a.numel()) if n is None else n
        dev = a.device
        a = _rows(a, n, dev, "major_mas")
        b = _rows(minor_mas, n, dev, "minor_mas")
        pa = torch.deg2rad(_rows(pa_deg, n, dev, "pa_deg"))
        s, c = torch.sin(pa), torch.cos(pa)  # major axis direction (east, north)
        cov = torch.full((n, 6, 6), math.nan, dtype=_F64, device=dev)
        cov[:, 0, 0] = a * a * s * s + b * b * c * c
        cov[:, 1, 1] = a * a * c * c + b * b * s * s
        cov[:, 0, 1] = cov[:, 1, 0] = (a * a - b * b) * s * c
        return cls(cov, floor_mas2)

    @classmethod
    def from_gaia(
        cls,
        columns: Mapping[str, Tensor],
        *,
        radial_velocity_error: Tensor | None = None,
        floor_mas2: Tensor | None = None,
    ) -> Astrometry:
        """Validated :meth:`gaia_covariance` (see there for the columns)."""
        return cls(
            cls.gaia_covariance(columns, radial_velocity_error=radial_velocity_error),
            floor_mas2,
        )

    @staticmethod
    def gaia_covariance(
        columns: Mapping[str, Tensor],
        *,
        radial_velocity_error: Tensor | None = None,
    ) -> Tensor:
        """Gaia 5-parameter covariance from the archive's error/correlation columns.

        Needs ``ra_error, dec_error, parallax_error, pmra_error, pmdec_error`` and
        the ten ``*_corr`` columns (``ra_dec_corr``, ``ra_parallax_corr``, ...).
        Two-parameter solutions (NaN parallax/pm errors) keep a position-only
        block. ``radial_velocity_error`` (km/s) fills the RV variance where finite;
        RV is taken as uncorrelated with the astrometry.
        """
        names = ("ra", "dec", "pmra", "pmdec", "parallax")
        slot = {"ra": 0, "dec": 1, "pmra": 2, "pmdec": 3, "parallax": 4}
        sig = {
            k: torch.as_tensor(columns[f"{k}_error"], dtype=_F64).reshape(-1)
            for k in names
        }
        n = int(sig["ra"].numel())
        dev = sig["ra"].device
        cov = torch.full((n, 6, 6), math.nan, dtype=_F64, device=dev)
        for k in names:
            cov[:, slot[k], slot[k]] = sig[k] * sig[k]
        gaia_order = ("ra", "dec", "parallax", "pmra", "pmdec")
        for i, a in enumerate(gaia_order):
            for b in gaia_order[i + 1 :]:
                rho = (
                    torch.as_tensor(columns[f"{a}_{b}_corr"], dtype=_F64)
                    .reshape(-1)
                    .to(dev)
                )
                value = rho * sig[a] * sig[b]
                cov[:, slot[a], slot[b]] = cov[:, slot[b], slot[a]] = value
        two_param = ~torch.isfinite(sig["parallax"]) | ~torch.isfinite(sig["pmra"])
        if bool(two_param.any()):
            cov[two_param, 2:, :] = math.nan
            cov[two_param, :, 2:] = math.nan
        if radial_velocity_error is not None:
            rv = _rows(radial_velocity_error, n, dev, "radial_velocity_error")
            ok = torch.isfinite(rv) & ~two_param
            cov[ok, 5, 5] = rv[ok] * rv[ok]
            cov[ok, 5, :5] = 0.0
            cov[ok, :5, 5] = 0.0
        return cov

    @staticmethod
    def validity(cov: Tensor) -> Tensor:
        """Per-row validity under the constructor's checks (``[N]`` bool).

        A row is valid when its position block is finite and positive definite,
        its matrix is symmetric, and every finite leading block is positive
        definite. Use it to drop rows explicitly; the constructor raises instead.
        """
        return _row_validity(torch.as_tensor(cov, dtype=_F64))


def _row_validity(cov: Tensor) -> Tensor:
    n = int(cov.shape[0])
    ok = torch.ones(n, dtype=torch.bool, device=cov.device)
    if n == 0:
        return ok
    pos = cov[:, _POS, _POS]
    ok &= torch.isfinite(pos).all(-1).all(-1)
    finite = torch.isfinite(cov)
    scale = cov.abs().nan_to_num(0.0).amax(dim=(-2, -1)).clamp(min=1e-300)
    ok &= ~(
        (cov - cov.mT).abs().nan_to_num(0.0).amax(dim=(-2, -1))
        > 64 * torch.finfo(_F64).eps * scale
    )
    for size in (2, 4, 5, 6):
        rows = finite[:, :size, :size].all(-1).all(-1)
        if not bool(rows.any()):
            continue
        _, info = torch.linalg.cholesky_ex(cov[rows][:, :size, :size])
        bad = torch.zeros_like(ok)
        bad[torch.nonzero(rows, as_tuple=True)[0]] = info != 0
        ok &= ~bad
    return ok


def _validate(cov: Tensor) -> None:
    """Fail closed: finite PD position block; symmetric PD measured blocks."""
    ok = _row_validity(cov)
    if not bool(ok.all()):
        bad = int((~ok).sum())
        pos_bad = int((~torch.isfinite(cov[:, _POS, _POS]).all(-1).all(-1)).sum())
        if pos_bad:
            raise ValueError(f"{pos_bad} row(s) lack a finite position covariance")
        raise ValueError(
            f"{bad} row(s) have a non-symmetric covariance or a block that is not positive "
            "definite; Astrometry.validity(cov) marks them"
        )


# -------------------------------------------------------------- motion prior


@dataclass(frozen=True)
class MotionPrior:
    """Proper-motion distribution for rows that have no measured proper motion.

    ``mean`` is ``(C, 2)`` and ``cov`` is ``(C, 2, 2)`` (mas/yr, east/north), with one entry per
    nested HEALPix cell at ``order`` (``C = 12 * 4**order``), or ``C = 1`` for a
    constant prior. It is used only through :func:`propagate_astrometry` for
    level-2 rows (no proper motion): the position moves by ``mean * dt``
    and the position covariance grows by ``dt^2 * cov``.
    """

    mean: Tensor
    cov: Tensor
    order: int | None = None

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean, dtype=_F64).reshape(-1, 2)
        cov = _as_tensor(self.cov, dtype=_F64, device=mean.device).reshape(-1, 2, 2)
        expected = 1 if self.order is None else 12 * 4**self.order
        if mean.shape[0] != expected or cov.shape[0] != expected:
            raise ValueError(f"prior needs {expected} cell(s) for order {self.order}")
        if not bool(torch.isfinite(mean).all() & torch.isfinite(cov).all()):
            raise ValueError("prior mean and covariance must be finite")
        _, info = torch.linalg.cholesky_ex(
            cov + 1e-30 * torch.eye(2, dtype=_F64, device=cov.device)
        )
        if bool((info != 0).any()):
            raise ValueError("prior covariances must be positive semi-definite")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "cov", cov)

    @classmethod
    def constant(
        cls, mean_mas_yr: tuple[float, float], cov_mas2_yr2: Tensor
    ) -> MotionPrior:
        return cls(
            torch.tensor([mean_mas_yr], dtype=_F64),
            torch.as_tensor(cov_mas2_yr2, dtype=_F64)[None],
        )

    @classmethod
    def from_reference(
        cls,
        lon_deg: Tensor,
        lat_deg: Tensor,
        pm_east_mas_yr: Tensor,
        pm_north_mas_yr: Tensor,
        *,
        order: int,
        min_count: int = 32,
    ) -> MotionPrior:
        """Empirical prior: pm mean and covariance of a reference catalogue per cell.

        Cells with fewer than ``min_count`` finite rows inherit their parent's
        moments, recursively up to order 0 and then the whole sky. The pooling
        rule is explicit, and each prior rests on at least ``min_count`` stars
        whenever the sky holds that many. Typical use: Gaia DR3 proper motions
        as the prior for a catalogue without them.
        """
        lon = torch.as_tensor(lon_deg, dtype=_F64).reshape(-1)
        lat = _as_tensor(lat_deg, dtype=_F64, device=lon.device).reshape(-1)
        pe = _as_tensor(pm_east_mas_yr, dtype=_F64, device=lon.device).reshape(-1)
        pn = _as_tensor(pm_north_mas_yr, dtype=_F64, device=lon.device).reshape(-1)
        ok = (
            torch.isfinite(pe)
            & torch.isfinite(pn)
            & torch.isfinite(lon)
            & torch.isfinite(lat)
        )
        if int(ok.sum()) < 2:
            raise ValueError(
                "the reference catalogue needs at least two finite proper motions"
            )
        lon, lat, pe, pn = lon[ok], lat[ok], pe[ok], pn[ok]
        pm = torch.stack((pe, pn), -1)
        pix = _healpix.ang2pix(2**order, lon, lat, nest=True, lonlat=True).to(
            torch.int64
        )

        def moments(cells: Tensor, ncell: int) -> tuple[Tensor, Tensor, Tensor]:
            count = torch.bincount(cells, minlength=ncell).to(_F64)
            s1 = torch.zeros(ncell, 2, dtype=_F64, device=lon.device).index_add_(
                0, cells, pm
            )
            s2 = torch.zeros(ncell, 2, 2, dtype=_F64, device=lon.device).index_add_(
                0, cells, pm[:, :, None] * pm[:, None, :]
            )
            return count, s1, s2

        level_moments = []
        for o in range(order, -1, -1):
            cells = pix >> (2 * (order - o))
            level_moments.append(moments(cells, 12 * 4**o))
        total = (
            torch.tensor([float(pm.shape[0])], dtype=_F64, device=lon.device),
            pm.sum(0, keepdim=True),
            (pm[:, :, None] * pm[:, None, :]).sum(0, keepdim=True),
        )
        ncell = 12 * 4**order
        count = torch.zeros(ncell, dtype=_F64, device=lon.device)
        s1 = torch.zeros(ncell, 2, dtype=_F64, device=lon.device)
        s2 = torch.zeros(ncell, 2, 2, dtype=_F64, device=lon.device)
        done = torch.zeros(ncell, dtype=torch.bool, device=lon.device)
        own = torch.arange(ncell, device=lon.device)
        for depth, (c, m1, m2) in enumerate(level_moments):
            parent = own >> (2 * depth)
            use = ~done & (c[parent] >= min_count)
            count[use], s1[use], s2[use] = (
                c[parent][use],
                m1[parent][use],
                m2[parent][use],
            )
            done |= use
        rest = ~done
        count[rest], s1[rest], s2[rest] = total[0][0], total[1][0], total[2][0]
        mean = s1 / count[:, None]
        cov = s2 / count[:, None, None] - mean[:, :, None] * mean[:, None, :]
        cov = cov * (count / (count - 1.0)).clamp(min=1.0)[:, None, None]
        cov = 0.5 * (cov + cov.mT)
        return cls(mean, cov, order)

    def at(self, lon_deg: Tensor, lat_deg: Tensor) -> tuple[Tensor, Tensor]:
        """Per-row prior ``(mean (N, 2), cov (N, 2, 2))``."""
        if self.order is None:
            n = int(torch.as_tensor(lon_deg).numel())
            return self.mean.expand(n, 2), self.cov.expand(n, 2, 2)
        pix = _healpix.ang2pix(
            2**self.order, lon_deg, lat_deg, nest=True, lonlat=True
        ).to(torch.int64)
        return self.mean[pix], self.cov[pix]


# --------------------------------------------------------------- propagation


def propagate_astrometry(
    state: CatalogState,
    astrometry: Astrometry,
    epoch: Tensor | float,
    *,
    motion_prior: MotionPrior | None = None,
) -> tuple[CatalogState, Tensor]:
    """Positions and ``(N, 2, 2)`` position covariance (mas^2) at ``epoch``.

    Rows are routed by the lower of their motion level
    (:meth:`CatalogState.motion_level`) and their covariance level
    (:attr:`Astrometry.level`), so a value is never propagated without its
    uncertainty:

    - 6: full space motion with the 6x6 Jacobian;
    - 4 or 5: linear angular motion with the 2x4 Jacobian (parallax unused);
    - 2: the :class:`MotionPrior` mean and ``dt^2`` times its covariance.
      Without a prior these rows must not change epoch.

    The returned covariance is in the local east/north basis at each row's new
    position, with ``Astrometry.floor_mas2`` added.
    """
    n = int(state.lon.numel())
    if len(astrometry) != n:
        raise ValueError("astrometry and state must have the same number of rows")
    dev = state.lon.device
    target = _as_tensor(epoch, dtype=_F64, device=dev)
    dt = torch.broadcast_to(
        target - _as_tensor(state.epoch, dtype=_F64, device=dev), (n,)
    )
    motion = state.motion_level()
    cov_level = astrometry.level.to(dev)
    level = torch.minimum(
        motion, torch.where(cov_level == 5, torch.full_like(cov_level, 4), cov_level)
    )
    cov = astrometry.cov.to(dev)
    out_cov = cov[:, _POS, _POS].clone()
    prior = None
    if motion_prior is not None:
        prior = motion_prior.at(state.lon, state.lat)
    capped = CatalogState(
        lon=state.lon,
        lat=state.lat,
        frame=state.frame,
        epoch=state.epoch,
        epoch_scale=state.epoch_scale,
        epoch_format=state.epoch_format,
        pm_lon_coslat_mas_yr=_mask(state.pm_lon_coslat_mas_yr, level >= 4),
        pm_lat_mas_yr=_mask(state.pm_lat_mas_yr, level >= 4),
        parallax_mas=_mask(state.parallax_mas, level == 6),
        radial_velocity_km_s=_mask(state.radial_velocity_km_s, level == 6),
        valid=state.valid,
    )
    moved = capped.propagate(
        target,
        prior_pm_mas_yr=None if prior is None else (prior[0][:, 0], prior[0][:, 1]),
    )
    moving = dt != 0
    nan = torch.full((n,), math.nan, dtype=_F64, device=dev)
    pm_e = nan if state.pm_lon_coslat_mas_yr is None else state.pm_lon_coslat_mas_yr
    pm_n = nan if state.pm_lat_mas_yr is None else state.pm_lat_mas_yr
    plx = nan if state.parallax_mas is None else state.parallax_mas
    rv = nan if state.radial_velocity_km_s is None else state.radial_velocity_km_s
    rows6 = torch.nonzero(moving & (level == 6), as_tuple=True)[0]
    if rows6.numel():
        _, jac = propagate_space_motion_with_jacobian(
            state.lon[rows6],
            state.lat[rows6],
            pm_e[rows6],
            pm_n[rows6],
            plx[rows6],
            rv[rows6],
            source_epoch_jyear=torch.broadcast_to(
                _as_tensor(state.epoch, dtype=_F64, device=dev), (n,)
            )[rows6],
            target_epoch_jyear=torch.broadcast_to(target, (n,))[rows6],
        )
        j = jac[:, :2, :]
        out_cov[rows6] = j @ cov[rows6] @ j.mT
    rows4 = torch.nonzero(moving & (level == 4), as_tuple=True)[0]
    if rows4.numel():
        _, jac = propagate_proper_motion_with_jacobian(
            state.lon[rows4],
            state.lat[rows4],
            pm_e[rows4],
            pm_n[rows4],
            source_epoch_jyear=torch.broadcast_to(
                _as_tensor(state.epoch, dtype=_F64, device=dev), (n,)
            )[rows4],
            target_epoch_jyear=torch.broadcast_to(target, (n,))[rows4],
        )
        c4 = cov[rows4, :4, :4]
        out_cov[rows4] = jac @ c4 @ jac.mT
    rows2 = torch.nonzero(moving & (level == 2), as_tuple=True)[0]
    if rows2.numel():
        assert prior is not None  # propagate() raised otherwise
        out_cov[rows2] = (
            out_cov[rows2] + (dt[rows2] ** 2)[:, None, None] * prior[1][rows2]
        )
    if astrometry.floor_mas2 is not None:
        out_cov = out_cov + astrometry.floor_mas2.to(dev)
    return moved, 0.5 * (out_cov + out_cov.mT)


def _mask(value: Tensor | None, keep: Tensor) -> Tensor | None:
    if value is None:
        return None
    return torch.where(keep, value, torch.full_like(value, math.nan))


__all__ = ["Astrometry", "MotionPrior", "propagate_astrometry"]
