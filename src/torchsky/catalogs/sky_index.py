"""Reusable sky index: within-radius pairs, k nearest neighbours and counts.

Results are exact, with no approximate step. Pairs come from the zone index in ``_zones`` and
are decided on great-circle separation in float64. Poles and the RA 0/360 wrap
need no special handling. Ties go to the lowest right row index.

Equivalents:

- ``match_within`` and ``SkyIndex.query_within``: astropy ``search_around_sky``
  and scipy ``cKDTree.query_ball_point`` on unit vectors.
- ``match_nearest`` and ``SkyIndex.query_nearest``: astropy ``match_to_catalog_sky``
  (``nthneighbor``) and ``cKDTree.query(k, distance_upper_bound)``.
- ``count_within`` and ``SkyIndex.count_within``: ``cKDTree.query_ball_point(return_length=True)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .._dtype import work_dtype as _work_dtype
from ..device import _as_tensor, _to
from ._zones import (
    DEFAULT_MAX_PAIRS,
    Combine,
    ZoneIndex,
    combine_radii,
    default_zone_height_deg,
    zone_query,
)
from .core import CrossmatchResult

_FULL_SKY_ARCSEC = 180.0 * 3600.0


@dataclass
class NearestResult:
    """``k`` nearest right rows per left row.

    ``index[i, n]`` is the right row of the ``n``-th nearest neighbour of left
    row ``i`` (``-1`` when there is none within ``max_radius_arcsec``).
    ``separation_arcsec`` is ``inf`` there.
    """

    index: Tensor
    separation_arcsec: Tensor


def _coords(
    lon: Tensor | float, lat: Tensor | float, device: torch.device | str | None
) -> tuple[Tensor, Tensor]:
    raw = torch.as_tensor(lon).reshape(-1)
    dev = torch.device(device) if device is not None else raw.device
    dtype = _work_dtype(dev)
    lon_t = _to(raw, device=dev, dtype=dtype)
    lat_t = _as_tensor(lat, dtype=dtype, device=dev).reshape(-1)
    if lon_t.shape != lat_t.shape:
        raise ValueError("lon and lat must have the same shape")
    if lon_t.numel() and not bool(
        torch.isfinite(lon_t).all() & torch.isfinite(lat_t).all()
    ):
        raise ValueError("positions must be finite; drop or mask NaN rows first")
    if lat_t.numel() and bool((lat_t.abs() > 90.0).any()):
        raise ValueError("latitudes must lie in [-90, 90] degrees")
    return lon_t, lat_t


def _radii(
    value: float | Tensor, n: int, ref: Tensor, name: str, *, allow_zero: bool
) -> Tensor:
    r = _as_tensor(value, dtype=ref.dtype, device=ref.device)
    r = r.reshape(1).expand(n) if r.ndim == 0 else r.reshape(-1)
    if int(r.numel()) != n:
        raise ValueError(f"{name} must be a scalar or have one entry per row")
    if n:
        ok = torch.isfinite(r) & ((r >= 0.0) if allow_zero else (r > 0.0))
        if not bool(ok.all()):
            raise ValueError(
                f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}"
            )
    return r


class SkyIndex:
    """Build once over a (right) catalogue, query many times.

    ``zone_height_deg`` only tunes cost, never results. The default is the
    mean point spacing; pass roughly your typical query radius for crowded
    catalogues.
    """

    def __init__(self, zones: ZoneIndex, lon: Tensor, lat: Tensor) -> None:
        self._zones = zones
        self.lon = lon
        self.lat = lat

    @classmethod
    def build(
        cls,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        zone_height_deg: float | None = None,
        device: torch.device | str | None = None,
    ) -> SkyIndex:
        lon, lat = _coords(lon_deg, lat_deg, device)
        h = (
            default_zone_height_deg(int(lon.numel()), None)
            if zone_height_deg is None
            else zone_height_deg
        )
        return cls(ZoneIndex(lon, lat, zone_height_deg=h), lon, lat)

    def __len__(self) -> int:
        return self._zones.n

    @property
    def device(self) -> torch.device:
        return self._zones.device

    def _check_same(self, lon: Tensor, lat: Tensor) -> None:
        if lon.shape != self.lon.shape or not (
            torch.equal(lon, self.lon) and torch.equal(lat, self.lat)
        ):
            raise ValueError(
                "index was built from different positions than the right catalogue"
            )

    def query_within(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        radius_arcsec: float | Tensor,
        *,
        right_radius_arcsec: float | Tensor | None = None,
        combine: Combine = "sum",
        find: Literal["all", "best"] = "all",
        max_pairs: int = DEFAULT_MAX_PAIRS,
    ) -> CrossmatchResult:
        """Pairs with ``separation <= combine(radius[i], right_radius[j])``.

        ``combine`` is ``"sum"`` (STILTS ``skyerr``), ``"quadrature"`` or
        ``"max"``; without ``right_radius_arcsec`` the threshold is
        ``radius[i]``. ``find="all"`` orders pairs by left, separation, then right
        index. ``"best"`` keeps the nearest right row per left row.
        """
        lon, lat = _coords(lon_deg, lat_deg, self.device)
        rl = _radii(
            radius_arcsec,
            int(lon.numel()),
            lon,
            "radius_arcsec",
            allow_zero=right_radius_arcsec is not None,
        )
        rr = (
            None
            if right_radius_arcsec is None
            else _radii(
                right_radius_arcsec,
                len(self),
                lon,
                "right_radius_arcsec",
                allow_zero=True,
            )
        )
        li, rj, sep = zone_query(
            self._zones,
            lon,
            lat,
            rl,
            radius_right=rr,
            combine=combine,
            find=find,
            max_pairs=max_pairs,
        )
        return CrossmatchResult(li, rj, sep)

    def count_within(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        radius_arcsec: float | Tensor,
        *,
        right_radius_arcsec: float | Tensor | None = None,
        combine: Combine = "sum",
        max_pairs: int = DEFAULT_MAX_PAIRS,
    ) -> Tensor:
        """Number of right rows within the threshold of each left row (int64 ``[N]``)."""
        lon, lat = _coords(lon_deg, lat_deg, self.device)
        n = int(lon.numel())
        rl = _radii(
            radius_arcsec,
            n,
            lon,
            "radius_arcsec",
            allow_zero=right_radius_arcsec is not None,
        )
        rr = (
            None
            if right_radius_arcsec is None
            else _radii(
                right_radius_arcsec,
                len(self),
                lon,
                "right_radius_arcsec",
                allow_zero=True,
            )
        )
        rows, counts, _ = zone_query(
            self._zones,
            lon,
            lat,
            rl,
            radius_right=rr,
            combine=combine,
            find="count",
            max_pairs=max_pairs,
        )
        out = torch.zeros(n, dtype=torch.int64, device=lon.device)
        out[rows] = counts
        return out

    def query_nearest(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        k: int = 1,
        *,
        max_radius_arcsec: float | None = None,
        max_pairs: int = DEFAULT_MAX_PAIRS,
    ) -> NearestResult:
        """Exact ``k`` nearest right rows per left row, nearest first.

        Search radii double from an initial guess set by the mean density. A row is
        settled as soon as it has ``k`` rows inside its radius: those are its true
        ``k`` nearest, since every other row lies farther out. Crowded rows settle in the first
        rounds, and each row costs about ``4/3`` of its final round. Without
        ``max_radius_arcsec`` the search can grow to the whole sky, so every row
        with ``k`` right rows on the sky gets ``k`` neighbours.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        lon, lat = _coords(lon_deg, lat_deg, self.device)
        n = int(lon.numel())
        dev = lon.device
        index = torch.full((n, k), -1, dtype=torch.int64, device=dev)
        sep_out = torch.full((n, k), math.inf, dtype=lon.dtype, device=dev)
        if n == 0 or len(self) == 0:
            return NearestResult(index, sep_out)
        cap = (
            _FULL_SKY_ARCSEC if max_radius_arcsec is None else float(max_radius_arcsec)
        )
        if not cap > 0.0:
            raise ValueError("max_radius_arcsec must be positive")
        # Expected k neighbours at uniform density: pi r^2 * M / (4 pi) = k.
        radius = min(cap, math.degrees(math.sqrt(4.0 * k / max(len(self), 1))) * 3600.0)
        todo = torch.arange(n, device=dev)
        while todo.numel():
            last = radius >= cap
            rl = torch.full((int(todo.numel()),), radius, dtype=lon.dtype, device=dev)
            li, rj, sep = zone_query(
                self._zones, lon[todo], lat[todo], rl, find="all", max_pairs=max_pairs
            )
            counts = torch.bincount(li, minlength=int(todo.numel()))
            done = (
                counts >= k if not last else torch.ones_like(counts, dtype=torch.bool)
            )
            first = torch.cumsum(counts, 0) - counts
            rank = torch.arange(li.numel(), device=dev) - first[li]
            take = (rank < k) & done[li]
            rows = todo[li[take]]
            index[rows, rank[take]] = rj[take]
            sep_out[rows, rank[take]] = sep[take]
            todo = todo[~done]
            radius = min(cap, radius * 2.0)
        return NearestResult(index, sep_out)


def _index_for(
    right_lon: Tensor,
    right_lat: Tensor,
    index: SkyIndex | None,
    typical_arcsec: float | None,
) -> SkyIndex:
    if index is not None:
        index._check_same(right_lon, right_lat)
        return index
    h = default_zone_height_deg(int(right_lon.numel()), typical_arcsec)
    return SkyIndex(
        ZoneIndex(right_lon, right_lat, zone_height_deg=h), right_lon, right_lat
    )


def match_within(
    left_lon_deg: Tensor | float,
    left_lat_deg: Tensor | float,
    right_lon_deg: Tensor | float,
    right_lat_deg: Tensor | float,
    radius_arcsec: float | Tensor,
    *,
    right_radius_arcsec: float | Tensor | None = None,
    combine: Combine = "sum",
    index: SkyIndex | None = None,
    device: torch.device | str | None = None,
) -> CrossmatchResult:
    """Every pair within the threshold (astropy ``search_around_sky``).

    See :meth:`SkyIndex.query_within`. Pass ``index`` to reuse a right-side
    :class:`SkyIndex`.
    """
    left = _coords(left_lon_deg, left_lat_deg, device)
    right = _coords(right_lon_deg, right_lat_deg, left[0].device)
    typical = float(
        combine_radii(
            torch.as_tensor(radius_arcsec, dtype=torch.float64).reshape(-1).median(),
            0.0
            if right_radius_arcsec is None
            else float(torch.as_tensor(right_radius_arcsec).max()),
            combine,
        )
    )
    idx = _index_for(*right, index, typical)
    return idx.query_within(
        *left, radius_arcsec, right_radius_arcsec=right_radius_arcsec, combine=combine
    )


def match_nearest(
    left_lon_deg: Tensor | float,
    left_lat_deg: Tensor | float,
    right_lon_deg: Tensor | float,
    right_lat_deg: Tensor | float,
    *,
    k: int = 1,
    max_radius_arcsec: float | None = None,
    index: SkyIndex | None = None,
    device: torch.device | str | None = None,
) -> NearestResult:
    """Exact ``k`` nearest right rows per left row (astropy ``match_to_catalog_sky``).

    ``nthneighbor=n`` in astropy is column ``n - 1`` here with ``k >= n``.
    See :meth:`SkyIndex.query_nearest`.
    """
    left = _coords(left_lon_deg, left_lat_deg, device)
    right = _coords(right_lon_deg, right_lat_deg, left[0].device)
    idx = _index_for(*right, index, None)
    return idx.query_nearest(*left, k, max_radius_arcsec=max_radius_arcsec)


def count_within(
    left_lon_deg: Tensor | float,
    left_lat_deg: Tensor | float,
    right_lon_deg: Tensor | float,
    right_lat_deg: Tensor | float,
    radius_arcsec: float | Tensor,
    *,
    right_radius_arcsec: float | Tensor | None = None,
    combine: Combine = "sum",
    index: SkyIndex | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Right rows within the threshold of each left row, as int64 ``[N]``."""
    left = _coords(left_lon_deg, left_lat_deg, device)
    right = _coords(right_lon_deg, right_lat_deg, left[0].device)
    idx = _index_for(*right, index, None)
    return idx.count_within(
        *left, radius_arcsec, right_radius_arcsec=right_radius_arcsec, combine=combine
    )


__all__ = [
    "NearestResult",
    "SkyIndex",
    "count_within",
    "match_nearest",
    "match_within",
]
