"""Sky partitions for survey-scale matching (no I/O; HATS-compatible, HATS-agnostic).

A :class:`PartitionPlan` is a set of disjoint nested HEALPix cells of mixed
orders, held as half-open ranges of order-29 indices (the HATS
``_healpix_29`` convention). Every row belongs to at most one cell
(:meth:`PartitionPlan.owner`).

To match a partitioned left catalogue against a right one, match partition
``p``'s owned left rows against every right row within the margin of ``p``:

- :func:`margin_arcsec` bounds how far a counterpart can lie (match radius,
  plus position errors, plus motion over the epoch gap);
- :meth:`PartitionPlan.neighbors` lists the partitions whose cells can hold
  such rows (one vectorised exact two-sided cap query over cell centres);
- :meth:`PartitionPlan.in_margin` keeps the rows within ``R_p + margin`` of
  ``p``'s centre, with ``R_p`` the analytic maximum pixel radius of its order.
  This is a superset of the cell dilated by the margin, so no counterpart is lost.

Pairs belong to the partition that owns their left row. The union over
partitions of ``all``, ``best`` and nearest-neighbour results equals the
single-pass result, with no de-duplication. Self-joins keep pairs where the
partition owns ``min(i, j)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..device import _as_tensor

from ..sphere import healpix as _healpix
from ..sphere.moc import MOC, uniq2nest
from .sky_index import match_within

_ORDER29 = 29
_F64 = torch.float64


def healpix29(lon_deg: Tensor, lat_deg: Tensor) -> Tensor:
    """Nested HEALPix index at order 29 (HATS ``_healpix_29``), int64."""
    return _healpix.ang2pix(2**_ORDER29, lon_deg, lat_deg, nest=True, lonlat=True).to(
        torch.int64
    )


def margin_arcsec(
    radius_arcsec: float,
    *,
    right_radius_arcsec: float = 0.0,
    pm_max_mas_yr: float = 0.0,
    dt_max_yr: float = 0.0,
) -> float:
    """Largest possible separation of a counterpart at the stored epochs.

    The sum of the left search radius, the largest right radius (errors or
    two-sided rules) and the largest motion over the largest epoch gap. Use it
    as the margin when the partitioned positions are not yet propagated.
    """
    if min(radius_arcsec, right_radius_arcsec, pm_max_mas_yr, dt_max_yr) < 0:
        raise ValueError("margin terms must be non-negative")
    return radius_arcsec + right_radius_arcsec + pm_max_mas_yr * abs(dt_max_yr) / 1000.0


@dataclass(frozen=True)
class PartitionPlan:
    """Disjoint nested HEALPix cells, sorted by order-29 start index.

    ``orders`` and ``pixels`` (nested) describe each cell. ``start`` and ``stop``
    are its half-open order-29 range.
    """

    orders: Tensor
    pixels: Tensor

    def __post_init__(self) -> None:
        orders = torch.as_tensor(self.orders, dtype=torch.int64).reshape(-1)
        pixels = _as_tensor(
            self.pixels, dtype=torch.int64, device=orders.device
        ).reshape(-1)
        if orders.shape != pixels.shape:
            raise ValueError("orders and pixels must have the same length")
        if orders.numel() and (int(orders.min()) < 0 or int(orders.max()) > _ORDER29):
            raise ValueError("orders must lie in [0, 29]")
        if bool((pixels < 0).any() | (pixels >= 12 * 4**orders).any()):
            raise ValueError("pixel out of range for its order")
        shift = 2 * (_ORDER29 - orders)
        start = pixels << shift
        stop = (pixels + 1) << shift
        order = torch.argsort(start)
        start, stop = start[order], stop[order]
        if start.numel() > 1 and bool((start[1:] < stop[:-1]).any()):
            raise ValueError("partition cells overlap")
        object.__setattr__(self, "orders", orders[order])
        object.__setattr__(self, "pixels", pixels[order])
        object.__setattr__(self, "_start", start)
        object.__setattr__(self, "_stop", stop)

    def __len__(self) -> int:
        return int(self.orders.numel())

    @property
    def start(self) -> Tensor:
        return self._start  # type: ignore[attr-defined]

    @property
    def stop(self) -> Tensor:
        return self._stop  # type: ignore[attr-defined]

    @classmethod
    def from_pixels(
        cls, orders: Tensor | list[int], pixels: Tensor | list[int]
    ) -> PartitionPlan:
        """Plan from explicit cells, for example a HATS catalogue's partition list."""
        return cls(torch.as_tensor(orders), torch.as_tensor(pixels))

    @classmethod
    def from_moc(cls, moc: MOC) -> PartitionPlan:
        """One cell per MOC cell."""
        order, pix = uniq2nest(moc.uniq)
        return cls(order.to(torch.int64), pix.to(torch.int64))

    @classmethod
    def from_counts(
        cls,
        lon_deg: Tensor,
        lat_deg: Tensor,
        *,
        max_rows: int,
        max_order: int = 12,
        min_order: int = 0,
    ) -> PartitionPlan:
        """Adaptive plan: split cells while they hold more than ``max_rows`` rows.

        Cells stop splitting at ``max_order``, so a cell may exceed ``max_rows``
        there. Empty sky gets no cell (HATS-like sparse coverage).
        """
        if max_rows < 1 or not 0 <= min_order <= max_order <= _ORDER29:
            raise ValueError("need max_rows >= 1 and 0 <= min_order <= max_order <= 29")
        h = healpix29(
            torch.as_tensor(lon_deg, dtype=_F64), torch.as_tensor(lat_deg, dtype=_F64)
        )
        orders, pixels = [], []
        active = h
        for o in range(min_order, max_order + 1):
            if active.numel() == 0:
                break
            pix = active >> (2 * (_ORDER29 - o))
            uniq, counts = torch.unique(pix, return_counts=True)
            final = (
                counts <= max_rows
                if o < max_order
                else torch.ones_like(counts, dtype=torch.bool)
            )
            orders.append(torch.full((int(final.sum()),), o, dtype=torch.int64))
            pixels.append(uniq[final])
            keep = ~torch.isin(pix, uniq[final])
            active = active[keep]
        return cls(torch.cat(orders), torch.cat(pixels))

    def owner(self, lon_deg: Tensor, lat_deg: Tensor) -> Tensor:
        """Partition of each row (``-1`` outside every cell)."""
        h = healpix29(
            torch.as_tensor(lon_deg, dtype=_F64), torch.as_tensor(lat_deg, dtype=_F64)
        )
        idx = torch.searchsorted(self.start, h, right=True) - 1
        safe = idx.clamp(min=0)
        inside = (idx >= 0) & (h < self.stop[safe])
        return torch.where(inside, safe, torch.full_like(idx, -1))

    def centers(self) -> tuple[Tensor, Tensor]:
        """Cell centres (degrees)."""
        lon = torch.empty(len(self), dtype=_F64)
        lat = torch.empty(len(self), dtype=_F64)
        for o in torch.unique(self.orders).tolist():
            sel = self.orders == o
            lo, la = _healpix.pix2ang(2**o, self.pixels[sel], nest=True, lonlat=True)
            lon[sel], lat[sel] = lo.to(_F64), la.to(_F64)
        return lon, lat

    def radii_arcsec(self) -> Tensor:
        """Analytic maximum centre-to-boundary angle of each cell (arcsec)."""
        table = torch.tensor(
            [_healpix.max_pixel_radius(2**o) for o in range(_ORDER29 + 1)], dtype=_F64
        ) * (180.0 * 3600.0 / math.pi)
        return table[self.orders]

    def neighbors(
        self, margin_arcsec: float, other: PartitionPlan | None = None
    ) -> tuple[Tensor, Tensor]:
        """Pairs ``(p, q)`` (cell of this plan, cell of ``other``) that can hold rows within ``margin``.

        Sound: ``q`` is listed whenever some point of cell ``q`` lies within
        ``margin_arcsec`` of some point of cell ``p`` (cap test
        ``|c_p - c_q| <= R_p + R_q + margin``). ``other`` defaults to this plan
        (then ``(p, p)`` is included); pass the right catalogue's own plan, for
        example its HATS partitions.
        """
        other = self if other is None else other
        lon, lat = self.centers()
        olon, olat = other.centers()
        res = match_within(
            lon,
            lat,
            olon,
            olat,
            self.radii_arcsec() + margin_arcsec,
            right_radius_arcsec=other.radii_arcsec(),
            combine="sum",
        )
        return res.left_index, res.right_index

    def in_margin(
        self,
        partition: int | Tensor,
        lon_deg: Tensor,
        lat_deg: Tensor,
        margin_arcsec: float,
    ) -> Tensor:
        """Rows that may lie within ``margin`` of a cell (cap ``R_p + margin``).

        ``partition`` is one cell index (returns ``[N]`` bool) or one per row.
        """
        lon_c, lat_c = self.centers()
        r = self.radii_arcsec() + margin_arcsec
        p = torch.as_tensor(partition, dtype=torch.int64)
        lon_t = torch.as_tensor(lon_deg, dtype=_F64).reshape(-1)
        lat_t = torch.as_tensor(lat_deg, dtype=_F64).reshape(-1)
        p = p.expand(lon_t.shape) if p.ndim == 0 else p
        from ..sphere import great_circle_distance

        sep = (
            great_circle_distance(lon_t, lat_t, lon_c[p], lat_c[p], degrees=True)
            * 3600.0
        )
        return sep <= r[p]


__all__ = ["PartitionPlan", "healpix29", "margin_arcsec"]
