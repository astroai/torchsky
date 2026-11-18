"""Zone index: exact within-radius candidate search on the sphere.

The right catalogue is sorted once by ``(zone, right ascension)``, where zones are
declination bands of height ``h`` (Gray, Szalay & Nieto-Santisteban 2006). A
left row with window radius ``W`` scans the zones that meet
``[dec - W, dec + W]``, and in each zone one right-ascension interval of
half-width ``asin(sin W / cos dec)``. When ``W >= 90 - |dec|`` the cap holds a
pole and the full circle is scanned. Near RA 0/360 the interval splits into two
pieces. Window edges are padded by a few ulp, so the candidate set is always
a superset of the true pairs, whatever ``h`` is.

Each candidate goes through a padded chord prefilter
``|a - b|^2 <= 4 sin^2(R / 2)``. The pair is accepted on its exact atan2
separation. ``h`` changes only the cost, never the result.

Cost is ``O((N + M) log M + K)`` for ``K`` candidates. Memory is bounded by
``max_pairs`` candidates per batch, and a single huge window is split across
batches too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor

from ..device import _scatter_reduce_int

Combine = Literal["sum", "quadrature", "max"]

_RA_BITS = 32
_RA_SCALE = float(1 << _RA_BITS)
_RA_TOP = (1 << _RA_BITS) - 1
_MAX_ZONES = 1 << 30
DEFAULT_MAX_PAIRS = 1 << 22
_ARCSEC_PER_RAD = 180.0 * 3600.0 / math.pi


def unit_vectors(ra_deg: Tensor, dec_deg: Tensor) -> Tensor:
    """``[N, 3]`` unit vectors from degrees (differentiable)."""
    ra, dec = torch.deg2rad(ra_deg), torch.deg2rad(dec_deg)
    cd = torch.cos(dec)
    return torch.stack((cd * torch.cos(ra), cd * torch.sin(ra), torch.sin(dec)), dim=-1)


def separation_arcsec(a: Tensor, b: Tensor) -> Tensor:
    """Great-circle angle between unit vectors, atan2 form, in arcsec."""
    cross = torch.linalg.norm(torch.cross(a, b, dim=-1), dim=-1)
    dot = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.atan2(cross, dot) * _ARCSEC_PER_RAD


def combine_radii(left: Tensor, right: Tensor | float, how: Combine) -> Tensor:
    if how == "sum":
        return left + right
    if how == "quadrature":
        return (
            torch.sqrt(left * left + right * right)
            if isinstance(right, Tensor)
            else torch.sqrt(left * left + float(right) ** 2)
        )
    if how == "max":
        if isinstance(right, Tensor):
            return torch.maximum(left, right)
        return left.clamp(min=float(right))
    raise ValueError(f"unknown radius combination {how!r}")


def occupied_spacing_deg(
    ra_deg: Tensor, dec_deg: Tensor, *, cell_deg: float = 0.25
) -> float:
    """Mean point spacing over the occupied sky, from ``cell_deg`` cells.

    The cells are a cost heuristic only: about ``cell_deg`` on a side, in declination
    bands. For data packed into a small footprint or clusters this is the
    spacing that sets candidate counts. The all-sky mean overestimates it by
    ``sqrt(41253 deg^2 / footprint)``.
    """
    n = int(ra_deg.numel())
    if n == 0:
        return 180.0
    with torch.no_grad():
        band = torch.floor((dec_deg.detach() + 90.0) / cell_deg)
        cos_c = torch.cos(torch.deg2rad((band + 0.5) * cell_deg - 90.0)).clamp(min=1e-3)
        per_band = torch.ceil(360.0 * cos_c / cell_deg)
        col = torch.floor(torch.remainder(ra_deg.detach(), 360.0) / 360.0 * per_band)
        cell = band.to(torch.int64) * (1 << 20) + col.to(torch.int64)
        occupied = int(torch.unique(cell).numel())
    return math.sqrt(occupied * cell_deg * cell_deg / n)


def default_zone_height_deg(
    n_points: int,
    typical_window_arcsec: float | None,
    spacing_deg: float | None = None,
) -> float:
    """Zone height: the typical query window, capped by the point spacing.

    Results never depend on it. With ``h ~ W`` a query scans about three zones
    and visits about ``density * 2W * (2W + 2h)`` candidates. Coarser zones
    are cheaper only when windows are far below the point spacing.
    ``spacing_deg`` defaults to the all-sky mean spacing; pass
    :func:`occupied_spacing_deg` for concentrated catalogues.
    """
    spacing = (
        math.degrees(math.sqrt(4.0 * math.pi / max(n_points, 1)))
        if spacing_deg is None
        else spacing_deg
    )
    h = (
        spacing
        if typical_window_arcsec is None
        else min(spacing, typical_window_arcsec / 3600.0)
    )
    return min(max(h, 180.0 / _MAX_ZONES), 180.0)


@dataclass
class EngineStats:
    """Work counters for one query (candidate reduction evidence)."""

    left_rows: int = 0
    segments: int = 0
    candidates: int = 0
    pairs: int = 0


class ZoneIndex:
    """Right catalogue sorted by ``(zone, quantised RA)``.

    Positions are kept as unit vectors in the input dtype, so autograd flows to
    the reported separations. Per-row right radii are a query argument, so one
    index serves every matcher.
    """

    def __init__(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        *,
        zone_height_deg: float,
    ) -> None:
        if not zone_height_deg > 0.0:
            raise ValueError("zone_height_deg must be positive")
        self.device = ra_deg.device
        self.dtype = ra_deg.dtype
        self.n = int(ra_deg.numel())
        self.zone_height = max(float(zone_height_deg), 180.0 / _MAX_ZONES)
        self.n_zones = int(math.ceil(180.0 / self.zone_height)) + 1
        with torch.no_grad():
            key = (self._zone_of(dec_deg.detach()) << _RA_BITS) | _ra_key(
                ra_deg.detach()
            )
            self.order = torch.argsort(key, stable=True)
            self.key = key[self.order]
        self.xyz = unit_vectors(ra_deg, dec_deg)[self.order]
        self._inv: Tensor | None = None

    def inverse_order(self) -> Tensor:
        """Sorted position of each original row."""
        if self._inv is None:
            inv = torch.empty_like(self.order)
            inv[self.order] = torch.arange(self.n, device=self.order.device)
            self._inv = inv
        return self._inv

    def local_density(self, ra_deg: Tensor, dec_deg: Tensor) -> Tensor:
        """Right rows per square degree around each query (cost heuristic).

        Counts rows in the three zones around the query, within ``4 h`` of it on
        the sky in RA. The RA wrap is ignored; the estimate only steers search radii.
        """
        h = self.zone_height
        with torch.no_grad():
            z = self._zone_of(dec_deg)
            cos_d = torch.cos(torch.deg2rad(dec_deg)).clamp(min=1e-6)
            half = (4.0 * h / cos_d).clamp(max=180.0)
            ra = torch.remainder(ra_deg, 360.0)
            klo = _ra_key(ra - half)
            khi = _ra_key(ra + half)
            whole = half >= 180.0
            klo = torch.where(whole, torch.zeros_like(klo), torch.minimum(klo, khi))
            khi = torch.where(
                whole, torch.full_like(khi, _RA_TOP), torch.maximum(klo, khi)
            )
            count = torch.zeros_like(z)
            for dz in (-1, 0, 1):
                zz = (z + dz).clamp(0, self.n_zones - 1)
                lo = torch.searchsorted(self.key, (zz << _RA_BITS) | klo, right=False)
                hi = torch.searchsorted(self.key, (zz << _RA_BITS) | khi, right=True)
                count = count + (hi - lo) * ((z + dz >= 0) & (z + dz < self.n_zones))
            area = 3.0 * h * torch.where(whole, 360.0 * cos_d, 8.0 * h)
            return count.to(ra_deg.dtype) / area

    def _zone_of(self, dec_deg: Tensor) -> Tensor:
        z = torch.floor((dec_deg + 90.0) / self.zone_height).to(torch.int64)
        return z.clamp_(0, self.n_zones - 1)

    def segments(
        self, ra_deg: Tensor, dec_deg: Tensor, window_deg: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``(left row, start, count)`` runs of the sorted right catalogue to scan.

        Exact superset of every right row within ``window_deg`` of each left row.
        """
        dtype = ra_deg.dtype
        eps = torch.finfo(dtype).eps
        pad_dec = 8.0 * eps * 90.0
        pad_ra = 8.0 * eps * 360.0
        n = int(ra_deg.numel())
        dev = ra_deg.device
        w = window_deg + pad_dec
        zlo = self._zone_of(dec_deg - w)
        zhi = self._zone_of(dec_deg + w)
        sin_w = torch.sin(torch.deg2rad(window_deg.clamp(max=90.0)))
        cos_d = torch.cos(torch.deg2rad(dec_deg))
        ratio = sin_w / cos_d.clamp(min=torch.finfo(dtype).tiny)
        dra = torch.rad2deg(torch.asin(ratio.clamp(max=1.0))) + pad_ra
        full = (
            (ratio * (1.0 + 8.0 * eps) >= 1.0)
            | (w >= 90.0 - dec_deg.abs())
            | (dra >= 180.0)
        )
        ra = torch.remainder(ra_deg, 360.0)
        lo = torch.where(full, torch.zeros_like(ra), ra - dra)
        hi = torch.where(full, torch.full_like(ra, 360.0), ra + dra)
        # Piece 1 always; piece 2 only for rows whose interval crosses 0/360.
        wrap_lo, wrap_hi = lo < 0.0, hi > 360.0
        two = wrap_lo | wrap_hi
        first_lo, first_hi = lo.clamp(min=0.0), hi.clamp(max=360.0)
        rows2 = torch.nonzero(two, as_tuple=True)[0]
        second_lo = torch.where(
            wrap_lo[rows2], lo[rows2] + 360.0, torch.zeros_like(lo[rows2])
        )
        second_hi = torch.where(
            wrap_lo[rows2], torch.full_like(hi[rows2], 360.0), hi[rows2] - 360.0
        )
        prow = torch.cat((torch.arange(n, device=dev), rows2))
        plo = torch.cat((first_lo, second_lo))
        phi = torch.cat((first_hi, second_hi))
        klo = torch.floor(plo * (_RA_SCALE / 360.0)).to(torch.int64).clamp_(0, _RA_TOP)
        khi = torch.floor(phi * (_RA_SCALE / 360.0)).to(torch.int64).clamp_(0, _RA_TOP)
        # Ragged expansion over zones: one sync for the segment count.
        nz = (zhi - zlo + 1)[prow]
        total = int(nz.sum())
        piece = torch.repeat_interleave(
            torch.arange(prow.numel(), device=dev), nz, output_size=total
        )
        first = torch.cumsum(nz, 0) - nz
        zone = zlo[prow][piece] + (torch.arange(total, device=dev) - first[piece])
        base = zone << _RA_BITS
        start = torch.searchsorted(self.key, base | klo[piece], right=False)
        stop = torch.searchsorted(self.key, base | khi[piece], right=True)
        count = stop - start
        keep = count > 0
        return prow[piece][keep], start[keep], count[keep]


def _ra_key(ra_deg: Tensor) -> Tensor:
    wrapped = torch.remainder(ra_deg, 360.0)
    return torch.floor(wrapped * (_RA_SCALE / 360.0)).to(torch.int64).clamp_(0, _RA_TOP)


def _candidate_batches(count: Tensor, max_pairs: int) -> list[tuple[int, int]]:
    total = int(count.sum())
    return [(s, min(s + max_pairs, total)) for s in range(0, total, max_pairs)]


def zone_query(
    index: ZoneIndex,
    left_ra: Tensor,
    left_dec: Tensor,
    radius_left: Tensor,
    *,
    radius_right: Tensor | None = None,
    combine: Combine = "sum",
    find: Literal["all", "best", "count"] = "all",
    accept: Callable[[Tensor, Tensor], Tensor] | None = None,
    left_chunk_size: int = 1 << 16,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    stats: EngineStats | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pairs ``(left, right original, sep_arcsec)`` with ``sep <= combine(r_l, r_r)``.

    ``radius_right`` (original right order) is optional; the window of each
    left row is ``combine(r_l, max r_r)``.

    ``find="all"`` returns every pair ordered by ``(left, sep, right)``.
    ``"best"`` returns the nearest right row per left row, with ties going to the
    lowest right index, ordered by left. ``"count"`` returns ``(left rows, counts, empty)``
    for left rows with at least one pair. ``accept(left_rows, right_rows)`` is an extra
    predicate on global left / original right indices.
    """
    dev = left_ra.device
    left_xyz = unit_vectors(left_ra, left_dec)
    n_left = int(left_ra.numel())
    out_l: list[Tensor] = []
    out_r: list[Tensor] = []
    out_s: list[Tensor] = []
    if index.n == 0 or n_left == 0:
        return _empty(dev, left_ra.dtype)
    big = torch.iinfo(torch.int64).max
    rr_sorted = None if radius_right is None else radius_right.detach()[index.order]
    rr_max = 0.0 if rr_sorted is None else float(rr_sorted.max())
    for c0 in range(0, n_left, left_chunk_size):
        c1 = min(c0 + left_chunk_size, n_left)
        with torch.no_grad():
            rl = radius_left[c0:c1].detach()
            window_arcsec = combine_radii(rl, rr_max, combine)
            seg_row, seg_start, seg_count = index.segments(
                left_ra[c0:c1].detach(),
                left_dec[c0:c1].detach(),
                window_arcsec / 3600.0,
            )
            incl = torch.cumsum(seg_count, 0)
            excl = incl - seg_count
            if stats is not None:
                stats.left_rows += c1 - c0
                stats.segments += int(seg_row.numel())
            lx = left_xyz[c0:c1].detach()
            if find == "best":
                best_sep = torch.full(
                    (c1 - c0,), math.inf, dtype=left_ra.dtype, device=dev
                )
                best_j = torch.full((c1 - c0,), big, dtype=torch.int64, device=dev)
            elif find == "count":
                counts = torch.zeros(c1 - c0, dtype=torch.int64, device=dev)
            for k0, k1 in _candidate_batches(seg_count, max_pairs):
                k = torch.arange(k0, k1, device=dev)
                seg = torch.searchsorted(incl, k, right=True)
                j = seg_start[seg] + (k - excl[seg])
                i = seg_row[seg]
                rxyz = index.xyz[j].detach()
                diff = lx[i] - rxyz
                chord2 = (diff * diff).sum(-1)
                rr = rr_sorted[j] if rr_sorted is not None else 0.0
                thr = combine_radii(rl[i], rr, combine)
                half = torch.deg2rad(thr / 3600.0).clamp(max=math.pi) * 0.5
                # Padded prefilter, then the exact atan2 test on survivors.
                pre = (
                    chord2
                    <= (2.0 * torch.sin(half)) ** 2
                    * (1.0 + 64.0 * torch.finfo(chord2.dtype).eps)
                    + 1e-300
                )
                i, j, thr = i[pre], j[pre], thr[pre]
                sep = separation_arcsec(lx[i], index.xyz[j].detach())
                ok = sep <= thr
                i, j, sep = i[ok], j[ok], sep[ok]
                jo = index.order[j]
                if accept is not None and i.numel():
                    keep = accept(i + c0, jo)
                    i, j, jo, sep = i[keep], j[keep], jo[keep], sep[keep]
                if stats is not None:
                    stats.candidates += k1 - k0
                    stats.pairs += int(i.numel())
                if find == "best":
                    new_best = best_sep.scatter_reduce(
                        0, i, sep, "amin", include_self=True
                    )
                    tie = sep == new_best[i]
                    cand_j = _scatter_reduce_int(
                        torch.full_like(best_j, big), 0, i[tie], jo[tie], "amin"
                    )
                    kept_old = torch.where(
                        best_sep == new_best, best_j, torch.full_like(best_j, big)
                    )
                    best_j = torch.minimum(kept_old, cand_j)
                    best_sep = new_best
                elif find == "count":
                    counts.index_add_(0, i, torch.ones_like(i))
                else:
                    out_l.append(i + c0)
                    out_r.append(jo)
                    out_s.append(sep)
        if find == "best":
            rows = torch.nonzero(best_j.ne(big), as_tuple=True)[0]
            out_l.append(rows + c0)
            out_r.append(best_j[rows])
        elif find == "count":
            rows = torch.nonzero(counts, as_tuple=True)[0]
            out_l.append(rows + c0)
            out_r.append(counts[rows])
    if not out_l:
        return _empty(dev, left_ra.dtype)
    li = torch.cat(out_l)
    rj = torch.cat(out_r)
    if find == "count":
        return li, rj, torch.empty(0, dtype=left_ra.dtype, device=dev)
    if find == "best":
        sep = _pair_separation(index, left_xyz, li, rj)
        return li, rj, sep
    # "all": order by (left, sep, right). Separations are recomputed with autograd.
    sep_d = torch.cat(out_s)
    order = torch.argsort(rj, stable=True)
    order = order[torch.argsort(sep_d[order], stable=True)]
    order = order[torch.argsort(li[order], stable=True)]
    li, rj = li[order], rj[order]
    return li, rj, _pair_separation(index, left_xyz, li, rj)


def _pair_separation(
    index: ZoneIndex, left_xyz: Tensor, li: Tensor, rj_orig: Tensor
) -> Tensor:
    right = index.xyz[index.inverse_order()[rj_orig]]
    if left_xyz.requires_grad or index.xyz.requires_grad:
        return separation_arcsec(left_xyz[li], right)
    with torch.no_grad():
        return separation_arcsec(left_xyz[li], right)


def _empty(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    e = torch.empty(0, dtype=torch.int64, device=device)
    return e, e, torch.empty(0, dtype=dtype, device=device)
