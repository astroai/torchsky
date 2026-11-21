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
    """Great-circle angle between unit vectors, atan2 form, in arcsec.

    Written element-wise (no reductions over the vector axis), so a pair's value
    does not depend on how pairs are batched: results are chunk-invariant.
    """
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    cross = torch.sqrt(cx * cx + cy * cy + cz * cz)
    dot = (ax * bx + ay * by + az * bz).clamp(-1.0, 1.0)
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
    self_upper: bool = False,
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
    predicate on global left / original right indices; ``self_upper`` keeps
    only ``right > left`` (self-joins, left and right the same catalogue).

    CPU float64 queries without ``accept`` run in the native kernel
    (``crossmatch.cpp``); everything else, and every other device, runs the
    torch path below. Both produce the same pairs.
    """
    dev = left_ra.device
    n_left = int(left_ra.numel())
    ext = _native_ext()
    if (
        ext is not None
        and accept is None
        and stats is None
        and dev.type == "cpu"
        and left_ra.dtype == torch.float64
        and index.xyz.dtype == torch.float64
        and index.n
        and n_left
    ):
        return _zone_query_native(
            ext,
            index,
            left_ra,
            left_dec,
            radius_left,
            radius_right,
            combine,
            find,
            self_upper,
        )
    return _zone_query_torch(
        index,
        left_ra,
        left_dec,
        radius_left,
        radius_right=radius_right,
        combine=combine,
        find=find,
        accept=accept,
        self_upper=self_upper,
        left_chunk_size=left_chunk_size,
        max_pairs=max_pairs,
        stats=stats,
        pad_tiles=torch.compiler.is_compiling(),
    )


def _candidate_tile(
    k: Tensor,
    incl: Tensor,
    excl: Tensor,
    seg_row: Tensor,
    seg_start: Tensor,
    lx: Tensor,
    rxyz: Tensor,
    rl: Tensor,
    rr_sorted: Tensor | None,
    combine: Combine,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Candidates ``k`` of the segment runs, tested (fixed shape; ``fullgraph=True``).

    Returns ``(i, j, chord2, keep)``: chunk-local left row, sorted right row,
    the exactly rounded chord^2 and whether the pair is within its radius (the
    padded chord prefilter and the exact atan2 separation). Candidates at or past
    the candidate count (tile padding) are not kept.
    """
    in_range = k < incl[-1]
    seg = torch.searchsorted(incl, k, right=True).clamp_(max=incl.numel() - 1)
    j = (seg_start[seg] + (k - excl[seg])).clamp_(max=rxyz.shape[0] - 1)
    i = seg_row[seg]
    li = lx[i]
    rj = rxyz[j]
    diff = li - rj
    # Element-wise (exactly rounded, batch-independent): the
    # ordering key and tie rule must not depend on chunking.
    chord2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2]
    rr = rr_sorted[j] if rr_sorted is not None else 0.0
    thr = combine_radii(rl[i], rr, combine)
    half = torch.deg2rad(thr / 3600.0).clamp(max=math.pi) * 0.5
    # Padded prefilter, then the exact atan2 test.
    pre = (
        chord2
        <= (2.0 * torch.sin(half)) ** 2 * (1.0 + 64.0 * torch.finfo(chord2.dtype).eps)
        + 1e-300
    )
    keep = in_range & pre & (separation_arcsec(li, rj) <= thr)
    return i, j, chord2, keep


@torch.compiler.disable(recursive=False)
def _zone_query_torch(
    index: ZoneIndex,
    left_ra: Tensor,
    left_dec: Tensor,
    radius_left: Tensor,
    *,
    radius_right: Tensor | None,
    combine: Combine,
    find: Literal["all", "best", "count"],
    accept: Callable[[Tensor, Tensor], Tensor] | None,
    self_upper: bool,
    left_chunk_size: int,
    max_pairs: int,
    stats: EngineStats | None,
    pad_tiles: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Torch path of :func:`zone_query`.

    Runs eagerly (skipped by ``torch.compile``): segment expansion, candidate
    counts, compaction of the kept pairs and the output ordering are
    data-dependent. The candidate test is the fixed-shape
    :func:`_candidate_tile`, which Dynamo compiles once; under compilation
    (``pad_tiles``) every candidate batch is padded to a common length.
    """
    dev = left_ra.device
    n_left = int(left_ra.numel())
    left_xyz = unit_vectors(left_ra, left_dec)
    if self_upper:
        accept = _upper if accept is None else _both(accept, _upper)
    out_l: list[Tensor] = []
    out_r: list[Tensor] = []
    out_s: list[Tensor] = []
    if index.n == 0 or n_left == 0:
        return _empty(dev, left_ra.dtype)
    big = torch.iinfo(torch.int64).max
    rxyz = index.xyz.detach()
    rr_sorted = None if radius_right is None else radius_right.detach()[index.order]
    rr_max = 0.0 if rr_sorted is None else float(rr_sorted.max())
    for c0 in range(0, n_left, left_chunk_size):
        c1 = min(c0 + left_chunk_size, n_left)
        chunk_start = len(out_l)
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
            batches = _candidate_batches(seg_count, max_pairs)
            tile = 0
            if pad_tiles and batches:
                from ..map.operators.geometry_drizzle import _padded_tile_length

                tile = _padded_tile_length(batches[-1][1], max_pairs)
            for k0, k1 in batches:
                k = torch.arange(k0, k0 + tile if pad_tiles else k1, device=dev)
                i, j, chord2, keep = _candidate_tile(
                    k, incl, excl, seg_row, seg_start, lx, rxyz, rl, rr_sorted, combine
                )
                i, j, chord2 = i[keep], j[keep], chord2[keep]
                jo = index.order[j]
                if accept is not None and i.numel():
                    keep = accept(i + c0, jo)
                    i, j, jo, chord2 = i[keep], j[keep], jo[keep], chord2[keep]
                if stats is not None:
                    stats.candidates += k1 - k0
                    stats.pairs += int(i.numel())
                if find == "best":
                    # Nearest by chord^2 (monotone in the angle, exactly rounded).
                    new_best = best_sep.scatter_reduce(
                        0, i, chord2, "amin", include_self=True
                    )
                    tie = chord2 == new_best[i]
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
                    out_s.append(chord2)
        if find == "all" and len(out_l) > chunk_start:
            # Order this chunk by (left, chord^2, right) now: chunks are contiguous in
            # left index, so the concatenation is ordered with no global sort
            # (and no global sort temporaries, which ran out of GPU memory).
            cl_ = torch.cat(out_l[chunk_start:])
            cr_ = torch.cat(out_r[chunk_start:])
            cs_ = torch.cat(out_s[chunk_start:])
            order = torch.argsort(cr_, stable=True)
            order = order[torch.argsort(cs_[order], stable=True)]
            order = order[torch.argsort(cl_[order], stable=True)]
            del out_l[chunk_start:], out_r[chunk_start:], out_s[chunk_start:]
            out_l.append(cl_[order])
            out_r.append(cr_[order])
            out_s.append(cs_[order])
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
    del out_s
    # Separations of the final pairs: with autograd in one call, otherwise in
    # fixed blocks aligned to output positions (bit-identical for any chunking,
    # bounded memory: pair-sized gathers ran a 10.75 GB GPU out of memory).
    if left_xyz.requires_grad or index.xyz.requires_grad:
        return li, rj, _pair_separation(index, left_xyz, li, rj)
    return li, rj, _blocked_separation(index, left_xyz, li, rj)


_COMBINE_CODE = {"sum": 0, "quadrature": 1, "max": 2}


def _native_ext():
    from .._native import cpp

    if cpp is None:
        return None
    from .. import _cpp_ext

    return _cpp_ext if hasattr(_cpp_ext, "xmatch_count") else None


def _upper(left_rows: Tensor, right_rows: Tensor) -> Tensor:
    return right_rows > left_rows


def _both(a: Callable[[Tensor, Tensor], Tensor], b: Callable[[Tensor, Tensor], Tensor]):
    return lambda li, rj: a(li, rj) & b(li, rj)


def _zone_query_native(
    ext,
    index: ZoneIndex,
    left_ra: Tensor,
    left_dec: Tensor,
    radius_left: Tensor,
    radius_right: Tensor | None,
    combine: Combine,
    find: str,
    self_upper: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """CPU float64 path: the native kernel finds pairs; torch computes separations."""
    n = int(left_ra.numel())
    f64, i64 = torch.float64, torch.int64
    with torch.no_grad():
        ra = left_ra.detach().contiguous()
        dec = left_dec.detach().contiguous()
        rl = radius_left.detach().to(f64).contiguous()
        rr = (
            radius_right.detach().to(f64)[index.order].contiguous()
            if radius_right is not None
            else torch.empty(0, dtype=f64)
        )
        rr_max = float(rr.max()) if rr.numel() else 0.0
        # Visit left rows in (zone, RA) order: neighbouring queries touch the
        # same right rows, which keeps them in cache.
        perm = torch.argsort(
            (index._zone_of(dec) << _RA_BITS) | _ra_key(ra)
        ).contiguous()
        common = (
            index.key,
            index.xyz.detach().contiguous(),
            index.order,
            rr,
            index.zone_height,
            index.n_zones,
            rr_max,
            ra,
            dec,
            rl,
            perm,
            _COMBINE_CODE[combine],
        )
        threads = torch.get_num_threads()
        if find == "best":
            j = torch.empty(n, dtype=i64)
            sep = torch.empty(n, dtype=f64)
            ext.xmatch_best(*common, j, sep, threads)
            if self_upper:
                raise ValueError("self_upper applies to find='all' or 'count'")
            rows = torch.nonzero(j >= 0, as_tuple=True)[0]
            li, rj, sep = rows, j[rows], sep[rows]
        else:
            counts = torch.empty(n, dtype=i64)
            ext.xmatch_count(*common, self_upper, 0, counts, threads)
            if find == "count":
                rows = torch.nonzero(counts, as_tuple=True)[0]
                return rows, counts[rows], torch.empty(0, dtype=left_ra.dtype)
            total = int(counts.sum())
            offsets = torch.cumsum(counts, 0) - counts
            rj = torch.empty(total, dtype=i64)
            sep = torch.empty(total, dtype=f64)
            ext.xmatch_fill(*common, self_upper, 0, offsets, rj, sep, threads)
            li = torch.repeat_interleave(torch.arange(n), counts, output_size=total)
    if left_ra.requires_grad or left_dec.requires_grad or index.xyz.requires_grad:
        # Separations of matched pairs, differentiable through the inputs.
        sep = _pair_separation(index, unit_vectors(left_ra[li], left_dec[li]), None, rj)
    return li, rj, sep


def native_knn(
    index: ZoneIndex,
    ra: Tensor,
    dec: Tensor,
    start_radius_arcsec: Tensor | None,
    k: int,
    cap_arcsec: float,
) -> tuple[Tensor, Tensor] | None:
    """Native exact k nearest neighbours (``None`` when the kernel does not apply).

    Without ``start_radius_arcsec`` the kernel picks each row's start radius from
    the local density of the index (expected ``k`` rows), then doubles it.
    """
    ext = _native_ext()
    if (
        ext is None
        or ra.device.type != "cpu"
        or ra.dtype != torch.float64
        or index.xyz.dtype != torch.float64
    ):
        return None
    n = int(ra.numel())
    with torch.no_grad():
        ra_c, dec_c = ra.detach().contiguous(), dec.detach().contiguous()
        perm = torch.argsort(
            (index._zone_of(dec_c) << _RA_BITS) | _ra_key(ra_c)
        ).contiguous()
        j = torch.empty(n * k, dtype=torch.int64)
        sep = torch.empty(n * k, dtype=torch.float64)
        ext.xmatch_knn(
            index.key,
            index.xyz.detach().contiguous(),
            index.order,
            index.zone_height,
            index.n_zones,
            ra_c,
            dec_c,
            torch.empty(0, dtype=torch.float64)
            if start_radius_arcsec is None
            else start_radius_arcsec.detach().to(torch.float64).contiguous(),
            perm,
            int(k),
            float(cap_arcsec),
            j,
            sep,
            torch.get_num_threads(),
        )
    return j.reshape(n, k), sep.reshape(n, k)


def _blocked_separation(
    index: ZoneIndex,
    left_xyz: Tensor,
    li: Tensor,
    rj_orig: Tensor,
    block: int = 1 << 20,
) -> Tensor:
    out = torch.empty(li.shape, dtype=left_xyz.dtype, device=li.device)
    inv = index.inverse_order()
    with torch.no_grad():
        for b0 in range(0, int(li.numel()), block):
            b1 = min(b0 + block, int(li.numel()))
            out[b0:b1] = separation_arcsec(
                left_xyz[li[b0:b1]], index.xyz[inv[rj_orig[b0:b1]]]
            )
    return out


def _pair_separation(
    index: ZoneIndex, left_xyz: Tensor, li: Tensor | None, rj_orig: Tensor
) -> Tensor:
    """Separations of pairs; ``li=None`` means ``left_xyz`` is already gathered."""
    right = index.xyz[index.inverse_order()[rj_orig]]
    left = left_xyz if li is None else left_xyz[li]
    if left.requires_grad or right.requires_grad:
        return separation_arcsec(left, right)
    with torch.no_grad():
        return separation_arcsec(left, right)


def _empty(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    e = torch.empty(0, dtype=torch.int64, device=device)
    return e, e, torch.empty(0, dtype=dtype, device=device)
