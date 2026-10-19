"""Tensor-native catalog helpers."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, cast

import torch

from .._dtype import work_dtype as _work_dtype
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from ..sphere import great_circle_distance
from ..sphere import healpix as _healpix
from .index import HealpixIndex
from .index import (
    candidate_coarse_pixels_for_left_pixel as _candidate_coarse_pixels_for_left_pixel,
)
from .index import nested_neighbor_rings as _nested_neighbor_rings
from .index import nside_for_cap as _nside_for_cap
from .index import _BATCHED_RING_HOPS_MAX


def _identity_partition(partition: Any) -> Any:
    return partition


class CatalogStreamDataset(IterableDataset):
    """Wrap a partition iterator into a PyTorch IterableDataset."""

    def __init__(
        self,
        partitions: Iterable[object] | Callable[[], Iterable[object]],
        loader: Callable[[object], object] = _identity_partition,
    ):
        self._partitions = partitions
        self._loader = loader

    def __iter__(self) -> Iterator[object]:
        worker = get_worker_info()
        partitions = (
            self._partitions() if callable(self._partitions) else self._partitions
        )
        if worker is not None:
            partitions = islice(iter(partitions), worker.id, None, worker.num_workers)
        for partition in partitions:
            yield self._loader(partition)

    @classmethod
    def from_arrow(
        cls,
        source: Any,
        *,
        batch_size: int = 65536,
        columns: tuple[str, str] = ("ra", "dec"),
    ) -> CatalogStreamDataset:
        """Build a CatalogStreamDataset that yields (ra, dec) Tensor pairs from Arrow.

        Parameters
        ----------
        source
            A ``pyarrow.Table``, ``pyarrow.ipc.RecordBatchReader``, or a path
            string / ``pathlib.Path`` to an Arrow IPC Feather file.
        batch_size
            Row count per yielded batch.
        columns
            Column names for RA (degrees) and Dec (degrees).
        """
        import pyarrow as pa

        if isinstance(source, (str, Path)):
            source = pa.ipc.open_file(str(source)).read_all()
        if isinstance(source, pa.Table):
            reader = source.to_batches(max_chunksize=batch_size)
        elif isinstance(source, pa.ipc.RecordBatchReader):
            reader = source
        else:
            batches = list(source) if hasattr(source, "__iter__") else [source]
            reader = iter(batches)

        def _loader(batch: pa.RecordBatch) -> tuple[Tensor, Tensor]:
            col_ra, col_dec = columns
            ra_np = batch.column(col_ra).to_numpy()
            dec_np = batch.column(col_dec).to_numpy()
            return (
                torch.from_numpy(ra_np.copy().astype("f8", copy=False)),
                torch.from_numpy(dec_np.copy().astype("f8", copy=False)),
            )

        return cls(partitions=lambda: iter(reader), loader=_loader)


@dataclass
class CrossmatchResult:
    left_index: Tensor
    right_index: Tensor
    separation_arcsec: Tensor

    @property
    def left_indices(self) -> Tensor:
        return self.left_index

    @property
    def right_indices(self) -> Tensor:
        return self.right_index


@dataclass
class _CrossmatchStats:
    left_points: int
    right_points: int
    candidate_groups: int = 0
    candidate_rows_considered: int = 0
    pairs_evaluated: int = 0
    matched_rows: int = 0


def _candidate_dec_window(
    right_dec_sorted: Tensor,
    *,
    dec_min_deg: Tensor,
    dec_max_deg: Tensor,
    radius_deg: float,
) -> tuple[int, int]:
    lower = max(float(dec_min_deg.item()) - radius_deg, -90.0)
    upper = min(float(dec_max_deg.item()) + radius_deg, 90.0)
    start = int(torch.searchsorted(right_dec_sorted, lower, right=False).item())
    stop = int(torch.searchsorted(right_dec_sorted, upper, right=True).item())
    return start, stop


def _empty_crossmatch(device: torch.device) -> CrossmatchResult:
    empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
    empty_flt = torch.empty(0, dtype=_work_dtype(device), device=device)
    return CrossmatchResult(empty_i64, empty_i64, empty_flt)


def _finalize_crossmatch(
    left_hits: list[Tensor],
    right_hits: list[Tensor],
    sep_hits: list[Tensor],
    *,
    device: torch.device,
    stats: _CrossmatchStats,
    sort_within_left: bool = False,
) -> CrossmatchResult:
    if not left_hits:
        return _empty_crossmatch(device)

    left_index = torch.cat(left_hits)
    right_index = torch.cat(right_hits)
    separation_arcsec = torch.cat(sep_hits)
    if sort_within_left:
        order = torch.argsort(right_index, stable=True)
        left_index = left_index[order]
        right_index = right_index[order]
        separation_arcsec = separation_arcsec[order]
        order = torch.argsort(separation_arcsec, stable=True)
        left_index = left_index[order]
        right_index = right_index[order]
        separation_arcsec = separation_arcsec[order]
    order = torch.argsort(left_index, stable=True)
    left_index = left_index[order]
    right_index = right_index[order]
    separation_arcsec = separation_arcsec[order]
    if sort_within_left and int(left_index.numel()) > 1:
        dup = (left_index[1:] == left_index[:-1]) & (
            right_index[1:] == right_index[:-1]
        )
        keep = torch.ones(left_index.shape[0], dtype=torch.bool, device=device)
        keep[1:] = ~dup
        left_index = left_index[keep]
        right_index = right_index[keep]
        separation_arcsec = separation_arcsec[keep]
    stats.matched_rows = int(left_index.numel())
    return CrossmatchResult(left_index, right_index, separation_arcsec)


def _normalize_radius_arcsec(
    radius_arcsec: float | Tensor,
    n_left: int,
    device: torch.device,
) -> tuple[Tensor, float]:
    """Return per-left radii and the prune radius (max)."""
    dtype = _work_dtype(device)
    radius = torch.as_tensor(radius_arcsec, dtype=dtype, device=device)
    if radius.ndim == 0:
        value = float(radius.item())
        if value <= 0.0:
            raise ValueError("radius_arcsec must be positive")
        return torch.full((n_left,), value, dtype=dtype, device=device), value
    flat = radius.reshape(-1)
    if int(flat.numel()) != n_left:
        raise ValueError("radius_arcsec must be a scalar or match the left catalog")
    if bool((flat <= 0.0).any()):
        raise ValueError("radius_arcsec must be positive")
    return flat, float(flat.max().item())


def _covering_hops(radius_rad: float, pixrad: float) -> int:
    return max(2, int(math.ceil(radius_rad / max(pixrad, 1e-12))))


def _unique_min_pairs(
    query: Tensor, right: Tensor, sep: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """One pair per query: min separation, then min right index."""
    if query.numel() <= 1:
        return query, right, sep
    order = torch.argsort(right, stable=True)
    query, right, sep = query[order], right[order], sep[order]
    order = torch.argsort(sep, stable=True)
    query, right, sep = query[order], right[order], sep[order]
    order = torch.argsort(query, stable=True)
    query, right, sep = query[order], right[order], sep[order]
    keep = torch.ones(query.shape[0], dtype=torch.bool, device=query.device)
    keep[1:] = query[1:] != query[:-1]
    return query[keep], right[keep], sep[keep]


def _accumulate_ring_chunk(
    *,
    left_ra: Tensor,
    left_dec: Tensor,
    left_abs: Tensor,
    radius_deg: Tensor,
    right_ra: Tensor,
    right_dec: Tensor,
    coarse_index: HealpixIndex,
    hops: int,
    find: str,
    left_hits: list[Tensor],
    right_hits: list[Tensor],
    sep_hits: list[Tensor],
    stats: _CrossmatchStats | None,
) -> None:
    left_pix = _healpix.ang2pix(
        coarse_index.nside,
        left_ra,
        left_dec,
        nest=True,
        lonlat=True,
    ).to(torch.int64)
    rings = _nested_neighbor_rings(coarse_index.nside, left_pix, hops)
    rings_flat = rings.reshape(-1)
    if not coarse_index.nest and rings_flat.numel() > 0:
        valid_r = rings_flat >= 0
        rings_lookup = rings_flat.clone()
        rings_lookup[valid_r] = _healpix.nest2ring(
            coarse_index.nside, rings_flat[valid_r]
        )
    else:
        rings_lookup = rings_flat
    n_left = int(left_ra.numel())
    query, right_ids = coarse_index.expand(rings_lookup)
    if stats is not None:
        stats.candidate_groups += 1
    if query.numel() == 0:
        return
    query = torch.remainder(query, n_left)
    sep = great_circle_distance(
        left_ra[query],
        left_dec[query],
        right_ra[right_ids],
        right_dec[right_ids],
        degrees=True,
    )
    if stats is not None:
        stats.pairs_evaluated += int(query.numel())
        stats.candidate_rows_considered += int(right_ids.numel())
    if find == "all":
        keep = sep <= radius_deg[query]
        if bool(keep.any()):
            kept = query[keep]
            left_hits.append(left_abs[kept])
            right_hits.append(right_ids[keep])
            sep_hits.append(sep[keep] * 3600.0)
        return
    query, right_ids, sep = _unique_min_pairs(query, right_ids, sep)
    mask = sep <= radius_deg[query]
    if bool(mask.any()):
        left_hits.append(left_abs[query[mask]])
        right_hits.append(right_ids[mask])
        sep_hits.append(sep[mask] * 3600.0)


def _choose_default_coarse_nside(
    left_ra: Tensor,
    left_dec: Tensor,
    right_ra: Tensor,
    right_dec: Tensor,
    *,
    radius_arcsec: float,
) -> int:
    del left_ra, left_dec, right_ra, right_dec
    if radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive")
    if radius_arcsec >= 180.0 * 3600.0:
        return 1
    return _nside_for_cap(math.radians(radius_arcsec / 3600.0))


def _build_coarse_healpix_index(
    right_ra: Tensor,
    right_dec: Tensor,
    *,
    coarse_nside: int,
) -> HealpixIndex:
    return HealpixIndex.build(right_ra, right_dec, nside=coarse_nside, nest=True)


def _crossmatch_sky_declination_pruned(
    left_ra: Tensor,
    left_dec: Tensor,
    right_ra: Tensor,
    right_dec: Tensor,
    *,
    radius_arcsec: float,
    left_chunk_size: int,
    right_chunk_size: int,
    return_stats: bool = False,
) -> CrossmatchResult | tuple[CrossmatchResult, _CrossmatchStats]:
    device = left_ra.device
    stats = _CrossmatchStats(
        left_points=int(left_ra.numel()),
        right_points=int(right_ra.numel()),
    )
    radius_deg = radius_arcsec / 3600.0
    right_order = torch.argsort(right_dec, stable=True)
    right_ra_sorted = right_ra[right_order]
    right_dec_sorted = right_dec[right_order]

    left_hits: list[Tensor] = []
    right_hits: list[Tensor] = []
    sep_hits: list[Tensor] = []

    for start, left_ra_chunk, left_dec_chunk in zip(
        range(0, int(left_ra.numel()), left_chunk_size),
        left_ra.split(left_chunk_size),
        left_dec.split(left_chunk_size),
        strict=False,
    ):
        cand_start, cand_stop = _candidate_dec_window(
            right_dec_sorted,
            dec_min_deg=left_dec_chunk.min(),
            dec_max_deg=left_dec_chunk.max(),
            radius_deg=radius_deg,
        )
        if cand_start >= cand_stop:
            continue

        ra_chunk = left_ra_chunk.unsqueeze(1)
        dec_chunk = left_dec_chunk.unsqueeze(1)
        chunk_len = left_ra_chunk.shape[0]
        best_sep = torch.full(
            (chunk_len,),
            torch.inf,
            dtype=torch.float64,
            device=device,
        )
        best_idx = torch.full(
            (chunk_len,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        candidate_count = cand_stop - cand_start
        stats.candidate_groups += 1
        stats.candidate_rows_considered += candidate_count

        cand_right_ra = right_ra_sorted[cand_start:cand_stop]
        cand_right_dec = right_dec_sorted[cand_start:cand_stop]
        cand_right_order = right_order[cand_start:cand_stop]
        if cand_right_ra.numel() == 0:
            continue

        for right_ra_chunk, right_dec_chunk, right_order_chunk in zip(
            cand_right_ra.split(right_chunk_size),
            cand_right_dec.split(right_chunk_size),
            cand_right_order.split(right_chunk_size),
            strict=False,
        ):
            right_chunk_len = right_ra_chunk.shape[0]
            sep = great_circle_distance(
                ra_chunk,
                dec_chunk,
                right_ra_chunk.unsqueeze(0),
                right_dec_chunk.unsqueeze(0),
                degrees=True,
            )
            stats.pairs_evaluated += chunk_len * right_chunk_len
            cand_sep, cand_idx = sep.min(dim=1)
            improved = cand_sep < best_sep
            if bool(improved.any()):
                best_sep = torch.where(improved, cand_sep, best_sep)
                best_idx = torch.where(
                    improved,
                    right_order_chunk[cand_idx.to(torch.int64)],
                    best_idx,
                )

        mask = best_sep <= radius_deg
        if bool(mask.any()):
            stop = start + chunk_len
            left_hits.append(
                torch.arange(start, stop, dtype=torch.int64, device=device)[mask]
            )
            right_hits.append(best_idx[mask])
            sep_hits.append(best_sep[mask] * 3600.0)

    result = _finalize_crossmatch(
        left_hits,
        right_hits,
        sep_hits,
        device=device,
        stats=stats,
    )
    if return_stats:
        return result, stats
    return result


def _ragged_candidate_batches(
    starts: Tensor,
    stops: Tensor,
    *,
    max_pairs: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield bounded ``(left-relative, right-sorted)`` candidate pairs."""
    counts = (stops - starts).to(torch.int64)
    cursor = 0
    rows = int(counts.numel())
    while cursor < rows:
        first_count = int(counts[cursor].item())
        if first_count > max_pairs:
            first_start = int(starts[cursor].item())
            first_stop = int(stops[cursor].item())
            for right_start in range(first_start, first_stop, max_pairs):
                right_stop = min(right_start + max_pairs, first_stop)
                right_sorted = torch.arange(
                    right_start,
                    right_stop,
                    dtype=torch.int64,
                    device=starts.device,
                )
                yield torch.full_like(right_sorted, cursor), right_sorted
            cursor += 1
            continue

        cumulative = torch.cumsum(counts[cursor:], dim=0)
        limit = torch.tensor(max_pairs, dtype=torch.int64, device=starts.device)
        take = int(torch.searchsorted(cumulative, limit, right=True).item())
        take = max(take, 1)
        stop_cursor = min(cursor + take, rows)
        group_counts = counts[cursor:stop_cursor]
        total = int(group_counts.sum().item())
        if total > 0:
            left_rows = torch.arange(
                cursor,
                stop_cursor,
                dtype=torch.int64,
                device=starts.device,
            )
            left_relative = torch.repeat_interleave(left_rows, group_counts)
            repeated_starts = torch.repeat_interleave(
                starts[cursor:stop_cursor], group_counts
            )
            group_bases = torch.repeat_interleave(
                torch.cumsum(group_counts, dim=0) - group_counts,
                group_counts,
            )
            right_sorted = repeated_starts + (
                torch.arange(total, dtype=torch.int64, device=starts.device)
                - group_bases
            )
            yield left_relative, right_sorted
        cursor = stop_cursor


def _best_hits(
    left_index: Tensor,
    right_index: Tensor,
    separation_arcsec: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Keep the deterministic nearest hit for each left row."""
    return _unique_min_pairs(left_index, right_index, separation_arcsec)


def _crossmatch_sky_declination_indexed(
    left_ra: Tensor,
    left_dec: Tensor,
    right_ra: Tensor,
    right_dec: Tensor,
    *,
    radius_arcsec: Tensor,
    left_chunk_size: int,
    right_chunk_size: int,
    find: str = "best",
    return_stats: bool = False,
) -> CrossmatchResult | tuple[CrossmatchResult, _CrossmatchStats]:
    """Crossmatch through exact per-row declination windows."""
    device = left_ra.device
    stats = _CrossmatchStats(
        left_points=int(left_ra.numel()),
        right_points=int(right_ra.numel()),
    )
    radius_deg = radius_arcsec / 3600.0
    right_order = torch.argsort(right_dec, stable=True)
    right_dec_sorted = right_dec[right_order]
    left_hits: list[Tensor] = []
    right_hits: list[Tensor] = []
    sep_hits: list[Tensor] = []

    for start, left_ra_chunk, left_dec_chunk, radius_chunk, radius_deg_chunk in zip(
        range(0, int(left_ra.numel()), left_chunk_size),
        left_ra.split(left_chunk_size),
        left_dec.split(left_chunk_size),
        radius_arcsec.split(left_chunk_size),
        radius_deg.split(left_chunk_size),
        strict=True,
    ):
        lower = torch.clamp(left_dec_chunk - radius_deg_chunk, min=-90.0, max=90.0)
        upper = torch.clamp(left_dec_chunk + radius_deg_chunk, min=-90.0, max=90.0)
        starts = torch.searchsorted(right_dec_sorted, lower, right=False)
        stops = torch.searchsorted(right_dec_sorted, upper, right=True)
        stats.candidate_rows_considered += int((stops - starts).sum().item())

        for left_relative, right_sorted in _ragged_candidate_batches(
            starts,
            stops,
            max_pairs=right_chunk_size,
        ):
            stats.candidate_groups += 1
            stats.pairs_evaluated += int(left_relative.numel())
            right_index = right_order[right_sorted]
            separation_arcsec = (
                great_circle_distance(
                    left_ra_chunk[left_relative],
                    left_dec_chunk[left_relative],
                    right_ra[right_index],
                    right_dec[right_index],
                    degrees=True,
                )
                * 3600.0
            )
            mask = separation_arcsec <= radius_chunk[left_relative]
            if not bool(mask.any()):
                continue
            batch_left = left_relative[mask] + start
            batch_right = right_index[mask]
            batch_sep = separation_arcsec[mask]
            if find == "best":
                batch_left, batch_right, batch_sep = _best_hits(
                    batch_left,
                    batch_right,
                    batch_sep,
                )
            left_hits.append(batch_left)
            right_hits.append(batch_right)
            sep_hits.append(batch_sep)

    result = _finalize_crossmatch(
        left_hits,
        right_hits,
        sep_hits,
        device=device,
        stats=stats,
        sort_within_left=True,
    )
    if find == "best":
        left_index, right_index, separation_arcsec = _best_hits(
            result.left_index,
            result.right_index,
            result.separation_arcsec,
        )
        result = CrossmatchResult(left_index, right_index, separation_arcsec)
        stats.matched_rows = int(left_index.numel())
    if return_stats:
        return result, stats
    return result


def _crossmatch_sky_coarse_indexed(
    left_ra: Tensor,
    left_dec: Tensor,
    right_ra: Tensor,
    right_dec: Tensor,
    *,
    radius_arcsec: float | Tensor,
    left_chunk_size: int,
    right_chunk_size: int,
    find: str = "best",
    return_stats: bool = False,
    right_index: HealpixIndex | None = None,
) -> CrossmatchResult | tuple[CrossmatchResult, _CrossmatchStats]:
    device = left_ra.device
    work_dtype = _work_dtype(device)
    stats = _CrossmatchStats(
        left_points=int(left_ra.numel()),
        right_points=int(right_ra.numel()),
    )
    radius_left, radius_max = _normalize_radius_arcsec(
        radius_arcsec, int(left_ra.numel()), device
    )
    if right_index is None:
        coarse_nside = _choose_default_coarse_nside(
            left_ra,
            left_dec,
            right_ra,
            right_dec,
            radius_arcsec=radius_max,
        )
        coarse_index = _build_coarse_healpix_index(
            right_ra, right_dec, coarse_nside=coarse_nside
        )
    else:
        if int(right_index.lon.numel()) != int(right_ra.numel()):
            raise ValueError("right_index does not match the right catalog")
        if right_index.lon.device != device:
            right_index = right_index.to(device)
        if not torch.equal(right_index.lon, right_ra) or not torch.equal(
            right_index.lat, right_dec
        ):
            raise ValueError("right_index coordinates do not match the right catalog")
        coarse_index = right_index
        coarse_nside = coarse_index.nside
    pixrad = _healpix.max_pixrad(coarse_nside, degrees=False)
    match_rad = math.radians(radius_max / 3600.0)
    hops = _covering_hops(match_rad + (2.0 * pixrad), pixrad)
    use_rings = hops <= _BATCHED_RING_HOPS_MAX
    candidate_cache: dict[tuple[int, float], Tensor] = {}

    left_hits: list[Tensor] = []
    right_hits: list[Tensor] = []
    sep_hits: list[Tensor] = []

    for start, left_ra_chunk, left_dec_chunk in zip(
        range(0, int(left_ra.numel()), left_chunk_size),
        left_ra.split(left_chunk_size),
        left_dec.split(left_chunk_size),
        strict=False,
    ):
        stop = start + left_ra_chunk.shape[0]
        left_abs_chunk = torch.arange(start, stop, dtype=torch.int64, device=device)
        radius_chunk = radius_left[start:stop]
        radius_chunk_deg = radius_chunk / 3600.0
        if use_rings:
            _accumulate_ring_chunk(
                left_ra=left_ra_chunk,
                left_dec=left_dec_chunk,
                left_abs=left_abs_chunk,
                radius_deg=radius_chunk_deg,
                right_ra=right_ra,
                right_dec=right_dec,
                coarse_index=coarse_index,
                hops=hops,
                find=find,
                left_hits=left_hits,
                right_hits=right_hits,
                sep_hits=sep_hits,
                stats=stats,
            )
            continue
        left_coarse_pixel = _healpix.ang2pix(
            coarse_nside,
            left_ra_chunk,
            left_dec_chunk,
            nest=True,
            lonlat=True,
        ).to(torch.int64)

        left_order = torch.argsort(left_coarse_pixel, stable=True)
        coarse_sorted = left_coarse_pixel[left_order]
        unique_pixels, counts = torch.unique_consecutive(
            coarse_sorted, return_counts=True
        )
        offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.cumsum(counts[:-1].to(torch.int64), dim=0),
            )
        )

        for coarse_pixel, offset, count in zip(
            unique_pixels.tolist(),
            offsets.tolist(),
            counts.tolist(),
            strict=False,
        ):
            rel_sorted = left_order[offset : offset + count]
            group_ra = left_ra_chunk[rel_sorted]
            group_dec = left_dec_chunk[rel_sorted]
            group_abs = left_abs_chunk[rel_sorted]
            group_radius = radius_chunk[rel_sorted]
            cover_arcsec = float(group_radius.max().item())
            cover_deg = cover_arcsec / 3600.0
            candidate_radius_rad = math.radians(cover_deg) + (2.0 * pixrad)
            cache_key = (int(coarse_pixel), cover_arcsec)
            candidate_pixels = candidate_cache.get(cache_key)
            if candidate_pixels is None:
                candidate_pixels = _candidate_coarse_pixels_for_left_pixel(
                    coarse_nside,
                    coarse_pixel,
                    radius_rad=candidate_radius_rad,
                    device=device,
                )
                if not coarse_index.nest and candidate_pixels.numel() > 0:
                    candidate_pixels = _healpix.nest2ring(
                        coarse_nside, candidate_pixels
                    )
                candidate_cache[cache_key] = candidate_pixels

            candidate_right_idx = coarse_index.lookup(candidate_pixels)
            stats.candidate_groups += 1
            stats.candidate_rows_considered += int(candidate_right_idx.numel())
            if candidate_right_idx.numel() == 0:
                continue

            group_radius_deg = group_radius / 3600.0
            if find == "all":
                group_ra_col = group_ra.unsqueeze(1)
                group_dec_col = group_dec.unsqueeze(1)
                radius_col = group_radius_deg.unsqueeze(1)
                for right_ids in candidate_right_idx.split(right_chunk_size):
                    right_chunk_len = right_ids.shape[0]
                    sep = great_circle_distance(
                        group_ra_col,
                        group_dec_col,
                        right_ra[right_ids].unsqueeze(0),
                        right_dec[right_ids].unsqueeze(0),
                        degrees=True,
                    )
                    stats.pairs_evaluated += count * right_chunk_len
                    left_rel, right_rel = torch.nonzero(
                        sep <= radius_col, as_tuple=True
                    )
                    if left_rel.numel() == 0:
                        continue
                    left_hits.append(group_abs[left_rel])
                    right_hits.append(right_ids[right_rel])
                    sep_hits.append(sep[left_rel, right_rel] * 3600.0)
                continue

            best_sep = torch.full(
                (count,),
                torch.inf,
                dtype=work_dtype,
                device=device,
            )
            best_idx = torch.full(
                (count,),
                -1,
                dtype=torch.int64,
                device=device,
            )
            group_ra_col = group_ra.unsqueeze(1)
            group_dec_col = group_dec.unsqueeze(1)

            for right_ids in candidate_right_idx.split(right_chunk_size):
                right_chunk_len = right_ids.shape[0]
                sep = great_circle_distance(
                    group_ra_col,
                    group_dec_col,
                    right_ra[right_ids].unsqueeze(0),
                    right_dec[right_ids].unsqueeze(0),
                    degrees=True,
                )
                stats.pairs_evaluated += count * right_chunk_len
                cand_sep, cand_idx = sep.min(dim=1)
                improved = cand_sep < best_sep
                if bool(improved.any()):
                    best_sep = torch.where(improved, cand_sep, best_sep)
                    best_idx = torch.where(improved, right_ids[cand_idx], best_idx)

            mask = best_sep <= group_radius_deg
            if bool(mask.any()):
                left_hits.append(group_abs[mask])
                right_hits.append(best_idx[mask])
                sep_hits.append(best_sep[mask] * 3600.0)

    result = _finalize_crossmatch(
        left_hits,
        right_hits,
        sep_hits,
        device=device,
        stats=stats,
        sort_within_left=find == "all",
    )
    if return_stats:
        return result, stats
    return result


def crossmatch_sky(
    left_ra_deg: Tensor,
    left_dec_deg: Tensor,
    right_ra_deg: Tensor,
    right_dec_deg: Tensor,
    *,
    radius_arcsec: float | Tensor,
    left_chunk_size: int = 65536,
    right_chunk_size: int = 65536,
    find: str = "best",
    right_index: HealpixIndex | None = None,
    device: torch.device | str | None = None,
) -> CrossmatchResult:
    """Chunked angular crossmatch on one in-memory tile.

    ``find="best"`` returns the nearest right-catalog row per left row.
    ``find="all"`` returns every pair within the radius, ordered by left
    index, separation, then right index.

    ``radius_arcsec`` may be a positive scalar or a per-left-row tensor.
    HEALPix covering uses the per-group maximum; the exact GC test uses the
    per-row radius. Pass ``right_index`` to reuse a nested ``HealpixIndex``.
    ``device`` moves inputs onto that device when given.
    """
    raw_left_ra = torch.as_tensor(left_ra_deg).reshape(-1)
    work_device = (
        torch.device(device) if device is not None else raw_left_ra.device
    )
    work_dtype = _work_dtype(work_device)
    left_ra = raw_left_ra.to(device=work_device, dtype=work_dtype)
    left_dec = torch.as_tensor(
        left_dec_deg, dtype=work_dtype, device=work_device
    ).reshape(-1)
    right_ra = torch.as_tensor(
        right_ra_deg, dtype=work_dtype, device=work_device
    ).reshape(-1)
    right_dec = torch.as_tensor(
        right_dec_deg, dtype=work_dtype, device=work_device
    ).reshape(-1)
    radius_left, _ = _normalize_radius_arcsec(
        radius_arcsec, int(left_ra.numel()), work_device
    )
    if left_chunk_size <= 0 or right_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if find not in ("best", "all"):
        raise ValueError("find must be 'best' or 'all'")
    if left_ra.shape != left_dec.shape:
        raise ValueError("left_ra_deg and left_dec_deg must have the same shape")
    if right_ra.shape != right_dec.shape:
        raise ValueError("right_ra_deg and right_dec_deg must have the same shape")
    if right_ra.numel() == 0 or left_ra.numel() == 0:
        return _empty_crossmatch(work_device)

    return cast(
        CrossmatchResult,
        _crossmatch_sky_coarse_indexed(
            left_ra,
            left_dec,
            right_ra,
            right_dec,
            radius_arcsec=radius_left,
            left_chunk_size=left_chunk_size,
            right_chunk_size=right_chunk_size,
            find=find,
            right_index=right_index,
        ),
    )


def crossmatch_sky_streamed(
    left_batches: Iterable[tuple[Tensor, Tensor]],
    right_ra_deg: Tensor,
    right_dec_deg: Tensor,
    *,
    radius_arcsec: float,
    right_chunk_size: int = 65536,
) -> CrossmatchResult:
    """Streamed crossmatch where the left catalog arrives in batches.

    The right catalog must fit in memory (required for the coarse HEALPix
    index). The left catalog is consumed incrementally from *left_batches*,
    which yields ``(ra_deg, dec_deg)`` tensor pairs.

    Parameters
    ----------
    left_batches
        Iterator yielding ``(ra_tensor, dec_tensor)`` pairs (each 1-D
        float64). Batches are concatenated in order and their indices are
        mapped back into the final ``left_index`` output.
    right_ra_deg, right_dec_deg
        Full right-catalog coordinates (1-D float64).
    radius_arcsec
        Match radius in arcseconds.
    right_chunk_size
        Right-side chunk size for the inner distance loop.
    """
    raw_right_ra = torch.as_tensor(right_ra_deg).reshape(-1)
    device = raw_right_ra.device
    work_dtype = _work_dtype(device)
    right_ra = raw_right_ra.to(dtype=work_dtype)
    right_dec = torch.as_tensor(
        right_dec_deg, dtype=work_dtype, device=device
    ).reshape(-1)

    if radius_arcsec <= 0.0:
        raise ValueError("radius_arcsec must be positive")
    if right_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")

    coarse_nside = _choose_default_coarse_nside(
        torch.empty(0, device=device),
        torch.empty(0, device=device),
        right_ra,
        right_dec,
        radius_arcsec=radius_arcsec,
    )
    coarse_index = HealpixIndex.build(
        right_ra, right_dec, nside=coarse_nside, nest=True
    )
    radius_deg = radius_arcsec / 3600.0
    pixrad = _healpix.max_pixrad(coarse_nside, degrees=False)
    match_rad = math.radians(radius_deg)
    candidate_radius_rad = match_rad + (2.0 * pixrad)
    hops = _covering_hops(candidate_radius_rad, pixrad)
    use_rings = hops <= _BATCHED_RING_HOPS_MAX

    left_hits: list[Tensor] = []
    right_hits: list[Tensor] = []
    sep_hits: list[Tensor] = []
    left_offset = 0
    candidate_cache: dict[int, Tensor] = {}

    for left_ra, left_dec in left_batches:
        left_ra = torch.as_tensor(left_ra, dtype=work_dtype, device=device).reshape(
            -1
        )
        left_dec = torch.as_tensor(
            left_dec, dtype=work_dtype, device=device
        ).reshape(-1)
        chunk_len = int(left_ra.numel())
        if chunk_len == 0:
            continue
        left_abs = torch.arange(
            left_offset, left_offset + chunk_len, dtype=torch.int64, device=device
        )
        if use_rings:
            _accumulate_ring_chunk(
                left_ra=left_ra,
                left_dec=left_dec,
                left_abs=left_abs,
                radius_deg=torch.full(
                    (chunk_len,), radius_deg, dtype=work_dtype, device=device
                ),
                right_ra=right_ra,
                right_dec=right_dec,
                coarse_index=coarse_index,
                hops=hops,
                find="best",
                left_hits=left_hits,
                right_hits=right_hits,
                sep_hits=sep_hits,
                stats=None,
            )
            left_offset += chunk_len
            continue

        left_coarse_pixel = _healpix.ang2pix(
            coarse_nside,
            left_ra,
            left_dec,
            nest=True,
            lonlat=True,
        ).to(torch.int64)

        left_order = torch.argsort(left_coarse_pixel, stable=True)
        coarse_sorted = left_coarse_pixel[left_order]
        unique_pixels, counts = torch.unique_consecutive(
            coarse_sorted, return_counts=True
        )
        offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.cumsum(counts[:-1].to(torch.int64), dim=0),
            )
        )

        for coarse_pixel, offset, count in zip(
            unique_pixels.tolist(), offsets.tolist(), counts.tolist(), strict=False
        ):
            rel_sorted = left_order[offset : offset + count]
            group_ra = left_ra[rel_sorted]
            group_dec = left_dec[rel_sorted]
            group_abs = left_abs[rel_sorted]

            candidate_pixels = candidate_cache.get(coarse_pixel)
            if candidate_pixels is None:
                candidate_pixels = _candidate_coarse_pixels_for_left_pixel(
                    coarse_nside,
                    coarse_pixel,
                    radius_rad=candidate_radius_rad,
                    device=device,
                )
                candidate_cache[coarse_pixel] = candidate_pixels

            candidate_right_idx = coarse_index.lookup(candidate_pixels)
            if candidate_right_idx.numel() == 0:
                continue

            best_sep = torch.full(
                (count,), torch.inf, dtype=work_dtype, device=device
            )
            best_idx = torch.full((count,), -1, dtype=torch.int64, device=device)
            group_ra_col = group_ra.unsqueeze(1)
            group_dec_col = group_dec.unsqueeze(1)

            for right_ids in candidate_right_idx.split(right_chunk_size):
                sep = great_circle_distance(
                    group_ra_col,
                    group_dec_col,
                    right_ra[right_ids].unsqueeze(0),
                    right_dec[right_ids].unsqueeze(0),
                    degrees=True,
                )
                cand_sep, cand_idx = sep.min(dim=1)
                improved = cand_sep < best_sep
                if bool(improved.any()):
                    best_sep = torch.where(improved, cand_sep, best_sep)
                    best_idx = torch.where(improved, right_ids[cand_idx], best_idx)

            mask = best_sep <= radius_deg
            if bool(mask.any()):
                left_hits.append(group_abs[mask])
                right_hits.append(best_idx[mask])
                sep_hits.append(best_sep[mask] * 3600.0)

        left_offset += chunk_len

    return _finalize_crossmatch(
        left_hits,
        right_hits,
        sep_hits,
        device=device,
        stats=_CrossmatchStats(
            left_points=left_offset,
            right_points=int(right_ra.numel()),
        ),
    )


def _tile_catalog_to_parquet(
    source: Any,
    *,
    ra_col: str,
    dec_col: str,
    coarse_nside: int,
    path: str | Path,
    batch_size: int = 131072,
) -> tuple[dict[int, list[int]], int]:
    """Stream a catalog through coarse HEALPix tiling into a single Parquet file.

    Each per-batch pixel group is written as its own row group, so that
    ``read_row_groups(groups)`` reads exactly one pixel's data.

    Returns ``(pixel_to_rgs, n_rows)`` — mapping from pixel to list of
    row-group indices and total row count.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    mmap = None
    if isinstance(source, (str, Path)):
        mmap = pa.memory_map(str(source), "r")
        source = pa.ipc.open_file(mmap)
    if isinstance(source, pa.Table):
        reader = source.to_batches(max_chunksize=batch_size)
    elif isinstance(source, pa.ipc.RecordBatchReader):
        reader = source
    elif isinstance(source, pa.ipc.RecordBatchFileReader):
        reader = (source.get_batch(i) for i in range(source.num_record_batches))
    else:
        if mmap is not None:
            mmap.close()
        raise TypeError(f"unsupported Arrow source: {type(source).__name__}")

    pixel_to_rgs: dict[int, list[int]] = {}
    writer: pq.ParquetWriter | None = None
    rg_idx = 0
    n_rows = 0

    try:
        batches = (
            batch.slice(start, batch_size)
            for batch in reader
            for start in range(0, batch.num_rows, batch_size)
        )
        for batch in batches:
            ra_np = batch.column(ra_col).to_numpy().astype("f8", copy=False)
            dec_np = batch.column(dec_col).to_numpy().astype("f8", copy=False)
            ra_t = torch.from_numpy(ra_np.copy())
            dec_t = torch.from_numpy(dec_np.copy())
            pixels = _healpix.ang2pix(
                coarse_nside, ra_t, dec_t, nest=True, lonlat=True
            )
            pixels_np = pixels.numpy().astype(np.int64)

            order = np.argsort(pixels_np, kind="stable")
            n = len(order)
            if n == 0:
                continue

            ra_sorted = ra_np[order]
            dec_sorted = dec_np[order]
            pix_sorted = pixels_np[order]
            row_sorted = order.astype(np.int64) + n_rows

            pixel_start = 0
            for i in range(1, n):
                if pix_sorted[i] != pix_sorted[pixel_start]:
                    pix = int(pix_sorted[pixel_start])
                    pixel_batch = pa.RecordBatch.from_arrays(
                        [
                            pa.array(ra_sorted[pixel_start:i]),
                            pa.array(dec_sorted[pixel_start:i]),
                            pa.array(pix_sorted[pixel_start:i]),
                            pa.array(row_sorted[pixel_start:i]),
                        ],
                        names=["lon", "lat", "coarse_pixel", "source_row"],
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(str(path), pixel_batch.schema)
                    assert writer is not None
                    writer.write_batch(pixel_batch)
                    n_rows += len(pixel_batch)
                    pixel_to_rgs.setdefault(pix, []).append(rg_idx)
                    rg_idx += 1
                    pixel_start = i

            # last pixel group in this batch
            pix = int(pix_sorted[pixel_start])
            pixel_batch = pa.RecordBatch.from_arrays(
                [
                    pa.array(ra_sorted[pixel_start:n]),
                    pa.array(dec_sorted[pixel_start:n]),
                    pa.array(pix_sorted[pixel_start:n]),
                    pa.array(row_sorted[pixel_start:n]),
                ],
                names=["lon", "lat", "coarse_pixel", "source_row"],
            )
            if writer is None:
                writer = pq.ParquetWriter(str(path), pixel_batch.schema)
            assert writer is not None
            writer.write_batch(pixel_batch)
            n_rows += len(pixel_batch)
            pixel_to_rgs.setdefault(pix, []).append(rg_idx)
            rg_idx += 1
    finally:
        if writer is not None:
            writer.close()
        if mmap is not None:
            mmap.close()

    return pixel_to_rgs, n_rows


def crossmatch_sky_from_arrow(
    left_source: Any,
    right_source: Any,
    *,
    radius_arcsec: float,
    left_columns: tuple[str, str] = ("ra", "dec"),
    right_columns: tuple[str, str] = ("ra", "dec"),
    batch_size: int = 131072,
    find: str = "best",
) -> CrossmatchResult:
    """Crossmatch Arrow inputs through temporary HEALPix tiles.

    Both catalogs are streamed once (partitioned by coarse HEALPix pixel
    into temporary Parquet files with a ``coarse_pixel`` column).  Per-pixel
    tiles are then loaded one at a time for fine-grained distance
    computation. Coordinate memory scales with the largest tile and its
    neighboring right tiles. Pixel metadata and returned pairs remain in RAM.
    Indices always refer to the original input row order, including across
    batches. Use them to select native ID columns without converting IDs to float.

    For uniformly distributed sky positions and a well-chosen coarse nside,
    per-pixel tile size is typically a few thousand rows regardless of
    whether the total catalog has 10^6 or 10^9 rows.

    Parameters
    ----------
    left_source, right_source
        Each can be a ``pyarrow.Table``, ``pyarrow.ipc.RecordBatchReader``,
        or a path string / ``pathlib.Path`` to an Arrow IPC file (Feather
        format v2).  File sources use memory-mapped I/O.
    radius_arcsec
        Match radius in arcseconds.
    left_columns, right_columns
        ``(ra_column_name, dec_column_name)`` for each catalog.
    batch_size
        Row count per streaming batch during the tiling phase.
    find
        ``"best"`` selects one nearest right row per left row, breaking ties
        by original right row index. ``"all"`` returns every pair in radius.
    """
    import pyarrow.parquet as pq

    if radius_arcsec <= 0.0:
        raise ValueError("radius_arcsec must be positive")
    if find not in ("best", "all"):
        raise ValueError("find must be 'best' or 'all'")
    tmpdir = tempfile.mkdtemp(prefix="torchsky_xmatch_")
    try:
        left_path = Path(tmpdir) / "left.parquet"
        right_path = Path(tmpdir) / "right.parquet"
        coarse_nside = _choose_default_coarse_nside(
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            radius_arcsec=radius_arcsec,
        )

        left_pixel_map, n_left = _tile_catalog_to_parquet(
            left_source,
            ra_col=left_columns[0],
            dec_col=left_columns[1],
            coarse_nside=coarse_nside,
            path=left_path,
            batch_size=batch_size,
        )
        right_pixel_map, n_right = _tile_catalog_to_parquet(
            right_source,
            ra_col=right_columns[0],
            dec_col=right_columns[1],
            coarse_nside=coarse_nside,
            path=right_path,
            batch_size=batch_size,
        )
        if n_left == 0 or n_right == 0:
            return _empty_crossmatch(torch.device("cpu"))

        left_pixels = set(left_pixel_map)
        right_pixels = set(right_pixel_map)

        candidate_radius_rad = math.radians(radius_arcsec / 3600.0) + (
            2.0 * _healpix.max_pixrad(coarse_nside, degrees=False)
        )

        left_hits: list[Tensor] = []
        right_hits: list[Tensor] = []
        sep_hits: list[Tensor] = []
        device = torch.device("cpu")

        left_pf = pq.ParquetFile(str(left_path))
        right_pf = pq.ParquetFile(str(right_path))

        for left_pixel in sorted(left_pixels):
            left_rgs = left_pixel_map[left_pixel]
            left_tables = left_pf.read_row_groups(
                left_rgs,
                columns=["lon", "lat", "source_row"],
            )
            left_ra = torch.from_numpy(
                left_tables.column(0).to_numpy().copy().astype("f8", copy=False)
            )
            left_dec = torch.from_numpy(
                left_tables.column(1).to_numpy().copy().astype("f8", copy=False)
            )
            left_rows = torch.from_numpy(left_tables.column(2).to_numpy().copy())
            count = int(left_ra.numel())
            if count == 0:
                continue

            cand_pixels = _candidate_coarse_pixels_for_left_pixel(
                coarse_nside, left_pixel, radius_rad=candidate_radius_rad, device=device
            )
            relevant_right = set(cand_pixels.tolist()) & right_pixels
            if not relevant_right:
                continue

            right_rg_indices = [
                rg for rp in relevant_right for rg in right_pixel_map[rp]
            ]
            right_rg_indices.sort()
            right_tables = right_pf.read_row_groups(
                right_rg_indices,
                columns=["lon", "lat", "source_row"],
            )
            right_ra = torch.from_numpy(
                right_tables.column(0).to_numpy().copy().astype("f8", copy=False)
            )
            right_dec = torch.from_numpy(
                right_tables.column(1).to_numpy().copy().astype("f8", copy=False)
            )
            right_rows = torch.from_numpy(right_tables.column(2).to_numpy().copy())
            # Input ordering makes tensor matcher's tie break the global row index.
            order = torch.argsort(right_rows, stable=True)
            right_rows = right_rows[order]
            matched = crossmatch_sky(
                left_ra,
                left_dec,
                right_ra[order],
                right_dec[order],
                radius_arcsec=radius_arcsec,
                find=find,
            )
            if matched.left_index.numel():
                left_hits.append(left_rows[matched.left_index])
                right_hits.append(right_rows[matched.right_index])
                sep_hits.append(matched.separation_arcsec)

        return _finalize_crossmatch(
            left_hits,
            right_hits,
            sep_hits,
            device=device,
            stats=_CrossmatchStats(
                left_points=n_left,
                right_points=n_right,
            ),
            sort_within_left=find == "all",
        )
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
