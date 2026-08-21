"""Tensor-native catalog helpers."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from ..sphere import great_circle_distance
from ..sphere import healpix as _healpix
from .index import HealpixIndex
from .index import candidate_coarse_pixels_for_left_pixel as _candidate_coarse_pixels_for_left_pixel
from .index import default_nside as _default_healpix_index_nside


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
    empty_f64 = torch.empty(0, dtype=torch.float64, device=device)
    return CrossmatchResult(empty_i64, empty_i64, empty_f64)


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
    stats.matched_rows = int(left_index.numel())
    return CrossmatchResult(left_index, right_index, separation_arcsec)


def _normalize_radius_arcsec(
    radius_arcsec: float | Tensor,
    n_left: int,
    device: torch.device,
) -> tuple[Tensor, float]:
    """Return per-left radii and the prune radius (max)."""
    radius = torch.as_tensor(radius_arcsec, dtype=torch.float64, device=device)
    if radius.ndim == 0:
        value = float(radius.item())
        if value <= 0.0:
            raise ValueError("radius_arcsec must be positive")
        return torch.full((n_left,), value, dtype=torch.float64, device=device), value
    flat = radius.reshape(-1)
    if int(flat.numel()) != n_left:
        raise ValueError("radius_arcsec must be a scalar or match the left catalog")
    if bool((flat <= 0.0).any()):
        raise ValueError("radius_arcsec must be positive")
    return flat, float(flat.max().item())


def _choose_default_coarse_nside(
    left_ra: Tensor,
    left_dec: Tensor,
    right_ra: Tensor,
    right_dec: Tensor,
    *,
    radius_arcsec: float,
) -> int:
    del left_ra, left_dec
    if radius_arcsec <= 0:
        return 2048
    if radius_arcsec >= 180.0 * 3600.0:
        return 1
    # Benchmark-backed selector tuned for the current CPU nested-HEALPix
    # coarse-index path. Smaller catalogs prefer lower group overhead,
    # while larger right catalogs benefit from finer coarse partitioning.
    return _default_healpix_index_nside(int(right_ra.numel()))


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
    if left_index.numel() == 0:
        return left_index, right_index, separation_arcsec
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
    keep = torch.ones_like(left_index, dtype=torch.bool)
    keep[1:] = left_index[1:] != left_index[:-1]
    return left_index[keep], right_index[keep], separation_arcsec[keep]


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
    radius_arcsec: float,
    left_chunk_size: int,
    right_chunk_size: int,
    find: str = "best",
    return_stats: bool = False,
    right_index: HealpixIndex | None = None,
) -> CrossmatchResult | tuple[CrossmatchResult, _CrossmatchStats]:
    device = left_ra.device
    stats = _CrossmatchStats(
        left_points=int(left_ra.numel()),
        right_points=int(right_ra.numel()),
    )
    if right_index is None:
        coarse_nside = _choose_default_coarse_nside(
            left_ra,
            left_dec,
            right_ra,
            right_dec,
            radius_arcsec=radius_arcsec,
        )
        coarse_index = _build_coarse_healpix_index(
            right_ra, right_dec, coarse_nside=coarse_nside
        )
    else:
        if int(right_index.lon.numel()) != int(right_ra.numel()):
            raise ValueError("right_index does not match the right catalog")
        if right_index.lon.device != device:
            right_index = right_index.to(device)
        coarse_index = right_index
        coarse_nside = coarse_index.nside
    radius_deg = radius_arcsec / 3600.0
    candidate_radius_rad = math.radians(radius_deg) + (
        2.0 * _healpix.max_pixrad(coarse_nside, degrees=False)
    )
    candidate_cache: dict[int, Tensor] = {}

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
            stats.candidate_groups += 1
            stats.candidate_rows_considered += int(candidate_right_idx.numel())
            if candidate_right_idx.numel() == 0:
                continue

            if find == "all":
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
                    left_rel, right_rel = torch.nonzero(
                        sep <= radius_deg, as_tuple=True
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
                dtype=torch.float64,
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

            mask = best_sep <= radius_deg
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
    left_chunk_size: int = 4096,
    right_chunk_size: int = 65536,
    find: str = "best",
    right_index: HealpixIndex | None = None,
) -> CrossmatchResult:
    """Chunked angular crossmatch suitable for catalog refinement stages.

    ``find="best"`` returns the nearest right-catalog row per left row.
    ``find="all"`` returns every pair within the radius, ordered by left
    index, separation, then right index.

    ``radius_arcsec`` may be a positive scalar or a per-left-row tensor.
    Coarse declination windows use the per-row radius; a global prune uses
    the maximum. Pass ``right_index`` to reuse a nested ``HealpixIndex`` on
    the right catalog (uniform radius only).
    """
    left_ra = torch.as_tensor(left_ra_deg, dtype=torch.float64).reshape(-1)
    device = left_ra.device
    left_dec = torch.as_tensor(
        left_dec_deg, dtype=torch.float64, device=device
    ).reshape(-1)
    right_ra = torch.as_tensor(
        right_ra_deg, dtype=torch.float64, device=device
    ).reshape(-1)
    right_dec = torch.as_tensor(
        right_dec_deg, dtype=torch.float64, device=device
    ).reshape(-1)
    radius_left, _ = _normalize_radius_arcsec(
        radius_arcsec, int(left_ra.numel()), device
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
        return _empty_crossmatch(device)

    # Scalar / uniform radius → HEALPix coarse index; varying per-row → dec windows.
    if bool(torch.all(radius_left == radius_left[0])):
        return _crossmatch_sky_coarse_indexed(
            left_ra,
            left_dec,
            right_ra,
            right_dec,
            radius_arcsec=float(radius_left[0].item()),
            left_chunk_size=left_chunk_size,
            right_chunk_size=right_chunk_size,
            find=find,
            right_index=right_index,
        )
    return _crossmatch_sky_declination_indexed(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=radius_left,
        left_chunk_size=left_chunk_size,
        right_chunk_size=right_chunk_size,
        find=find,
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
    right_ra = torch.as_tensor(right_ra_deg, dtype=torch.float64).reshape(-1)
    right_dec = torch.as_tensor(right_dec_deg, dtype=torch.float64).reshape(-1)
    device = right_ra.device

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
    candidate_radius_rad = math.radians(radius_deg) + (
        2.0 * _healpix.max_pixrad(coarse_nside, degrees=False)
    )

    left_hits: list[Tensor] = []
    right_hits: list[Tensor] = []
    sep_hits: list[Tensor] = []
    left_offset = 0
    candidate_cache: dict[int, Tensor] = {}

    for left_ra, left_dec in left_batches:
        left_ra = torch.as_tensor(left_ra, dtype=torch.float64, device=device).reshape(
            -1
        )
        left_dec = torch.as_tensor(
            left_dec, dtype=torch.float64, device=device
        ).reshape(-1)
        chunk_len = int(left_ra.numel())
        if chunk_len == 0:
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
        left_abs = torch.arange(
            left_offset, left_offset + chunk_len, dtype=torch.int64, device=device
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
                (count,), torch.inf, dtype=torch.float64, device=device
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
    from ..sphere import healpix as _healpix

    if isinstance(source, (str, Path)):
        source = pa.ipc.open_file(pa.memory_map(str(source), "r")).read_all()
    if isinstance(source, pa.Table):
        reader = source.to_batches(max_chunksize=batch_size)
    elif isinstance(source, pa.ipc.RecordBatchReader):
        reader = source
    else:
        raise TypeError(f"unsupported Arrow source: {type(source).__name__}")

    pixel_to_rgs: dict[int, list[int]] = {}
    writer: pq.ParquetWriter | None = None
    rg_idx = 0
    n_rows = 0

    for batch in reader:
        ra_np = batch.column(ra_col).to_numpy().astype("f8", copy=False)
        dec_np = batch.column(dec_col).to_numpy().astype("f8", copy=False)
        ra_t = torch.from_numpy(ra_np.copy())
        dec_t = torch.from_numpy(dec_np.copy())
        pixels = _healpix.ang2pix(coarse_nside, ra_t, dec_t, nest=True, lonlat=True)
        pixels_np = pixels.numpy().astype(np.int64)

        order = np.argsort(pixels_np, kind="stable")
        n = len(order)
        if n == 0:
            continue

        ra_sorted = ra_np[order]
        dec_sorted = dec_np[order]
        pix_sorted = pixels_np[order]

        pixel_start = 0
        for i in range(1, n):
            if pix_sorted[i] != pix_sorted[pixel_start]:
                pix = int(pix_sorted[pixel_start])
                pixel_batch = pa.RecordBatch.from_arrays(
                    [
                        pa.array(ra_sorted[pixel_start:i]),
                        pa.array(dec_sorted[pixel_start:i]),
                        pa.array(pix_sorted[pixel_start:i]),
                    ],
                    names=[ra_col, dec_col, "coarse_pixel"],
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
            ],
            names=[ra_col, dec_col, "coarse_pixel"],
        )
        if writer is None:
            writer = pq.ParquetWriter(str(path), pixel_batch.schema)
        assert writer is not None
        writer.write_batch(pixel_batch)
        n_rows += len(pixel_batch)
        pixel_to_rgs.setdefault(pix, []).append(rg_idx)
        rg_idx += 1

    if writer is not None:
        writer.close()
    else:
        raise ValueError("catalog is empty")

    return pixel_to_rgs, n_rows


def crossmatch_sky_from_arrow(
    left_source: Any,
    right_source: Any,
    *,
    radius_arcsec: float,
    left_columns: tuple[str, str] = ("ra", "dec"),
    right_columns: tuple[str, str] = ("ra", "dec"),
    batch_size: int = 131072,
) -> CrossmatchResult:
    """Bounded-memory crossmatch via HEALPix tiling — no catalog is loaded
    into memory at once.

    Both catalogs are streamed once (partitioned by coarse HEALPix pixel
    into temporary Parquet files with a ``coarse_pixel`` column).  Per-pixel
    tiles are then loaded one at a time for fine-grained distance
    computation.  Memory scales with the largest per-pixel tile, not the
    total catalog size.

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
    """
    import pyarrow.parquet as pq

    tmpdir = tempfile.mkdtemp(prefix="torchsky_xmatch_")
    try:
        left_path = Path(tmpdir) / "left.parquet"
        right_path = Path(tmpdir) / "right.parquet"

        left_pixel_map, _ = _tile_catalog_to_parquet(
            left_source,
            ra_col=left_columns[0],
            dec_col=left_columns[1],
            coarse_nside=64,
            path=left_path,
            batch_size=batch_size,
        )
        right_pixel_map, n_right = _tile_catalog_to_parquet(
            right_source,
            ra_col=right_columns[0],
            dec_col=right_columns[1],
            coarse_nside=64,
            path=right_path,
            batch_size=batch_size,
        )

        left_pixels = set(left_pixel_map)
        right_pixels = set(right_pixel_map)

        radius_deg = radius_arcsec / 3600.0
        candidate_radius_rad = math.radians(radius_deg) + (
            2.0 * _healpix.max_pixrad(64, degrees=False)
        )

        left_hits: list[Tensor] = []
        right_hits: list[Tensor] = []
        sep_hits: list[Tensor] = []
        left_offset = 0
        device = torch.device("cpu")

        left_pf = pq.ParquetFile(str(left_path))
        right_pf = pq.ParquetFile(str(right_path))

        for left_pixel in sorted(left_pixels):
            left_rgs = left_pixel_map[left_pixel]
            left_tables = left_pf.read_row_groups(
                left_rgs,
                columns=[left_columns[0], left_columns[1]],
            )
            left_ra = torch.from_numpy(
                left_tables.column(0).to_numpy().copy().astype("f8", copy=False)
            )
            left_dec = torch.from_numpy(
                left_tables.column(1).to_numpy().copy().astype("f8", copy=False)
            )
            count = int(left_ra.numel())
            if count == 0:
                continue

            cand_pixels = _candidate_coarse_pixels_for_left_pixel(
                64, left_pixel, radius_rad=candidate_radius_rad, device=device
            )
            relevant_right = set(cand_pixels.tolist()) & right_pixels
            if not relevant_right:
                left_offset += count
                continue

            right_rg_indices = [
                rg for rp in relevant_right for rg in right_pixel_map[rp]
            ]
            right_rg_indices.sort()
            right_tables = right_pf.read_row_groups(
                right_rg_indices,
                columns=[right_columns[0], right_columns[1]],
            )
            right_ra = torch.from_numpy(
                right_tables.column(0).to_numpy().copy().astype("f8", copy=False)
            )
            right_dec = torch.from_numpy(
                right_tables.column(1).to_numpy().copy().astype("f8", copy=False)
            )
            n_right = int(right_ra.numel())
            if n_right == 0:
                left_offset += count
                continue

            left_abs = torch.arange(
                left_offset, left_offset + count, dtype=torch.int64, device=device
            )
            left_ra_2d = left_ra.unsqueeze(1)
            left_dec_2d = left_dec.unsqueeze(1)

            best_sep = torch.full((count,), torch.inf, dtype=torch.float64)
            best_idx = torch.full((count,), -1, dtype=torch.int64)
            for right_ids in range(0, n_right, 65536):
                chunk = slice(right_ids, min(right_ids + 65536, n_right))
                sep = great_circle_distance(
                    left_ra_2d,
                    left_dec_2d,
                    right_ra[chunk].unsqueeze(0),
                    right_dec[chunk].unsqueeze(0),
                    degrees=True,
                )
                cand_sep, cand_idx = sep.min(dim=1)
                improved = cand_sep < best_sep
                if bool(improved.any()):
                    best_sep = torch.where(improved, cand_sep, best_sep)
                    best_idx = torch.where(improved, right_ids + cand_idx, best_idx)

            mask = best_sep <= radius_deg
            if bool(mask.any()):
                left_hits.append(left_abs[mask])
                right_hits.append(best_idx[mask])
                sep_hits.append(best_sep[mask] * 3600.0)

            left_offset += count

        return _finalize_crossmatch(
            left_hits,
            right_hits,
            sep_hits,
            device=device,
            stats=_CrossmatchStats(
                left_points=left_offset,
                right_points=n_right,
            ),
        )
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
