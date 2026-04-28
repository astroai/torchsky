"""Tensor-native catalog helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Iterable, Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from ..geom import great_circle_distance
from ..wcs import healpix as _healpix


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


@dataclass
class _CoarseHealpixIndex:
    nside: int
    right_order: Tensor
    unique_pixels: Tensor
    bucket_starts: Tensor
    bucket_stops: Tensor

    def lookup(self, coarse_pixels: Tensor) -> Tensor:
        pixels = torch.as_tensor(
            coarse_pixels,
            dtype=torch.int64,
            device=self.unique_pixels.device,
        ).reshape(-1)
        if pixels.numel() == 0 or self.unique_pixels.numel() == 0:
            return self.right_order[:0]

        pixels = torch.unique(pixels, sorted=True)
        idx = torch.searchsorted(self.unique_pixels, pixels)
        idx_safe = idx.clamp(max=max(int(self.unique_pixels.numel()) - 1, 0))
        mask = (idx < self.unique_pixels.numel()) & (
            self.unique_pixels[idx_safe] == pixels
        )
        if not bool(mask.any()):
            return self.right_order[:0]

        starts = self.bucket_starts[idx_safe[mask]]
        stops = self.bucket_stops[idx_safe[mask]]
        lengths = (stops - starts).to(torch.int64)
        total = int(lengths.sum().item())
        if total == 0:
            return self.right_order[:0]

        out = torch.empty(total, dtype=torch.int64, device=self.right_order.device)
        offset = 0
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=False):
            span = int(stop - start)
            out[offset : offset + span] = self.right_order[start:stop]
            offset += span
        return out


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
) -> CrossmatchResult:
    if not left_hits:
        return _empty_crossmatch(device)

    left_index = torch.cat(left_hits)
    right_index = torch.cat(right_hits)
    separation_arcsec = torch.cat(sep_hits)
    order = torch.argsort(left_index, stable=True)
    left_index = left_index[order]
    right_index = right_index[order]
    separation_arcsec = separation_arcsec[order]
    stats.matched_rows = int(left_index.numel())
    return CrossmatchResult(left_index, right_index, separation_arcsec)


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
    right_points = int(right_ra.numel())
    # Benchmark-backed selector tuned for the current CPU nested-HEALPix
    # coarse-index path. Smaller catalogs prefer lower group overhead,
    # while larger right catalogs benefit from finer coarse partitioning.
    if right_points <= 25_000:
        return 64
    if right_points <= 90_000:
        return 128
    if right_points <= 180_000:
        return 512
    if right_points <= 1_000_000:
        return 1024
    return 2048


def _build_coarse_healpix_index(
    right_ra: Tensor,
    right_dec: Tensor,
    *,
    coarse_nside: int,
) -> _CoarseHealpixIndex:
    coarse_pixels = _healpix.ang2pix(
        coarse_nside,
        right_ra,
        right_dec,
        nest=True,
        lonlat=True,
    ).to(torch.int64)
    right_order = torch.argsort(coarse_pixels, stable=True)
    sorted_pixels = coarse_pixels[right_order]
    unique_pixels, counts = torch.unique_consecutive(sorted_pixels, return_counts=True)
    bucket_starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=right_ra.device),
            torch.cumsum(counts[:-1].to(torch.int64), dim=0),
        )
    )
    bucket_stops = bucket_starts + counts.to(torch.int64)
    return _CoarseHealpixIndex(
        nside=coarse_nside,
        right_order=right_order,
        unique_pixels=unique_pixels.to(torch.int64),
        bucket_starts=bucket_starts,
        bucket_stops=bucket_stops,
    )


def _candidate_coarse_pixels_for_left_pixel(
    coarse_nside: int,
    coarse_pixel: int,
    *,
    radius_rad: float,
    device: torch.device,
) -> Tensor:
    center_pix = torch.tensor([coarse_pixel], dtype=torch.int64, device=device)
    pixrad = _healpix.max_pixrad(coarse_nside, degrees=False)
    hops = max(1, int(math.ceil(radius_rad / max(pixrad, 1e-12)))) + 1

    visited = center_pix
    frontier = center_pix
    for _ in range(hops):
        neigh = _healpix.get_all_neighbours(
            coarse_nside,
            frontier,
            nest=True,
            lonlat=False,
        )
        neigh = torch.as_tensor(neigh, dtype=torch.int64, device=device).reshape(-1)
        neigh = neigh[neigh >= 0]
        if neigh.numel() == 0:
            break
        merged = torch.unique(torch.cat((visited, neigh)))
        if merged.numel() == visited.numel():
            break
        visited = merged
        frontier = neigh

    lon_c, lat_c = _healpix.pix2ang(coarse_nside, center_pix, nest=True, lonlat=True)
    cand_lon, cand_lat = _healpix.pix2ang(coarse_nside, visited, nest=True, lonlat=True)
    sep = great_circle_distance(
        lon_c.expand_as(cand_lon),
        lat_c.expand_as(cand_lat),
        cand_lon,
        cand_lat,
        degrees=False,
    )
    return visited[sep <= radius_rad]


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


def _crossmatch_sky_coarse_indexed(
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
    radius_arcsec: float,
    left_chunk_size: int = 4096,
    right_chunk_size: int = 65536,
) -> CrossmatchResult:
    """Chunked nearest-neighbour angular crossmatch suitable for refinement stages."""
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
    if radius_arcsec <= 0.0:
        raise ValueError("radius_arcsec must be positive")
    if left_chunk_size <= 0 or right_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if left_ra.shape != left_dec.shape:
        raise ValueError("left_ra_deg and left_dec_deg must have the same shape")
    if right_ra.shape != right_dec.shape:
        raise ValueError("right_ra_deg and right_dec_deg must have the same shape")
    if right_ra.numel() == 0 or left_ra.numel() == 0:
        return _empty_crossmatch(device)

    return _crossmatch_sky_coarse_indexed(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=radius_arcsec,
        left_chunk_size=left_chunk_size,
        right_chunk_size=right_chunk_size,
    )
