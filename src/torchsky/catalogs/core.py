"""Tensor-native catalog helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Iterable, Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from ..geom import great_circle_distance


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
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        empty_f64 = torch.empty(0, dtype=torch.float64, device=device)
        return CrossmatchResult(empty_i64, empty_i64, empty_f64)

    radius_deg = radius_arcsec / 3600.0
    right_order = torch.argsort(right_dec)
    right_ra_sorted = right_ra[right_order]
    right_dec_sorted = right_dec[right_order]

    left_hits: list[Tensor] = []
    right_hits: list[Tensor] = []
    sep_hits: list[Tensor] = []

    for start in range(0, int(left_ra.numel()), left_chunk_size):
        stop = min(start + left_chunk_size, int(left_ra.numel()))
        left_ra_chunk = left_ra[start:stop]
        left_dec_chunk = left_dec[start:stop]
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
        best_sep = torch.full(
            (stop - start,),
            torch.inf,
            dtype=torch.float64,
            device=left_ra.device,
        )
        best_idx = torch.full(
            (stop - start,),
            -1,
            dtype=torch.int64,
            device=left_ra.device,
        )

        for right_start in range(cand_start, cand_stop, right_chunk_size):
            right_stop = min(right_start + right_chunk_size, cand_stop)
            sep = great_circle_distance(
                ra_chunk,
                dec_chunk,
                right_ra_sorted[right_start:right_stop].unsqueeze(0),
                right_dec_sorted[right_start:right_stop].unsqueeze(0),
                degrees=True,
            )
            cand_sep, cand_idx = sep.min(dim=1)
            improved = cand_sep < best_sep
            if improved.any():
                best_sep = torch.where(improved, cand_sep, best_sep)
                best_idx = torch.where(
                    improved,
                    right_order[right_start:right_stop][cand_idx.to(torch.int64)],
                    best_idx,
                )

        mask = best_sep <= radius_deg
        if mask.any():
            left_hits.append(
                torch.arange(start, stop, dtype=torch.int64, device=left_ra.device)[
                    mask
                ]
            )
            right_hits.append(best_idx[mask])
            sep_hits.append(best_sep[mask] * 3600.0)

    if not left_hits:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        empty_f64 = torch.empty(0, dtype=torch.float64, device=device)
        return CrossmatchResult(empty_i64, empty_i64, empty_f64)

    return CrossmatchResult(
        left_index=torch.cat(left_hits),
        right_index=torch.cat(right_hits),
        separation_arcsec=torch.cat(sep_hits),
    )
