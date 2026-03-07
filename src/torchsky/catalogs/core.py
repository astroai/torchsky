"""Tensor-native catalog helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geom import great_circle_distance


class CatalogStreamDataset(IterableDataset):
    """Wrap a partition iterator into a PyTorch IterableDataset."""

    def __init__(
        self, partitions: Iterable[object], loader: Callable[[object], object]
    ):
        self._partitions = partitions
        self._loader = loader

    def __iter__(self) -> Iterator[object]:
        for partition in self._partitions:
            yield self._loader(partition)


@dataclass
class CrossmatchResult:
    left_index: Tensor
    right_index: Tensor
    separation_arcsec: Tensor


def crossmatch_sky(
    left_ra_deg: Tensor,
    left_dec_deg: Tensor,
    right_ra_deg: Tensor,
    right_dec_deg: Tensor,
    *,
    radius_arcsec: float,
    chunk_size: int = 4096,
) -> CrossmatchResult:
    """Chunked brute-force angular crossmatch suitable for refinement stages."""
    left_ra = torch.as_tensor(left_ra_deg, dtype=torch.float64)
    left_dec = torch.as_tensor(left_dec_deg, dtype=torch.float64)
    right_ra = torch.as_tensor(right_ra_deg, dtype=torch.float64)
    right_dec = torch.as_tensor(right_dec_deg, dtype=torch.float64)
    radius_deg = radius_arcsec / 3600.0

    left_hits: list[Tensor] = []
    right_hits: list[Tensor] = []
    sep_hits: list[Tensor] = []

    for start in range(0, int(left_ra.numel()), chunk_size):
        stop = min(start + chunk_size, int(left_ra.numel()))
        ra_chunk = left_ra[start:stop].unsqueeze(1)
        dec_chunk = left_dec[start:stop].unsqueeze(1)
        sep = great_circle_distance(
            ra_chunk,
            dec_chunk,
            right_ra.unsqueeze(0),
            right_dec.unsqueeze(0),
            degrees=True,
        )
        best_sep, best_idx = sep.min(dim=1)
        mask = best_sep <= radius_deg
        if mask.any():
            left_hits.append(torch.arange(start, stop, dtype=torch.int64)[mask])
            right_hits.append(best_idx[mask].to(torch.int64))
            sep_hits.append(best_sep[mask] * 3600.0)

    if not left_hits:
        empty = torch.empty(0, dtype=torch.int64)
        return CrossmatchResult(empty, empty, empty.to(torch.float64))

    return CrossmatchResult(
        left_index=torch.cat(left_hits),
        right_index=torch.cat(right_hits),
        separation_arcsec=torch.cat(sep_hits),
    )
