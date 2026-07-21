"""FITS datasets — torchfits-backed with a TableChunkDataset for chunked table access."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import IterableDataset

from torchfits.data import (
    FitsImageDataset as FitsImageDataset,
    FitsImageIterableDataset as FitsImageIterableDataset,
)
from torchfits import read_header as _tf_get_header
from torchfits.table import scan_torch as _tf_scan_torch

# Compatibility names belong to torchsky, not torchfits.  New code should use
# the explicit image-dataset names above.
FITSDataset = FitsImageDataset
IterableFITSDataset = FitsImageIterableDataset


class TableChunkDataset(IterableDataset):
    """Iterable dataset that yields table chunks (dict[str, Tensor]) from FITS tables.

    Parameters
    ----------
    file_paths:
        One or more FITS file paths.
    hdu:
        HDU index or name.
    columns:
        Column names to select (``None`` → all).
    chunk_rows:
        Number of rows per chunk.
    max_chunks:
        Cap on total chunks per file (``None`` → unlimited).
    mmap:
        Memory-map table data when ``True``.
    device:
        Target device for tensors.
    non_blocking_transfer:
        Use non-blocking ``.to(device)`` when ``True``.
    pin_memory_transfer:
        Pin host memory before transfer when ``True``.
    transform:
        Optional callable applied to each chunk dict.
    include_header:
        Attach the table header as ``chunk["_header"]`` when ``True``.
    """

    def __init__(
        self,
        file_paths: list[str],
        hdu: int = 1,
        columns: list[str] | None = None,
        chunk_rows: int = 1000,
        max_chunks: int | None = None,
        mmap: bool = False,
        device: str = "cpu",
        non_blocking_transfer: bool = True,
        pin_memory_transfer: bool = False,
        transform: Any = None,
        include_header: bool = False,
    ) -> None:
        super().__init__()
        self.file_paths = list(file_paths)
        self.hdu = hdu
        self.columns = columns
        self.chunk_rows = chunk_rows
        self.max_chunks = max_chunks
        self.mmap = mmap
        self.device = device
        self.non_blocking_transfer = non_blocking_transfer
        self.pin_memory_transfer = pin_memory_transfer
        self.transform = transform
        self.include_header = include_header

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Each worker handles its shard of files.
            paths = [
                self.file_paths[i]
                for i in range(len(self.file_paths))
                if i % worker_info.num_workers == worker_info.id
            ]
        else:
            paths = self.file_paths

        for file_path in paths:
            header = (
                _tf_get_header(file_path, self.hdu) if self.include_header else None
            )
            for chunk_idx, batch in enumerate(
                _tf_scan_torch(
                    file_path,
                    hdu=self.hdu,
                    columns=self.columns,
                    batch_size=self.chunk_rows,
                    mmap=self.mmap,
                    device=self.device,
                    non_blocking=self.non_blocking_transfer,
                    pin_memory=self.pin_memory_transfer,
                )
            ):
                if self.max_chunks is not None and chunk_idx >= self.max_chunks:
                    break
                chunk = dict(batch)
                if self.include_header and header is not None:
                    chunk["_header"] = header
                if self.transform is not None:
                    chunk = self.transform(chunk)
                yield chunk
