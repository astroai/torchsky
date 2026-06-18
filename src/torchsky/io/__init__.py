"""Canonical torchsky I/O convenience surface.

FITS file operations are owned by :mod:`torchsky.io.fits`. This namespace keeps
the established convenience imports for datasets, table helpers, interop, and
root-compatible I/O wrappers.
"""

from __future__ import annotations

from .. import table
from ..dataloader import (
    create_dataloader,
    create_fits_dataloader,
    create_streaming_dataloader,
    create_table_dataloader,
)
from ..datasets import FITSDataset, IterableFITSDataset, TableChunkDataset
from ..frame import read_tensor_frame, to_tensor_frame, write_tensor_frame
from ..hdu import Card, HDUList, Header, TableHDU, TableHDURef, TensorHDU
from ..interop import to_arrow, to_pandas, to_polars
from . import fits as fits
from .fits import (
    IO_CACHE_SUBSYSTEMS,
    _HAS_READ_FULL_NOCACHE,
    _HAS_READ_FULL_RAW,
    _HAS_READ_FULL_RAW_WITH_SCALE,
    _HAS_READ_FULL_UNMAPPED,
    _HAS_READ_FULL_UNMAPPED_RAW,
    _HAS_READ_HDUS_BATCH,
    _get_image_meta,
    _invalidate_path_caches,
    _resolve_image_mmap,
    _should_use_cold_nommap,
    cache_subsystem_policy,
    clear_cache_subsystem,
    clear_file_cache,
    delete_hdu,
    get_batch_info,
    get_cache_performance,
    get_header,
    get_wcs,
    insert_hdu,
    open,
    open_subset_reader,
    read_batch,
    read_fast,
    read_hdus,
    read_image,
    read_large_table,
    read_subset,
    read_table,
    replace_hdu,
    stream_table,
    verify_checksums,
    write,
    write_checksums,
)


def _sync_fits_compat_overrides() -> None:
    """Forward legacy `torchsky.io` monkeypatch hooks to the owning FITS module."""
    fits._get_image_meta = _get_image_meta
    fits._should_use_cold_nommap = _should_use_cold_nommap
    fits._resolve_image_mmap = _resolve_image_mmap


def read(*args, **kwargs):
    _sync_fits_compat_overrides()
    return fits.read(*args, **kwargs)

__all__ = [
    "FITSDataset",
    "HDUList",
    "IO_CACHE_SUBSYSTEMS",
    "Header",
    "Card",
    "IterableFITSDataset",
    "TableChunkDataset",
    "TableHDU",
    "TableHDURef",
    "TensorHDU",
    "_HAS_READ_FULL_NOCACHE",
    "_HAS_READ_FULL_RAW",
    "_HAS_READ_FULL_RAW_WITH_SCALE",
    "_HAS_READ_FULL_UNMAPPED",
    "_HAS_READ_FULL_UNMAPPED_RAW",
    "_HAS_READ_HDUS_BATCH",
    "_get_image_meta",
    "_invalidate_path_caches",
    "_resolve_image_mmap",
    "_should_use_cold_nommap",
    "clear_file_cache",
    "cache_subsystem_policy",
    "clear_cache_subsystem",
    "create_dataloader",
    "create_fits_dataloader",
    "create_streaming_dataloader",
    "create_table_dataloader",
    "delete_hdu",
    "fits",
    "get_batch_info",
    "get_cache_performance",
    "get_header",
    "get_wcs",
    "insert_hdu",
    "open",
    "open_subset_reader",
    "read",
    "read_batch",
    "read_fast",
    "read_hdus",
    "read_image",
    "read_large_table",
    "read_subset",
    "read_table",
    "read_tensor_frame",
    "replace_hdu",
    "stream_table",
    "table",
    "to_arrow",
    "to_pandas",
    "to_polars",
    "to_tensor_frame",
    "verify_checksums",
    "write",
    "write_checksums",
    "write_tensor_frame",
]
