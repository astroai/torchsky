"""Canonical torchsky I/O convenience surface.

FITS operations are delegated to torchfits. This namespace keeps
convenience imports for datasets, table helpers, interop, and
root-compatible I/O wrappers.
"""

from __future__ import annotations

import torchfits

from .. import table
from ..dataloader import (
    create_dataloader,
    create_fits_dataloader,
    create_streaming_dataloader,
    create_table_dataloader,
)
from ..datasets import (
    FITSDataset,
    FitsImageDataset,
    FitsImageIterableDataset,
    IterableFITSDataset,
    TableChunkDataset,
)
from ..frame import read_tensor_frame, to_tensor_frame, write_tensor_frame
from torchfits.interop import to_arrow, to_pandas
from torchfits.io import (
    _get_image_meta as _get_image_meta,
    _resolve_image_mmap as _resolve_image_mmap,
    _should_use_cold_nommap as _should_use_cold_nommap,
)
from ..hdu import Card, HDUList, Header, TableHDU, TableHDURef, TensorHDU
from . import fits as fits
from .fits import (
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
    read_subset,
    read_table,
    replace_hdu,
    stream_table,
    verify_checksums,
    write,
    write_checksums,
)

read = torchfits.read

__all__ = [
    "FITSDataset",
    "FitsImageDataset",
    "FitsImageIterableDataset",
    "HDUList",
    "Header",
    "Card",
    "IterableFITSDataset",
    "TableChunkDataset",
    "TableHDU",
    "TableHDURef",
    "TensorHDU",
    "clear_file_cache",
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
    "read_subset",
    "read_table",
    "read_tensor_frame",
    "replace_hdu",
    "stream_table",
    "table",
    "to_arrow",
    "to_pandas",
    "to_tensor_frame",
    "verify_checksums",
    "write",
    "write_checksums",
    "write_tensor_frame",
]
