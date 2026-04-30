"""Canonical torchsky I/O surface."""

from __future__ import annotations

from typing import Any

import torchsky.cpp as cpp

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


def read_fast(*args: Any, **kwargs: Any):
    from .._fastio import read as _read_fast

    return _read_fast(*args, **kwargs)


def read(*args: Any, **kwargs: Any):
    from .. import read as _read

    return _read(*args, **kwargs)


def read_image(*args: Any, **kwargs: Any):
    from .. import read_image as _read_image

    return _read_image(*args, **kwargs)


def read_table(*args: Any, **kwargs: Any):
    from .. import read_table as _read_table

    return _read_table(*args, **kwargs)


def read_hdus(*args: Any, **kwargs: Any):
    from .. import read_hdus as _read_hdus

    return _read_hdus(*args, **kwargs)


def read_subset(*args: Any, **kwargs: Any):
    from .. import read_subset as _read_subset

    return _read_subset(*args, **kwargs)


def open(*args: Any, **kwargs: Any):
    from .. import open as _open

    return _open(*args, **kwargs)


def write(*args: Any, **kwargs: Any):
    from .. import write as _write

    return _write(*args, **kwargs)


def get_header(*args: Any, **kwargs: Any):
    from .. import get_header as _get_header

    return _get_header(*args, **kwargs)


def get_wcs(*args: Any, **kwargs: Any):
    from .. import get_wcs as _get_wcs

    return _get_wcs(*args, **kwargs)


def stream_table(*args: Any, **kwargs: Any):
    from .. import stream_table as _stream_table

    return _stream_table(*args, **kwargs)


def read_large_table(*args: Any, **kwargs: Any):
    from .. import read_large_table as _read_large_table

    return _read_large_table(*args, **kwargs)


__all__ = [
    "FITSDataset",
    "HDUList",
    "Header",
    "Card",
    "IterableFITSDataset",
    "TableChunkDataset",
    "TableHDU",
    "TableHDURef",
    "TensorHDU",
    "create_dataloader",
    "create_fits_dataloader",
    "create_streaming_dataloader",
    "create_table_dataloader",
    "get_header",
    "get_wcs",
    "open",
    "read",
    "read_fast",
    "read_hdus",
    "read_image",
    "read_large_table",
    "read_subset",
    "read_table",
    "read_tensor_frame",
    "stream_table",
    "table",
    "to_arrow",
    "to_pandas",
    "to_polars",
    "to_tensor_frame",
    "write",
    "write_tensor_frame",
]

# Internal capability flags for tests
_HAS_READ_HDUS_BATCH = hasattr(cpp, "read_hdus_batch")
_HAS_READ_FULL_RAW_WITH_SCALE = hasattr(cpp, "read_full_raw_with_scale")
_HAS_READ_FULL_RAW = hasattr(cpp, "read_full_raw")
_HAS_READ_FULL_UNMAPPED_RAW = hasattr(cpp, "read_full_unmapped_raw")
_HAS_READ_FULL_UNMAPPED = hasattr(cpp, "read_full_unmapped")
_HAS_READ_FULL_NOCACHE = hasattr(cpp, "read_full_nocache")
