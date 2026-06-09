"""Canonical torchsky I/O surface."""

from __future__ import annotations

import logging as _stdlib_logging
import os
import sys
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
from .._io_engine.caches import (
    check_read_cache as _check_read_cache_impl,
    invalidate_path_caches as _invalidate_path_caches_impl,
)
from .._io_engine.hdu_api import autodetect_hdu as _autodetect_hdu_impl
from .._io_engine.hdu_api import open_hdulist as _open_hdulist_impl
from .._io_engine.hdu_api import read_header_fast as _read_header_fast_impl
from .._io_engine.image import batch_to_device as _batch_to_device_impl
from .._io_engine.image import read_hdus as _read_hdus_impl
from .._io_engine.image import read_image as _read_image_impl
from .._io_engine.image_meta import (
    get_image_meta as _get_image_meta_impl,
    resolve_image_mmap as _resolve_image_mmap_impl,
    should_use_cold_nommap as _should_use_cold_nommap_impl,
)
from .._io_engine.read_dispatch import read_unified as _read_unified_impl
from .._io_engine.table_api import read_table as _read_table_impl
from .._io_engine.table_streaming import (
    read_large_table as _read_large_table_impl,
)
from .._io_engine.table_streaming import stream_table as _stream_table_impl

_log = _stdlib_logging.getLogger(__name__)
_DEBUG_SCALE = os.environ.get("TORCHSKY_DEBUG_SCALE") == "1"
_COLD_NOMMAP = os.environ.get("TORCHSKY_COLD_NOMMAP") == "1"
_COLD_NOCACHE = os.environ.get("TORCHSKY_COLD_NOCACHE") == "1"
_READ_EXC_TYPES = (
    RuntimeError,
    OSError,
    ValueError,
    TypeError,
    AttributeError,
    MemoryError,
)


def read_fast(*args: Any, **kwargs: Any):
    from .._fastio import read as _read_fast

    return _read_fast(*args, **kwargs)


def _invalidate_path_caches(path: str) -> None:
    _invalidate_path_caches_impl(path, table)


def _cpp_module():
    root = sys.modules.get("torchsky")
    return getattr(root, "cpp", cpp)


def _read_check_cache(*args: Any, **kwargs: Any):
    return _check_read_cache_impl(
        path=args[0],
        hdu=args[1],
        device=args[2],
        fp16=args[3],
        bf16=args[4],
        columns=args[5],
        start_row=args[6],
        num_rows=args[7],
        return_header=args[8],
        cache_capacity=args[9],
        invalidate_path=_invalidate_path_caches,
    )


def _get_image_meta(path: str, hdu: int):
    return _get_image_meta_impl(path, hdu, cpp_module=_cpp_module())


def _should_use_cold_nommap(
    path: str, hdu: int, cache_capacity: int, mmap: bool
) -> bool:
    return _should_use_cold_nommap_impl(
        path,
        hdu,
        cache_capacity,
        mmap,
        force_cold_nommap=_COLD_NOMMAP,
        get_image_meta_func=_get_image_meta,
    )


def _resolve_image_mmap(path: str, hdu: int, mmap: bool | str, cache_capacity: int):
    return _resolve_image_mmap_impl(
        path,
        hdu,
        mmap,
        cache_capacity,
        get_image_meta_func=_get_image_meta,
        should_use_cold_nommap_func=_should_use_cold_nommap,
    )


def read(
    path: Any,
    hdu: Any = None,
    device: str = "cpu",
    mmap: bool | str = "auto",
    options: Any = None,
    return_header: bool = False,
    **kwargs: Any,
):
    return _read_unified_impl(
        cpp_module=_cpp_module(),
        path=path,
        hdu=hdu,
        device=device,
        mmap=mmap,
        options=options,
        return_header=return_header,
        kwargs=dict(kwargs),
        autodetect_hdu=_autodetect_hdu_impl,
        batch_to_device=_batch_to_device_impl,
        resolve_image_mmap=_resolve_image_mmap,
        read_check_cache=_read_check_cache,
        read_header=_read_header_fast_impl,
        debug_scale=_DEBUG_SCALE,
        cold_nocache=_COLD_NOCACHE,
        read_exc_types=_READ_EXC_TYPES,
        logger=_log,
    )


def read_image(*args: Any, **kwargs: Any):
    from .. import get_header as _get_header

    kwargs.setdefault("fallback_get_header", _get_header)
    return _read_image_impl(*args, **kwargs)


def read_table(*args: Any, **kwargs: Any):
    return _read_table_impl(read, *args, **kwargs)


def read_hdus(*args: Any, **kwargs: Any):
    return _read_hdus_impl(*args, **kwargs)


def read_subset(*args: Any, **kwargs: Any):
    from .. import read_subset as _read_subset

    return _read_subset(*args, **kwargs)


def open(*args: Any, **kwargs: Any):
    return _open_hdulist_impl(*args, **kwargs)


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
    from .. import get_header as _get_header

    return _stream_table_impl(_get_header, *args, **kwargs)


def read_large_table(*args: Any, **kwargs: Any):
    from .. import get_header as _get_header

    return _read_large_table_impl(_get_header, *args, **kwargs)


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
