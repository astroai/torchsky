"""
torchsky: High-performance FITS I/O for PyTorch

This module provides efficient FITS file reading and writing capabilities
optimized for PyTorch tensors and pytorch-frame TensorFrames.
"""

import importlib
import logging as _stdlib_logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Iterator,
    Tuple,
    Sequence,
    overload,
    Literal,
    Unpack,
)

import torch
import os
import warnings
from torch import Tensor

from .buffer import clear_buffers, configure_buffers, get_buffer_stats
from .cache import clear_cache, configure_for_environment, get_cache_stats
from .core import ChecksumVerifier, CompressionType, FITSCore
from .dataloader import (
    create_dataloader,
    create_fits_dataloader,
    create_streaming_dataloader,
    create_table_dataloader,
)
from .datasets import FITSDataset, IterableFITSDataset, TableChunkDataset
from .frame import read_tensor_frame, to_tensor_frame, write_tensor_frame
from .hdu import Card, HDUList, Header, TableHDU, TableHDURef, TensorHDU
from .interop import to_arrow, to_pandas, to_polars
from . import table
from . import sphere
from . import io
from ._io_engine.batch import (
    get_batch_info as _get_batch_info_impl,
    read_batch as _read_batch_impl,
)
from ._io_engine.caches import (
    auto_hdu_cache as _auto_hdu_cache,
    auto_mmap_cache as _auto_mmap_cache,
    cache_stats as _cache_stats,
    check_read_cache as _check_read_cache_impl,
    clear_python_caches as _clear_python_file_caches,
    cold_nommap_cache as _cold_nommap_cache,
    file_cache as _file_cache,
    file_handle_cache as _file_handle_cache,
    file_handle_sig_cache as _file_handle_sig_cache,
    get_cached_handle as _get_cached_handle,
    get_cached_hdu_type as _get_cached_hdu_type,
    get_cache_performance as _get_io_cache_performance,
    hdu_type_cache as _hdu_type_cache,
    image_meta_cache as _image_meta_cache,
    invalidate_path_caches as _invalidate_io_path_caches,
    path_signature as _path_signature,
    set_cached_hdu_type as _set_cached_hdu_type,
)
from ._io_engine.checksum_api import verify_checksums as _verify_checksums_impl
from ._io_engine.checksum_api import write_checksums as _write_checksums_impl
from ._io_engine.hdu_api import (
    autodetect_hdu as _autodetect_hdu,
    find_first_hdu as _find_first_hdu,
    get_header as _get_header_impl,
    get_wcs as _get_wcs_impl,
    open_hdulist as _open_hdulist_impl,
    read_header_fast as _read_header_fast,
)
from ._io_engine.image import (
    batch_to_device as _batch_to_device,
    read_hdus as _read_hdus_impl,
    read_image as _read_image_impl,
)
from ._io_engine.image_meta import (
    get_image_meta as _get_image_meta_impl,
    get_image_meta_from_handle as _get_image_meta_from_handle_impl,
    resolve_image_mmap as _resolve_image_mmap_impl,
    should_use_cold_nommap as _should_use_cold_nommap_impl,
)
from ._io_engine.options import ReadOptions, ReadOptionsKwargs
from ._io_engine.read_dispatch import (
    read_batch_hdus as _read_batch_hdus_impl,
    read_batch_paths as _read_batch_paths_impl,
    read_cpu_fast_path as _read_cpu_fast_path_impl,
    read_fallback as _read_fallback_impl,
    read_fallback_image as _read_fallback_image_impl,
    read_fallback_table as _read_fallback_table_impl,
    read_generic_fast_path as _read_generic_fast_path_impl,
    read_scaled_cpu_fast as _read_scaled_cpu_fast_impl,
    read_unified as _read_unified_impl,
)
from ._io_engine.subset import (
    SubsetReader,
    open_subset_reader as _open_subset_reader_impl,
    read_subset as _read_subset_impl,
)
from ._io_engine.table_api import read_table as _read_table_impl
from ._io_engine.table_streaming import (
    read_large_table as _read_large_table_impl,
    stream_table as _stream_table_impl,
)
from ._io_engine.write_api import (
    _can_use_cpp_table_writer,
    _coerce_compressed_hdu_item,
    _detach_hdus_for_rewrite,
    _host_tensor_for_fits_write,
    _normalize_cpp_table_data,
    _normalize_list_sequence,
    _normalize_ndarray_column,
    _normalize_vla_item,
    _resolve_compression_algorithm,
    _sanitize_header_for_compressed_write,
    _sanitize_table_header_for_write,
    _TableHDUWriteProxy,
    _write_hdus_with_optional_compression,
    _write_header_cards_if_supported,
)
from ._io_engine.write_api import delete_hdu as _delete_hdu_impl
from ._io_engine.write_api import insert_hdu as _insert_hdu_impl
from ._io_engine.write_api import replace_hdu as _replace_hdu_impl
from ._io_engine.write_api import write as _write_impl
from .surfaces import (
    APISurface,
    API_SURFACES,
    SurfacePillar,
    SurfaceTier,
    api_surfaces_by_pillar,
    api_surfaces_by_tier,
    get_api_surface,
)
from .header_parser import fast_parse_header
from .transforms import (
    AsinhStretch,
    CenterCrop,
    Compose,
    GaussianNoise,
    LogStretch,
    MinMaxScale,
    Normalize,
    PerturbByError,
    PoissonNoise,
    PowerStretch,
    RandomCrop,
    RandomFlip,
    RandomRotation,
    RedshiftShift,
    RobustScale,
    ToDevice,
    ZScale,
    create_inference_transform,
    create_training_transform,
    create_validation_transform,
)
from .wcs import WCS

_log = _stdlib_logging.getLogger(__name__)

_PRESERVED_PRIVATE_CACHE_ALIASES = (
    _auto_hdu_cache,
    _auto_mmap_cache,
    _cold_nommap_cache,
    _file_cache,
    _file_handle_cache,
    _file_handle_sig_cache,
    _find_first_hdu,
    _get_cached_hdu_type,
    _hdu_type_cache,
    _image_meta_cache,
    _path_signature,
    _set_cached_hdu_type,
)

_PRESERVED_PRIVATE_WRITE_ALIASES = (
    _can_use_cpp_table_writer,
    _coerce_compressed_hdu_item,
    _detach_hdus_for_rewrite,
    _host_tensor_for_fits_write,
    _normalize_cpp_table_data,
    _normalize_list_sequence,
    _normalize_ndarray_column,
    _normalize_vla_item,
    _resolve_compression_algorithm,
    _sanitize_header_for_compressed_write,
    _sanitize_table_header_for_write,
    _TableHDUWriteProxy,
    _write_hdus_with_optional_compression,
    _write_header_cards_if_supported,
)

# Recoverable errors when probing C++/FITS image fast paths (avoid bare ``except Exception``).
_READ_EXC_TYPES = (
    RuntimeError,
    OSError,
    ValueError,
    TypeError,
    AttributeError,
    MemoryError,
)

# Suppress noisy torch.jit.script deprecation warnings on Python 3.14+
# This is a temporary measure until the codebase is migrated to torch.compile
warnings.filterwarnings(
    "ignore", message=".*torch.jit.script.*", category=DeprecationWarning
)

def _invalidate_path_caches(path: str) -> None:
    """Invalidate Python-side caches/handles for a path that is being modified."""
    _invalidate_io_path_caches(path, table)


# Auto-configure cache and buffers on import
configure_for_environment()
try:
    # Optional CFITSIO cache tuning via env (safe defaults if unset)
    import torchsky.cpp as cpp

    _cache_mb = os.environ.get("TORCHSKY_CFITSIO_CACHE_MB")
    _cache_files = os.environ.get("TORCHSKY_CFITSIO_CACHE_FILES")
    if _cache_mb is not None or _cache_files is not None:
        max_files = int(_cache_files) if _cache_files is not None else 32
        max_mb = int(_cache_mb) if _cache_mb is not None else 256
        cpp.configure_cache(max_files, max_mb)
except Exception:
    pass

__version__ = "0.4.0.dev0"
__all__ = [
    # Core I/O functions
    "read",
    "read_image",
    "read_table",
    "write",
    "insert_hdu",
    "replace_hdu",
    "delete_hdu",
    "open",
    "get_header",
    "get_wcs",
    "read_subset",
    "read_hdus",
    "open_subset_reader",
    "SubsetReader",
    "io",
    "APISurface",
    "API_SURFACES",
    "SurfacePillar",
    "SurfaceTier",
    "api_surfaces_by_pillar",
    "api_surfaces_by_tier",
    "get_api_surface",
    # Batch operations
    "read_batch",
    "get_batch_info",
    "get_cache_performance",
    "clear_file_cache",
    "read_large_table",
    "stream_table",
    # HDU classes
    "HDUList",
    "TensorHDU",
    "TableHDU",
    "TableHDURef",
    "Header",
    "Card",
    # WCS functionality
    "WCS",
    "ChecksumVerifier",
    # Dataset classes
    "FITSDataset",
    "IterableFITSDataset",
    "TableChunkDataset",
    # DataLoader factories
    "create_dataloader",
    "create_fits_dataloader",
    "create_streaming_dataloader",
    "create_table_dataloader",
    # Transforms
    "ZScale",
    "AsinhStretch",
    "LogStretch",
    "PowerStretch",
    "Normalize",
    "MinMaxScale",
    "RobustScale",
    "RandomCrop",
    "CenterCrop",
    "RandomFlip",
    "GaussianNoise",
    "PoissonNoise",
    "RandomRotation",
    "RedshiftShift",
    "PerturbByError",
    "ToDevice",
    "Compose",
    "create_training_transform",
    "create_validation_transform",
    "create_inference_transform",
    # Core types
    "FITSCore",
    "CompressionType",
    # Header parsing
    "fast_parse_header",
    # Utility functions
    "configure_for_environment",
    "get_cache_stats",
    "clear_cache",
    "configure_buffers",
    "get_buffer_stats",
    "clear_buffers",
    # Integration
    "to_tensor_frame",
    "read_tensor_frame",
    "write_tensor_frame",
    "to_pandas",
    "to_polars",
    "to_arrow",
    "table",
    # Split-ready namespaces
    "sphere",
    "coords",
    "geom",
    "map",
    "encoders",
    "catalogs",
]


def _get_vla_columns(header_data: Optional[Dict[str, Any]]) -> set:
    """Extract VLA column names from header metadata."""
    vla_columns = set()
    if not header_data:
        return vla_columns
    for key, value in header_data.items():
        if not isinstance(key, str) or not key.startswith("TFORM"):
            continue
        if not isinstance(value, str) or not value:
            continue
        code = value.strip().upper()
        if code.startswith("P") or code.startswith("Q"):
            idx = key[5:]
            name = header_data.get(f"TTYPE{idx}")
            if isinstance(name, str) and name:
                vla_columns.add(name)
            else:
                vla_columns.add(f"COL{idx}")
    return vla_columns


def _get_scaled_columns(header_data: Optional[Dict[str, Any]]) -> set:
    """Extract scaled column names (TSCALn/TZEROn) from header metadata."""
    scaled_columns = set()
    if not header_data:
        return scaled_columns
    for key, value in header_data.items():
        if not isinstance(key, str):
            continue
        key_upper = key.upper()
        if not (key_upper.startswith("TSCAL") or key_upper.startswith("TZERO")):
            continue
        idx = key_upper[5:]
        if not idx:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            val = None
        is_scaled = False
        if key_upper.startswith("TSCAL"):
            is_scaled = val is None or val != 1.0
        else:
            is_scaled = val is None or val != 0.0
        if is_scaled:
            name = header_data.get(f"TTYPE{idx}")
            if isinstance(name, str) and name:
                scaled_columns.add(name)
            else:
                scaled_columns.add(f"COL{idx}")
    return scaled_columns


def _get_image_meta(path: str, hdu: int):
    return _get_image_meta_impl(path, hdu, cpp_module=globals().get("cpp"))


def _get_image_meta_from_handle(file_handle, path: str, hdu: int):
    """Fetch image metadata using an already-open FITS handle (avoids open/close)."""
    return _get_image_meta_from_handle_impl(
        file_handle, path, hdu, read_header=_read_header_fast
    )


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


def _resolve_image_mmap(
    path: str,
    hdu: int,
    mmap: Union[bool, str],
    cache_capacity: int,
) -> bool:
    """Resolve mmap mode for image reads."""
    return _resolve_image_mmap_impl(
        path,
        hdu,
        mmap,
        cache_capacity,
        get_image_meta_func=_get_image_meta,
        should_use_cold_nommap_func=_should_use_cold_nommap,
    )


def _read_scaled_cpu_fast(path: str, hdu: int = 0, mmap: bool = True) -> Tensor:
    """Internal helper for the CPU scaled fast path."""
    return _read_scaled_cpu_fast_impl(globals().get("cpp"), path, hdu, mmap)


def read_image(
    path: str,
    hdu: int = 0,
    device: str = "cpu",
    mmap: bool = True,
    handle_cache: bool = True,
    fp16: bool = False,
    bf16: bool = False,
    raw_scale: bool = False,
    return_header: bool = False,
) -> Union[Tensor, Tuple[Tensor, Header]]:
    """Read image data through a direct low-level path (no runtime policy dispatch).

    This path is intended for deterministic performance profiling and advanced users.
    It bypasses Python-side policy logic and calls C++ image readers directly.
    By default it reuses C++ shared FITS handles (`handle_cache=True`) for repeated reads.
    """
    return _read_image_impl(
        path=path,
        hdu=hdu,
        device=device,
        mmap=mmap,
        handle_cache=handle_cache,
        fp16=fp16,
        bf16=bf16,
        raw_scale=raw_scale,
        return_header=return_header,
        fallback_get_header=get_header,
    )


def read_table(
    path: str,
    hdu: int = 1,
    columns: Optional[List[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    cache_capacity: int = 10,
    handle_cache_capacity: int = 16,
    fast_header: bool = True,
    return_header: bool = False,
    policy: str = "default",
):
    """Read a table HDU as a dictionary of tensors/lists.

    This is a convenience wrapper around `read()` with table-centric arguments.
    Uses the same ``mmap='auto'`` default as :func:`read` (explicit ``True``/``False``
    still overrides).
    """
    return _read_table_impl(
        read_func=read,
        path=path,
        hdu=hdu,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        device=device,
        mmap=mmap,
        cache_capacity=cache_capacity,
        handle_cache_capacity=handle_cache_capacity,
        fast_header=fast_header,
        return_header=return_header,
        policy=policy,
    )


@overload
def read(  # noqa: PLR0913
    path: str,
    hdu: Union[int, str, None] = None,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    options: Optional[ReadOptions] = None,
    return_header: Literal[False] = False,
    **kwargs: Unpack[ReadOptionsKwargs],
) -> Union[Tensor, Dict[str, Any]]: ...


@overload
def read(  # noqa: PLR0913
    path: str,
    hdu: Union[int, str, None] = None,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    options: Optional[ReadOptions] = None,
    return_header: Literal[True] = ...,
    **kwargs: Unpack[ReadOptionsKwargs],
) -> Tuple[Union[Tensor, Dict[str, Any]], Header]: ...


@overload
def read(  # noqa: PLR0913
    path: Union[List[str], Tuple[str, ...]],
    hdu: Union[int, str, None] = None,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    options: Optional[ReadOptions] = None,
    return_header: Literal[False] = False,
    **kwargs: Unpack[ReadOptionsKwargs],
) -> List[Union[Tensor, Dict[str, Any]]]: ...


@overload
def read(  # noqa: PLR0913
    path: Union[List[str], Tuple[str, ...]],
    hdu: Union[int, str, None] = None,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    options: Optional[ReadOptions] = None,
    return_header: Literal[True] = ...,
    **kwargs: Unpack[ReadOptionsKwargs],
) -> List[Tuple[Union[Tensor, Dict[str, Any]], Header]]: ...


def _read_batch_paths(
    path,
    hdu,
    device,
    mmap,
    fp16,
    bf16,
    raw_scale,
    scale_on_device,
    columns,
    start_row,
    num_rows,
    cache_capacity,
    handle_cache_capacity,
    fast_header,
    return_header,
    mode,
    policy,
):
    return _read_batch_paths_impl(
        cpp_module=globals().get("cpp"),
        path=path,
        hdu=hdu,
        device=device,
        mmap=mmap,
        fp16=fp16,
        bf16=bf16,
        raw_scale=raw_scale,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        cache_capacity=cache_capacity,
        handle_cache_capacity=handle_cache_capacity,
        fast_header=fast_header,
        return_header=return_header,
        mode=mode,
        policy=policy,
        autodetect_hdu=_autodetect_hdu,
        batch_to_device=_batch_to_device,
        read_func=read,
    )


def _read_batch_hdus(
    path,
    hdu,
    device,
    mmap,
    fp16,
    bf16,
    raw_scale,
    scale_on_device,
    columns,
    start_row,
    num_rows,
    cache_capacity,
    handle_cache_capacity,
    fast_header,
    return_header,
    mode,
    policy,
):
    return _read_batch_hdus_impl(
        cpp_module=globals().get("cpp"),
        path=path,
        hdu=hdu,
        device=device,
        mmap=mmap,
        fp16=fp16,
        bf16=bf16,
        raw_scale=raw_scale,
        scale_on_device=scale_on_device,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        cache_capacity=cache_capacity,
        handle_cache_capacity=handle_cache_capacity,
        fast_header=fast_header,
        return_header=return_header,
        policy=policy,
        batch_to_device=_batch_to_device,
        read_func=read,
    )


def _read_cpu_fast_path(
    path, hdu, mmap, cache_capacity, handle_cache_capacity, fp16, bf16, force_image
):
    return _read_cpu_fast_path_impl(
        cpp_module=globals().get("cpp"),
        path=path,
        hdu=hdu,
        mmap=mmap,
        cache_capacity=cache_capacity,
        handle_cache_capacity=handle_cache_capacity,
        fp16=fp16,
        bf16=bf16,
        force_image=force_image,
        resolve_image_mmap=_resolve_image_mmap,
        get_cached_handle=_get_cached_handle,
        cache_stats=_cache_stats,
        read_exc_types=_READ_EXC_TYPES,
        debug_scale=_DEBUG_SCALE,
        logger=_log,
    )


def _read_generic_fast_path(
    path,
    hdu,
    device,
    mmap,
    cache_capacity,
    handle_cache_capacity,
    fp16,
    bf16,
    raw_scale,
    scale_on_device,
    force_image,
    debug_scale,
):
    return _read_generic_fast_path_impl(
        cpp_module=globals().get("cpp"),
        path=path,
        hdu=hdu,
        device=device,
        mmap=mmap,
        cache_capacity=cache_capacity,
        fp16=fp16,
        bf16=bf16,
        raw_scale=raw_scale,
        scale_on_device=scale_on_device,
        force_image=force_image,
        debug_scale=debug_scale,
        cold_nocache=_COLD_NOCACHE,
        resolve_image_mmap=_resolve_image_mmap,
        read_exc_types=_READ_EXC_TYPES,
        logger=_log,
    )


def _read_check_cache(
    path,
    hdu,
    device,
    fp16,
    bf16,
    columns,
    start_row,
    num_rows,
    return_header,
    cache_capacity,
):
    return _check_read_cache_impl(
        path=path,
        hdu=hdu,
        device=device,
        fp16=fp16,
        bf16=bf16,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        return_header=return_header,
        cache_capacity=cache_capacity,
        invalidate_path=_invalidate_path_caches,
    )


def _read_fallback_image(
    file_handle,
    path,
    hdu_num,
    device,
    mmap,
    fp16,
    bf16,
    cache_capacity,
    fast_header,
    return_header,
    cache_key,
    use_cache,
):
    return _read_fallback_image_impl(
        cpp_module=globals().get("cpp"),
        file_handle=file_handle,
        path=path,
        hdu_num=hdu_num,
        device=device,
        mmap=mmap,
        fp16=fp16,
        bf16=bf16,
        cache_capacity=cache_capacity,
        fast_header=fast_header,
        return_header=return_header,
        cache_key=cache_key,
        use_cache=use_cache,
        resolve_image_mmap=_resolve_image_mmap,
        read_header=_read_header_fast,
    )


def _read_fallback_table(
    file_handle,
    path,
    hdu_num,
    device,
    mmap,
    cache_capacity,
    fast_header,
    return_header,
    cache_key,
    use_cache,
    columns,
    start_row,
    num_rows,
    header_data,
    header,
):
    return _read_fallback_table_impl(
        cpp_module=globals().get("cpp"),
        file_handle=file_handle,
        path=path,
        hdu_num=hdu_num,
        device=device,
        mmap=mmap,
        cache_capacity=cache_capacity,
        fast_header=fast_header,
        return_header=return_header,
        cache_key=cache_key,
        use_cache=use_cache,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        header_data=header_data,
        header=header,
        read_header=_read_header_fast,
    )


def _read_fallback(
    path,
    hdu,
    device,
    mmap,
    fp16,
    bf16,
    cache_capacity,
    handle_cache_capacity,
    fast_header,
    return_header,
    force_image,
    force_table,
    hdu_type_hint,
    columns,
    start_row,
    num_rows,
):
    return _read_fallback_impl(
        cpp_module=globals().get("cpp"),
        path=path,
        hdu=hdu,
        device=device,
        mmap=mmap,
        fp16=fp16,
        bf16=bf16,
        cache_capacity=cache_capacity,
        handle_cache_capacity=handle_cache_capacity,
        fast_header=fast_header,
        return_header=return_header,
        force_image=force_image,
        force_table=force_table,
        hdu_type_hint=hdu_type_hint,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        read_check_cache=_read_check_cache,
        resolve_image_mmap=_resolve_image_mmap,
        read_header=_read_header_fast,
    )


def read(  # noqa: PLR0913
    path: Union[str, List[str], Tuple[str, ...]],
    hdu: Union[int, str, List[int], Tuple[int, ...], None] = None,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    options: Optional[ReadOptions] = None,
    return_header: bool = False,
    **kwargs: Unpack[ReadOptionsKwargs],
) -> Any:
    """Read FITS data with optimizations.

    Args:
        path: File path or cutout specification
        hdu: HDU index, name, `None`, or `"auto"` (default: first HDU with payload)
        mode: Read intent: `'auto'` (default), `'image'`, or `'table'`
        policy: Dispatch mode compatibility flag: `'default'` or `'smart'`.
        device: Target device ('cpu', 'cuda')
        mmap: Memory-mapping mode. `True`/`False` are explicit, `'auto'` chooses
            per-HDU defaults (compressed images default to non-mmap).
        fp16: Convert to half precision
        bf16: Convert to bfloat16
        columns: List of column names for table reading (None = all columns)
        start_row: Starting row for table reading (1-based)
        num_rows: Number of rows to read (-1 = all rows)
        cache_capacity: File cache capacity (max entries)
        handle_cache_capacity: Open-handle cache capacity (max open files). This caches
            CFITSIO handles only (no data caching) and is a big win for repeated reads,
            especially for compressed images.
        fast_header: Use fast bulk header parsing (default: True)
        return_header: Whether to return header dict (default: False)

    Returns:
        torch.Tensor for images or dict for tables; header dict if return_header=True

    Raises:
        ValueError: Invalid parameters
        FileNotFoundError: File not found
        RuntimeError: FITS reading errors
    """
    return _read_unified_impl(
        cpp_module=globals().get("cpp"),
        path=path,
        hdu=hdu,
        device=device,
        mmap=mmap,
        options=options,
        return_header=return_header,
        kwargs=dict(kwargs),
        autodetect_hdu=_autodetect_hdu,
        batch_to_device=_batch_to_device,
        resolve_image_mmap=_resolve_image_mmap,
        read_check_cache=_read_check_cache,
        read_header=_read_header_fast,
        debug_scale=_DEBUG_SCALE,
        cold_nocache=_COLD_NOCACHE,
        read_exc_types=_READ_EXC_TYPES,
        logger=_log,
    )


def read_subset(
    path: str,
    hdu: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    handle_cache_capacity: int = 16,
) -> Tensor:
    """Read a rectangular subset of an image HDU.

    Args:
        path: Path to FITS file
        hdu: HDU index
        x1, y1: Start coordinates (0-based, inclusive)
        x2, y2: End coordinates (0-based, exclusive)
        handle_cache_capacity: Open-handle cache capacity (max open files). Reusing
            handles can significantly reduce overhead for many small cutouts.

    Returns:
        Tensor containing the subset
    """
    return _read_subset_impl(
        get_cached_handle=_get_cached_handle,
        path=path,
        hdu=hdu,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        handle_cache_capacity=handle_cache_capacity,
    )


def read_hdus(
    path: str,
    hdus: Sequence[Union[int, str]],
    *,
    device: str = "cpu",
    mmap: bool = True,
    return_header: bool = False,
):
    """Read multiple image HDUs from one file using a direct one-handle path.

    This is an explicit user-facing fast path for multi-HDU image reads.
    """
    return _read_hdus_impl(
        path=path,
        hdus=hdus,
        device=device,
        mmap=mmap,
        return_header=return_header,
    )


def open_subset_reader(path: str, hdu: int = 0, device: str = "cpu") -> SubsetReader:
    """Open a persistent cutout reader for repeated subsets on one HDU."""
    return _open_subset_reader_impl(path=path, hdu=hdu, device=device)


def write(
    path: str,
    data: Any,
    header: Header = None,
    overwrite: bool = False,
    compress: Union[bool, str] = False,
):
    """Write data to FITS file.

    Args:
        path: Output file path
        data: Data to write (Tensor, TensorFrame, or HDUList)
        header: Optional FITS header dictionary
        overwrite: Whether to overwrite existing files
        compress: Whether to use tile compression (Rice algorithm)

    Image tensors on non-CPU devices are detached and copied to CPU before
    the CFITSIO writer runs (in-memory input tensors are not modified).
    """
    return _write_impl(
        path,
        data,
        header=header,
        overwrite=overwrite,
        compress=compress,
    )


def insert_hdu(
    path: str,
    data: Any,
    index: int = 1,
    header: Optional[Dict[str, Any]] = None,
    compress: Union[bool, str] = False,
) -> None:
    """Insert a new HDU into an existing FITS file."""
    return _insert_hdu_impl(
        path,
        data,
        index=index,
        header=header,
        compress=compress,
    )


def replace_hdu(
    path: str,
    hdu: Union[int, str],
    data: Any,
    header: Optional[Dict[str, Any]] = None,
    compress: Union[bool, str] = False,
) -> None:
    """Replace an HDU by index or EXTNAME."""
    return _replace_hdu_impl(
        path,
        hdu,
        data,
        header=header,
        compress=compress,
    )


def delete_hdu(
    path: str,
    hdu: Union[int, str],
    compress: Union[bool, str] = False,
) -> None:
    """Delete an HDU by index or EXTNAME."""
    return _delete_hdu_impl(path, hdu, compress=compress)


def read_batch(
    file_paths: List[str],
    hdu: int = 0,
    device: str = "cpu",
    *,
    strict: bool = False,
) -> List[Tensor]:
    """Read multiple FITS files in batch.

    Args:
        file_paths: List of file paths to read
        hdu: HDU index to read from each file
        device: Target device for tensors
        strict: If True, raise on the first read failure instead of skipping files.

    Returns:
        List of tensors from each file (only successfully read when strict=False).
    """
    return _read_batch_impl(
        read_func=read,
        read_exc_types=_READ_EXC_TYPES,
        log=_log,
        file_paths=file_paths,
        hdu=hdu,
        device=device,
        strict=strict,
    )


def get_batch_info(file_paths: List[str]) -> Dict[str, Any]:
    """Get information about a batch of FITS files.

    Args:
        file_paths: List of file paths

    Returns:
        Dictionary with batch statistics
    """
    return _get_batch_info_impl(file_paths)


def get_cache_performance() -> Dict[str, Any]:
    """Get cache performance statistics.

    Returns:
        Dictionary with cache statistics
    """
    return _get_io_cache_performance()


def clear_file_cache(
    data: bool = True,
    handles: bool = True,
    meta: bool = True,
    hdu_types: bool = True,
    stats: bool = True,
    cpp: bool = True,
):
    """Clear Python/C++ caches.

    - data: cached returned tensors/tables (Python LRU)
    - handles: cached open CFITSIO handles (Python LRU)
    - meta: cached image metadata (BITPIX/NAXIS/BSCALE/BZERO)
    - hdu_types: cached HDU type hints (IMAGE/TABLE)
    - stats: reset cache performance counters
    - cpp: clear C++-side caches (if available)
    """
    _clear_python_file_caches(
        data=data,
        handles=handles,
        meta=meta,
        hdu_types=hdu_types,
        stats=stats,
    )

    if cpp:
        try:
            import torchsky.cpp as _cpp

            _cpp.clear_file_cache()
            if hasattr(_cpp, "clear_shared_read_meta_cache"):
                _cpp.clear_shared_read_meta_cache()
        except (AttributeError, RuntimeError):
            pass


def write_checksums(path: str, hdu: int = 0) -> None:
    """Compute and write DATASUM/CHECKSUM keywords for an HDU (CFITSIO)."""
    return _write_checksums_impl(path, hdu)


def verify_checksums(path: str, hdu: int = 0) -> Dict[str, Any]:
    """Verify DATASUM/CHECKSUM keywords for an HDU (CFITSIO).

    Returns a dict with `datastatus`, `hdustatus`, and `ok`. Status values:
    - 1: verification correct
    - 0: keyword not present / undefined
    - -1: verification not correct
    """
    return _verify_checksums_impl(path, hdu)


def read_large_table(
    file_path: str,
    hdu: int = 1,
    max_memory_mb: int = 100,
    streaming: bool = False,
    return_iterator: bool = False,
) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """Read large FITS table with memory management.

    Args:
        file_path: Path to FITS file
        hdu: HDU index
        max_memory_mb: Maximum memory usage in MB
        streaming: Whether to use streaming mode
        return_iterator: If True, return an iterator over chunks instead of a dict

    Returns:
        Dictionary with table data, or an iterator if return_iterator=True
    """
    return _read_large_table_impl(
        get_header_func=get_header,
        file_path=file_path,
        hdu=hdu,
        max_memory_mb=max_memory_mb,
        streaming=streaming,
        return_iterator=return_iterator,
    )


def stream_table(
    file_path: str,
    hdu: int = 1,
    columns: Optional[List[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    chunk_rows: int = 65536,
    mmap: bool = False,
    max_chunks: Optional[int] = None,
):
    """Yield table data in row chunks.

    Default ``chunk_rows`` matches :func:`torchsky.table.read` ``batch_size`` (65536).

    Falls back to a single full-table read if row streaming is unavailable.
    """
    yield from _stream_table_impl(
        get_header_func=get_header,
        file_path=file_path,
        hdu=hdu,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        chunk_rows=chunk_rows,
        mmap=mmap,
        max_chunks=max_chunks,
    )


def open(path: str, mode: str = "r") -> HDUList:
    """Open FITS file for reading/writing.

    Args:
        path: File path to open
        mode: File mode ('r' for read, 'w' for write)

    Returns:
        HDUList object for accessing HDUs

    Raises:
        FileNotFoundError: If file doesn't exist in read mode
        PermissionError: If insufficient permissions
        RuntimeError: For other FITS-related errors
    """
    return _open_hdulist_impl(path, mode)


def get_header(path: str, hdu: Union[int, str, None] = None) -> Header:
    """Get the header of a FITS file.

    Args:
        path: Path to the FITS file.
        hdu: HDU index, name, or `None`/`"auto"` for the first payload HDU
            (same convention as :func:`read` and :func:`get_wcs`).

    Returns:
        Header object.
    """
    return _get_header_impl(path, hdu, autodetect_hdu=_autodetect_hdu)


def get_wcs(
    path: str,
    hdu: Union[int, str, None] = "auto",
    device: Optional[Union[str, torch.device]] = None,
) -> WCS:
    """Build a WCS object directly from a FITS file/header HDU.

    Args:
        path: Path to FITS file.
        hdu: HDU index/name, or `"auto"`/`None` for first payload HDU.
        device: Optional torch device for WCS tensor buffers.

    Returns:
        WCS object initialized from the selected HDU header.
    """
    return _get_wcs_impl(path, hdu, device, get_header_func=get_header)


_DEBUG_SCALE = os.environ.get("TORCHSKY_DEBUG_SCALE") == "1"
_COLD_NOMMAP = os.environ.get("TORCHSKY_COLD_NOMMAP") == "1"
_COLD_NOCACHE = os.environ.get("TORCHSKY_COLD_NOCACHE") == "1"

_LAZY_NAMESPACES = {"io", "coords", "geom", "map", "encoders", "catalogs"}


def __getattr__(name: str):
    if name in _LAZY_NAMESPACES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
