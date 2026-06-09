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
from dataclasses import fields

import numpy as np
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
    read_fallback_image as _read_fallback_image_impl,
    read_fallback_table as _read_fallback_table_impl,
    read_generic_fast_path as _read_generic_fast_path_impl,
    read_scaled_cpu_fast as _read_scaled_cpu_fast_impl,
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
    _hdu_type_cache,
    _image_meta_cache,
    _path_signature,
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
    import torchsky.cpp as cpp

    hit, cached_res, cache_key = _read_check_cache(
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
    )
    if hit:
        return cached_res

    if isinstance(hdu, int) and hdu < 0:
        raise ValueError("HDU index must be non-negative")
    if start_row < 1:
        raise ValueError("start_row must be >= 1 (FITS uses 1-based indexing)")
    if num_rows < -1 or num_rows == 0:
        raise ValueError("num_rows must be > 0 or -1 for all rows")
    if device not in ["cpu", "cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")

    try:
        file_handle, cached_handle = _get_cached_handle(path, handle_cache_capacity)
        try:
            if isinstance(hdu, str):
                hdu_num = None
                if hasattr(cpp, "resolve_hdu_name_cached"):
                    try:
                        hdu_num = int(cpp.resolve_hdu_name_cached(path, hdu))
                    except Exception:
                        hdu_num = None

                if hdu_num is None:
                    num_hdus = cpp.get_num_hdus(file_handle)
                    for i in range(num_hdus):
                        try:
                            hdr = cpp.read_header(file_handle, i)
                            if hdr.get("EXTNAME") == hdu:
                                hdu_num = i
                                break
                        except Exception:
                            continue

                if hdu_num is None:
                    raise ValueError(f"HDU '{hdu}' not found in file")
            else:
                hdu_num = hdu

            header = None
            header_data = None
            hdu_type = hdu_type_hint if isinstance(hdu_num, int) else None
            if isinstance(hdu_num, int) and hdu_type is None:
                try:
                    hdu_type = cpp.get_hdu_type(file_handle, hdu_num)
                    _set_cached_hdu_type(path, hdu_num, hdu_type)
                except Exception:
                    hdu_type = None

            is_table_hdu = force_table or (hdu_type in {"ASCII_TABLE", "BINARY_TABLE"})
            if force_image:
                is_table_hdu = False

            if not is_table_hdu:
                try:
                    return _read_fallback_image(
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
                        cache_capacity > 0,
                    )
                except (RuntimeError, TypeError):
                    if force_image:
                        raise
                    if isinstance(hdu_num, int):
                        try:
                            hdu_type = cpp.get_hdu_type(file_handle, hdu_num)
                            _set_cached_hdu_type(path, hdu_num, hdu_type)
                        except Exception:
                            hdu_type = None
                    is_table_hdu = force_table or (
                        hdu_type in {"ASCII_TABLE", "BINARY_TABLE"}
                    )
                    if not is_table_hdu:
                        raise

            try:
                return _read_fallback_table(
                    file_handle,
                    path,
                    hdu_num,
                    device,
                    mmap,
                    cache_capacity,
                    fast_header,
                    return_header,
                    cache_key,
                    cache_capacity > 0,
                    columns,
                    start_row,
                    num_rows,
                    header_data,
                    header,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to read table extension: {e}")

        finally:
            if not cached_handle:
                try:
                    file_handle.close()
                except Exception:
                    pass

    except Exception as e:
        raise RuntimeError(f"Failed to read FITS file '{path}': {e}") from e


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
    opts = options if options is not None else ReadOptions()
    for field in fields(ReadOptions):
        if field.name in kwargs:
            setattr(opts, field.name, kwargs[field.name])

    fp16 = opts.fp16
    bf16 = opts.bf16
    raw_scale = opts.raw_scale
    scale_on_device = opts.scale_on_device
    use_cache = opts.use_cache
    columns = opts.columns
    start_row = opts.start_row
    num_rows = opts.num_rows
    cache_capacity = opts.cache_capacity
    handle_cache_capacity = opts.handle_cache_capacity
    fast_header = opts.fast_header
    mode = opts.mode
    policy = opts.policy

    if not path:
        raise ValueError("Path must be a non-empty string")

    if use_cache is not None and not isinstance(use_cache, bool):
        raise ValueError("use_cache must be bool when provided")
    if use_cache is True:
        if cache_capacity <= 0:
            cache_capacity = 10
        # Preserve historical behavior of reading by path in cached mode.
        handle_cache_capacity = 0
    elif use_cache is False:
        cache_capacity = 0
        handle_cache_capacity = 0

    if isinstance(path, (list, tuple)):
        if any(not isinstance(p, str) or not p for p in path):
            raise ValueError("Path must be a string or list of strings")
        return _read_batch_paths(
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
        )

    if not isinstance(path, str):
        raise ValueError("Path must be a string or list of strings")
    if isinstance(hdu, int) and hdu < 0:
        raise ValueError("HDU index must be a non-negative integer")
    if device not in ["cpu", "cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")
    if isinstance(mmap, str) and mmap.strip().lower() != "auto":
        raise ValueError("mmap must be bool or 'auto'")
    if not isinstance(mmap, (bool, str)):
        raise ValueError("mmap must be bool or 'auto'")
    mode = str(mode).strip().lower()
    if mode not in {"auto", "image", "table"}:
        raise ValueError("mode must be 'auto', 'image', or 'table'")
    policy = str(policy).strip().lower()
    if policy not in {"default", "smart"}:
        raise ValueError("policy must be 'default' or 'smart'")
    force_image = mode == "image"
    force_table = mode == "table"
    if force_image and (columns is not None or start_row != 1 or num_rows != -1):
        raise ValueError("mode='image' does not support table row/column options")

    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == "auto"):
        hdu = _autodetect_hdu(path, handle_cache_capacity)

    hdu_type_hint = _get_cached_hdu_type(path, hdu) if isinstance(hdu, int) else None
    is_cached_table_hdu = force_table or (
        hdu_type_hint in {"ASCII_TABLE", "BINARY_TABLE"}
    )

    skip_generic_image_fast_path = is_cached_table_hdu

    if (
        isinstance(hdu, (list, tuple))
        and not return_header
        and columns is None
        and start_row == 1
        and num_rows == -1
    ):
        return _read_batch_hdus(
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
        )

    debug_scale = _DEBUG_SCALE
    cpp_is_mocked = False
    if isinstance(hdu, int):
        try:
            import unittest.mock as _unittest_mock
            import torchsky.cpp as _cpp_mod

            cpp_is_mocked = isinstance(
                getattr(_cpp_mod, "read_full", None), _unittest_mock.Mock
            )
        except Exception:
            cpp_is_mocked = False

    if (
        scale_on_device
        and not raw_scale
        and not cpp_is_mocked
        and device == "cpu"
        and not return_header
        and isinstance(hdu, int)
        and columns is None
        and start_row == 1
        and num_rows == -1
        and not is_cached_table_hdu
    ):
        res, fallback = _read_cpu_fast_path(
            path,
            hdu,
            mmap,
            cache_capacity,
            handle_cache_capacity,
            fp16,
            bf16,
            force_image,
        )
        if not fallback:
            return res
        skip_generic_image_fast_path = True

    if (
        not return_header
        and isinstance(hdu, int)
        and columns is None
        and start_row == 1
        and num_rows == -1
        and not skip_generic_image_fast_path
    ):
        res = _read_generic_fast_path(
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
        )
        if res is not None:
            return res

    return _read_fallback(
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


def _host_tensor_for_fits_write(tensor: Tensor) -> Tensor:
    """Move image tensors to CPU for CFITSIO-backed writes (GPU tensors are not supported)."""
    if tensor.device.type != "cpu":
        return tensor.detach().cpu()
    return tensor


def _write_header_cards_if_supported(
    path: str,
    hdu: int,
    header: Optional[Dict[str, Any]],
) -> None:
    if not header:
        return
    header_obj = header if isinstance(header, Header) else Header(header)
    if not header_obj.cards:
        return
    try:
        import torchsky.cpp as cpp

        writer = getattr(cpp, "write_hdu_header_cards", None)
        if writer is None:
            return
        writer(path, int(hdu), list(header_obj.cards))
        _invalidate_path_caches(path)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return


def write(
    path: str,
    data,
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
    import os

    if not overwrite and os.path.exists(path):
        raise FileExistsError(
            f"File '{path}' already exists. Use overwrite=True to overwrite."
        )

    # The unified C++ cache and the Python-side handle cache can otherwise return
    # stale views of an overwritten file (mtime/size can be unchanged).
    _invalidate_path_caches(path)

    try:
        # Import C++ backend
        import torchsky.cpp as cpp

        hdus_to_write = []

        if compress:
            compressed_hdus: List[Any] = []
            if isinstance(data, Tensor):
                compressed_hdus = [
                    TensorHDU(
                        data=_host_tensor_for_fits_write(data),
                        header=Header(header or {}),
                    )
                ]
            elif isinstance(data, HDUList):
                compressed_hdus = list(getattr(data, "_hdus", []))
            elif isinstance(data, dict):
                if "data" in data:
                    item_hdu = _coerce_compressed_hdu_item(data)
                    compressed_hdus.append(item_hdu)
                else:
                    compressed_hdus.append(
                        _TableHDUWriteProxy(data, Header(header or {}))
                    )
            elif isinstance(data, (list, tuple)):
                for item in data:
                    compressed_hdus.append(_coerce_compressed_hdu_item(item))
            else:
                raise NotImplementedError(
                    "Compressed FITS writing supports tensors, tables, or HDU lists."
                )

            if header and compressed_hdus:
                first = compressed_hdus[0]
                merged = Header(dict(getattr(first, "header", {})))
                merged.update(dict(header))
                if isinstance(first, TensorHDU):
                    first._header = merged
                else:
                    first.header = merged

            _write_hdus_with_optional_compression(
                path, compressed_hdus, compress=compress
            )
            for out_hdu, item_hdu in enumerate(compressed_hdus, start=1):
                _write_header_cards_if_supported(
                    path, out_hdu, getattr(item_hdu, "header", None)
                )
            return

        if isinstance(data, HDUList):
            data.write(path, overwrite=overwrite)
            return

        # Check for table dictionary (col_name -> tensor)
        if isinstance(data, dict) and "data" not in data:
            if _can_use_cpp_table_writer(data):
                data = _normalize_cpp_table_data(data)
                header_obj = Header(header) if header else Header()
                cpp.write_fits_table(path, data, header_obj, overwrite)
                _write_header_cards_if_supported(path, 1, header_obj)
                return
            raise ValueError(
                "Dictionary table writes currently require CFITSIO-native column types "
                "(numeric/bool/complex, strings, or VLA lists). Unsupported object/structure "
                "columns should be converted before writing."
            )

        if isinstance(data, Tensor):
            # Single image HDU
            hdu_dict = {"data": _host_tensor_for_fits_write(data)}
            if header:
                hdu_dict["header"] = header
            hdus_to_write.append(hdu_dict)

        elif hasattr(data, "__iter__") and not isinstance(data, (str, Tensor)):
            # List of HDUs
            for item in data:
                if isinstance(item, dict):
                    # Already a dict (e.g. from previous read)
                    # Ensure 'data' is present
                    if "data" in item:
                        payload = item["data"]
                        if isinstance(payload, Tensor):
                            merged = dict(item)
                            merged["data"] = _host_tensor_for_fits_write(payload)
                            hdus_to_write.append(merged)
                        else:
                            hdus_to_write.append(item)
                elif isinstance(item, Tensor):
                    hdus_to_write.append({"data": _host_tensor_for_fits_write(item)})
                # Handle objects with .data attribute (like TensorHDU)
                elif hasattr(item, "data") and isinstance(item.data, Tensor):
                    hdu_dict = {"data": _host_tensor_for_fits_write(item.data)}
                    if hasattr(item, "header"):
                        hdu_dict["header"] = item.header
                    hdus_to_write.append(hdu_dict)
        else:
            raise ValueError(f"Unsupported data type for FITS writing: {type(data)}")

        # Call C++ writer
        cpp.write_fits_file(path, hdus_to_write, overwrite)
        for idx, item in enumerate(hdus_to_write):
            item_header = item.get("header") if isinstance(item, dict) else None
            _write_header_cards_if_supported(path, idx, item_header)

    except Exception as e:
        # Clean up partial file on error
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        raise RuntimeError(f"Failed to write FITS file '{path}': {e}") from e


def _can_use_cpp_table_writer(table_dict: Dict[str, Any]) -> bool:
    """Return True when all table columns can use the fast C++ writer."""
    if not table_dict:
        return False

    for value in table_dict.values():
        if isinstance(value, torch.Tensor):
            if value.dim() > 2:
                return False
            if value.is_complex():
                if value.dtype not in {torch.complex64, torch.complex128}:
                    return False
                continue
            if value.dtype not in {
                torch.bool,
                torch.uint8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.float32,
                torch.float64,
            }:
                return False
            continue

        if isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
            except ValueError:
                arr = np.asarray(value, dtype=object)
            if arr.dtype != np.object_:
                value = arr
            else:
                for item in value:
                    if item is None:
                        continue
                    if isinstance(item, (str, bytes, np.str_, np.bytes_)):
                        continue
                    if isinstance(item, torch.Tensor):
                        t = item.detach()
                        if t.dim() > 2:
                            return False
                        if t.is_complex():
                            if t.dtype not in {torch.complex64, torch.complex128}:
                                return False
                            continue
                        if t.dtype not in {
                            torch.bool,
                            torch.uint8,
                            torch.int16,
                            torch.int32,
                            torch.int64,
                            torch.float32,
                            torch.float64,
                        }:
                            return False
                        continue
                    arr_item = np.asarray(item)
                    if arr_item.ndim > 2:
                        return False
                    if np.iscomplexobj(arr_item):
                        if arr_item.dtype not in (np.complex64, np.complex128):
                            return False
                        continue
                    kind = arr_item.dtype.kind
                    itemsize = arr_item.dtype.itemsize
                    if kind in {"U", "S"}:
                        continue
                    if kind == "b":
                        continue
                    if kind == "u" and itemsize == 1:
                        continue
                    if kind == "i" and itemsize in (2, 4, 8):
                        continue
                    if kind == "f" and itemsize in (4, 8):
                        continue
                    return False
                continue

        if not isinstance(value, np.ndarray):
            return False
        if value.ndim > 2:
            return False
        if np.iscomplexobj(value):
            if value.dtype not in (np.complex64, np.complex128):
                return False
            continue
        if value.dtype.kind in {"U", "S"}:
            continue
        kind = value.dtype.kind
        itemsize = value.dtype.itemsize
        if kind == "b":
            continue
        if kind == "u" and itemsize == 1:
            continue
        if kind == "i" and itemsize in (2, 4, 8):
            continue
        if kind == "f" and itemsize in (4, 8):
            continue
        return False

    return True


def _normalize_vla_item(item: Any) -> np.ndarray:
    """Normalize a single VLA item."""
    if isinstance(item, torch.Tensor):
        t = item.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if t.dim() == 0:
            t = t.reshape(1)
        return np.ascontiguousarray(t.numpy())
    elif isinstance(item, np.ndarray):
        return np.ascontiguousarray(item)
    elif item is None:
        return np.asarray([], dtype=np.float32)
    else:
        return np.asarray(item)


def _normalize_list_sequence(items: list[Any]) -> Any:
    """Normalize a list or tuple sequence."""
    if items and all(
        isinstance(item, (str, bytes, np.str_, np.bytes_)) or item is None
        for item in items
    ):
        return items
    if any(isinstance(item, (list, tuple, np.ndarray, torch.Tensor)) for item in items):
        return [_normalize_vla_item(item) for item in items]
    return np.asarray(items)


def _normalize_ndarray_column(value: np.ndarray) -> Any:
    """Normalize an ndarray column."""
    if value.dtype == np.object_:
        return list(value)
    if value.dtype.kind in {"U", "S"}:
        return value.astype(str).tolist()
    return value


def _normalize_cpp_table_data(table_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize table data for the C++ writer (strings/VLA/object arrays)."""
    out: Dict[str, Any] = {}
    for name, value in table_dict.items():
        if isinstance(value, (list, tuple)):
            out[name] = _normalize_list_sequence(list(value))
        elif isinstance(value, np.ndarray):
            out[name] = _normalize_ndarray_column(value)
        else:
            out[name] = value
    return out


def _resolve_compression_algorithm(compress: Union[bool, str]) -> Optional[str]:
    """Normalize compress flag to a backend algorithm string or None."""
    if compress is False:
        return None
    if compress is True:
        return "RICE_1"
    if isinstance(compress, str):
        algo = compress.strip()
        return algo if algo else "RICE_1"
    raise TypeError("compress must be bool or compression algorithm string")


def _coerce_compressed_hdu_item(item: Any) -> Any:
    """Normalize compressed-write inputs to TensorHDU/TableHDU objects."""
    if isinstance(item, (TensorHDU, TableHDU)):
        return item
    if isinstance(item, TableHDURef):
        return item.materialize(device="cpu")
    if isinstance(item, Tensor):
        return TensorHDU(data=_host_tensor_for_fits_write(item), header=Header())
    if isinstance(item, dict):
        if "data" in item:
            img = item["data"]
            if not isinstance(img, Tensor):
                raise NotImplementedError(
                    "Compressed FITS writing supports tensor image payloads for dict HDUs."
                )
            return TensorHDU(
                data=_host_tensor_for_fits_write(img),
                header=Header(item.get("header", {})),
            )
        return _TableHDUWriteProxy(item, Header())
    raise NotImplementedError(
        f"Unsupported HDU payload for compressed write: {type(item)}"
    )


class _TableHDUWriteProxy:
    """Small table-HDU proxy for writer paths that should not build TensorFrame."""

    def __init__(self, raw_data: Dict[str, Any], header: Header):
        self._raw_data = _normalize_cpp_table_data(dict(raw_data))
        self.header = header


def _detach_hdus_for_rewrite(path: str) -> List[Any]:
    """Materialize file-backed HDUs so rewrite paths never hold stale handles."""
    with open(path) as hdul:
        detached: List[Any] = []
        for hdu in list(hdul._hdus):
            if isinstance(hdu, TensorHDU):
                detached.append(TensorHDU(data=hdu.to_tensor("cpu"), header=hdu.header))
            elif isinstance(hdu, TableHDU):
                detached.append(
                    TableHDU(dict(getattr(hdu, "_raw_data", {})), header=hdu.header)
                )
            elif isinstance(hdu, TableHDURef):
                mat = hdu.materialize(device="cpu")
                detached.append(
                    TableHDU(dict(getattr(mat, "_raw_data", {})), header=hdu.header)
                )
            else:
                detached.append(hdu)
    return detached


def _sanitize_header_for_compressed_write(
    header: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Drop structural/compression keys so CFITSIO can emit canonical metadata."""
    if not header:
        return {}

    skip_exact = {
        "SIMPLE",
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "EXTEND",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "THEAP",
        "BSCALE",
        "BZERO",
        "DATASUM",
        "CHECKSUM",
        "ZIMAGE",
        "ZCMPTYPE",
        "ZBITPIX",
        "ZNAXIS",
        "ZPCOUNT",
        "ZGCOUNT",
        "ZHECKSUM",
        "ZDATASUM",
    }
    skip_prefix = (
        "NAXIS",
        "ZNAXIS",
        "ZTILE",
        "ZNAME",
        "ZVAL",
        "TTYPE",
        "TFORM",
        "TDIM",
        "TSCAL",
        "TZERO",
        "TNULL",
        "TUNIT",
        "TDISP",
    )

    out: Dict[str, Any] = {}
    for key, value in dict(header).items():
        key_str = str(key)
        key_upper = key_str.upper()
        if key_upper in skip_exact or any(
            key_upper.startswith(prefix) for prefix in skip_prefix
        ):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="ignore")
        out[key_str] = value
    return out


def _sanitize_table_header_for_write(
    header: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Drop FITS structural keywords before delegating table writes to CFITSIO."""
    skip_keys = {
        "SIMPLE",
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "EXTEND",
        "THEAP",
        "DATASUM",
        "CHECKSUM",
    }
    out: Dict[str, Any] = {}
    for key, value in dict(header or {}).items():
        key_upper = str(key).upper()
        if key_upper in skip_keys or key_upper.startswith("NAXIS"):
            continue
        out[str(key)] = value
    return out


def _write_hdus_with_optional_compression(
    path: str, hdus: List[Any], compress: Union[bool, str] = False
) -> None:
    """Rewrite HDUs, optionally using CFITSIO compressed-image writer."""
    algorithm = _resolve_compression_algorithm(compress)
    if algorithm is None:
        write(path, HDUList(hdus), overwrite=True)
        return

    import torchsky.cpp as cpp

    class _TableWriteProxy:
        def __init__(self, raw_data, header):
            self._raw_data = raw_data
            self.header = header

    payload = []
    for idx, hdu in enumerate(hdus):
        if isinstance(hdu, TableHDURef):
            hdu = hdu.materialize(device="cpu")

        if isinstance(hdu, TableHDU):
            raw_data = dict(getattr(hdu, "_raw_data", {}))
            raw_data = _normalize_cpp_table_data(raw_data)
            payload.append(
                _TableWriteProxy(raw_data, _sanitize_table_header_for_write(hdu.header))
            )
            continue

        if hasattr(hdu, "_raw_data") and hasattr(hdu, "header"):
            raw_data = dict(getattr(hdu, "_raw_data", {}))
            raw_data = _normalize_cpp_table_data(raw_data)
            payload.append(
                _TableWriteProxy(raw_data, _sanitize_table_header_for_write(hdu.header))
            )
            continue

        if not isinstance(hdu, TensorHDU):
            raise ValueError(
                f"Unsupported HDU type for rewrite at index {idx}: {type(hdu).__name__}"
            )

        # A compressed FITS file uses an empty primary HDU followed by compressed
        # image extensions; skip this placeholder to avoid duplicating it.
        naxis_value = getattr(hdu, "header", {}).get("NAXIS", -1)
        try:
            naxis = int(naxis_value)
        except Exception:
            naxis = -1
        if idx == 0 and naxis == 0:
            xtension = (
                str(getattr(hdu, "header", {}).get("XTENSION", "")).strip().upper()
            )
            if not xtension:
                continue

        hdu_dict: Dict[str, Any] = {"data": hdu.to_tensor("cpu")}
        header = getattr(hdu, "header", None)
        if header:
            hdu_dict["header"] = _sanitize_header_for_compressed_write(header)
        payload.append(hdu_dict)

    _invalidate_path_caches(path)
    cpp.write_fits_file_compressed_images(path, payload, True, algorithm)


def insert_hdu(
    path: str,
    data: Any,
    index: int = 1,
    header: Optional[Dict[str, Any]] = None,
    compress: Union[bool, str] = False,
) -> None:
    """Insert a new HDU into an existing FITS file."""
    if not isinstance(index, int):
        raise TypeError("index must be an integer HDU position")

    if isinstance(data, TableHDU) or isinstance(data, TensorHDU):
        new_hdu = data
        if header is not None:
            new_hdu.header = Header(header)
    elif isinstance(data, dict) and "data" not in data:
        new_hdu = TableHDU(data, header=Header(header or {}))
    elif isinstance(data, Tensor):
        new_hdu = TensorHDU(data=data, header=Header(header or {}))
    else:
        raise ValueError(f"Unsupported HDU data type: {type(data)}")

    hdus = _detach_hdus_for_rewrite(path)

    if index < 0 or index > len(hdus):
        raise IndexError(f"index {index} out of range for {len(hdus)} HDUs")
    hdus.insert(index, new_hdu)
    _write_hdus_with_optional_compression(path, hdus, compress=compress)


def replace_hdu(
    path: str,
    hdu: Union[int, str],
    data: Any,
    header: Optional[Dict[str, Any]] = None,
    compress: Union[bool, str] = False,
) -> None:
    """Replace an HDU by index or EXTNAME."""

    preserve_header = header is None and not isinstance(data, (TableHDU, TensorHDU))

    if isinstance(data, TableHDU) or isinstance(data, TensorHDU):
        new_hdu = data
        if header is not None:
            new_hdu.header = Header(header)
    elif isinstance(data, dict) and "data" not in data:
        new_hdu = TableHDU(data, header=Header(header or {}))
    elif isinstance(data, Tensor):
        new_hdu = TensorHDU(data=data, header=Header(header or {}))
    else:
        raise ValueError(f"Unsupported HDU data type: {type(data)}")

    hdus = _detach_hdus_for_rewrite(path)

    if isinstance(hdu, int):
        if hdu < 0 or hdu >= len(hdus):
            raise IndexError(f"hdu index {hdu} out of range for {len(hdus)} HDUs")
        target = hdu
    elif isinstance(hdu, str):
        target = None
        for idx, item in enumerate(hdus):
            if item.header.get("EXTNAME") == hdu:
                target = idx
                break
        if target is None:
            raise KeyError(f"HDU '{hdu}' not found")
    else:
        raise TypeError("hdu must be an int index or EXTNAME string")

    if preserve_header:
        # Keep the original header (e.g. EXTNAME/WCS) unless the caller overrides it.
        old_header = getattr(hdus[target], "header", None)
        if old_header is not None:
            if isinstance(new_hdu, TensorHDU):
                new_hdu._header = old_header
            else:
                try:
                    new_hdu.header = old_header
                except Exception:
                    pass

    hdus[target] = new_hdu
    _write_hdus_with_optional_compression(path, hdus, compress=compress)


def delete_hdu(
    path: str,
    hdu: Union[int, str],
    compress: Union[bool, str] = False,
) -> None:
    """Delete an HDU by index or EXTNAME."""
    hdus = _detach_hdus_for_rewrite(path)

    if isinstance(hdu, int):
        if hdu < 0 or hdu >= len(hdus):
            raise IndexError(f"hdu index {hdu} out of range for {len(hdus)} HDUs")
        target = hdu
    elif isinstance(hdu, str):
        target = None
        for idx, item in enumerate(hdus):
            if item.header.get("EXTNAME") == hdu:
                target = idx
                break
        if target is None:
            raise KeyError(f"HDU '{hdu}' not found")
    else:
        raise TypeError("hdu must be an int index or EXTNAME string")

    del hdus[target]
    _write_hdus_with_optional_compression(path, hdus, compress=compress)


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
    ChecksumVerifier.write_hdu_checksums(path, hdu)


def verify_checksums(path: str, hdu: int = 0) -> Dict[str, Any]:
    """Verify DATASUM/CHECKSUM keywords for an HDU (CFITSIO).

    Returns a dict with `datastatus`, `hdustatus`, and `ok`. Status values:
    - 1: verification correct
    - 0: keyword not present / undefined
    - -1: verification not correct
    """
    return dict(ChecksumVerifier.verify_hdu_checksums(path, hdu))


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
