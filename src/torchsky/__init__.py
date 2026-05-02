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
)
from collections import OrderedDict
from dataclasses import dataclass

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
from .surfaces import (
    APISurface,
    API_SURFACES,
    SurfaceTier,
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


@dataclass
class ReadOptions:
    path: str
    hdu: Union[int, str, None]
    device: str
    mmap: Union[bool, str]
    fp16: bool
    bf16: bool
    raw_scale: bool
    scale_on_device: bool
    columns: Optional[List[str]]
    start_row: int
    num_rows: int
    cache_capacity: int
    handle_cache_capacity: int
    fast_header: bool
    return_header: bool
    force_image: bool = False
    force_table: bool = False
    hdu_type_hint: Optional[str] = None
    debug_scale: bool = False



_log = _stdlib_logging.getLogger(__name__)

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

# Simple cache tracking for tests
_cache_stats = {"total_requests": 0, "hits": 0, "misses": 0, "cache_size": 0}
_file_cache = OrderedDict()
_file_handle_cache = OrderedDict()
_file_handle_sig_cache = OrderedDict()
_image_meta_cache = OrderedDict()
_hdu_type_cache = OrderedDict()
_cold_nommap_cache = OrderedDict()
_auto_mmap_cache = OrderedDict()
_auto_hdu_cache = OrderedDict()


def _batch_to_device(
    tensors: list[torch.Tensor], device: str | torch.device
) -> list[torch.Tensor]:
    if not tensors:
        return []
    if len(tensors) == 1:
        return [tensors[0].to(device)]
    try:
        return list(torch.stack(tensors).to(device).unbind(0))
    except Exception:
        return [t.to(device) for t in tensors]


def _invalidate_path_caches(path: str) -> None:
    """Invalidate Python-side caches/handles for a path that is being modified."""
    global \
        _file_cache, \
        _file_handle_cache, \
        _file_handle_sig_cache, \
        _image_meta_cache, \
        _hdu_type_cache, \
        _cold_nommap_cache, \
        _auto_mmap_cache, \
        _auto_hdu_cache

    stale_data_keys = [
        k
        for k in list(_file_cache.keys())
        if isinstance(k, tuple) and k and k[0] == path
    ]
    for key in stale_data_keys:
        _file_cache.pop(key, None)
    handle = _file_handle_cache.pop(path, None)
    _file_handle_sig_cache.pop(path, None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass

    for key in [k for k in _image_meta_cache.keys() if k[0] == path]:
        _image_meta_cache.pop(key, None)
    for key in [k for k in _hdu_type_cache.keys() if k[0] == path]:
        _hdu_type_cache.pop(key, None)
    for key in [k for k in _cold_nommap_cache.keys() if k[0] == path]:
        _cold_nommap_cache.pop(key, None)
    for key in [k for k in _auto_mmap_cache.keys() if k[0] == path]:
        _auto_mmap_cache.pop(key, None)
    _auto_hdu_cache.pop(path, None)
    for key in [
        k for k in _auto_hdu_cache.keys() if isinstance(k, tuple) and k and k[0] == path
    ]:
        _auto_hdu_cache.pop(key, None)

    # Keep Arrow table reader/handle caches coherent across write/append/update paths.
    try:
        if hasattr(table, "_invalidate_caches_for_path"):
            table._invalidate_caches_for_path(path)
    except Exception:
        pass


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
    "SurfaceTier",
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


def _read_header_fast(file_handle, hdu_index: int, fast_header: bool = True):
    """Read header using fast bulk parsing or fallback to slow method."""
    import torchsky.cpp as cpp

    if fast_header:
        try:
            # Try fast bulk header reading
            header_string = cpp.read_header_string(file_handle, hdu_index)
            if header_string:
                return fast_parse_header(header_string)
        except (AttributeError, RuntimeError, IOError):
            # Fall back to slow method if fast parsing fails
            pass

    # Fall back to keyword-by-keyword reading
    return cpp.read_header(file_handle, hdu_index)


def _find_first_hdu(
    path: str,
    handle_cache_capacity: int = 16,
) -> Optional[int]:
    """Find first payload HDU, preferring image/compressed-image over table."""
    import torchsky.cpp as cpp

    file_handle, cached = _get_cached_handle(path, handle_cache_capacity)
    first_table_hdu: Optional[int] = None
    try:
        num_hdus = cpp.get_num_hdus(file_handle)
        for i in range(num_hdus):
            hdu_type = _get_cached_hdu_type(path, i)
            if hdu_type is None:
                try:
                    hdu_type = cpp.get_hdu_type(file_handle, i)
                    _set_cached_hdu_type(path, i, hdu_type)
                except Exception:
                    hdu_type = None
            if hdu_type == "IMAGE":
                try:
                    shape = file_handle.get_shape(i)
                except Exception:
                    shape = []
                if shape and all(int(dim) > 0 for dim in shape):
                    return i
                continue

            if hdu_type in {"ASCII_TABLE", "BINARY_TABLE"}:
                try:
                    hdr = _read_header_fast(file_handle, i, fast_header=True)
                except Exception:
                    hdr = {}
                zimage = _header_truthy(hdr.get("ZIMAGE"))
                has_compression_keys = any(
                    k in hdr for k in ("ZCMPTYPE", "ZBITPIX", "ZNAXIS", "ZTILE1")
                )
                if zimage or has_compression_keys:
                    return i
                if first_table_hdu is None:
                    first_table_hdu = i
    finally:
        if not cached:
            try:
                file_handle.close()
            except Exception:
                pass

    return first_table_hdu


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


def _path_signature(path: str):
    try:
        st = os.stat(path)
    except Exception:
        return None
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
    return (int(st.st_size), int(mtime_ns), int(st.st_ino))


def _get_cached_handle(path: str, handle_cache_capacity: int):
    import torchsky.cpp as cpp

    if handle_cache_capacity <= 0:
        return cpp.open_fits_file(path, "r"), False

    # Keep a small Python-level LRU of open CFITSIO handles. This avoids open/close
    # overhead on repeated reads and is intentionally very cheap per call.
    cur_sig = _path_signature(path)
    handle = _file_handle_cache.get(path)
    if handle is not None:
        prev_sig = _file_handle_sig_cache.get(path)
        if cur_sig is not None and prev_sig is not None and prev_sig != cur_sig:
            # File changed on disk (size/mtime/inode): drop stale handle.
            try:
                handle.close()
            except Exception:
                pass
            _file_handle_cache.pop(path, None)
            _file_handle_sig_cache.pop(path, None)
            handle = None
        else:
            _file_handle_cache.move_to_end(path)
            _file_handle_sig_cache.move_to_end(path)

    if handle is None:
        handle = cpp.open_fits_file(path, "r")
        _file_handle_cache[path] = handle
        _file_handle_sig_cache[path] = cur_sig

    while len(_file_handle_cache) > handle_cache_capacity:
        old_path, old_handle = _file_handle_cache.popitem(last=False)
        _file_handle_sig_cache.pop(old_path, None)
        try:
            old_handle.close()
        except Exception:
            pass

    return handle, True


def _get_image_meta(path: str, hdu: int):
    import torchsky.cpp as cpp

    sig = (path, hdu)

    cached = _image_meta_cache.get(sig)
    if cached is not None:
        return cached

    try:
        header = Header(cpp.read_header_dict(path, hdu))
        bitpix = int(header.get("BITPIX", 0))
        naxis = int(header.get("NAXIS", 0))
        try:
            bscale = float(header.get("BSCALE", 1.0))
        except Exception:
            bscale = 1.0
        try:
            bzero = float(header.get("BZERO", 0.0))
        except Exception:
            bzero = 0.0
        dims = []
        for i in range(1, naxis + 1):
            key = f"NAXIS{i}"
            if key in header:
                try:
                    dims.append(int(header.get(key)))
                except Exception:
                    break
        zimage = header.get("ZIMAGE", False)
        if isinstance(zimage, str):
            zimage = zimage.strip().upper() in {"T", "TRUE", "1"}
        xtension = str(header.get("XTENSION", "")).strip().upper()
        has_compression_keys = any(
            k in header for k in ("ZCMPTYPE", "ZBITPIX", "ZNAXIS", "ZTILE1")
        )
        is_compressed = bool(zimage) or (
            xtension == "BINTABLE" and has_compression_keys
        )
        meta = (bitpix, naxis, tuple(dims), bscale, bzero, is_compressed)
    except Exception:
        meta = None

    _image_meta_cache[sig] = meta
    # Keep a bounded cache to avoid unbounded growth on many files.
    while len(_image_meta_cache) > 256:
        _image_meta_cache.popitem(last=False)
    return meta


def _get_image_meta_from_handle(file_handle, path: str, hdu: int):
    """Fetch image metadata using an already-open FITS handle (avoids open/close)."""
    sig = (path, hdu)
    cached = _image_meta_cache.get(sig)
    if cached is not None:
        return cached

    try:
        header_data = _read_header_fast(file_handle, hdu, fast_header=True)
        bitpix = int(header_data.get("BITPIX", 0))
        naxis = int(header_data.get("NAXIS", 0))
        try:
            bscale = float(header_data.get("BSCALE", 1.0))
        except Exception:
            bscale = 1.0
        try:
            bzero = float(header_data.get("BZERO", 0.0))
        except Exception:
            bzero = 0.0
        dims = []
        for i in range(1, naxis + 1):
            key = f"NAXIS{i}"
            if key in header_data:
                try:
                    dims.append(int(header_data.get(key)))
                except Exception:
                    break
        zimage = header_data.get("ZIMAGE", False)
        if isinstance(zimage, str):
            zimage = zimage.strip().upper() in {"T", "TRUE", "1"}
        xtension = str(header_data.get("XTENSION", "")).strip().upper()
        has_compression_keys = any(
            k in header_data for k in ("ZCMPTYPE", "ZBITPIX", "ZNAXIS", "ZTILE1")
        )
        is_compressed = bool(zimage) or (
            xtension == "BINTABLE" and has_compression_keys
        )
        meta = (bitpix, naxis, tuple(dims), bscale, bzero, is_compressed)
    except Exception:
        meta = None

    _image_meta_cache[sig] = meta
    while len(_image_meta_cache) > 256:
        _image_meta_cache.popitem(last=False)
    return meta


def _get_cached_hdu_type(path: str, hdu: int) -> Optional[str]:
    sig = (path, hdu)
    cached = _hdu_type_cache.get(sig)
    if cached is not None:
        _hdu_type_cache.move_to_end(sig)
    return cached


def _set_cached_hdu_type(path: str, hdu: int, hdu_type: Optional[str]) -> None:
    if not hdu_type:
        return
    sig = (path, hdu)
    _hdu_type_cache[sig] = hdu_type
    _hdu_type_cache.move_to_end(sig)
    while len(_hdu_type_cache) > 512:
        _hdu_type_cache.popitem(last=False)


def _header_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().upper() in {"T", "TRUE", "1", "YES", "Y"}
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


def _autodetect_hdu(path: str, handle_cache_capacity: int = 16) -> int:
    """Return the first HDU with payload, preferring image/compressed-image HDUs."""
    sig = _path_signature(path)
    cache_key = (path, "payload")
    cached = _auto_hdu_cache.get(cache_key)
    if cached is not None:
        cached_sig, cached_hdu = cached
        if sig is None or cached_sig is None or cached_sig == sig:
            _auto_hdu_cache.move_to_end(cache_key)
            return int(cached_hdu)
        _auto_hdu_cache.pop(cache_key, None)

    resolved = _find_first_hdu(path, handle_cache_capacity=handle_cache_capacity)
    if resolved is None:
        return 0

    _auto_hdu_cache[cache_key] = (sig, int(resolved))
    _auto_hdu_cache.move_to_end(cache_key)
    while len(_auto_hdu_cache) > 512:
        _auto_hdu_cache.popitem(last=False)
    return int(resolved)


def _should_use_cold_nommap(
    path: str, hdu: int, cache_capacity: int, mmap: bool
) -> bool:
    # Keep cache_capacity in the signature for backward compatibility with
    # existing call sites; nommap selection now depends on file/HDU traits.
    _ = cache_capacity
    if not mmap:
        return False
    if _COLD_NOMMAP:
        return True

    cached = _cold_nommap_cache.get((path, hdu))
    if cached is not None:
        return bool(cached)

    # Auto-select non-mmap for common uncompressed integer/float32 cold reads where
    # mmap has consistently higher latency than direct reads.
    # Keep this scoped to larger files so tiny-image latency is unaffected.
    try:
        file_size = os.path.getsize(path)
        if file_size < (1 << 20):  # 1 MiB
            _cold_nommap_cache[(path, hdu)] = False
            return False
    except Exception:
        _cold_nommap_cache[(path, hdu)] = False
        return False

    meta = _image_meta_cache.get((path, hdu))
    if meta is None:
        meta = _get_image_meta(path, hdu)
    if not meta:
        _cold_nommap_cache[(path, hdu)] = False
        return False

    try:
        bitpix = int(meta[0])
    except Exception:
        _cold_nommap_cache[(path, hdu)] = False
        return False

    is_compressed = False
    if len(meta) >= 6:
        try:
            is_compressed = bool(meta[5])
        except Exception:
            is_compressed = False
    if is_compressed:
        _cold_nommap_cache[(path, hdu)] = False
        return False

    # TARGET: common medium/large science images where direct reads are often
    # lower-latency than mmap+byteswap in real workloads.
    if bitpix in (16, 32, -32):
        _cold_nommap_cache[(path, hdu)] = True
        _cold_nommap_cache.move_to_end((path, hdu))
        while len(_cold_nommap_cache) > 512:
            _cold_nommap_cache.popitem(last=False)
        return True

    _cold_nommap_cache[(path, hdu)] = False
    _cold_nommap_cache.move_to_end((path, hdu))
    while len(_cold_nommap_cache) > 512:
        _cold_nommap_cache.popitem(last=False)
    return False


def _resolve_image_mmap(
    path: str,
    hdu: int,
    mmap: Union[bool, str],
    cache_capacity: int,
) -> bool:
    """Resolve mmap mode for image reads.

    - mmap=True/False are explicit user overrides.
    - mmap='auto' uses heuristics:
      * compressed HDUs -> mmap disabled by default
      * otherwise use cold-read nommap heuristic
    """
    if isinstance(mmap, bool):
        return mmap

    if isinstance(mmap, str):
        mode = mmap.strip().lower()
        if mode != "auto":
            raise ValueError("mmap must be bool or 'auto'")

        sig = (path, hdu)
        cached = _auto_mmap_cache.get(sig)
        if cached is not None:
            _auto_mmap_cache.move_to_end(sig)
            return bool(cached)

        meta = _image_meta_cache.get((path, hdu))
        if meta is None:
            meta = _get_image_meta(path, hdu)

        if meta is not None and len(meta) >= 6:
            try:
                if bool(meta[5]):  # compressed image HDU
                    _auto_mmap_cache[sig] = False
                    _auto_mmap_cache.move_to_end(sig)
                    while len(_auto_mmap_cache) > 512:
                        _auto_mmap_cache.popitem(last=False)
                    return False
            except Exception:
                pass

        resolved = not _should_use_cold_nommap(path, hdu, cache_capacity, True)
        _auto_mmap_cache[sig] = bool(resolved)
        _auto_mmap_cache.move_to_end(sig)
        while len(_auto_mmap_cache) > 512:
            _auto_mmap_cache.popitem(last=False)
        return bool(resolved)

    raise ValueError("mmap must be bool or 'auto'")


def _read_scaled_cpu_fast(path: str, hdu: int = 0, mmap: bool = True) -> Tensor:
    """Internal helper for the CPU scaled fast path."""
    import torchsky.cpp as cpp

    if not hasattr(cpp, "read_full_raw_with_scale"):
        raise RuntimeError("Scaled fast path unavailable in this build")

    data, scaled, bscale, bzero = cpp.read_full_raw_with_scale(path, hdu, mmap)
    if scaled:
        data = data.to(dtype=torch.float32)
        if bscale != 1.0:
            data.mul_(bscale)
        if bzero != 0.0:
            data.add_(bzero)
    return data


def _validate_read_image_args(
    path: str, hdu: int, mmap: bool, handle_cache: bool, device: str
) -> None:
    """Validate arguments for low-level read_image."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if not isinstance(hdu, int) or hdu < 0:
        raise ValueError("hdu must be a non-negative integer")
    if not isinstance(mmap, bool):
        raise ValueError("read_image requires explicit mmap=True/False")
    if not isinstance(handle_cache, bool):
        raise ValueError("handle_cache must be bool")
    if device not in ["cpu", "cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")


def _dispatch_read_image_cpp(
    cpp: Any, path: str, hdu: int, mmap: bool, handle_cache: bool, raw_scale: bool
) -> Tensor:
    """Dispatch the correct C++ function for low-level image reading."""
    if raw_scale:
        if not mmap and hasattr(cpp, "read_full_unmapped_raw"):
            return cpp.read_full_unmapped_raw(path, hdu)
        elif hasattr(cpp, "read_full_raw"):
            return cpp.read_full_raw(path, hdu, mmap)
        else:
            return cpp.read_full(path, hdu, mmap)
    else:
        if handle_cache and hasattr(cpp, "read_full_cached"):
            return cpp.read_full_cached(path, hdu, mmap)
        elif not mmap and hasattr(cpp, "read_full_unmapped"):
            return cpp.read_full_unmapped(path, hdu)
        elif hasattr(cpp, "read_full_nocache"):
            return cpp.read_full_nocache(path, hdu, mmap)
        else:
            return cpp.read_full(path, hdu, mmap)


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
    import torchsky.cpp as cpp

    _validate_read_image_args(path, hdu, mmap, handle_cache, device)

    data = _dispatch_read_image_cpp(cpp, path, hdu, mmap, handle_cache, raw_scale)

    if fp16:
        data = data.to(torch.float16)
    elif bf16:
        data = data.to(torch.bfloat16)

    if device != "cpu" and data.device.type == "cpu":
        data = data.to(device)

    if return_header:
        try:
            return data, Header(cpp.read_header_dict(path, hdu))
        except Exception:
            return data, get_header(path, hdu)
    return data


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
    if not isinstance(hdu, int) or hdu < 0:
        raise ValueError("hdu must be a non-negative integer")

    out = read(
        path=path,
        hdu=hdu,
        mode="table",
        device=device,
        mmap=mmap,
        columns=columns,
        start_row=start_row,
        num_rows=num_rows,
        cache_capacity=cache_capacity,
        handle_cache_capacity=handle_cache_capacity,
        fast_header=fast_header,
        return_header=return_header,
        policy=policy,
    )
    data = out[0] if return_header else out
    if isinstance(data, torch.Tensor):
        raise ValueError(
            f"HDU {hdu!r} is an image HDU. Use read_image(...) or read(...)."
        )
    return out


@overload
def read(
    path: str,
    hdu: Union[int, str, None] = None,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    fp16: bool = False,
    bf16: bool = False,
    raw_scale: bool = False,
    scale_on_device: bool = True,
    use_cache: Optional[bool] = None,
    columns: Optional[List[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    cache_capacity: int = 10,
    handle_cache_capacity: int = 16,
    fast_header: bool = True,
    return_header: Literal[False] = False,
    mode: str = "auto",
    policy: str = "default",
) -> Union[Tensor, Dict[str, Any]]: ...


@overload
def read(
    path: str,
    hdu: Union[int, str, None] = None,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    fp16: bool = False,
    bf16: bool = False,
    raw_scale: bool = False,
    scale_on_device: bool = True,
    use_cache: Optional[bool] = None,
    columns: Optional[List[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    cache_capacity: int = 10,
    handle_cache_capacity: int = 16,
    fast_header: bool = True,
    return_header: Literal[True] = ...,
    mode: str = "auto",
    policy: str = "default",
) -> Tuple[Union[Tensor, Dict[str, Any]], Header]: ...


@overload
def read(
    path: Union[List[str], Tuple[str, ...]],
    hdu: Union[int, str, None] = None,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    fp16: bool = False,
    bf16: bool = False,
    raw_scale: bool = False,
    scale_on_device: bool = True,
    use_cache: Optional[bool] = None,
    columns: Optional[List[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    cache_capacity: int = 10,
    handle_cache_capacity: int = 16,
    fast_header: bool = True,
    return_header: Literal[False] = False,
    mode: str = "auto",
    policy: str = "default",
) -> List[Union[Tensor, Dict[str, Any]]]: ...


@overload
def read(
    path: Union[List[str], Tuple[str, ...]],
    hdu: Union[int, str, None] = None,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    fp16: bool = False,
    bf16: bool = False,
    raw_scale: bool = False,
    scale_on_device: bool = True,
    use_cache: Optional[bool] = None,
    columns: Optional[List[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    cache_capacity: int = 10,
    handle_cache_capacity: int = 16,
    fast_header: bool = True,
    return_header: Literal[True] = ...,
    mode: str = "auto",
    policy: str = "default",
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
    import torchsky.cpp as cpp

    hdu_batch = hdu
    if hdu_batch is None or (
        isinstance(hdu_batch, str) and hdu_batch.strip().lower() == "auto"
    ):
        if not path:
            raise ValueError("Batch read requires a non-empty path list")
        hdu_batch = _autodetect_hdu(path[0], handle_cache_capacity)
    if not isinstance(hdu_batch, int):
        raise ValueError("Batch read requires a single integer HDU")
    hdu = hdu_batch
    # C++ batch API does not expose mmap mode controls. Keep the default fast
    # path for mmap='auto' and explicit mmap=True.
    if mmap is not False:
        try:
            data_list = cpp.read_images_batch(list(path), hdu)
            if device != "cpu":
                data_list = _batch_to_device(data_list, device)
            return data_list
        except Exception:
            pass

    data_list = []
    for p in path:
        data_list.append(
            read(
                p,
                hdu=hdu,
                mode=mode,
                policy=policy,
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
            )
        )
    return data_list


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
    import torchsky.cpp as cpp

    if hasattr(cpp, "read_hdus_batch"):
        try:
            data = cpp.read_hdus_batch(path, list(hdu))
        except TypeError:
            effective_mmap = True if isinstance(mmap, str) else mmap
            data = cpp.read_hdus_batch(path, list(hdu), effective_mmap)
        if device != "cpu":
            data = _batch_to_device(data, device)
        return data
    return [
        read(
            path,
            hdu=h,
            device=device,
            policy=policy,
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
        )
        for h in hdu
    ]


def _read_cpu_fast_path(opts: ReadOptions):
    import torchsky.cpp as cpp

    try:
        effective_mmap = _resolve_image_mmap(opts.path, opts.hdu, opts.mmap, opts.cache_capacity)
        # Prefer handle-cache fast path when enabled: cache_capacity only controls
        # data caching, while handle_cache_capacity controls open-handle reuse.
        if opts.handle_cache_capacity > 0:
            if hasattr(cpp, "read_full_cached"):
                # Keep the fast path fully inside C++ (shared handle/meta cache)
                # to avoid Python-side handle-cache overhead per read.
                data = cpp.read_full_cached(opts.path, opts.hdu, effective_mmap)
            else:
                file_handle, _cached = _get_cached_handle(opts.path, opts.handle_cache_capacity)
                data = cpp.read_full(file_handle, opts.hdu, effective_mmap)
        elif opts.cache_capacity == 0 and hasattr(cpp, "read_full_nocache"):
            # Explicit no-cache mode when handle-cache is disabled.
            data = cpp.read_full_nocache(opts.path, opts.hdu, effective_mmap)
        else:
            file_handle = cpp.open_fits_file(opts.path, "r")
            try:
                data = cpp.read_full(file_handle, opts.hdu, effective_mmap)
            finally:
                try:
                    file_handle.close()
                except Exception:
                    pass

        if opts.fp16:
            data = data.to(torch.float16)
        elif opts.bf16:
            data = data.to(torch.bfloat16)

        # Keep cache performance stats consistent even on fast-path returns.
        try:
            _cache_stats["total_requests"] += 1
            _cache_stats["misses"] += 1
        except Exception:
            pass
        return data, False
    except _READ_EXC_TYPES as exc:
        # Not an image HDU or reader mismatch; skip generic image fast path and fall through.
        if _DEBUG_SCALE or _log.isEnabledFor(_stdlib_logging.DEBUG):
            _log.debug(
                "read: CPU image fast-path fallback for %r hdu=%s: %s",
                opts.path,
                opts.hdu,
                exc,
                exc_info=True,
            )
        if opts.force_image:
            raise
        return None, True


def _read_generic_fast_path(opts: ReadOptions):
    import torchsky.cpp as cpp

    try:
        effective_mmap = _resolve_image_mmap(opts.path, opts.hdu, opts.mmap, opts.cache_capacity)
        if opts.scale_on_device and not opts.raw_scale:
            if hasattr(cpp, "read_full_raw_with_scale"):
                if opts.debug_scale:
                    print("TORCHSKY_DEBUG_SCALE: fast_path_scaled")
                data, scaled, bscale, bzero = cpp.read_full_raw_with_scale(
                    opts.path, opts.hdu, effective_mmap
                )
                if scaled:
                    data = data.to(device=opts.device, dtype=torch.float32)
                    if bscale != 1.0:
                        data.mul_(bscale)
                    if bzero != 0.0:
                        data.add_(bzero)
                else:
                    data = data.to(opts.device)
            else:
                data = cpp.read_full(opts.path, opts.hdu, effective_mmap)
                data = data.to(opts.device)
        elif opts.raw_scale:
            if opts.debug_scale:
                print("TORCHSKY_DEBUG_SCALE: raw_scale")
            # Keep raw-scale path on the unified read_full_raw entry point.
            # This avoids a crash-prone unmapped-only variant on scaled HDUs.
            if not effective_mmap and hasattr(cpp, "read_full_unmapped_raw"):
                data = cpp.read_full_unmapped_raw(opts.path, opts.hdu)
            else:
                data = cpp.read_full_raw(opts.path, opts.hdu, effective_mmap)
        else:
            if opts.debug_scale:
                print("TORCHSKY_DEBUG_SCALE: unscaled")
            if not effective_mmap and hasattr(cpp, "read_full_unmapped"):
                data = cpp.read_full_unmapped(opts.path, opts.hdu)
            else:
                if (
                    _COLD_NOCACHE
                    and opts.cache_capacity == 0
                    and hasattr(cpp, "read_full_nocache")
                ):
                    data = cpp.read_full_nocache(opts.path, opts.hdu, effective_mmap)
                else:
                    if opts.cache_capacity == 0 and hasattr(cpp, "read_full_nocache"):
                        # Explicitly disable caching when cache_capacity==0.
                        data = cpp.read_full_nocache(opts.path, opts.hdu, effective_mmap)
                    else:
                        data = cpp.read_full(opts.path, opts.hdu, effective_mmap)

        if opts.fp16:
            data = data.to(torch.float16)
        elif opts.bf16:
            data = data.to(torch.bfloat16)

        if opts.device != "cpu" and data.device.type == "cpu":
            data = data.to(opts.device)

        try:
            _cache_stats["total_requests"] += 1
            _cache_stats["misses"] += 1
        except Exception:
            pass

        return data, False
    except ValueError:
        raise
    except _READ_EXC_TYPES as exc:
        if opts.force_image:
            raise
        if _log.isEnabledFor(_stdlib_logging.DEBUG):
            _log.debug(
                "read: generic image fast-path fallback for %r hdu=%s: %s",
                opts.path,
                opts.hdu,
                exc,
                exc_info=True,
            )
        return None, True


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
    global _cache_stats
    _cache_stats["total_requests"] += 1
    use_cache = cache_capacity > 0

    cache_key = None
    if use_cache:
        try:
            cache_key = (
                path,
                hdu,
                device,
                fp16,
                bf16,
                tuple(columns) if columns else None,
                start_row,
                num_rows,
                return_header,
            )
        except TypeError:
            cache_key = None

        if cache_key is not None:
            if cache_key in _file_cache:
                cached_entry = _file_cache.pop(cache_key)
                if isinstance(cached_entry, tuple) and len(cached_entry) == 3:
                    cached_data, cached_header, cached_sig = cached_entry
                else:
                    cached_data, cached_header = cached_entry
                    cached_sig = None

                cur_sig = _path_signature(path)
                stale_cache_entry = (
                    cached_sig is not None
                    and cur_sig is not None
                    and cached_sig != cur_sig
                )
                if stale_cache_entry:
                    _invalidate_path_caches(path)
                    _cache_stats["misses"] += 1
                else:
                    _cache_stats["hits"] += 1
                    _file_cache[cache_key] = (cached_data, cached_header, cached_sig)

                    if device != "cpu":
                        if isinstance(cached_data, torch.Tensor):
                            cached_data = cached_data.to(device)
                        elif isinstance(cached_data, dict):
                            new_data = {}
                            for k, v in cached_data.items():
                                if isinstance(v, torch.Tensor):
                                    new_data[k] = v.to(device)
                                else:
                                    new_data[k] = v
                            cached_data = new_data

                    return (
                        True,
                        (
                            (cached_data, cached_header)
                            if return_header
                            else cached_data
                        ),
                        cache_key,
                    )
            else:
                _cache_stats["misses"] += 1
        else:
            _cache_stats["misses"] += 1
    else:
        _cache_stats["misses"] += 1

    return False, None, cache_key


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
    import torchsky.cpp as cpp

    effective_mmap = _resolve_image_mmap(path, hdu_num, mmap, cache_capacity)
    data = cpp.read_full(file_handle, hdu_num, effective_mmap)

    if fp16:
        data = data.to(torch.float16)
    elif bf16:
        data = data.to(torch.bfloat16)

    if device != "cpu":
        data = data.to(device)

    header = None
    if return_header:
        header_data = _read_header_fast(file_handle, hdu_num, fast_header)
        header = Header(header_data)

    if use_cache and cache_key is not None:
        path_sig = _path_signature(path)
        _file_cache[cache_key] = (
            data.cpu() if device != "cpu" else data,
            header,
            path_sig,
        )
        while len(_file_cache) > cache_capacity:
            _file_cache.popitem(last=False)
        _cache_stats["cache_size"] = len(_file_cache)

    if isinstance(hdu_num, int):
        _set_cached_hdu_type(path, hdu_num, "IMAGE")
    return (data, header) if return_header else data


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
    import torchsky.cpp as cpp

    if (return_header or isinstance(hdu_num, str)) and header_data is None:
        header_data = _read_header_fast(file_handle, hdu_num, fast_header)
        header = Header(header_data)

    col_list = columns if columns else []
    table_result = None
    table_mmap = mmap if isinstance(mmap, bool) else True
    if table_mmap:
        try:
            if start_row > 1 or num_rows != -1:
                if hasattr(cpp, "read_fits_table_rows"):
                    table_result = cpp.read_fits_table_rows(
                        path, hdu_num, col_list, start_row, num_rows, True
                    )
                else:
                    table_result = cpp.read_fits_table(path, hdu_num, col_list, True)
            else:
                table_result = cpp.read_fits_table(path, hdu_num, col_list, True)
        except Exception:
            table_result = None

    if table_result is None:
        if columns is None and start_row == 1 and num_rows == -1:
            table_result = cpp.read_fits_table_from_handle(file_handle, hdu_num)
        elif hasattr(cpp, "read_fits_table_rows_from_handle"):
            table_result = cpp.read_fits_table_rows_from_handle(
                file_handle, hdu_num, col_list, start_row, num_rows
            )
        elif start_row > 1 or num_rows != -1:
            if hasattr(cpp, "read_fits_table_rows"):
                table_result = cpp.read_fits_table_rows(
                    path, hdu_num, col_list, start_row, num_rows, False
                )
            else:
                table_result = cpp.read_fits_table(path, hdu_num, col_list, False)
        else:
            table_result = cpp.read_fits_table(path, hdu_num, col_list, False)

    table_data = table_result

    if (start_row > 1 or num_rows != -1) and not hasattr(cpp, "read_fits_table_rows"):
        for k, v in table_data.items():
            if isinstance(v, torch.Tensor):
                end_row = start_row + num_rows - 1 if num_rows != -1 else len(v)
                table_data[k] = v[start_row - 1 : end_row]

    if use_cache and cache_key is not None:
        path_sig = _path_signature(path)
        _file_cache[cache_key] = (table_data, header, path_sig)
        while len(_file_cache) > cache_capacity:
            _file_cache.popitem(last=False)
        _cache_stats["cache_size"] = len(_file_cache)

    if device != "cpu":
        new_data = {}
        for k, v in table_data.items():
            if isinstance(v, torch.Tensor):
                new_data[k] = v.to(device)
            elif isinstance(v, list):
                new_list = []
                for item in v:
                    if isinstance(item, torch.Tensor):
                        new_list.append(item.to(device))
                    else:
                        new_list.append(item)
                new_data[k] = new_list
            else:
                new_data[k] = v
        table_data = new_data

    if isinstance(hdu_num, int):
        _set_cached_hdu_type(path, hdu_num, "BINARY_TABLE")
    return (table_data, header) if return_header else table_data


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

    global _cache_stats

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


def read(
    path: Union[str, List[str], Tuple[str, ...]],
    hdu: Union[int, str, List[int], Tuple[int, ...], None] = None,
    device: str = "cpu",
    mmap: Union[bool, str] = "auto",
    fp16: bool = False,
    bf16: bool = False,
    raw_scale: bool = False,
    scale_on_device: bool = True,
    use_cache: Optional[bool] = None,
    columns: Optional[List[str]] = None,
    start_row: int = 1,
    num_rows: int = -1,
    cache_capacity: int = 10,
    handle_cache_capacity: int = 16,
    fast_header: bool = True,
    return_header: bool = False,
    mode: str = "auto",
    policy: str = "default",
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
    global _cache_stats, _file_cache

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

    opts = ReadOptions(
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
        force_image=force_image,
        force_table=force_table,
        hdu_type_hint=hdu_type_hint,
        debug_scale=debug_scale,
    )

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
        res, fallback = _read_cpu_fast_path(opts)
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
        res, fallback = _read_generic_fast_path(opts)
        if not fallback:
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
    try:
        file_handle, cached = _get_cached_handle(path, handle_cache_capacity)
        try:
            return file_handle.read_subset(hdu, x1, y1, x2, y2)
        finally:
            if not cached:
                try:
                    file_handle.close()
                except Exception:
                    pass
    except Exception as e:
        raise RuntimeError(f"Failed to read subset from '{path}': {e}") from e


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
    import torchsky.cpp as cpp

    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(hdus, (list, tuple)) or len(hdus) == 0:
        raise ValueError("hdus must be a non-empty list/tuple of HDU indices or names")
    if device not in ["cpu", "cuda", "mps"] and not str(device).startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")
    if not isinstance(mmap, bool):
        raise ValueError("mmap must be a bool for read_hdus")

    resolved_hdus: List[int] = []
    for hdu in hdus:
        if isinstance(hdu, int):
            if hdu < 0:
                raise ValueError("HDU index must be non-negative")
            resolved_hdus.append(int(hdu))
            continue
        if isinstance(hdu, str):
            if hasattr(cpp, "resolve_hdu_name_cached"):
                resolved_hdus.append(int(cpp.resolve_hdu_name_cached(path, hdu)))
                continue
            raise ValueError("named HDUs require resolve_hdu_name_cached support")
        raise ValueError("each item in hdus must be an int or str")

    data = cpp.read_hdus_batch(path, resolved_hdus, mmap)
    if device != "cpu":
        data = _batch_to_device(data, device)

    if not return_header:
        return data

    headers = [Header(cpp.read_header_dict(path, hdu_num)) for hdu_num in resolved_hdus]
    return data, headers


class SubsetReader:
    """Persistent subset reader for repeated cutouts on one image HDU."""

    def __init__(self, path: str, hdu: int = 0, device: str = "cpu"):
        import torchsky.cpp as cpp

        if not isinstance(path, str):
            raise ValueError("path must be a string")
        if not isinstance(hdu, int) or hdu < 0:
            raise ValueError("hdu must be a non-negative integer")
        if device not in ["cpu", "cuda", "mps"] and not str(device).startswith("cuda:"):
            raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")

        self._reader = cpp.SubsetReader(path, int(hdu))
        self._device = device

    @property
    def hdu(self) -> int:
        return int(self._reader.hdu)

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self._reader.height), int(self._reader.width)

    def read_subset(self, x1: int, y1: int, x2: int, y2: int) -> Tensor:
        out = self._reader.read(int(x1), int(y1), int(x2), int(y2))
        if self._device != "cpu":
            out = out.to(self._device)
        return out

    def close(self) -> None:
        self._reader.close()

    def __call__(self, x1: int, y1: int, x2: int, y2: int) -> Tensor:
        return self.read_subset(x1, y1, x2, y2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def open_subset_reader(path: str, hdu: int = 0, device: str = "cpu") -> SubsetReader:
    """Open a persistent cutout reader for repeated subsets on one HDU."""
    return SubsetReader(path=path, hdu=hdu, device=device)


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
    if not file_paths:
        return []

    if device not in ["cpu", "cuda", "mps"] and not device.startswith("cuda:"):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")

    try:
        if isinstance(hdu, int) and hdu >= 0:
            import torchsky.cpp as cpp

            tensors = cpp.read_images_batch(list(file_paths), hdu)
            if device != "cpu":
                tensors = _batch_to_device(tensors, device)
            return tensors
    except _READ_EXC_TYPES as exc:
        if strict:
            raise
        _log.debug("read_batch: C++ batch path failed, falling back per file: %s", exc)

    results = []
    for path in file_paths:
        try:
            tensor = read(path, hdu=hdu, device=device, return_header=False)
            results.append(tensor)
        except _READ_EXC_TYPES as exc:
            if strict:
                raise
            _log.debug("read_batch: skipped %r: %s", path, exc, exc_info=True)
            continue
    return results


def get_batch_info(file_paths: List[str]) -> Dict[str, Any]:
    """Get information about a batch of FITS files.

    Args:
        file_paths: List of file paths

    Returns:
        Dictionary with batch statistics
    """
    import os

    valid_files = 0
    for path in file_paths:
        try:
            if os.path.exists(path):
                valid_files += 1
        except Exception:
            continue

    return {"num_files": len(file_paths), "valid_files": valid_files}


def get_cache_performance() -> Dict[str, Any]:
    """Get cache performance statistics.

    Returns:
        Dictionary with cache statistics
    """
    global _cache_stats
    total = _cache_stats["total_requests"]
    hits = _cache_stats["hits"]
    misses = _cache_stats["misses"]

    return {
        "cache_size": _cache_stats["cache_size"],
        "hit_rate": hits / total if total > 0 else 0.0,
        "miss_rate": misses / total if total > 0 else 0.0,
        "total_requests": total,
        "hits": hits,
        "misses": misses,
    }


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
    global \
        _cache_stats, \
        _file_cache, \
        _file_handle_cache, \
        _file_handle_sig_cache, \
        _image_meta_cache, \
        _hdu_type_cache, \
        _cold_nommap_cache, \
        _auto_mmap_cache, \
        _auto_hdu_cache

    if data:
        _file_cache.clear()

    if handles:
        for _, handle in list(_file_handle_cache.items()):
            try:
                handle.close()
            except Exception:
                pass
        _file_handle_cache.clear()
        _file_handle_sig_cache.clear()

    if meta:
        _image_meta_cache.clear()
    if hdu_types:
        _hdu_type_cache.clear()
        _auto_hdu_cache.clear()
    if meta:
        _cold_nommap_cache.clear()
        _auto_mmap_cache.clear()

    if stats:
        _cache_stats = {"total_requests": 0, "hits": 0, "misses": 0, "cache_size": 0}

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
    try:
        import os

        import torchsky.cpp as cpp

        if not os.path.exists(file_path):
            return {}

        if not streaming:
            if return_iterator:
                return iter([cpp.read_fits_table(file_path, hdu)])
            return cpp.read_fits_table(file_path, hdu)

        # Streaming-like chunked read (returns full dict, but limits peak memory)
        if not hasattr(cpp, "read_fits_table_rows"):
            return cpp.read_fits_table(file_path, hdu)

        try:
            header = get_header(file_path, hdu)
            total_rows = header.get("NAXIS2", 0)
            try:
                if isinstance(total_rows, str):
                    total_rows = int(float(total_rows))
                else:
                    total_rows = int(total_rows)
            except Exception:
                total_rows = 0
            if total_rows == 0:
                return cpp.read_fits_table(file_path, hdu)

            # Sample a small batch to estimate bytes per row
            sample_rows = min(256, total_rows)
            if hasattr(cpp, "read_fits_table_rows_from_handle"):
                file_handle = cpp.open_fits_file(file_path, "r")
                try:
                    if hasattr(cpp, "TableReader"):
                        reader = cpp.TableReader(file_handle, hdu)
                        sample = reader.read_rows([], 1, sample_rows)
                    else:
                        sample = cpp.read_fits_table_rows_from_handle(
                            file_handle, hdu, [], 1, sample_rows
                        )
                finally:
                    file_handle.close()
            else:
                sample = cpp.read_fits_table_rows(
                    file_path, hdu, [], 1, sample_rows, False
                )

            def _estimate_bytes(data: Dict[str, Any], rows: int) -> float:
                total = 0
                for v in data.values():
                    if isinstance(v, torch.Tensor):
                        total += v.numel() * v.element_size()
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, torch.Tensor):
                                total += item.numel() * item.element_size()
                return total / max(1, rows)

            bytes_per_row = _estimate_bytes(sample, sample_rows)
            if bytes_per_row <= 0:
                bytes_per_row = 1024.0  # Fallback guess

            max_bytes = max_memory_mb * 1024 * 1024
            rows_per_chunk = max(1, int(max_bytes / bytes_per_row))

            if return_iterator:
                return stream_table(
                    file_path,
                    hdu=hdu,
                    columns=None,
                    start_row=1,
                    num_rows=-1,
                    chunk_rows=rows_per_chunk,
                    mmap=False,
                )

            accum = {}
            start = 1
            if hasattr(cpp, "read_fits_table_rows_from_handle"):
                file_handle = cpp.open_fits_file(file_path, "r")
                try:
                    reader = None
                    if hasattr(cpp, "TableReader"):
                        reader = cpp.TableReader(file_handle, hdu)

                    while start <= total_rows:
                        remaining = total_rows - start + 1
                        num = min(rows_per_chunk, remaining)
                        if reader is not None:
                            chunk = reader.read_rows([], start, num)
                        else:
                            chunk = cpp.read_fits_table_rows_from_handle(
                                file_handle, hdu, [], start, num
                            )

                        for k, v in chunk.items():
                            if isinstance(v, torch.Tensor):
                                if k not in accum:
                                    out_shape = (total_rows,) + tuple(v.shape[1:])
                                    accum[k] = torch.empty(out_shape, dtype=v.dtype)
                                accum[k][start - 1 : start - 1 + v.shape[0]] = v
                            elif isinstance(v, list):
                                accum.setdefault(k, []).extend(v)
                            else:
                                accum.setdefault(k, []).append(v)

                        start += num
                finally:
                    file_handle.close()
            else:
                while start <= total_rows:
                    remaining = total_rows - start + 1
                    num = min(rows_per_chunk, remaining)
                    chunk = cpp.read_fits_table_rows(
                        file_path, hdu, [], start, num, False
                    )

                    for k, v in chunk.items():
                        if isinstance(v, torch.Tensor):
                            if k not in accum:
                                out_shape = (total_rows,) + tuple(v.shape[1:])
                                accum[k] = torch.empty(out_shape, dtype=v.dtype)
                            accum[k][start - 1 : start - 1 + v.shape[0]] = v
                        elif isinstance(v, list):
                            accum.setdefault(k, []).extend(v)
                        else:
                            accum.setdefault(k, []).append(v)

                    start += num

            return accum
        except Exception:
            return cpp.read_fits_table(file_path, hdu)

    except Exception:
        return {}


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
    import os

    import torchsky.cpp as cpp

    if not os.path.exists(file_path):
        return

    col_list = columns if columns else []

    if not hasattr(cpp, "read_fits_table_rows"):
        result = cpp.read_fits_table(file_path, hdu, col_list, mmap)
        yield result
        return

    header = get_header(file_path, hdu)
    total_rows = header.get("NAXIS2", 0)
    try:
        if isinstance(total_rows, str):
            total_rows = int(float(total_rows))
        else:
            total_rows = int(total_rows)
    except Exception:
        total_rows = 0
    if total_rows == 0:
        return

    if num_rows != -1:
        total_rows = min(total_rows, start_row + num_rows - 1)

    row = start_row
    emitted = 0
    if mmap and hasattr(cpp, "read_fits_table_rows"):
        while row <= total_rows:
            remaining = total_rows - row + 1
            size = min(chunk_rows, remaining)
            yield cpp.read_fits_table_rows(file_path, hdu, col_list, row, size, mmap)
            row += size
            emitted += 1
            if max_chunks is not None and emitted >= max_chunks:
                return
    elif hasattr(cpp, "read_fits_table_rows_from_handle"):
        file_handle = cpp.open_fits_file(file_path, "r")
        try:
            reader = None
            if hasattr(cpp, "TableReader"):
                reader = cpp.TableReader(file_handle, hdu)
            while row <= total_rows:
                remaining = total_rows - row + 1
                size = min(chunk_rows, remaining)
                if reader is not None:
                    yield reader.read_rows(col_list, row, size)
                else:
                    yield cpp.read_fits_table_rows_from_handle(
                        file_handle, hdu, col_list, row, size
                    )
                row += size
                emitted += 1
                if max_chunks is not None and emitted >= max_chunks:
                    return
        finally:
            file_handle.close()
    else:
        while row <= total_rows:
            remaining = total_rows - row + 1
            size = min(chunk_rows, remaining)
            yield cpp.read_fits_table_rows(file_path, hdu, col_list, row, size, mmap)
            row += size
            emitted += 1
            if max_chunks is not None and emitted >= max_chunks:
                return


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
    import os

    if mode == "r" and not os.path.exists(path):
        raise FileNotFoundError(f"FITS file not found: {path}")

    try:
        return HDUList.fromfile(path, mode)
    except PermissionError:
        raise PermissionError(f"Permission denied accessing file: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to open FITS file '{path}': {e}") from e


def get_header(path: str, hdu: Union[int, str, None] = None) -> Header:
    """Get the header of a FITS file.

    Args:
        path: Path to the FITS file.
        hdu: HDU index, name, or `None`/`"auto"` for the first payload HDU
            (same convention as :func:`read` and :func:`get_wcs`).

    Returns:
        Header object.
    """
    import torchsky.cpp as cpp

    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == "auto"):
        hdu = _autodetect_hdu(path, handle_cache_capacity=16)

    # Resolve HDU index if string
    if isinstance(hdu, str):
        if hasattr(cpp, "resolve_hdu_name_cached"):
            try:
                hdu = int(cpp.resolve_hdu_name_cached(path, hdu))
                return Header(cpp.read_header_dict(path, hdu))
            except Exception:
                pass

        for i in range(100):  # Fallback scan
            try:
                h_list = cpp.read_header_dict(path, i)
                if not h_list:
                    break
                h = Header(h_list)
                if h.get("EXTNAME") == hdu:
                    return h
            except Exception:
                break
        raise ValueError(f"HDU '{hdu}' not found")

    return Header(cpp.read_header_dict(path, hdu))


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
    wcs = WCS(get_header(path, hdu=hdu))
    if device is not None:
        wcs = wcs.to(torch.device(device))
    return wcs


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
