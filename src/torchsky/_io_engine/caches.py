"""Python-side cache state for root FITS I/O dispatch."""

from __future__ import annotations

import os
from collections import OrderedDict
from types import ModuleType
from typing import Any


_CACHE_STATS_DEFAULT = {
    "total_requests": 0,
    "hits": 0,
    "misses": 0,
    "cache_size": 0,
}

cache_stats: dict[str, int] = dict(_CACHE_STATS_DEFAULT)
file_cache = OrderedDict()
file_handle_cache = OrderedDict()
file_handle_sig_cache = OrderedDict()
image_meta_cache = OrderedDict()
hdu_type_cache = OrderedDict()
cold_nommap_cache = OrderedDict()
auto_mmap_cache = OrderedDict()
auto_hdu_cache = OrderedDict()


def path_signature(path: str) -> tuple[int, int, int] | None:
    """Return a compact file identity for stale-cache detection."""
    try:
        st = os.stat(path)
    except Exception:
        return None
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
    return (int(st.st_size), int(mtime_ns), int(st.st_ino))


def get_cached_handle(path: str, handle_cache_capacity: int) -> tuple[Any, bool]:
    """Return an open CFITSIO handle, reusing a small Python-side LRU when enabled."""
    import torchsky.cpp as cpp

    if handle_cache_capacity <= 0:
        return cpp.open_fits_file(path, "r"), False

    cur_sig = path_signature(path)
    handle = file_handle_cache.get(path)
    if handle is not None:
        prev_sig = file_handle_sig_cache.get(path)
        if cur_sig is not None and prev_sig is not None and prev_sig != cur_sig:
            try:
                handle.close()
            except Exception:
                pass
            file_handle_cache.pop(path, None)
            file_handle_sig_cache.pop(path, None)
            handle = None
        else:
            file_handle_cache.move_to_end(path)
            file_handle_sig_cache.move_to_end(path)

    if handle is None:
        handle = cpp.open_fits_file(path, "r")
        file_handle_cache[path] = handle
        file_handle_sig_cache[path] = cur_sig

    while len(file_handle_cache) > handle_cache_capacity:
        old_path, old_handle = file_handle_cache.popitem(last=False)
        file_handle_sig_cache.pop(old_path, None)
        try:
            old_handle.close()
        except Exception:
            pass

    return handle, True


def get_cached_hdu_type(path: str, hdu: int) -> str | None:
    """Return a cached HDU payload type for path/HDU dispatch, if known."""
    sig = (path, hdu)
    cached = hdu_type_cache.get(sig)
    if cached is not None:
        hdu_type_cache.move_to_end(sig)
    return cached


def set_cached_hdu_type(path: str, hdu: int, hdu_type: str | None) -> None:
    """Record an HDU payload type for path/HDU dispatch."""
    if not hdu_type:
        return
    sig = (path, hdu)
    hdu_type_cache[sig] = hdu_type
    hdu_type_cache.move_to_end(sig)
    while len(hdu_type_cache) > 512:
        hdu_type_cache.popitem(last=False)


def invalidate_path_caches(path: str, table_module: ModuleType | None = None) -> None:
    """Invalidate Python-side caches and open handles for one path."""
    stale_data_keys = [
        key
        for key in list(file_cache.keys())
        if isinstance(key, tuple) and key and key[0] == path
    ]
    for key in stale_data_keys:
        file_cache.pop(key, None)

    handle = file_handle_cache.pop(path, None)
    file_handle_sig_cache.pop(path, None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass

    for key in [key for key in image_meta_cache.keys() if key[0] == path]:
        image_meta_cache.pop(key, None)
    for key in [key for key in hdu_type_cache.keys() if key[0] == path]:
        hdu_type_cache.pop(key, None)
    for key in [key for key in cold_nommap_cache.keys() if key[0] == path]:
        cold_nommap_cache.pop(key, None)
    for key in [key for key in auto_mmap_cache.keys() if key[0] == path]:
        auto_mmap_cache.pop(key, None)

    auto_hdu_cache.pop(path, None)
    for key in [
        key
        for key in auto_hdu_cache.keys()
        if isinstance(key, tuple) and key and key[0] == path
    ]:
        auto_hdu_cache.pop(key, None)

    if table_module is not None:
        try:
            invalidate = getattr(table_module, "_invalidate_caches_for_path", None)
            if invalidate is not None:
                invalidate(path)
        except Exception:
            pass


def get_cache_performance() -> dict[str, Any]:
    """Return root FITS I/O cache performance statistics."""
    total = cache_stats["total_requests"]
    hits = cache_stats["hits"]
    misses = cache_stats["misses"]

    return {
        "cache_size": cache_stats["cache_size"],
        "hit_rate": hits / total if total > 0 else 0.0,
        "miss_rate": misses / total if total > 0 else 0.0,
        "total_requests": total,
        "hits": hits,
        "misses": misses,
    }


def reset_cache_stats() -> None:
    """Reset root FITS I/O cache counters in place."""
    cache_stats.clear()
    cache_stats.update(_CACHE_STATS_DEFAULT)


def clear_python_caches(
    *,
    data: bool = True,
    handles: bool = True,
    meta: bool = True,
    hdu_types: bool = True,
    stats: bool = True,
) -> None:
    """Clear Python-side root FITS I/O caches."""
    if data:
        file_cache.clear()

    if handles:
        for _, handle in list(file_handle_cache.items()):
            try:
                handle.close()
            except Exception:
                pass
        file_handle_cache.clear()
        file_handle_sig_cache.clear()

    if meta:
        image_meta_cache.clear()
        cold_nommap_cache.clear()
        auto_mmap_cache.clear()

    if hdu_types:
        hdu_type_cache.clear()
        auto_hdu_cache.clear()

    if stats:
        reset_cache_stats()
