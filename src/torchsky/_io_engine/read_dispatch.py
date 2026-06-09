"""Private read dispatch helpers for root FITS I/O."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


def read_scaled_cpu_fast(
    cpp_module: Any, path: str, hdu: int = 0, mmap: bool = True
) -> Tensor:
    """Internal helper for the CPU scaled fast path."""
    if not hasattr(cpp_module, "read_full_raw_with_scale"):
        raise RuntimeError("Scaled fast path unavailable in this build")

    data, scaled, bscale, bzero = cpp_module.read_full_raw_with_scale(path, hdu, mmap)
    if scaled:
        data = data.to(dtype=torch.float32)
        if bscale != 1.0:
            data.mul_(bscale)
        if bzero != 0.0:
            data.add_(bzero)
    return data


def read_cpu_fast_path(
    *,
    cpp_module: Any,
    path: str,
    hdu: int,
    mmap: bool | str,
    cache_capacity: int,
    handle_cache_capacity: int,
    fp16: bool,
    bf16: bool,
    force_image: bool,
    resolve_image_mmap: Callable[[str, int, bool | str, int], bool],
    get_cached_handle: Callable[[str, int], tuple[Any, bool]],
    cache_stats: dict[str, int],
    read_exc_types: tuple[type[BaseException], ...],
    debug_scale: bool,
    logger: Any,
) -> tuple[Tensor | None, bool]:
    """Try the CPU image fast path; return (data, fallback_required)."""
    try:
        effective_mmap = resolve_image_mmap(path, hdu, mmap, cache_capacity)
        if handle_cache_capacity > 0:
            if hasattr(cpp_module, "read_full_cached"):
                data = cpp_module.read_full_cached(path, hdu, effective_mmap)
            else:
                file_handle, _cached = get_cached_handle(path, handle_cache_capacity)
                data = cpp_module.read_full(file_handle, hdu, effective_mmap)
        elif cache_capacity == 0 and hasattr(cpp_module, "read_full_nocache"):
            data = cpp_module.read_full_nocache(path, hdu, effective_mmap)
        else:
            file_handle = cpp_module.open_fits_file(path, "r")
            try:
                data = cpp_module.read_full(file_handle, hdu, effective_mmap)
            finally:
                try:
                    file_handle.close()
                except Exception:
                    pass

        if fp16:
            data = data.to(torch.float16)
        elif bf16:
            data = data.to(torch.bfloat16)

        try:
            cache_stats["total_requests"] += 1
            cache_stats["misses"] += 1
        except Exception:
            pass
        return data, False
    except read_exc_types as exc:
        if debug_scale or logger.isEnabledFor(10):
            logger.debug(
                "read: CPU image fast-path fallback for %r hdu=%s: %s",
                path,
                hdu,
                exc,
                exc_info=True,
            )
        if force_image:
            raise
        return None, True


def read_generic_fast_path(
    *,
    cpp_module: Any,
    path: str,
    hdu: int,
    device: str,
    mmap: bool | str,
    cache_capacity: int,
    fp16: bool,
    bf16: bool,
    raw_scale: bool,
    scale_on_device: bool,
    force_image: bool,
    debug_scale: bool,
    cold_nocache: bool,
    resolve_image_mmap: Callable[[str, int, bool | str, int], bool],
    read_exc_types: tuple[type[BaseException], ...],
    logger: Any,
) -> Tensor | None:
    """Try the generic image fast path; return None when fallback is required."""
    try:
        effective_mmap = resolve_image_mmap(path, hdu, mmap, cache_capacity)
        if scale_on_device and not raw_scale:
            if hasattr(cpp_module, "read_full_raw_with_scale"):
                if debug_scale:
                    print("TORCHSKY_DEBUG_SCALE: fast_path_scaled")
                data, scaled, bscale, bzero = cpp_module.read_full_raw_with_scale(
                    path, hdu, effective_mmap
                )
                if scaled:
                    data = data.to(device=device, dtype=torch.float32)
                    if bscale != 1.0:
                        data.mul_(bscale)
                    if bzero != 0.0:
                        data.add_(bzero)
                else:
                    data = data.to(device)
            else:
                data = cpp_module.read_full(path, hdu, effective_mmap)
                data = data.to(device)
        elif raw_scale:
            if debug_scale:
                print("TORCHSKY_DEBUG_SCALE: raw_scale")
            if not effective_mmap and hasattr(cpp_module, "read_full_unmapped_raw"):
                data = cpp_module.read_full_unmapped_raw(path, hdu)
            else:
                data = cpp_module.read_full_raw(path, hdu, effective_mmap)
        else:
            if debug_scale:
                print("TORCHSKY_DEBUG_SCALE: unscaled")
            if not effective_mmap and hasattr(cpp_module, "read_full_unmapped"):
                data = cpp_module.read_full_unmapped(path, hdu)
            else:
                if (
                    cold_nocache
                    and cache_capacity == 0
                    and hasattr(cpp_module, "read_full_nocache")
                ):
                    data = cpp_module.read_full_nocache(path, hdu, effective_mmap)
                elif cache_capacity == 0 and hasattr(cpp_module, "read_full_nocache"):
                    data = cpp_module.read_full_nocache(path, hdu, effective_mmap)
                else:
                    data = cpp_module.read_full(path, hdu, effective_mmap)

        if fp16:
            data = data.to(torch.float16)
        elif bf16:
            data = data.to(torch.bfloat16)

        if device != "cpu" and data.device.type == "cpu":
            data = data.to(device)

        return data
    except ValueError:
        raise
    except read_exc_types as exc:
        if force_image:
            raise
        if logger.isEnabledFor(10):
            logger.debug(
                "read: generic image fast-path fallback for %r hdu=%s: %s",
                path,
                hdu,
                exc,
                exc_info=True,
            )
        return None
