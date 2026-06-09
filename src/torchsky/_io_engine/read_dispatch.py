"""Private read dispatch helpers for root FITS I/O."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from ..hdu import Header
from .caches import cache_stats, file_cache, path_signature, set_cached_hdu_type


def read_fallback_image(
    *,
    cpp_module: Any,
    file_handle: Any,
    path: str,
    hdu_num: Any,
    device: str,
    mmap: bool | str,
    fp16: bool,
    bf16: bool,
    cache_capacity: int,
    fast_header: bool,
    return_header: bool,
    cache_key: Any,
    use_cache: bool,
    resolve_image_mmap: Callable[[str, int, bool | str, int], bool],
    read_header: Callable[[Any, int, bool], Any],
) -> Any:
    """Read an image HDU in the generic fallback path."""
    effective_mmap = resolve_image_mmap(path, hdu_num, mmap, cache_capacity)
    data = cpp_module.read_full(file_handle, hdu_num, effective_mmap)

    if fp16:
        data = data.to(torch.float16)
    elif bf16:
        data = data.to(torch.bfloat16)

    if device != "cpu":
        data = data.to(device)

    header = None
    if return_header:
        header_data = read_header(file_handle, hdu_num, fast_header)
        header = Header(header_data)

    if use_cache and cache_key is not None:
        file_cache[cache_key] = (
            data.cpu() if device != "cpu" else data,
            header,
            path_signature(path),
        )
        while len(file_cache) > cache_capacity:
            file_cache.popitem(last=False)
        cache_stats["cache_size"] = len(file_cache)

    if isinstance(hdu_num, int):
        set_cached_hdu_type(path, hdu_num, "IMAGE")
    return (data, header) if return_header else data


def read_fallback_table(
    *,
    cpp_module: Any,
    file_handle: Any,
    path: str,
    hdu_num: Any,
    device: str,
    mmap: bool | str,
    cache_capacity: int,
    fast_header: bool,
    return_header: bool,
    cache_key: Any,
    use_cache: bool,
    columns: Any,
    start_row: int,
    num_rows: int,
    header_data: Any,
    header: Any,
    read_header: Callable[[Any, int, bool], Any],
) -> Any:
    """Read a table HDU in the generic fallback path."""
    if (return_header or isinstance(hdu_num, str)) and header_data is None:
        header_data = read_header(file_handle, hdu_num, fast_header)
        header = Header(header_data)

    col_list = columns if columns else []
    table_result = None
    table_mmap = mmap if isinstance(mmap, bool) else True
    if table_mmap:
        try:
            if start_row > 1 or num_rows != -1:
                if hasattr(cpp_module, "read_fits_table_rows"):
                    table_result = cpp_module.read_fits_table_rows(
                        path, hdu_num, col_list, start_row, num_rows, True
                    )
                else:
                    table_result = cpp_module.read_fits_table(
                        path, hdu_num, col_list, True
                    )
            else:
                table_result = cpp_module.read_fits_table(
                    path, hdu_num, col_list, True
                )
        except Exception:
            table_result = None

    if table_result is None:
        if columns is None and start_row == 1 and num_rows == -1:
            table_result = cpp_module.read_fits_table_from_handle(file_handle, hdu_num)
        elif hasattr(cpp_module, "read_fits_table_rows_from_handle"):
            table_result = cpp_module.read_fits_table_rows_from_handle(
                file_handle, hdu_num, col_list, start_row, num_rows
            )
        elif start_row > 1 or num_rows != -1:
            if hasattr(cpp_module, "read_fits_table_rows"):
                table_result = cpp_module.read_fits_table_rows(
                    path, hdu_num, col_list, start_row, num_rows, False
                )
            else:
                table_result = cpp_module.read_fits_table(
                    path, hdu_num, col_list, False
                )
        else:
            table_result = cpp_module.read_fits_table(path, hdu_num, col_list, False)

    table_data = table_result

    if (start_row > 1 or num_rows != -1) and not hasattr(
        cpp_module, "read_fits_table_rows"
    ):
        for key, value in table_data.items():
            if isinstance(value, torch.Tensor):
                end_row = start_row + num_rows - 1 if num_rows != -1 else len(value)
                table_data[key] = value[start_row - 1 : end_row]

    if use_cache and cache_key is not None:
        file_cache[cache_key] = (table_data, header, path_signature(path))
        while len(file_cache) > cache_capacity:
            file_cache.popitem(last=False)
        cache_stats["cache_size"] = len(file_cache)

    if device != "cpu":
        new_data = {}
        for key, value in table_data.items():
            if isinstance(value, torch.Tensor):
                new_data[key] = value.to(device)
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        new_list.append(item.to(device))
                    else:
                        new_list.append(item)
                new_data[key] = new_list
            else:
                new_data[key] = value
        table_data = new_data

    if isinstance(hdu_num, int):
        set_cached_hdu_type(path, hdu_num, "BINARY_TABLE")
    return (table_data, header) if return_header else table_data


def read_batch_paths(
    *,
    cpp_module: Any,
    path: list[str] | tuple[str, ...],
    hdu: Any,
    device: str,
    mmap: bool | str,
    fp16: bool,
    bf16: bool,
    raw_scale: bool,
    columns: Any,
    start_row: int,
    num_rows: int,
    cache_capacity: int,
    handle_cache_capacity: int,
    fast_header: bool,
    return_header: bool,
    mode: str,
    policy: str,
    autodetect_hdu: Callable[[str, int], int],
    batch_to_device: Callable[[list[Tensor], str], list[Tensor]],
    read_func: Callable[..., Any],
) -> list[Any]:
    """Dispatch a list of FITS paths through batch C++ or recursive reads."""
    hdu_batch = hdu
    if hdu_batch is None or (
        isinstance(hdu_batch, str) and hdu_batch.strip().lower() == "auto"
    ):
        if not path:
            raise ValueError("Batch read requires a non-empty path list")
        hdu_batch = autodetect_hdu(path[0], handle_cache_capacity)
    if not isinstance(hdu_batch, int):
        raise ValueError("Batch read requires a single integer HDU")
    hdu = hdu_batch

    if mmap is not False:
        try:
            data_list = cpp_module.read_images_batch(list(path), hdu)
            if device != "cpu":
                data_list = batch_to_device(data_list, device)
            return data_list
        except Exception:
            pass

    data_list = []
    for item_path in path:
        data_list.append(
            read_func(
                item_path,
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


def read_batch_hdus(
    *,
    cpp_module: Any,
    path: str,
    hdu: list[int] | tuple[int, ...],
    device: str,
    mmap: bool | str,
    fp16: bool,
    bf16: bool,
    raw_scale: bool,
    scale_on_device: bool,
    columns: Any,
    start_row: int,
    num_rows: int,
    cache_capacity: int,
    handle_cache_capacity: int,
    fast_header: bool,
    return_header: bool,
    policy: str,
    batch_to_device: Callable[[list[Tensor], str], list[Tensor]],
    read_func: Callable[..., Any],
) -> Any:
    """Dispatch multiple HDUs from one FITS path."""
    if hasattr(cpp_module, "read_hdus_batch"):
        try:
            data = cpp_module.read_hdus_batch(path, list(hdu))
        except TypeError:
            effective_mmap = True if isinstance(mmap, str) else mmap
            data = cpp_module.read_hdus_batch(path, list(hdu), effective_mmap)
        if device != "cpu":
            data = batch_to_device(data, device)
        return data
    return [
        read_func(
            path,
            hdu=item_hdu,
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
        for item_hdu in hdu
    ]


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
