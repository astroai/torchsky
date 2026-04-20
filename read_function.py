def read(path: Union[str, List[str], Tuple[str, ...]], hdu: Union[int, str, List[int], Tuple[int, ...], None]=None, device: str='cpu', mmap: Union[bool, str]='auto', fp16: bool=False, bf16: bool=False, raw_scale: bool=False, scale_on_device: bool=True, columns: Optional[List[str]]=None, start_row: int=1, num_rows: int=-1, cache_capacity: int=10, handle_cache_capacity: int=16, fast_header: bool=True, return_header: bool=False, mode: str='auto', policy: str='default') -> Any:
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
    import torchsky.cpp as cpp
    global _cache_stats, _file_cache
    if not path:
        raise ValueError('Path must be a non-empty string')
    if isinstance(path, (list, tuple)):
        hdu_batch = hdu
        if hdu_batch is None or (isinstance(hdu_batch, str) and hdu_batch.strip().lower() == 'auto'):
            if not path:
                raise ValueError('Batch read requires a non-empty path list')
            hdu_batch = _autodetect_hdu(path[0], handle_cache_capacity)
        if not isinstance(hdu_batch, int):
            raise ValueError('Batch read requires a single integer HDU')
        hdu = hdu_batch
        if mmap is True:
            try:
                data_list = cpp.read_images_batch(list(path), hdu)
                if device != 'cpu':
                    data_list = [t.to(device) for t in data_list]
                return data_list
            except Exception:
                pass
        data_list = []
        for p in path:
            data_list.append(read(p, hdu=hdu, mode=mode, policy=policy, device=device, mmap=mmap, fp16=fp16, bf16=bf16, raw_scale=raw_scale, columns=columns, start_row=start_row, num_rows=num_rows, cache_capacity=cache_capacity, handle_cache_capacity=handle_cache_capacity, fast_header=fast_header, return_header=return_header))
        return data_list
    if not isinstance(path, str):
        raise ValueError('Path must be a string or list of strings')
    if isinstance(mmap, str) and mmap.strip().lower() != 'auto':
        raise ValueError("mmap must be bool or 'auto'")
    if not isinstance(mmap, (bool, str)):
        raise ValueError("mmap must be bool or 'auto'")
    mode = str(mode).strip().lower()
    if mode not in {'auto', 'image', 'table'}:
        raise ValueError("mode must be 'auto', 'image', or 'table'")
    policy = str(policy).strip().lower()
    if policy not in {'default', 'smart'}:
        raise ValueError("policy must be 'default' or 'smart'")
    force_image = mode == 'image'
    force_table = mode == 'table'
    if force_image and (columns is not None or start_row != 1 or num_rows != -1):
        raise ValueError("mode='image' does not support table row/column options")
    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == 'auto'):
        hdu = _autodetect_hdu(path, handle_cache_capacity)
    hdu_type_hint = _get_cached_hdu_type(path, hdu) if isinstance(hdu, int) else None
    is_cached_table_hdu = force_table or hdu_type_hint in {'ASCII_TABLE', 'BINARY_TABLE'}
    skip_generic_image_fast_path = is_cached_table_hdu
    if isinstance(hdu, (list, tuple)) and (not return_header) and (columns is None) and (start_row == 1) and (num_rows == -1):
        if isinstance(mmap, bool) and hasattr(cpp, 'read_hdus_batch'):
            data = cpp.read_hdus_batch(path, list(hdu), mmap)
            if device != 'cpu':
                data = [t.to(device) for t in data]
            return data
        return [read(path, hdu=h, device=device, policy=policy, mmap=mmap, fp16=fp16, bf16=bf16, raw_scale=raw_scale, scale_on_device=scale_on_device, columns=columns, start_row=start_row, num_rows=num_rows, cache_capacity=cache_capacity, handle_cache_capacity=handle_cache_capacity, fast_header=fast_header, return_header=return_header) for h in hdu]
    debug_scale = _DEBUG_SCALE
    if scale_on_device and (not raw_scale) and (device == 'cpu') and (not return_header) and isinstance(hdu, int) and (columns is None) and (start_row == 1) and (num_rows == -1) and (not is_cached_table_hdu):
        try:
            effective_mmap = _resolve_image_mmap(path, hdu, mmap, cache_capacity)
            if handle_cache_capacity > 0:
                if hasattr(cpp, 'read_full_cached'):
                    data = cpp.read_full_cached(path, hdu, effective_mmap)
                else:
                    file_handle, _cached = _get_cached_handle(path, handle_cache_capacity)
                    data = cpp.read_full(file_handle, hdu, effective_mmap)
            elif cache_capacity == 0 and hasattr(cpp, 'read_full_nocache'):
                data = cpp.read_full_nocache(path, hdu, effective_mmap)
            else:
                file_handle = cpp.open_fits_file(path, 'r')
                try:
                    data = cpp.read_full(file_handle, hdu, effective_mmap)
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
                _cache_stats['total_requests'] += 1
                _cache_stats['misses'] += 1
            except Exception:
                pass
            return data
        except _READ_EXC_TYPES as exc:
            if _DEBUG_SCALE or _log.isEnabledFor(_stdlib_logging.DEBUG):
                _log.debug('read: CPU image fast-path fallback for %r hdu=%s: %s', path, hdu, exc, exc_info=True)
            if force_image:
                raise
            skip_generic_image_fast_path = True
    if not return_header and isinstance(hdu, int) and (columns is None) and (start_row == 1) and (num_rows == -1) and (not skip_generic_image_fast_path):
        try:
            effective_mmap = _resolve_image_mmap(path, hdu, mmap, cache_capacity)
            if scale_on_device and (not raw_scale):
                handle = None
                if handle_cache_capacity > 0:
                    try:
                        handle, _ = _get_cached_handle(path, handle_cache_capacity)
                    except _READ_EXC_TYPES:
                        handle = None
                if handle is not None:
                    meta = _get_image_meta_from_handle(handle, path, hdu)
                else:
                    meta = _image_meta_cache.get((path, hdu))
                    if meta is None:
                        meta = _get_image_meta(path, hdu)
                needs_scale = None
                if meta is not None and len(meta) >= 5:
                    try:
                        needs_scale = not (float(meta[3]) == 1.0 and float(meta[4]) == 0.0)
                    except Exception:
                        needs_scale = None
                if needs_scale is False:
                    if debug_scale:
                        print('TORCHSKY_DEBUG_SCALE: fast_path_unscaled_via_read_full')
                    if handle_cache_capacity > 0:
                        if handle is None:
                            handle, _ = _get_cached_handle(path, handle_cache_capacity)
                        data = cpp.read_full(handle, hdu, effective_mmap)
                    elif _COLD_NOCACHE and cache_capacity == 0 and hasattr(cpp, 'read_full_nocache'):
                        data = cpp.read_full_nocache(path, hdu, effective_mmap)
                    elif cache_capacity == 0 and hasattr(cpp, 'read_full_nocache'):
                        data = cpp.read_full_nocache(path, hdu, effective_mmap)
                    else:
                        data = cpp.read_full(path, hdu, effective_mmap)
                    data = data.to(device)
                elif hasattr(cpp, 'read_full_raw_with_scale'):
                    if debug_scale:
                        print('TORCHSKY_DEBUG_SCALE: fast_path_scaled')
                    data, scaled, bscale, bzero = cpp.read_full_raw_with_scale(path, hdu, effective_mmap)
                    if scaled:
                        data = data.to(device=device, dtype=torch.float32)
                        if bscale != 1.0:
                            data.mul_(bscale)
                        if bzero != 0.0:
                            data.add_(bzero)
                    else:
                        data = data.to(device)
            elif raw_scale:
                if debug_scale:
                    print('TORCHSKY_DEBUG_SCALE: raw_scale')
                data = cpp.read_full_raw(path, hdu, effective_mmap)
            else:
                if debug_scale:
                    print('TORCHSKY_DEBUG_SCALE: unscaled')
                if not effective_mmap and hasattr(cpp, 'read_full_unmapped'):
                    data = cpp.read_full_unmapped(path, hdu)
                elif _COLD_NOCACHE and cache_capacity == 0 and hasattr(cpp, 'read_full_nocache'):
                    data = cpp.read_full_nocache(path, hdu, effective_mmap)
                elif cache_capacity == 0 and hasattr(cpp, 'read_full_nocache'):
                    data = cpp.read_full_nocache(path, hdu, effective_mmap)
                else:
                    data = cpp.read_full(path, hdu, effective_mmap)
            if fp16:
                data = data.to(torch.float16)
            elif bf16:
                data = data.to(torch.bfloat16)
            if device != 'cpu' and data.device.type == 'cpu':
                data = data.to(device)
            return data
        except ValueError:
            raise
        except _READ_EXC_TYPES as exc:
            if force_image:
                raise
            if _log.isEnabledFor(_stdlib_logging.DEBUG):
                _log.debug('read: generic image fast-path fallback for %r hdu=%s: %s', path, hdu, exc, exc_info=True)
    _cache_stats['total_requests'] += 1
    use_cache = cache_capacity > 0
    cache_key = None
    if use_cache:
        try:
            cache_key = (path, hdu, device, fp16, bf16, tuple(columns) if columns else None, start_row, num_rows, return_header)
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
                stale_cache_entry = cached_sig is not None and cur_sig is not None and (cached_sig != cur_sig)
                if stale_cache_entry:
                    _invalidate_path_caches(path)
                    _cache_stats['misses'] += 1
                else:
                    _cache_stats['hits'] += 1
                    _file_cache[cache_key] = (cached_data, cached_header, cached_sig)
                    if device != 'cpu':
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
                    return (cached_data, cached_header) if return_header else cached_data
            else:
                _cache_stats['misses'] += 1
        else:
            _cache_stats['misses'] += 1
    else:
        _cache_stats['misses'] += 1
    if isinstance(hdu, int) and hdu < 0:
        raise ValueError('HDU index must be non-negative')
    if start_row < 1:
        raise ValueError('start_row must be >= 1 (FITS uses 1-based indexing)')
    if num_rows < -1 or num_rows == 0:
        raise ValueError('num_rows must be > 0 or -1 for all rows')
    if device not in ['cpu', 'cuda', 'mps'] and (not device.startswith('cuda:')):
        raise ValueError("device must be 'cpu', 'cuda', 'mps' or 'cuda:N'")
    try:
        file_handle, cached_handle = _get_cached_handle(path, handle_cache_capacity)
        try:
            if isinstance(hdu, str):
                hdu_num = None
                if hasattr(cpp, 'resolve_hdu_name_cached'):
                    try:
                        hdu_num = int(cpp.resolve_hdu_name_cached(path, hdu))
                    except Exception:
                        hdu_num = None
                if hdu_num is None:
                    num_hdus = cpp.get_num_hdus(file_handle)
                    for i in range(num_hdus):
                        try:
                            hdr = cpp.read_header(file_handle, i)
                            if hdr.get('EXTNAME') == hdu:
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
            is_table_hdu = force_table or hdu_type in {'ASCII_TABLE', 'BINARY_TABLE'}
            if force_image:
                is_table_hdu = False
            if not is_table_hdu:
                try:
                    effective_mmap = _resolve_image_mmap(path, hdu_num, mmap, cache_capacity)
                    data = cpp.read_full(file_handle, hdu_num, effective_mmap)
                    if fp16:
                        data = data.to(torch.float16)
                    elif bf16:
                        data = data.to(torch.bfloat16)
                    if device != 'cpu':
                        data = data.to(device)
                    if return_header:
                        header_data = _read_header_fast(file_handle, hdu_num, fast_header)
                        header = Header(header_data)
                    if use_cache and cache_key is not None:
                        path_sig = _path_signature(path)
                        _file_cache[cache_key] = (data.cpu() if device != 'cpu' else data, header, path_sig)
                        while len(_file_cache) > cache_capacity:
                            _file_cache.popitem(last=False)
                        _cache_stats['cache_size'] = len(_file_cache)
                    if isinstance(hdu_num, int):
                        _set_cached_hdu_type(path, hdu_num, 'IMAGE')
                    return (data, header) if return_header else data
                except (RuntimeError, TypeError):
                    if force_image:
                        raise
                    if isinstance(hdu_num, int):
                        try:
                            hdu_type = cpp.get_hdu_type(file_handle, hdu_num)
                            _set_cached_hdu_type(path, hdu_num, hdu_type)
                        except Exception:
                            hdu_type = None
                    is_table_hdu = force_table or hdu_type in {'ASCII_TABLE', 'BINARY_TABLE'}
                    if not is_table_hdu:
                        raise
            try:
                if (return_header or isinstance(hdu, str)) and header_data is None:
                    header_data = _read_header_fast(file_handle, hdu_num, fast_header)
                    header = Header(header_data)
                col_list = columns if columns else []
                table_result = None
                table_mmap = mmap if isinstance(mmap, bool) else True
                if table_mmap:
                    try:
                        if start_row > 1 or num_rows != -1:
                            if hasattr(cpp, 'read_fits_table_rows'):
                                table_result = cpp.read_fits_table_rows(path, hdu_num, col_list, start_row, num_rows, True)
                            else:
                                table_result = cpp.read_fits_table(path, hdu_num, col_list, True)
                        else:
                            table_result = cpp.read_fits_table(path, hdu_num, col_list, True)
                    except Exception:
                        table_result = None
                if table_result is None:
                    if columns is None and start_row == 1 and (num_rows == -1):
                        table_result = cpp.read_fits_table_from_handle(file_handle, hdu_num)
                    elif hasattr(cpp, 'read_fits_table_rows_from_handle'):
                        table_result = cpp.read_fits_table_rows_from_handle(file_handle, hdu_num, col_list, start_row, num_rows)
                    elif start_row > 1 or num_rows != -1:
                        if hasattr(cpp, 'read_fits_table_rows'):
                            table_result = cpp.read_fits_table_rows(path, hdu_num, col_list, start_row, num_rows, False)
                        else:
                            table_result = cpp.read_fits_table(path, hdu_num, col_list, False)
                    else:
                        table_result = cpp.read_fits_table(path, hdu_num, col_list, False)
                table_data = table_result
                if (start_row > 1 or num_rows != -1) and (not hasattr(cpp, 'read_fits_table_rows')):
                    for k, v in table_data.items():
                        if isinstance(v, torch.Tensor):
                            end_row = start_row + num_rows - 1 if num_rows != -1 else len(v)
                            table_data[k] = v[start_row - 1:end_row]
                if use_cache and cache_key is not None:
                    path_sig = _path_signature(path)
                    _file_cache[cache_key] = (table_data, header, path_sig)
                    while len(_file_cache) > cache_capacity:
                        _file_cache.popitem(last=False)
                    _cache_stats['cache_size'] = len(_file_cache)
                if device != 'cpu':
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
                    _set_cached_hdu_type(path, hdu_num, 'BINARY_TABLE')
                return (table_data, header) if return_header else table_data
            except Exception as e:
                raise RuntimeError(f'Failed to read table extension: {e}')
        finally:
            if not cached_handle:
                try:
                    file_handle.close()
                except Exception:
                    pass
    except Exception as e:
        raise RuntimeError(f"Failed to read FITS file '{path}': {e}") from e
