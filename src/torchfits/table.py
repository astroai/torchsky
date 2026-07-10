_handle_cache = {}
_reader_cache = {}


def _not_supported(*args, **kwargs):
    raise NotImplementedError("Local torchfits shim does not support table I/O.")


def _can_use_mmap_row_path_for_full_read(*args, **kwargs):
    return False


def _can_use_torch_table_path_for_full_read(*args, **kwargs):
    return False


def _close_cpp_handle(*args, **kwargs):
    return None


def _column_name_index_map(*args, **kwargs):
    return {}


def _close_all_cached_handles(*args, **kwargs):
    _handle_cache.clear()
    _reader_cache.clear()
    return None


append_rows = _not_supported
dataset = _not_supported
delete_rows = _not_supported
drop_columns = _not_supported
duckdb_query = _not_supported
insert_column = _not_supported
insert_rows = _not_supported
read = _not_supported
reader = _not_supported
rename_columns = _not_supported
replace_column = _not_supported
scan = _not_supported
scan_torch = _not_supported
scanner = _not_supported
schema = _not_supported
to_duckdb = _not_supported
to_pandas = _not_supported
to_polars = _not_supported
to_polars_lazy = _not_supported
update_rows = _not_supported
write = _not_supported
write_parquet = _not_supported
