"""Table I/O delegated through torchfits' public table namespace."""

from torchfits.table import (
    TABLE_BACKENDS,
    append_rows,
    can_use_mmap_row_path_for_full_read as _can_use_mmap_row_path_for_full_read,  # noqa: F401
    can_use_torch_table_path_for_full_read as _can_use_torch_table_path_for_full_read,  # noqa: F401
    clear_cache,
    column_name_index_map as _column_name_index_map,  # noqa: F401
    close_table_handle as _close_cpp_handle,  # noqa: F401
    dataset,
    delete_rows,
    drop_columns,
    duckdb_query,
    handle_cache as _handle_cache,  # noqa: F401
    insert_column,
    insert_rows,
    reader,
    reader_cache as _reader_cache,  # noqa: F401
    read,
    rename_columns,
    replace_column,
    scan,
    scan_torch,
    scanner,
    schema,
    to_duckdb,
    to_pandas,
    to_polars,
    to_polars_lazy,
    update_rows,
    write,
    write_parquet,
)


def _close_all_cached_handles() -> None:
    """Compatibility alias for torchsky's legacy cache test hooks."""

    clear_cache()


__all__ = [
    "TABLE_BACKENDS",
    "append_rows",
    "clear_cache",
    "dataset",
    "delete_rows",
    "drop_columns",
    "duckdb_query",
    "insert_column",
    "insert_rows",
    "read",
    "reader",
    "rename_columns",
    "replace_column",
    "scan",
    "scan_torch",
    "scanner",
    "schema",
    "to_duckdb",
    "to_pandas",
    "to_polars",
    "to_polars_lazy",
    "update_rows",
    "write",
    "write_parquet",
]
