"""Table I/O — delegated to torchfits."""

# Public table APIs — re-exported from torchfits._table submodules
# (torchfits 0.6.0 no longer re-exports internals from torchfits.table).

from torchfits._table.read import (  # noqa: F401
    _can_use_mmap_row_path_for_full_read as _can_use_mmap_row_path_for_full_read,
    _can_use_torch_table_path_for_full_read as _can_use_torch_table_path_for_full_read,
    dataset as dataset,
    read as read,
    reader as reader,
    scan as scan,
    scan_torch as scan_torch,
    scanner as scanner,
    schema as schema,
)

from torchfits._table.write import (  # noqa: F401
    _column_name_index_map as _column_name_index_map,
    write as write,
)

from torchfits._table.cache import (  # noqa: F401
    _close_all_cached_handles as _close_all_cached_handles,
    _close_cpp_handle as _close_cpp_handle,
    _handle_cache as _handle_cache,
    _reader_cache as _reader_cache,
)

from torchfits._table.mutation import (  # noqa: F401
    append_rows as append_rows,
    delete_rows as delete_rows,
    drop_columns as drop_columns,
    insert_column as insert_column,
    insert_rows as insert_rows,
    rename_columns as rename_columns,
    replace_column as replace_column,
    update_rows as update_rows,
)

from torchfits._table.interop import (  # noqa: F401
    duckdb_query as duckdb_query,
    to_duckdb as to_duckdb,
    to_pandas as to_pandas,
    to_polars as to_polars,
    to_polars_lazy as to_polars_lazy,
    write_parquet as write_parquet,
)
