"""Compatibility facade for the public :mod:`torchfits.table` API.

FITS table implementation and dataframe interchange belong to torchfits.
Torchsky keeps this thin import path for one migration release.
"""

from torchfits.table import (
    TABLE_BACKENDS,
    append_rows,
    clear_cache,
    dataset,
    delete_rows,
    drop_columns,
    duckdb_query,
    insert_column,
    insert_rows,
    read,
    reader,
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
