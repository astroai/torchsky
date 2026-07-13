"""Table I/O delegated through torchfits' public table namespace.

The latest published torchfits wheel is 0.7.x and predates a few cache and
full-read heuristics.  Those optional helpers have conservative local
fallbacks so the table surface remains usable until the 0.8 wheel is
published.  Core reads/scans always continue to come from torchfits.
"""

from torchfits.table import (
    TABLE_BACKENDS,
    append_rows,
    dataset,
    delete_rows,
    drop_columns,
    duckdb_query,
    insert_column,
    insert_rows,
    reader,
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
import torchfits.table as _fits_table

try:
    from torchfits.table import (
        can_use_mmap_row_path_for_full_read as _can_use_mmap_row_path_for_full_read,
        can_use_torch_table_path_for_full_read as _can_use_torch_table_path_for_full_read,
        clear_cache,
        column_name_index_map as _column_name_index_map,
        close_table_handle as _close_cpp_handle,
        handle_cache as _handle_cache,
        reader_cache as _reader_cache,
    )
except ImportError:
    _fits_clear_cache = getattr(_fits_table, "clear_cache", None)
    _handle_cache: dict[str, object] = {}
    _reader_cache: dict[tuple[str, int], object] = {}

    def _close_cpp_handle(handle: object) -> None:
        close = getattr(handle, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def clear_cache() -> None:
        """Clear torchsky compatibility caches and any published cache."""
        handles = list(_handle_cache.values())
        _handle_cache.clear()
        _reader_cache.clear()
        for handle in handles:
            _close_cpp_handle(handle)
        if callable(_fits_clear_cache):
            _fits_clear_cache()

    def _column_name_index_map(header: object) -> dict[int, str]:
        """Build the stable FITS-column index map absent from torchfits 0.7."""
        values = getattr(header, "get", lambda *_args: None)
        try:
            count = int(values("TFIELDS", 0) or 0)
        except (TypeError, ValueError):
            count = 0
        result: dict[int, str] = {}
        if count:
            for index in range(1, count + 1):
                name = values(f"TTYPE{index}")
                if name not in (None, ""):
                    result[index] = str(name).strip()
            return result
        items = getattr(header, "items", lambda: ())()
        for key, value in items:
            text = str(key).upper()
            if text.startswith("TTYPE"):
                try:
                    index = int(text[5:])
                except ValueError:
                    continue
                if value not in (None, ""):
                    result[index] = str(value).strip()
        return result

    def _can_use_mmap_row_path_for_full_read(*_args: object, **_kwargs: object) -> bool:
        return False

    def _can_use_torch_table_path_for_full_read(
        *_args: object, **_kwargs: object
    ) -> bool:
        return False


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
