# `torchsky.table`

`torchsky.table` is the table-oriented surface for FITS binary tables, Arrow
scans, predicate pushdown, file-backed mutation, and dataframe interop.

## What This Namespace Is For

Use `torchsky.table` when the primary object is a table HDU or an Arrow-style
scan. Use root `torchsky.read(...)` when you want auto-detection across image and
table HDUs.

## Core Concepts

- **Projection:** request only the columns you need.
- **Predicate pushdown:** `where=` filters rows before data reaches Python.
- **Streaming:** scan large tables in chunks.
- **Root streaming:** `torchsky.stream_table(...)` and
  `torchsky.read_large_table(...)` are convenience wrappers for package-level
  workflows; `torchsky.table.scan(...)` and `torchsky.table.reader(...)` are the
  table-namespace scan/reader APIs.
- **Interop:** convert to PyArrow, Polars, DuckDB, Pandas, or tensors when those
  optional dependencies are installed.
- **Mutation:** file-backed row and column mutation is explicit and validates
  unsupported columns before touching data.

## Recipes

### Read A Filtered Table

```python
import torchsky

table = torchsky.table.read(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20 AND DEC > 0",
)
```

### Stream Row Chunks

```python
for batch in torchsky.table.scan("catalog.fits", hdu=1, batch_size=100_000):
    process(batch["RA"], batch["DEC"])
```

### Use Polars Or DuckDB

```python
import polars as pl

lazy = torchsky.table.to_polars_lazy("catalog.fits", hdu=1)
bright = lazy.filter(pl.col("MAG_G") < 20).collect()

rel = torchsky.table.to_duckdb("catalog.fits", hdu=1, relation_name="cat")
```

## API Surface

Read/write:

- `torchsky.table.read(...)`
- `torchsky.table.scan(...)`
- `torchsky.table.reader(...)`
- `torchsky.table.write(...)`

Mutation:

- `torchsky.table.append_rows(...)`
- `torchsky.table.update_rows(...)`
- `torchsky.table.insert_rows(...)`
- `torchsky.table.delete_rows(...)`
- `torchsky.table.insert_column(...)`
- `torchsky.table.replace_column(...)`
- `torchsky.table.rename_columns(...)`
- `torchsky.table.drop_columns(...)`

Interop:

- `to_polars_lazy(...)`
- `to_duckdb(...)`
- `duckdb_query(...)`
- `scanner(...)`
- root helpers `torchsky.to_arrow(...)`, `torchsky.to_pandas(...)`,
  `torchsky.to_polars(...)`

## Predicate Pushdown

Supported operators include `=`, `!=`, `<`, `>`, `<=`, `>=`, `AND`, `OR`,
`NOT`, `IN (...)`, `NOT IN (...)`, `BETWEEN ... AND ...`, `IS NULL`, and
`IS NOT NULL`.

## Examples

- [Table read/write example](../examples/example_table.py)
- [Astropy interop example](../examples/example_table_interop.py)
- [Polars example](../examples/example_polars.py)
- [Table recipes](../examples/example_table_recipes.py)

## Validation And Performance

Table behavior is covered by table tests, Arrow/Polars interop tests, and
`benchmarks/bench_fitstable_io.py`, `benchmarks/bench_arrow_tables.py`, and
`benchmarks/bench_pipeline_table_sphere.py`.
