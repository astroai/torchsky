# `torchsky.io` And Root FITS I/O

`torchsky` keeps only the core FITS operations at the package root:
`read`, `write`, `open`, `get_header`, and `get_wcs`. `torchsky.io.fits` owns
the CFITSIO-backed FITS file operations; `torchsky.io` re-exports those helpers
alongside datasets, dataloaders, interop helpers, and table conveniences.

## What This Namespace Is For

Use this surface for FITS images, HDUs, headers, checksums, cutouts, streaming
table reads, and writes. Results are PyTorch tensors where possible, with
optional placement on CPU, CUDA, or MPS after data are read.

## Core Concepts

- **Unified read:** `torchsky.read(...)` auto-detects image or table HDUs.
- **Explicit read:** `read_image(...)`, `read_table(...)`, `read_hdus(...)`, and
  `read_subset(...)` skip auto-detection when you know the payload.
- **Lazy handles:** `torchsky.open(...)` returns an `HDUList`; table HDUs expose
  `TableHDURef` for column reads, Arrow scans, and materialization.
- **Headers:** `Header` preserves ordered FITS cards, comments, history, and
  WCS metadata needed downstream.
- **Checksums:** checksum helpers write and verify FITS `DATASUM` / `CHECKSUM`
  keywords at file/HDU level.

## Recipes

### Read An Image And Header

```python
import torchsky

data, header = torchsky.read("science.fits", hdu="auto", return_header=True)
```

### Read A Cutout Without Loading The Full Image

```python
stamp = torchsky.io.read_subset(
    "mosaic.fits", hdu=0, x1=1000, y1=1000, x2=1256, y2=1256
)
```

### Keep A File Open For Repeated Cutouts

```python
with torchsky.io.open_subset_reader("mosaic.fits", hdu=0) as reader:
    a = reader(0, 0, 256, 256)
    b = reader(256, 256, 512, 512)
```

### Work With HDUs

```python
with torchsky.open("mef.fits") as hdul:
    sci = hdul["SCI"].data
    header = hdul["SCI"].header
```

### Write Images Or Tables

```python
torchsky.write("out.fits", data, header={"OBJECT": "M31"}, overwrite=True)
torchsky.write("cat.fits", {"RA": ra, "DEC": dec}, overwrite=True)
```

## API Surface

Root readers:

- `torchsky.read(...)`
- `torchsky.open(...)`
- `torchsky.get_header(...)`
- `torchsky.get_wcs(...)`

Root writer:

- `torchsky.write(...)`

Specialized readers, writers, and mutations live under both `torchsky.io.fits`
and the compatibility convenience namespace `torchsky.io`:

- `torchsky.io.fits.read(...)`
- `torchsky.io.fits.write(...)`
- `torchsky.io.fits.open(...)`
- `torchsky.io.read_image(...)`
- `torchsky.io.read_table(...)`
- `torchsky.io.read_hdus(...)`
- `torchsky.io.read_subset(...)`
- `torchsky.io.open_subset_reader(...)`
- `torchsky.io.stream_table(...)`
- `torchsky.io.read_large_table(...)`
- `torchsky.io.insert_hdu(...)`
- `torchsky.io.replace_hdu(...)`
- `torchsky.io.delete_hdu(...)`
- `torchsky.io.write_checksums(...)`
- `torchsky.io.verify_checksums(...)`

HDU and header objects:

- `torchsky.HDUList`
- `torchsky.TensorHDU`
- `torchsky.TableHDU`
- `torchsky.TableHDURef`
- `torchsky.Header`
- `torchsky.Card`

Batch and low-level FITS cache helpers:

- `torchsky.io.fits.read_batch(...)`
- `torchsky.io.read_batch(...)`
- `torchsky.io.get_batch_info(...)`
- `torchsky.io.cache_subsystem_policy(...)`
- `torchsky.io.clear_cache_subsystem(...)`
- `torchsky.io.get_cache_performance(...)`
- `torchsky.io.clear_file_cache(...)`
- `torchsky.cache.clear(...)`
- `torchsky.cache.stats(...)`

Named FITS I/O cache subsystems are:

- `fits_image_data`
- `fits_table_data`
- `fits_header_metadata`
- `fits_wcs_extraction`
- `all`

`torchsky.io.fits` is the implementation-owning FITS namespace. `torchsky.io`
re-exports the complete FITS I/O helper set (`read`,
`read_image`, `read_table`, `read_hdus`, `read_subset`,
`open_subset_reader`, `open`, `write`, `insert_hdu`, `replace_hdu`,
`delete_hdu`, `write_checksums`, `verify_checksums`, `get_header`, `get_wcs`,
`read_batch`, `get_batch_info`, `get_cache_performance`, `clear_file_cache`,
streaming table helpers, HDU/header types, table interop helpers, and
named cache-subsystem helpers).

## Examples

- [Image read example](../examples/example_image.py)
- [Cutout example](../examples/example_image_cutouts.py)
- [Image cube example](../examples/example_image_cube.py)
- [Multi-extension FITS example](../examples/example_image_mef.py)

## Validation And Performance

FITS I/O parity and throughput are covered by `tests/test_fitsio_upstream_smoke.py`,
`tests/test_real_data_validation.py`, `benchmarks/bench_fits_io.py`, and
`benchmarks/bench_fitstable_io.py`. See [benchmarks.md](benchmarks.md) for
commands and comparator policy.
