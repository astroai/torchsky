# `torchsky.io` And Root FITS I/O

`torchsky` keeps high-frequency FITS operations at the package root for
compatibility. `torchsky.io` is the canonical namespace-oriented FITS surface:
it exposes the same core readers/writers plus HDU mutation, checksum, batch,
streaming, and cache helpers through direct private-engine routes.

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
stamp = torchsky.read_subset("mosaic.fits", hdu=0, x1=1000, y1=1000, x2=1256, y2=1256)
```

### Keep A File Open For Repeated Cutouts

```python
with torchsky.open_subset_reader("mosaic.fits", hdu=0) as reader:
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
- `torchsky.read_image(...)`
- `torchsky.read_table(...)`
- `torchsky.read_hdus(...)`
- `torchsky.read_subset(...)`
- `torchsky.open_subset_reader(...)`
- `torchsky.stream_table(...)`
- `torchsky.read_large_table(...)`

Root writers and mutations:

- `torchsky.write(...)`
- `torchsky.insert_hdu(...)`
- `torchsky.replace_hdu(...)`
- `torchsky.delete_hdu(...)`
- `torchsky.write_checksums(...)`
- `torchsky.verify_checksums(...)`

Checksum helpers are package-callable as qualified root names and through
`torchsky.io`.

HDU and header objects:

- `torchsky.HDUList`
- `torchsky.TensorHDU`
- `torchsky.TableHDU`
- `torchsky.TableHDURef`
- `torchsky.Header`
- `torchsky.Card`

Root batch/runtime helpers:

- `torchsky.read_batch(...)`
- `torchsky.get_batch_info(...)`
- `torchsky.get_cache_performance(...)`
- `torchsky.clear_file_cache(...)`
- `torchsky.configure_for_environment(...)`

`torchsky.io` re-exports the complete FITS I/O helper set (`read`,
`read_image`, `read_table`, `read_hdus`, `read_subset`,
`open_subset_reader`, `open`, `write`, `insert_hdu`, `replace_hdu`,
`delete_hdu`, `write_checksums`, `verify_checksums`, `get_header`, `get_wcs`,
`read_batch`, `get_batch_info`, `get_cache_performance`, `clear_file_cache`,
streaming table helpers, HDU/header types, table interop helpers, and
dataloader helpers).

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
