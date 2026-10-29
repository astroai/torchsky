# FITS I/O Boundary

FITS file and table I/O is owned by
[`torchfits`](https://github.com/astroai/torchfits). `torchsky` does not define a
second FITS backend. Its compute APIs consume tensors, FITS-like header
mappings, masks, weights, and streamed table batches.

## Install

Install both packages for FITS-backed sky workflows:

```bash
pip install torchsky torchfits
```

`torchsky` remains usable without `torchfits` when inputs are already in memory
or come from another storage system.

## Version Contract

The supported integration baseline is `torchfits 1.3.1` or newer. Dataset
classes live under `torchfits.data`; use `FitsImageDataset`,
`FitsImageIterableDataset`, `FitsTableDataset`, or
`FitsTableIterableDataset` rather than removed root-level legacy aliases.

## Image And WCS Workflow

```python
import torchfits
from torchsky.wcs import WCS

image, header = torchfits.read(
    "science.fits",
    hdu="auto",
    device="cuda",
    return_header=True,
)

wcs = WCS(header)
ra, dec = wcs.pixel_to_world(x, y)
```

The boundary is deliberately data-oriented:

- `torchfits` owns file selection, HDU semantics, dtype decoding, compression,
  checksums, cutouts, caches, and writes.
- `torchsky.wcs` owns coordinate transforms constructed from metadata.
- `torchsky.map`, `torchsky.scene`, and `torchsky.field` own tensor-domain
  observation and modelling semantics.

## Tables And Catalogs

```python
import torchfits
from torchsky.catalogs import crossmatch_sky

for batch in torchfits.table.scan(
    "catalog.fits",
    columns=["RA", "DEC"],
    batch_size=100_000,
):
    matches = crossmatch_sky(batch["RA"], batch["DEC"], ref_ra, ref_dec)
```

FITS table mutation, Arrow conversion, and predicate pushdown belong to
`torchfits.table`. Sky partitioning, spherical indexing, crossmatch, and catalog
models belong to `torchsky.catalogs`.

## Contract Requirements

Adapters crossing the package boundary must preserve:

- tensor dtype, shape, device, and byte-order normalization;
- ordered header values needed by WCS, including alternate keys and `-TAB`
  references;
- mask and weight meaning—never infer inverse variance or validity silently;
- spectral/time units and axis ordering;
- chunk boundaries without changing row order or duplicating rows.

Ambiguous metadata must fail closed. Core torchsky code should not inspect file
paths to guess semantics.

## Compatibility Boundary

`torchsky` no longer exposes `torchsky.read`, `torchsky.open`, `torchsky.io`,
`torchsky.table`, `torchsky.hdu`, `torchsky.cache`, or dataset/dataloader
aliases. Use the public `torchfits` API directly. Any future compatibility
surface must be a new, documented design with its own removal policy; it must
not restore the removed aliases.

## Validation And Performance

- FITS image/table correctness and throughput are validated in `torchfits`.
- `torchsky` keeps one integration gate covering the supported `torchfits 1.3.1+`
  output passed into WCS/map/catalog computation; 0.8 adds optimized helpers.
- End-to-end torchsky benchmarks start after data is decoded, but must still
  report transfer volume and device movement when those affect the workflow.
