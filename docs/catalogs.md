# `torchsky.catalogs`

`torchsky.catalogs` contains scalable catalog helpers such as sky crossmatching
and streaming dataset adapters.

## What This Namespace Is For

Use this namespace when tabular sky data needs spatial matching or partitioned
streaming. Use [table.md](table.md) for the `torchfits.table` boundary and
Arrow/tensor batch interchange.

## Core Concepts

- Crossmatching uses simple public inputs: RA/Dec tensors in degrees plus a
  radius in arcseconds.
- Candidate pruning is internal. The current implementation uses a coarse
  nested-HEALPix right-catalog index before exact great-circle refinement.
- Streaming helpers are intentionally thin adapters, not full catalog workflow
  orchestration.

## Recipes

### Crossmatch Two Catalogs

```python
from torchsky.catalogs import crossmatch_sky

matches = crossmatch_sky(
    left_ra_deg=left["RA"],
    left_dec_deg=left["DEC"],
    right_ra_deg=right["RA"],
    right_dec_deg=right["DEC"],
    radius_arcsec=1.5,
)
```

Use `find="all"` when downstream association policy needs every candidate
within the radius. Pair rows are deterministically ordered by left index,
separation, then right index. The default `find="best"` returns only the
nearest candidate per left row.

`radius_arcsec` may be a positive scalar or a per-left-row tensor. Coarse
declination windows use each row's own radius; the global prune uses the
maximum. Streamed and Arrow helpers (`crossmatch_sky_streamed`,
`crossmatch_sky_from_arrow`) still take a scalar radius.

### Stream Catalog Partitions

```python
from torch.utils.data import DataLoader
from torchsky.catalogs import CatalogStreamDataset

dataset = CatalogStreamDataset(partitions=range(8), loader=load_partition)
for partition in DataLoader(dataset, num_workers=2, batch_size=None):
    process(partition)
```

## API Surface

- `crossmatch_sky(...)`
- `CatalogStreamDataset`

`crossmatch_sky(...)` returns an object with `left_index`, `right_index`, and
`separation_arcsec` fields. The concrete `CrossmatchResult` type currently
lives in `torchsky.catalogs.core`; use the function return value rather than
importing that class from `torchsky.catalogs`.

## Validation And Performance

Catalog compatibility is tested against Astropy-style fixtures where available
and benchmarked in `benchmarks/bench_catalog_crossmatch.py` and
`benchmarks/bench_catalog_compat.py`.
