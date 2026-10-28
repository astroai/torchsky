# Catalogue crossmatching

`torchsky.catalogs` matches sky positions with an exact great-circle predicate.
Covering is always a *candidate* step: a pair is returned only after the exact
angular test, so candidate pruning can never drop a true match inside the stated
radius.

Use this page when you are choosing an entry point or sizing a match. For frames
and epochs, see [coordinates](coordinates.md); for coverage masks, see
[sphere](sphere.md); for the Arrow boundary, see [table](table.md). Full
signatures are in the [`torchsky.catalogs` reference](reference/catalogs.md).

## Choosing an entry point

| Situation | Entry point | Radius | Index reuse |
|---|---|---|---|
| In-memory `Catalog` objects with frame/epoch metadata | [`Catalog.crossmatch`](reference/catalogs.md#torchsky.catalogs.Catalog) or [`crossmatch`](reference/catalogs.md#torchsky.catalogs.crossmatch) | scalar or per-left-row tensor | `right_index=` |
| Raw longitude/latitude tensors, one tile | [`crossmatch_sky`](reference/catalogs.md#torchsky.catalogs.crossmatch_sky) | scalar or per-left-row tensor | `right_index=` |
| Left catalog arrives in batches, right fits in memory | [`crossmatch_sky_streamed`](reference/catalogs.md#torchsky.catalogs.crossmatch_sky_streamed) | scalar only | none |
| Both catalogs are Arrow tables, readers, or IPC files | [`crossmatch_sky_from_arrow`](reference/catalogs.md#torchsky.catalogs.crossmatch_sky_from_arrow) | scalar only | none |

`crossmatch(...)` dispatches on its arguments: two `Catalog` objects, four
coordinate arrays, two `(lon, lat)` tuples, or Arrow inputs. Prefer the explicit
function when you want the call site to state which path is used.

## Result contract

Every path returns a [`CrossmatchResult`](reference/catalogs.md#torchsky.catalogs.CrossmatchResult)
with `left_index`, `right_index`, and `separation_arcsec`. Indices always refer
to the **original input row order**, including Arrow batches and streamed left
partitions; no spatial sort leaks into the result.

- `find="best"` (default) returns one nearest right row per matched left row.
  Exact ties are broken by the original right row index, so the result does not
  depend on chunking.
- `find="all"` returns every pair inside the radius, ordered by left index, then
  separation, then right index.
- Separations are great-circle angles in arcseconds. Longitude wrap-around
  (359.999° vs 0.001°) and the poles need no special handling at the call site.
- An empty input returns empty tensors rather than raising.
- Matching is not one-to-one: two left rows may select the same right row.
  Association probabilities and covariance propagation are out of scope.

<!-- docs: run -->
```python
import torch

from torchsky.catalogs import crossmatch_sky

left_ra = torch.tensor([10.0, 30.0], dtype=torch.float64)
left_dec = torch.zeros(2, dtype=torch.float64)
right_ra = torch.tensor([10.0001, 10.0002, 29.9998, 80.0], dtype=torch.float64)
right_dec = torch.zeros(4, dtype=torch.float64)

best = crossmatch_sky(
    left_ra, left_dec, right_ra, right_dec, radius_arcsec=2.0
)
assert best.left_index.tolist() == [0, 1]
assert best.right_index.tolist() == [0, 2]
assert best.separation_arcsec.max() < 1.0

every = crossmatch_sky(
    left_ra, left_dec, right_ra, right_dec, radius_arcsec=2.0, find="all"
)
assert every.left_index.tolist() == [0, 0, 1]
assert every.right_index.tolist() == [0, 1, 2]
print(best.separation_arcsec)
```

## Radius semantics

`radius_arcsec` must be positive. For `crossmatch_sky` and
`Catalog.crossmatch` it may be a scalar or a tensor with one entry per left row.
Candidate generation uses the largest radius in a group, but the exact test uses
each row's own radius, so a small per-row radius is not widened by its
neighbours. Streamed and Arrow paths accept a scalar radius only.

Beware of units: `Catalog.crossmatch` takes `radius_arcsec`, while
`Catalog.cone` and `HealpixIndex.query_disc` take `radius_deg`.

## Indexing and candidate pruning

[`HealpixIndex`](reference/catalogs.md#torchsky.catalogs.HealpixIndex) buckets
rows by nested HEALPix pixel. Build it once and reuse it for cone queries and
matches:

- `HealpixIndex.build(lon, lat, nside=..., nest=True)` sorts rows into buckets;
  without `nside` it targets roughly 64 rows per pixel.
- `query_disc(lon, lat, radius_deg)` returns row indices whose positions are
  inside the cap. `lookup(pixels)` returns rows in the given pixels.
- `lookup_moc(footprint)` prunes with a MOC first, then applies the exact MOC
  predicate.
- `occupied_moc()` exposes the occupied buckets as coverage.

Two helpers describe the covering used by the matcher itself:
`nside_for_cap(radius_rad)` returns the coarsest power-of-two NSIDE whose maximum
pixel radius fits the cap, and `cap_covering_pixels(nside, lon, lat, radius_rad)`
returns pixel *centres* inside the cap. Candidate sets pad the match radius by
twice the maximum pixel radius; pass that padded radius if you build a covering
yourself.

<!-- docs: run -->
```python
import math

import torch

from torchsky.catalogs import (
    Catalog,
    HealpixIndex,
    cap_covering_pixels,
    nside_for_cap,
)

lon = torch.tensor([10.0, 10.2, 80.0], dtype=torch.float64)
lat = torch.tensor([0.0, 0.1, 20.0], dtype=torch.float64)
catalog = Catalog(lon=lon, lat=lat, columns={"id": torch.tensor([7, 8, 9])})

index = catalog.index(nside=64)
assert sorted(index.query_disc(10.0, 0.0, 0.5).tolist()) == [0, 1]

self_matches = catalog.crossmatch(catalog, radius_arcsec=1.0, right_index=index)
assert self_matches.left_index.tolist() == [0, 1, 2]
assert self_matches.right_index.tolist() == [0, 1, 2]
assert catalog.columns["id"][self_matches.left_index].tolist() == [7, 8, 9]

nside = nside_for_cap(math.radians(0.5), max_nside=4096)
covering = cap_covering_pixels(
    nside, torch.tensor([10.0]), torch.tensor([0.0]), math.radians(0.5)
)
assert covering.numel() > 0
assert isinstance(HealpixIndex.build(lon, lat), HealpixIndex)
```

An index carries the coordinates it was built from. Reusing a `right_index`
built on different positions is rejected, and a prebuilt index cannot be
combined with `target_epoch` or frame conversion, because those change the
evaluated positions.

## Per-row radii and streaming

<!-- docs: run -->
```python
import torch

from torchsky.catalogs import crossmatch_sky, crossmatch_sky_streamed

left_ra = torch.tensor([10.0, 10.0], dtype=torch.float64)
left_dec = torch.tensor([0.0, 0.001], dtype=torch.float64)
right_ra = torch.tensor([10.0], dtype=torch.float64)
right_dec = torch.tensor([0.0], dtype=torch.float64)

radius = torch.tensor([1.0, 1.0], dtype=torch.float64)
per_row = crossmatch_sky(
    left_ra, left_dec, right_ra, right_dec, radius_arcsec=radius
)
assert per_row.left_index.tolist() == [0]

batches = iter([(left_ra[:1], left_dec[:1]), (left_ra[1:], left_dec[1:])])
streamed = crossmatch_sky_streamed(
    batches, right_ra, right_dec, radius_arcsec=1.0
)
assert streamed.left_index.tolist() == [0]
```

`crossmatch_sky_streamed` always returns the nearest right row per matched left
row; it has no `find` argument. The right catalogue must fit in memory because it
is indexed once. Left indices are offsets into the concatenation of all batches,
not into one batch.

## Arrow inputs

`crossmatch_sky_from_arrow` accepts a `pyarrow.Table`, a
`pyarrow.ipc.RecordBatchReader`, or a path to an Arrow IPC file. Both inputs are
streamed once into temporary HEALPix tiles (Parquet row groups), then matched one
tile at a time. Coordinate memory scales with the largest tile and its
neighbouring right tiles; pixel metadata and the returned pairs stay in RAM.
This is not a distributed output sink.

Select native identifiers with the returned indices instead of turning them into
floats:

```python
pairs = crossmatch_sky_from_arrow(left, right, radius_arcsec=2.0, find="all")
left_ids = left["source_id"].take(pairs.left_index.numpy())
right_ids = right["source_id"].take(pairs.right_index.numpy())
```

`left_columns` / `right_columns` name the `(ra, dec)` columns when they are not
`("ra", "dec")`. `pyarrow` must be installed; it is not a default `torchsky`
runtime dependency. The unified `crossmatch(..., streamed=True)` form expects an
iterable of `(ra, dec)` batches, not an Arrow table.

## Frames, epochs, and motion

`Catalog.crossmatch` matches stored positions by default. When two catalogues
use different astrometric reference epochs, align them explicitly:

```python
pairs = left.crossmatch(
    right,
    radius_arcsec=2.0,
    find="all",
    target_epoch=2016.0,
    motion="angular",
)
```

Angular propagation needs both proper-motion components; `motion="physical"`
additionally needs positive parallax and radial velocity. `target_epoch` must be
a scalar. The match runs on the propagated positions, the original catalogues are
left unchanged, and payload columns and row indices are preserved. See
[catalogue foundations](catalogue-foundations.md) for the propagation contract and
the cases that are rejected (partial proper motion, FK4 time-dependent
conversion).

## Scaling checklist

- Build the right-side index once and pass `right_index=`; do not rebuild it per
  chunk.
- Tune `left_chunk_size` / `right_chunk_size` for the working set; they do not
  change results.
- Pass `device=` to `crossmatch_sky` to move inputs and outputs together.
  Indices agree between CPU and CUDA for the same inputs; separations can differ
  by floating-point tolerance.
- For catalogues that do not fit in memory, stream the left side; for two large
  Arrow catalogues, use the Arrow path.
- Measure candidate reduction and throughput on your own data; reference
  measurements and large-catalogue campaigns live in
  [`torchsky-harness`](https://github.com/astroai/torchsky-harness).
