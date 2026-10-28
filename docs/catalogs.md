# `torchsky.catalogs`

`torchsky.catalogs` is a batched sky table plus the index and match kernels it
calls. Coordinates are tensors. There is no `SkyCoord` class.

Use [coordinates.md](coordinates.md) for frames and observed place,
[sphere.md](sphere.md) for MOC coverage, and [table.md](table.md) for the
`torchfits.table` / Arrow interchange boundary.

## Types that compose

```python
from torchsky.catalogs import Catalog, HealpixIndex, crossmatch_sky
from torchsky.sphere import MOC

cat = Catalog(lon=ra, lat=dec, frame="ICRS", columns={"id": ids})
idx = cat.index()                       # HealpixIndex, build once
hits = cat.cone(10.0, 20.0, radius_deg=0.5)
inside = cat.inside(MOC.from_lonlat(ra, dec, max_order=8))
gal = cat.to_frame("galactic")
matches = cat.crossmatch(other, radius_arcsec=1.5)
subset = cat.take(inside)
```

- **`Catalog`** — `lon`/`lat` in degrees, canonical `frame`, `epoch`. Optional
  proper motion, parallax, radial velocity, `valid`, and extra `columns`.
  Missing kinematics stay `None`. `to(device)` follows the WCS work dtype
  (float64 CPU/CUDA, float32 MPS).
- **`HealpixIndex`** — nested HEALPix buckets over rows. `query_disc` and
  `lookup` are the primitives. Crossmatch uses the same index; pass
  `right_index=` to `crossmatch_sky` to reuse one.
- **`CrossmatchResult`** — `left_index`, `right_index`, `separation_arcsec`.
  Import it from `torchsky.catalogs`.
- **Coverage** — `MOC.contains` / `Catalog.inside` return a boolean mask.
  `Catalog.inside(stmoc, time_mjd=...)` is the STMOC predicate. `Catalog.take`
  copies rows. `filter_catalog` remains as a convenience.

`from_arrow` / `CatalogStreamDataset` feed tensors into `Catalog`.
There is no `from_polars` / `to_polars` wrapper; Polars and HATS stay in
xmatch / torchfits. Scale covering and the exact-pair design are documented in
[Matching design](#matching-design-exact-covering-first).
`Catalog.to_state()` is the `CatalogState` used by `torchsky.coords`.

## Recipes

### Crossmatch two catalogs

```python
from torchsky.catalogs import Catalog, crossmatch

left = Catalog(lon=left_ra, lat=left_dec)
right = Catalog(lon=right_ra, lat=right_dec)
matches = crossmatch(left, right, 1.5)
# or directly on coordinate tensors:
matches = crossmatch(
    left.lon, left.lat, right.lon, right.lat, radius_arcsec=1.5
)
```

`find="all"` returns every pair within the radius, ordered by left index,
separation, then right index. The default `find="best"` returns the nearest
right row per left row.

`radius_arcsec` may be a positive scalar or a per-left-row tensor. A scalar
(or all-equal) radius uses a nested-HEALPix index on the right catalog.
Per-row radii keep declination-window pruning. Streamed and Arrow helpers
(`crossmatch(left_arrow, right_arrow, radius_arcsec, streamed=True)`) take a scalar
radius.

### Stream catalog partitions

```python
from torch.utils.data import DataLoader
from torchsky.catalogs import CatalogStreamDataset

dataset = CatalogStreamDataset(partitions=range(8), loader=load_partition)
for partition in DataLoader(dataset, num_workers=2, batch_size=None):
    process(partition)
```

## API surface

- `Catalog` (`.state`, `.to_state()`, `Catalog.from_state(...)`, `.crossmatch(...)`, `.cone(...)`, `.inside(...)`)
- `crossmatch(...)` (unified entry point for `Catalog`, coordinate tensors, and Arrow/streamed tables)
- `HealpixIndex.build` / `query_disc` / `lookup`
- `nside_for_cap` / `cap_covering_pixels`
- `CrossmatchResult`
- `crossmatch_sky(...)`, `crossmatch_sky_streamed(...)`, `crossmatch_sky_from_arrow(...)`
- `CatalogStreamDataset`

## Validation and performance

Core catalog tests cover bounded API, regression, and geometric contracts.
Comparator parity, large-catalog throughput, candidate-reduction measurements,
and Linux/real-data campaigns live in
[torchsky-harness](https://github.com/astroai/torchsky-harness). No benchmark
result is inferred from a passing core test.

## Matching design: exact covering first

Matching needs **recall = 1** under a stated metric. On the unit sphere the
metric is the great-circle angle; torchsky uses the stable
`atan2(||a x b||, a . b)` form. Euclidean k-NN in R^3 on unit vectors is an
*exact* spherical-cap query only with the chord radius `2 sin(theta/2)`.

The join is partition-then-refine:

1. Cover the query and catalog with HEALPix cells of bounded radius.
2. Generate candidates whose cell centres are within `radius + 2 * max_pixel_radius`.
3. Apply the exact great-circle (or Mahalanobis) test to surviving pairs.

Dropping a true pair during candidate generation is a correctness bug, never an
optimization. Per-row positional error must cover that row's radius; a global
maximum radius is only a candidate-generation shortcut, not the final test.
Approximate nearest-neighbour indexes are not a substitute for the exact
predicate, and HEALPix, HTM, Q3C, and LSST sphgeom are alternative coverings,
not multiple torchsky APIs.
