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
xmatch / torchfits. Scale covering is documented in
[catalog matching research](research/catalog_matching.md).
`Catalog.to_state()` is the `CatalogState` used by `torchsky.coords`.

## Recipes

### Crossmatch two catalogs

```python
from torchsky.catalogs import Catalog, crossmatch_sky

left = Catalog(lon=left_ra, lat=left_dec)
right = Catalog(lon=right_ra, lat=right_dec)
matches = left.crossmatch(right, radius_arcsec=1.5)
# or the function kernel:
matches = crossmatch_sky(
    left.lon, left.lat, right.lon, right.lat, radius_arcsec=1.5
)
```

`find="all"` returns every pair within the radius, ordered by left index,
separation, then right index. The default `find="best"` returns the nearest
right row per left row.

`radius_arcsec` may be a positive scalar or a per-left-row tensor. A scalar
(or all-equal) radius uses a nested-HEALPix index on the right catalog.
Per-row radii keep declination-window pruning. Streamed and Arrow helpers
(`crossmatch_sky_streamed`, `crossmatch_sky_from_arrow`) still take a scalar
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

- `Catalog`
- `HealpixIndex.build` / `query_disc` / `lookup`
- `nside_for_cap` / `cap_covering_pixels`
- `CrossmatchResult`
- `crossmatch_sky(...)`
- `crossmatch_sky_streamed` / `crossmatch_sky_from_arrow`
- `CatalogStreamDataset`

## Validation and performance

Catalog compatibility is tested against Astropy-style fixtures where available
and benchmarked in `benchmarks/bench_catalog_crossmatch.py` and
`benchmarks/bench_catalog_compat.py`. Linux catalog benches stay on CANFAR;
this API wave does not claim new Linux leadership. SHA `7564070c`
`catalog_compat.json`: synthetic 10k×85k is 0.116× SkyCoord (not claimed).
