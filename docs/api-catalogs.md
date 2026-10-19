# Catalog API reference

`torchsky.catalogs` is the sky-domain catalog layer: it owns indexing,
cone queries, crossmatching, and streaming datasets. Table parsing belongs to
[`torchfits.table`](table.md); this namespace consumes tensors and Arrow
batches.

---

## Catalog model

```python
from torchsky.catalogs import Catalog

catalog = Catalog(lon_deg, lat_deg)
```

| Name | Contract |
|---|---|
| `Catalog` | Batched sky rows on one device: `lon`, `lat`, optional proper motion, parallax, radial velocity, validity mask, and extra tensor columns. |
| `CatalogStreamDataset` | Wrap a partition iterator into a `torch.utils.data.IterableDataset` of `Catalog` batches. |
| `CrossmatchResult` | `left_index`, `right_index`, and `separation_arcsec` for matched pairs. |
| `HealpixIndex` | Bucketed NESTED HEALPix index over catalog rows; the engine behind crossmatch and cone queries. |

`Catalog` stores spherical coordinates in degrees and keeps frames/epochs
explicit (`frame`, `epoch`). There is no implicit coordinate conversion in a
crossmatch: pass catalogs already in the same frame and epoch.

---

## Crossmatching

| Name | Contract |
|---|---|
| `crossmatch(left, right, radius_arcsec, *, find="best")` | Unified entry point. `find="best"` returns one best match per left row; `find="all"` returns every pair within the radius. |
| `crossmatch_sky(...)` | Chunked all-pairs crossmatch over two in-memory tiles. |
| `crossmatch_sky_from_arrow(...)` | Crossmatch Arrow inputs by materializing HEALPix tiles. |
| `crossmatch_sky_streamed(...)` | Stream the left catalog in batches; the right side is indexed once. |

```python
from torchsky.catalogs import crossmatch

matched = crossmatch(catalog_a, catalog_b, radius_arcsec=1.0, find="best")
print(matched.left_index[:5], matched.separation_arcsec.max())
```

Chunking and the HEALPix index bound memory; the index resolution is chosen from
the search radius by `nside_for_cap`.

---

## Indexing and coverage helpers

| Name | Contract |
|---|---|
| `cap_covering_pixels(nside, lon, lat, radius_rad)` | HEALPix pixels whose centers lie in a spherical cap. |
| `nside_for_cap(radius_rad)` | Coarsest power-of-two NSIDE whose pixel radius is small enough for the cap. |

These are the primitives a partitioner uses to assign catalog rows to HEALPix
tiles; they are exposed so survey-specific partition schemes can reuse the same
bounds.

---

See also: [catalogs guide](catalogs.md), [catalogue foundations](catalogue-foundations.md),
[HEALPix reference](api-sphere.md), and the [table boundary](table.md).
