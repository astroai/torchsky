# Catalogs: overview
<!-- TODO(0.5.0 docs): intro -->


`torchsky.catalogs` is a batched sky table plus the index and match kernels it
calls. Coordinates are tensors; there is no `SkyCoord` class. Matching uses an
exact great-circle predicate: covering is only ever a *candidate* step, so
pruning can never drop a true match inside the stated radius.

Frames and epochs are covered in [Coordinate frames and motion](../coordinates/frames.md), MOC coverage
in [MOC and sparse maps](../sphere/moc.md), and the table boundary in
[Concepts](../introduction/concepts.md#file-boundary). Signatures are in the
[`torchsky.catalogs` reference](../reference/catalogs.md).

## Choosing an entry point

| Situation | Entry point | Radius | Index reuse |
|---|---|---|---|
| In-memory `Catalog` objects with frame/epoch metadata | [`Catalog.crossmatch`](../reference/catalogs.md#torchsky.catalogs.Catalog) or [`crossmatch`](../reference/catalogs.md#torchsky.catalogs.crossmatch) | scalar or per-left-row tensor | `right_index=` |
| Raw longitude/latitude tensors, one tile | [`crossmatch_sky`](../reference/catalogs.md#torchsky.catalogs.crossmatch_sky) | scalar or per-left-row tensor | `right_index=` |
| Left catalog arrives in batches, right fits in memory | [`crossmatch_sky_streamed`](../reference/catalogs.md#torchsky.catalogs.crossmatch_sky_streamed) | scalar only | none |
| Both catalogs are Arrow tables, readers, or IPC files | [`crossmatch_sky_from_arrow`](../reference/catalogs.md#torchsky.catalogs.crossmatch_sky_from_arrow) | scalar only | none |

`crossmatch(...)` dispatches on its arguments: two `Catalog` objects, four
coordinate arrays, two `(lon, lat)` tuples, or Arrow inputs. Prefer the explicit
function when you want the call site to state which path is used.
