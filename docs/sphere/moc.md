# Multi-order coverage (MOC)
<!-- TODO(0.5.0 docs): intro -->


Survey work rarely fills the sky. `torchsky.sphere` gives you three explicit
representations for partial coverage — multi-order coverage (MOC), sparse
HEALPix maps, and multi-order value maps — plus dense masks when you really want
them. They compose: build coverage once, then attach values, filter catalogs, or
gate expensive map operations without materializing a full-sky array.

## Which representation?

| Representation | Stores | Use when | Reference |
|---|---|---|---|
| [`MOC`](../reference/sphere.md#torchsky.sphere.MOC) | IVOA UNIQ cells, NESTED ordering | Coverage only: footprints, exposure unions, region membership, cross-match pruning | `MOC` |
| [`SparseHealpixMap`](../reference/sphere.md#torchsky.sphere.SparseHealpixMap) | Values on a sorted pixel list, optional coverage map | Sparse values at one resolution (masks, weights, depth maps) | `SparseHealpixMap` |
| [`MultiOrderHealpixMap`](../reference/sphere.md#torchsky.sphere.MultiOrderHealpixMap) | Values on disjoint UNIQ cells | Probabilistic sky maps whose resolution varies with density | `MultiOrderHealpixMap` |
| [`PixelizedRegion`](../reference/sphere.md#torchsky.sphere.PixelizedRegion) | A set of pixels at one `nside` | Boolean geometry at a fixed resolution | `PixelizedRegion` |
| Dense `torch.Tensor` | Every pixel | The map is genuinely full-sky and fits in memory | — |

MOC stores topology, not values. `MultiOrderHealpixMap` layers values on top of
sorted, disjoint UNIQ cells; it does not define a second sky index. Sparse maps
keep the healsparse-compatible `(nside, nside_coverage, sentinel)` contract but
compute with torchsky kernels.

## Build coverage

Coverage constructors are hierarchical: interior cells stay coarse and only the
boundary is subdivided to `max_order`, so a footprint does not enumerate every
leaf pixel.

<!-- docs: run -->
```python
import torch
from torchsky.sphere import MOC

circle = MOC.from_circle(10.0, 20.0, 1.0, max_order=8)
inside = circle.contains(
    torch.tensor([10.0, 200.0]), torch.tensor([20.0, -40.0])
)
assert inside.tolist() == [True, False]
assert circle.area > 0.0

polygon = MOC.from_polygon(
    torch.tensor([0.0, 2.0, 2.0, 0.0]),
    torch.tensor([0.0, 0.0, 2.0, 2.0]),
    max_order=6,
)
assert bool(polygon.contains(torch.tensor(1.0), torch.tensor(1.0)))

points = MOC.from_lonlat(
    torch.tensor([15.0, 16.0]), torch.tensor([-3.0, 4.0]), max_order=6
)
assert bool(points.contains(torch.tensor([15.0]), torch.tensor([-3.0])))
print(round(circle.area_sq_deg, 3), polygon.to_ascii())
```

Other constructors cover the rest of the workflow:

| Constructor | Input |
|---|---|
| [`MOC.from_region`](../reference/sphere.md#torchsky.sphere.MOC.from_region) | Any torchsky region exposing `pixelize`/`query_pixels`, or a `PixelizedRegion` |
| [`MOC.from_sparse_map`](../reference/sphere.md#torchsky.sphere.MOC.from_sparse_map) | Coverage of a sparse map |
| [`MOC.from_threshold`](../reference/sphere.md#torchsky.sphere.MOC.from_threshold) | Pixels of a sparse map above/below a value |
| [`MOC.from_wcs`](../reference/sphere.md#torchsky.sphere.MOC.from_wcs) | A 2D WCS plus optional shape/mask |
| `MOC.empty()` / `MOC.full()` | Empty coverage / full sky |
| `MOC.from_ascii` / `from_json` / `read_fits` | IVOA interchange |

When you already have a pixelized region at a known resolution, `from_region`
preserves it instead of repixelizing; pass `max_order` to assert the expected
resolution.

## Query coverage

`contains` is the point predicate and broadcasts over tensors. For indexed
catalogs, `contains_pixels` tests HEALPix pixels directly; `overlap_fraction`
measures how much of a UNIQ cell is covered without expanding to leaves.

<!-- docs: run -->
```python
import torch
from torchsky.sphere import MOC, nest2uniq

moc = MOC.from_circle(0.0, 0.0, 5.0, max_order=6)
assert bool(moc.contains(torch.tensor(0.0), torch.tensor(0.0)))
assert not bool(moc.contains(torch.tensor(180.0), torch.tensor(0.0)))

cell = nest2uniq(4, torch.tensor([31]))
fraction = moc.overlap_fraction(cell)
assert 0.0 <= float(fraction) <= 1.0

pixels = torch.tensor([31, 32])
covered = moc.contains_pixels(64, pixels, nest=True)
assert covered.shape == pixels.shape
print(round(float(fraction), 4), covered.tolist())
```

`MOC.overlaps(other)` answers the exposure-intersection question without
materializing an intersection; `contains_moc` is the subset test. For catalogs,
prefer `Catalog.inside(moc)` or `moc.contains(lon, lat)` and index the boolean
mask, rather than filtering coordinates twice.

## Boolean algebra and resolution

MOCs are closed under union, intersection, and difference, including cells at
different orders. Operators `|`, `&`, and `-` are aliases for the method calls.
`degrade` drops fine cells onto nested parents, and `flatten` realizes a dense
boolean mask at a chosen `nside`.

On CPU, the set operations, UNIQ-to-range conversion, and `degrade` run in small
native kernels over sorted half-open pixel ranges (microseconds for footprints of
thousands of cells; the torch implementation remains the path for CUDA/MPS and
the parity oracle in the tests). Results are identical to the torch path and to
mocpy.

<!-- docs: run -->
```python
import torch
from torchsky.sphere import MOC

a = MOC.from_circle(0.0, 0.0, 2.0, max_order=6)
b = MOC.from_circle(3.0, 0.0, 2.0, max_order=6)

union = a | b
intersection = a & b
difference = a - b

assert union.area >= max(a.area, b.area)
assert intersection.area <= min(a.area, b.area)
assert difference.area <= a.area
assert a.overlaps(b)

coarse = union.degrade(4)
assert coarse.max_order <= 4

mask = a.flatten(64)
assert mask.dtype == torch.bool
assert mask.numel() == 12 * 64**2
print(round(union.area, 4), int(mask.sum()))
```

`to_pixelized_region(nside, nest=True)` converts coverage into a
`PixelizedRegion` at a fixed resolution, and `to_sparse_map(nside, ...)` realizes
it as a sparse map with a sentinel outside the footprint. Both are explicit
resampling steps; neither mutates the MOC.

## Persist and interchange

MOC serialization follows IVOA MOC: ASCII (`order/pixel` ranges), JSON, and FITS.
Writing goes through `torchfits`; reading accepts the same representations.
`overlaps`, set algebra, and `contains` work on the deserialized object.

<!-- docs: run -->
```python
import torch
from torchsky.sphere import MOC

moc = MOC.from_circle(120.0, -30.0, 0.5, max_order=7)

ascii_text = moc.to_ascii(compact=True)
json_text = moc.to_json()

assert abs(MOC.from_ascii(ascii_text).area - moc.area) < 1e-12
assert abs(MOC.from_json(json_text).area - moc.area) < 1e-12
assert bool(MOC.from_ascii(ascii_text).contains(torch.tensor(120.0), torch.tensor(-30.0)))
print(ascii_text[:40], "…")
```

`BMOC` is the boundary-MOC variant for footprints that need their edges
represented explicitly; `STMOC` covers space-time (next section).

## Space-time coverage

`STMOC` is IVOA MOC 2.0: spatial MOC cells paired with microsecond time cells
(1 µs since JD = 0). Use it for exposure windows, transient localization, and
time-filtered sky queries.

<!-- docs: run -->
```python
from torchsky.sphere import MOC, STMOC

spatial = MOC.from_circle(10.0, 0.0, 1.0, max_order=6)
stmoc = STMOC.from_spatial_coverages(
    59000.0, 59001.0, spatial, time_depth=16
)

assert bool(stmoc.contains(59000.5, 10.0, 0.0))
assert not bool(stmoc.contains(60000.0, 10.0, 0.0))

window = stmoc.query_by_time(59000.0, 59000.5)
assert window.area > 0.0
print(round(window.area, 6))
```

`from_times_positions` quantizes catalog samples onto time × HEALPix cells;
`union`, `intersection`, and `difference` sweep over time breakpoints. Time
conversion helpers `mjd_to_time_us` / `time_us_to_mjd` use the same epoch.
`STMOC` is implemented natively, not as a mocpy wrapper.

## Performance and ownership

- MOC set operations work on sorted NEST ranges, so unions and intersections cost
  time proportional to the number of ranges, not the number of leaf pixels.
- Coverage queries use binary search over ranges; point containment does not
  allocate a dense mask.
- Sparse lookups use a sorted pixel list, with a fused CPU gather when a coverage
  map is present. Boolean and integer maps keep the pixel-list path so an
  uncovered child cannot be confused with a stored `False`/`0`.
- `flatten` and `to_dense` are the only operations that allocate a full-sky
  array; call them once at the boundary rather than inside a loop.

Comparative throughput and parity evidence for sparse maps and MOCs live in
`torchsky-harness`. `healsparse` and `mocpy` are optional host interchange
targets, never runtime backends.

## See also

- [Sphere and HEALPix guide](index.md)
- [`torchsky.sphere` reference](../reference/sphere.md)
- [Spherical harmonic transforms](sht.md)
- [Catalogs and crossmatch](../catalogs/index.md)
- [Scientific evaluation ownership](../performance/benchmarks.md)
