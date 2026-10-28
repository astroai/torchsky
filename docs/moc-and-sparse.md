# Multi-order coverage and sparse maps

Survey work rarely fills the sky. `torchsky.sphere` gives you three explicit
representations for partial coverage — multi-order coverage (MOC), sparse
HEALPix maps, and multi-order value maps — plus dense masks when you really want
them. They compose: build coverage once, then attach values, filter catalogs, or
gate expensive map operations without materializing a full-sky array.

## Which representation?

| Representation | Stores | Use when | Reference |
|---|---|---|---|
| [`MOC`](reference/sphere.md#torchsky.sphere.MOC) | IVOA UNIQ cells, NESTED ordering | Coverage only: footprints, exposure unions, region membership, cross-match pruning | `MOC` |
| [`SparseHealpixMap`](reference/sphere.md#torchsky.sphere.SparseHealpixMap) | Values on a sorted pixel list, optional coverage map | Sparse values at one resolution (masks, weights, depth maps) | `SparseHealpixMap` |
| [`MultiOrderHealpixMap`](reference/sphere.md#torchsky.sphere.MultiOrderHealpixMap) | Values on disjoint UNIQ cells | Probabilistic sky maps whose resolution varies with density | `MultiOrderHealpixMap` |
| [`PixelizedRegion`](reference/sphere.md#torchsky.sphere.PixelizedRegion) | A set of pixels at one `nside` | Boolean geometry at a fixed resolution | `PixelizedRegion` |
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
| [`MOC.from_region`](reference/sphere.md#torchsky.sphere.MOC.from_region) | Any torchsky region exposing `pixelize`/`query_pixels`, or a `PixelizedRegion` |
| [`MOC.from_sparse_map`](reference/sphere.md#torchsky.sphere.MOC.from_sparse_map) | Coverage of a sparse map |
| [`MOC.from_threshold`](reference/sphere.md#torchsky.sphere.MOC.from_threshold) | Pixels of a sparse map above/below a value |
| [`MOC.from_wcs`](reference/sphere.md#torchsky.sphere.MOC.from_wcs) | A 2D WCS plus optional shape/mask |
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

## Sparse HEALPix maps

A sparse map stores values only for a sorted pixel list. `from_pixels` takes
explicit indices, `from_dense` extracts the valid pixels of a dense array, and
`from_moc` realizes coverage without building the dense mask first.

<!-- docs: run -->
```python
import torch
from torchsky.sphere import SparseHealpixMap

nside = 16
dense = torch.randn(12 * nside**2, dtype=torch.float64)
pixels = torch.arange(100, dtype=torch.int64)
sparse = SparseHealpixMap.from_pixels(
    pixels=pixels,
    values=dense[:100],
    nside_sparse=nside,
    sentinel=0.0,
)

assert sparse.valid_pixels.numel() == 100
assert int(sparse.coverage_mask.sum()) == 100
assert sparse.get_values(torch.tensor([0, 99, 5000])).shape == (3,)

nearest = sparse.interpolate(
    torch.tensor([0.0]), torch.tensor([0.0]), method="nearest"
)
assert nearest.shape == (1,)

coarse = sparse.ud_grade(8)
assert coarse.nside == 8

roundtrip = sparse.to_dense(fill_value=0.0)
assert torch.equal(roundtrip[:100], dense[:100])
assert float(roundtrip[roundtrip.numel() - 1]) == 0.0
print(coarse.valid_pixels.numel())
```

`get_values` returns the sentinel for uncovered pixels and preserves the leading
value dimensions, so multi-band sparse maps work the same way. `interpolate`
supports `"nearest"` and `"bilinear"`; bilinear weights drop uncovered
neighbours and renormalize. `ud_grade` uses `power`, `pess`, or `reduction`
semantics when downgrading, and preserves the coverage hierarchy when the output
resolution allows it.

`coverage_mask` is a dense boolean mask at `nside_coverage` when one is set, or
at `nside` otherwise. Use `valid_pixels` for the stored indices and
`get_covered_pixels()` for the covered set, including pixels reached through a
coverage map.

## Coverage to values and back

`MOC.from_sparse_map`, `sparse.to_moc()`, and `MOC.to_sparse_map` convert between
topology and values at an explicit resolution. The conversion never expands a
high-order MOC into dense pixels; it realizes only the requested resolution.

<!-- docs: run -->
```python
import torch
from torchsky.sphere import MOC, SparseHealpixMap

nside = 16
sparse = SparseHealpixMap.from_pixels(
    pixels=torch.arange(64, dtype=torch.int64),
    values=torch.ones(64, dtype=torch.float64),
    nside_sparse=nside,
    sentinel=0.0,
)

moc = MOC.from_sparse_map(sparse)
assert moc.area > 0.0

coverage = moc.to_sparse_map(nside=16, nside_coverage=2)
assert coverage.valid_pixels.numel() > 0
assert coverage.nside_coverage == 2
print(moc.max_order, coverage.valid_pixels.numel())
```

## Multi-order value maps

`MultiOrderHealpixMap` attaches tensor values to sorted, disjoint UNIQ cells —
the shape of probabilistic sky maps from gravitational-wave or neutrino
localization, where resolution follows density. `values_are_densities=True`
declares values per steradian, so `integral(footprint)` multiplies by cell area
and by the covered fraction of each cell. Values remain differentiable.

<!-- docs: run -->
```python
import torch
from torchsky.sphere import MOC, MultiOrderHealpixMap, nest2uniq

uniq = nest2uniq(torch.tensor([1, 2]), torch.tensor([0, 4]))
values = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
multi = MultiOrderHealpixMap(uniq, values, values_are_densities=True)

footprint = MOC(nest2uniq(2, torch.tensor([0, 4])))
total = multi.integral(footprint)
assert float(total) > 0.0
total.backward()

assert multi.uniq.numel() == 2
assert values.grad is not None
print(float(total), values.grad.tolist())
```

`from_sparse_map` and `to_sparse_map` exchange fixed-resolution maps;
`from_columns`/`to_columns` exchange `UNIQ` plus value columns. Optional
host-side adapters for `mhealpy` and `cdshealpix` detach and copy to CPU; the
tensor path has no peer dependency.

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

- [Sphere and HEALPix guide](sphere.md)
- [`torchsky.sphere` reference](reference/sphere.md)
- [Spherical harmonic transforms](sht.md)
- [Catalog crossmatching](crossmatch.md)
- [Scientific evaluation ownership](evaluation/benchmarks.md)
