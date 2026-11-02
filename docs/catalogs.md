# Catalogs and crossmatch

`torchsky.catalogs` is a batched sky table plus the index and match kernels it
calls. Coordinates are tensors; there is no `SkyCoord` class. Matching uses an
exact great-circle predicate: covering is only ever a *candidate* step, so
pruning can never drop a true match inside the stated radius.

Frames and epochs are covered in [Coordinate frames and motion](coordinates.md), MOC coverage
in [MOC and sparse maps](moc-and-sparse.md), and the table boundary in
[Concepts](concepts.md#file-boundary). Signatures are in the
[`torchsky.catalogs` reference](reference/catalogs.md).

## Types That Compose

<!-- docs: run -->
```python
import torch

from torchsky.catalogs import Catalog, HealpixIndex, crossmatch
from torchsky.sphere import MOC

g = torch.Generator().manual_seed(0)
ra = torch.rand(1000, generator=g, dtype=torch.float64) * 20 + 10
dec = torch.rand(1000, generator=g, dtype=torch.float64) * 20 - 10

cat = Catalog(lon=ra, lat=dec, frame="ICRS", columns={"id": torch.arange(1000)})
idx = cat.index()                                   # build once, reuse
assert isinstance(idx, HealpixIndex)
hits = cat.cone(20.0, 0.0, radius_deg=0.5)
inside = cat.inside(MOC.from_lonlat(ra[:100], dec[:100], max_order=8))
gal = cat.to_frame("galactic")
other = Catalog(lon=ra + 1e-4, lat=dec)
matches = cat.crossmatch(other, radius_arcsec=1.5)
subset = cat.take(inside)

assert matches.left_index.numel() == 1000
assert subset.lon.shape[0] == int(inside.sum())
```

- **`Catalog`**: `lon`/`lat` in degrees, a canonical `frame`, and `epoch`.
  Optional proper motion, parallax, radial velocity, `valid`, and extra
  `columns`. Missing kinematics stay `None`. `to(device)` follows the WCS work
  dtype (float64 on CPU/CUDA, float32 on MPS).
- **`HealpixIndex`**: nested HEALPix buckets over rows. Crossmatch uses the same
  index; pass `right_index=` to reuse one.
- **`CrossmatchResult`**: `left_index`, `right_index`, `separation_arcsec`.
  STILTS-style `find`, errors, joins, self and multi-catalogue matching are in
  [STILTS-style matching](#stilts-style-matching).
- **Coverage**: `MOC.contains` and `Catalog.inside` return a boolean mask;
  `Catalog.take` copies rows.

`from_arrow` and `CatalogStreamDataset` feed tensors into `Catalog`. There is no
Polars wrapper; read tables with `torchfits.table` and pass batches in.

```python
# fragment: partition loader is user code
from torch.utils.data import DataLoader
from torchsky.catalogs import CatalogStreamDataset

dataset = CatalogStreamDataset(partitions=range(8), loader=load_partition)
for partition in DataLoader(dataset, num_workers=2, batch_size=None):
    process(partition)
```

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

- `find="best"` (default, same as `"best1"`) returns one nearest right row per
  matched left row. Exact ties are broken by the original right row index, so
  the result does not depend on chunking. The other `find` modes are listed in
  [STILTS-style matching](#stilts-style-matching).
- `find="all"` returns every pair inside the radius, ordered by left index, then
  separation, then right index.
- Separations are great-circle angles in arcseconds. Longitude wrap-around
  (359.999° vs 0.001°) and the poles need no special handling at the call site.
- An empty input returns empty tensors rather than raising.
- `best`/`best1` is not one-to-one: two left rows may select the same right row
  (use `find="symmetric"` for one-to-one). Association probabilities and covariance propagation are out of scope.

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

## STILTS-style matching

The same strip index drives the matchers STILTS users rely on. Every call is
exact (a hit is a hit under the great-circle test, never a pruned
approximation), `O((N + M) log M + K)` for `K` candidate pairs, tensor-native,
and runs on CPU, CUDA and MPS (float32 on MPS, so thresholds below the float32
position resolution are not meaningful there). Indices always refer to the
input row order.

| torchsky | STILTS |
|---|---|
| `crossmatch_sky(..., radius_arcsec=r)` | `tmatch2 matcher=sky params=r` / `tskymatch2` |
| `crossmatch_sky(..., left_err_arcsec=, right_err_arcsec=)` | `tmatch2 matcher=skyerr` (match if `sep <= err1 + err2`) |
| `crossmatch_sky(..., left_distance=, right_distance=, distance_tol=)` | sky plus a 1-d cut (not the Euclidean metric of `sky3d`) |
| `crossmatch_exact(key1, key2)` | `tmatch2 matcher=exact` |
| `find="best1"` / `"best2"` / `"symmetric"` / `"all"` | `find=best1` / `best2` / `best` / `all` |
| `join_rows(result, n1, n2, join)` | `join=1and2`, `1or2`, `all1`, `all2`, `1not2`, `2not1` |
| `crossmatch_self(..., group=True)` | `tmatch1 action=identify` |
| `crossmatch_many(cats, r, mode="pairs")` / `"group"` | `tmatchn multimode=pairs` / `group` |

Not provided: `skyellipse`, and error-normalised scores (ranking always uses
plain separation in arcseconds).

**`find`.** The name `"best"` keeps its older torchsky meaning (`best1`), which
differs from STILTS `find=best`; the STILTS one is `"symmetric"`.

- `best1`: nearest right row per left row; ties go to the lowest right index.
- `best2`: nearest left row per right row; ties go to the lowest left index.
- `symmetric`: one-to-one, greedy by ascending separation (ties by lowest left,
  then lowest right index); each row appears at most once. Rounds resolve all
  locally dominant pairs at once, which equals the sequential greedy pass.
  Typical inputs need a few `O(K)` rounds; only an adversarial chain of strictly
  monotone separations is slower.
- Results are ordered by left index then right index (`all`: left, separation,
  right).

**Per-row errors.** With `left_err_arcsec` / `right_err_arcsec` (scalars or
per-row tensors, either side optional, finite and `>= 0`) a pair matches when
`separation <= err_left[i] + err_right[j]`. Candidate windows use
`max(err_left) + max(err_right)`; the exact test uses each pair's own sum.
`radius_arcsec` and the error options are alternatives. `Catalog.crossmatch`
accepts the same options, and its error arguments may name a column.

**Joins.** `join_rows` turns pairs into index tensors with `-1` for the absent
side. Order: matched pairs as given, then unmatched left rows, then unmatched
right rows.

**Self-match and groups.** `crossmatch_self` returns each pair `i < j` once. With
`group=True` it adds friends-of-friends groups (connected components, scatter-min
union-find): per-row `group_id` (`-1` for rows with no partner, otherwise
`0..n_groups-1` by smallest member row) and `group_size` (1 for singletons).

**Many catalogues.** `crossmatch_many` returns one output row per group and an
`index[g, c]` row of catalogue `c` (`-1` if absent).

- `mode="pairs"`: each other catalogue is matched to `catalogs[reference]`
  one-to-one (`symmetric`). Rows are all reference rows, then each other
  catalogue's reference-unmatched rows alone. Two non-reference rows are never
  linked directly.
- `mode="group"`: friends-of-friends over all cross-catalogue pairs (same-catalogue
  pairs are not links), one row per group ordered by its smallest stacked row. A
  chain may hold several rows of one catalogue; the representative is the one
  with the smallest separation to any linked row of another catalogue, ties by
  lowest row index. `group_size` counts every member.

<!-- docs: run -->
```python
import torch

from torchsky.catalogs import (
    crossmatch_exact,
    crossmatch_many,
    crossmatch_self,
    crossmatch_sky,
    join_rows,
)

d = torch.float64
ra1 = torch.tensor([10.0, 10.0, 359.9999, 50.0], dtype=d)
ra2 = torch.tensor([10.0002, 0.0001, 80.0], dtype=d)
z1, z2 = torch.zeros(4, dtype=d), torch.zeros(3, dtype=d)

sym = crossmatch_sky(ra1, z1, ra2, z2, radius_arcsec=2.0, find="symmetric")
assert sym.left_index.tolist() == [0, 2] and sym.right_index.tolist() == [0, 1]
left, right = join_rows(sym, 4, 3, "1or2")
assert left.tolist() == [0, 2, 1, 3, -1] and right.tolist() == [0, 1, -1, -1, 2]

# The three pairs are 0.72" apart: 0.5" + 0.3" accepts them, 0.3" + 0.3" does not.
kw = dict(find="all")
near = crossmatch_sky(ra1, z1, ra2, z2, left_err_arcsec=0.5, right_err_arcsec=0.3, **kw)
assert near.left_index.tolist() == [0, 1, 2]
far = crossmatch_sky(ra1, z1, ra2, z2, left_err_arcsec=0.3, right_err_arcsec=0.3, **kw)
assert far.left_index.numel() == 0

own = crossmatch_self(ra1, z1, 2.0, group=True)
assert own.group_id.tolist() == [0, 0, -1, -1] and own.group_size.tolist() == [2, 2, 1, 1]

keys = crossmatch_exact(torch.tensor([1, 2, 2]), torch.tensor([2, 3]))
assert keys.left_index.tolist() == [1, 2] and keys.right_index.tolist() == [0, 0]

pairs = crossmatch_many([(ra1, z1), (ra2, z2)], 2.0, mode="pairs")
assert pairs.index.tolist() == [[0, 0], [1, -1], [2, 1], [3, -1], [-1, 2]]
groups = crossmatch_many([(ra1, z1), (ra2, z2)], 2.0, mode="group")
assert groups.index.tolist() == [[0, 0], [2, 1], [3, -1], [-1, 2]]
assert groups.group_size.tolist() == [3, 2, 1, 1]
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
# fragment: needs pyarrow tables `left` and `right`
from torchsky.catalogs import crossmatch_sky_from_arrow

pairs = crossmatch_sky_from_arrow(left, right, radius_arcsec=2.0, find="all")
left_ids = left["source_id"].take(pairs.left_index.numpy())
right_ids = right["source_id"].take(pairs.right_index.numpy())
```

`left_columns` / `right_columns` name the `(ra, dec)` columns when they are not
`("ra", "dec")`. `pyarrow` must be installed; it is not a default `torchsky`
runtime dependency. The unified `crossmatch(..., streamed=True)` form expects an
iterable of `(ra, dec)` batches, not an Arrow table.

## Frames, epochs, and motion

`Catalog.crossmatch` defaults to matching positions as stored. For observations
at different astrometric reference epochs, explicitly align to a common Julian
year:

```python
# fragment: continues from the catalogs above
pairs = left_catalog.crossmatch(
    right_catalog,
    radius_arcsec=2.0,
    find="all",
    target_epoch=2016.0,
    motion="angular",
)
```

Angular propagation requires both proper-motion components when the epoch
changes. Physical propagation (`motion="physical"`) additionally requires
positive parallax and radial velocity, using the existing space-motion kernel.
A catalogue already at the target epoch does not need motion imputation.
The operation preserves input row indices and payload columns and leaves the
original catalogues unchanged. A prebuilt right index cannot be supplied when
requesting frame or epoch conversion; matching builds an index on the evaluated
positions. In-memory matching also rejects indexes built from other coordinates.

`CatalogState.to_frame` rotates proper motion through Cartesian tangent vectors,
then projects into the target frame's longitude/latitude tangent basis. The
rigid ICRS, Galactic, FK5-at-J2000 and ecliptic transforms preserve motion norm
and differentiability. Ecliptic-of-date conversion is a rotation at the supplied
fixed `tt_mjd`; this does not include the rotation rate of a moving frame.
Partial proper motion is rejected during conversion. FK4 proper-motion
conversion is rejected because a complete time-dependent astrometric transform
is not implemented; position-only FK4 conversion remains available.

These APIs do not propagate catalogue covariance, model unknown proper-motion
priors, identify blends, or produce association probabilities. Use the existing
motion Jacobian functions for covariance calculations in the association layer.

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
