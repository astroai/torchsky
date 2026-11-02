# Catalog containers
<!-- TODO(0.5.0 docs): intro -->


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
  [STILTS-style matching](crossmatch.md#stilts-style-matching).
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
