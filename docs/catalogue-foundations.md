# Catalogue identity and astrometric matching

`crossmatch_sky_from_arrow` returns indices in the original input row order.
Temporary spatial sorting never changes the meaning of either index. Select
native IDs from the original Arrow table with these indices; integer or string
IDs need not enter a floating-point tensor. This also applies to IPC files and
record-batch readers: the index counts rows across all input batches.

```python
from torchsky.catalogs import crossmatch_sky_from_arrow

pairs = crossmatch_sky_from_arrow(left, right, radius_arcsec=2.0, find="all")
left_ids = left["source_id"].take(pairs.left_index.numpy())
right_ids = right["source_id"].take(pairs.right_index.numpy())
```

`find="best"` returns one nearest right row per left row and resolves exact
ties by original right row index. `find="all"` retains every candidate in the
radius, ordered by left row, separation, then right row. Empty catalogues
return empty results. IPC input files are read batch by batch.

The Arrow helper keeps coordinate tiles and their right-neighbour tiles in
memory, along with pixel-to-row-group metadata and all returned pairs. It is
not a distributed output sink. Very fine partitions can produce many small
Parquet row groups; an adaptive HATS layout and streamed candidate output are
still needed for a production full-sky union.

## Epochs and proper motion

`Catalog.crossmatch` defaults to matching positions as stored. For observations
at different astrometric reference epochs, explicitly align to a common Julian
year:

```python
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
