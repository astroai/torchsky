# FITS Tables And Catalog Batches

FITS table I/O is owned by
[`torchfits.table`](https://github.com/sfabbro/torchfits). The former
`torchsky.table` implementation is not a torchsky roadmap surface.
The integration target is `torchfits 0.5.0`; verify exact scan/read signatures
against its release-candidate docs before treating the examples below as stable.

Use `torchfits.table` for:

- FITS BINTABLE/ASCII table reads and writes;
- projection and predicate pushdown;
- bounded-memory scans;
- row/column mutation;
- Arrow, Polars, DuckDB, and Pandas interchange.

Use `torchsky.catalogs` and `torchsky.sphere` for:

- sky-coordinate validation and conversion;
- spherical partitioning and indices;
- crossmatch and neighborhood queries;
- HEALPix/MOC joins and map accumulation;
- tensor-native catalog modelling.

## Stream Into Torchsky

```python
import torchfits
from torchsky.catalogs import crossmatch_sky

for batch in torchfits.table.scan(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20",
    batch_size=100_000,
):
    result = crossmatch_sky(batch["RA"], batch["DEC"], ref_ra, ref_dec)
    consume(result)
```

The batch boundary must preserve row order, null/mask semantics, units, dtype,
and device placement. `torchsky` must not depend on a particular dataframe
implementation in its core compute paths.

## Scale And Performance Policy

Catalog work is evaluated by:

- asymptotic complexity and avoidance of all-pairs materialization;
- candidate-reduction ratio for indexed searches;
- peak memory and bounded batch behavior;
- CPU/CUDA/MPS correctness where claimed;
- parity against brute force on synthetic cases and trusted real fixtures;
- throughput at small, medium, and survey-scale row counts.

FITS table decode/encode and predicate-pushdown benchmarks belong to
`torchfits`. Torchsky benchmarks begin with decoded tensor or Arrow batches.
