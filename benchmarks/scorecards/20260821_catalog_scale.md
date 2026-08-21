# Catalog-scale exact match (2026-08-21)

Mac numbers are **wiring only**. Linux CANFAR is the claim machine. Do not
quote these walls vs SkyCoord as leadership. Pair-ID vs unit-sphere cKDTree
(`--engine-fast`) is the correctness gate.

## Synthetic pair-ID (batched neighbour rings)

After replacing the Python-per-pixel covering driver with batched nested
rings + aligned gather. Same-machine Mac wiring:

| N | layout | oracle | pair-ID | torchsky wall | oracle wall |
|---|---|---|---|---|---|
| 2e4 | sparse | unit-sphere cKDTree | true | 0.058 s | 0.134 s |
| 5e4 | sparse | unit-sphere cKDTree | true | 0.140 s | 0.145 s |
| 2e4 | dense | unit-sphere cKDTree | true | 0.082 s | 0.139 s |
| 2e5 | dense | unit-sphere cKDTree | true | 2.88 s | 0.23 s |

The previous Python-per-pixel driver on this host was 35.7 s at 5e4 sparse
(~250× slower than the batched path above). Linux 1e6/1e7 of that driver
are not performance claims.

JSON: `catalog_scale_20k.json`, `catalog_scale_50k.json`,
`catalog_scale_20k_dense.json`, `catalog_scale_200k_dense.json`.

**1e7 × 1e7 Linux** still needs a CANFAR rerun of this kernel
(`pixi run canfar-catalog-scale-1e7` after push). 1e7 sparse uses `--find best`.

## Real Gaia × 2MASS / unWISE

Not run: no Parquet/HATS roots on this host. Runner:

```
CATALOG_SCALE_LEFT=/path/gaia CATALOG_SCALE_RIGHT=/path/2mass \
  pixi run -e test -- python scripts/run_catalog_scale_real.py
```

HATS native `engine=torchsky` is unit-tested on a two-row fixture.
