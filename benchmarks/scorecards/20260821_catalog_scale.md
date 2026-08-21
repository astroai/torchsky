# Catalog-scale exact match (2026-08-21)

Mac numbers are **wiring only**. Linux CANFAR (`ebe8f8ed`) is the claim
machine. Pair-ID vs unit-sphere cKDTree (`--engine-fast`). Do not quote vs
SkyCoord.

## Linux synthetic pair-ID (batched neighbour rings)

| N | layout | find | pair-ID | torchsky | cKDTree `fast` | RSS |
|---|---|---|---|---|---|---|
| 1e6 | sparse | best | true | 5.84 s | 1.67 s | 0.95 GB |
| 1e6 | dense | best | true | 45.8 s | 1.04 s | 2.70 GB |
| 1e7 | sparse | best | true | 43.9 s | 12.4 s | 5.28 GB |

Runs: `catalog-scale-1e6-20260821-225017`, `catalog-scale-1e7-20260821-225023`
(SHA `ebe8f8ed`). Later covering uses compacted `r+2ρ` neighbour rings; pair-ID
tests still pass. Do not mix those walls with this SHA without a rerun.

JSON: `catalog_scale_1e6_sparse_best.json`, `catalog_scale_1e6_dense_best.json`,
`catalog_scale_1e7_sparse_best.json`.

## Mac wiring (same kernel)

| N | layout | pair-ID | torchsky | cKDTree |
|---|---|---|---|---|
| 2e4 | sparse | true | 0.058 s | 0.134 s |
| 5e4 | sparse | true | 0.140 s | 0.145 s |
| 2e5 | dense | true | 2.88 s | 0.23 s |

## Real Gaia × 2MASS / unWISE

`pixi run canfar-catalog-scale-real` scans `/arc/projects/hats` (or
`CATALOG_SCALE_LEFT` / `CATALOG_SCALE_RIGHT`). Not yet ingested.
