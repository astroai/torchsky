# Catalog-scale exact match (2026-08-21)

Mac numbers are **wiring only**. Linux CANFAR is the claim machine. Do not
quote these walls vs SkyCoord or vs xmatch `fast` as leadership.

## Synthetic pair-ID

| N | layout | oracle | pair-ID | torchsky wall | oracle wall |
|---|---|---|---|---|---|
| 5e3 | sparse | unit-sphere cKDTree | true | 5.64 s | 0.22 s |
| 5e4 | sparse | unit-sphere cKDTree | true | 35.69 s | 0.31 s |
| 2e5 | dense | unit-sphere cKDTree | true | 53.82 s | 0.79 s |
| 1e4 × 1e4 | dense | xmatch `fast` best/all | true | ~1.6 s | ~0.008–0.047 s |

JSON: `catalog_scale_5k.json`, `catalog_scale_50k.json`,
`catalog_scale_200k_dense.json`, `xmatch_contract_10k.json`.

xmatch 1e4 RSS: `fast` ~151 MB, `torchsky` ~445 MB (timed_runs=1).

**1e7 × 1e7 Linux not run on this host.** CANFAR jobs: `pixi run canfar-catalog-scale`,
`pixi run canfar-catalog-scale-1e6`, `pixi run canfar-catalog-scale-1e7`
(after push). Dense 1e6 find=all on this Mac did not finish in 11 min.
1e7 sparse uses `--find best`.

## Real Gaia × 2MASS / unWISE

Not run: no Parquet/HATS roots on this host. Runner:

```
CATALOG_SCALE_LEFT=/path/gaia CATALOG_SCALE_RIGHT=/path/2mass \
  pixi run -e test -- python scripts/run_catalog_scale_real.py
```

HATS native `engine=torchsky` is unit-tested on a two-row fixture.
