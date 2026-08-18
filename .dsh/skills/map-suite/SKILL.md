---
name: map-suite
description: torchsky map tests and benches — map pytest exclusion from pixi test, map acceptance gates, SMACS0723 real-data repro, foam/field-fit benches
metadata:
  project: torchsky
---

# Map suite (tests + benches)

The map subsystem (image-plane coadds, mosaics, observation operators, field
fits, foam) has its own gates.

## Map tests are excluded from `pixi run test`

`pixi run test` runs `pytest -n auto --ignore-glob='tests/test_map_*.py'`. A
green default suite does NOT mean map tests pass — they run in the GitHub CI
map matrix. Run them explicitly:

- `pixi run -e test -- pytest tests/test_map_*.py`
- or via the acceptance gates below.

## Map gates

- `pixi run bench-map-acceptance-smoke` — ground-truth, multifield, field-fit,
  compat smoke gates vs `benchmarks/baselines/map_*_smoke.json`
- `pixi run bench-map-acceptance` — rep gates vs
  `benchmarks/baselines/map_*_rep.json`
- `pixi run bench-map-compat-external` — optical/IR stacker comparators
  (reproject, drizzle, Montage, SWarp, casutools). Bootstrap external tools
  first: `pixi run bootstrap-map-compat-tools` (or `--tool swarp|montage|casutools`).
- Foam / operator / spectral / training: `pixi run bench-map-foam-*`; field
  fit: `pixi run bench-map-field-fit`; projection/mosaic and TAN-SIP suites:
  `pixi run bench-map-projection-mosaic-suite` and `pixi run bench-map-tan-sip-suite`.

## Real-data repro (SMACS0723)

`benchmarks/bench_map_smacs0723_*.py` reproduce field-fit / stacker / ladder /
joint results on real JWST SMACS0723 data. Setup scripts in `scripts/`:
`download_jwst_smacs0723_mast.py`, `download_smacs0723_multiband_public.py`,
`build_smacs0723_stack_baseline.py`, `smacs0723_full_mosaic_pipeline.py`,
`visualize_smacs0723_products.py`, `download_updec_webb.py`. Heavy artifacts go
to `data/benchmark_results/` (not committed). CFHT Megacam MEF mosaics:
`scripts/bench_cfht_mef_mosaic.py`.
