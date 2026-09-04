# Map overlap-background spill — 2026-08-31

Focused CPU scorecard for the global overlap-graph scratch path. Four aligned
1024×1024 calibrated exposures carry distinct affine backgrounds and explicit
inverse variance; graph reduction uses 256×256 tiles. The updated path fits a
compact robust polynomial and renders corrections by tile instead of retaining
one full fitted model plane.

| Metric | Full model plane | Compact/tiled model |
| --- | ---: | ---: |
| Wall time | 3.355 s | 2.565 s |
| Projected throughput | 1.250 Mpix/s | 1.635 Mpix/s |
| Peak RSS delta | 259,227,648 bytes | 243,777,536 bytes |
| Exact scratch requirement | 84,934,656 bytes | 83,886,080 bytes |
| Scratch clean after run | yes | yes |
| Scene RMSE | 1.9133e-4 | 2.3484e-6 |
| Maximum IVAR error | 2.8610e-6 | 2.8610e-6 |

Command:

```bash
pixi run -e test -- python benchmarks/bench_map_overlap_background.py \
  --size 1024 --n-observations 4 --tile 256
```

This snapshot was taken from the uncommitted overlap-spill working tree based
on `85010fda295c`; it is implementation evidence, not a release comparison.
The compact column is the per-metric median of three fresh subprocesses.
Relative to the same local fixture, compact/tiled fitting improves throughput by
about 31% and reduces the measured RSS delta by about 15 MB while preserving
scratch cleanup and propagated IVAR. Polynomial and default mesh estimators now
have compact renderers; custom estimators without that contract fail closed in
the bounded tiled path instead of silently retaining a full model plane.
