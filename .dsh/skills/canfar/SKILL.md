---
name: canfar
description: CANFAR Linux job suite for torchsky — launch/fetch WCS+sphere jobs, cuHPX SHT spike, ADASS depth runner, run-id artifacts
metadata:
  project: torchsky
---

# CANFAR job suite

Linux CPU/CUDA confirmation runs as CANFAR/AstroAI headless sessions that
execute an exact pushed commit of torchsky. Full workflow is in
`docs/benchmarks.md`; do NOT reuse the ADASS/SMACS runner for WCS/sphere
domains.

## Launch

- `pixi run canfar-wcs-sphere` — `scripts/launch_canfar_wcs_sphere.sh` (mode `all`)
- `pixi run canfar-wcs-sphere-cpu` — cpu job
- `pixi run canfar-wcs-sphere-cuda` — cuda job
- `pixi run canfar-wcs-sphere-suite` — parallel spectral, sphere, sparse, spin,
  geometry, and cuHPX jobs
- Accepted modes: `cpu`, `cuda`, `all`, `suite`, `sht` (spectral + sparse +
  spin only).
- Jobs clone from a pushed GitHub SHA (`TORCHSKY_COMMIT`) — push the commit
  first; the suite no longer clones from VOS.

## Artifacts

- Each run writes `/arc/projects/ots/torchsky-runs/<run_id>/` and copies the
  same tree to `vos:sfabbro/torchsky-runs/<run_id>/`.
- Pull a finished run: `bash scripts/fetch_canfar_wcs_sphere_run.sh <run_id>`
  (needs `vcp` / `pip install vos`).
- Current Linux ingest scorecard:
  `benchmarks/scorecards/20260817_world_layer.md`.

## Rules

- `cpu`/`spin`/`geometry` jobs pass `--no-throughput-gate` (Mac-tuned healpy
  ratio thresholds must not fail the job); relative L2 and index/angle mismatch
  gates still apply. Mac pixi tasks keep the ratio gates.
- The cuHPX scalar SHT spike is a separate CUDA-only job:
  `scripts/run_canfar_cuhpx_sht_spike.sh` (also submitted via `suite`).
- ADASS depth runs use `scripts/run_canfar_adass_depth.sh` — domain-specific,
  not for WCS/sphere.

## Driving from dsh

Long CANFAR jobs are background work: launch via `pixi run canfar-wcs-sphere-*`
in a background job, then poll/fetch with `fetch_canfar_wcs_sphere_run.sh`.
The dsh headless profile (see `.dsh/README.md`) is the intended driver on a
CANFAR VM — no localhost web UI needed.
