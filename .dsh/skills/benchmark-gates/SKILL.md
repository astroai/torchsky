---
name: benchmark-gates
description: Run torchsky benchmarks and compare against committed baseline gates — bench_all.py scopes, baseline JSONs, acceptance gates, scorecards
metadata:
  project: torchsky
---

# Benchmark gates and baselines

Methodology lives in `docs/benchmarks.md`. Tracked gate baselines under
`benchmarks/baselines/` (edit only with evidence): `map_*.json` / `map_*.md`,
`spin_matrix_extended_baseline.json`, `sphere_geom_exact_contains_gate.json`.
Other per-run baseline outputs there are gitignored. Heavy real-data artifacts
go to `data/benchmark_results/` (never committed).

## Main benches

- `pixi run bench` — `benchmarks/bench_all.py --profile user --scope all`
  (WCS + sphere, env `bench-all`)
- `pixi run bench-wcs` — WCS scope only
- `pixi run bench-sphere` — sphere scope only
- Narrow benches: `pixi run -e <env> -- python benchmarks/bench_*.py` with the
  matching feature env (`bench-all`, `sphere-bench`, `map-compat`,
  `bench-legacy`, `sht-bench`).

## Acceptance gates

- `pixi run wcs-sphere-acceptance` — quick WCS + sphere orchestrator, WCS/sphere
  pytest globs, spin-matrix smoke (local Mac smoke).
- `pixi run wcs-sphere-acceptance-full` — full Linux/CANFAR variant
  (`--no-gpu --no-legacy-wcs-bridge`).
- `pixi run bench-map-acceptance-smoke` — the four map smoke gates vs
  `benchmarks/baselines/map_*_smoke.json`.
- `pixi run bench-map-acceptance` — the reproducible (rep) map gates vs
  `benchmarks/baselines/map_*_rep.json`.
- Single gate example: `pixi run bench-map-ground-truth-gate` compares against
  `benchmarks/baselines/map_ground_truth_smoke.json`.

## Recording results

- Ephemeral full runs write under `benchmarks_results/<run_id>/` (gitignored).
- Commit a compact diff-friendly snapshot after a representative sweep:
  `pixi run bench-archive-scorecard -- --label <name> --runs wcs=<id>,sphere=<id>`
  → `benchmarks/scorecards/`.
- Record comparator versions with `pixi run bench-comparator-versions-all`
  (writes `benchmarks_results/VERSIONS_*.json`).

## Policy (from AGENTS.md)

- Performance changes must answer: what algorithmic bottleneck is removed, why
  it scales at large N, and what the benchmark evidence is.
- Prefer algorithmic improvements → structured indices → data layout →
  chunking/streaming → selective native kernels. Benchmark small and large
  workloads; do not force a native fast path that regresses medium sizes.
- Update gate baselines deliberately and only with evidence. Sphere/catalog
  work requires explicit parity against trusted upstream behavior, and
  throughput claims report candidate reduction, not just wall time.
