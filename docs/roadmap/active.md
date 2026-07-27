# Active Torchsky Execution Checklist

> Started 2026-07-26 from `main` at `0b10008`.
>
> This is the durable handoff for the current implementation cycle. Update a
> checkbox in the same commit that supplies its evidence. Leave unfinished work
> unchecked and keep the exact continuation command beneath it.

## Update Contract

- `[x]` means code or evidence exists at the commit recorded beside the item.
- `[ ]` means the item is not complete, even if exploratory work has started.
- Every implementation commit updates this file before it is pushed.
- Local performance numbers are smoke evidence only. Scale claims require a
  CANFAR staging headless artifact with commit, image, device, dtype, seed,
  command, wall time, and peak memory.
- `pixi run ci-local` is mandatory before every push. GitHub Comprehensive CI
  must also be green before the release/infrastructure box is checked.

## Checkpoint 0 — Freeze The Plan And Baseline

- [x] Audit repository state and freeze the starting commit.
  Evidence: clean `main` at `0b10008`.
- [x] Inspect the last GitHub Comprehensive CI instead of assuming the local
  pre-push gate was sufficient.
  Evidence: run
  [29942943655](https://github.com/sfabbro/torchsky/actions/runs/29942943655)
  failed only the Linux `Map mosaic (integration)` job; 6 optional external
  `drizzle_square` parity cases failed at target boundaries while the main
  build/test, code-quality, and other map shards passed.
- [x] Confirm the remote execution target before creating work.
  Evidence: the active CANFAR client server is `staging` at
  `https://staging.canfar.net/skaha`; existing sessions are user work and must
  not be reused or deleted.
- [x] Commit this checklist before implementation.
  Evidence: the commit containing this file, verified with
  `pixi run preflight-push`.
- [x] Push this checklist before implementation.
  Evidence: `23c2008`; pre-push gate passed with 1617 tests and the commit is
  on `origin/main`.

## Checkpoint 1 — Restore Releasable Drizzle Semantics

The current square-drop culling conflates an implementation tile boundary with
the science product's outer boundary. Finite drops need halo support across
internal tiles, but the external Drizzle contract still controls the outer
canvas. The fix must express both boundaries rather than widening outer support.

- [ ] Reproduce the six Linux external-Drizzle failures in a CANFAR staging
  headless session using the exact commit and optional comparator environment.
- [x] Add a regression that distinguishes global target boundaries from internal
  spatial-tile boundaries for rotated and shifted square drops.
  Evidence: `test_square_drizzle_tiles_keep_internal_halos_but_not_outer_canvas_halos`.
- [x] Restore external `drizzle_square` weight/data parity for surface-brightness,
  flux, weight-map, mask-map, and uniform-field cases.
  Evidence: all 27 optional external-Drizzle cases passed locally.
- [x] Preserve tiled/non-tiled equality for bilinear, flux, square Drizzle, and
  conservative polygon paths.
  Evidence: the exposure-major tiled parameter matrix and the new rotated
  square-drop regression passed.
- [x] Run the focused local gate:

  ```bash
  pixi run -e sphere-bench -- pytest \
    tests/test_map_drizzle_external_parity.py \
    tests/test_map_image_plane.py -q
  ```

  Evidence: 140 passed in 52.16 seconds.
- [ ] Run the same focused gate on CANFAR staging Linux.
- [ ] Commit the repair and update the evidence here.

## Checkpoint 2 — Sparse Čech Candidates And Neighbor CSR

The exact dense `cech_adjacency` remains the correctness oracle. The production
path will reduce pair tests with a Torch-native uniform-grid index and retain the
exact disk-overlap predicate.

- [x] Define an experimental result contract containing deterministic unique
  candidate pairs, exact overlap edges, symmetric neighbor CSR, and candidate
  reduction counters without adding a runtime dependency.
  Evidence: `SparseCechGraph` in `map.field.foam`.
- [x] Bin finite non-negative-radius disks by AABB coverage, sort
  `(bin_key, site_id)` pairs with Torch operations, and generate unique pairs
  only from shared bins.
  Evidence: `sparse_cech_graph`; the default bin width bounds each disk to at
  most four memberships while explicit widths remain allocation-guarded.
- [x] Apply the exact Čech predicate after candidate generation; never use the
  binning approximation as the final topology.
  Evidence: squared-distance overlap is applied only after unique candidates.
- [ ] Produce deterministic symmetric CSR (`indptr`, `indices`) on CPU and CUDA.
  Local CPU and MPS parity/determinism pass; CUDA remains a CANFAR gate.
- [x] Gate exact equality against the dense oracle on randomized, equal-center,
  zero-radius, touching, separated, large-radius, and chunked inputs.
  Evidence: `tests/test_map_field_foam.py`, 11 passed and one CUDA skip.
- [ ] Gate translation, rotation, uniform-scale, dtype, empty-input, and
  device-parity behavior.
  CPU/MPS and float32/float64 behavior pass; CUDA remains a CANFAR gate.
- [x] Add a benchmark that records total possible pairs, candidate pairs, exact
  edges, reduction ratio, wall time, and peak device memory.
  Evidence: `benchmarks/bench_map_foam_sparse.py` and Pixi task
  `bench-map-foam-sparse`.
- [ ] Run small/medium local benchmarks and large CPU/CUDA CANFAR staging runs.
  Local evidence at 100,000 sites: 4,999,950,000 total pairs, 84,647 candidates,
  25,389 exact edges, 99.9983% candidate reduction, 0.424 s median CPU wall time,
  and exact parity on a 2,000-site dense oracle. At 1,000,000 sites: 848,574
  candidates from 499,999,500,000 pairs, 253,942 exact edges, 99.99983%
  reduction, 4.36 s median, and 1.50 GiB peak process RSS. CANFAR remains
  pending.
- [ ] Check the parent roadmap's survey-scale overlap-index box only after the
  exact parity and CANFAR evidence are recorded.

Focused continuation:

```bash
pixi run -e test -- pytest tests/test_map_field_foam.py -q
pixi run preflight-push
```

## Checkpoint 3 — Tile-To-Cell Candidate CSR

This is separate from cell-to-cell adjacency: detector/map tiles need a bounded
list of disks that can affect them before exact cell/pixel intersection begins.

- [ ] Define tile bounds and disk support in one local tangent-plane convention.
- [ ] Build deterministic tile-to-cell CSR without allocating `tiles × sites`.
- [ ] Prove equality with brute-force disk/AABB intersection, including empty
  tiles, boundary touches, and disks spanning many tiles.
- [ ] Report occupancy distribution, maximum candidates per tile, reduction
  ratio, wall time, and peak memory locally and on CANFAR.
- [ ] Check the parent roadmap's neighbor-CSR/tile-candidate box only after both
  CSR directions are gated.

## Checkpoint 4 — Absorb The Former Parallel Work

### Exposure-major map scaling

- [ ] Run untiled and `256`/`512` tiled multi-exposure benchmarks on CANFAR CPU
  and CUDA where advertised.
- [ ] Cover at least 512/2048-square with 6/24 exposures; attempt 4096-square only
  in the bounded tiled mode.
- [ ] Decide from repeated evidence whether a caller-scoped byte-bounded MEF
  cache is justified. Do not implement one from assumption.
- [ ] Record the tile-size crossover and peak-memory reduction in
  `docs/roadmap/status.md`.

### SMACS J0723 six-band readiness

- [ ] Inventory the calibrated `SCI`, `ERR`, and `DQ` planes on CANFAR through
  Torchfits/Torchsky boundaries.
- [ ] Verify common support for F090W, F150W, F200W, F277W, F356W, and F444W
  under the pre-registered split:
  `train={1,2,4,5,7}`, `validation={3,8}`, `test={6,9}`.
- [ ] Record a header/coverage-only crop decision. Do not use science pixels to
  choose the crop and do not treat a deeper coadd as truth.
- [ ] Run the existing six-band joint-fit smoke if the staged data and image
  provide the required environment.

### Torchfits consumer stress

- [ ] Exercise repeated MEF `SCI`/`ERR`/`DQ` reads, overwrite freshness, header
  preservation, dtype/device transfer, and large-image memory through Torchsky.
- [ ] Record Torchfits defects in the sibling repository; keep FITS/cache fixes
  out of Torchsky.

## Checkpoint 5 — Close And Hand Off

- [ ] Update every checkbox above with commit IDs, artifact paths, and exact
  rerun commands.
- [ ] Update [`next.md`](next.md) and [`status.md`](status.md) only for gates that
  actually passed.
- [ ] Write the harness trajectory under `.cursor/harness/trajectories/`.
- [ ] Run `pixi run ci-local`.
- [ ] Commit and push all completed work on `main`.
- [ ] Confirm GitHub Comprehensive CI is green.
- [ ] Leave the tree clean and report:
  scientific/technical status, infrastructure status, next unchecked item, and
  explicit parallel work with repository, branch, data, and commands.

## CANFAR Staging Run Template

Do not reuse or delete unrelated sessions. Select an image only after confirming
it is advertised by staging. Every job writes its artifact to persistent `/arc`
storage and exits.

```bash
canfar server use staging
canfar image ls
canfar create headless IMAGE \
  --name torchsky-CHECKPOINT-DATE \
  --cpu CPUS --memory RAM_GB --gpu GPUS \
  -- /bin/bash /arc/projects/ots/torchsky-runs/SCRIPT.sh
canfar ps --all --json
canfar logs SESSION_ID
canfar events SESSION_ID
```

The script must clone or update Torchsky under the session's ephemeral source
directory, check out the recorded commit, use Pixi, write logs/JSON/Markdown to
`/arc/projects/ots/torchsky-runs/<run-id>/`, and record `git rev-parse HEAD`,
`pixi info`, device information, and the exact command.
