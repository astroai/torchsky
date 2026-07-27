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

- [x] Reproduce the six Linux external-Drizzle failures in a CANFAR staging
  headless session using the exact commit and optional comparator environment.
  Evidence: session `vw6i462r`, commit `0b10008`, image
  `images.canfar.net/skaha/base-notebook:latest`, 6 expected and observed
  failures in 72.70 s with 1,479,056 KiB peak RSS. Durable artifacts:
  `/arc/projects/ots/torchsky-runs/20260726-drizzle-baseline-v2-0b10008/`.
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
- [x] Run the same focused gate on CANFAR staging Linux.
  Evidence: session `np228qfy`, exact commit `380ecdd`, 139 passed and one
  platform skip in 118.15 s with 2,705,668 KiB peak RSS. Durable artifacts:
  `/arc/projects/ots/torchsky-runs/20260726-drizzle-fixed-v4-380ecdd/`.
- [x] Commit the repair and update the evidence here.
  Evidence: `380ecdd`, pushed to `origin/main` after 1625 tests passed in
  `ci-local`.

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
- [x] Produce deterministic symmetric CSR (`indptr`, `indices`) on CPU and CUDA.
  Evidence: local CPU/MPS parity and determinism plus CANFAR staging session
  `dhepgojh` at `d8839ed`, where both CUDA CSR parity tests passed on an
  NVIDIA H100 NVL MIG 1g.12gb.
- [x] Gate exact equality against the dense oracle on randomized, equal-center,
  zero-radius, touching, separated, large-radius, and chunked inputs.
  Evidence: `tests/test_map_field_foam.py`, 11 passed and one CUDA skip.
- [x] Gate translation, rotation, uniform-scale, dtype, empty-input, and
  device-parity behavior.
  Evidence: CPU/MPS and float32/float64 behavior pass locally; CUDA parity
  passed in session `dhepgojh`.
- [x] Add a benchmark that records total possible pairs, candidate pairs, exact
  edges, reduction ratio, wall time, and peak device memory.
  Evidence: `benchmarks/bench_map_foam_sparse.py` and Pixi task
  `bench-map-foam-sparse`.
- [x] Run small/medium local benchmarks and large CPU/CUDA CANFAR staging runs.
  Local evidence at 100,000 sites: 4,999,950,000 total pairs, 84,647 candidates,
  25,389 exact edges, 99.9983% candidate reduction, 0.424 s median CPU wall time,
  and exact parity on a 2,000-site dense oracle. At 1,000,000 sites: 848,574
  candidates from 499,999,500,000 pairs, 253,942 exact edges, 99.99983%
  reduction, 4.36 s median, and 1.50 GiB peak process RSS. CANFAR staging CPU
  session `vet25ml3` reproduced exact
  one-million-site parity in 3.683 s median with 1.29 GiB benchmark peak RSS;
  artifacts: `/arc/projects/ots/torchsky-runs/20260726-foam-cpu-aeabf02/`.
  CUDA session `dhepgojh` at `d8839ed` passed the 2,000-site dense oracle and
  reduced 499,999,500,000 possible pairs to 847,623 candidates and 253,290
  exact edges in 0.0465 s median, using 675,155,456 bytes peak CUDA allocation.
  Artifacts:
  `/arc/projects/ots/torchsky-runs/20260726-foam-cuda-v5-d8839ed/`.
- [x] Check the parent roadmap's survey-scale overlap-index box only after the
  exact parity and CANFAR evidence are recorded.
  Evidence: local and CANFAR CPU dense-oracle parity, candidate reduction, and
  peak memory are recorded above; `docs/roadmap/next.md` is updated in the same
  evidence commit.

Focused continuation:

```bash
pixi run -e test -- pytest tests/test_map_field_foam.py -q
pixi run preflight-push
```

## Checkpoint 3 — Tile-To-Cell Candidate CSR

This is separate from cell-to-cell adjacency: detector/map tiles need a bounded
list of disks that can affect them before exact cell/pixel intersection begins.

- [x] Define tile bounds and disk support in one local tangent-plane convention.
  Evidence: `tile_disk_candidate_csr` uses `(x, y)` lower-left origin and tile
  size, `(rows, columns)` shape, row-major tile IDs, and closed support so exact
  boundary touches are conservatively retained.
- [x] Build deterministic tile-to-cell CSR without allocating `tiles × sites`.
  Evidence: the production path enumerates clipped per-disk AABB tile ranges,
  applies exact disk/AABB intersection, and sorts `(tile_id, site_id)` before
  constructing CSR. An explicit AABB-membership allocation guard fails closed.
- [x] Prove equality with brute-force disk/AABB intersection, including empty
  tiles, boundary touches, and disks spanning many tiles.
  Evidence: randomized brute-force parity plus outside, zero-radius
  edge/corner, whole-grid, empty-input, similarity-scale, determinism, limit,
  and device tests in `tests/test_map_field_foam.py`; local focused gate passed
  15 tests with 2 device skips.
- [x] Report occupancy distribution, maximum candidates per tile, reduction
  ratio, wall time, and peak memory locally and on CANFAR.
  Local million-site/1,048,576-tile evidence: 2,789,942 exact candidates from
  1,048,576,000,000 possible tests (99.999734% reduction), 2.66 mean,
  15 maximum, p50/p90/p99 = 2/5/7 candidates per tile, 0.569 s median CPU wall
  time, 1.11 GiB peak RSS, and a 2,000-site dense oracle pass. CANFAR staging
  CPU session `vet25ml3` reproduced the result
  in 0.181 s median with 1.18 GiB benchmark peak RSS; artifacts:
  `/arc/projects/ots/torchsky-runs/20260726-foam-cpu-aeabf02/`. CUDA session
  `dhepgojh` at `d8839ed` reproduced the million-site result with 2,791,311
  exact candidates, mean/max/p50/p90/p99 occupancy 2.662/14/2/5/7, a 0.0269 s
  median, 675,804,160 bytes peak CUDA allocation, and dense-oracle parity.
- [x] Check the parent roadmap's neighbor-CSR/tile-candidate box only after both
  CSR directions are gated.
  Evidence: CPU, MPS, and CUDA parity plus million-site CPU/CUDA artifacts are
  recorded above; `docs/roadmap/next.md` is updated in the same commit.

Focused continuation:

```bash
pixi run -e test -- pytest tests/test_map_field_foam.py -q
pixi run -e sphere-bench -- python benchmarks/bench_map_foam_tiles.py \
  --n-sites 1000000 --rows 1024 --columns 1024 --runs 3 \
  --oracle-sites 2000 --device cpu \
  --json-out benchmark_results/map_foam_tiles.json
pixi run preflight-push
```

## Checkpoint 3B — Exact Bounded Power-Cell/Pixel Operator

This is the next scientific implementation after both sparse index directions.
The hard renderer must integrate the intersection of a detector pixel, a
power-cell's radical half-planes, and its finite-radius disk. Sampling a few
points inside the pixel is not an exact-area implementation.

- [x] Define the sparse operator contract: candidate `(pixel, cell)` pairs,
  exact non-negative intersection areas, pixel-row CSR, cell-row transpose CSR,
  and explicit surface-brightness versus integrated-flux semantics.
  Evidence: `PowerPixelOperator` and `bounded_power_pixel_operator`, including
  pixel-integrated `matvec`/`rmatvec` semantics and sparse candidate/neighbor
  CSR inputs.
- [x] Clip local tangent-plane pixel polygons by the candidate cell's radical
  half-planes, handling equal sites, dominated coincident sites, exact boundary
  touches, empty intersections, and finite input validation.
  Evidence: deterministic convex half-plane clipping and lower-index tie
  ownership, covered by analytic partition, coincident-site, sparse-candidate,
  validation, and empty-contract tests.
- [x] Integrate the remaining polygon against the cell's circular bound with
  analytic segment/circular-arc area contributions. Keep documented adaptive
  quadrature only for curved WCS pixel footprints.
  Evidence: exact straight-edge segment/circular-arc integration; the API
  explicitly requires curved WCS edges to be adaptively tessellated to a named
  error budget and forbids reporting that approximation as analytic area.
- [x] Gate partition of area, single-cell analytic disk/rectangle cases,
  translation/rotation/uniform-scale invariance, permutation invariance,
  non-overlap, and randomized agreement with a high-accuracy independent
  numerical oracle.
  Evidence: `tests/test_map_field_foam_intersection.py`; eight CPU tests pass,
  including a 1024-square independent dense ownership oracle.
- [x] Gate the forward/transpose dot product for both value semantics and prove
  tile-chunk invariance. Hard topology may be piecewise differentiable, but
  exact inference must not be labelled as the soft training surrogate.
  Evidence: parametrized surface-brightness/integrated-flux dot products and
  one-pixel chunk invariance.
- [ ] Benchmark candidates/s, intersections/s, peak memory, and numerical error
  on CPU and CUDA; profile before adding a native kernel.
  Local CPU reference at 256 sites and 256 pixels: 2,539 candidates, 931
  positive intersections, 2.405 s median, 1,056 candidates/s, 387 positive
  intersections/s, 1.07 GB peak RSS, and 0.044 square-unit error against the
  independent 256-square coverage oracle. CUDA and CANFAR remain unchecked.
- [ ] Only then check the parent roadmap's exact clipped
  power-cell/pixel-intersection box.

Focused continuation:

```bash
pixi run -e test -- pytest tests/test_map_field_foam_intersection.py -q
pixi run preflight-push
```

## Checkpoint 4 — Absorb The Former Parallel Work

### CANFAR CUDA enablement

- [x] Diagnose the advertised staging CUDA image rather than accepting skipped
  device tests as evidence.
  Evidence: H100 NVL session `zyaylbxy` reports driver `610.43.02`; session
  `b65nwd8h` proves the prior `sphere-bench` lock installed
  `pytorch-2.10.0-cpu_mkl`, so both CUDA tests skipped and the explicit CUDA
  benchmark failed closed.
- [x] Add an isolated Linux-only `cuda-bench` Pixi environment while preserving
  the normal CPU/MPS solve.
  Evidence: a rich `linux-64-cuda` platform with CUDA 12.9 and
  `pytorch-gpu >=2.10,<2.11` resolves PyTorch 2.10.0 CUDA 12.9; the environment
  excludes the default CPU feature and retains `torchfits>=1.0.0rc4`. The first
  native build (`uuqpxybv`) then exposed a missing full-toolkit constraint and a
  CMake 4.4 `FindCUDAToolkit` failure; the lock now explicitly includes
  `cuda-nvcc 12.9` and CMake 4.3.x. Session `xv84arv5` then proved that NVCC
  alone does not install CMake's `CUDA::nvrtc` development target; the isolated
  lock now also includes `cuda-nvrtc-dev 12.9`.
- [x] Build Torchsky in `cuda-bench` on CANFAR, require
  `torch.cuda.is_available()`, pass both sparse-index CUDA parity tests, and
  record million-site CUDA benchmarks.
  Evidence: session `dhepgojh`, exact commit `d8839ed`, PyTorch 2.10.0 with
  CUDA 12.9 on an H100 NVL MIG 1g.12gb; editable native build, both CUDA tests,
  and both million-site benchmarks passed. Durable artifact:
  `/arc/projects/ots/torchsky-runs/20260726-foam-cuda-v5-d8839ed/`.

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
