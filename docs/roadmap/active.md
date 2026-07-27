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
- [x] Benchmark candidates/s, intersections/s, peak memory, and numerical error
  on CPU and CUDA; profile before adding a native kernel.
  Local CPU reference at 256 sites and 256 pixels: 2,539 candidates, 931
  positive intersections, 2.405 s median, 1,056 candidates/s, 387 positive
  intersections/s, 1.07 GB peak RSS, and 0.044 square-unit error against the
  independent 256-square coverage oracle. CANFAR session `m7ln4v57`, exact
  commit `f16f939`, passed nine focused contracts plus the CUDA parity test.
  Its matched-size reference benchmarks measured 3.924 s CPU
  (647 candidates/s, 1.467 GB peak RSS) and 14.256 s H100 CUDA
  (184 candidates/s, 705,991,168 bytes peak CUDA allocation), with independent
  coverage-oracle absolute errors 0.044 and 0.0023 square units respectively.
  The different CPU/CUDA random streams are not a parity comparison; the
  focused same-input device test is. Artifacts:
  `/arc/projects/ots/torchsky-runs/20260726-foam-operator-cpu-cuda-f16f939/`.
  A local `cProfile` run attributes 77% of operator time to per-candidate cell
  clipping, including 46% in half-plane clipping; scalar device decisions make
  the exact reference unsuitable as the production CUDA renderer.
- [x] Only then check the parent roadmap's exact clipped
  power-cell/pixel-intersection box.
  Evidence: exact analytic/invariance/oracle/adjoint gates and matched
  CPU/CUDA CANFAR measurements are recorded above; the parent roadmap is
  updated without claiming production throughput.

Focused continuation:

```bash
pixi run -e test -- pytest tests/test_map_field_foam_intersection.py -q
pixi run preflight-push
```

## Checkpoint 3C — Production Batched Exact Foam Renderer

The reference establishes the mathematics but synchronizes once or more per
candidate/edge on accelerators. Production must batch the same topology and
analytic area calculation; soft ownership or fixed point sampling is not an
acceptable speed substitute.

- [x] Represent candidate polygons with bounded packed/padded storage and batch
  radical-half-plane clipping without per-candidate `.item()` or host/device
  synchronization.
  Evidence: `batched_bounded_power_pixel_operator` uses a caller-bounded padded
  vertex capacity, 4,096-candidate default chunks, CSR-padded neighbor planes,
  and batched scatter compaction. It synchronizes once for the maximum neighbor
  count, not per candidate or edge.
- [x] Batch the exact segment/circular-arc disk intersection contribution,
  preserving clockwise/counter-clockwise, tangent, zero-length, zero-radius,
  coincident, and dominated-cell behavior.
  Evidence: batched quadratic segment/disk roots and triangle/arc contributions
  match the scalar analytic oracle; a scale-aware area tolerance removes
  floating zero-area boundary ghosts in both paths.
- [x] Prove bitwise topology and tight area agreement with
  `bounded_power_pixel_operator` across randomized/adversarial CPU, MPS, and
  CUDA cases. Retain the current implementation as the correctness oracle.
  CPU and MPS randomized, coincident/tie, zero-radius, tangent, transform, and
  chunk cases pass. CANFAR session `o5ubrs02` at exact commit `d5408df` passed
  all eleven non-MPS focused contracts, including the same-input CUDA oracle;
  CPU and CUDA produced identical 3,825-entry topology, total area, and
  independent-oracle error in the matched medium benchmark.
- [x] Route pixel-major/transpose sparse assembly and both value semantics
  through the batched result; re-run adjoint and tile-chunk gates.
  Local pixel/transpose CSR, area, `matvec`, and `rmatvec` equality pass for both
  value semantics, including candidate chunk size 3 versus the 4,096 default.
  The focused CUDA parity gate passed in session `o5ubrs02`.
- [x] Benchmark eager, `torch.compile`, and—only if still justified by
  profiling—one native CPU/CUDA kernel. Record candidates/s, intersections/s,
  peak allocation, synchronization count, and crossover size.
  Eager CPU at 256 sites/256 pixels: 0.136 s median and 18,699 candidates/s,
  versus 2.164 s and 1,173 candidates/s for the matched scalar oracle, with
  identical 930-entry topology and total area. At 1,000 sites/1,024 pixels:
  0.707 s, 14,781 candidates/s, and 591 MB operator-only peak RSS under
  4,096-candidate chunks. CANFAR session `o5ubrs02` reproduced that medium case
  on identical CPU/CUDA inputs: 10,448 candidates, 3,825 intersections, and
  1,017.414935 square units exact area on both devices. CPU measured 0.721 s
  (14,486 candidates/s, 895,537,152-byte operator peak RSS); H100 CUDA measured
  0.240 s (43,535 candidates/s, 85,896,704-byte peak CUDA allocation).
  Artifacts:
  `/arc/projects/ots/torchsky-runs/20260726-foam-batched-cpu-cuda-d5408df/`.
  Whole-builder `torch.compile` at commit `1a47854` reduced the same CPU case to
  0.230 s (45,378 candidates/s) after a 423.1 s warm-up, but raised operator
  peak RSS to 1,495,138,304 bytes. H100 CUDA fell to 0.0414 s
  (252,514 candidates/s) after a 232.5 s warm-up, with 32,103,936-byte peak CUDA
  allocation and 4,489,248,768-byte process peak RSS. Both retained identical
  topology and tight area agreement. Artifacts:
  `/arc/projects/ots/torchsky-runs/20260726-foam-smooth-compile-1a47854/`.
  The steady-state gain is real, but whole-builder compilation is unsuitable
  for short CLI jobs; targeted numeric-kernel compilation remains the next gate
  before deciding whether any native kernel is justified. Targeted compilation
  at commit `8baa1b4` cut CPU warm-up/steady/RSS to 215.3 s, 0.208 s
  (50,306 candidates/s), and 1,132,908,544 bytes; H100 CUDA fell to 87.3 s
  warm-up and 0.0843 s steady (123,887 candidates/s), with 32,192,000-byte CUDA
  allocation and 2,190,807,040-byte process RSS. Artifacts:
  `/arc/projects/ots/torchsky-runs/20260726-foam-targeted-compile-8baa1b4/`.
  There are no per-candidate host synchronizations in either batched execution;
  trust-boundary validation and one maximum-neighbor scalar remain eager. Eager
  stays the short-job default, targeted compile is the long-lived compromise,
  and whole-builder compile is the maximum-throughput option. A native kernel
  is not justified before a real downstream profile exceeds these baselines.
- [x] Run small/medium/large CANFAR CPU/CUDA cases, define a production
  throughput baseline, and only then use the renderer in spectral Foam fitting.
  Evidence: sessions `o5ubrs02` and `idnlgxh7`, exact commit `d5408df`.
  Small CPU/CUDA: 2,539 candidates, 930 intersections, 0.178/0.0649 s,
  14,226/39,128 candidates/s, and equal area within 3e-14. Medium CPU/CUDA:
  10,448 candidates, 3,825 intersections, 0.721/0.240 s, and
  14,486/43,535 candidates/s with bitwise-equal area. Large CPU/CUDA:
  108,379 candidates, 51,572 intersections, 2.995/1.547 s, and
  36,192/70,051 candidates/s with bitwise-equal area. Large operator-only
  peaks were 910,188,544-byte process RSS and 84,674,560-byte CUDA allocation.
  Artifacts:
  `/arc/projects/ots/torchsky-runs/20260726-foam-batched-scale-d5408df/`.

Focused continuation:

```bash
pixi run -e test -- pytest tests/test_map_field_foam_intersection.py -q
pixi run bench-map-foam-operator
pixi run preflight-push
```

## Checkpoint 3D — Exact Inference And Differentiable Foam Training

Exact inference and differentiable geometry optimization have different
contracts. The exact pixel operator remains the reporting path; the training
surrogate must expose useful gradients across both radical and radius boundaries
and must never be described as exact area integration.

- [x] Add an explicitly named smooth bounded-power weighting primitive whose
  weights sum to a differentiable union-occupancy probability. Preserve
  `soft_bounded_power_weights` as the compact, hard-support compatibility path.
  Evidence: `smooth_bounded_power_weights` returns conditional power ownership
  times a separately testable union occupancy; its missing mass is background.
- [x] Use numerically stable log-domain support probabilities so sites, radii,
  and query coordinates receive finite gradients near support boundaries,
  including when no hard disk currently contains the query.
  Evidence: log-sigmoid membership and outside-union accumulation remain finite
  in the outside-support gradient and autograd finite-difference gates.
- [ ] Gate permutation/similarity equivariance, background occupancy, hard-limit
  convergence away from boundaries, coincident sites, float32/float64, and
  CPU/MPS/CUDA parity.
  CPU invariance, hard-limit, coincident-site, and dtype gates pass. CANFAR
  session `nc1qx6se` at exact commit `1a47854` passed the CUDA same-input gate
  within 35 focused passes; MPS remains unchecked because the locked local
  environments currently report MPS unavailable.
- [x] Gate autograd against finite differences for sites, radii, and queries;
  include a toy optimization where a radius starts outside an observed point
  and learns to include it.
  Evidence: `torch.autograd.gradcheck` covers all three geometry inputs; a
  deterministic Adam fixture grows a radius from 0.4 past an initially excluded
  observation.
- [x] Demonstrate the handoff from smooth training to exact hard inference on a
  seeded pixel fixture, reporting both surrogate loss and exact rendered error.
  Evidence: the smooth 31-square radius fit converges to 0.72 within 0.002, then
  the analytic pixel operator recovers the target disk area within 0.01 square
  units. Focused local Foam gates pass 31 tests with eight device skips.

Focused continuation:

```bash
pixi run -e test -- pytest tests/test_map_field_foam.py -q
pixi run preflight-push
```

## Checkpoint 3E — Shared-Geometry Spectral Foam

Multi-wavelength Foam must reuse one exact spatial operator. Spectral samples
or basis coefficients vary by cell and wavelength; bandpass integration remains
the existing deterministic linear response rather than rebuilding geometry per
channel.

- [x] Render arbitrary per-cell spectral samples through one
  `PowerPixelOperator` and one or many response-weight rows, preserving leading
  batch dimensions and both surface-brightness/integrated-flux semantics.
  Evidence: `render_power_spectral_samples` contracts response weights before
  one shared-geometry `matvec`; single/multi-channel and leading-batch results
  equal explicit per-channel operator application under both value semantics.
- [x] Contract compact per-cell spectral basis coefficients with sampled basis
  functions before response integration, without introducing a Foam-only
  bandpass abstraction.
  Evidence: `render_power_spectral_basis` is two Torch `einsum` contractions
  plus the same sampled-spectra path; it accepts weights from the existing
  deterministic spectral response without renormalizing them.
- [x] Gate monochromatic, top-hat, delta-line, overlapping-band, broadband/IFU
  equivalence, shape validation, exact spatial adjoints, and gradients through
  cell spectra, coefficients, and basis samples.
  Evidence: `tests/test_map_field_foam_spectral.py`; five local tests pass,
  including manual response contraction, the exact `rmatvec` adjoint, and finite
  coefficient/basis gradients.
- [x] Run CPU/CUDA matched multi-band benchmarks and compare shared-geometry
  execution against independent per-band operator construction.
  Evidence: CANFAR session `so72vztv`, exact commit `e8accfd`, six channels,
  nine spectral samples, 1,000 sites, 1,024 pixels, and identical inputs. Shared
  geometry exactly matched six independent operator outputs while reducing CPU
  median time from 2.638 to 0.527 s (5.01x) and H100 CUDA from 1.034 to
  0.244 s (4.24x). Shared CUDA peak allocation was 115,358,720 bytes. Artifacts:
  `/arc/projects/ots/torchsky-runs/20260726-foam-spectral-e8accfd/`.
- [ ] Only after those gates, build a trainable shared-geometry spectral Foam
  field and the seeded multi-band depth experiment.

Focused continuation:

```bash
pixi run -e test -- pytest tests/test_map_field_foam_spectral.py -q
pixi run preflight-push
```

## Checkpoint 3F — Trainable Shared-Geometry Spectral Foam

The trainable field reuses the gated smooth/hard split. Geometry is shared over
wavelength; cell appearance is expressed in caller-supplied spectral bases so
physical templates, learned decoders, broadband quadrature, and IFU channels
remain interoperable.

- [x] Add one trainable module containing shared 2-D sites, strictly positive
  radii, and per-cell spectral-basis coefficients without owning WCS, bandpass,
  or FITS concerns.
  Evidence: `SpectralPowerFoam`; radii use a stable inverse-softplus
  initialization plus an explicit non-negative floor, while sites and spectral
  coefficients are ordinary Torch parameters.
- [x] Render point/wavelength samples through smooth bounded ownership with
  finite joint gradients for sites, radii, coefficients, queries, and basis
  values; keep missing union mass explicitly zero/background.
  Evidence: functional equality and joint-gradient tests cover all five
  differentiable inputs; the smooth union occupancy supplies zero uncovered
  mass without a hidden diffuse component.
- [x] Render the same learned coefficients through a caller-supplied exact
  `PowerPixelOperator`, and gate equality with the functional spectral path.
  Evidence: `SpectralPowerFoam.render_exact` matches
  `render_power_spectral_basis` under both surface-brightness and
  integrated-flux semantics; five focused model tests pass.
- [x] Recover a seeded two-cell geometry/colour fixture from held-out spatial
  and spectral samples, then report exact-pixel held-out error after the hard
  inference handoff.
  Evidence: CANFAR session `sy8bl0ii`, exact commit `7e78c57`, 1,500 samples
  with a fixed modulo-five validation split and 0.002 noise. Both CPU and H100
  CUDA reproduce validation MSE `0.396 -> 4.00e-6` and exact two-band pixel
  relative RMSE `2.45e-4`, while recovering sites, radii, and spectral
  coefficients to the recorded tolerance. CPU/CUDA training times are
  0.503/2.732 s; the tiny float64 workload is launch-bound on CUDA. Peak CUDA
  allocation is 67,414,528 bytes. Artifacts:
  `/arc/projects/ots/torchsky-runs/20260726-foam-training-7e78c57/`.
- [ ] Build the larger pre-registered multi-band injection/recovery benchmark
  before making any depth claim.

Focused continuation:

```bash
pixi run -e test -- pytest tests/test_map_field_foam_model.py -q
pixi run preflight-push
```

## Checkpoint 3G — Pre-Registered Multi-Band Depth Harness

The complete protocol is frozen in
[`../../.cursor/harness/proposals/multiband-depth-demo.md`](../../.cursor/harness/proposals/multiband-depth-demo.md).
The first implementation milestone is Gate 0/M0 only. Keep source generation,
detection, matching, and catalogue scoring benchmark-local; do not add a source
extractor to the Torchsky runtime.

- [x] Add a deterministic truth catalogue for point sources and compact
  galaxies with smooth continua, lines, dropouts, close pairs, and blank/null
  scenes. Serialize scene, noise, pointing, and nuisance seeds separately.
  Evidence: `benchmarks/bench_multiband_depth.py` generates all five required
  spectral/morphology strata, a close pair, stable source identifiers, and an
  explicit blank catalogue. `SeedManifest` derives and serializes four
  independent streams; deterministic artifact and flux-conservation tests pass.
- [x] Render truth with denser spatial/spectral integration than any fitted
  method, and split complete scene/exposure identifiers into disjoint
  train/validation/test sets.
  Evidence: truth uses 4x4 subpixel integration and 129-sample normalized
  tabulated responses over smooth, line, and dropout spectra; `band_flux` is
  recomputed and gated from the dense integral. `split_identifiers` shuffles
  complete stable identifiers into deterministic, complete, pairwise-disjoint
  partitions.
- [x] Add one common Torch detector: propagated/empirical whitening, a fixed PSF
  template bank, local maxima, angular non-maximum suppression, and a threshold
  calibrated on validation nulls to 95% measured purity.
  Evidence: per-band propagated noise whitening, a unit-energy Gaussian
  template bank, positive chi statistic, deterministic local maxima/NMS, and
  tied-score-safe validation purity thresholding are benchmark-local. NMS
  accepts an explicit target-WCS pixel scale and angular radius; angular/pixel
  equivalence and null/bright-source fixtures pass.
- [x] Add deterministic maximum-cardinality one-to-one truth association within
  the pre-registered radius; count unmatched and duplicate detections as false
  positives and gate close-pair/duplicate fixtures.
  Evidence: benchmark-local augmenting-path association returns an exact
  maximum-cardinality matching with deterministic distance/index edge order.
  Close-pair, permutation-cardinality, unmatched-duplicate, completeness,
  purity, false-density, and astrometric-score tests pass.
- [ ] Report completeness, purity, false detections/area, astrometric residuals,
  forced flux/color bias, held-out detector likelihood, runtime, and peak memory.
  Image RMSE remains supporting evidence, never the depth claim.
- [ ] Implement only the smoke comparison first: weighted bilinear, square
  Drizzle, chi-square coadd detection, independent per-band Foam, and shared
  spectral Foam under matched seeds and a declared capacity/compute view.
- [ ] Gate zero-source false positives, noiseless bright-source flux/position,
  split disjointness, deterministic artifacts, shuffled-band loss of sharing
  gain, and CPU/CUDA parity.
- [ ] Run three smoke scene seeds and two noise realizations. Do not check the
  parent depth-demonstration box or claim a magnitude gain from fewer scenes.
- [ ] Only after Gate 0 passes, add joint INR/hybrid rows, the 30-scene
  publication profile, bootstrap intervals, and detector-level SMACS injections.

Focused continuation:

```bash
pixi run -e test -- pytest tests/test_multiband_depth_benchmark.py -q
```

Current Gate 0/M0 evidence:

- [x] Independent deterministic seed manifest, blank scene, required truth
  strata, stable JSON, and 4x4 subpixel truth integration with band-flux
  conservation.
- [x] Propagated per-band whitening, fixed Torch Gaussian template bank, local
  maxima, pixel-space NMS on the common target plane, and validation-only
  measured-purity threshold selection.
- [x] Noiseless bright-source position recovery and zero-valued null control.
- [x] Extend truth from band-integrated SED samples to a denser tabulated
  spectral truth integral than every fitted method.
- [x] Express NMS and association radii through the target WCS angular scale.
- [x] Add forced flux/color scoring and held-out detector likelihood before the
  reporting checkbox above can be checked.
- [x] Pass the focused CUDA parity test on CANFAR; local CPU passes and the
  locked local environment has no CUDA. Exact evidence below covers the first
  Gate 0 implementation commit; rerun after each material detector change.
- [x] H100 session `of2wd541`, exact commit `3874b9a`, passed all 10 focused
  Gate 0 tests plus the deterministic 96-square/six-band artifact. Artifacts:
  `/arc/projects/ots/torchsky-runs/20260727-multiband-gate0-3874b9a/`.
  Peak child RSS was 1,756,592 KiB on CUDA 12.9.

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

- [x] Run untiled and `256`/`512` tiled multi-exposure benchmarks on CANFAR CPU
  and CUDA where advertised.
- [x] Cover at least 512/2048-square with 6/24 exposures; attempt 4096-square only
  in the bounded tiled mode.
- [ ] Decide from repeated evidence whether a caller-scoped byte-bounded MEF
  cache is justified. Do not implement one from assumption.
- [x] Record the tile-size crossover and peak-memory reduction in
  `docs/roadmap/status.md`.

Active exact-commit runs:

- [x] CPU session `lf94h6du`, commit `7e78c57`, was attempted for the
  512/2048 matrix but failed before environment setup: an ephemeral partial
  clone could not fetch a promised Git object after the runner removed its
  temporary credential, and the image lacked its configured `gh` helper. It
  produced no performance evidence. Diagnostic artifact directory:
  `/arc/projects/ots/torchsky-runs/20260726-map-scaling-cpu-7e78c57/`.
- [x] H100 session `epn3drw0`, commit `7e78c57`, same matrix and artifacts:
  `/arc/projects/ots/torchsky-runs/20260726-map-scaling-cuda-7e78c57/`.
  Evidence: all five deterministic cases passed. At 512-square/6 exposures,
  untiled/256/512 measured 0.035/0.155/0.052 s and 34/26/34 MB peak CUDA
  allocation. At 2048-square/24 exposures, tile 256/512 measured
  6.967/1.978 s and 279/290 MB. The job manifest reports 1,765,264 KiB peak
  child RSS.
- [x] Replacement CPU session `i3lhh6tk`, exact commit `63be9b3`, ran the same
  512/2048 matrix with the required per-command Pixi environment. Artifacts:
  `/arc/projects/ots/torchsky-runs/20260727-map-scaling-cpu-v4-63be9b3/`.
  Evidence: deterministic 512-square/6-exposure untiled/256/512 times were
  0.073/0.230/0.083 s. Deterministic 2048-square/24-exposure tile 256/512 times
  were 14.073/4.335 s. The manifest reports 1,478,308 KiB peak child RSS.
- [x] Bounded 4096-square/24-exposure H100 session `r0j9aehr`, exact commit
  `63be9b3`, compares tile 256 and 512 only. Artifacts:
  `/arc/projects/ots/torchsky-runs/20260727-map-scaling-cuda-4096-v3-63be9b3/`.
  Evidence: both deterministic cases passed in 29.239/8.538 s with
  1,095/1,106 MB peak CUDA allocation. The manifest reports 1,907,664 KiB peak
  child RSS.
- [x] Record replacement-launch failures `w7m1xl20`/`vdhsc9f8` (the same
  filtered-clone credential lifetime) and `axvovt1a`/`nibxmavo` (bootstrap
  passed, but raw commands omitted the required Pixi prefix). None produced
  performance evidence; `i3lhh6tk`/`r0j9aehr` correct both launch contracts.

Continuation:

```bash
canfar logs lf94h6du
canfar logs epn3drw0
canfar logs i3lhh6tk
canfar logs r0j9aehr
canfar ps --all --json
```

Conclusion: untiled wins for 512-square products. For 24-exposure products,
tile 512 is 3.25x faster than tile 256 on CPU at 2048 square, 3.52x faster on
H100 at 2048 square, and 3.42x faster on H100 at 4096 square. At large sizes,
the measured CUDA allocation differs by only 1–4% because the benchmark retains
all input exposure tensors; output tiling does not solve input residency.
Retain 512 as the operational default tile and leave the MEF-cache box open
until repeated `SCI`/`ERR`/`DQ` reads measure bytes saved versus stale-data risk.

### SMACS J0723 six-band readiness

- [ ] Inventory the calibrated `SCI`, `ERR`, and `DQ` planes on CANFAR through
  Torchfits/Torchsky boundaries.
  Located data root: `/arc/projects/ots/jwst_smacs0723`. The prior guessed
  `/arc/projects/ots/torchsky/data/...` roots do not exist. Run the committed
  Torchfits-backed inventory against this exact root.
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
  Partial evidence: the SMACS benchmark no longer imports Astropy FITS; filter
  discovery, `SCI`/`ERR`/`DQ` reads, and generated science/weight/mask products
  now use `torchsky.get_header`/`read`/`write`, hence Torchfits. A focused MEF
  fixture gates headers, float/int plane reads, invalid-ERR/DQ masking, weights,
  and round-trip auxiliary outputs. Repeated overwrite/device/large-real-file
  stress remains open.
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
