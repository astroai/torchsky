# Torchsky Execution Roadmap

> Ordered engineering and science tasks after the 2026-07-22 map/INR milestone.
> This is a planning document, not a public API promise. Current shipped status
> remains in [`status.md`](status.md).

## North Star

Torchsky should provide one tensor-native observation model for classical
coaddition, continuous neural fields, and adaptive bounded power-cell fields.
The same WCS, bandpass, PSF, calibration, mask, weight, and background operators
must be used by every representation. A new method earns promotion by improving
truth recovery or operational scale under those shared operators, not by making
a visually sharper image.

## P0 — Exact And Scalable Map Substrate

These tasks protect every later INR/Foam result from operator or memory errors.

- [x] Remove the stale module-global image cache and make weighted/mean tiled
  mosaics exposure-major: each input image and its calibration/weight preparation
  are held for one exposure, reused across tiles, then released.
- [ ] Add a caller-scoped byte-bounded cache only where repeated MEF access proves
  it useful, and add a spillable/tile-local reducer so exact median and
  cross-exposure background correction do not trade bounded memory for repeated
  input reads.
- [ ] Adaptively sample curved SIP/TPV footprint boundaries when planning union
  products; gate containment across the RA=0 seam and rotated/sheared grids.
- [ ] Propagate inverse variance through resampling and PSF kernels explicitly,
  and report correlated-noise/effective-noise products where interpolation makes
  diagonal variance incomplete.
- [ ] Emit science-ready data, weight/inverse-variance, overlap/context, mask,
  and optional coverage-MOC products from `torchsky map`.
- [ ] Make saved execution plans replayable and reject changed/missing inputs by
  recorded identity rather than silently running a different job.
- [ ] Keep weighted-mean execution observation-streaming and tile-bounded; make
  the unavoidable memory cost of exact median stacking explicit.

## P1 — Full Bounded Power-Foam Representation

The shipped `foam` primitives are the exact geometry foundation, not yet a full
renderer. `ResidualTileBasisField` remains a Gaussian residual baseline and must
not be presented as Foam.

### Geometry and sparse execution

- [x] Tensor-native power distance, exact bounded-cell ownership, Čech overlap
  edges, radical-plane equations, and an explicitly labelled soft optimization
  surrogate.
- [ ] Build a survey-scale overlap index before pair testing. Validate it against
  the exact chunked Čech reference and report candidate reduction and peak memory.
- [ ] Construct per-cell neighbor CSR and tile-to-cell candidate lists without a
  mandatory graphics or CUDA-only runtime dependency.
- [ ] Implement exact clipped power-cell/pixel intersection in the local tangent
  plane, including flux-conserving area integration and an adjoint/dot-product
  gate. Use quadrature only as a documented fallback for curved WCS footprints.
- [ ] Add exact hard-cell inference and a well-defined differentiable training
  path; never conflate a soft partition with the exact renderer.

### Astronomy-native appearance model

- [ ] Share spatial sites, radii, and adjacency across wavelength while keeping
  spectral appearance parameters separate from geometry.
- [ ] Add per-cell spectral bases or compact spectral decoders evaluated through
  the existing deterministic bandpass operator; support broadband images, narrow
  bands, IFU channels, and energy bins through the same contract.
- [ ] Add within-cell detail sites with normalized soft-Voronoi interpolation for
  high-frequency spatial/spectral appearance, following the Power Foam separation
  of coarse geometry from texture without importing its view-dependent RGB model.
- [ ] Compose Foam with a diffuse INR branch so cells model compact/residual
  structure and the INR models smooth emission; add identifiability and sparsity
  constraints so the branches cannot exchange arbitrary flux.

### Optimization and uncertainty

- [ ] Initialize sites from persistent sky evidence across multiple exposures,
  not a residual from one detector image.
- [ ] Track per-cell held-out residual error and contribution EMAs; implement
  deterministic densification/pruning with optimizer-state preservation.
- [ ] Add overlap/connectivity, contribution sparsity, radius, and component
  separation losses with ablations; document every heuristic and its upgrade path.
- [ ] Produce posterior/ensemble uncertainty and coverage diagnostics for cell
  fluxes, colors, positions, radii, and diffuse-field predictions.

## P1 — State-Of-The-Art Continuous Fields

- [ ] Benchmark the existing SIREN, WIRE, Fourier, hash-grid, sky-encoder,
  sky-lambda plane/K-plane, feature-atlas, and operator-constrained families on
  equal parameter, optimizer-step, and wall-time budgets. Promote presets only
  from held-out exposure performance.
- [ ] Add footprint-aware/integrated encodings beyond fixed 2x2 pixel quadrature
  when compact-source phase tests show a measured gain.
- [ ] Make training observation-streaming and patch/tile-bounded, with mixed
  precision and `torch.compile`/CUDA paths enabled only after numerical parity.
- [ ] Strengthen heteroscedastic and robust likelihoods, calibration nuisance
  parameters, and posterior predictive checks; distinguish epistemic model error
  from detector noise.
- [ ] Add physically optional spectral structure—smooth continua, emission-line
  components, redshift/kinematic transforms, and non-negative mixtures—without
  forcing physical approximations when data support a more expressive decoder.
- [ ] Train and score hybrid diffuse-INR plus Power-Foam models against each branch
  alone, classical stacks, and explicit-grid inverse solutions.

## P1 — Multi-Wavelength Depth Demonstration

The central claim is improved inference at fixed input information, not merely a
higher-S/N detection image.

- [ ] Build a seeded injection suite with sub-threshold point sources, compact
  galaxies, diffuse color gradients, line emitters, band dropouts, PSF variation,
  dithers, masks, calibration offsets, and realistic correlated noise.
- [ ] Compare matched single-band stacks, classical multi-band detection with
  forced photometry, independent per-band INRs, a shared multiband INR, a shared
  spectral Foam, and the hybrid model under equal train/validation information.
- [ ] Reserve complete exposures and sky regions for held-out prediction; use
  blank-sky negative controls and never tune on the final test field.
- [ ] Report completeness versus intrinsic flux, purity/false detections, flux and
  color bias/scatter, morphology error, limiting depth at fixed purity, interval
  coverage, calibration residuals, runtime, and peak memory.
- [ ] Require the joint model to improve limiting depth at fixed purity while
  preserving calibrated flux/color uncertainty; publish negative results and
  source-class-specific failures.
- [ ] Reproduce the winning protocol on a real multi-band field using raw
  exposures plus independent external catalogs, not a deeper coadd as truth.

## P2 — SWarp, Drizzle, And Montage Workflow Closure

Torchsky already covers the central reprojection/coaddition algorithms. Remaining
work should target operational semantics and measurable parity, not clone every
legacy option.

- [ ] Maintain source-built official-tool comparison rows for representative
  TAN/TPV, dithered undersampling, cylindrical, and all-sky cases; keep unsupported
  comparisons as explicit skips.
- [ ] Close SWarp-style background matching, photometric scaling, clipping/combine
  behavior, weight/mask interpretation, and large-mosaic streaming where those
  workflows are scientifically useful.
- [ ] Close Drizzle context maps, pixfrac/kernel semantics, units, fill/support,
  and correlated-noise accounting on upstream-derived translated/dithered cases.
- [ ] Close Montage-style footprint planning, background rectification, plate
  aggregation, and spherical large-mosaic execution, while retaining Torchsky's
  tensor-native internals.
- [ ] Promote a parity item only when analytic truth, the external comparator,
  edge/interior diagnostics, flux conservation, and deterministic repeats agree.

## P2 — WCS, Sphere, And All-Sky Scale

- [ ] Finish named Paper III/IV and frame-precision gaps with concrete upstream
  fixtures and device-parity tests.
- [ ] Generalize map products across tangent patches, HEALPix/MOC partitions, and
  sparse all-sky outputs without densifying the sky.
- [ ] Route Foam/INR queries through patch-local tangent coordinates with explicit
  seam/overlap blending and prove equivalence on shared patch boundaries.
- [ ] Benchmark CPU, CUDA, and MPS only where the numerical contract is supported;
  fail closed elsewhere.

## Release And Infrastructure Gates

- [ ] Keep `main` releasable: targeted test during development, then
  `pixi run ci-local` before every push.
- [ ] Add small/medium/large benchmark baselines for map streaming, sparse Foam
  candidates, Foam rendering, INR fitting, and the multi-band depth experiment.
- [ ] Record environment, commit, device, dtype, seeds, input identities, config,
  wall time, and peak memory in every benchmark artifact.
- [ ] Exercise `torchfits` at every FITS boundary and keep all FITS ownership and
  throughput work in the sibling repository.
- [ ] Validate Linux headless CPU and CUDA wheels on CANFAR before declaring a
  release candidate; retain CPU/MPS correctness on local CI.
- [ ] Promote experimental field APIs only after their truth, held-out, uncertainty,
  memory, and device gates are green.

## Immediate Task Order

1. Land and benchmark exposure-major, bounded-memory classical coaddition.
2. Implement sparse Čech candidates and exact 2D power-cell/pixel intersections.
3. Add the shared-geometry spectral Foam field through existing bandpass and
   observation operators.
4. Build the seeded multiband injection/recovery benchmark before tuning models.
5. Add Foam densification/pruning and diffuse-INR composition.
6. Run the equal-budget INR/Foam/classical ablation, then optimize measured hot
   paths and select user-facing presets.
7. Close external-tool workflow gaps in the order revealed by real survey use,
   with background matching, variance/context outputs, and large-mosaic streaming
   ahead of low-impact option breadth.
