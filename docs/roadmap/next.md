# Torchsky Execution Roadmap

## Accepted execution plan — 2026-09-05

This section supersedes earlier scheduling, branch, calibration-default, and
model-selection instructions below. Historical results retain their original
code identities; a new science run records a new exact pushed commit.

### Consolidate the fork first

- Preserve the three pending roadmap/sprint edits before switching branches.
- Reconcile all local and remote branches on `main` in `sfabbro/torchsky`:
  current and remote SMACS, WCS integration, SVO caching, field research, and
  the stale UPDC history. Preserve relevant unique work without restoring
  obsolete parallel implementations or FITS ownership in Torchsky.
- Track and push `origin/main`; do not force-push, push upstream, or delete
  branch references as part of consolidation. Record starting commits and
  conflict decisions in `active.md`.
- Repair the inconsistent lint job and spectral zero-weight test failures.
  Confirm WCS, sphere, and matching behavior after integration. Document the
  limits of finite-resolution native exact-region operations, matching epoch
  and validity handling, and device coverage.

### Shared scientific assumptions

All methods predict the same calibrated native detector observations through
the same independently calibrated response. Learn a static celestial
`I_nu(position, wavelength)` scene, including persistent intracluster light.
Flag moving or variable sources. A joint INR couples position and spectrum;
Foam combines localized spatial components with spectral coefficients. Hybrid
component images remain diagnostic until component-specific injections support
physical interpretation. Six measured band integrals do not establish a unique
fine spectrum.

| Component | Initial treatment | Controlled extensions |
| --- | --- | --- |
| Passbands | Measured tabulated photon throughput with the actual calibration reference spectrum; fixed across methods. | Throughput corrections need independent calibration. |
| PSF and pixels | Independently calibrated effective PSFs, frozen during scene comparison; explicitly record whether pixel response is included and integrate pixels exactly once. | Spatial, chromatic, and temporal response variation require calibration support. |
| Brightness | Predict calibrated SCI, normally MJy/sr; preserve zero points and pixel-area conventions. | Independently fitted relative exposure scaling with an explicit reference. |
| Background | Persistent celestial light stays in the scene. Detector offsets act in detector coordinates; optical foregrounds pass through the optical response. | Independently calibrated plane/neural backgrounds, frozen across scene representations. |
| Noise | Gaussian likelihood using valid calibrated error/variance planes and identical masks; covariance only where independently supported. | Separate calibration-uncertainty propagation; no free variance inflation to hide residuals. |

For a flat-`I_nu` calibration and dimensionless photon throughput `T(lambda)`,
the normalized band weights are proportional to `T(lambda)/lambda`. Verify
against the input product convention and otherwise use its reference-spectrum
normalization. Keep throughput shape, absolute calibration, and numerical
scaling separate. Do not multiply calibrated MJy/sr by gain and call it counts.

Fit numerical normalization on training data only, scale SCI/ERR/VAR
consistently, and reverse it on export without subtracting an unexplained
diffuse pedestal. Relative background offsets have weighted zero mean; this
does not determine absolute sky brightness. Use an independent foreground
reference or explicitly report brightness relative to the adopted mean
foreground. Calibration for held-out observations comes from independent
evidence or training-based predictions, never unrestricted fits to their
science residuals.

Replace misleading nominal defaults: Gaussian PSFs are approximation controls,
summary-shaped passbands are not measured curves, and a plane option must
actually apply a plane. Each run identifies fixed, independently fitted, and
scene-learned quantities. Temporal dependence belongs to instrument/background
terms initially, not an unrestricted astronomical position/wavelength/time field.

### Ordered implementation

1. Recover actual ARC log contents and products; verify all 135 aligned detector
   inputs, UPDC references, calibration identities, and code versions before
   restarting jobs.
2. Preserve WCS, units, passbands, PSF convention, masks, variance, and background
   metadata through fitting, rendering, normalization, save/resume, replay, and
   export. Fix sampling inconsistencies and invalid spectral integration.
3. Finish PSF-aware patches with sufficient margins, streamed backward
   evaluation, and recomputation for full-observation noise treatment. Verify
   tiled/full predictions and gradients. FITS subset/streaming writing belongs
   in Torchfits; core computation remains tensor-native.
4. Produce independent official Drizzle, SWarp, and Montage products alongside
   Torchsky on identical inputs and WCS. Check flux, support, weights, counts,
   provenance, time, and memory. Missing comparators remain incomplete.
5. Screen existing field architectures and compare explicit grids, independent
   and joint INRs, independent and joint Foam, and diffuse/Foam hybrid against
   strong classical multiband and native detector-space detection. Resolve
   Foam flux recovery before expensive runs. Compare equal total parameter and
   optimization budgets, and separately accuracy versus elapsed time.
6. Select on validation, evaluate the final test once, then refit on all
   exposures and produce six-band reconstructions and UPDC comparisons.
7. Finish general AstroDrizzle registration/background/outlier/combination,
   SWarp background/clipping/weights, and Montage overlap-background/curved
   footprint workflows. Validate beyond SMACS before claiming general parity.

Keep public interfaces thin: retain `field` and deprecated `inr`, extend
existing observation/product records, provide bounded tile iteration and the
documented AstroDrizzle workflow preset.

### Delegation and CANFAR scheduling

Use Astra-medium for most implementation and coordination, Astra-high/max for
difficult numerical work, merge conflicts and substantive review, and
Luna-xhigh for bounded routine implementation, checks, logs, documentation and
verified commits. Assign disjoint file ownership; serialize integration and
commits on the shared `main`.

Use CANFAR within available allocations: one GPU validation run, then up to six
independent single-GPU jobs by default, adapting to measured capacity and needs.
Continue independent coding/debugging/testing/reporting while jobs run. Wait
only for results that actually determine the next decision. Multi-GPU training
requires measured single-device limitations first.

### Products and continuous measurements

Every completed method produces FITS reconstruction, valid support/coverage,
available labelled uncertainty, and applicable residual/background/PSF products.
Produce matched full-field and fixed-region PNGs with common per-band scales,
linear/asinh views, signed residuals and distinct missing coverage. Legends
show method, band, units, orientation, angular scale, measured prediction error,
PSF width, flux/color bias, runtime and memory; missing values say "not measured".

Use Torchfits for FITS and RGB. Freeze stretch and channel mapping across
methods: blue averages F090W/F150W, green F200W/F277W, red F356W/F444W. Label false
color and provide native and resolution-matched comparisons. Extend the existing
visualizer; freeze display scales from the classical reference.

Track validation prediction error, flux/color recovery, PSF width/encircled
energy, diffuse recovery, noise correlations, completeness/purity, uncertainty
coverage, time and memory. Save diagnostics at 0%, 10%, 25%, 50%, 75% and completion.
Keep immutable run reports and compact history; report improved/worse/uncertain
with uncertainty intervals. Sharpness alone is not improvement; the test set
is not a development dashboard. Use plain-language reports.

### Verification and selection

Keep whole PATT groups together: training `{1,2,4,5,7}`, validation `{3,8}`,
test `{6,9}`. Split before fitting preprocessing statistics. Verify units,
surface brightness, flux, passband integration, PSF centroid/wings/normalization,
single pixel integration, applicable adjoints, tiled/full behavior and replay.
Use real data and independent point, extended, blended, dropout and diffuse
injections: at least 20 paired batches and 800 continuum sources, seven or more
flux levels, four fixed science regions, and independent truth PSFs.

Select by validation detector likelihood after scientific-fidelity checks.
Bright-source flux bias must be at most 2% and adjacent-band color bias at most
0.02 mag for the claimed continuum population. Measure depth at matched
effective resolution and false-positive density with paired intervals and
blank, negative-injection and shuffled-band controls. Published UPDC products
are secondary image/photometry/morphology references, not truth or an injected
depth competitor without rerunning the same injected inputs.

Run targeted tests, preflight and changed-signature type checks. Verify CUDA on
CANFAR; unavailable devices remain unverified. Broad superiority claims need an
independent field. Completion requires reconciled fork `main`, passing relevant
checks, reproducible six-band products/comparisons, validated general workflows,
and clear separation of measured, experimental and incomplete capabilities.

> Ordered engineering and science tasks after the 2026-07-22 map/INR milestone.
> This is the authoritative remaining-work roadmap, not a public API promise.
> The execution/evidence log lives in [`active.md`](active.md), and the
> scientific contract lives in [`../formulation_map.tex`](../formulation_map.tex).
> [`status.md`](status.md) is a historical status snapshot and must be
> re-baselined when a release is cut.
> The user-authored FOSSIL/BISOU science and primitive work is incorporated as
> a separate mission-neutral track in
> [`../projects/fossil.md`](../projects/fossil.md); its immediate ordering is
> interleaved with the active exact/bounded map-operator gates.

## Live SMACS parity recovery checkpoint — 2026-09-04

This is the durable restart point for the current work. The classical map code
under test is commit `ff2afba21efa5126978d647990ce60b6d943307f` on
`sfabbro/smacs-map-parity`; it is committed and pushed. Keep every band in this
comparison pinned to that science-code commit even if documentation-only commits
advance the branch.

The job is testing Torchsky equivalents against the actual official comparator
implementations, not comparing Torchsky modes with one another:

- Drizzle: STScI `drizzle==2.2.0` through its compiled Python API, compared with
  Torchsky exact square drops at `pixfrac=0.8`.
- SWarp: the source-built executable at
  `bf4f496f18c04a8d32022b45449ef8675ab9b3da`, using `LANCZOS3`, compared with
  Torchsky Lanczos3.
- Montage: source-built `mProject`/`mProjectPP` at
  `0fc9e5ff55ece4ee0b5d731157cf07cc31325724`, compared with Torchsky's
  conservative polygon remapper.

Each side writes independent `SCI`/`WHT` products on the same supplied target
WCS. The strict harness then compares their pixels, support, flux, exposure
count, provenance, and acceptance bounds. This is workflow parity on calibrated
aligned exposures; it is not raw JWST calibration and it is not the full
AstroDrizzle orchestration surface.

Last observed CANFAR state before this checkpoint:

| Band | Session | Persistent run | Last observed state |
| --- | --- | --- | --- |
| F277W | `lwm8klfq` | `depth-20260904-213335` | H100 proved; installing the single locked `map-compat-cuda` environment. |
| F090W | `b02dnl1p` | `depth-20260904-213526` | Session created; science stage not yet inspected. |
| F150W | `mrfyfau0` | `depth-20260904-213553` | Session created; science stage not yet inspected. |
| F200W | unresolved | launch begun after 21:35 UTC | Local client was interrupted; resolve from `canfar ps --all --json` before retrying. |
| F356W | not submitted | — | Submit only after the F277W canary clears environment and comparator setup. |
| F444W | not submitted | — | Submit only after the F277W canary clears environment and comparator setup. |

Artifacts are expected under
`/arc/projects/ots/torchsky-runs/<run-id>/mosaic-parity/b-<band>/`, including
separate official/Torchsky products plus `parity.json` and `parity.md`. None of
the active sessions had produced accepted science evidence at the time of this
checkpoint. Earlier failed attempts stopped before science execution and are not
parity results.

The classical SMACS gate is complete only when all six band rows report the
expected exposure counts (36 for F090W/F150W/F200W and 9 for
F277W/F356W/F444W), every required official comparator has status `ok`, every
acceptance row passes, every Torchsky row records `compute_device=cuda`, and the
persisted products can be reopened through Torchfits. A skipped comparator is a
failure. Until that evidence exists, Drizzle/SWarp/Montage are implemented and
locally gated but are not claimed ready for the full SMACS mosaic.

Neural-field and Foam contracts are usable for bounded experiments, but the
full-size SMACS field product, equal-budget bakeoff, calibrated uncertainty,
and scientific depth gate remain open. Do not call the system SOTA or deeper
from the classical parity run.

## Reconciliation register — 2026-08-30

This register reconciles the unchecked work in the active, project, formulation,
and archived planning documents. A row names a contiguous checklist group; all
unchecked bullets in that group have the same disposition. `implement` means
Torchsky work remains; `absorbed` means the current implementation or another
roadmap item covers it; `downstream` belongs in `torchfits`, FOSSIL, USPM, or
cosmodist; `conditional` waits for a declared profile/consumer/benchmark gate;
and `rejected` is intentionally out of scope or conflicts with the ownership
contract. No row is a scientific claim without its evidence gate.

| Source and unchecked group | Disposition | Evidence gate / next proof |
| --- | --- | --- |
| `active.md`: staged-calibration identifiability, learned-PSF covariance, CUDA conditioning profile, and public tile sink | implement / downstream | Held-out calibration score; covariance test; CUDA profile; `torchfits` subset-writer contract. |
| `active.md`: MPS/spherical/delay-axis variants and the remaining Foam blank, contracted-grid, exact-refinement, and blind-test gates | implement / conditional | Device parity where available; frozen real-exposure artifacts; activate new axes only with a FOSSIL consumer. |
| `active.md`: Checkpoint 6 `TensorSpec`, response, protected subspace, cross-fit, full-sky, and cosmology rows | implement / conditional / downstream | Shared Torchsky consumer plus adjoint/chunk/replay gates; cosmology certification remains downstream. |
| `adass_updec_depth_week.md`: independent/shared real Foam, common photometer, frozen configs, injection batches, claim gates, and submission rows | implement / conditional | Exact frozen exposure/split artifacts; official comparators; 20 paired batches and the preregistered 95% purity/depth/coverage gates. |
| `fossil.md`: typed operators, `TensorSpec`, `SpectralResponse`, protected subspaces, cross-fit/injections, full-sky/open residuals, and FOSSIL benchmark phases | implement / conditional / downstream | Implement only shared mission-neutral primitives; require adjoint, held-out, injection, memory, and consumer evidence before promotion. |
| `torchsky-uspm-cosmodist-roadmap.md`: TS-02 | implement | Cross-project adapter fixtures, units/time/provenance, device movement, and query/splat adjoint gates. |
| `torchsky-uspm-cosmodist-roadmap.md`: USPM-06–12 | downstream | USPM-owned real manifests, physical state, nuisance/calibration, predictive uncertainty, and science-case certification. |
| `torchsky-uspm-cosmodist-roadmap.md`: COSMO-01–09 and INT-01 | downstream / conditional | cosmodist-owned representation/mode/certification artifacts; Torchsky contributes only accepted shared primitives and evidence schemas. |
| `formulation_map.tex`: normative equations and 52-assumption contract | absorbed | Treat the formulation as the contract; every implementation claim must pass its operator, information, uncertainty, and held-out gates. No unchecked task list exists in the TeX source. |
| `docs/archive/*` roadmap/whitepaper plans | absorbed / conditional / rejected | Preserve as rationale/history; promote an item only when it appears in `next.md` with a current owner and gate. Legacy duplicate abstractions and FITS ownership are rejected. |

The current working-tree implementation has already absorbed the P0 map fixes
for adaptive curved footprints, exact square-drop edge support and diagonal
variance, honest optional map uncertainty/context/provenance products, MEF
writing through Torchfits, workflow/background presets, and plan-v2 identity
replay. Field plan replay now requires and content-verifies its tensor-only
artifact and renders without adding optimizer steps; changed artifacts are
rejected. These are still subject to the targeted gates recorded in `active.md`;
the large-output subset writer remains a downstream `torchfits` prerequisite.
The field trainer also has an opt-in CUDA `--field-amp` path with local
autocast/GradScaler, FP32 diagnostics, unscale-before-clipping, overflow
accounting, and resumable scaler state. Actual CUDA parity and overflow evidence
remain open because this host has no CUDA device.

## Status guardrails — 2026-09-01

The current working tree is an implementation checkpoint, not a scientific
promotion. The immediate aim is one tensor-native, bounded observation path that
can be compared fairly across classical coadds, continuous fields, and Foam
under the same WCS, passband, PSF, mask, weight, background, and uncertainty
contracts.

What is complete enough to execute:

- the map substrate has adaptive curved-footprint planning, finite-kernel
  squared-coefficient variance propagation, optional `NCORR` diagnostics,
  capacity-checked reducer scratch, plan-v2 identity/artifact replay, and
  non-skippable Drizzle/SWarp/SWarp-`CLIPPED`/Montage smoke gates;
- the field surface has tensor-only artifacts, default joint multiband fitting,
  a common constructor/device/gradient contract for every registered candidate,
  and a fast benchmark that reserves a complete exposure and removes PSF-guarded
  NE/SW sky regions before product, domain, initializer, or fit construction;
- Foam lifecycle, the combined diffuse/Foam renderer, and opt-in AMP have
  executable contracts, but their scientific promotion gates remain open.

The remaining work that can change a scientific conclusion is: FFT/spectral
uncertainty, medium/large memory scorecards, the
distorted/dithered/cylindrical/all-sky
comparator matrix, equal-parameter/step/time field bakeoffs, blank and
detector-space controls, component-specific injections, at least three scene
seeds with two noise realizations, and replication on an independent real field.
Multi-GPU DDP is conditional work after the single-device streaming and
deterministic-reduction contracts are stable; no DDP result exists yet. Large
FITS output remains downstream of a `torchfits` subset-writer/tile-export
contract.

No SOTA, depth, or production-preset claim is allowed from the current smoke
artifacts, a one-seed result, a one-field result, visual sharpness, or image
RMSE. A parent checklist stays unchecked until its complete evidence gate passes:
matched input information and budgets, held-out exposures and sky regions,
negative controls, calibrated uncertainty, deterministic repeats, and the
required CPU/CUDA/device and memory evidence. Test or blind identifiers may not
be used for selection; a null or negative result is a valid outcome.

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
- [x] Decide caller-cache policy from repeated real MEF evidence: warmed
  `SCI`/`ERR`/`DQ` reads are already 0.060–0.068 ms, so do not add a second
  caller cache.
- [x] Spill exact median and iterative sigma-clipped weighted stacks to a
  tile-local scratch store. Capacity is checked before projected contributions
  are consumed; values are reduced in bounded pixel slabs and scratch is removed
  after success or failure. Evidence: masked/even-median parity, outlier/context,
  tiled parity, scratch exhaustion, and cleanup tests.
- [x] Spill cross-exposure background correction so overlap rectification does
  not trade bounded memory for repeated input reads. The implementation now
  projects every exposure once into a capacity-checked linear-size scratch
  store, accumulates pairwise models directly into one anchored graph RHS, and
  solves/applies corrections by output tile without retaining an edge-model
  stack. Global chain-graph parity, explicit-IVAR parity, single-read behavior,
  scratch exhaustion, and cleanup tests pass. Robust polynomial and default
  mesh estimators now retain only coefficients/the filtered mesh and render
  correction windows on demand; unsupported custom estimators fail closed in
  the tiled path. The updated three-run-median four-exposure 1024-square
  scorecard records 2.565 s, 1.635 projected Mpix/s, 83,886,080 exact scratch
  bytes, and a 243,777,536-byte peak-RSS delta, with unchanged propagated-IVAR
  error.
- [x] Adaptively sample curved SIP/TPV footprint boundaries when planning union
  products; gate containment across the RA=0 seam and rotated/sheared grids.
  Evidence: recursive quarter-pixel chord-bound sampling, seam-safe buffering,
  invalid-WCS fail-closed behavior, and focused curved-edge/seam/rotation tests.
- [ ] Complete diagonal variance propagation for interpolation, conservative,
  and forward-splat kernels. Direct copy, exact square/point drizzle,
  nearest/bilinear/bicubic, Lanczos/cubic, conservative polygon, and point/box
  splat paths now propagate squared coefficients only when an explicit scalar
  or map noise weight is supplied. Evidence includes basis-coefficient oracles,
  half-pixel surface-brightness/flux cases, tiled parity, and seeded bilinear
  Monte Carlo. Distorted-WCS flux quadrature sums repeated source-pixel
  coefficients before squaring and passes real-SIP tiled parity. The gate
  remains open for FFT/spectral remappers; those paths return `None` rather than
  a proxy.
- [x] Report correlated-noise/effective-noise products where interpolation makes
  diagonal variance incomplete. Gate: local/aperture correlation diagnostics
  and a variance Monte Carlo parity fixture.
  - [x] Emit optional `NCORR` as the local first-coefficient/squared-coefficient
    ratio whenever exact propagated detector variance exists. Identity,
    half-pixel Drizzle/interpolation/polygon/splat, tile/cube, FITS metadata, and
    seeded large-aperture Monte Carlo gates pass. It remains explicitly labelled
    as a local diagnostic; exact requested apertures use the evaluator below.
  - [x] Add an exact finite-aperture evaluator for every finite-support map
    kernel. It applies the shared differentiable observation operator's adjoint,
    accumulates reused source-pixel coefficients across output tiles before
    squaring, and never materializes a map covariance matrix. Evidence covers
    all twelve finite kernels, signed apertures, flux and surface-brightness
    units, spatial inverse-variance maps, a distorted SIP source-basis oracle,
    tile-boundary covariance, and a seeded 10,000-draw Monte Carlo. Mismatched
    provenance, unsupported coverage, data-dependent backgrounds, and global
    FFT/spectral kernels fail closed.
- [x] Emit science-ready data, weight/inverse-variance, overlap/context, and
  mask planes from `torchsky map`; large outputs still require the downstream
  Torchfits subset writer.
  Evidence: MEF writer and context-overflow/provenance tests.
- [ ] Emit optional coverage-MOC products. Gate: a sparse all-sky consumer and
  RING/NEST/MOC parity fixture.
- [x] Make saved execution plans replayable and reject changed/missing inputs by
  recorded identity rather than silently running a different job.
  Evidence: plan-v2 schema and trust-boundary validation, checksum/SHA-256
  identity, auxiliary-map identity pairing, artifact-bound field replay without
  extra optimizer steps, malformed-plan rejection, and input/artifact tampering
  tests.
- [x] Keep weighted-mean execution observation-streaming and tile-bounded.
  Evidence: exposure-major execution and tiled parity tests.
- [x] Make exact median and sigma-clipped scratch requirements explicit in plan
  v2 (`5` and `12` bytes per observation-pixel respectively) and fail before
  consuming projections when free capacity is insufficient.
- [ ] Record medium/large reducer runtime, peak RSS, and scratch-usage scorecards.

## P1 — Full Bounded Power-Foam Representation

The shipped `foam` primitives are the exact geometry foundation, not yet a full
renderer. `ResidualTileBasisField` remains a Gaussian residual baseline and must
not be presented as Foam.

### Geometry and sparse execution

- [x] Tensor-native power distance, exact bounded-cell ownership, Čech overlap
  edges, radical-plane equations, and an explicitly labelled soft optimization
  surrogate.
- [x] Build a survey-scale overlap index before pair testing. Validate it against
  the exact chunked Čech reference and report candidate reduction and peak memory.
- [x] Construct per-cell neighbor CSR and tile-to-cell candidate lists without a
  mandatory graphics or CUDA-only runtime dependency.
- [x] Implement exact clipped power-cell/pixel intersection in the local tangent
  plane, including flux-conserving area integration and an adjoint/dot-product
  gate. Use quadrature only as a documented fallback for curved WCS footprints.
- [x] Add exact hard-cell inference and a well-defined differentiable training
  path; never conflate a soft partition with the exact renderer.

### Astronomy-native appearance model

- [x] Share spatial sites, radii, and adjacency across wavelength while keeping
  spectral appearance parameters separate from geometry.
- [x] Add per-cell spectral bases or compact spectral decoders evaluated through
  the existing deterministic bandpass operator; support broadband images, narrow
  bands, IFU channels, and energy bins through the same contract.
- [ ] Add within-cell detail sites with normalized soft-Voronoi interpolation for
  high-frequency spatial/spectral appearance, following the Power Foam separation
  of coarse geometry from texture without importing its view-dependent RGB model.
- [ ] Compose Foam with a diffuse INR branch so cells model compact/residual
  structure and the INR models smooth emission; add identifiability and sparsity
  constraints so the branches cannot exchange arbitrary flux.
  - [x] Implement the experimental combined-sky model, explicit
    diffuse/Foam/joint stages, low-capacity diffuse default, non-negative
    localized Foam, declared sparsity/correlation penalties, exact hard-cell
    inference, and tensor-only artifacts. The parent gate remains open until
    component-specific injections show that the decomposition is identifiable.

### Optimization and uncertainty

- [ ] Initialize sites from persistent sky evidence across multiple exposures,
  not a residual from one detector image.
  - [x] Add the exposure-balanced, order-invariant core initializer over
    normalized sky coordinates and whitened residuals. The remaining gate is
    wiring it into a frozen production recipe and held-out injection benchmark.
- [x] Track per-cell held-out residual error, contribution, and residual-position
  EMAs; implement deterministic densification/pruning with survivor optimizer
  state preserved and new-child moments zeroed. Evidence: focused model tests,
  tensor-only artifact round-trip after topology change, and resumed Adam step.
- [ ] Add overlap/connectivity, contribution sparsity, radius, and component
  separation losses with ablations; document every heuristic and its upgrade path.
- [ ] Produce posterior/ensemble uncertainty and coverage diagnostics for cell
  fluxes, colors, positions, radii, and diffuse-field predictions.
  - [x] Add bounded-memory seeded ensemble rendering with separate detector,
    epistemic, and independently seeded calibration-draw variance components.
    Per-cell posterior coverage and real held-out coverage remain open.

## P1 — State-Of-The-Art Continuous Fields

- [x] Replace the INR-only product surface with the `field` method (retain
  `inr` as a deprecated alias), field presets, and tensor-only resumable field
  artifacts. Evidence: CLI/API, JSON-manifest, weights-only load, input
  identity, and split/resume-equivalence tests.
- [x] Make joint multiband fitting the default while retaining per-band fitting
  as an explicit control (`--per-band`/`--per-filter`). Evidence: the field CLI
  derives and preserves each exposure's passband by default, and two-band
  operator/CLI tests pass without collapsing exposure-specific metadata.
- [ ] Benchmark the existing SIREN, WIRE, Fourier, hash-grid, sky-encoder,
  sky-lambda plane/K-plane, feature-atlas, and operator-constrained families on
  equal parameter, optimizer-step, and wall-time budgets. Promote presets only
  from held-out exposure performance.
  - [x] Give all registered models one API/device/gradient/tensor-state contract;
    keep corrected BACON and support-capped SASNet candidates experimental until
    the equal-budget bakeoff is complete.
  - [x] Add deterministic complete-exposure holdouts to the common benchmark,
    split before product/domain/initializer construction, and record parameter,
    trainable-parameter, wall-time, split identity, and held-out detector RMSE.
    The fast contract covers SIREN, spectral Foam, the diffuse/Foam hybrid, and
    the equal-information coadd product; the full equal-budget model zoo remains
    open and opt-in because its slow candidates are not a smoke gate.
- [ ] Add footprint-aware/integrated encodings beyond fixed 2x2 pixel quadrature
  when compact-source phase tests show a measured gain.
- [ ] Make training observation-streaming and patch/tile-bounded, with mixed
  precision and `torch.compile`/CUDA paths enabled only after numerical parity.
  - [x] Add opt-in CUDA `--field-amp none|float16|bfloat16` with local
    autocast/GradScaler, FP32 diagnostics, unscale-before-clipping, skipped-step
    overflow accounting, and scaler-state artifact persistence/resume. CPU and
    MPS fail closed.
  - [ ] Run CUDA FP32-parity and overflow scorecards before enabling AMP in a
    production preset. Gate: exact loss/gradient tolerance, finite artifacts,
    and matched held-out results.
- [ ] Strengthen heteroscedastic and robust likelihoods, calibration nuisance
  parameters, and posterior predictive checks; distinguish epistemic model error
  from detector noise.
- [ ] Add physically optional spectral structure—smooth continua, emission-line
  components, redshift/kinematic transforms, and non-negative mixtures—without
  forcing physical approximations when data support a more expressive decoder.
- [ ] Train and score hybrid diffuse-INR plus Power-Foam models against each branch
  alone, classical stacks, and explicit-grid inverse solutions.
  - [x] Implement the combined-only hybrid and its staged-fit/exact-render path.
    Equal-budget held-out scoring and component-identifiability injections remain.

## P1 — Multi-Wavelength Depth Demonstration

The central claim is improved inference at fixed input information, not merely a
higher-S/N detection image.

- [ ] Build a seeded injection suite with sub-threshold point sources, compact
  galaxies, diffuse color gradients, line emitters, band dropouts, PSF variation,
  dithers, masks, calibration offsets, and realistic correlated noise.
  - [x] Add deterministic paired manifest plumbing for positive, blank,
    explicitly sign-flipped, and per-band source-permutation controls. The same
    detector injection and classical/field adapters consume every mode, and the
    shuffled control records exact permutations while preserving per-band flux
    marginals. Real frozen-batch execution and pass/fail thresholds remain open.
- [ ] Compare matched single-band stacks, classical multi-band detection with
  forced photometry, independent per-band INRs, a shared multiband INR, a shared
  spectral Foam, and the hybrid model under equal train/validation information.
- [ ] Reserve complete exposures and sky regions for held-out prediction; use
  blank-sky negative controls and never tune on the final test field.
  - [x] Implement seeded complete-exposure reservation and PSF/resampling-guarded
    sky-coordinate quadrant holdouts in the common synthetic field benchmark.
    The fast contract removes NE and SW before every initializer/operator and
    verifies zero training support in the evaluation cores. Blank controls,
    frozen final-test identities, arbitrary-region splits, and a detector-space
    classical prediction baseline remain open.
- [ ] Report completeness versus intrinsic flux, purity/false detections, flux and
  color bias/scatter, morphology error, limiting depth at fixed purity, interval
  coverage, calibration residuals, runtime, and peak memory.
- [ ] Require the joint model to improve limiting depth at fixed purity while
  preserving calibrated flux/color uncertainty; publish negative results and
  source-class-specific failures.
- [ ] Reproduce the winning protocol on a real multi-band field using raw
  exposures plus independent external catalogs, not a deeper coadd as truth.
  This field-to-field replication is required before any SOTA or depth language,
  even if the first real field has a positive interval.

## P2 — SWarp, Drizzle, And Montage Workflow Closure

Torchsky already covers the central reprojection/coaddition algorithms. Remaining
work should target operational semantics and measurable parity, not clone every
legacy option.

- [x] Expose `native|drizzle|swarp|montage` workflow presets and
  `none|per-exposure|overlap` background modes while retaining `--sky` as an
  alias. Evidence: resolved-configuration CLI tests.
- [ ] Maintain source-built official-tool comparison rows for representative
  TAN/TPV, dithered undersampling, cylindrical, and all-sky cases; keep unsupported
  comparisons as explicit skips.
  - [x] Add non-skippable, version-locked acceptance rows for upstream-derived
    Drizzle point/square translation, SWarp bilinear/Lanczos3 shifted TAN, and
    Montage conservative SIN projection. Evidence: `bench-map-drizzle-required`,
    `bench-map-swarp-required`, and `bench-map-montage-required` pass their
    committed absolute bounds; the weekly job source-builds pinned SWarp and
    Montage revisions before running them.
  - [ ] Extend the required matrix to distorted TPV/dithered, cylindrical, and
    spherical/RA-seam workloads rather than treating the three smoke fixtures as
    workflow-wide closure.
- [ ] Close SWarp-style background matching, photometric scaling, clipping/combine
  behavior, weight/mask interpretation, and large-mosaic streaming where those
  workflows are scientifically useful.
  Current evidence: `sigma_clipped` is a bounded scratch-backed iterative
  MAD/inverse-weight-noise clipped weighted mean with retained-observation
  `WHT`/`NEXP`/`CTX`. On the locked shifted-TAN fixture, complete-footprint
  bilinear and Lanczos3 data differences are at most `1.50e-5` and `6.74e-6`,
  respectively, and weight differences are below `1.84e-4`; explicit support
  IoU/full-grid diagnostics retain SWarp's border-policy disagreement. A locked
  five-exposure cosmic-ray fixture also passes official SWarp `CLIPPED` with
  `8.59e-5` maximum data difference and `1.44e-3` maximum median-normalized
  weight difference on complete support. Dithered clipping and the broader
  gain/noise/`CLIP_AMPFRAC` matrix remain open.
- [ ] Close Drizzle context maps, pixfrac/kernel semantics, units, fill/support,
  and correlated-noise accounting on upstream-derived translated/dithered cases.
  Current official translated-fixture evidence is exact for point drops and
  `1.68e-12` maximum data difference for square drops, with zero weight
  difference. Dithered context overflow, units, and correlation acceptance remain
  open.
- [ ] Close Montage-style footprint planning, background rectification, plate
  aggregation, and spherical large-mosaic execution, while retaining Torchsky's
  tensor-native internals.
  Current official SIN projection evidence is exact on data and support against
  pinned `mProjectPP`; overlap rectification and spherical/plate scale gates remain
  open.
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
- [ ] Add a conditional multi-GPU DDP runner and deterministic reduction/parity
  scorecard only after the single-device field path is streaming and bounded;
  DDP is an execution/scaling gate, not evidence of scientific improvement.
- [ ] Promote experimental field APIs only after their truth, held-out, uncertainty,
  memory, and device gates are green.

## Immediate Task Order

The detailed scientific protocol remains
[`../projects/adass_updec_depth_week.md`](../projects/adass_updec_depth_week.md).
Resume from the live checkpoint above in this order:

1. inspect the three known sessions and resolve whether the interrupted F200W
   launch created a fourth session; never submit a duplicate blindly;
2. require F277W to prove the H100/CUDA device, finish the locked environment,
   source-build the pinned comparators, and enter the real mosaic stage;
3. submit or resume the other five bands at exact science commit `ff2afba2`,
   using one GPU per band and no more than the available CANFAR quota;
4. collect `parity.json`, `parity.md`, logs, manifests, and both official and
   Torchsky products for all bands; reject missing, skipped, CPU, changed-input,
   wrong-count, or incomplete rows;
5. diagnose numerical failures by workflow and rerun only after committing and
   pushing a tested fix, with every replacement run pinned to one new exact
   science SHA;
6. after all classical rows pass, record runtime/peak-memory scorecards and cut
   the first usable single-node map checkpoint;
7. then run the frozen independent-field, joint-field, Foam, and hybrid matrix
   under equal information and compute budgets; full-size execution alone is
   not a promotion gate;
8. run paired detector injections, blank/negative/shuffled controls, calibrated
   uncertainty, and independent-field replication before any SOTA/depth claim;
9. keep DDP conditional until the single-device streaming path and artifacts
   are stable, then add deterministic one-rank/eight-GPU equivalence evidence;
10. resume FOSSIL typed spectral/component-separation primitives and broader
    survey scaling after the SMACS map checkpoint is closed.
