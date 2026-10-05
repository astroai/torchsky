# Continuous Cube Reconstruction Execution Plan

## Current continuation order — 2026-09-23

Follow [the SMACS evidence-first reconstruction plan](next.md#smacs-evidence-first-reconstruction-plan--2026-09-24)
for the next work, and [verified results](active.md#verified-results-and-handoff--2026-09-23)
for evidence and run identities. These supersede the older checkpoint and sprint
ordering below; the scientific objective and fidelity requirements remain.
The precision-only experiment completed without a meaningful improvement; the
full-image experiment timed out. Begin by recovering its cost/failure evidence
and identifying the earlier good reference, then verify the observation model,
useful grid resolution, sampling objective and initialization before another
architecture sweep. No representation has passed final SMACS quality criteria.

## Objective

Produce a calibrated continuous scene estimate `s(x, y, lambda)` from the six
JWST/NIRCam bands and show that integrating it through a declared transmission
curve yields either:

1. higher effective spatial resolution at matched noise and false-positive rate,
   or
2. higher signal-to-noise at matched effective resolution and false-positive
   rate,

than the best matched per-band stack built from the same detector exposures.

A result is promotable only if held-out detector prediction improves or remains
consistent, bright-source flux bias is at most 2%, adjacent-band colour bias is
at most 0.02 mag, support and masks are identical, and uncertainty coverage is
calibrated. RGB images are required diagnostic products, not evidence of
scientific superiority.

Six broad bands do not identify an unconstrained fine spectrum. The supported
claim is a regularized continuous field in the measured passband subspace.
Transmission curves outside that span, or spectral structure finer than the
measured response permits, must be labelled extrapolative and cannot enter the
promotion gate.

## Historical checkpoint — 2026-09-18 (not live session status)

- Pushed implementation tip: `27cd94ce5fbfd8798ac7f24118451603b20c93a1`.
- GPU reconstruction canary `lxc5ook9` is running on healthy node
  `keel-prod-g0126` with an H100 MIG, 16 CPUs, and 64 GiB RAM.
- Full F277W and F090W classical parity sessions `rr4w3dib` and `po465gbo` are
  still running.
- Repeated jobs assigned to `keel-prod-g0123` failed before user code because
  NVIDIA CDI could not initialize NVML (`Driver/library version mismatch`).
- The working tree now wires the scene initializer through a temporary latent
  explicit spectral-grid fit using the active frozen observation bundles, then
  renders the result without detector operators onto one common WCS. The
  benchmark exposes `--scene-initialization common_product|none`, records common
  and model-specific initialization time, and persists physical-unit data,
  inverse variance, support, passband samples, normalization, operator
  convention, and hashes. `.freebuff/` remains unrelated and untracked.
- Focused local verification on 2026-09-17 passes 84 tests across the joint
  adapter, real-field helpers, artifact replay, and CLI contracts. Typecheck has
  zero errors and one pre-existing `__all__` warning. The synthetic nontrivial-
  PSF regression confirms that the latent estimate predicts the detector image
  better than directly reusing the blurred coadd.
- A paired real-data training pilot completed on 2026-09-18 at exact commit
  `27cd94ce`: initialized session `qoc2xsyh` / run
  `depth-20260918-000358` and no-init session `t6rpgosq` / run
  `depth-20260918-000423`. Both used the same 18 F150W/F277W detector exposures,
  64-square crop, seed 101, and 40-step budget. Initial loss fell by 79.0% for
  the explicit grid, 76.3% for SIREN, 30.5% for SASNet, and 35.4% for Foam.
  Final loss improved by 79.2%, 95.9%, and 41.4% for the explicit grid, SIREN,
  and Foam, but worsened by 51.4% for SASNet. Initialization added 31.54 s of
  common CPU work and raised per-model elapsed time by 1.49--3.20x.
- This pilot does not close the Phase 1 scientific gate. It used nominal Gaussian
  responses and no held-out split. Strict calibrated attempts failed closed
  because the archived tree has no measured-passband files and the old NRCB4
  background checkpoint lacks current metadata. The initializer therefore
  remains enabled for experimentation but is not promoted as uniformly better;
  SASNet requires a model-specific diagnosis and a newly compliant frozen-
  calibration run must supply held-out flux, colour, support, and likelihood.
- F090W science-validation exposures still lack acceptable independent PSF
  evidence. No six-band scientific claim may bypass or silently drop them.

## Execution principles

1. Use identical detector inputs, splits, masks, WCS, measured passbands, frozen
   calibration, and information budgets for every representation.
2. Keep PATT groups intact: train `{1,2,4,5,7}`, validation `{3,8}`, final test
   `{6,9}`. Fit preprocessing and thresholds on training data only.
3. Select using validation detector likelihood after flux, colour, support, PSF,
   and uncertainty checks. Evaluate the final test once.
4. Compare both equal parameter/step budgets and equal elapsed-time budgets.
5. Separate two claims: resolution gain at matched SNR and SNR gain at matched
   resolution. Do not combine them into an unqualified “better image” claim.
6. Every remote run uses production CANFAR, an exact pushed commit, immutable
   inputs, one declared device, and persistent ARC artifacts.
7. Missing comparators, unsupported calibration, missing products, NaNs, changed
   inputs, or incomplete rows fail closed.
8. Use real SMACS detector observations and the existing high-quality stacks as
   the primary architecture bakeoff. Synthetic scenes are bounded numerical and
   failure-mode checks, not the model-ranking arena. Existing stacks remain
   references rather than truth, so paired injections into real detector data
   are still required for resolution, SNR, completeness, and bias claims.

## Phase 0 — Recover and classify current jobs

### Tasks

- Monitor `lxc5ook9`, `rr4w3dib`, and `po465gbo` without launching duplicates.
- Recover each session log, exit status, JSON/Markdown report, manifests, FITS
  products, and hashes from `/arc/projects/ots/torchsky-runs/<run-id>/`.
- Record failed `g0123` sessions as infrastructure failures only; retain the
  verbatim OCI/CDI event and do not treat them as Torchsky measurements.
- If a job completes, validate artifact schema, exact commit, device, input
  identity, exposure count, finite values, and Torchfits reopen before review.

### Gate

No result enters later comparisons until its persistent artifact is complete and
hash-verifiable. Session state or visually plausible output is insufficient.

## Phase 1 — Finish the common scene initializer

### Scope

Complete the existing helpers in
`benchmarks/bench_map_smacs0723_field_fit_joint.py`:

- `_training_scene_initialization_product`
- `_initialize_joint_scene`

### Tasks

1. Write failing tests for order invariance, train-only construction, exclusion
   of calibration and held-out pixels, detector-background subtraction, common
   WCS alignment, band ordering, support propagation, and zero-weight handling.
2. Verify units: the initializer must estimate the same calibrated
   surface-brightness quantity predicted by the scene model.
3. Use active frozen operator bundles and their pixel-response convention; do
   not apply detector response twice.
4. Wire the initializer into explicit-grid, SIREN/SASNet, Foam, and hybrid arms
   through one declared initialization policy.
5. Include initialization cost in elapsed-time comparisons and persist the
   initialization product, support, hashes, and diagnostics.
6. Add a no-initialization control. Initialization must improve convergence
   without changing the final objective or leaking validation/test data.

### Gate

On synthetic and real training crops, initialized models must start with lower
training residual without worsening held-out likelihood, flux bias, colour bias,
or support. The initializer must reproduce under shuffled observation order.

### Commit boundary

One commit containing tests, initializer wiring, artifact metadata, and roadmap
update. Run targeted tests, `pixi run preflight-push`, and `pixi run typecheck`
before push.

## Phase 2 — Make transmission-integrated products first-class

### Scope

Use the existing `MapBandpassSpec`, `Passband`, and field product renderer rather
than introducing a parallel spectral API.

### Tasks

1. Add a benchmark-local product specification for arbitrary tabulated
   transmission curves with explicit wavelength units and photon/energy
   convention.
2. Render the shared field through each measured NIRCam passband and through
   predeclared synthetic/intermediate transmission curves.
3. Render the independent per-band control only where its information supports
   the requested curve. Interpolation between independent bands must be explicit
   and separately labelled.
4. Verify constant-spectrum, delta/narrow-line, linear-spectrum, and zero-support
   analytic cases; test gradients and device parity.
5. Record the effective spectral response and warn or fail when a requested
   transmission curve is outside the measured passband span.
6. Persist FITS products, support, uncertainty, passband samples, normalization,
   and provenance through Torchfits.

### Gate

For analytic spectra, band-integrated flux must agree with an independent dense
quadrature oracle within the predeclared numerical tolerance. CPU and CUDA must
agree within dtype-appropriate bounds.

### Commit boundary

One commit for transmission-product behavior and tests; no visualization changes
in this commit.

## Phase 3 — Add reproducible RGB and matched visual diagnostics

### Scope

Extend `scripts/visualize_smacs0723_products.py` and reuse Torchfits-backed
products. Do not use RGB appearance as a selection metric.

### Tasks

1. Produce native and resolution-matched false-colour images using the frozen
   mapping:
   - blue: mean of F090W and F150W;
   - green: mean of F200W and F277W;
   - red: mean of F356W and F444W.
2. Freeze linear and asinh display limits from the classical reference and reuse
   them for all methods.
3. Emit shared-field, independent-field, Foam, hybrid, and classical RGBs with
   common crop, WCS, orientation, angular scale, and missing-coverage alpha.
4. Emit signed residual panels and per-band prediction/reference/support images.
5. Include method, units, effective PSF, flux/colour bias, held-out NLL, runtime,
   memory, commit, and input hashes in legends or sidecar metadata.
6. Add image-dimension, channel-mapping, common-scale, support-alpha, and
   deterministic-byte/hash tests where stable.

### Gate

Every six-band candidate and classical reference produces reopenable FITS and
matched PNG/RGB outputs. Resolution-matched comparisons must use a declared
common target PSF.

### Commit boundary

One visualization-only commit with focused tests and no scientific ranking.

## Phase 4 — Repair and freeze calibration evidence

### Tasks

1. Re-extract PSF evidence with the full extraction-footprint blend rejection.
2. Preserve `pixel_response_included` and provenance through evidence manifests,
   checkpoint schema v2, loading, rendering, and reports.
3. Inspect all supported exposure/filter PSFs, not only detector averages.
4. Resolve F090W validation coverage using additional acceptable stars or an
   independently validated exposure-conditioned prediction. Do not use the
   contaminated kernels or a detector-average fallback by convenience.
5. Fit stationary per-exposure/filter PSF and detector-background controls.
6. Compare stationary and adaptive calibration only where coordinate coverage
   supports the adaptive model.
7. Freeze calibration before any scene-model comparison; record exclusions and
   unsupported observations explicitly.

### Gate

Every training, validation, and test exposure used by a scene run has independent
frozen calibration with acceptable centroid, normalization, encircled-flux, and
held-out residual checks. F090W must pass or the six-band campaign remains
blocked.

### CANFAR schedule

- Use one bounded calibration canary first.
- Fan out by exposure/filter after the canary passes.
- Request 16 CPU / 64 GiB / one GPU when needed to avoid the known broken-node
  scheduling pattern; record actual assigned node.
- Never convert an infrastructure failure into a calibration exclusion.

## Phase 5 — Close six-band classical parity

### Tasks

1. Recover F277W and F090W results when complete.
2. After `g0123` repair or healthy placement is available, rerun
   F150W/F200W/F356W/F444W from one exact commit.
3. Require official Drizzle, SWarp, and Montage plus corresponding Torchsky
   methods on identical inputs and target WCS.
4. Validate expected exposure counts: 36 for F090W/F150W/F200W and 9 for
   F277W/F356W/F444W.
5. Check flux, support, weights, count maps, runtime, peak memory, provenance,
   and Torchfits reopen.
6. Produce per-band products and frozen-scale six-band RGB diagnostics.

### Gate

All official comparator statuses are `ok`; every acceptance row passes; every
Torchsky row records CUDA; no method is skipped; all products are persistent and
hash-verified.

### Commit boundary

Commit only tested numerical fixes. Evidence-only roadmap updates use a separate
commit after artifacts are recovered and checked.

## Phase 6 — Primary real-data architecture bakeoff

### Tasks

1. Use the frozen train/validation/test split and calibration from Phase 4.
2. Compare explicit grids, selected independent/joint INRs, independent/shared
   Foam, and diffuse/Foam hybrid under both budget regimes.
3. Use identical target patches, guards, masks, passbands, PSFs, backgrounds,
   and likelihoods.
4. Include the best available per-band stacks as strong practical baselines and
   render every candidate onto their common WCS at native and matched resolution.
5. Score held-out native detector observations first; use stack differences and
   RGBs as diagnostics rather than treating a stack as scene truth.
6. Save diagnostics at 0%, 10%, 25%, 50%, 75%, and completion.
7. Select architecture and hyperparameters from validation only.
8. Reject models with better image appearance but worse detector likelihood,
   source attenuation, flux/colour bias, or support.
9. Run two budget regimes: equal trainable parameters/optimizer steps and equal
   end-to-end elapsed time including initialization.

### Gate

Choose at most one primary shared representation and one independent control.
Selection requires stable rankings across validation regions and seeds, not one
best image.

## Phase 7 — Targeted synthetic invariants and failure controls

### Existing entry point

`benchmarks/bench_multiband_depth.py --profile classical-smoke --include-foam`
emits the explicit `shared_vs_independent` gate.

### Scope

Do not run a broad synthetic architecture tournament. Use small deterministic
cases to falsify the real-data finalists and verify behavior that real stacks
cannot identify uniquely:

- constant, linear, narrow-feature, and zero spectra through declared passbands;
- noiseless point/extended source flux and position recovery;
- zero-source, negative-source, and shuffled-band controls;
- PSF, pixel-integration, support-edge, and WCS perturbation cases;
- one shared-field and one independent-field control under identical budgets;
- CPU/CUDA parity and deterministic artifact replay.

### Gate

Synthetic checks may reject a finalist but may not promote one over a real-data
competitor. All analytic invariants must pass, shuffled sharing must lose its
advantage, blanks must not create sources, and flux/colour limits must hold. A
synthetic advantage without real held-out detector improvement is ignored.

## Phase 8 — Paired real detector injections

### Tasks

1. Run at least 20 paired batches and 800 continuum sources over seven or more
   flux levels and four fixed science regions.
2. Include point, extended, blended, dropout, diffuse, blank, negative, and
   shuffled-band controls.
3. Use independent truth PSFs and preserve common support.
4. Fit detection thresholds on validation nulls at 95% purity.
5. Bootstrap paired differences by independent batch/PATT group.
6. Compare transmission-integrated cube products with the best per-band stack at:
   - matched effective resolution for SNR/depth;
   - matched SNR for effective resolution.
7. Measure uncertainty coverage separately from reconstruction error.

### Gate

The 95% paired interval must support at least one declared gain while all fidelity
limits pass. If intervals overlap zero, report the result as uncertain rather
than tuning on the final test.

## Phase 9 — Final test, refit, and six-band products

### Tasks

1. Freeze architecture, calibration, preprocessing, transmission curves,
   thresholds, display settings, and resource budget.
2. Evaluate the final test split once.
3. If the frozen gate passes, refit the selected model on all eligible exposures
   without changing settings.
4. Produce six measured-band FITS products, requested transmission-integrated
   products, support, uncertainty, residuals, PSF diagnostics, and native plus
   resolution-matched RGBs.
5. Compare with official classical products and published UPDC products using the
   correct role for each reference.
6. Publish a compact report separating measured improvements, null results,
   infrastructure failures, unsupported calibration, and extrapolative products.

### Gate

No depth/SOTA statement until the final test and all product/provenance checks
pass. Published claims must name field, bands, population, resolution matching,
purity level, uncertainty interval, and compute budget.

## Phase 10 — Independent replication and generalization

- Repeat the frozen workflow on at least one independent field.
- Verify Linux CUDA and CPU behavior; retain local CPU/MPS correctness where
  supported.
- Complete broader distorted/dithered/cylindrical/all-sky parity independently of
  the SMACS claim.
- Add DDP only after the single-device path is bounded and deterministic; require
  one-rank/eight-GPU equivalence before using DDP for scale claims.
- Keep FITS subset streaming and large-output ownership in Torchfits.

## Artifact contract for every run

Each artifact records:

- exact pushed commit and dirty status;
- container image, node, device, dtype, CUDA/driver versions;
- deterministic seeds and split identities;
- detector input paths, sizes, checksums, WCS, units, masks, and exposure counts;
- measured passband samples and normalization convention;
- PSF/background calibration identities and pixel-response convention;
- model kind, parameter counts, initialization, optimizer, steps, and budget;
- wall time, peak accelerator memory, peak RSS, and scratch use;
- held-out metrics, fidelity gates, and pass/fail reasons;
- FITS/PNG/RGB paths and hashes;
- explicit `depth_claim: false` until Phase 9 passes.

## Commit and review discipline

- Keep each phase in reviewable commits: tests/contracts, implementation,
  visualization, runner, then evidence/docs.
- Never commit `.freebuff/`, credentials, caches, or transient session state.
- Preserve unrelated user work; stage exact files or reviewed hunks only.
- Before every push: inspect status/diff/log, run targeted tests,
  `pixi run preflight-push`, and `pixi run typecheck` for signature changes.
- Push only to `origin/main`, never force-push, and pin every CANFAR replacement
  run to the resulting exact commit.

## Immediate next actions

1. Continue monitoring `lxc5ook9`, `rr4w3dib`, and `po465gbo`; validate and record
   artifacts immediately when they finish.
2. Finish Phase 1 tests and wiring for the uncommitted common scene initializer.
3. Add Phase 2 arbitrary-transmission rendering and analytic parity tests.
4. Add Phase 3 RGB/resolution-matched exports to the reconstruction canary.
5. Re-extract and inspect calibration evidence, prioritizing F090W coverage.
6. Resume the four blocked classical bands only on healthy GPU placement.
7. Start the bounded Phase 6 real-data architecture canary as soon as its bands
   have frozen calibration and classical products; do not wait for a broad
   synthetic matrix.
8. Run Phase 7 targeted synthetic invariants against the real-data finalists in
   parallel with the paired real-detector injection preparation.
