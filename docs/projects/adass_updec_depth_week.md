# ADASS UPdec Depth Sprint

> Active focus: 2026-07-28 through 2026-08-04. This sprint temporarily pauses
> FOSSIL/component-separation work and broad Torchsky roadmap expansion.
>
> ADASS XXXVI oral/tutorial abstracts are due **2026-08-01 Anywhere on Earth**:
> <https://adass2026.icrar.org/call-for-contrib>. Submit a conservative version
> by August 1 local time. The remaining sprint validates or strengthens it.

## Outcome

Produce one defensible detector-level result:

> On six-band JWST/NIRCam exposures underlying the UPdec-Webb SMACS J0723
> products, a shared spectral Power-Foam recovers fainter injected continuum
> sources than the strongest matched classical pipeline at fixed purity, while
> preserving fluxes and predicting held-out exposures.

The abstract may say “deeper” only if the blind-test gates below pass. The
strongest classical pipeline includes multiband detection from Drizzle/SWarp
products; beating separate per-band images alone is not sufficient.

## Scope Lock

- [x] Use the existing 135 calibrated SMACS J0723 exposures: 36 exposures in
  each of F090W/F150W/F200W and nine in each of
  F277W/F356W/F444W.
- [x] Use the header-selected common-support 512-square F277W target-grid crop
  at `(x0=112, y0=0)`. Do not select the field from science-pixel appearance.
- [x] Keep simultaneous short/long observations together:
  `train={1,2,4,5,7}`, `validation={3,8}`, `test={6,9}` by `PATT_NUM`.
- [x] Treat published UPdec/iter0 mosaics as registration and visual-QA
  references only. They are neither independent truth nor the primary depth
  comparator.
- [x] Keep all FITS access through Torchfits/Torchsky.
- [x] Make no FOSSIL, component-separation, COSMOS, release, or unrelated API
  changes during this sprint.
- [x] Fix only defects that block the experiment. Time-box exact Torchsky
  Drizzle optimization; the official upstream Drizzle comparator can establish
  the scientific baseline while the 3,691-second Torchsky path remains an
  explicitly reported engineering result.

## Existing Evidence

- [x] Deterministic seed manifests, dense spectral truth, source strata, common
  detector, validation-null thresholding, angular NMS, exact one-to-one
  association, forced flux/color metrics, NLL, and classical rows exist in
  `benchmarks/bench_multiband_depth.py`.
- [x] The production weighted-bilinear, exact square-Drizzle, and chi-square
  classical floor passed on a CANFAR H100 at commit `addb925`.
- [x] Official Drizzle, SWarp, and Montage backends are wired in the existing
  map-compat/SMACS stacker harness.
- [x] `SpectralPowerFoam` has differentiable smooth fitting, shared spectral
  coefficients, exact hard pixel rendering, sparse candidate geometry, and
  CPU/CUDA primitive gates.
- [x] Pre-sprint `main` (`cb01c62`) has green local and GitHub comprehensive CI
  ([run 30396973217](https://github.com/sfabbro/torchsky/actions/runs/30396973217)).
- [x] Freeze the complete calibrated-input inventory: 135 CAL files, with
  15 files for every `PATT_NUM`; band counts are
  F090W/F150W/F200W=`36/36/36` and F277W/F356W/F444W=`9/9/9`.
- [x] Inventory the existing CANFAR CRFs: 27 long-channel products under
  `/arc/projects/ots/jwst_smacs0723/`, covering three observation groups and
  nine patterns.

## Blocking Gaps

- [x] Close the detector injection calibration/value-semantics boundary without
  trusting the public-mosaic metric. Local real-data smoke evidence conserves
  injected Jy through distorted-WCS `MJy/sr` pixels to about `1e-8` relative;
  the existing 4.7-to-18.7 public-mosaic ratios remain excluded from claims.
- [x] Add detector-level real-image injection/recovery. The
  `updec-adass-smoke` profile reads unchanged `SCI/ERR/DQ`, injects into `SCI`,
  and keeps the control image available. Never inject only into a coadd.
- [ ] Add an approved band-dependent PSF contract. Injection truth must use a
  more accurate/oversampled PSF than the fitted approximation so this is not an
  inverse-PSF experiment. A six-band oversampled Gaussian proxy and deliberate
  5% width-mismatch control now gate the plumbing; science approval or an
  empirical/WebbPSF truth model is still required before the blind campaign.
- [ ] Route independent and shared Foam through the same WCS, bandpass, pixel
  footprint, PSF, mask, weight, and exposure-nuisance operators used by the INR.
- [ ] Apply one common detector/photometer to every final band product.
- [ ] Freeze validation-selected hyperparameters and thresholds before
  generating blind injection locations and noise/control identifiers.

## Claim Contract

### Primary endpoint

At empirically calibrated 95% purity, report

```text
Delta m50 = m50(shared spectral Foam) - m50(best classical multiband pipeline)
```

Fit `m50` from per-source binomial recovery probabilities. Bootstrap complete
injection-batch identifiers with 10,000 paired resamples.

The abstract may claim greater depth only when all are true:

- [ ] `Delta m50 > 0` and its paired 95% bootstrap interval excludes zero.
- [ ] Measured test purity is at least 95%; its lower 95% confidence bound is at
  least 93%.
- [ ] Bright injected-source median absolute flux bias is at most 2% per band
  and no more than one percentage point worse than the best classical method.
- [ ] Adjacent-band median color bias is at most 0.02 mag for the claimed
  smooth-continuum population.
- [ ] Held-out test-exposure predictive NLL is improved or statistically
  indistinguishable from the best independent model.
- [ ] The gain appears in at least two of four science-blind 256-square
  quadrants, not only in one favorable patch.
- [ ] Blank, negative-injection, and shuffled-band controls pass.

`0.2 mag` is a useful target, not a threshold that may be imposed after seeing
the results. Report the measured effect, interval, wall time, parameters/atoms,
peak RSS, and peak accelerator memory.

### Required method matrix

All methods receive identical exposures, masks, inverse variances, WCS,
bandpasses, PSF metadata, split identifiers, injections, and final target grids.

| Priority | Method | Role |
| --- | --- | --- |
| required | weighted per-band Torchsky stack | internal classical control |
| required | upstream square Drizzle; validation-selected `pixfrac` | standard dithered-imaging baseline |
| required | SWarp bilinear/LANCZOS3, chosen on validation only | standard survey-stacking baseline |
| required | whitened chi-square detection over the best per-band stack products | strongest classical use of all six bands |
| required | independent per-band Foam with union/best-band detection | isolates the gain from spectral sharing |
| required | shared spectral Foam | primary candidate |
| required fallback | shared spectral INR and independent per-band INR | protects the abstract if Foam integration is not ready or is not the winner |
| secondary | Montage | reprojection/mosaicking context, not a blocker for the depth endpoint |
| stretch | diffuse INR + compact Foam | run only after the primary matrix freezes |

Compare both capacity-matched and compute-matched independent/shared models.
Do not count independent models as receiving six times the optimization budget.

## Real Injection Design

- [x] Start from the unchanged `SCI`, `ERR`, and `DQ` detector planes. The
  real-data profile records `input_unchanged` and uses `ERR^-2` with valid
  `DQ==0` support for its robust planar background fit.
- [ ] Cache a no-injection control fit/catalogue for every method.
- [x] Build one frozen common-support mask from the intersection of every
  selected method's final per-band weight maps, eroded by the injection/PSF
  footprint. Generate source positions only inside this mask. The first F277W
  plumbing batch deliberately remains evidence for this gate: one AB 27.5
  source accepted by both input detectors was absent from SWarp's final
  product.
  - [x] Add deterministic support intersection, configurable erosion, target-WCS
    validation, persisted mask FITS/hashing, and support-constrained source
    placement to the existing `updec-adass-manifest` CLI. Three focused
    contracts pass; the remaining unchecked work is freezing the selected
    six-band mask.
  - [x] Complete all 42 blank products for six bands times seven classical
    candidates on the exact TAN grid. Session `kkpxfdxb` at `3ff8604` persisted
    every SCI/weight product under
    `/arc/projects/ots/torchsky-runs/20260729-smacs-sixband-controls-3ff8604/`.
    Zero-source probe `wxz900hl` measured only 984 pixels after nine-pixel
    erosion (mask SHA-256 `99943000...2b9dc68`), so this candidate crop is not
    frozen for science injections. Diagnose the limiting band/method and
    support topology without using recovery outcomes.
  - [x] Replace whole-quadrant rejection sampling with deterministic shuffled
    valid-pixel pools per quadrant. Sparse valid masks now either place sources
    reproducibly or report the unsupported quadrant directly.
  - [x] Add an explicit minimum local footprint-coverage fraction per method.
    The default remains strict 100% erosion; a lower predeclared fraction may
    tolerate sparse Drizzle holes while still requiring positive center weight
    in every candidate. The CLI records a predeclared coverage-grid sweep so it
    can be frozen from weight topology before any recovery run.
  - [x] Freeze `min_footprint_fraction=0.99` from weight topology alone.
    CANFAR session `vedlmudw` at `a0a40bf` retained 109,393 pixels (41.73%)
    with 22,175–31,681 pixels in every quadrant. This is the strictest
    predeclared non-degenerate threshold: it permits at most four unsupported
    pixels in a 19-square footprint per method while still requiring positive
    center weight in all 42 products. No injection recovery was inspected.
  - [x] Persist the selected placement mask and exercise it with eight
    four-quadrant, 36-pixel-separated calibration sources. Session `ry4c9xv5`
    wrote `common-support-r9-f0p99.fits` (SHA-256
    `adc31bd3...d2dfa253`) and manifest mask hash
    `37bb326a...742efd30` under
    `/arc/projects/ots/torchsky-runs/20260729-smacs-psf-calibration-8cc3fb9/`.
- [ ] Inject sources through each exposure WCS, oversampled PSF, pixel footprint,
  calibration, and bandpass. Preserve the original background/noise/artifacts.
- [ ] Use smooth continuum point sources for the abstract endpoint. Add compact
  galaxies, dropouts, emission-line objects, and close pairs only after the
  primary endpoint is frozen.
- [ ] Use seven or more flux levels densely bracketing the validation estimate
  of `m50`.
- [ ] Use at most 5% injected-source areal occupancy per fit and reject isolated
  injections within two PSF FWHM of the frozen control catalogue.
  - [x] Add target-WCS control-catalogue ingestion and deterministic circular
    exclusion to the manifest CLI. It persists the catalogue, radius, source
    count, and pre/post-exclusion support counts. The remaining gate is deriving
    and freezing the real blank-field catalogue after effective PSFs exist.
- [ ] Run at least 20 independent batches distributed deterministically over the
  four 256-square quadrants. Aim for at least 800 total continuum injections.
- [ ] Run blank, sign-flipped/negative, and shuffled-band batches through the
  identical fitting and detection path.
- [ ] Detect on the full reconstructed products. Use the control catalogue to
  distinguish pre-existing real objects; do not claim depth from an
  injected-minus-control image with the real sky artificially removed.

## Four-Day Submission Sprint

### Day 1 — Tuesday 28 July: make the experiment trustworthy

- [x] Add a shipped `updec-adass` profile or equivalent stable CLI to
  `benchmarks/bench_multiband_depth.py`; do not create a separate analysis
  framework.
- [x] Add the real exposure/injection adapter by reusing discovery, cropping,
  and `SCI/ERR/DQ` helpers from the SMACS joint benchmark.
- [x] Reconcile `MJy/sr`, pixel-area, photometric scale, inverse variance, and
  surface-brightness versus integrated-flux semantics with bright noiseless
  detector-level injections.
- [x] Add a band-dependent oversampled Gaussian truth/fitted-PSF plumbing
  contract and a deliberately perturbed-PSF control. The final science PSF
  model remains an approval gate above.
- [x] Pass one 32-square, two-band, two-exposure CPU/CUDA smoke and write a
  deterministic JSON artifact.
  - [x] CPU on real F150W/F277W calibrated exposures: rendered-flux error below
    `1e-8`, post-background exact-PSF flux error below `2e-6`, centroid error
    below `2e-8` pixel; the 5% PSF-width control shifts flux by 3–4%.
  - [x] CUDA exact-commit CANFAR staging artifact: session `jaannbqt`, commit
    `3bc6ca6e2b36b325dfad53686eaf8f9c01c53df6`, image
    `images.canfar.net/skaha/astroflow-cuda:latest`, `cuda-bench`, Torch 2.10.0,
    CUDA 12.9, one visible GPU, 33.10 s command wall time, and 1,675,628 KiB
    peak child RSS. The two exact MAST detector products, runner result, logs,
    and science JSON are under
    `/arc/projects/ots/torchsky-runs/20260728-updec-day1-3bc6ca6-v3/`.

Continue the unfinished Day-1 gate from a clean checkout of this checkpoint:

```bash
pixi run python benchmarks/bench_multiband_depth.py \
  --profile updec-adass-smoke \
  --input-root data/jwst_smacs0723 \
  --updec-bands F150W,F277W \
  --exposures 1 --cutout-size 32 --device cpu \
  --json-out benchmark_results/updec_adass_smoke_local.json
pixi run pytest tests/test_multiband_depth_benchmark.py \
  tests/test_map_operator_execution.py tests/test_map_background_ops.py -q
```

Day-1 stop gate: a bright injected source must recover position and flux through
each final method boundary. Otherwise do not begin model tuning.

### Day 2 — Wednesday 29 July: freeze the strongest classical baseline

- [x] Add a dependency-free, atomic, resumable staging command for the frozen
  135-file MAST grid and canonical links to existing CRFs:
  `benchmarks/stage_smacs0723_canfar.py`.
- [x] Complete and verify
  `/arc/projects/ots/torchsky-data/jwst_smacs0723/{cal,crf,aligned,manifest.json}`
  from one exact pushed commit.
  Evidence: session `tmbk7b72` at commit `35af22e` downloaded all 135 CAL
  products and linked all 27 CRFs. Session `e74yyfec` at commit `46c1ce5`
  linked all 135 registered aligned products and atomically refreshed the
  canonical manifest.
- [x] Freeze the validation comparator interface: `--input-product aligned`,
  `--patt-nums 3,8`, `--crop-x0 112 --crop-y0 0`, and
  `--validation-classical --drizzle-pixfracs 0.6,0.8,1.0`.
- [x] Run weighted Torchsky, upstream Drizzle, and SWarp on identical validation
  exposures and method-compatible target grids.
  Evidence: Torchsky session `axix9wz3` at `803a200`, upstream Drizzle session
  `z7t27ff0` at `46c1ce5`, and locked SWarp 2.41.5 session `gckyhwco` at
  `62f572a`. Durable reports are under
  `/arc/projects/ots/torchsky-runs/20260728-smacs-{torchsky-803a200,drizzle-46c1ce5,swarp-62f572a}/`.
  These are backend/geometry diagnostics against an aligned detector image,
  not depth measurements and not a scientific kernel selection. They used the
  native published crop for Torchsky/Drizzle and its north-up TAN equivalent
  for SWarp, so they are not an exact common-grid comparison.
- [x] Add deterministic common-world smooth-continuum manifests, transient
  detector-SCI injection overrides, and exact-target product persistence.
  Archived CANFAR inputs remain unchanged; all classical families receive the
  same injected detector pixels. Evidence: commits `278735e` and `419c94a`,
  including the focused common-world conservation/blank-control tests.
- [x] Run one 28-source validation plumbing batch and its zero-source blank
  through Torchsky bilinear, upstream Drizzle `pixfrac=1`, and SWarp bilinear.
  Evidence: CANFAR session `py7dl02x` at exact commit `419c94a`; every
  detector audit preserves the archived input, all six positive/blank
  products and weights were written and hashed under
  `/arc/projects/ots/torchsky-runs/20260728-smacs-injection-419c94a/`.
  This proves common-pixel delivery and artifact persistence only; completeness,
  purity, and kernel selection remain unmeasured.
- [x] Repair and revalidate SWarp astrometry on SIP-distorted JWST inputs.
  SWarp 2.41.5 does not consume SIP input distortion, so commit `fb779ef`
  translates each SIP polynomial to its equivalent TPV convention before
  invoking SWarp. The synthetic WCS parity test agrees to `2e-9` degree across
  the detector. Exact CANFAR session `lf5xzbso` reran the same positive/blank
  manifest: median peak offset fell from about 8.57 pixels to 0.39 pixels
  (0.56 pixels for the bright `AB<=28` stratum). Detailed audit session
  `pnk5hrr4` also found one unsupported bright source and a lower fixed-PSF
  bright median flux ratio for SWarp (0.71 versus 0.86 for Torchsky/Drizzle).
  Therefore this is an astrometric/plumbing pass, not a photometric pass or
  kernel selection. Hashed artifacts are under
  `/arc/projects/ots/torchsky-runs/20260728-smacs-swarp-sip-fb779ef/`.
- [x] Remove the target-grid ambiguity for every subsequent classical sweep.
  The validation profile now forces Torchsky, upstream Drizzle, and both SWarp
  kernels onto the same frozen north-up TAN header; the injection-manifest CLI
  uses that target by default and rejects weight maps whose WCS differs at the
  four corners or centre. This code gate must be exercised by the six-band
  control sweep before configuration freeze.
- [x] Generate the frozen per-band/per-method common-support mask before the
  next injection manifests. Exclude the unsupported-source failure by design,
  not after examining recovery.
  - [x] Implement `--support-weight-paths`, `--support-weight-threshold`, and
    `--support-erode-radius-px` plus local footprint coverage in the existing
    manifest profile. The emitted artifact records the resolved paths,
    individual/raw/eroded support diagnostics, retained quadrant counts, and
    SHA-256 of the exact boolean placement mask.
  - [x] Produce all 42 exact-grid blank products at `3ff8604`. The zero-source
    probe retained only 984 pixels after erosion, so the crop remains a
    diagnostic while limiting-map and quadrant support are audited.
  - [x] Audit local coverage at 80/85/90/95/98/99/100% and freeze 99% before
    positive calibration. Exact session `vedlmudw`; selected support retains
    109,393 pixels across all quadrants.
- [x] Fit or propagate each method's effective PSF before applying the 2% flux
  gate. A fixed injected-PSF fit is not comparable after different resampling
  kernels broaden the product differently.
  - [x] Add fractional-phase registration and robust stacking of isolated
    injected-minus-blank responses into a normalized effective PSF. The
    `updec-adass-effective-psf` profile persists one hashed PSF and size/lobe
    diagnostics per band and method.
  - [x] Run eight isolated AB=25 sources through all 42 band/method products
    and persist every measured ePSF. Session `ry4c9xv5` at exact commit
    `8cc3fb9`; Gaussian-equivalent FWHM spans 0.756–2.742 target pixels and the
    largest measured negative-lobe fraction is 4.58% for undersampled F090W
    Torchsky Lanczos3. These are PSF diagnostics, not kernel selection.
- [ ] Select Drizzle `pixfrac` from a small predeclared grid and SWarp kernel
  from `{BILINEAR, LANCZOS3}` using validation injections only.
- [ ] Build the whitened chi-square six-band detector from each winning stack
  family and select the empirical 95%-purity threshold on validation nulls.
  - [x] Add spatial-inverse-variance matched filtering with one effective PSF
    per band and positive per-band signal-to-noise combination in quadrature.
    Robust clipped per-band response centering/scaling comes from the frozen
    blank to absorb resampling-correlated noise; the final purity threshold
    still comes from validation control products.
  - [x] Add `updec-adass-classical-detection`: one Torchsky mesh-background,
    effective-PSF matched-filter, blank-whitening, positive-chi, angular-WCS
    catalogue path for every classical product family. It persists the
    statistic image, detections, scores, WCS coordinates, hashes, and nuisance
    diagnostics; the real threshold/catalogue run remains unchecked.
  - [x] Add `updec-adass-control-union`: validate that every control catalogue
    shares the frozen support mask, deterministically cover duplicate
    cross-method peaks within the predeclared six-pixel radius, and persist the
    input hashes plus exact world-coordinate avoidance catalogue. The seven
    real blank catalogues remain an execution gate.
- [ ] Include Montage if its exact target grid is supported; otherwise record an
  explicit secondary skip and continue.
- [ ] Run at least three validation injection batches and one blank control.
- [ ] Freeze `adass_updec_classical_config.json`, including commits, external
  tool versions, target header hash, inputs, split, PSF, seeds, and thresholds.

Day-2 stop gate: classical completeness curves must be monotonic within sampling
uncertainty, blank controls must behave, and bright-source flux must satisfy the
2% gate.

### Day 3 — Thursday 30 July: independent versus shared Foam

- [ ] Add the minimum field/observation adapter for `SpectralPowerFoam`; reuse
  current rendering and training operators rather than creating a Foam pipeline.
- [ ] Initialize geometry without test pixels. Use a fixed deterministic grid
  for the first fair run; adaptive residual initialization is a validation-only
  ablation.
- [ ] Fit independent per-band Foam and shared spectral Foam under both matched
  capacity and matched sampled-pixel/step budgets.
- [ ] Hard-render final products with detached exact geometry; never score the
  smooth training surrogate as the final Foam image.
- [ ] Run the same matrix for the strongest current shared/independent INR as a
  fallback and architectural control.
- [ ] Tune only atom density/capacity, spectral basis size, learning rate,
  temperature schedule, step budget, and stopping rule on validation batches.
- [ ] Freeze `adass_updec_model_config.json` and commit before blind test IDs are
  generated.

Day-3 noon fallback: if exact Foam integration or flux recovery is still broken,
freeze the joint INR as the primary submitted method and describe Foam as an
experimental adaptive representation. Never rename an INR result as Foam.

### Day 4 — Friday 31 July: blind CANFAR result and abstract numbers

- [ ] Generate blind test injection/control identifiers only after both config
  files are committed.
- [ ] Run at least 20 batches on CANFAR H100 sessions from one exact commit.
- [ ] Persist every batch under
  `/arc/projects/ots/torchsky-runs/20260731-adass-updec-depth-<commit>/`.
- [ ] Generate one merged JSON and Markdown report, including all failures.
- [ ] Compute paired batch-bootstrap intervals and the complete claim gate.
- [ ] Produce:
  1. completeness versus AB magnitude at fixed purity;
  2. `Delta m50` with 95% interval for each comparator;
  3. flux/color bias and held-out NLL;
  4. runtime/peak-memory/parameter-or-atom table;
  5. one clearly labelled visual example as supporting evidence only.

### Submission day — Saturday 1 August

- [ ] Re-run one frozen batch and verify its artifact hash.
- [ ] Fill the claim sentence only from the generated blind report.
- [ ] Submit the conservative abstract by August 1 local time rather than using
  the final AoE hours as operational margin.
- [ ] If a claim gate fails, use the fallback wording below. Do not tune on blind
  test failures.

## Remaining Sprint — 2 through 4 August

- [ ] Repeat the frozen endpoint with additional batches/quadrants.
- [ ] Add compact galaxies, dropout/emission-line, crowded, and PSF-misspecified
  strata.
- [ ] Resolve any transient CANFAR/GitHub failures and obtain green exact-commit
  CI.
- [ ] Update the abstract only if the submission system still permits it.
- [ ] Freeze a reproducibility bundle and define the later COSMOS scale-up; do
  not start downloading or re-engineering COSMOS during this sprint.

## Abstract Claim Ladder

Use the strongest statement whose gates actually pass.

### A — Desired quantitative Foam claim

> On detector-level injections into six-band JWST/NIRCam observations of
> SMACS J0723, shared spectral Power-Foam improves the 50%-completeness limit by
> `X [95% CI: L, U]` mag at 95% purity relative to the strongest
> Drizzle/SWarp multiband baseline, with `Y%` bright-source flux bias and no
> degradation in held-out exposure likelihood.

### B — Safe joint-INR result

> A joint differentiable continuous-field model improves the
> 50%-completeness limit by `X [L, U]` mag over matched classical multiband
> stacks; Power-Foam is evaluated as an adaptive representation within the same
> operator framework.

Use this if INR wins or Foam is not fully gated. Do not put “Foam reaches
deeper” in the title or abstract.

### C — Methods-only fallback

> We present a reproducible detector-level benchmark for classical coadds,
> implicit fields, and adaptive Power-Foam representations, with common
> completeness, purity, photometric-fidelity, held-out-likelihood, and
> computational-cost measurements on JWST/NIRCam data.

Use this if no positive blind depth interval exists. A correct negative result is
safer than a post-selected depth claim.

## Proposed Stable Commands

These are the intended end-state CLI contracts; check the boxes only after the
profile exists and `--help` confirms them.

```bash
cd /Users/fabbros/src/torchsky

pixi run -e test -- pytest tests/test_multiband_depth_benchmark.py -q

pixi run -e sphere-bench python benchmarks/bench_multiband_depth.py \
  --profile updec-adass --stage validation --device cuda \
  --input-root data/jwst_smacs0723 \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --train-patt-num 1,2,4,5,7 \
  --validation-patt-num 3,8 --test-patt-num 6,9 \
  --methods weighted,drizzle,swarp,chi2,independent_inr,joint_inr,independent_foam,joint_foam \
  --config-out benchmark_results/adass_updec_frozen_config.json \
  --json-out benchmark_results/adass_updec_validation.json

pixi run -e sphere-bench python benchmarks/bench_multiband_depth.py \
  --profile updec-adass --stage test --device cuda \
  --config benchmark_results/adass_updec_frozen_config.json \
  --injection-batches 20 --bootstrap-resamples 10000 \
  --json-out benchmark_results/adass_updec_blind.json \
  --markdown-out benchmark_results/adass_updec_blind.md
```

## Parallel Work

### User/science and abstract

- [ ] Create the Pretalx submission now and enter a conservative methods-first
  title/summary so the deadline is operationally safe.
- [ ] Confirm authors, affiliations, presenting author, requested format, and
  Theme 2 (“AI as a tool for scientific discovery”) versus Theme 8.
- [ ] Confirm how UPdec-Webb should be credited. The detector exposures are
  JWST data used for the UPdec-Webb field; do not imply that UPdec produced the
  raw exposures.
- [ ] Approve the abstract endpoint population: smooth-continuum point sources
  first, with extended/dropout/line sources labelled as robustness work.
- [ ] Approve the PSF truth model and acceptable instrument systematic floor.
- [ ] Draft the motivation/methods paragraphs without a numerical depth claim;
  leave `X`, confidence interval, photometric bias, and runtime placeholders.

### Torchsky engineering

- [ ] Implement and test the stable profile, real injection adapter, unit/PSF
  gates, method adapters, blind-run lock, report generator, and CANFAR runs.
- [ ] Keep `main` releasable with focused tests during development,
  `pixi run ci-local` before every push, and exact-commit artifacts.
