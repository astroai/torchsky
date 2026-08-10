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
- [x] Use the complete 135 pre-existing aligned
  `*_crf_rshift_tweakreg.fits` detector products for the matched scientific
  method matrix. They retain detector `SCI/ERR/DQ` planes with registered WCS
  metadata. The separately staged 135 CAL products remain a complete
  detector-level fallback/audit set; no CRF refetch is required for this
  experiment.
- [x] Use the header-selected common-support 512-square F277W target-grid crop
  at `(x0=112, y0=0)`. Do not select the field from science-pixel appearance.
- [x] Keep simultaneous short/long observations together:
  `train={1,2,4,5,7}`, `validation={3,8}`, `test={6,9}` by `PATT_NUM`.
- [x] Treat published UPdec/iter0 mosaics as registration and visual-QA
  references only. They are neither independent truth nor the primary depth
  comparator. The stacker harness also reports static residual diagnostics
  against their pixels, but injected completeness/purity is measured only on
  products regenerated from the injected detector exposures.
- [x] Keep the public UPdec iter10 artifact stacks as a secondary static
  iter0-versus-iter10 context comparison. They cannot enter the injected-depth
  method matrix unless the same injected detector exposures are rerun through
  the UPdec pipeline.
  Evidence: session `yeym14dy` at exact commit `e87cb11` compared all six
  published pairs after Torchfits read and analytic TAN alignment including
  CRPIX, scale, rotation, and shape changes. Iter10/iter0 central-crop gradient
  ratios are 1.48--2.11, Laplacian ratios are 1.29--4.22, and summed flux ratios
  are 1.0008--1.0243. Artifacts:
  `/arc/projects/ots/torchsky-runs/20260729-updec-iter0-vs-iter10-e87cb11/`.
- [x] Keep all FITS access through Torchfits/Torchsky. The standalone public
  artifact-pair benchmark calls Torchfits directly, so it does not accidentally
  require Torchsky's compiled mapping extension merely to read FITS.
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
  - [x] Re-audit all 135 local CAL headers through the production Torchfits
    calibration-manifest path on 2026-08-09. It recovered 75 train, 30
    validation, and 30 test exposures, all with canonical `PRIMARY:EXPMID`
    timing; manifest SHA-256
    `642576612af33a7df264409abbafc2d24e05f78ba9b5b0f2b434cba1c2e4d276`.
  - [x] Add the reviewed-catalog CAL extractor and held-out PSF/background
    metric contract. The remaining real-data gate is catalog review followed
    by one extraction and staged fit; synthetic CAL-to-FITS target closure
    passes locally.
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
- [x] Add an approved band-dependent PSF contract. Injection truth must use a
  more accurate/oversampled PSF than the fitted approximation so this is not an
  inverse-PSF experiment. For the abstract endpoint, freeze the existing
  four-times oversampled Gaussian truth at FWHM
  `0.033/0.049/0.066/0.091/0.116/0.145` arcsec for
  F090W/F150W/F200W/F277W/F356W/F444W and retain the deliberate 5% wider fitted
  model as the misspecification control. This deterministic relative-depth
  endpoint minimizes new instrument-model systematics; WebbPSF and empirical
  stars are post-abstract robustness work, not tuning inputs.
- [ ] Route independent and shared Foam through the same WCS, bandpass, pixel
  footprint, PSF, mask, weight, and exposure-nuisance operators used by the INR.
  - [x] Add `SpectralPowerFoamField`, the minimum local-tangent field adapter:
    smooth fitting now enters the existing `FieldObservation` WCS, bandpass,
    PSF, mask/weight, and nuisance path; detached hard rendering reuses the same
    Legendre spectral basis and learned coefficients.
  - [x] Add deterministic regular-grid initialization without science-pixel
    tuning and hard-render target WCS pixel polygons through the common
    bandpass, PSF, and background operators. The real adapter now records its
    aligned/CAL input choice and declared Foam capacity.
  - [x] Activate the frozen fitted-PSF contract in the real joint adapter. Each
    detector crop and target product now converts the approved 5%-wider
    band-dependent Gaussian FWHM through its local Torchsky-WCS pixel area;
    fitting uses the existing 2-by-2 Gauss-Legendre pixel-footprint quadrature,
    and JSON rows record truth/fitted FWHM, input/target pixel sigmas, and the
    spatial-sampling mode. A shared-sky-patch fit regression exercises both
    pixel integration and PSF convolution.
  - [ ] Exercise independent and shared instances on the frozen real exposure
    adapter and persist their hard-rendered final products.
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
  canonical manifest. The experiment consumes the 135 aligned
  `*_crf_rshift_tweakreg.fits` products; the CAL inventory is a reproducibility
  fallback, and the 27 unregistered CRFs are the long-channel originals found
  at `/arc/projects/ots/jwst_smacs0723/`. Manifest audit session `rrkjowzw`
  confirms that the aligned set is linked—not downloaded—from
  `results/wcs_aware_multiband_inr/official_no_psf_whitened_l2_batch100_400_tweakreg_rshift_8plus3_run2/alignment/tweakreg_rshift`.
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
  - [x] Extend the same effective-PSF report with exact-WCS, border-subtracted
    aperture fluxes and robust fractional-bias diagnostics against the manifest
    flux in every band.
  - [x] Rescore the existing 42 products in session `q82737l1` at `aa0aa21`.
    Upstream Drizzle pixfrac 0.8/1.0 and SWarp bilinear pass the 2% median-bias
    gate in all six bands, while inverse Torchsky bilinear/Lanczos fail in the
    undersampled short bands (F090W +15.3%/+12.1%). This is a gate failure, not
    a kernel selection. Session `g1ajbo0l` is testing the native
    flux-conserving Torchsky square-Drizzle variants before the candidate set is
    frozen.
    The primary comparison rows use the external Python `drizzle` package and
    the SWarp 2.41.5 executable. Torchsky square Drizzle is a separately named
    native candidate, never an alias for either external baseline.
  - [x] Checkpoint and fan out the slow native calibration without changing its
    frozen inputs or methods. Serial session `g1ajbo0l` completed both
    positive/blank F090W and F150W products plus the F200W blank before being
    stopped. The first fanout `z4yz1lrz`/`wgy6yrh6`/`fpkjir10`/`k1nm430k`
    failed during bootstrap source parsing before scientific work. Corrected
    sessions `kcumoz89`, `glns4y2r`, `x32kchen`, and `ys4tjw3v` independently
    finish F200W/F277W/F356W/F444W in the same durable artifact root. Low-resource
    continuation session `y5cbuwjy` waits for all 12 band/mode reports and then
    generates `effective-psf-flux-report.json` automatically.
- [x] Select Drizzle `pixfrac` from the predeclared grid and SWarp kernel from
  `{BILINEAR, LANCZOS3}` using validation injections plus the predeclared flux
  gate. At pooled purity at least 95%, upstream Drizzle `pixfrac=0.8` recovers
  74/84 sources (88.1%; 50% at AB=30), tied with `pixfrac=0.6` in recovery but
  passing the six-band 2% median-flux gate; it also out-recovers the other
  flux-valid candidate, `pixfrac=1.0` (73/84). SWarp Lanczos3 recovers 76/84
  but fails that gate in F200W (2.15%); the valid SWarp winner is bilinear at
  72/84 (85.7%; 41.7% at AB=30).
- [x] Build the whitened chi-square six-band detector from each winning stack
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
    input hashes plus exact world-coordinate avoidance catalogue.
  - [x] Run all seven real blank catalogues and freeze their union before any
    depth validation. Session `c2iartol` at `32e0940` completed with 1,882
    permissive method detections covered by 282 distinct six-pixel avoidance
    neighbourhoods under
    `/arc/projects/ots/torchsky-runs/20260729-smacs-control-catalogues-32e0940/`.
  - [x] Make `MeshSkyEstimator` fill fully masked coarse cells from finite
    neighbouring cells and fail closed only when the whole mesh is masked.
    Session `bgjkc3jm` exposed the former NaN propagation before producing a
    catalogue; no scientific outcome was inspected.
  - [x] Add `updec-adass-classical-validation`: remove detections near the
    same-method frozen control catalogue, exact-match the remaining incremental
    detections to injections, pool batches to select the lowest 95%-purity
    threshold, and report per-batch and per-magnitude completeness.
  - [x] Score all three completed batches in session `j4xy3vos` at `aa0aa21`.
    Every method reached the pooled 95%-purity target. The seven persisted
    thresholds and completeness tables are under
    `/arc/projects/ots/torchsky-runs/20260729-smacs-validation-score-aa0aa21/`;
    the upstream Drizzle `pixfrac=0.8` threshold is 4.9025 and the valid SWarp
    bilinear threshold is 4.9117.
- [ ] Include Montage if its exact target grid is supported; otherwise record an
  explicit secondary skip and continue.
- [x] Run at least three validation injection batches and one blank control.
  Session `oi5mskje` completed three deterministic 28-source positive batches
  from the exact frozen control union; session `c2iartol` supplied the seven
  blank controls. Products and detections are under
  `/arc/projects/ots/torchsky-runs/20260729-smacs-validation-32e0940/`, with
  validation summaries under
  `/arc/projects/ots/torchsky-runs/20260729-smacs-validation-score-aa0aa21/`.
- [ ] Freeze `adass_updec_classical_config.json`, including commits, external
  tool versions, target header hash, inputs, split, PSF, seeds, and thresholds.

Day-2 stop gate: classical completeness curves must be monotonic within sampling
uncertainty, blank controls must behave, and bright-source flux must satisfy the
2% gate.

### Day 3 — Thursday 30 July: independent versus shared Foam

- [x] Add the minimum field/observation adapter for `SpectralPowerFoam`; reuse
  current rendering and training operators rather than creating a Foam pipeline.
  `SpectralPowerFoamField` supplies normalized local-tangent geometry,
  well-conditioned Legendre spectral coefficients, temperature curriculum,
  common field-forward compatibility, and detached exact-pixel handoff. Seven
  focused tests cover gradients, broadcast spectra, hard rendering, and the
  common bandpass/PSF render path.
- [ ] Initialize geometry without test pixels. Use a fixed deterministic grid
  for the first fair run; adaptive residual initialization is a validation-only
  ablation.
  - [x] Fix the real-data field domain to use the union of detector-edge WCS
    footprints rather than the first detector `CRVAL`. H100 session `snrobkst`
    at `51b757e` produced nonzero coefficients and exact hard products for the
    first time; this is a geometry gate, not a depth result.
  - [x] Apply the frozen `PATT_NUM={3,8}` validation split to Foam, matching
    the classical stackers' 8/2 short-/long-band exposure counts. The adapter
    records the selected patterns and the CANFAR validation runner defaults to
    them; all-exposure Foam products are excluded from method comparison.
  - [x] Pass the exact 512-square blank execution gate with bounded smooth
    rendering.
    The first all-exposure and matched-split attempts failed in the same
    pre-fit dense diagnostic render, before the sparse hard operator. Exact
    tiled initial/final diagnostics and streamed PSF-aware training patches
    reduced session `omf0qjkg` at `eb6cd59` to 186,634,752 bytes peak H100
    allocation and 41.3 s for 50 steps on the frozen 512-square WCS. Both hard
    products were finite and about 98.7% nonzero.
  - [ ] Pass the separate exact 512-square scientific-fidelity gate. The same
    16-square-grid run recovered source-contrast slopes near `-4.6e-6` and only
    1.6--2.5% of support flux. Physical-unit output scaling was present; this
    256-cell model spans roughly 75--140 target pixels per cell and is an
    under-capacity diagnostic, not a depth result. Record normalization and
    initial/final losses, contract the WCS-only grid to the footprint bound,
    scale smooth tiles inversely with cell count, and express soft-ownership
    temperature in inverse-cell-area units before testing a denser fixed grid.
    A fixed absolute temperature makes finer grids less locally identifiable
    because power distance is measured in squared coordinate units.
    The unscaled-temperature capacity control `lhztxhsb` at `1662f7c`
    increased the grid from 256 to 4,096 cells and completed in 253.7 s at
    691,726,848 bytes peak H100 allocation. Full normalized loss improved from
    0.9416 to 0.8754, but mean source-contrast slope remained only `1.18e-5`
    and mean support flux only 2.06%; capacity alone does not repair the
    non-local training surrogate.
    The temperature-only rerun `s0thrsmm` at `1687187` kept the same 4,096
    cells, 50 steps, and `lr=0.001`. Cell-area-normalized temperatures raised
    the hard prediction peak to 0.039 and mean source-contrast slope to
    `1.71e-5`, but support flux remained 2.28% and coefficients stopped at
    0.0276. Localization helps, but the optimizer amplitude ceiling still
    dominates.
  - [x] Re-run a true fixed-geometry control with a coefficient learning rate
    commensurate with unit-RMS normalized inputs. The earlier nominal
    fixed-grid runs also optimized sites/radii and used `lr=0.001`; after 50
    steps their maximum coefficients remained near 0.03, which explains
    physical predictions near 0.01 after restoring the measured 0.211 intensity
    scale. Geometry is now frozen by default and adaptive geometry is a
    separately declared validation-only ablation. A local uniform two-band
    control confirms that this is an optimization-scale issue: after 50 steps,
    `lr=0.001` leaves MSE at 2.34 from 2.50 and coefficients below 0.049,
    whereas `lr=0.02` reaches 0.429, coefficients 0.868, and a strong wavelength
    contrast. This control is not a depth result. The benchmark runner now
    fails closed on any requested model error or missing row. Exact H100
    session `vhevkgna` at `23e5ee9` completed the corrected PSF-aware,
    2-by-2 detector-pixel-integrated control in 1,130.6 s with 6,722,270,208
    bytes peak CUDA allocation. Loss fell from 0.9416 to 0.8195 and coefficients
    reached 0.1803, but mean support flux was only 0.0347, mean source flux was
    -0.0297, and mean source-contrast slope was `-5.94e-4`. The fixed-LR
    execution question is closed; scientific fidelity still fails. The one
    predeclared dense 256-square-grid session `j2fmxlln` was stopped after 48
    minutes because its 32-square memory-bounded patches sampled 16 times fewer
    sky pixels per step than the 64-grid control; it was not a clean capacity
    experiment. Frozen-geometry Foam now provides exact compact-support
    appearance fitting and fails closed for trainable geometry. The replacement
    256-square control must use that path with the original 128-square patch,
    unchanged data/PSF/pixel integration/steps/LR, and record
    `foam_compact_fixed_geometry=true` before its score is interpreted. Exact
    H100 session `mjsil8q2` completed that replacement from commit `42f0744` in
    172.7 s with 937,856,512 bytes peak CUDA allocation, versus 1,130.6 s and
    6,722,270,208 bytes for grid 64. Mean support flux remained 0.0332, mean
    source flux reached only 0.0802, mean source-contrast slope was `4.68e-4`,
    and only 49.62% of hard-render pixels were nonzero. Capacity is not the
    missing ingredient. Commit `4829a0a` then added an opt-in coadd initializer
    that uses the exact bandpass basis and a ridge-stabilized spectral solve per
    fixed cell. Exact H100 session `lera4b7j` initialized 18.13% of the full
    grid (the target footprint) to `1.12e-6` band reconstruction RMSE and
    restored 100% nonzero hard coverage. Initial loss improved from 0.9416 to
    0.5090 and mean source-contrast slope from `4.68e-4` to 0.337. It still
    failed: mean support/source flux overshot to 2.273/2.527 and mean support
    RMSE worsened to 0.662. Do not introduce an empirical scale or run another
    grid/LR sweep. Next solve fixed-cell appearance directly in pixel space
    through the exact `PowerPixelOperator.matvec/rmatvec`, then perform
    observation-level refinement. Artifacts are under
    `/arc/projects/ots/torchsky-runs/20260802-smacs-foam-coaddinit-g256-4829a0a`.
- [x] Replace centre sampling with an exact weighted pixel-footprint inverse.
  Commit `d10627f` adds a reusable Jacobi-PCG solve over the existing exact
  `PowerPixelOperator`, converts surface brightness to pixel-integrated units,
  and carries exact valid-pixel cell coverage into a scale-invariant spectral
  solve. Focused checks passed 29/29 and `ci-local` passed 1,739 tests with 214
  skips and zero type errors.
- [x] Score the one matched exact-pixel blank before any sweep or PSF change.
  H100 session `yxy529gn` completed full commit
  `d10627f7238c6f1381bc9e4be86a07aa322ebe3d` in 173.2 s at 937,856,512 peak
  accelerator bytes. Exact initialization covered 18.63% of the full grid and
  retained 100% hard-render coverage. It corrected the centre-sampled
  2.273/2.527 mean support/source flux overshoot to 0.872/0.889 while retaining
  mean source-contrast slope 0.276. It does not pass: mean support RMSE is
  0.414, flux is still 11--13% low, and normalized pixel initialization RMSE is
  2.047. Artifacts are under
  `/arc/projects/ots/torchsky-runs/20260802-smacs-foam-pixelinit-g256-d10627f`.
- [x] Compose the known symmetric target PSF into the exact initializer forward
  and adjoint, then run one otherwise identical blank. Commit `5c99a33` added
  the area-correct `D P D^-1` forward and `D^-1 P^T D` adjoint and passed 1,739
  tests. H100 session `cub30m3e` completed in 175.4 s at the same 937,856,512
  peak accelerator bytes. It is scientifically unchanged: mean support/source
  flux is 0.875/0.893, source-contrast slope 0.278, support RMSE 0.414, and
  normalized initialization RMSE 2.044. Target-PSF mismatch is not the limit.
  Artifacts are under
  `/arc/projects/ots/torchsky-runs/20260802-smacs-foam-psfinit-g256-5c99a33`.
- [x] Make one representation-resolution test, not a grid sweep. The initial
  session `zoypfeli` correctly stopped because its measured 16,423,554 raw
  candidate pairs exceeded the 10-million allocation guard. Commit `e935868`
  exposed that existing safety ceiling, and H100 session `e0do2myt` completed
  the frozen grid-512 blank with an explicit 20-million limit. Fit time was
  255.1 s and peak accelerator allocation was 3,147,516,928 bytes. Relative to
  grid-256, mean source-contrast slope improved from 0.278 to 0.515,
  initialization RMSE from 2.044 to 1.656, mean support RMSE from 0.414 to
  0.344, and mean support/source flux from 0.875/0.893 to 1.075/1.078. The
  representation-resolution hypothesis is confirmed, but the blank fidelity
  gate remains closed at roughly 7--8% mean flux excess. Artifacts are under
  `/arc/projects/ots/torchsky-runs/20260802-smacs-foam-psfinit-g512-e935868`.
- [ ] Score the frozen grid-512 configuration on the three existing injection
  batches before changing representation or tuning. H100 session `tsu81ekc`
  completed all three batches at exact commit
  `1d055cfba3d5a89841f6cee61747cf079ec10a2b` in 24 minutes. Their global blank
  metrics are deliberately nearly unchanged because the injected catalogue is
  sparse; recovery must be measured at the registered source positions.
  Artifacts are under
  `/arc/projects/ots/torchsky-runs/20260802-smacs-foam-g512-injections-1d055cf`.
  Scorer session `zuzdu3cp` failed before import because its fresh environment
  lacked the local Torchsky build; `quw7q4b3` then reached the science products
  but correctly rejected an ePSF stacked across faint non-positive responses.
  Commit `226b399` calibrates the ePSF on the predeclared brightest AB-magnitude
  bin without post-selecting on recovered response and passed 1,714 tests with
  246 skips and zero type errors. Its `@canfar` launch is currently blocked by
  the platform error `CONFIG: servops certificate is invalid`; the queued output
  root is
  `/arc/projects/ots/torchsky-runs/20260809-smacs-foam-g512-score-226b399`.
  If recovery is not competitive, stop regular-grid scaling and add exact
  within-cell first moments plus an affine spatial basis.
- [x] Implement the frozen spectral-Foam scorer while the injection fanout runs.
  `benchmarks/score_smacs0723_foam.py` stages exact products, intersects and
  erodes support, derives band-dependent ePSFs from injected-minus-blank model
  responses, and reuses the established matched-filter, blank-whitening,
  95%-purity, and completeness-by-AB-magnitude machinery. The `foam-score`
  headless runner mode executes it on CANFAR. Six focused tests include an
  end-to-end float32 FITS/ePSF check. Foam currently exports binary support
  rather than spatial inverse variance; the scorer records that deliberate
  ceiling and must be upgraded when propagated model uncertainty is available.
- [x] Add independent and joint spectral Foam rows to the synthetic smoke
  tournament. Commit `da06bc3` uses one shared footprint operator and enforces a
  target-response gate, so lower reconstruction loss cannot hide source
  attenuation. Commit `fbd8a99` replaces implicit band indices with
  response-weighted spectral coordinates, rejects invalid or unordered axes and
  passbands, records those coordinates in the artifact, and verifies that the
  dimensionless Legendre basis is invariant to affine wavelength-unit changes.
  This remains representation plumbing, not real-data depth evidence; its
  automatic grid density is only a bounded test heuristic.
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
- [x] Approve the abstract PSF truth model: the frozen four-times oversampled
  six-band Gaussian contract plus the 5% width-mismatch control. Treat
  WebbPSF/empirical-star results as later robustness evidence and do not tune
  the abstract endpoint on them.
- [ ] Draft the motivation/methods paragraphs without a numerical depth claim;
  leave `X`, confidence interval, photometric bias, and runtime placeholders.

### Torchsky engineering

- [ ] Implement and test the stable profile, real injection adapter, unit/PSF
  gates, method adapters, blind-run lock, report generator, and CANFAR runs.
- [ ] Keep `main` releasable with focused tests during development,
  `pixi run ci-local` before every push, and exact-commit artifacts.
