# Multi-wavelength INR / Power-Foam depth experiment

Date: 2026-07-22
Status: experiment design; no production API commitment

## Decision

The first scientific demonstration must answer a catalogue-level question:

> At a fixed false-discovery rate, do shared multi-wavelength INR and Power-Foam
> scene models recover fainter real sky sources than independent single-band
> processing, without biasing fluxes, colors, or uncertainty?

Image RMSE, PSNR, SSIM, or a visually sharper reconstruction are supporting
diagnostics. None is evidence for “going deeper” by itself. The primary result is
a paired injection/recovery selection function, measured on independent noise and
reported as completeness versus source amplitude at fixed purity. A held-out
exposure likelihood test is required so a regularized model cannot win by drawing
plausible-looking sources unsupported by detector data.

The experiment should be implemented as one benchmark-local harness that reuses
Torchsky's observation operators. Do not add a detection framework to the public
API or a new runtime dependency for this milestone.

## What Torchsky already has, and what is missing

Existing foundations:

- `src/torchsky/map/field/types.py`: weighted/masked observations, WCS headers,
  tabulated throughput, PSF, backgrounds, band labels, and spectral quantity.
- `src/torchsky/map/field/rendering.py`: WCS, spectral quadrature, detector-pixel
  quadrature, PSF, background, exposure pose, and tiled field rendering.
- `src/torchsky/map/field/training.py`: joint observation fitting, band-balanced
  sampling, global/per-band reductions, patches, spectral curricula, Gaussian NLL,
  and pose refinement.
- `benchmarks/bench_map_field_fit.py`: synthetic multiband broad/exact bandpasses,
  TAN/SIP/TPV geometry, variable PSF/noise, joint fields, and coadd baselines.
- `benchmarks/bench_map_multifield_truth.py`: known latent-cube truth and cube-level
  reconstruction metrics.
- `benchmarks/bench_map_smacs0723_field_fit_joint.py`: a shared field fitted to real
  JWST calibrated exposures and compared per band with public reference mosaics.
- `benchmarks/bench_inr_perf.py`: INR throughput and PSNR/SSIM versus native or
  SWarp coadds.

Missing evidence:

- no injected-source truth catalogue;
- no common source detector applied to every method;
- no one-to-one truth/detection association, completeness, purity, or false
  discoveries;
- no limiting magnitude or depth-gain confidence interval;
- no independent single-band INR baseline;
- no classical multiband chi-square detection baseline;
- no flux/color recovery or interval-coverage result;
- no train/validation/test split by exposure;
- no held-out observation-space likelihood or residual-whiteness guard against
  hallucination;
- no SED-stratified failure test for dropout, emission-line, or strong color-gradient
  sources.

The current SMACS joint benchmark's RMSE and source-flux slopes remain useful
regressions, but they cannot establish depth. The public reference mosaic is also
not statistically independent of the calibrated exposures and therefore must not
serve as the decisive test truth.

## Pre-registered hypotheses

Use three claims, in order. Do not promote a method on a later claim if an earlier
one fails.

1. **Detection depth:** for continuum sources with spatial support shared across
   bands, the joint method has a positive paired gain in 50% completeness depth
   relative to the best independent single-band pipeline at 95% measured purity.
   The lower endpoint of the scene-bootstrap 95% confidence interval for the gain
   must exceed zero.
2. **Scientific fidelity:** bright-source flux and color bias do not worsen, and
   empirical 68% and 95% intervals have calibrated coverage on unseen synthetic
   scenes/noise and real-image injections.
3. **Robustness:** on sources that violate shared-continuum assumptions—single-band
   emission lines, dropouts, and chromatic morphology—the joint method is not
   materially less complete than the best relevant single-band method and does not
   leak significant flux into unsupported bands.

A 0.2 mag gain in continuum-source `m50` is a useful stretch target, not a pass
threshold. Statistical evidence for a positive gain is the acceptance condition;
the measured effect size and compute cost must always be reported.

## Generative experiment

### Latent scene

For sky direction `s = (RA, Dec)` and wavelength `lambda`, generate

```text
F(s, lambda) = D(s, lambda) + sum_j S_j(s, lambda)
```

where `D` is a diffuse component and `S_j` are compact sources. Render observation
`e` through the same physical contract used by Torchsky, but with a more accurate
truth integrator than the fitted model:

```text
y_e[p] = calibration_e * integral_pixel integral_lambda
         R_e[p, s, lambda] F(s, lambda) T_e(lambda) ds dlambda
         + background_e[p] + noise_e[p].
```

Truth rendering should use at least 4x4 spatial quadrature and dense tabulated
spectral integration. Reconstruction uses the public 2x2 spatial quadrature and its
configured spectral budget. This avoids an inverse crime in which generator and
fit share the same discretization error.

### Source population

Each scene contains both isolated and crowded regions. Sample source amplitudes
dense around the expected threshold, with enough bright anchors to measure
photometric bias. Store the noiseless band-integrated flux of every object.

Required strata:

1. point sources with smooth continuum SEDs;
2. compact galaxies with one achromatic morphology and smooth SED;
3. two-component galaxies with bulge/disk color gradients;
4. narrow emission-line sources visible mainly in one band;
5. blue and red dropout sources;
6. close pairs spanning 0.5–2.0 output PSF FWHM;
7. diffuse low-surface-brightness sources, scored separately from point-source
   depth.

SEDs should come from a benchmark-local mixture of smooth non-negative basis
functions plus breaks and lines. Do not initialize or constrain the fitted model
with the generator's exact basis. Positions must be uniform over usable support,
with a separate clustered/crowded subset. Exclude moving and variable objects from
the first static-sky claim; add them later as an explicit nuisance/robustness case,
not silently as failed static sources.

### Observation diversity

Use six bands with overlapping and non-overlapping throughput, at least four
dithered exposures per band, rotations, TAN plus SIP/TPV distortion, chip gaps,
heteroscedastic inverse variance, band-dependent PSFs, photometric scales, smooth
backgrounds, correlated noise, cosmic-ray masks, and small astrometric residuals.

The publication profile must contain both:

- a correctly specified set where fit WCS/PSF/throughput/noise match truth; and
- a misspecified set with bounded PSF width, astrometric, zero-point, and background
  errors representative of calibrated imaging.

Noise realization, source catalogue, pointing, and nuisance seeds are separate and
serialized in the result artifact.

## Shared and band-specific parameters

The comparison is only meaningful if the sharing contract is explicit.

| Quantity | Joint INR | Joint Power-Foam / hybrid | Independent baseline |
| --- | --- | --- | --- |
| sky coordinates / WCS | fixed per exposure; optional learned pose nuisance | same | same |
| diffuse morphology | shared continuous `D(s, lambda)` | shared INR `D(s, lambda)` | separate field/image per band |
| compact positions | emerge in shared field | atom centers shared across wavelength | independent per band |
| compact support geometry | implicit in field | power sites/weights shared initially | independent per band |
| compact spectrum | continuous spectral decoder | per-atom non-negative spectral coefficients/decoder | independent amplitudes |
| chromatic morphology | allowed by spatial-spectral field | low-rank wavelength-dependent atom features; tested against fixed-geometry ablation | independent, unconstrained |
| PSF, pixel response, throughput | per exposure | per exposure | per exposure |
| calibration, background, detector artifacts | per exposure nuisance | per exposure nuisance | per exposure nuisance |
| uncertainty | predictive/ensemble distribution | predictive/ensemble distribution | matched bootstrap/analytic baseline |

The full-power Foam hypothesis for this experiment is not “one achromatic blob per
object.” It is shared adaptive spatial partition/support with wavelength-dependent
coefficients, plus enough chromatic residual capacity to represent color gradients.
The fixed-geometry version is an ablation and is expected to fail on some galaxies.

## Methods and fairness matrix

Every method receives identical calibrated exposures, masks, inverse variances,
WCS, throughput, PSF information, and train/validation/test exposure partitions.

Required baselines:

1. native per-band weighted bilinear coadd;
2. native per-band `drizzle_square` coadd at pre-registered pixfrac values;
3. SWarp per-band coadd when available, clearly marked external;
4. per-band coadds followed by the Szalay-style whitened chi-square multiband
   detection image;
5. independent single-band INR models, with detections unioned and also scored in
   the best band for each truth SED stratum;
6. joint spectral INR;
7. joint Power-Foam;
8. diffuse INR plus Power-Foam compact correction;
9. an oracle SED-matched multi-band filter on synthetic data only, reported as an
   upper bound and never as a deployable competitor.

Run two fairness views:

- **capacity matched:** total trainable parameter count (or total Foam atom/feature
  count) across all independent models is within 5% of the joint model;
- **compute matched:** equal total optimizer steps times sampled detector pixels,
  with actual wall time and accelerator memory also reported.

Hyperparameters and detection thresholds are selected on validation scenes only,
then frozen. The test seeds and real injection locations are generated after the
configuration is frozen. Report failed/non-converged runs; do not drop them from
the denominator.

## Common detection and measurement layer

Keep this code benchmark-local. It exists to compare representations, not to create
a second Source Extractor package.

1. Render each method to the same target WCS and band-integrated products.
2. Whiten each band by its propagated or empirically measured noise.
3. Convolve with the same small bank of PSF and extended-source templates using
   Torch operations.
4. For independent bands, record the maximum/union statistic and a chi-square
   statistic. For joint fields, evaluate the same band products plus a joint
   likelihood-ratio statistic. Account for the template and band trials when
   calibrating the threshold.
5. Find local maxima and apply the same angular non-maximum suppression.
6. Associate detections to truth one-to-one within the smaller of 0.5 target PSF
   FWHM and one output pixel. Use benchmark-only exact bipartite assignment; unmatched
   detections and duplicate detections are false positives.
7. Measure fluxes at all injected truth positions as well as at detected positions.
   This exposes selection bias. Integrate the latent scene through each true
   bandpass; also perform a common forced linear fit in held-out detector images.

Detection thresholds come from validation null fields/locations and target 95%
purity. They are not retuned on the test catalogue. Correlated noise means a nominal
“5 sigma” threshold is not interchangeable across methods; empirical purity is the
contract.

## Train, validation, and test split

### Synthetic

- smoke: 3 scene seeds, 2 noise realizations, small 96x96 target;
- development: 10 scene seeds, 3 noise realizations;
- publication: at least 30 independent scene seeds and 4 noise realizations, or a
  larger count selected by a pre-run power calculation using validation effect
  variance.

Split at the scene level: generator seeds used for hyperparameter selection never
appear in the test set. Within every scene, withhold complete exposures from fitting.
Candidate discovery uses train exposures; threshold choice/early stopping uses
validation exposures/scenes; final likelihood confirmation uses test exposures.

### Real JWST SMACS J0723 injection/recovery

The local fixture at `data/jwst_smacs0723/` currently contains 135 calibrated
files. A read-only header audit found nine dither positions in each of F090W, F150W,
F200W, F277W, F356W, and F444W; the three short-wave bands cover four B-module
detectors while each long-wave band uses NRCBLONG. This is enough for a six-band
experiment, although the existing joint benchmark defaults to four bands.

Use the calibrated `SCI`, `ERR`, and `DQ` planes and inject sources into detector
images through each exposure's WCS, tabulated throughput, pixel footprint, and a
measured/approved PSF. Preserve real backgrounds, masks, correlated noise, and
artifacts. Do not inject into coadds.

Pre-register the dither split by `PATT_NUM` so simultaneous short/long observations
stay together:

```text
train = {1, 2, 4, 5, 7}
validation = {3, 8}
test = {6, 9}
```

Before fitting, verify that each band has non-zero common support in all three
partitions for the chosen crop. If it does not, select a crop using only headers
and coverage—not science pixels—or reduce the claim to the supported bands.

Use sparse injections (less than 5% added-source areal occupancy per run) and many
independent injection batches so synthetic sources do not change crowding or
background statistics unrealistically. Include blank runs and sign-flipped/negative
source injections to measure false positives and flux boosting. Mask a small radius
around bright real objects for the isolated-source stratum, then run a separate
crowded stratum without that mask.

The public UPdec/reference mosaics are useful for registration and visual QA only.
The decisive real-data evidence is recovery of known detector-level injections and
prediction of held-out calibrated exposures.

## Metrics

### Primary detection metrics

- completeness and purity versus truth normalization magnitude and SED stratum;
- `m50` and `m90`, from a binomial sigmoid fit to per-source recovery rather than
  interpolating noisy histogram bins;
- paired depth gain `Delta m50 = m50_joint - m50_best_independent` at 95% measured
  purity;
- precision-recall curve and area under it;
- false detections per square arcminute, including duplicates;
- astrometric median, 68th, and 95th percentile angular error;
- close-pair resolution/completeness versus separation in PSF FWHM.

Bootstrap whole scene/injection-batch identifiers, not individual sources, because
sources in one image share noise and fit parameters. Report paired 95% confidence
intervals with at least 10,000 bootstrap resamples in the publication report.

### Flux and color

Report both all-injection forced measurements and detected-only measurements:

- median fractional flux bias, NMAD, and catastrophic outlier fraction per band;
- magnitude bias versus truth magnitude;
- color bias/NMAD for every adjacent band pair and selected long-baseline colors;
- band-to-band residual covariance;
- flux leakage into truth-zero dropout/line-free bands;
- total-flux and surface-brightness bias for extended objects;
- blend flux-allocation error and centroid exchange rate for close pairs.

### Uncertainty and data consistency

- empirical coverage of nominal 50%, 68%, 90%, and 95% flux/color intervals;
- interval width, calibration error, Gaussian NLL or CRPS, and standardized-residual
  mean/std;
- held-out detector-pixel reduced chi-square and predictive NLL;
- held-out residual autocorrelation / whitened power spectrum;
- change in held-out likelihood when each detected source component is removed;
- support versus extrapolation uncertainty, retained as a diagnostic but not used
  as a substitute for coverage.

For deterministic methods, obtain uncertainty from a matched visit/bootstrap
ensemble or analytic forced-photometry covariance. A method with no meaningful
interval must be marked “uncertainty unavailable”; do not treat zero variance as a
successful result. Current `forward_distribution` support statistics are not yet a
coverage test.

### Cost and scale

- fit, render, and detection wall time separately;
- peak CPU RSS and accelerator memory;
- trainable parameters, Foam atom count, queried candidate pairs, and effective
  pixels/s;
- performance versus image area, exposure count, band count, and source density;
- quality/cost Pareto frontier for each model family.

## Ablations

Run paired ablations with identical seeds:

1. joint spectral model versus one model per band;
2. exact/tabulated bandpasses versus center wavelengths;
3. pixel-center versus 2x2 pixel-footprint rendering;
4. known PSF versus deliberately perturbed PSF;
5. fixed WCS versus gauge-anchored pose refinement;
6. shared Foam geometry versus independent per-band Foam geometry;
7. fixed achromatic atom geometry versus chromatic low-rank atom features;
8. diffuse INR only, Foam only, and INR+Foam;
9. dense all-atom ownership versus sparse candidate ownership, with exact parity on
   a small case;
10. likelihood choice (`mse`, robust loss, Gaussian NLL);
11. full six bands versus leave-one-band-out fits;
12. continuum-only training versus mixed SED population;
13. correct versus misspecified PSF/WCS/calibration/background;
14. capacity-matched versus compute-matched comparisons.

The leave-one-band-out render is an interpolation test, not a depth result. Report
it separately and never use an unobserved wavelength prediction as detection truth.

## Acceptance and stopping gates

### Gate 0: harness correctness

- deterministic manifests and results for a fixed seed;
- zero-source/null scenes have measured false-positive rates consistent with the
  validation-calibrated threshold;
- noiseless bright injections recover position and flux to the truth integrator's
  stated tolerance;
- truth association is one-to-one and tested on duplicates/close pairs;
- train/validation/test exposure identifiers are disjoint;
- shuffled band labels remove the expected continuum-sharing gain.

Stop and fix the harness if any Gate 0 item fails.

### Gate 1: synthetic scientific claim

- lower 95% paired bootstrap endpoint of continuum `Delta m50` is greater than zero
  at test purity >= 95%;
- test purity's lower 95% confidence bound is >= 93% (allowing finite-sample
  variation around the validation target);
- bright-source median absolute flux bias is <= 2% in every band and no worse than
  the matched independent baseline by more than 1 percentage point;
- adjacent-color median bias is <= 0.02 mag for smooth-continuum sources;
- nominal 68%/95% interval coverage is within 5 percentage points after calibration
  on validation scenes;
- held-out predictive NLL improves over the matched independent baseline, or is
  statistically indistinguishable while depth improves;
- dropout/emission-line `m50` is not worse than the relevant best-band baseline by
  more than 0.1 mag, and truth-zero-band leakage remains consistent with uncertainty.

The numerical fidelity bounds above are pre-registered engineering targets. If a
baseline already fails them because of an invalid simulator/noise contract, revise
the harness and record the reason before evaluating joint methods; do not relax a
bound after seeing the test winner.

### Gate 2: real-image injection claim

- the continuum depth gain remains positive with a scene/injection-batch bootstrap
  interval excluding zero;
- blank-run purity and negative-injection controls agree with synthetic calibration;
- held-out dither likelihood and residual whiteness do not degrade;
- gains appear in at least two spatial/background strata, not only one hand-picked
  crop;
- flux/color biases remain within Gate 1 bounds or a documented instrument-level
  systematic floor measured from bright injections.

### Gate 3: scale and release evidence

- sparse Foam candidate execution is numerically equivalent to dense ownership on
  the small exact case;
- peak memory scales with configured tile/candidate budget, not output area times
  all atoms;
- one 512x512 six-band, multi-exposure development case completes on the target
  CANFAR GPU without OOM;
- JSON includes environment, Torch/Torchsky/Torchfits revisions, device, seeds,
  model configuration, split manifest, and all failed runs;
- a committed Markdown report is generated from JSON, not edited by hand.

## Minimal implementation sequence

### M0 — scorer and synthetic truth catalogue

Candidate files:

- new `benchmarks/bench_multiband_depth.py`;
- new `tests/test_multiband_depth_benchmark.py` for null rate, source matching,
  noiseless flux, split disjointness, and deterministic artifact checks;
- update `docs/benchmarks.md` only after the smoke result exists.

Reuse observation/spec builders from `benchmarks/bench_map_field_fit.py` where they
are stable. Do not move benchmark-only source population or detector code into
`src/torchsky`.

### M1 — fair classical and independent-INR baselines

Add bilinear, drizzle, optional SWarp, chi-square coadd detection, and independent
per-band INR rows. Freeze thresholds and hyperparameters on development seeds.

### M2 — joint INR result

Add current recommended spectral INR kinds using the same common scorer. The first
publishable plot is completeness/purity plus flux/color calibration, not a PNG.

### M3 — Power-Foam and hybrid

Once the production Foam primitive has dense/sparse parity and lifecycle tests, add
Foam-only and INR+Foam rows. Record atom count, candidate pairs, split/merge history,
and ownership entropy/overlap diagnostics in the artifact.

### M4 — SMACS detector-level injection

Reuse discovery/header/auxiliary helpers from
`benchmarks/bench_map_smacs0723_field_fit_joint.py`, but write a separate benchmark
so the current real-data regression remains stable. FITS reads/writes stay through
Torchfits/Torchsky boundaries.

### M5 — scorecard and gate

Candidate files:

- `benchmarks/baselines/multiband_depth_smoke.json` only after repeated stable runs;
- generated `benchmark_results/multiband_depth_*.json` and `.md` locally/CANFAR;
- Pixi tasks `bench-multiband-depth-smoke`, `bench-multiband-depth`, and
  `bench-multiband-depth-smacs` after commands stabilize.

Do not put the publication profile in `ci-local`; it is an explicit human/CANFAR
benchmark. CI gets deterministic helper tests and, optionally, the tiny smoke gate.

## Exact candidate commands

Current joint-field floor, using existing code:

```bash
cd /Users/fabbros/src/torchsky
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py \
  --size 48 --steps 120 --runs 1 \
  --cases tpv_field_fit_deep_multiband_broad \
  --models coadd_bilinear,sky_lambda_feature_field,hashgrid_spectral_field \
  --use-spectral-budget \
  --json-out benchmark_results/field_fit_multiband_floor.json
```

Current six-band real exposure inventory and joint-fit smoke:

```bash
cd /Users/fabbros/src/torchsky
pixi run -e sphere-bench python benchmarks/bench_map_smacs0723_field_fit_joint.py \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --input-product cal --module b --crop-size 32 --steps 4 \
  --models coadd_bilinear,sky_lambda_feature_field \
  --skip-missing \
  --json-out data/benchmark_results/smacs0723_field_fit_joint_6band_smoke.json \
  --markdown-out data/benchmark_results/smacs0723_field_fit_joint_6band_smoke.md
```

Planned deterministic tests:

```bash
cd /Users/fabbros/src/torchsky
pixi run -e test -- pytest tests/test_multiband_depth_benchmark.py -q
```

Current production-operator classical floor:

```bash
cd /Users/fabbros/src/torchsky
pixi run -e sphere-bench python benchmarks/bench_multiband_depth.py \
  --profile classical-smoke --device cpu --seed 101 --size 20 \
  --bands 3 --sources 3 --exposures 2 \
  --json-out benchmark_results/multiband_depth_classical_tiny.json
```

The tiny command is the local correctness gate. The exact square-Drizzle
96-square/six-band/four-exposure run, including its separate null product, is a
CANFAR accelerator gate because the exact polygon clipping path is intentionally
not replaced by bilinear interpolation for convenience. The three-seed/two-noise
smoke command is added only after the independent and shared Foam rows exist.

Planned CANFAR publication run:

```bash
cd /path/to/torchsky
pixi run -e sphere-bench python benchmarks/bench_multiband_depth.py \
  --profile publication --device cuda --seed-start 1000 --num-seeds 30 \
  --noise-realizations 4 \
  --methods bilinear,drizzle,chi2_coadd,independent_inr,joint_inr,joint_foam,hybrid \
  --json-out benchmark_results/multiband_depth_publication.json \
  --markdown-out benchmark_results/multiband_depth_publication.md
```

Planned real-image injection run:

```bash
cd /path/to/torchsky
TORCHSKY_DATA=/path/to/torchsky/data \
pixi run -e sphere-bench python benchmarks/bench_multiband_depth.py \
  --profile smacs-injection --device cuda \
  --input-root "$TORCHSKY_DATA/jwst_smacs0723" \
  --bands F090W,F150W,F200W,F277W,F356W,F444W \
  --train-patt-num 1,2,4,5,7 --validation-patt-num 3,8 --test-patt-num 6,9 \
  --injection-batches 20 \
  --methods bilinear,drizzle,chi2_coadd,independent_inr,joint_inr,joint_foam,hybrid \
  --json-out benchmark_results/multiband_depth_smacs.json \
  --markdown-out benchmark_results/multiband_depth_smacs.md
```

The `bench_multiband_depth.py` commands are a proposed CLI contract; implement and
test them before documenting them as shipped Pixi tasks.

## Expected report

The generated report should contain:

1. a method/configuration and fairness table;
2. completeness and purity curves by SED/morphology stratum;
3. paired `m50`/`m90` depth gains with confidence intervals;
4. flux and color bias/scatter curves;
5. uncertainty coverage and held-out predictive metrics;
6. null/negative-injection controls;
7. PSF/WCS/bandpass misspecification ablations;
8. time/memory/parameter/atom scaling;
9. representative input, truth, reconstruction, and residual images;
10. failures and unsupported claims.

The headline is allowed only if Gate 2 passes. Otherwise the report should state
which source classes benefit, which fail, and whether the limitation is the
representation, observation model, uncertainty, or detector.

## Primary methodological references

- Szalay, Connolly & Szokoly, *Simultaneous Multicolor Detection of Faint Galaxies
  in the Hubble Deep Field*: https://arxiv.org/abs/astro-ph/9811086
- Melchior et al., *SCARLET: Source separation in multi-band images by Constrained
  Matrix Factorization*: https://arxiv.org/abs/1802.10157
- Chiu, Desai & Liu, *ComEst: a Completeness Estimator of Source Extraction on
  Astronomical Imaging*: https://arxiv.org/abs/1605.02650
- LSST DESC, *The DC2 Simulated Sky Survey*: https://arxiv.org/abs/2010.05926

These motivate multiband detection, shared-morphology scene modelling, direct
real-image injection/recovery, and truth-catalogue end-to-end validation. Torchsky's
distinct contribution should be a continuous, differentiable, exposure-level
multi-wavelength scene with modern INR/Foam representations—not merely reproducing
their software boundaries.
