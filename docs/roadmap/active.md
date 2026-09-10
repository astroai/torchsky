# Active Torchsky Execution Checklist

CANFAR execution uses **production only**, as requested on 2026-09-09.
Do not submit jobs to staging or use it as a fallback.

## Production follow-up — 2026-09-10

Staging session `lbjhoo6j` never started: its events reported a missing storage
mount secret. It was cancelled before the production-only instruction.
Production replacement `byso1yb5` was created successfully but is now absent
from session and event APIs. Its ARC directory contains only the uploaded
bootstrap and source bundle, with no execution log, exit record or science
products. The cause is unknown; this does not demonstrate another GPU-memory
failure. The source bundle verifies locally as complete at `3b92e55d`.
The next attempt records startup output and exit status before cloning the
bundle, in its separate `depth-20260910-softtiles-r2` directory.
That retry, production session `eihik1s9`, failed before container startup.
Its events show successful scheduling and image pull, followed by
`failed to initialize NVML: Driver/library version mismatch` on
`keel-prod-g0123.arbutus.uvic.ca`. Torchsky and the bootstrap never ran.
This is a confirmed host/runtime failure, not a reconstruction failure;
the smaller-tile CUDA correction remains unverified on real SMACS data.
One production retry, `pt215xlb`, landed on the same host and failed with the
same pre-start NVIDIA mismatch. Further identical submissions are paused until
the host is repaired or a healthy allocation is available. The launcher now
selects production explicitly before submitting, rather than inheriting a
possibly stale local server choice. No science run is currently in progress.

## Reconstruction work — 2026-09-07

Use the existing AstroAI improc image for SMACS comparison tools.
The reported SWarp build was official SWarp, not Torchsky's implementation;
rebuilding it was unnecessary. Depth launches now default to this image,
mosaic comparisons no longer build external tools, and Montage discovery
also checks the image's PATH. The previous CPU build session `isy3usqe` is
no longer present; no replacement build was launched.

Checkpoint `e085b5a3` is pushed to fork main: 107 focused tests pass,
preflight passes, and typecheck reports no errors (one existing warning).
CANFAR rejected both `astroai/improc:latest` and its registry-qualified form
as unknown/private. These requests failed, but that does not establish that
the image name is wrong: the image listing omits headless-only labelled images.
The user-confirmed `astroai/improc:latest` remains the depth default. The
temporary substitution of `astroai/improc-notebook:latest` was incorrect.
One Torchsky-only
GPU validation session `my8tv91c`, run `depth-20260908-032847`, was created from
`e085b5a3` using the previously working base notebook image. It uses frozen
stationary calibration, measured passbands, two bands, a 32-pixel crop and two
optimization steps. This checks real-data execution, not scientific convergence.
No external comparison tools were built or invoked by this run. Follow-up
found its ARC `b/` output directory empty and session/status/events unavailable
(404). The persistent log ends after the CUDA-environment Torchsky build;
there is no recorded scientific result or termination reason. Do not call
this an out-of-memory failure without evidence.

Fork CI is fully green at `ba862f17` (run `34183869145`), including all map,
core and sphere jobs. The follow-up implementation removes avoidable retention
of full detector SCI/noise arrays while preparing small crops, skips unused
validation auxiliary FITS writes, and keeps validation bands without training
overlap out of scoring. Frozen validation operators move to the scene device.
The runner now records commands and ordinary exit status, enables Python fault
tracebacks, and builds directly in the CUDA environment for Foam runs. A shell
SIGKILL still cannot write an exit record. These changes address concrete
resource and diagnostic limitations; they do not establish the old run's cause.
The changes pass 53 focused tests, preflight and type checking (one existing
export-list warning). Whole-band injection overrides still retain full arrays;
the preparation memory improvement currently applies to uninjected science.
Checkpoint `433c1db6` contains these fixes. The replacement two-band validation
run `depth-20260908-054220` (session `y2l4autn`) was created from that commit
using the working base image after the improc launch request again returned
HTTP 400. The configured default remains `astroai/improc:latest`. The new run
is still setting up; no reconstruction result has been measured yet.
The plain-text report now places held-out detector scores before secondary
reference-image comparisons and labels missing scores “not measured.”

The replacement run recorded exit code 1 after the benchmark command, with
no score products. Inspection found the direct cause: the script entry point
only called `main()` when `TORCHSKY_GPU_MEM_LOG=1`. Ordinary invocations exited
successfully without fitting; the runner then failed its missing-output check.
The normal entry point is restored, with subprocess checks that `--help`
prints options and invalid arguments fail without memory logging enabled.
This explains the observed empty-output behavior without attributing it to
memory exhaustion. Historical six-band native Drizzle products were also
located on ARC; historical Foam products had poor source-flux recovery and
are diagnostic evidence, not promoted reconstructions.

The repaired-entry run is `depth-20260908-152136`, session `qx6jqxkw`, from
`7ef600a5`; it passed CUDA import/device checks and entered the benchmark
command. Science results are pending. An independent analytic Gaussian
continuum check now tests the default linear-affine Foam with pixel integration
and a PSF: both cold Adam and coadd/whitening controls recover aperture flux
within 1% after the existing exact refinement. The positive-log variant
underconverged at 200 steps and improved at 800 in a separate experiment;
this does not reproduce or resolve the historical real-data flux deficit.
Real masks, measured passbands, independent PSFs and larger injection batches
remain necessary before making a SMACS recovery claim.

Historical uninjected Drizzle science and weights from
`20260729-smacs-torchsky-drizzle-aa0aa21/blank` were recovered for all six bands.
They share a 512-pixel WCS at 0.06300066 arcsec/pixel. Local products under
`benchmarks_results/recovered_smacs_20260908/` include original FITS files,
SHA256 provenance, a masked six-band FITS cube and Torchfits RGB in
`quicklooks_v2/`. They are labelled as historical, not new neural results;
unmeasured metrics are not inferred from the image appearance.

The repaired run completed successfully with exit code 0. It exported two-band
FITS products and validated on two overlapping exposures per band, evaluating
no test exposures. F090W has 9,504 valid validation pixels, mean Gaussian NLL
-2.162687 and chi-squared 1.513361 per pixel; F150W has 9,249 pixels, NLL
-2.516363 and chi-squared 1.237603. The density is expressed in the original
MJy/sr brightness units. Two optimization steps took 17.08 seconds, with
72,892,928 bytes peak tracked CUDA allocation. This is execution evidence:
the source-flux comparison shows a severely underfitted scene, not a usable
neural map. Nominal Gaussian widths in subsequent reports are explicitly
labelled as assumptions, not measured or fitted PSF widths.

The next pilot uses a single predeclared 128-pixel common target grid centered
on the historical crop, six measured bands, 64-by-64 fixed Foam cells, six
spectral modes, affine spatial coefficients, coadd initialization and spectral
basis whitening, then 200 Adam steps with stationary independent calibration.
The target FITS is stored at
`/arc/projects/ots/torchsky-runs/20260908-six-band-pilot-inputs/target_128.fits`,
SHA256 `46842f1b9edbdadc717199367434dfe8faf40130f245aab2ddda729ac3b423c2`.
Its upload was read back and hash-verified. The launcher now forwards these
existing benchmark controls. This pilot remains a cropped experiment, not
the final cluster reconstruction or the full injection comparison.

This pilot is running as session `z93v5gab`, run
`depth-20260908-202557`, from fork commit `5dc782d3`. The Torchsky-only job
uses the explicitly selected working base notebook image; the comparison
image default remains `astroai/improc:latest`. The run failed during training.
The inspected log recorded CUDA out-of-memory failures on the 11.54 GB slice,
then an NVML allocator assertion in the Foam support-weight calculation.
No completed map was produced; exact refinement was off.

The follow-up preserves the 128-pixel training patch and 64-by-64 model cells,
but renders and backpropagates 16-pixel cores with PSF margins. The per-call
point-by-cell limit is a working-memory control, not a total GPU-memory bound.
Unrefined Foam exports now retain the trained soft field instead of silently
switching to hard cell ownership. These are target-grid projections with an
assumed target PSF; validation scores describe calibrated detector predictions.
Hard export remains reserved for explicitly exact-refined models.

Checkpoint `3b92e55d` contains the correction. All 67 targeted sampling,
joint-benchmark and streamed-render tests pass; preflight passes and type
checking has zero errors (one existing warning). Replacement session `lbjhoo6j`
was created on the currently selected CANFAR **staging** server and is Pending.
Its run directory is `depth-20260909-softtiles-3b92e55d`. It retains the same
six-band pilot configuration and starts from a Git bundle of the pushed commit
on ARC, without passing GitHub credentials to the job. No replacement science
result has been measured yet.

Fork checkpoint `25d7ae12` passed Comprehensive CI `34275561108` and the
WCS/sphere weekly run `34349930928`. Those checks do not establish CUDA viability
for this larger SMACS fit.

Joint fits now save registered scene models immediately after fitting, before
validation and image export. Artifacts include optimizer and sampling state,
training normalization, run configuration, and hashes of detector inputs,
passbands and frozen calibration references. Loading a trained Foam reproduces
its predictions in the focused tests. Replaying detector predictions still
requires rebuilding observations and operators from those verified references;
there is no automatic benchmark resume command yet. Unregistered exposure-bias
wrappers are explicitly unsupported, unfrozen PSF/noise extras are marked as
scene-only exports, and exact-refined scenes omit stale optimizer state.
The artifact and sampling tests pass (23 tests); type checking reports zero
errors and the existing export-list warning.
The independent numerical review confirmed that exact observation refinement
cannot consume the calibrated PSF/background bundles used by Adam. The
existing rejection now also happens during argument checking, before reading
data or fitting. Measured passbands alone remain supported by that refiner;
consistency of its quadrature does not establish spectral convergence.

Structured-noise training now supports PSF-aware tiled recomputation. It
evaluates the likelihood on the full detector image, then re-renders each
tile to propagate the full-image likelihood gradient. This preserves
cross-tile noise correlations and bounds field activations by tile size;
prediction and noise tensors still scale with image size. CPU tests compare
scene, noise and background gradients across all four reductions, replay
random draws, and check saved activation memory. CUDA AMP and MPS remain
unverified, and rendering must not mutate model buffers.

Every reconstruction supplied with a calibration manifest now excludes
validation and test exposures before coaddition, domain construction and
fitting. Previously this depended on the independent-calibration flag.
Unknown splits and selections without training data are rejected.

The joint benchmark now scores frozen model predictions on validation
detector crops, using all valid pixels and the supplied ERR planes. It
records Gaussian NLL and chi-squared per band and exposure, reverses the
training-only numerical scale in the likelihood density, and never renders
test exposures for selection. This is a diagonal-noise score: correlations,
the actual photometric reference spectrum and pixel-response calibration
still need independent verification. Training-specific free-offset models
are explicitly excluded from this scoring path. INR configurations now use
PSF-aware streamed tiles instead of incompatible independent pixel samples.

CI at `9c467265` passed every map group and sphere spectral checks. The
remaining failures were a FITS facade unnecessarily importing Torchsky's
native extension, and the old small-cutout visualization test lacking
physical units. The facade now gets its FITS capability flags directly
from Torchfits; the test supplies units and expects RGB to remain pending
when only two bands are available. Remote verification of these repairs
is pending.

## Branch consolidation and SMACS implementation — 2026-09-05

The accepted plan is the 2026-09-05 section of `next.md`. Work now targets
`main` on `sfabbro/torchsky`, tracking `origin/main`; upstream is read-only.
Older exact-SHA runs below are historical evidence, not a requirement to launch
new jobs from old code. New jobs must name an exact pushed commit.

Starting tips: main `b4356594`, local SMACS `15ae3d3b`, remote SMACS `ff2afba2`,
SVO `95751d02`, WCS integration `849d0784`, field research `612e4cf8`, stale UPDC
`fb77669a`. Commit `03064951` preserves the three pre-existing documentation
edits before switching branches. SMACS merge `7aea4b34` retains the newer docs;
SVO merge `01b41f6f` records equivalent already-applied work. Research merge
`7b0c684a` retains the corrected finite-bandwidth field implementation and its
expanded tests rather than the earlier Softplus version.

Legacy reconciliation retains history and ports all-observation MAST product
discovery/deduplication into the existing downloader, plus exposure/PSF manifest
pairing. The parallel WCS-aware trainer, embedded `src/torchfits`, notebooks,
backups and unconfigured HybPSF gitlink are superseded or unsuitable as current
production code. Useful HybPSF calibration, UPDC rejection-mask handling and
pixel-complete epoch sampling need explicit current-API ports; current epoch
sampling cycles observations and is not yet equivalent to the legacy sampler.

Before implementation, focused WCS/sphere/catalog tests passed 257 with 11
skipped. WCS and sphere code match starting main; catalog changes are result
aliases. This used the current Pixi environment, not a fresh main environment.
Exact-main GitHub core and sphere spectral jobs passed, but lint and two broad
map spectral tests failed. The integrated SMACS branch already pins Ruff to the
working 0.15 series: its exact fast-lane Ruff command now passes. Both previously
failing broad spectral cases pass locally after integration (63.64-second run,
8 passing checks; a separate new manifest-path assertion failed and was corrected
to account for macOS temporary-directory symlinks). Remote CI is still pending.

ARC recovery found all four September launches failed before science because
the loaded system `libstdc++.so.6` lacks `CXXABI_1.3.15`. Both August 12 exact
Foam jobs exceeded the uniform-grid candidate-pair guard (12,015,480 versus
12,000,000), not GPU memory. These runs are not successful science evidence.

All inspected branch references are ancestors of fork main `e6be1ba0`, pushed
without rewriting history. Focused merge checks passed 40 tests with four device
skips. Runtime repair now selects each Pixi environment's own C++ library after
activation, disables user-site Python packages, and prints loaded libraries and
GPU information before science. Its shell behavior and launcher checks pass
10 tests. Run `depth-20260906-052224`, from pushed commit `cc822530`, confirmed
Torchsky and Torchfits imports on Linux and loaded Pixi's `libstdc++.so.6.0.36`.
It detected an H100 NVL MIG 1g.12gb device with 10.64 GiB free. Science did not
start: SWarp configuration could not link `ffopen` from CFITSIO. The compiler's
detailed configuration log must be retained before diagnosing this failure.

The next implementation passes 137 focused field tests with 10 device skips.
PSF-aware tiled residual training matches full predictions and gradients for
all four loss reductions tested. A CPU saved-tensor check measured 91,624 bytes
for both 40- and 80-row tiled images versus 1,892,572 bytes for the 80-row full
image. This measures saved activations, not process/GPU peak memory. Structured
noise and learned-variance losses still require full-observation evaluation;
recomputation for those losses remains unfinished. Pixel-only sampling now
rejects PSF or non-center quadrature settings it previously failed to honor.

The measured-passband adapter preserves calibration metadata and rescales
errors, variances and background planes consistently. It explicitly assumes
constant-I_nu photon-response normalization, numerically checked against an
analytic band integral. Matching the actual CRDS photometric reference-spectrum
convention remains required before claiming calibrated colors. The misleading
background option `plane` is replaced by `none`; a real independently fitted
plane remains separate work. Frozen calibration can use stationary or adaptive
families and derives numerical normalization from training exposures only.

Implementation checkpoint `8b02d5cb` is pushed to fork main. The subsequent
numerical review also fixed untiled diagnostics, zero pixel-sample handling and
unsupported full-image auxiliary losses; the expanded tile checks pass 14 tests.
CPU build diagnostic `depth-20260907-020758` captured the actual SWarp failure:
`/opt/conda/bin/ld` selected its sysroot's incompatible `libpthread.so.0`, producing
undefined `GLIBC_PRIVATE` symbols. Repair `a2f8c6a1` binds the compiler and linker
to host tools. CPU job `depth-20260907-021433` successfully configured, built and
installed official SWarp 2.42.0. This is build evidence; astronomical comparison
results still require the next GPU run.

The visualization update passes three checks using synthetic FITS cubes. It
produces common-scale linear/asinh panels, signed reference differences,
explicit missing metrics, fixed-region reports and six-band Torchfits RGB using
intersection coverage. Reference checksums protect reused display limits, and
completed report metadata cannot be overwritten. These are renderer checks,
not real SMACS products or scientific-fidelity measurements. Native RGB is
implemented; independently resolution-matched RGB, evolving metric histories,
diagnostic training snapshots, and final six-band measurements remain pending.

Further CI investigation reproduced eight deterministic-render benchmark errors:
the reports assumed every product had an uncertainty-derived weight. These now
evaluate uniformly on rendered support when no uncertainty exists, without
inventing an uncertainty plane. The minimal operator CI environment also lacked
the required Torchfits dependency. The Lanczos discrepancy was float32 reduction
roundoff (2.38e-7 versus a 2e-7 assertion). Its normalization already preserves a
constant mathematically. A trial extra normalization cost about 35% at 128x128
in a seven-repeat CPU check, so the algorithm is retained. The test now checks
both float32 and float64 against a signed-stencil floating-point error bound.

All 27 external Drizzle checks pass when its square-drop reference uses a padded
canvas and crops back to the requested WCS. With the previous tight canvas,
Drizzle 2.2's scanner omitted contributions at the rotated outer edge. In the
11-degree example, padding reduced maximum weight disagreement from 0.818608
to zero and maximum brightness disagreement to 1.19e-7. This records a real
boundary-behavior difference, not general bitwise equivalence. The independent
SMACS comparator runners still need this explicit boundary convention recorded
and applied before claiming strict full-field equivalence.

GPU job `depth-20260907-021737` (`dhkmtim0`, commit `a2f8c6a1`) passed the
Torchsky/Torchfits runtime checks and SWarp build. Its log stops during Montage's
bundled CFITSIO build; session info and events now return 404. No termination
reason or completed astronomical comparison is available. Isolate Montage in
a CPU-only comparator-build job before allocating another GPU.

The repaired truth-report suites pass all 38 tests; the expanded Lanczos
precision checks pass four cases. Preflight and type checking pass (one existing
export-list warning). Remote CI after these repairs is still required before
calling consolidated main fully healthy.

Recovered inputs: aligned 135, CAL 135, CRF 27, matching manifest filenames.
Independent calibration evidence is at
`/arc/projects/ots/torchsky-runs/20260810-smacs-auto-calibration-6ca88db-r8/evidence/calibration_evidence_manifest.json`;
its SHA256 `88f412774876ed62f537d300ad4ebced574cb1baeb24aceae084190ebe8398b4`
matches the completed
`20260811-smacs-operator-fit-3f11e6c-r3/fit/operator_calibration_fit.json` report.
It contains 135 exposures, 4,757 evidence records and the agreed PATT splits.
Twenty calibration checkpoints exist. Retain the stationary-background control:
the prior adaptive NRCB4 background test loss worsened by 4%. Earlier test scores
are historical diagnostics, not fresh blind evidence for choosing new models.
Measured SVO curves live at
`/arc/projects/ots/jwst_smacs0723/data/calibration/svo/JWST_NIRCam2025.{band}.npz`.
The July effective-PSF products are reconstruction injection responses and must
not substitute for independent detector PSFs. STPSF 2.2.0 and CRDS assets exist,
but availability alone does not establish exposure-matched PSF calibration.

## Live execution log — 2026-09-04 SMACS full-mosaic parity

Crash-recovery invariant: all current science jobs use exact pushed commit
`ff2afba21efa5126978d647990ce60b6d943307f` from
`sfabbro/smacs-map-parity`. Documentation commits may advance the branch, but
must not silently change the SHA used by the six-band comparison.

Completed and pushed implementation checkpoints:

| Commit | Change |
| --- | --- |
| `16ccba3b` | Shared map workflows, uncertainty/products, plan replay, and field foundations. |
| `05640a58` | Strict full-SMACS parity gate. |
| `273e2abd` | Forward SMACS parity inputs to CANFAR. |
| `2cfdf856` | Probe SMACS headers through Torchfits. |
| `399e8ad5` | Bound full-mosaic execution. |
| `23b93bdf` | Run parity in the CUDA comparator environment. |
| `a5bcb380` | Accept currently authorized CANFAR compute images. |
| `19a1d100` | Preserve base-image tools in the launcher `PATH`. |
| `fd7fc9cc` | Refresh the persisted GitHub credential helper. |
| `57393355` | Pin Pixi 0.78.0 for reproducible CANFAR jobs. |
| `ff2afba2` | Build the SMACS mosaic directly in the selected CUDA environment. |

Local evidence at `ff2afba2`: focused launcher/runner tests pass (latest set:
9 passed), `pixi run preflight-push` passes, and commit hooks report Ruff and
Pyright green. This verifies the launch contract, not the astronomical parity
result.

Official comparator contract:

- STScI `drizzle==2.2.0` compiled library API versus exact square Torchsky
  Drizzle with `pixfrac=0.8`;
- source-built SWarp executable at
  `bf4f496f18c04a8d32022b45449ef8675ab9b3da` with `LANCZOS3` versus Torchsky
  Lanczos3;
- source-built Montage `mProject`/`mProjectPP` at
  `0fc9e5ff55ece4ee0b5d731157cf07cc31325724` versus Torchsky conservative
  polygon overlap.

The runner persists independent official and Torchsky `SCI`/`WHT` products on
the same WCS, then evaluates pixel, support, flux, provenance, and exposure-count
acceptance. It records the selected Torch device on every Torchsky row. An
unavailable or skipped comparator does not pass.

Active/recoverable jobs at the moment this entry was written:

| Band | Session | Run ID | Evidence state |
| --- | --- | --- | --- |
| F277W | `lwm8klfq` | `depth-20260904-213335` | H100 NVL/MIG 11 GB observed; locked CUDA environment installing. |
| F090W | `b02dnl1p` | `depth-20260904-213526` | Created; inspect logs before interpreting. |
| F150W | `mrfyfau0` | `depth-20260904-213553` | Created; inspect logs before interpreting. |
| F200W | unknown | unknown | Submission client was interrupted; discover the session/run before retry. |
| F356W | — | — | Not yet submitted. |
| F444W | — | — | Not yet submitted. |

Expected persistent directory for each row:
`/arc/projects/ots/torchsky-runs/<run-id>/mosaic-parity/b-<band>/`. Required
evidence is the job log and manifest, independent products, `parity.json`, and
`parity.md`. Expected input counts are 36 in each short band and 9 in each long
band.

Pre-science failures that have been diagnosed and fixed, and therefore are not
accepted parity evidence:

1. the old private AstroAI image was no longer authorized;
2. the base image's `PATH` hid `/opt/conda/bin/gh`;
3. the persistent Git credential helper referenced a missing `/usr/bin/gh`;
4. the image shipped obsolete Pixi 0.20;
5. the first runner built a CPU default environment before the CUDA one;
6. PyTorch 2.13's SymPy lane required `mpmath<1.4`.

The authorized base image plus refreshed helper, pinned Pixi, constrained
`mpmath`, fail-closed CUDA assertion, and direct `map-compat-cuda` build address
those failures. Probe sessions established an NVIDIA H100 NVL and private-repo
access. Obsolete sessions were removed; persistent failure logs remain only as
diagnostics.

Recovery commands, to run only after reading this entry and checking for an
already-created F200W session:

```bash
canfar ps --all --json
canfar logs lwm8klfq
canfar logs b02dnl1p
canfar logs mrfyfau0
```

Next gate: complete all six exact-SHA rows, require official status `ok`,
acceptance `pass`, correct counts, `compute_device=cuda`, and Torchfits-reopenable
outputs. No row had reached that accepted state when this checkpoint was
written. Numerical failures must produce a committed/pushed fix and a new
six-band exact-SHA run. Only after this gate closes should the equal-budget
field/Foam/hybrid and detector-injection science matrix resume.

## Reconciliation index — 2026-08-30

`next.md` is the authoritative remaining-work list; this file records execution
and evidence. The unchecked groups below are deliberately retained until their
gates pass. Their dispositions are: `implement` (Torchsky work remains),
`absorbed` (covered by a completed implementation or another active gate),
`downstream` (Torchfits or an application repository owns it), `conditional`
(only activated by a declared consumer/profile/benchmark), or `rejected`
(outside the ownership/scientific contract).

| Unchecked group in this file | Disposition | Required evidence |
| --- | --- | --- |
| Conditioned PSF/background identifiability, non-stationary covariance, CUDA batching, and public streaming FITS sink | implement / downstream | Held-out calibration; covariance propagation; CUDA profile; Torchfits subset-writer API and tile export. |
| MPS/sphere/delay-native axes and the remaining local Foam/INR sweeps | conditional / implement | Device parity; a FOSSIL consumer for new axes; matched compute/accuracy scorecards. |
| ADASS Day 2 official-comparator closure, Day 3 exact blank/contracted-grid/refinement gates, and Day 4 blind claim | implement / conditional | Versioned comparator runs, frozen split/config, paired blank/injection artifacts, and all preregistered claim controls. |
| Smooth Foam equivariance/MPS, larger shared spectral benchmark, multiband depth metrics, and AMP parity | implement / conditional | Complete CPU/CUDA gate plus held-out exposure/sky and uncertainty evidence; AMP additionally requires a CUDA FP32-parity/overflow scorecard. No SOTA language before replication. |
| Field-to-field replication and multi-GPU DDP execution | implement / conditional | Independent real field plus repeated scene/noise seeds; deterministic single-device/DDP parity and scaling after bounded streaming is stable. |
| FOSSIL Checkpoint 6 typed response/protected-subspace/cross-fit/full-sky/cosmology items | implement / conditional / downstream | Shared mission-neutral consumer, exact adjoints/chunk parity, then application-owned science certification. |
| Historical release/CI, CANFAR, and submission checkboxes | absorbed / implement | Historical entries stay as evidence; current release gate is rerun on the final commit and GitHub CI must be green. |

The current working tree has additionally implemented and tested plan-v2 input
identity/replay, honest map uncertainty/context/provenance products, adaptive
curved footprints, exact square-drop edge/variance semantics, workflow and
background presets, tensor-only field artifacts, and the default joint
multiband field path. Field fitting now derives and preserves each exposure's
passband; `--per-band`/`--per-filter` is the explicit independent-fit control.
Field plan-v2 replay requires and content-verifies the recorded artifact, adds
no optimizer steps, and rejects changed artifacts. Actual CLI save/resume/replay
and two-band metadata tests pass. These changes are not marked as historical
`[x]` entries here until the parent implementation commit records their exact
test/evidence references.

The field trainer now also exposes opt-in CUDA `--field-amp none|float16|bfloat16`
with local autocast/GradScaler handling, FP32 diagnostics, unscale-before-clip,
skipped-step overflow accounting, and persisted/resumable scaler state. CPU and
MPS requests fail closed. Targeted CPU training/rejection tests, pyright, and
preflight pass; the CUDA overflow regression is present but skipped on this
host, so the AMP parity/overflow gate remains open.

Power-Foam now has exposure-balanced persistent-site proposals from normalized
sky coordinates and whitened residuals, plus held-out-only per-cell residual,
contribution, and residual-position EMA buffers. Explicit densify/prune counts
use stable cell-id tie breaking, split radii by equal 2-D area, retain optimizer
state for surviving parameters, initialize child moments to zero, and reject
training-split evidence. Topology-changing state remains tensor-only and resumes
through the field artifact boundary. `tests/test_map_field_foam_model.py` passes
32 tests and the artifact/registered-model contract set passes 33. Production
initializer wiring and held-out scientific recovery remain open; this is not a
model-promotion or depth claim.

The experimental `diffuse_foam_hybrid` is a low-capacity diffuse SIREN plus a
localized non-negative spectral Foam. It exposes diffuse, Foam, and joint fit
stages, explicit Foam sparsity and component-correlation penalties, safe
tensor-only artifacts, and a final renderer that keeps Foam cells exact before
applying the common PSF/background operators. Its public scientific product is
the combined sky: component outputs remain diagnostics until component-specific
injections establish identifiability. No production preset has been promoted.

Field product uncertainty no longer inherits a fabricated unit log-variance.
Deterministic fields return `uncertainty=None`; explicit likelihood variance is
labelled `detector`. Seeded ensemble rendering processes members sequentially
and reports unbiased epistemic variance plus the mean detector component and,
when supplied, independently seeded calibration-draw variance. The product/API
set passes 40 tests. Coverage scorecards and Foam parameter posteriors remain
open.

The classical map reducer now spills exact medians and iterative sigma-clipped
weighted stacks into a capacity-checked tile-local scratch directory. Plan v2
records the exact required bytes; reduction uses bounded pixel slabs, accepted
observations define `WHT`, `NEXP`, and `CTX`, and cleanup is tested on both
success and failure. The expanded map/operator/adjoint/CLI/product run passes
278 tests with one device skip. This closes the reducer implementation gate, not the
medium/large memory scorecard, overlap-background spill, or official SWarp
`CLIPPED` comparator gate. Squared-coefficient diagonal variance now covers
direct copy, Drizzle, nearest/bilinear/bicubic, Lanczos/cubic, conservative
polygon, and point/box splat paths, with basis, half-pixel flux, tiled, and
Monte Carlo evidence. Distorted-flux quadrature aggregates coefficient rows
before squaring and passes real-SIP tiled parity. The same work removed an
incorrect absolute clamp on squared steradian pixel areas that suppressed
small-pixel flux weights and IVAR by orders of magnitude. FFT/spectral
remappers return absent uncertainty. Exact finite-aperture covariance is now
available for finite-support kernels through a tile-bounded adjoint evaluator.
Support masks and default unit coadd denominators likewise no longer masquerade
as detector-noise measurements; explicit scalar or map weights are required
for classical `IVAR`.

The propagated detector uncertainty now also carries an optional local noise-
correlation factor, written as `NCORR` with `NCORTYPE=LOCAL_ENBW` and
`NCORDEF=W1/W2`. It is the ratio of the first and squared spatial-coefficient
weight moments: identity sampling is one and a one-axis half-pixel linear shift
is two. A seeded 63-pixel aperture Monte Carlo agrees with the exact finite-
aperture factor, and Drizzle, bilinear, conservative, splat, tiled, cube, and
MEF tests pass. This is exact for locally shift-invariant large apertures and is
labelled only as a diagnostic for distorted/edge-truncated mappings. The public
finite-aperture evaluator supplies exact detector variance there: it replays the
shared finite-support geometry, accumulates source-pixel adjoints across tiles
before squaring, and passes all-kernel single-pixel checks, signed and flux-unit
source-basis oracles, spatial-IVAR and nonlinear-SIP oracles, plus a seeded
10,000-draw Monte Carlo. It rejects data-dependent backgrounds and FFT/spectral
remappers rather than reporting incomplete covariance.

Official workflow smoke comparators are now non-skippable numerical gates rather
than warning-only rows. The harness records the matching torchsky reference
variant, device/dtype/seeds, package versions, and the pinned SWarp/Montage source
commits; committed acceptance JSON enforces identity plus absolute bounds. Local
locked runs pass: Drizzle point is exact and square differs by `1.67e-12` in data
with exact weights; Montage `mProjectPP` is exact on the representative SIN case;
SWarp complete-footprint bilinear/Lanczos3 differ by at most `1.50e-5`/`6.74e-6`
in data and `1.84e-4` in weight. SWarp common-support and full-grid metrics remain
in the artifacts and expose the known border/support-policy difference instead of
hiding it. A separate non-skippable SWarp `CLIPPED` row uses five aligned exposures
with a localized high-amplitude contaminant; it agrees with the streamed torchsky
sigma-clipped product to `8.59e-5` in data and `1.44e-3` after positive-median
weight normalization on complete support. The same diagnosis found and fixed two harness/kernel defects: external
variants were previously compared only with bilinear, and Lanczos coefficients did
not preserve a constant field before edge masking. The three locked smoke cases do
not close TPV/dithered/cylindrical/all-sky, the broader SWarp clipping-control
matrix, or workflow-level science gates.

Overlap rectification now avoids both prior failure modes: it neither retains
every projected exposure in memory nor rereads every FITS exposure for every
output tile. Projected numerator/weight/variance/overlap planes spill once;
pairwise difference models accumulate directly into an anchored weighted graph
RHS; and graph corrections are solved and reduced tilewise. Scratch is exact
and capacity-checked before projection, grows linearly with observation count,
and is cleaned after success or failure. Tests cover one-read execution,
global chain connectivity across locally disconnected tiles, untiled parity,
explicit propagated IVAR, exhaustion, and cleanup. Difference models are now
compact: robust polynomials retain coefficients and mesh estimators retain the
filtered mesh, then both render only the active output tile. Custom estimators
without the compact contract fail closed in tiled execution. The updated
three-run-median four-exposure 1024-square CPU scorecard records 2.565 s, 1.635
projected Mpix/s, 83,886,080 exact scratch bytes, a 243,777,536-byte peak-RSS
delta, scene RMSE 2.3484e-6, and maximum IVAR error 2.8610e-6.

Plan-v2 construction now invokes its validation contract instead of merely
carrying validation helpers. Malformed target shapes/header dimensions,
unknown execution modes, zero/non-finite weights and calibrations, unpaired
auxiliary identities, invalid runtime devices/seeds, duplicate or unknown
product planes, and malformed field spectral metadata fail before replay.

The common field benchmark now reserves complete observations with a seeded,
recorded split before product planning, support coaddition, domain estimation,
initialization, or optimization. All registered candidates can be constructed
through the same benchmark contract; rows now record total/trainable parameters,
optimizer steps, wall time, observation indices, and held-out detector weighted
RMSE. The fast `bench-map-field-heldout-contract` run fits two of three synthetic
exposures and holds one out for SIREN, spectral Foam, and the diffuse/Foam hybrid,
with an equal-information bilinear coadd product. It also removes NE and SW sky
quadrants from every training observation before product/domain/initializer
construction, then evaluates PSF/resampling-guarded cores with verified zero
training support. Its one-step results are an execution/leakage contract, not a
ranking: the full registry is deliberately opt-in after a sequential
all-candidate smoke exceeded five minutes. Equal-parameter/equal-wall-time
bakeoffs, arbitrary held-out regions, classical detector-space prediction,
negative controls, and science injections remain open.

## Latest working-tree reconciliation — 2026-09-01

The durable fast field artifact is
`benchmark_results/map_field_heldout_contract.json` (with its Markdown summary).
It contains four rows: SIREN, shared spectral Foam, the combined diffuse/Foam
hybrid, and an equal-information bilinear coadd. Each row holds out one of three
complete exposures, removes NE and SW before any product/domain/initializer/fit
construction, and evaluates PSF/resampling-guarded cores with zero training
support. The artifact records split seed, exposure indices, guard size,
parameter counts, optimizer steps, wall time, peak memory, and held-out metrics.
This is a leakage/execution contract; it is not an equal-budget ranking or a
depth result. The registered-model constructor tests cover the full registry,
while the fast artifact intentionally does not run every registered or
experimental candidate.

The same evidence boundary applies to the map work. Finite-kernel diagonal
variance, optional local `NCORR`, and exact requested-aperture covariance are
tested, but FFT/spectral uncertainty remains open. Official Drizzle, SWarp bilinear/
Lanczos3, SWarp `CLIPPED`, and Montage smoke rows pass their locked acceptance
bounds, while distorted/dithered/cylindrical/all-sky comparator matrices and
workflow-level science closure remain open. The overlap-background scorecard
(`benchmarks/scorecards/20260831_map_overlap_background.md`) demonstrates
linear-size projection scratch, compact tile-rendered difference models, and
cleanup; the bounded overlap-background implementation gate is closed.

The next evidence sequence is: close the contracted-grid/exact-refinement blank
gates; run capacity- and compute-matched independent/shared field bakeoffs;
execute blank, detector-space, component-identifiability, and paired science
injection controls; repeat across at least three scene seeds and two noise
realizations; then replicate the selected protocol on an independent real field.
Only after those results are complete should DDP parity/scaling be measured.
Large FITS/tile output remains a downstream `torchfits` subset-writer contract,
not a Torchsky FITS implementation task.

No SOTA, limiting-depth, or production-preset claim is supported by the current
smoke rows, one-seed artifacts, one real field, image RMSE, visual sharpness, or
the locked comparator rows. Keep parent checkboxes unchecked until their full
held-out, negative-control, uncertainty, deterministic-repeat, device, and
memory gates pass; if a preregistered interval is null, report that result.

## Conditioned PSF and sky-background observation models

- [x] Add a differentiable spatial/spectral/temporal encoder with independent
  coordinate scales and Fourier bandwidths on every axis. PSF and background
  instantiate separate encoders so their physical variability scales need not
  agree.
- [x] Promote fixed variability scales to bounded adaptive multiscale banks:
  learn sparse gates and a continuous dilation independently on detector-x,
  detector-y, spectral, and MJD axes while retaining explicit physical
  minimum/maximum length contracts and unit-rescaling invariance.
- [x] Integrate adaptive-scale regularization into `fit_field_model`, deduplicate
  shared conditioners, include their parameters in gradient clipping, and
  persist learned gates, feature wavelengths, and explicitly labelled
  gate-weighted scales in fit diagnostics.
- [x] Remove hand-derived axis bounds from the normal setup path. Derive a
  conservative coverage guard patterned after regular Nyquist sampling from
  actual calibration coordinates, profile the initial bank dilation by weighted
  harmonic fit, and initialize gates from band-ablation BIC. Freeze sourced
  STScI NIRCam detector, version-7 passband, and on-orbit PSF metadata without
  confusing PSF width with focal-plane PSF-variation scale.
- [x] Add a non-stationary neural PSF as source-position-conditioned convex
  mixtures of positive, unit-sum trainable kernels. Apply it to each spectral
  quadrature plane before bandpass integration so chromatic morphology is not
  replaced by a band-average blur.
- [x] Add signed neural background residuals in either detector or sky
  coordinates, conditioned on wavelength and MJD, while preserving explicit
  sky/instrument/detector background components.
- [x] Use the same conditioned operator bundle in ordinary INR rendering,
  exact spectral Power-Foam rendering, spatially tiled rendering, and
  `fit_field_model`; optimize unique operator parameters alongside scene
  parameters and preserve detector coordinates across training crops.
- [x] Make learned-PSF tiling and patch fitting exact for finite kernel support;
  validate output and gradient parity. Fix the float32 exact-Foam containment
  tolerance exposed by tiny pixels inside much larger cells.
- [x] Establish a local four-thread CPU checkpoint for five wavelengths, four
  PSF bases, 15-square kernels, and both conditioned operators. At 128 square,
  full/tiled medians were 0.0285/0.0281 s; at 512 square they were
  0.6686/0.7346 s with identical checksums. Exact 128-square tiling therefore
  bounds activation extent at a measured 9.9% CPU time cost in the larger case;
  this is a baseline, not a CUDA or end-to-end speed claim.
- [x] Fail closed for heteroscedastic/distribution rendering with learned PSF or
  background until covariance propagation is mathematically implemented.
- [x] Add measurable calibration anchors and staged freeze/unfreeze helpers.
  PSF terms operate on normalized effective kernels, background terms operate
  on signed predicted values, both accept fixed sample weights, and explicit
  `scene`/`background`/`psf`/`joint` stages support separate fits without a
  second optimizer framework. Final calibration losses are persisted.
- [ ] Demonstrate that the staged calibration keeps diffuse scene, background,
  and PSF corrections identifiable on held-out exposures; the API and training
  invariants are complete, but real validation/test evidence is not.
- [x] Add certified joint x/y calibration fill-distance bounds. The adaptive
  axes retain conservative marginal contracts, while the calibration manifest
  records an explicit lower/upper interval over each rectangular detector
  footprint using the nearest-sample distance's 1-Lipschitz bound.
- [x] Add a Torchfits-backed JWST/NIRCam calibration-manifest boundary with
  canonical MJD UTC midpoints, zero-indexed detector coordinates, frozen
  `PATT_NUM` splits, official instrument metadata, atomic output, deterministic
  provenance hashing, and persisted two-dimensional coverage diagnostics.
- [x] Add a Torchfits-backed observational evidence extractor. A reviewed sky
  catalog can be proper-motion propagated into each detector exposure, but is
  no longer mandatory: a catalog-free GLS matched-filter path automatically
  selects isolated compact PSF targets and source-excluded background controls
  from `SCI/ERR/DQ`. Phase-registered targets, invalid-pixel weights, selection
  diagnostics, FITS products, and hashes are attached to the frozen `PATT_NUM`
  manifest. The scientific path is the complete 135 aligned CRF set; CAL
  products are not used in this experiment.
- [x] Add a split-labelled held-out scoring contract for PSF/background terms:
  normalized PSF MSE, centroid RMSE, maximum encircled-energy error, valid-pixel
  fraction, and background MSE/bias/MAE. This is evaluation plumbing, not a
  real predictive result.
- [x] Run the automatic aligned-CRF extraction and staged fit, optionally audit
  against the published UPDec PSF star and an external reviewed catalog, then
  publish validation/test scores by band, detector, position, and time. Coverage
  certifies where evidence exists; the r3 split-safe results below measure
  prediction on withheld exposures.
  - [x] Add a train-only evidence initializer and exact-commit CANFAR fit mode.
    Compare a coordinate-invariant one-kernel/one-bias control against adaptive
    detector/wavelength/time models with one training-derived loss scale across
    train, validation, and untouched test splits.
  - [x] Audit the persisted 135-CRF extraction and every referenced evidence
    hash. Session `e3yy6n9v` at `64d5994` fit all five detectors from 4,757
    accepted targets; the manifest SHA-256 is
    `88f412774876ed62f537d300ad4ebced574cb1baeb24aceae084190ebe8398b4`.
  - [x] Fit every detector/role group and publish aggregate held-out
    adaptive-to-stationary ratios. Adaptive PSF loss improved for all five
    detectors, with median 20.2% validation and 23.2% untouched-test gains.
    Adaptive background loss improved for all validation detectors and four of
    five test detectors, with median 63.6%/34.5% gains. The initial NRCB4 test
    regression was 4.0%; validation-only pruning below resolves it without
    using test evidence for selection.
  - [x] Add natural-band, training-median detector-quadrant, and training-median
    time-epoch slices. A slice is published only when both validation and test
    contain at least 20 targets; skipped slices retain their counts instead of
    producing unstable metrics.
  - [x] Run the coverage-gated slice/pruning endpoint, investigate NRCB4 and
    freeze the selected operator family before coupling it to the scene fit.
    Exact session `x9pxiw45` at `3f11e6c` used all 135 aligned CRFs and report
    `/arc/projects/ots/torchsky-runs/20260811-smacs-operator-fit-3f11e6c-r3/fit/operator_calibration_fit.json`.
    All declared NRCB4 band/quadrant/time slices met the 20-target gate. The
    unpruned failure localizes to F090W and early test exposures. Validation
    pruning removed NRCB4 temporal band 0 plus four negligible spatial bands,
    improved validation loss 16.6%, and improved test loss 43.2% relative to
    the full adaptive model; its test loss is 40.9% below the stationary
    control. Freeze adaptive raw coordinate conditioning plus detector-specific
    validation-pruned Fourier gates for backgrounds. Freeze the smooth
    coordinate-conditioned PSF family: all five detector test losses improve
    19.8--28.6%, while its near-zero Fourier gates remain effectively disabled
    and must not be interpreted as identified periodic scales.
  - [x] Publish the NRCB4 visual inspection bundle (zscale PNG contact sheets,
    per-exposure DS9 regions, compact CSV/Markdown indices, model residual
    plots, and Torchfits-written SCI/ERR/DQ FITS cubes) beside the r3 artifact.
    Exact headless session `qe322mqv` at `5847231` produced 108 displayed
    cutouts from 480 accepted NRCB4 background targets, nine split/band zscale
    contact sheets, 26 per-exposure DS9 region files, the complete prediction
    and inspection CSVs, a diagnostic PNG, and three matching Torchfits cubes
    under
    `/arc/projects/ots/torchsky-runs/20260811-smacs-nrcb4-inspection-5847231-r4/inspection/`.
    On untouched test data, validation-pruned weighted RMSE is 0.01829 versus
    0.02379 stationary and 0.02426 unpruned adaptive. The aggregate gain does
    not hide the remaining F090W risk: pruned F090W RMSE is 0.02106 versus
    0.01352 stationary, so any band-specific fallback must be selected from
    validation evidence alone before another test opening.
  - [x] Wire the frozen selection into the joint SMACS Foam/INR benchmark.
    Checkpoint hashes and manifest identity are verified; pruning is replayed
    from the validation report; PSF/background parameters start frozen; CRF
    detector and MJD context is explicit; background amplitudes follow the
    benchmark intensity normalization; and initial WCS crops retain their true
    JWST detector origin. Matched 512-square, six-band, 20-step blank sessions
    `b1922nx6` and `warc69r8` at `5847231` completed with all rows OK. Calibrated
    versus uncalibrated mean support RMSE was 0.40662 versus 0.40658 and mean
    source slope was 0.0354 versus 0.0351, while fit time rose from 297 to 531 s.
    This establishes numerical wiring/stability, not a calibration gain; do not
    spend a 200-step matrix on the underfit constant-cell representation.
- [ ] Implement learned non-stationary PSF covariance propagation, including
  cross-basis terms, before enabling Gaussian NLL for these bundles.
- [ ] Measure CPU/CUDA throughput and peak memory from small stamps through
  survey-sized tiled fields; then select basis count, kernel support, mixed
  precision, compilation, and spectral batching from evidence.
  - [x] Equal-width four-thread CPU checkpoint with five wavelengths and exact
    128-square tiling: fixed/adaptive medians are 0.0278/0.0355 s at 128 square
    and 0.6492/0.8096 s at 512 square (9/7 timed runs after warmup), with
    identical initialized outputs. The measured adaptive cost is 27.7%/24.7%;
    no acceleration claim is made.
  - [ ] Profile and batch repeated spectral-plane conditioning on CUDA before
    deciding whether compilation or a fused encoder is warranted.
- [x] Add held-out band-ablation and output-Jacobian scale diagnostics before
  interpreting gate-weighted feature scales as physical correlation lengths.
  Ablation disables one Fourier band without refitting; scalar-background
  Jacobians are exact and PSF-kernel Jacobian norms use deterministic
  Rademacher projections. Session `t123u804` at `8d87f38` shows the PSF Fourier
  gates collapsed near `3.2e-4` with negligible single-band held-out effect, so
  the PSF gains currently support smooth raw coordinate conditioning, not
  identified periodic scales. Background models select temporal Fourier bands
  strongly, but NRCB3/4 test ablations expose time-band overfitting.
- [x] Add whole-axis centering ablations and conservative validation-only
  Fourier pruning. The selector greedily removes a band only when held-out
  validation loss improves by at least 0.5%, restores the fitted model after
  evaluation, and opens test only once for the selected configuration. In r3,
  NRCB4's raw spectral and time trends remain predictive (centering them raises
  test loss 37.1% and 43.6%), while removing its short temporal Fourier band
  improves test loss 42.8%; this distinguishes smooth physical conditioning
  from a non-generalizing periodic term.
- [ ] Add axis-native adaptive variants required beyond image patches: spherical
  harmonic/needlet angular scales for full-sky FOSSIL/CMB/21-cm work and an
  optical-delay axis for interferometric spectroscopy.

## Local Foam/INR physical-fit performance (CANFAR-independent)

- [x] Preserve spatial sky, instrument, and detector background components in
  random and shared-sky training crops.
- [x] Make PSF-aware patch fitting exact on the requested loss core: render the
  finite Gaussian or kernel halo, clip only at real detector boundaries, and
  exclude halo pixels from both value and gradient-residual losses.
- [x] Add a reproducible full-observation versus PSF-halo patch control to
  `bench_map_field_fit.py`. On the local CPU, the 256-square, three-exposure
  `tan_field_fit_psf_mismatch`/`sine_implicit` case at 12 steps and three runs
  improved from 8.224 s to 0.962 s median fit time with a 32-square loss core
  (8.55x). This is a throughput result, not a depth or accuracy claim.
- [x] Avoid allocating and caching full WCS coordinate grids when diagnostic
  rendering is explicitly tiled; image, log-variance, and parameter-gradient
  parity pass. At 1024 square, 128-square diagnostic tiles reduced fit-phase
  peak RSS from 1.677 GB to 1.330 GB (20.7%) with identical final metrics.
- [x] Correct predictive-variance propagation through PSF convolution using
  squared normalized weights; test heteroscedastic INR rendering with both
  finite-kernel and Gaussian PSFs.
- [x] Stop benchmark target-product and observation-residual scoring from
  retaining autograd graphs across spatial tiles; scoring is inference-only.
  On the identical 1024-square tiled run this reduced total process peak RSS
  from 18.07 GB to 1.488 GB (91.8%) with identical final metrics.
- [ ] Add a public streaming tile sink for target-product FITS export so maps
  larger than device memory need not materialize the complete cube on device.
- [ ] Run matched-compute and matched-accuracy Foam/INR patch-size sweeps on
  CUDA after CANFAR service recovery; report peak accelerator memory from the
  existing fit performance profile.

## Current Focus Override — ADASS UPdec Depth Sprint

From 2026-07-28 through 2026-08-04, pause FOSSIL/component separation and broad
roadmap work. The sole scientific priority is the detector-level UPdec/SMACS
stacking-versus-INR/Foam depth result defined in
[`../projects/adass_updec_depth_week.md`](../projects/adass_updec_depth_week.md).

- [x] Freeze the field, target crop, six bands, exposure split, primary endpoint,
  claim gates, required comparator matrix, four-day submission sprint, and
  fallback abstract language.
- [x] Execute Day 1: real detector injection, unit/calibration closure, PSF
  contract, and tiny CPU/CUDA smoke.
  - [x] Add the stable real-data profile, distorted-WCS local pixel-area
    primitive, flux-conserving `MJy/sr` injection, frozen `PATT_NUM` split,
    weighted sigma-clipped background plane, and perturbed-PSF control.
    Evidence: the commit containing this checkpoint; 38 focused tests pass and
    the real F150W/F277W CPU artifact meets the bright-source unit/position gate.
  - [x] Run the same two-band/two-exposure smoke on CANFAR CUDA from the exact
    pushed commit and record the session, command, runtime, memory, and artifact.
    Evidence: session `jaannbqt`, commit `3bc6ca6`, `cuda-bench`, Torch 2.10.0 /
    CUDA 12.9, 33.10 s, 1,675,628 KiB peak child RSS, durable artifact
    `/arc/projects/ots/torchsky-runs/20260728-updec-day1-3bc6ca6-v3/`.
  - [x] Approve the deterministic abstract PSF endpoint: four-times
    oversampled band-dependent Gaussian truth with a deliberately 5% wider
    fitted model. WebbPSF/empirical stars remain post-abstract robustness work.
- [ ] Execute Day 2: validation-selected upstream Drizzle/SWarp plus classical
  multiband chi-square baseline.
  - [x] Freeze and locally verify the 135-file CAL filename grid; inventory 27
    existing long-channel CRFs on CANFAR.
  - [x] Add the atomic MAST staging/link utility and focused tests. It writes
    CAL files under `torchsky-data/jwst_smacs0723/cal`, links the existing CRFs
    under `crf`, links the 135 registered experiment inputs under `aligned`,
    and atomically emits `manifest.json`.
  - [x] Add the frozen aligned-input, validation-pattern, fixed-crop, upstream
    Drizzle pixfrac, and SWarp-kernel controls to the real comparator.
  - [x] Push the staging commit, complete missing downloads on CANFAR, and
    verify 135 CAL, 27 linked CRF, and 135 linked aligned entries from the
    persisted manifest.
    Evidence: headless session `tmbk7b72`, exact commit `35af22e`, completed
    successfully in 145 seconds of command time; aligned completion session
    `e74yyfec` at `46c1ce5`. The scientific runs use the 135 aligned
    `*_crf_rshift_tweakreg.fits` files. CAL is the complete fallback inventory;
    the 27 plain CRFs are the available unregistered long-channel originals.
    Manifest audit `rrkjowzw` confirms the aligned 135 are links to the
    pre-existing local `official_no_psf.../alignment/tweakreg_rshift` pipeline
    directory; the staging utility downloads only CAL products from MAST.
  - [x] Run the existing public UPdec iter0-versus-iter10 artifact comparison
    as secondary static context. Do not mix it into injected completeness:
    fixed public iter10 stacks did not receive the injected detector exposures.
    Session `yeym14dy` at exact commit `e87cb11` used Torchfits plus an analytic
    full-linear-matrix TAN alignment on all six bands. Iter10/iter0 central-crop
    gradient ratios are 1.48--2.11, Laplacian ratios are 1.29--4.22, and summed
    flux ratios are 1.0008--1.0243. Durable JSON/Markdown:
    `/arc/projects/ots/torchsky-runs/20260729-updec-iter0-vs-iter10-e87cb11/`.
  - [x] Run the validation-only Torchsky/Drizzle/SWarp backend sweep.
    Evidence: Torchsky `axix9wz3` at `803a200`, upstream Drizzle `z7t27ff0` at
    `46c1ce5`, and locked SWarp 2.41.5 `gckyhwco` at `62f572a`; durable reports
    live below `/arc/projects/ots/torchsky-runs/20260728-smacs-*`. These
    detector-reference diagnostics establish runnable backends but do not
    select a scientific kernel. Torchsky/Drizzle used the native crop while
    SWarp required its north-up TAN equivalent, so this first sweep was not an
    exact common-grid comparison.
  - [x] Add deterministic common-world manifests, transient detector-level SCI
    overrides, and persisted exact-grid products for identical classical
    injection/control runs. Evidence: commits `278735e` and `419c94a`, with 37
    focused passing tests (one optional skip).
  - [x] Run one positive 28-source/seven-level/four-quadrant batch and one
    zero-source blank through Torchsky bilinear, upstream Drizzle
    `pixfrac=1`, and SWarp bilinear. Evidence: CANFAR session `py7dl02x`,
    exact commit `419c94a`, with hashed manifests, SCI, weights, JSON, and
    Markdown under
    `/arc/projects/ots/torchsky-runs/20260728-smacs-injection-419c94a/`.
    This is a plumbing gate, not completeness/purity or kernel-selection
    evidence.
  - [x] Diagnose and repair the SWarp SIP-registration failure. Commit
    `fb779ef` translates SIP input distortion to equivalent TPV coefficients;
    exact CANFAR session `lf5xzbso` reduced the common-world median peak offset
    from about 8.57 to 0.39 pixels. Session `pnk5hrr4` stratified the audit and
    confirmed a 0.56-pixel bright-source median. Artifacts are under
    `/arc/projects/ots/torchsky-runs/20260728-smacs-swarp-sip-fb779ef/`.
    This remains a plumbing result: one bright source had no final SWarp signal
    and fixed injected-PSF flux recovery differs by kernel.
  - [x] Force every subsequent validation method onto one canonical north-up
    TAN target and make the manifest CLI use and validate that same WCS. The
    complete six-band control sweep remains the execution evidence gate.
  - [x] Freeze the intersection of all selected methods' six-band output weight
    maps, erode it by the PSF/injection footprint, and use only that mask for
    subsequent source placement.
    - [x] Add the support-intersection/erosion primitive and wire final weight
      maps into `updec-adass-manifest`; validate target WCS, persist the mask
      FITS/hash/count/fraction, and reject placements outside it. The remaining
      work is freezing the complete selected-method six-band mask.
    - [x] Produce all 42 exact-grid blank products at `3ff8604`. CANFAR session
      `kkpxfdxb` completed the products; zero-source probe `wxz900hl` measured
      only 984 pixels after nine-pixel erosion, so the current crop is retained
      as a diagnostic and not falsely frozen.
    - [x] Make support-constrained placement sample shuffled valid pixels in
      each quadrant instead of depending on low-occupancy rejection.
    - [x] Add per-method local footprint-coverage gating so isolated Drizzle
      holes can be tolerated only through an explicit pre-recovery threshold;
      strict 100% erosion remains the default and a diagnostic coverage grid is
      persisted before selection.
    - [x] Freeze 99% local footprint coverage from weights only. Session
      `vedlmudw` at `a0a40bf` retained 109,393 pixels with at least 22,175 in
      every quadrant; no recovery outcome contributed to this choice.
    - [x] Persist and exercise the selected mask in session `ry4c9xv5`; exact
      FITS SHA-256 `adc31bd3...d2dfa253`.
  - [x] Propagate or fit the method-dependent effective PSF before comparing
    bright-source flux bias or selecting a classical kernel.
    - [x] Add robust fractional-phase stacking of isolated
      injected-minus-blank responses into normalized effective PSFs, with a
      stable CLI that persists their hashes and shape diagnostics.
    - [x] Session `ry4c9xv5` at `8cc3fb9` persisted all 42 ePSFs from eight
      isolated AB=25 sources; no completeness or kernel outcome was used.
    - [x] Add exact-WCS aperture-flux recovery to the effective-PSF report so
      the existing bright batch can enforce the pre-registered 2% flux-bias
      gate rather than inferring photometry from PSF width.
    - [x] Session `q82737l1` at `aa0aa21` rescored all 42 products. The
      inverse-interpolation Torchsky candidates fail short-band flux
      conservation, so native Torchsky square-Drizzle pixfrac 0.6/0.8/1.0 is
      being calibrated in `g1ajbo0l` before any classical selection.
  - [x] Add spatial-inverse-variance matched filtering and positive six-band
    chi detection that accepts the measured per-band effective PSFs, plus
    robust blank-response whitening for resampling-correlated noise. Empirical
    purity-threshold calibration remains part of the validation run.
    - [x] Wire these primitives into one stable classical-detection profile
      with the existing Torchsky mesh sky estimator and hashed
      statistic/catalogue artifacts.
    - [x] Add a deterministic control-union profile that validates the common
      support contract, hashes every method catalogue, and covers
      cross-method duplicate peaks inside the frozen six-pixel radius.
    - [x] Harden `MeshSkyEstimator` for sparse common support by propagating
      finite neighbouring mesh values into fully masked cells. The primitive
      now stays finite on partial masks and fails closed on an entirely masked
      image; CANFAR session `bgjkc3jm` exposed this before catalogue creation.
    - [x] Add a validation aggregator that subtracts the same-method frozen
      control neighbourhoods, exact-matches incremental detections, calibrates
      the lowest pooled 95%-purity threshold, and reports completeness by AB
      magnitude without consulting blind identifiers.
  - [x] Add frozen blank-catalogue neighborhood exclusion to common-support
    manifests so science injections cannot be placed beside pre-existing
    sources.
    - [x] Session `c2iartol` at `32e0940` completed all seven method controls:
      1,882 permissive detections covered by 282 distinct six-pixel frozen
      neighbourhoods. Positive validation session `oi5mskje` uses that exact
      union.
  - [x] Run three positive validation batches plus one blank, build the
    six-band whitened chi-square detector, and freeze the winning classical
    configuration using only those validation batches.
    - [x] Session `oi5mskje` completed all three deterministic 28-source
      positive batches; session `c2iartol` supplied all seven blank controls.
      Exact products live under
      `/arc/projects/ots/torchsky-runs/20260729-smacs-validation-32e0940/`.
    - [x] Session `j4xy3vos` at `aa0aa21` selected pooled 95%-purity thresholds.
      Upstream Drizzle `pixfrac=0.8` passes the six-band flux gate and recovers
      74/84 (88.1%) sources; SWarp bilinear is the flux-valid SWarp winner at
      72/84 (85.7%). Summaries live under
      `/arc/projects/ots/torchsky-runs/20260729-smacs-validation-score-aa0aa21/`.
    - [x] Finish native Torchsky square-Drizzle effective-PSF/flux calibration.
      Sessions `kcumoz89`, `glns4y2r`, `x32kchen`, and `ys4tjw3v` completed the
      non-overlapping band fanout; aggregator `y5cbuwjy` emitted
      `/arc/projects/ots/torchsky-runs/20260729-smacs-torchsky-drizzle-aa0aa21/effective-psf-flux-report.json`.
      `pixfrac=0.8` and `1.0` both meet the absolute 2% median flux-bias gate in
      all six bands; `0.6` fails F090W at +2.19%.
    - [x] Rebuild common-support/control and all three positive validation
      batches for native `pixfrac=0.8/1.0`, select between them from validation
      recovery at fixed purity, then freeze `adass_updec_classical_config.json`.
      Do not select from effective-PSF sharpness alone. Session `yk0lwhjj` at
      scoring commit `4816987` selected native `pixfrac=0.8` from source commit
      `7612d32`: both candidates recover 73/84 sources (86.9%) and 5/12 in the
      faintest AB=30 bin at the frozen 95%-purity gate; `0.8` reaches 96.1%
      pooled purity and `1.0` reaches 100%, so the predeclared tie-break selects
      `0.8`. The frozen configuration is
      `/arc/projects/ots/torchsky-runs/20260802-smacs-native-score-4816987/score/adass_updec_classical_config.json`.
    - [x] Add the versioned headless entry point
      `scripts/run_canfar_adass_depth.sh native-band RUN_ID`, with exact-SHA
      checkout, durable logs, frozen manifests, AstroAI-image enforcement, and
      a `src`-layout import path. Launch one non-overlapping session per band
      with `SMACS_BAND` and an exact `TORCHSKY_COMMIT`. The first AstroAI
      authentication gate, session `bmvmo076` at `c4afdee`, reached the
      benchmark and exposed the missing import path before any science run;
      sessions `vjvoehq7` and `pkb3eptv` at `ba55f37` then exposed the missing
      editable native build. Session `uhhhnba0` at `c768be8` built the wheel
      successfully and exposed that the clean-environment verifier reused the
      stale pre-install interpreter. Relaunch only through the fresh-interpreter
      `scripts/run_local_sphere_bench.py` build gate.
    - [x] Complete all six non-overlapping native validation fanouts from exact
      source commit `d8fece0`: sessions `pw97y59a`, `cy3nnl82`, `ngpfssn5`,
      `r3qhpsj6`, `wt8f7twf`, and `nfk3aihr` persisted blank plus
      `validation-000/001/002` SCI/weight products for `pixfrac=0.8/1.0`.
      Every session used `images.canfar.net/astroai/notebook:latest` headless.
    - [x] Add the fail-closed native scoring entry point
      `benchmarks/score_smacs0723_native_classical.py` and runner mode
      `native-score`. It rebuilds common 99%-footprint support, rejects any
      frozen injection outside that support, reuses the measured effective PSFs
      and blank whitening, calibrates pooled 95%-purity thresholds, and hashes
      the complete configuration provenance. The real scorer retained 183,613
      pixels (70.0%) after nine-pixel erosion at 99% footprint coverage and
      accepted every frozen validation source.
    - [x] Fail the first scorer attempt before selection. Session `a0qpgq4s`
      found that the per-band native products used six different band-derived
      TAN headers. Add `--target-header-path` to the stacker and pin subsequent
      native products directly to the authoritative
      `validation-000/common-support-r9-f0p99.fits` WCS; do not relax the
      cross-band WCS check or add a post-hoc interpolation.
    - [x] Regenerate the native common-grid products after the `cbbbab4`
      fanout failed scientifically. Its 24 JSON/Markdown artifacts existed, but
      every inspected product directory was empty; F090W blank records both
      requested square-Drizzle methods as `FAIL` with no finite support. The
      benchmark previously returned exit zero after method failures, so session
      completion was not product completion. Native validation now exits
      nonzero on any selected-method error; exact square Drizzle falls back to
      a correctness-preserving full detector scan when its footprint scanner
      cannot represent distorted outer corners, and batches overlap clipping
      instead of looping over source pixels in Python. Gate a single real F090W
      blank product before repeating the other bands/batches.
      A local rotated-WCS exact-path check now completes at 256 square in
      0.309 s with 65,533 supported pixels and unit surface brightness; the
      pinned pre-change path was interrupted after 60 s on the identical input
      inside `_batch_quad_pixel_areas`, a greater-than-190x lower-bound speedup.
      The new path completes the same check at 512 square in 1.136 s with
      262,140 supported pixels. These are engineering timings, not depth
      evidence; the single-band real-data gate remains required. The headless
      runner now accepts `NATIVE_VALIDATION_BATCHES=blank` so that gate does not
      silently expand into all four frozen batches. The first strict F090W
      blank gate, session `qp8j31vb` at `2634a33`, correctly failed both
      pixfracs with zero support instead of persisting empty products. Local
      diagnosis found that the four-corner footprint scanner can reject a
      curved SIP/TPV detector whose interior overlaps the target. Distorted
      exact Drizzle now bypasses that approximation and scans the full detector
      through the batched exact kernel; a real SIP regression has finite
      support. The corresponding real F090W retry, session `hnhtlukr` at
      `4becb1f`, still produced zero support for both pixfracs, so the scanner
      was not the root cause. Stop remote retries. The next gate now includes
      Torchsky bilinear as a WCS control and persists a fixed 5-by-5
      source-to-target mapping comparison against Astropy for every exposure,
      including SIP/TPV flags, finite/inside counts, coordinate ranges, and
      maximum target-pixel disagreement. Session `yywpr350` at `09dda05`
      resolved that gate: Torchsky and Astropy agree within
      `3.46e-9` target pixel, both find the same overlapping detector samples,
      and bilinear plus both square-Drizzle variants reach `max_overlap=2`.
      The apparent zero-support failure was therefore in the benchmark adapter,
      not WCS or either projection kernel: it tested the valid explicit-grid
      product only after reprojecting its weight onto the unrelated per-band
      UPDec metric crop, then discarded the product when those grids did not
      overlap. Native generation now uses explicit `--product-only` semantics,
      validates finite support on the authoritative common grid, persists the
      SCI/weight pair before any optional reference metric, records product
      support counts, and retains method timing on downstream failures. The
      focused adapter regression passes locally. Rerun one F090W blank at this
      commit; fanout remains blocked until its three product pairs exist.
      That gate passed in AstroAI headless session `lyoh21qy` at `7612d32`:
      bilinear supports 261,494 pixels, square Drizzle supports 262,041 pixels
      at `pixfrac=0.8` and 262,092 at `pixfrac=1.0`, every method reaches two
      overlapping exposures, and all six SCI/weight files are durable under
      `/arc/projects/ots/torchsky-runs/20260802-smacs-native-f090-product-7612d32/`.
      The full blank plus validation-000/001/002 fanout completed at the same
      exact commit in sessions `i7jv3hil`, `d15df5ns`, `wt2iq4wt`, `rv96bwhn`,
      `ehrn0vzr`, and `m9mx5qh5` for F090W through F444W; each session recorded
      12/12 successful method rows and persisted every requested product under
      `/arc/projects/ots/torchsky-runs/20260802-smacs-native-{band_lower}-7612d32/`.
- [ ] Execute Day 3: capacity/compute-matched independent/shared Foam and INR;
  freeze configuration before blind identifiers.
  - [x] Add `SpectralPowerFoamField` as the minimum common-operator bridge:
    normalized local-tangent geometry and a shared Legendre spectral basis now
    run through the existing field WCS/bandpass/PSF/mask/weight fitter, with
    detached hard-pixel rendering for final products.
  - [x] Add science-pixel-independent regular-grid initialization and a
    hard-pixel observation renderer that analytically converts each target WCS
    footprint into normalized Foam geometry, restores surface-brightness
    semantics from exact pixel integrals, then applies the common PSF/background
    operators. The real joint adapter defaults explicitly to the canonical
    aligned-CRF directory and persists input product, capacity, spectral basis,
    optimizer steps, seed, parameter count, runtime, peak accelerator memory,
    and render mode. Seventeen focused adapter/benchmark contracts pass.
  - [ ] Run the shared adapter on the frozen real exposures, then add the
    capacity- and compute-matched independent per-band matrix. The real
    independent/shared products remain unchecked.
    The versioned headless entry point is
    `scripts/run_canfar_adass_depth.sh foam RUN_ID`; its first gate uses two
    bands, a 32-square crop, two optimizer steps, and a 4-square site grid
    before scaling capacity or compute. Session `iaz5l4js` at `d8fece0`
    completed the native CUDA build and exposed compressed-image `NAXIS1`
    leaking the binary-table row width into the joint overlap grid; relaunch
    only after normalizing `ZNAXIS1/2`. Session `nfhln02y` at `bcbd5d3` passed
    that gate and exposed the resulting 5.7-million-window Python crop scan;
    replace it with the equivalent vectorized integral-image reduction before
    scaling.
  - [x] Pass the first real CUDA end-to-end gate. Session `tqkr0w47` at
    `fd03e67` fit and hard-rendered a two-band, 32-square, two-step Foam with 96
    parameters and about 149 MB peak accelerator memory under
    `/arc/projects/ots/torchsky-runs/20260729-smacs-shared-foam-smoke-fd03e67/`.
    This is an execution gate only, not a depth result.
  - [ ] Re-run the capacity-matched shared/independent validation matrix through
    the sparse exact operator.
    - [x] Launch and diagnose the first 128-square matrix. Shared session
      `muha9pim` and independent sessions `h3ap2flp`, `v7hefzh3`, `ddw1i4dw`,
      `yz4qxmb4`, `xskdsbp2`, and `u6zppq7m` reached final hard rendering but
      failed closed because the adapter omitted candidate CSR (4,194,304 shared
      and 1,327,104 independent dense pairs). Major session `vrvm9kbp` exposed
      the same dense path earlier as a CUDA allocator failure. These are
      geometry-scaling failures, not scientific outcomes.
    - [x] Add a Torch-native uniform-grid polygon/disk sparse index and a
      production helper that builds conservative pixel candidates, exact sparse
      Čech neighbours, and the batched analytic power-cell/pixel operator.
      Arbitrary convex WCS polygons are supported; a 128-square/256-cell local
      check reduced 4,194,304 dense pairs to 106,504 candidates while preserving
      integrated area. Also replace the Foam-only full-frame spectral anchor
      with deterministic quadrature over sampled detector pixels.
    - [x] Complete the corrected independent 128-square execution matrix at
      `7eee607`. Sessions `akbnf5c1`, `id61mt1s`, `f9luvv7y`, `dgq3oh2u`,
      `pe4hbzgw`, and `u132cqex` all fit and hard-rendered successfully at
      about 106 MB peak accelerator memory and 5.9--7.4 s fit time per band.
      Their hard products are identically zero, so this is a sparse-operator
      execution pass and a model/configuration failure, not scientific
      evidence. Persist coefficient, geometry/target-range, and output-support
      diagnostics in the next run before changing capacity or learning rate.
    - [x] Diagnose the zero hard products before tuning. Session `boc6g6x5` at
      `cbbbab4` found zero learned coefficients because the actual target
      footprint normalized to `x=-11.05..-10.22`, `y=1.44..2.27` while the
      Foam sites occupy `[-0.94,0.94]`. The field-domain estimator used the
      first detector `CRVAL` and detector size but ignored shifted cutout
      footprints and pointing separation. This is a domain-geometry defect,
      not evidence against Foam or for a larger model.
    - [x] Re-run the two-band diagnostic after deriving the domain from the
      union of actual detector-edge WCS footprints. Session `snrobkst` at
      `51b757e` moved the target from roughly `(-11,+2)` to
      `x=-0.26..0.19`, `y=-0.22..0.23`; learned coefficients reached 0.0272,
      and both hard products have 100% nonzero pixels. This passes the geometry
      and execution gate, but its near-zero source-contrast slopes do not yet
      support a depth claim.
    - [x] Run one two-band blank fit on the exact frozen 512-square validation
      WCS and pass the execution/memory gate.
      Sessions `y139noqg` at `618f064` (all exposures) and `sv73lk92` at
      `700c3e3` (matched `PATT_NUM=3,8`) both failed before optimization:
      the diagnostic initial-loss render formed dense smooth memberships for
      every 512-square pixel at once and tripped the H100 allocator. The exact
      hard renderer was not reached. Session `omf0qjkg` at `eb6cd59` then
      completed the same matched-split blank with exact 64-square diagnostics
      and streamed 128-square PSF-aware patches: 41.3 s for 50 steps,
      186,634,752 bytes peak accelerator memory, and finite hard products with
      about 98.7% nonzero pixels.
    - [ ] Pass the separate blank scientific-fidelity gate before launching
      positive validation batches. The successful 16-square-grid execution
      run recovered source-contrast slopes near `-4.6e-6` and support-flux
      ratios of 0.0155/0.0245. Output rescaling was already correct; 256 cells
      are far too coarse for the 512-square JWST target. The next exact commit
      records intensity scale plus initial/final loss, uses the WCS-defined
      `[-0.5,0.5]` footprint bound with a 10% edge margin, and shrinks smooth
      patch/diagnostic tiles as grid capacity grows. It also scales the
      temperature curriculum as inverse cell area: the previous fixed `4..24`
      schedule becomes increasingly global as grid spacing shrinks and cannot
      identify local source structure at high capacity.
      The exact 4,096-cell control `lhztxhsb` at `1662f7c` confirmed this:
      253.7 s and 691,726,848 bytes peak H100 allocation reduced normalized
      full loss from 0.9416 to 0.8754, but mean source-contrast slope remained
      `1.18e-5` and mean support flux 0.0206. More cells under the unscaled
      temperature are not a scientific fix.
      The temperature-only H100 rerun `s0thrsmm` at `1687187` increased the
      hard prediction peak to 0.039 and mean source-contrast slope to
      `1.71e-5`, but mean support flux remained 0.0228 and the largest
      coefficient only 0.0276. This isolates the remaining 50-step
      `lr=0.001` amplitude ceiling; it is not evidence for more capacity.
      The validation adapter also now freezes sites/radii for the actual fixed
      grid control and records its coefficient learning rate. Previous 50-step
      runs at `lr=0.001` could only grow zero-initialized coefficients to about
      0.03, matching their roughly 0.01 physical prediction peaks after the
      measured 0.211 intensity scale was restored. Trainable geometry remains
      an explicit later ablation, never silently mixed into the fixed control.
      A local two-band fixed-geometry control independently confirms the scale:
      in 50 steps, `lr=0.001` reduced MSE from 2.50 to 2.34 with coefficient
      maximum 0.048, while `lr=0.02` reached 0.429 with coefficient maximum
      0.868 and recovered the expected red/blue contrast. This is an optimizer
      sanity check, not real-data depth evidence. The recovered real H100 blank
      artifact at `39a894b` then removed the amplitude ceiling: 50 steps at
      `lr=0.02` reduced full loss from 0.942 to 0.176, reached coefficient
      maximum 0.489 and mean support-flux ratio 0.392, but recovered only 2.8%
      mean source flux. It therefore still fails scientific fidelity. Its fit
      metadata exposed that the generic 1,024-pixel option bypassed the intended
      PSF-aware Foam patches; Foam now disables point sampling and explicitly
      uses shared-world patches across the selected exposures. Re-run this
      exact fixed-geometry blank before changing capacity or loss weighting.
      The first rerun, session `wamzjylm` at `1f9d7fd`, failed before loading
      data or training because the dependency-light Foam adapter imported two
      input helpers through the Astropy-dependent classical-comparator module.
      The helpers now live in the shared SMACS input module, so the next rerun
      can use the unchanged slim `smacs-cuda` environment; this failed session
      provides no scientific evidence. A follow-up forward-path audit then
      found that the real adapter left every input and target observation at
      `PSFSpec(mode="none")`; the sampler supported PSFs, but the claimed real
      PSF path was not active. The adapter now converts the frozen 5%-wider
      six-band Gaussian FWHM contract to each exposure's local Torchsky-WCS
      pixel scale, trains through that PSF with existing 2-by-2 Gauss-Legendre
      detector-pixel integration, applies the corresponding target PSF to hard
      products, and records truth/fitted FWHM plus input/target sigma and
      sampling provenance. The PSF-only commit `0c808d9` passes its full local
      gate (1,735 passed, 214 skipped, zero type errors), but its first exact
      AstroAI H100 request was rejected before pod creation with HTTP 400
      `Illegal group reference`. Admission probes isolated this to CANFAR
      interpreting shell `$name` references in submitted commands as platform
      groups, not to the image, account, resource shape, or Torchsky. Session
      `vhevkgna` then ran the corrected pixel-integrated commit `23e5ee9`
      through an encoded bootstrap with the frozen F090W/F150W blank,
      512-square target, 64-square grid, four spectral terms, 50 steps,
      `lr=0.02`, and fixed geometry. Both rows are `OK`; the recorded PSF
      widths and `gauss_legendre_2x2` sampling are exact, and CUDA peak
      allocation was 6,722,270,208 bytes. The 1,130.6 s fit only reduced loss
      from 0.9416 to 0.8195, however: mean support flux was 0.0347, mean source
      flux was -0.0297, and mean source-contrast slope was `-5.94e-4`. This is
      the first valid PSF-plus-detector-pixel result and it fails scientific
      fidelity. Dense 256-square session `j2fmxlln` was stopped after 48 minutes:
      its memory guard reduced the training patch from 128 square to 32 square,
      so it sampled 16 times fewer sky pixels per step and could not isolate
      representation capacity. Frozen-geometry Foam now has an opt-in exact
      compact-support path that rejects trainable geometry and evaluates only
      cells whose disks intersect the current point bounding box. The fair
      256-square control therefore restores 128-square patches while keeping
      every scientific setting fixed. Exact H100 session `mjsil8q2` completed
      commit `42f0744` with `foam_compact_fixed_geometry=true` and the matched
      128-square patch. Fit time fell from 1,130.6 s to 172.7 s and peak CUDA
      allocation from 6,722,270,208 to 937,856,512 bytes. Capacity did not close
      the science gate: final loss was 0.8892, mean support flux 0.0332, mean
      source flux 0.0802, and mean source-contrast slope `4.68e-4`; only 49.62%
      of hard-render pixels were nonzero. Commit `4829a0a` added an opt-in
      coverage-aware coadd initializer using the exact bandpass basis and a
      ridge-stabilized per-cell spectral solve. Exact H100 session `lera4b7j`
      initialized 18.13% of the full grid (the target footprint) with `1.12e-6`
      band reconstruction RMSE and restored 100% nonzero hard coverage. Initial
      loss improved from 0.9416 to 0.5090 and mean source-contrast slope from
      `4.68e-4` to 0.337, but mean support/source flux overshot to 2.273/2.527
      and mean support RMSE worsened to 0.662. The blank gate remains closed.
      Do not apply an empirical scalar or launch a grid/LR sweep. Next solve the
      fixed-cell appearance directly in pixel space with the existing exact
      `PowerPixelOperator.matvec/rmatvec`, then refine against observations.
      Artifacts are under `20260802-smacs-foam-coaddinit-g256-4829a0a`.
      Foam headless runs now also fail nonzero when a requested model fails or
      is missing.
    - [x] Replace centre-sampled Foam initialization with a weighted exact
      pixel-footprint inverse. Commit `d10627f` adds Jacobi-preconditioned CG
      over `PowerPixelOperator.matvec/rmatvec`, solves in pixel-integrated
      units, propagates exact valid-pixel cell coverage into the spectral
      solve, and makes both ridge terms scale invariant. The local gate passed
      1,739 tests with 214 skips and zero type errors.
    - [x] Close the matched exact-pixel blank before changing any tuning
      setting. H100 session `yxy529gn` completed full commit
      `d10627f7238c6f1381bc9e4be86a07aa322ebe3d` in 173.2 s at 937,856,512
      peak accelerator bytes. Exact initialization covered 18.63% of the full
      grid and kept 100% hard-render coverage. It materially corrected the
      previous 2.273/2.527 flux overshoot: mean support/source flux became
      0.872/0.889 and mean source-contrast slope remained 0.276. The blank gate
      still fails because mean support RMSE is 0.414 and flux remains 11--13%
      low; initialization RMSE in normalized pixel space is 2.047. Artifacts
      are under `20260802-smacs-foam-pixelinit-g256-d10627f`.
    - [x] Compose the known symmetric target PSF into the initializer's exact
      forward and adjoint as the only next scientific change. Commit `5c99a33`
      added the area-correct `D P D^-1` pixel forward and
      `D^-1 P^T D` adjoint and passed 1,739 tests. Matched H100 session
      `cub30m3e` completed in 175.4 s at the same 937,856,512-byte peak. The
      result is effectively unchanged: mean support/source flux is 0.875/0.893,
      contrast slope 0.278, support RMSE 0.414, and initialization RMSE 2.044.
      Target-PSF mismatch is therefore not the limiting error, and the blank
      gate remains closed. Artifacts are under
      `20260802-smacs-foam-psfinit-g256-5c99a33`.
    - [x] Test one explicit representation-resolution change, not a grid sweep.
      The first session `zoypfeli` stopped at the existing 10-million-pair
      allocation guard after measuring 16,423,554 raw candidates; this was a
      bounded resource guard, not a science result. Commit `e935868` exposed the
      existing ceiling through the benchmark and CANFAR runner. H100 session
      `e0do2myt` then completed the otherwise frozen grid-512 blank at exact
      commit `e9358682d2c86a06792acfbec94a04c7e6110e6e` in 255.1 s with
      3,147,516,928 peak accelerator bytes. Mean source-contrast slope improved
      from 0.278 to 0.515, initialization RMSE from 2.044 to 1.656, and mean
      support/source flux from 0.875/0.893 to 1.075/1.078. Mean support RMSE
      improved from 0.414 to 0.344. This confirms hard-cell resolution was a
      major limiter, but the 7--8% mean flux excess and RMSE still keep the
      fidelity gate closed. Artifacts are under
      `20260802-smacs-foam-psfinit-g512-e935868`.
    - [x] Score the frozen grid-512 model on all three existing injection
      batches before changing representation or tuning. Session `tsu81ekc`
      completed `validation-000` through `validation-002` at exact commit
      `1d055cfba3d5a89841f6cee61747cf079ec10a2b` in 24 minutes; artifacts are
      under `20260802-smacs-foam-g512-injections-1d055cf`. Their global metrics
      remain close to blank, as expected for sparse injections, so registered
      source recovery is the relevant statistic. Scorer `zuzdu3cp` failed before
      import because its fresh environment lacked the local build; replacement
      `quw7q4b3` reached the products but correctly rejected an ePSF polluted by
      faint non-positive responses. Commit `226b399` now uses the predeclared
      brightest AB-magnitude bin without response post-selection and passed
      1,714 tests, 246 skips, and zero type errors. After CANFAR recovery,
      session `qpqphhms` at `5847231` reached the frozen products and failed the
      predeclared scientific response gate: even the brightest injection bin
      has non-positive total injected-minus-blank response, so no effective PSF
      can be normalized without outcome-based source selection. The current
      piecewise-constant Foam is therefore not a depth candidate.
    - [x] Implement exact within-cell first moments and an opt-in affine spatial
      basis rather than scaling the regular grid again. Scalar and batched
      power-cell/pixel operators now persist analytic x/y moments, expose exact
      affine forward/adjoint maps, and hard-render wavelength-dependent local
      slopes. The constant-cell API remains the default. Analytic rectangle and
      half-disk integrals, scalar/batched parity, gradients, and SMACS CLI
      plumbing pass locally; exact-commit CUDA evidence remains the next gate.
    - [x] Warm-start the affine basis from coadds with the same exact forward
      model. A joint value-plus-gradient PCG solve uses area and x/y moment
      columns, optional symmetric PSF forward/adjoint composition, and then
      fits the shared spectral basis for both values and slopes. This avoids
      spending early optimizer steps discovering gradients from zero.
    - [x] Recover and score the pre-joint-initializer affine blank ablation.
      The durable exact-affine artifact completed all six bands at 512 square
      and 20 steps in 332.9 s. Its mean support/source-contrast RMSE
      (0.406581/1.247304) is numerically unchanged from the matched constant-cell
      control (0.406582/1.247307), while the maximum learned gradient
      coefficient remained only 0.0968. Affine capacity alone is therefore not
      activated by a value-only warm start plus 20 optimizer steps.
    - [ ] Run the affine-cell 512-square blank and one injection batch on the
      frozen validation split. Require positive bright-bin response, improved
      source slope, and acceptable flux/RMSE before expanding to all batches.
      Blank session `jin6xhv6` completed exact commit
      `0b73f1dd35afbbd46102208393c6ca8921a79943`, the joint value/slope coadd
      initializer, per-parameter relative ridge scaling, and artifact root
      `/arc/projects/ots/torchsky-runs/20260812-smacs-foam-affine-init-0b73f1d-c512-r1`.
      Relative to the value-only affine control, mean source-contrast slope
      increased from 0.0351 to 0.0737 and support/source flux ratios from
      0.804/0.608 to 0.916/0.825, while both RMSEs improved by 1.7% at matched
      20-step fit time (336 versus 333 s). The blank gate remains closed because
      source flux is still 17.5% low. Do not launch an injection yet. The next
      exact run must use cell-area-standardized slope parameters and record
      per-band within-cell variation plus spectral-basis conditioning.
    - [x] Run the standardized affine inverse at exact commit
      `5d5fd69a0a9c5f5d6afa412318101154fd356e3c`. Session `brxk9n79` completed
      the matched six-band blank under
      `/arc/projects/ots/torchsky-runs/20260812-smacs-foam-affine-standardized-5d5fd69-c512-r2`.
      Its mean support/source-contrast RMSE (0.399745/1.226135), source slope
      (0.07369), support/source flux (0.9162/0.8250), and 331 s fit time are
      effectively unchanged, so no injection was launched. Spectral-basis
      conditioning is benign at 3.86. The actionable failure is spatial:
      only 19.1% of the 1,024 fixed cells overlap the target, giving about 13
      cells across the scored crop and leaving extreme variation in boundary
      cells. Contract the deterministic grid from WCS geometry alone and rerun
      one blank before any capacity or optimizer sweep.
    - [x] Add the WCS-only target-footprint placement option. Seventeen samples
      per target-WCS edge define normalized tangent-plane bounds; the same
      rectangular 32x32 deterministic grid, cell-unit disk overlap, temperature,
      affine basis, and ridge are retained. This uses no image values, support
      masks, or source detections and exposes its choice in the JSON contract.
    - [ ] Run the matched contracted-grid blank and require near-unity initialized
      cell coverage plus materially improved source flux/slope before injection.
    - [x] Make the real-data independent/shared spectral control identifiable.
      Passband-only SVD whitening now orthonormalizes the integrated continuous
      basis, rejects numerical rank deficiency, and records raw/final condition,
      rank, and trainable capacity. For six bands, only a whitened six-mode run
      with recorded rank six may be interpreted as independent per-band
      appearance on shared geometry.
    - [ ] After the contracted-grid gate, run shared affine `32x32x4`, the
      within-1% trainable-capacity control `26x26x6` whitened, and the
      spatial-resolution control `32x32x6` whitened under identical data,
      optimizer, sampling, and seed contracts.
    - [x] Implement a joint exact detector-pixel inverse for fixed spectral
      affine Foam appearance. Matrix-free PCG composes each exposure's sparse
      WCS footprint operator, inverse variance, passband design, and symmetric
      PSF forward/adjoint without resampling the input or materializing a dense
      design matrix. The benchmark exposes it as an opt-in post-fit refinement
      and records its native-pixel residual contract.
    - [ ] Compare coadd initialization + smooth refinement against coadd
      initialization + exact detector-pixel refinement on the contracted blank.
      If the exact path wins fidelity and is computationally viable, make it
      the fixed-geometry default and reserve smooth rendering for adaptive-site
      geometry only.
    - [x] Bound exact-refinement accelerator memory by separating operator
      storage from solve placement and accumulating unchanged normal equations
      in observation batches. Operators expose exact tensor-storage bytes; PCG
      artifacts record total/peak-batch operator bytes, completed iterations,
      relative residual, and convergence. A coupled six-observation CPU check
      at 32 square reached identical `1.03e-7` residuals after ten iterations;
      resident/batch-one medians were 22.5/20.2 ms over five timed runs after
      warmup, so batching has negligible CPU overhead but no speed claim is
      made. CUDA transfer/memory evidence remains required.
    - [x] Add an opt-in per-cell block-Jacobi preconditioner for the exact
      spectral-affine solve. It captures within-cell spectral/value/slope
      coupling without changing the forward model or converged solution. On a
      coupled 32-square, eight-observation, four-mode CPU case it reduced PCG
      from 51 to 12 iterations and batch-one median time from 0.184 to 0.055 s
      (3.3x), while increasing preconditioner storage from 24,576 to 294,912
      bytes for an unwhitened polynomial response with condition 6.93. With the
      same response whitened to condition 1.00, diagonal/block took 14/12
      iterations but 20.5/24.2 ms: block setup outweighed the saved maps. This
      is therefore an ill-conditioning result, not a universal speed claim.
      Keep diagonal as the large-grid default until GPU CRF evidence justifies
      the O(cells * basis^2) block storage.
      Exact-refinement profiles now default to `auto`, which selects the block
      only when the response condition is at least 2 and its estimated
      construction peak fits a conservative device budget, recording the
      condition/threshold, selection, estimate, budget, and resident bytes;
      the reusable low-level solver itself retains its diagonal default. Any
      numerically non-positive local block falls back to its regularized local
      diagonal and is counted rather than aborting the entire field solve.
    - [ ] Validate host-batched exact refinement on aligned CRFs. Initial
      session `nuy2ew60` pulled and started the AstroAI image but emitted no
      runner log: CANFAR passed only the first token of the nested `bash -lc`
      payload. Direct-command smoke `snj8yvi9` and persistent-repository stage
      `rfsw1kyr` completed, proving the replacement launch contract. Session
      `eqql3aav` now runs exact commit
      `361e2b273bfdf1057667000eaa3d43b2f4f1924e` from that persistent script,
      with CPU operator storage, batch size one, auto preconditioning, grid 32,
      basis four, PATT 3/8, and the contracted blank. Artifacts target
      `/arc/projects/ots/torchsky-runs/20260812-foam-exact-auto-361e2b2/`.
      Baseline `eqql3aav` reached the exact operator but stopped at the declared
      allocation guard: 10,018,775 raw candidate pairs versus the 10,000,000
      ceiling. This is not a science result. First proximal submission
      `xj5945ab` used an incorrect manually expanded SHA and failed at checkout,
      also without science output. The actual fully verified proximal commit is
      `1c3328b506b2cce7fbb2a0f980a1192aec50ce89`; retry both configurations
      with the same explicit 12,000,000 candidate ceiling. Artifacts target
      `/arc/projects/ots/torchsky-runs/20260812-foam-exact-proximal-1c3328b/`.
      Compare the two as a single controlled objective test; do not tune from
      either blank result.
      Corrected submissions are baseline `aosl7dqt` at
      `/arc/projects/ots/torchsky-runs/20260812-foam-exact-base-r2/` and
      proximal `q8g1oyzx` at
      `/arc/projects/ots/torchsky-runs/20260812-foam-exact-prox-r2/`.
    - [x] Make exact-refinement scaling diagnosable before larger fields. The
      solver records preconditioner assembly, RHS assembly, initial residual,
      and PCG wall times; initial/final relative residuals; exact normal-map
      application count; and initial/final weighted native-pixel RMSE. CUDA
      timings synchronize only when diagnostics are requested. These fields
      distinguish geometry/setup, host transfer plus PSF work, and convergence
      costs without a profiler-specific artifact.
      All fields are propagated into per-band JSON rows and a deduplicated
      Markdown exact-refinement table, including initial/final RMSE, iterations,
      residual, phase times, peak operator batch MiB, and ridge center.
    - [x] Make post-fit regularization scientifically conservative. When an
      initial coadd/smooth model is supplied, exact refinement now minimizes a
      proximal ridge around that model rather than around zero; this prevents
      the refinement prior itself from shrinking recovered source flux. A
      cold solve still uses the original zero-centered ridge, and artifacts
      state which contract was used.
    - [x] Remove manual observation-batch sizing for new fields. Exact profiles
      default to `auto`, selecting the largest batch whose measured maximum
      operator+values+weights payload fits a conservative fraction of current
      free CUDA memory (capped at 2 GiB). Explicit positive sizes remain
      reproducible overrides, and artifacts record selection, budget, and
      maximum per-observation transfer bytes.
    - [x] Add the frozen Foam scoring path before the products finish.
      `benchmarks/score_smacs0723_foam.py` stages the exact blank/injection
      products, enforces common eroded support, measures each effective PSF from
      injected-minus-blank responses, and reuses the established matched-filter,
      blank-whitening, purity, and completeness-by-magnitude gates. The
      `foam-score` CANFAR mode runs it beside the products. Six focused scorer
      checks include a float32 FITS/ePSF self-check. The current ceiling is
      explicit: binary Foam support stands in for spatial inverse variance until
      the model exports propagated uncertainty.
    - [x] Add independent per-band and shared spectral Foam rows to the
      synthetic smoke tournament. Commit `da06bc3` reuses one exact footprint
      operator and applies a target-response gate so smoothing cannot masquerade
      as depth. It remains plumbing rather than evidence: the synthetic basis is
      ordered by band index and the automatic grid density is a bounded smoke
      heuristic, not the frozen SMACS configuration.
    - [x] Enforce the frozen exposure split in the Foam adapter. The previous
      joint benchmark discovered all 36/9 exposures per band, while classical
      validation uses only `PATT_NUM={3,8}` (8/2 exposures). `--patt-nums`
      now reuses the stacker filename filter, records the selection, and
      `foam-validation` defaults to `3,8`. Any exact-grid run without this
      filter remains an execution diagnostic, not comparator evidence.
    - [x] Wire the same frozen common-world detector injection into the joint
      field adapter without modifying archived inputs. `--injection-manifest`
      now feeds transient SCI overrides to both the differentiable observations
      and the coadd diagnostic; `--target-header-path` plus fixed crop
      coordinates pins every hard-rendered band to the exact classical WCS.
      Runner mode `foam-validation` persists blank and all three validation
      batches separately; `FOAM_VALIDATION_BATCHES` can select a comma-separated
      subset and rejects unknown batch names. Do not launch the four-fit matrix
      until the exact-grid blank passes.
- [ ] Execute Day 4: 20-batch CANFAR blind test, paired intervals, report, and
  figures.
- [ ] Submit only the strongest abstract claim whose pre-registered gate passes.

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
- Git pre-push hook is `pixi run preflight-push`. GitHub Comprehensive CI
  must be green before the release/infrastructure box is checked. `pixi run
  ci-local` is optional local parity with `build-test`.

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
  - [x] Implement the classical floor through production WCS, Torchfits, and
    coadd boundaries. `classical-smoke` now creates subpixel-dithered TAN
    detector images, runs weighted bilinear and exact square Drizzle separately,
    and adds independent-union and chi-square detection rows. The chi-square row
    explicitly accounts for reusing the bilinear product.
  - [x] Keep validation nulls and method-specific correlated-noise effects in
    the first floor. Every row freezes its threshold above a separate null
    realization and reports null detections, catalogue scores, forced
    photometry, held-out Gaussian NLL, RMSE, wall time, and CUDA allocation.
  - [x] Pass the exact 96-square/six-band/four-exposure classical floor on H100.
    Exact commit `addb925`, session `bcnedhqg`, completed 15 focused CUDA tests
    and the full benchmark in 3,702.98 s. Weighted bilinear took 0.281 s,
    chi-square detection including the reused bilinear product took 0.330 s,
    and exact square Drizzle took 3,691.03 s. Peak child RSS was 1,934,984 KiB;
    peak reported CUDA allocation was 47,494,144 bytes. Artifact:
    `/arc/projects/ots/torchsky-runs/20260727-multiband-classical-addb925/multiband_classical_smoke.json`.
    This one-seed floor is correctness/runtime evidence, not a depth claim.
  - [ ] Profile and compile/vectorize the exact square-Drizzle clipping path
    before expanding the benchmark. CANFAR telemetry showed about 12% of eight
    CPUs and 0% GPU utilization during the 3,691 s row: the CUDA tensors are
    device-correct, but polygon clipping is effectively serial/launch-bound.
    Preserve exact geometry and its separate validation null.
  - [ ] Add capacity- and compute-declared independent per-band Foam and shared
    spectral Foam rows, with hard-renderer inference after smooth fitting.
    The reusable `SpectralPowerFoamField` adapter is complete; the remaining
    work is deterministic grid initialization, real fitting, and matrix
    persistence rather than another observation pipeline.
- [ ] Gate zero-source false positives, noiseless bright-source flux/position,
  split disjointness, deterministic artifacts, shuffled-band loss of sharing
  gain, and CPU/CUDA parity.
  - [x] Implement the paired control boundary without accepting arbitrary
    negative flux: generated manifests label positive, blank, explicit
    sign-flipped, or shuffled-band mode; every band shuffle uses a distinct
    deterministic non-identity source permutation and records it. Both real
    classical and joint-field adapters use the same detector injection path.
    Focused conservation, determinism, marginal-preservation, CLI, and
    fail-closed sign tests pass. The parent scientific gate remains open until
    the frozen control batches run and demonstrate the expected null/loss-of-
    sharing behavior.
- [ ] Run three smoke scene seeds and two noise realizations. Do not check the
  parent depth-demonstration box or claim a magnitude gain from fewer scenes.
- [ ] Only after Gate 0 passes, add joint INR/hybrid rows, the 30-scene
  publication profile, bootstrap intervals, and detector-level SMACS injections.

Focused continuation:

```bash
pixi run -e test -- pytest tests/test_multiband_depth_benchmark.py -q
pixi run -e sphere-bench -- python benchmarks/bench_multiband_depth.py \
  --profile classical-smoke --device cpu --seed 101 --size 20 \
  --bands 3 --sources 3 --exposures 2 \
  --json-out benchmark_results/multiband_depth_classical_tiny.json
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
- [x] Record H100 session `xqrm5rlw`, exact commit `a482f9f`: the CUDA and SMACS
  boundary tests ran, but the deterministic-artifact gate exposed
  cross-platform LAPACK roundoff (`~1e-16`) in noiseless forced-photometry
  report values. The artifact now canonicalizes values below its declared
  `1e-12` reporting tolerance; scientific tensors retain their tight tests.
- [x] Rerun the focused H100 gate after the deterministic-report fix is pushed.
  Evidence: session `longvrs3`, exact commit `9b58106`, passed all 23 Gate 0
  and SMACS Torchfits-boundary tests in 44.75 s, then wrote the deterministic
  96-square/six-band artifact in 2.11 s. CUDA 12.9 was available and peak child
  RSS was 1,796,104 KiB. Artifacts:
  `/arc/projects/ots/torchsky-runs/20260727-multiband-gate0-v2-9b58106/`.

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
- [x] Decide from repeated evidence whether a caller-scoped byte-bounded MEF
  cache is justified. Do not implement one from assumption.
  Decision: do not add a second Torchsky cache. Session `lzp0324s` repeatedly
  read 2048-square `SCI`, `ERR`, and `DQ` planes through Torchfits with warm
  medians of 0.068/0.060/0.066 ms. NaN masks, finite values, headers, dtypes,
  and a 128-square CUDA round trip were exact. Retaining 405 full planes would
  instead create a large stale-data and residency liability; exposure-major
  callers already read each plane once.
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
Retain 512 as the operational default tile. The real repeated-MEF measurement
above rejects a second caller cache; revisit only if a later exposure-major
profile shows genuine repeated cold I/O.

### SMACS J0723 six-band readiness

- [x] Inventory the calibrated `SCI`, `ERR`, and `DQ` planes on CANFAR through
  Torchfits/Torchsky boundaries.
  Evidence: the data root is `/arc/projects/ots/jwst_smacs0723`; the 135
  aligned/tweakreg products actually used by the existing six-band work are
  under `results/wcs_aware_multiband_inr/official_no_psf_whitened_l2_batch100_400_tweakreg_rshift_8plus3_run2/alignment/tweakreg_rshift`.
  Session `g3s3wkdm`, exact commit `a482f9f`, found all 135 products and read
  every named plane through Torchfits. Session `g8ol0mcu` separately explained
  the recursive count of 162 as those 135 plus 27 duplicated root CRF products.
  Artifacts:
  `/arc/projects/ots/torchsky-runs/20260727-smacs-inventory-a482f9f/` and
  `/arc/projects/ots/torchsky-runs/20260727-smacs-aligned-inventory-a482f9f/`.
- [x] Verify common support for F090W, F150W, F200W, F277W, F356W, and F444W
  under the pre-registered split:
  `train={1,2,4,5,7}`, `validation={3,8}`, `test={6,9}`.
  Evidence: the aligned inventory has short-band counts 20/8/8 and long-band
  counts 5/2/2 for train/validation/test. Session `l3ux6x9i` then proved every
  one of the 18 band/split groups has non-zero coverage throughout the selected
  crop.
- [x] Record a header/coverage-only crop decision. Do not use science pixels to
  choose the crop and do not treat a deeper coadd as truth.
  Decision: use the 512-square target-grid crop `(x0=112, y0=0)` in the F277W
  pattern-1 exposure
  `108_jw02736001001_02101_00001_nrcblong_o001_crf_rshift_tweakreg.fits`.
  Session `l3ux6x9i`, exact commit `9b58106`, chose it from headers and WCS
  footprint coverage only at a declared 16-pixel coarse step. All 1,024 coarse
  crop samples have support in every band/split group; 98.17% of the complete
  target detector has the same common support. Artifact:
  `/arc/projects/ots/torchsky-runs/20260727-smacs-header-crop-9b58106/header_crop.json`.
- [ ] Run the existing six-band joint-fit smoke if the staged data and image
  provide the required environment.

### Torchfits consumer stress

- [x] Keep the Torchfits requirement open across releases while respecting its
  PyTorch ABI lane. Torchsky declares `torchfits>=1.0.0rc4` with no upper bound,
  but currently declares PyTorch 2.10.x. A locally built Torchfits 1.0.0rc5
  wheel declares `torch>=2.13,<2.14` and its runtime guard correctly rejects
  PyTorch 2.10, explaining why the Pixi solve retains rc4. Do not add a
  Torchfits upper bound; move Torchsky and its native-extension matrix to the
  PyTorch 2.13 lane as one explicit migration before expecting rc5/rc6 here.
- [x] Exercise repeated MEF `SCI`/`ERR`/`DQ` reads, overwrite freshness, header
  preservation, dtype/device transfer, and large-image memory through Torchsky.
  Evidence: the SMACS benchmark no longer imports Astropy FITS; filter
  discovery, `SCI`/`ERR`/`DQ` reads, and generated science/weight/mask products
  now use `torchsky.get_header`/`read`/`write`, hence Torchfits. A focused MEF
  fixture gates headers, float/int plane reads, invalid-ERR/DQ masking, weights,
  and round-trip auxiliary outputs. Session `lzp0324s`, exact commit `9b58106`,
  added real 2048-square SCI/ERR/DQ repetition: NaN masks and finite values were
  exact, float32/uint32 dtypes were preserved, headers repeated exactly, and the
  CUDA round trip was exact. Peak child RSS was 1,902,796 KiB. Existing
  `test_coadd_image_plane_does_not_reuse_overwritten_input` supplies the
  overwrite-freshness gate. Artifact:
  `/arc/projects/ots/torchsky-runs/20260727-smacs-repeat-read-9b58106/repeat_read.json`.
- [x] Record any Torchfits defects in the sibling repository; keep FITS/cache
  fixes out of Torchsky.
  Evidence: no Torchfits defect was exposed by the real-file stress, so no
  sibling issue or speculative cache patch was created. The earlier apparent
  repeat mismatch was solely `torch.equal` treating equal-position NaNs as
  unequal; the corrected NaN-aware test passes.

## Checkpoint 5 — Close And Hand Off

- [x] Update every checkbox above with commit IDs, artifact paths, and exact
  rerun commands.
- [x] Update [`next.md`](next.md) and [`status.md`](status.md) only for gates that
  actually passed.
- [x] Write the harness trajectory under `.cursor/harness/trajectories/`.
- [x] Run `pixi run ci-local`.
  Evidence: 1,669 passed, 215 skipped, 12 warnings in 545.03 s; Pyright reported
  zero errors and the existing dynamic-`__all__` warning.
- [x] Commit and push all completed work on `main`.
  Evidence: implementation/roadmap commit `c5e657d` is on `origin/main`.
- [ ] Confirm GitHub Comprehensive CI is green.
  Pending run: `30382664582` for `c5e657d`.
- [x] Track the user-authored FOSSIL sources as Markdown in
  [`docs/projects/`](../projects/fossil.md) and report:
  scientific/technical status, infrastructure status, next unchecked item, and
  explicit parallel work with repository, branch, data, and commands.

## Checkpoint 6 — Common Differentiable Spatial-Spectral Primitives

Source and scope: [`../projects/fossil.md`](../projects/fossil.md). These
primitives must serve CMB spectral-distortion/component separation and 21 cm
foreground inference first, while remaining useful for MOS/IFU sky/telluric
removal and static-sky versus moving-object separation. Classical and neural
separators share the same differentiable operator, split, injection, and open
residual contracts.

- [x] Read and distil the user-authored
  [`fossil_primitives_plan.md`](../projects/fossil_primitives_plan.md) and
  [`fossil_science.md`](../projects/fossil_science.md).
- [x] Harden the existing `SpectralAxis` instead of adding a parallel axis or
  units dependency: normalized physical quantity, strict ascending/descending
  and duplicate validation, regularity, explicit/derived edges, channel and
  quadrature weights, interval selection, device/dtype transfer, and
  differentiable wavelength/frequency/energy/wavenumber conversion.
  Evidence: 36 focused spectral/FITS-WCS tests pass, including irregular
  descending channels, explicit edges, energy/wavenumber round trips, and
  autograd.
- [ ] Add the minimum immutable `TensorSpec` needed to distinguish spectral,
  spatial, detector, and time axes and reject physically incompatible operator
  composition. Reuse existing metadata; do not build a units framework.
- [ ] Promote current spectral transmission code into a differentiable
  `SpectralResponse`/mixing operator with tabulated curves, channel mixing,
  detector batches, photon/energy conventions, forward/adjoint, and dense
  analytic oracles.
- [ ] Add a weighted `ProtectedSubspace` primitive shared by parametric,
  PCA/NMF, neural, and hybrid component separators. Report target attenuation,
  nuisance overlap, retained target norm, rank, and conditioning.
- [ ] Add deterministic cross-fit partitions and paired positive/negative
  injections across sky, detector, time, scan, and frequency. Leakage-prone
  random sample splits are not acceptable.
- [ ] Add full-sky spectral products with explicit monopole/dipole/anisotropy,
  masks/weights/coverage, RING/NEST, beams, splits/coadds, and open residuals.
- [ ] Gate the common spine end to end:
  `axis/response -> differentiable mixing -> protected target+nuisance ->
  held-out cross-fit -> paired injection -> open residual`, including exact
  adjoints, gradcheck, eager/chunk parity, CPU/CUDA parity, and bounded memory.
- [ ] Only then add cosmology profiles: `mu ~ 2e-8` preservation, 21 cm/LIM
  smooth-foreground leakage and wedge-aware validation, plus smaller MOS and
  moving-object transfer demonstrations.

## CANFAR Run Template

Do not reuse or delete unrelated sessions. `canfar image ls` does not currently
report every pure headless image, so it is not an authoritative allowlist.
Confirm the exact resolved image through a successful disposable session pull
and the session metadata returned by `canfar ps --all --json`. Every job writes
its artifact to persistent `/arc` storage and exits.

CANFAR expands `$name` tokens in submitted command arguments as platform group
references before the container starts. Do not put shell-variable references in
the `canfar create ... -- CMD ARGS` payload. Use a script already on `/arc`, or
encode the complete bootstrap and decode it inside the container with a
single-quoted, YAML-safe standard-library command.

```bash
canfar server use canfar
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

Use `astroai/improc:latest` for the installed comparison tools. Headless-only
images may be absent from image listings. A separately recorded working image
override is acceptable for Torchsky-only jobs; it does not change the
comparison-image default or establish that the default name is incorrect.

Use `map-compat` for classical comparator jobs and `smacs-cuda` for real-data
Foam jobs. Both run through `scripts/run_local_sphere_bench.py`, which installs
the local project editable without build isolation and fails unless the native
`torchsky.cpp` extension imports successfully.
