# UPDC-vs-Drizzle SMACS0723 Benchmark

Consolidated runbook for the specialized SMACS J0723 benchmark: reproduce the
UPDC (Wang et al. 2024, arXiv:2411.13891) comparison chain end-to-end with
torchsky-native primitives, and score native-pixel field reconstruction
against classical Drizzle products.

Pipeline overview and glossary:
`docs/projects/wcs-aware-smacs0723-pipeline.md`.
Background: `docs/projects/hyperspectral-fusion.md` (UPDC context).
Run log: `docs/projects/adass_updec_depth_week.md`.

## Run order

1. **External calibration, then Chain C preparation** (per band):
   Run the survey/JWST 1/f correction upstream and place the resulting CAL
   products in `<cal_dir>`. Torchsky does not own that survey-specific step.
   ```bash
   pixi run python benchmarks/updc/prepare_inputs.py \
       --input-dir <cal_dir> --output-dir <prep_dir> \
       --target-reference <updc_grid.fits> \
       --bands F277W,F356W,F444W [--dq-fix]
   ```
   Fits two-pass, leave-one-exposure-out 5-sigma statistics on a factor-2
   downsampled sky grid projected back per detector through
   `bind_image_plane_target_operator`, completes bad pixels with seeded
   Gaussian draws (`--seed`), and writes `*_cal_completed*.fits` plus a
   `UPDCREJ` rejection extension (`1=replaced`; never pass it directly as a
   valid-support mask). `--dq-fix` uses the conservative policy
   (science aperture & finite & DO_NOT_USE bit only).

2. **Alignment** — either the official JWST step or the native primitive:
   ```bash
   pixi run -e smacs python scripts/tweakreg_align_exposures.py \
       --inputs <exposures...> --output-dir <aligned_dir> --fitgeometry rshift
   ```
   Produces `{index:03d}_{stem}_rshift_tweakreg.fits` — exactly the glob
   consumed by `benchmarks/bench_map_smacs0723_real.py`. For a pure-torch
   correction of matched catalogs, fit
   `torchsky.map.exposure_pose.fit_pose_correction` in the exposure's
   CRVAL-centered tangent frame; pair sources with
   `torchsky.catalogs.crossmatch_sky`.

3. **Native-pixel training** — `map/field` (`fit_field_model`) trains
   continuous fields directly on native SCI/ERR/DQ pools; joint multiband via
   `benchmarks/bench_map_smacs0723_field_fit_joint.py`; exposure sampling
   across epochs via `observation_sampling="epoch"` (stateless coprime-stride
   cycles, every exposure visited once per epoch).

4. **Mosaic render** — full-field coadds on the UPDC grid:
   `bash scripts/run_canfar_adass_depth.sh mosaic` (see that script's env
   knobs for bands/methods).

5. **QA vs UPDC iter10**:
   ```bash
   pixi run python benchmarks/updc/compare_catalogs_vs_reference.py \
       --candidate <detected.fits> --reference <updc_iter10.fits> \
       --radius-arcsec 0.25 --out qa_report.json
   ```
   Radius candidates come from `crossmatch_sky(find="all")`, then a
   deterministic maximum-cardinality one-to-one association reports
   matched/candidate-only/reference-only counts, separation median+p95, and
   flux-ratio median/q16/q84 overall and per SNR bin.
   Optional photutils detection from rendered SCI images:
   `--detect-images <dir>` (run in the `smacs` Pixi environment).

## Role map of existing assets (unchanged locations)

| Asset | Role |
|---|---|
| `benchmarks/bench_multiband_depth.py` | Injections, dither splits (train/validation/test), completeness & false-density metrics |
| `benchmarks/score_smacs0723_native_classical.py` | Drizzle pf0.8/pf1.0 selector over blank/validation batches |
| `benchmarks/stage_smacs0723_canfar.py` | CANFAR staging |
| `scripts/run_canfar_adass_depth.sh` | Orchestration incl. new `mosaic` mode |
| `docs/projects/hyperspectral-fusion.md` | UPDC background |
| `docs/projects/adass_updec_depth_week.md` | Run log |

## Gate checklist (docs/formulation_map.tex §gates)

- [ ] **Gate 1 — Prevalence-independent completeness depth**: Δm50 > 0 with
      cluster-bootstrap 95% CI excluding zero at fixed false-positive surface
      density (`bench_multiband_depth.py` injections + blank batches).
- [ ] **Gate 2 — Resolution-matched sensitivity**: injection transfer function
      H_eff(x) and effective PSF matched before depth claims.
- [ ] **Gate 3 — Photometric & color invariants**: bright point-source median
      absolute flux bias ≤ 2% per band; adjacent-band color bias ≤ 0.02 mag.
- [ ] **Gate 4 — Population-specific headroom**: m50 reported per class
      (point / extended / blended / dropouts).
- [ ] **Gate 5 — Two-stage out-of-sample falsification**: discoveries from
      training exposures must predict photons in untouched test exposures
      (held-out likelihood-ratio statistic).
