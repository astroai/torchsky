# WCS-Aware SMACS0723 Pipeline

JWST/NIRCam deep-field reconstruction over SMACS J0723 that trains continuous
spatial-spectral fields directly on NATIVE detector pixels, benchmarked
against published UPdec-Webb products. Describes the student-fork pipeline
and its torchsky-native equivalents.

## Glossary
- CRF/CAL: JWST corrected-rate / calibrated NATIVE exposures (no resampling).
- i2d: resampled Drizzle product (already-projected mosaic plane).
- UPDC (Wang et al. 2024, arXiv:2411.13891): Upsampling-and-PSF-Deconvolution
  Coaddition. iter0 = Drizzle baseline; iter10 = deconvolved reference.
- HybPSF: WebbPSF (WSS OPD by date) + PCA residual fitted on science-frame stars.
- ERR/DQ: pipeline per-pixel error and data-quality planes.

## Original fork chains
### A — science faint-source (native pixels end-to-end)
CRFs -> [TweakReg frozen alignment; anchor gauge] -> native SCI/ERR/DQ pools ->
whitened-L2 INR pretrain -> frozen-PSF finetune -> render to target grid
(only resampling, post-fit) -> detection -> QA vs UPDC iter10.
Variants: dense native residuals; experimental learnable PSF; multi-GPU DDP.
### B — tutorial (pre-drizzled inputs)
i2d crops -> INR on ALREADY-DRAZZLED mosaics -> drizzle baselines -> tile QA.
Tutorial only: unsuitable for depth claims.
### C — UPDC reproduction prep
survey/JWST 1/f correction -> calibrated CAL inputs -> two-pass
leave-one-exposure-out 5-sigma outliers (half-scale sky grid, projected back)
-> Gaussian completion + `UPDCREJ` (`1=replaced`) -> feeds A and PSF stars.

## Torchsky-native equivalents
| Stage | Torchsky |
|---|---|
| Native-pixel training | `map/field` `fit_field_model`; joint multiband `bench_map_smacs0723_field_fit_joint.py`; epoch sampling via `observation_sampling="epoch"` |
| Alignment | native `map/exposure_pose.fit_pose_correction`; official JWST step: `scripts/tweakreg_align_exposures.py` (`tweakreg` extra) |
| UPDC prep | `benchmarks/updc/prepare_inputs.py` |
| QA vs iter10 | `benchmarks/updc/compare_catalogs_vs_reference.py` (`catalogs.crossmatch_sky`) |
| Nominal filter/detector PSFs | `scripts/export_stpsf_kernels.py` -> `kernel2d`; not exposure/time/OPD matched |
| Mosaic render | mosaic mode in `run_canfar_adass_depth.sh` (full-field coadds on UPDC grid) |
| Gates | `bench_multiband_depth.py` dither splits/injections/blank batches; `score_field_models.py` m50 bootstrap CIs |

Runbook: `benchmarks/updc/README.md`. Background: `hyperspectral-fusion.md`,
run log: `adass_updec_depth_week.md`.

## Limitations / deferred
- DDP launchers and learnable-PSF arm not ported; prefer `learned_lowrank` +
  `PSFCalibrationTerm` (star-anchored, held-out gated).
- `poisson_gaussian_nll` / `StructuredDetectorNoise` shipped but unwired in
  reconstruction defaults.
- Native-pixel GLRT baseline unimplemented (formulation gate row); closest:
  whitened matched filter on classical stacks.
- Chain B trains on pre-resampled mosaics: tutorial only.
