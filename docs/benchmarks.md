# torchsky Benchmarks

This page documents the benchmark methodology. The snapshot tables below reflect the **0.3.1** release; update with each release run.

## Main Scripts

| Script | Description |
|--------|-------------|
| `bench_all.py` | 4-domain orchestrator: FITS I/O, FITS Table I/O, WCS, Sphere. |
| `bench_fits_io.py` | Authoritative FITS image-domain runner (1D/2D/3D, dtype/size/compression/scaled, MEFs, cutouts, headers). |
| `bench_fitstable_io.py` | Authoritative FITS table-domain runner (row scales, schema mixes, varlen, projection/slice/filter/scan). |
| `bench_wcs_suite.py` | Authoritative WCS projection sweep runner with forward+inverse and `N={1k..10M}` tiers. |
| `bench_sphere_suite.py` | Authoritative sphere-domain runner aggregating geometry/advanced/sparse/spectral/polygon/core benches. |
| `bench_legacy_all.py` | Legacy monolithic exhaustive harness (opt-in only, no longer default contract). |
| `bench_sentinel.py` | Fast 10-case gate with early-stop significance checks for optimization loops. |
| `bench_ml_loader.py` | End-to-end DataLoader throughput benchmark (uncompressed + compressed). |
| `bench_fast.py` | Fast regression loop for image-path changes. |
| `bench_compressed_stability.py` | Randomized-order compressed benchmark for RICE/HCOMPRESS tuning with lower order/cache bias (`library_default` vs forced config modes). |
| `bench_table.py` | FITS table-path read/scan/column access checks. |
| `bench_arrow_tables.py` | Arrow conversion benchmark for table workloads. |
| `bench_map_image_plane.py` | Image-plane mapmaking benchmark covering synthetic identical/shifted WCS coadds, a synthetic low-overlap TPV MEF case, a grouped synthetic multi-band compatibility cube, a header-derived planned spectral-cube row on a shared target grid, plus both single-HDU and generated MEF real Astropy SIP FITS cases in identical-WCS and shifted-WCS variants, including overlap and footprint diagnostics. |
| `bench_updec_webb_real.py` | Real-data smoke benchmark on public UPdec-Webb JWST SMACS J0723 products. It compares the published iteration-0 Drizzle product against the iteration-10 UPDC product on the same WCS grid, reports product-format details plus same-grid delta and sharpness ratios, and can auto-download selected public bands from NADC. |
| `bench_map_smacs0723_real.py` | Real-exposure JWST SMACS J0723 smoke benchmark. It coadds local public NIRCam exposures onto a cropped target grid derived from the published F277W Drizzle product, preferring local `*_crf.fits` inputs when present and falling back to `*_cal.fits`, then reports support-domain agreement against that real reference patch. The current real JWST path also accepts `drizzle_point` / `drizzle_square` plus explicit `--drizzle-pixfrac` so the in-tree map code can be compared against the published drizzle contract directly. |
| `bench_map_smacs0723_reproduce.py` | Real-data reproduction sweep against the published SMACS J0723 F277W Drizzle patch. It runs several `torchsky.map` resampling/value-semantics variants on the same local calibrated exposures, prefers local `*_crf.fits` inputs when present, and can optionally add true external `drizzle` rows on the same crop to separate accumulation-contract bugs from benchmark mismatches. For the public JWST CRF inputs, the drizzle rows are treated as `surface_brightness` products rather than flux-per-pixel products. |
| `bench_map_smacs0723_ladder.py` | Real-data exposure-ladder benchmark against the published SMACS J0723 F277W Drizzle patch. It runs one chosen `torchsky.map` configuration on prefixes of a matched real-exposure set, preferring local `*_crf.fits` when present, so overlap, coverage, and support error can be tracked as more real visits are added. |
| `bench_map_smacs0723_field_fit.py` | Real-data single-band field-fit smoke on JWST SMACS J0723 exposures. It compares `coadd_bilinear` against selected INR field models on a crop of the public drizzle target, now using band-aware field observations with finite bandpass bounds plus weighted-RMS normalization instead of a delta-wavelength proxy, and it also supports an `exact_wavelength` observation mode as a 2D no-integration control. |
| `bench_map_smacs0723_field_fit_joint.py` | Real-data joint multiband field-fit benchmark on JWST SMACS J0723 products. It fits one latent field across several public bands simultaneously, then renders each band back onto its own published reference crop so per-band coadds can be compared against a shared multiband INR. |
| `bench_map_smacs0723_field_fit_band_sweep.py` | Real-data per-band field-fit sweep on JWST SMACS J0723 module-B products. It iterates across multiple public UPdec-Webb reference bands, auto-discovers matching local public exposures by filter and module, and reports per-band `coadd_bilinear` vs INR field-fit results on a common crop. It supports both finite-bandpass fitting and an `exact_wavelength` 2D control mode. |
| `bench_map_smacs0723_stackers.py` | Real-data stacker comparison benchmark against the published SMACS J0723 Drizzle patch. It runs legacy stackers (`drizzle`, `reproject`, `dfreproject`, `Montage`, `SWarp`) plus matching `torchsky` kernels and selected INR rows on the same paper-faithful crop, with backend-appropriate target projections and band-aware INR rendering. |
| `bench_map_smacs0723_stackers_band_sweep.py` | Real-data multiband non-INR stacker sweep on public SMACS J0723 products. It iterates over multiple public UPdec-Webb reference bands, auto-discovers matching local public JWST exposures by filter and module, then runs the full legacy/`torchsky` stacker matrix sequentially per band without INR rows so the baseline stack pipeline can be validated before multiband field fitting. |
| `bench_map_compat.py` | Mapmaking compatibility benchmark comparing torchsky image-plane coadds against an explicit external-tool matrix. `reproject_interp`, `dfreproject`, low-level `drizzle`, source-built `Montage`, source-built `SWarp`, and local-source `casutools imstack` are wired where their target/grid semantics are comparable; `drizzlepac` and unsupported or backend-failing target/grid combinations appear as explicit `SKIPPED` rows with concrete reasons. |
| `bench_map_ground_truth.py` | Synthetic mapmaking correctness benchmark against known sky truth. It evaluates final requested map products on a shared `MapProductSpec`, emits machine-readable JSON plus a concise Markdown summary table, and can optionally enforce representative baseline JSON gates. |
| `diagnose_map_large_union_reprojection.py` | Focused diagnostic runner for the large-union truth cases. It emits per-observation reprojection rows for `torchsky` and `dfreproject`, pairwise diffs before stacking, and final stack rows on the same target grid so TPV or large-rectangle gaps can be localized to specific inputs instead of inferred from the final coadd alone. |
| `diagnose_map_large_union_accumulation.py` | Follow-up diagnostic runner for the large-union TAN weighted cases. It compares grouped vs per-observation accumulation for `torchsky` and `SWarp`, bins residuals by overlap count, and reports numerator/denominator hotspots so overlap-region edge-support mismatches can be localized to specific pixels and observations. |
| `diagnose_map_projection_tail.py` | Focused wide-projection stack diagnostic. It compares stack methods on one projection case, bins residuals by overlap count, and reports the worst improved pixel with contributing single-observation values so small tail differences can be separated from broad reprojection failures. |
| `bench_map_boundary_support.py` | Focused edge-support sweep for bilinear TAN weighted coadds on the large-union truth cases. It evaluates explicit source-boundary margins against ground-truth reconstruction metrics and `SWarp` agreement so edge-policy ideas can be benchmarked before touching the production resampler. |
| `bench_map_multifield_truth.py` | Synthetic multiwavelength latent-field cube benchmark. It simulates wavelength-dependent continuous sky fields with mixed bandpasses, PSFs, WCS distortions, throughput, and overlap, using both analytic fields and an initial frozen implicit-neural-field variant, then scores `coadd_image_plane_cube(...)` against latent cube truth and emits per-case torchsky kernel recommendations. |
| `bench_map_field_fit.py` | Experimental continuous-field fitting benchmark. It renders fitted field models and direct stack baselines onto the same requested final product, emits JSON plus Markdown summaries, and can optionally enforce representative baseline JSON gates. |
| `bench_wcs.py` | WCS transform throughput checks, including small-batch center-projection `default vs python-only` fields on CPU. |
| `bench_wcs_tpv_inverse.py` | Dedicated large-`N` TPV inverse benchmark comparing torchsky fused TPV `world_to_pixel`, torchsky tensor-core TPV inverse, and Astropy on affine-only and non-linear TPV headers. |
| `bench_healpix.py` | HEALPix CPU/CUDA parity + throughput + quality gates (`ang2pix`, `pix2ang`, `ring/nest`). |
| `bench_healpix_advanced.py` | Advanced HEALPix parity + throughput checks (`get_all_neighbours`, `get_interp_weights`, `get_interp_val`). |
| `bench_sphere_geometry.py` | Cross-library spherical geometry benchmark (torchsky vs HEALPix ecosystem releases). |
| `bench_sphere_core.py` | Sphere-core primitives benchmark (multi-band sampling, pairwise distances, ellipse query). |
| `bench_sphere_polygons.py` | Non-convex spherical polygon benchmark (contains/intersects/query/area with optional spherical-geometry parity). |
| `bench_catalog_crossmatch.py` | Catalog crossmatch refinement benchmark comparing declination pruning against the internal coarse HEALPix candidate index. |
| `bench_catalog_compat.py` | Catalog crossmatch compatibility benchmark against `astropy.coordinates.SkyCoord`, covering both synthetic clustered catalogs and vendored real Astropy VOTable fixtures, plus a real-data FITS roundtrip when native table I/O is available. |
| `bench_sphere_spectral.py` | CPU-first spherical harmonic primitive benchmark (`map2alm`, `map2alm_lsq`, `alm2map`, `almxfl`, `alm2cl`, `anafast`, `map2alm_spin`, `alm2map_spin`) plus compatibility-generation paths (`synalm`, `synfast`, `bl2beam`, `beam2bl`) as needed. |
| `bench_pipeline_table_sphere.py` | End-to-end benchmark: table predicate pushdown (`where`) + HEALPix spherical reduction pipeline. |
| `replay_upstream_healpy_interp_edges.py` | Replays interpolation edge cases (lon wrap, poles, pixel-center inputs) against official healpy with parity/perf gates. |
| `replay_upstream_healpy_spin.py` | Replays spin transform parity/throughput against official healpy releases (`map2alm_spin`, `alm2map_spin`). |
| `replay_upstream_spherical_geometry_polygons.py` | Replays spherical-geometry upstream polygon fixtures/data (`test_intersects_*`, `difficult_intersections.txt`) against torchsky with multi-NSIDE difficult-overlap area gates. |

## Standard Commands

```bash
# Contract run (all four domains)
pixi run bench-all

# Exact scope aliases
pixi run bench-fits       # == pixi run bench-all -- --fits-only
pixi run bench-fitstable  # == pixi run bench-all -- --fitstable-only
pixi run bench-wcs        # == pixi run bench-all -- --wcs-only
pixi run bench-sphere     # == pixi run bench-all -- --sphere-only

# Direct CLI (equivalent)
pixi run python benchmarks/bench_all.py --scope all
pixi run python benchmarks/bench_all.py --fits-only
pixi run python benchmarks/bench_all.py --fitstable-only
pixi run python benchmarks/bench_all.py --wcs-only
pixi run python benchmarks/bench_all.py --sphere-only

# Output contract
# benchmarks_results/<run_id>/
#   - results.csv
#   - torchsky_deficits.csv
#   - summary.md

# Fast optimization gate (10 diverse cases, early-stop enabled)
pixi run python benchmarks/bench_sentinel.py --initial-repeats 3 --full-repeats 9 --seed 123

# Compressed-path stability run (randomized config order)
pixi run python benchmarks/bench_compressed_stability.py --rounds 4 --repeats 12 --warmup 3

# ML loader throughput run
pixi run python benchmarks/bench_ml_loader.py --device cpu --shape 2048,2048 --n-files 50 --batch-size 4 --num-workers 4 --epochs 3 --repeats 5 --warm-cache

# Image-plane mapmaking run (synthetic + real FITS fixture; native extension required)
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_image_plane.py --size 512 --n-obs 4 --runs 5 --footprint-order 8

# Image-plane mapmaking correctness run against synthetic truth (native extension required)
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_ground_truth.py --size 128 --runs 3

# Same run, but also emit the concise Markdown summary next to the JSON artifact
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_ground_truth.py --size 128 --runs 3 --json-out /tmp/map_truth.json

# Image-plane method matrix against synthetic truth, including optional external backends
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_ground_truth.py --size 128 --runs 3 --include-external

# Diagnose per-observation reprojection differences on the rectangular large-union truth case
pixi run -e map-compat python benchmarks/diagnose_map_large_union_reprojection.py --size 512 --width 512 --height 384

# Diagnose grouped-vs-per-observation accumulation differences on the same large-union TAN weighted case
pixi run -e map-compat python benchmarks/diagnose_map_large_union_accumulation.py --size 2048 --width 2048 --height 1536 --scenario no_mask

# Diagnose residual tail differences between stack methods on a wide-projection row
pixi run -e map-compat python benchmarks/diagnose_map_projection_tail.py --case ARC:tan_multires_noise@768 --left-method mean --right-method median

# Sweep bilinear source-boundary support margins against truth and SWarp on the large-union TAN weighted case
pixi run -e map-compat python benchmarks/bench_map_boundary_support.py --size 2048 --width 2048 --height 1536 --scenario no_mask

# Multiwavelength latent-field cube truth run
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_multifield_truth.py --size 96 --runs 3

# Rectangular large-scale multifield run with mixed input image sizes
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_multifield_truth.py --cases tan_multifield_rectangular_scale,tpv_multifield_rectangular_scale --width 2048 --height 1536 --runs 1 --torchsky-kernels bilinear

# Ten-image large-scale random multifield stress run with mixed resolutions and some non-overlap
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_multifield_truth.py --cases mixed_multifield_random10 --width 2048 --height 1536 --runs 1 --torchsky-kernels bilinear

# Multiwavelength latent-field cube smoke baseline gate
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_multifield_truth.py --size 48 --runs 2 --cases tan_multifield_cube --torchsky-kernels bilinear --baseline-json benchmarks/baselines/map_multifield_truth_smoke.json

# Larger representative multifield gate with timing checks
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_multifield_truth.py --size 96 --runs 3 --cases tan_multifield_cube,tpv_multifield_cube --torchsky-kernels bilinear --baseline-json benchmarks/baselines/map_multifield_truth_rep.json

# Experimental latent-field fitting run
# Includes exact-transmission spectral rows such as `tan_field_fit_exact_transmission`
# and `tpv_field_fit_deep_multiband_exact`, plus harder same-field broad-band
# deep-field rows such as `tan_sip_field_fit_deep_multiband_broad` and
# `tpv_field_fit_deep_multiband_broad`, for direct projected-field vs per-band
# stack comparisons on the same product grid. New sparse larger-mosaic rows such as
# `tan_sip_field_fit_sparse_large_mosaic_broad` and
# `tpv_field_fit_sparse_large_mosaic_exact` keep the visits small while scaling
# the requested product to a larger field. Experimental pure-INR families
# include fixed and learned spectral bases, factorized sky-lambda planes,
# multiscale sky-lambda K-planes with refinement, transmission-conditioned latent
# fields, sky-encoder bandpass aggregators, conditioned spectral decoders, and
# hybrid hash-grid spatial encoders with explicit wavelength decoders.
# The broad-band deep-field rows currently use training-time shared mixture
# Monte Carlo supervision plus a light deterministic full-band anchor on one
# observation per step, while keeping deterministic quadrature for evaluation.
# The tuned `sky_encoder_resmlp` branch is now case-split too: the TAN-SIP broad
# row keeps the stronger long-budget non-multiscale setting, while the TPV broad
# row keeps the multiscale readout variant that currently wins its support-aware
# benchmark.
# The sparse broad-band mosaic row remains on deterministic quadrature because
# both sampled training paths regressed that larger regime in current benchmarks.
# The structured sparse TPV exact `sky_lambda_feature_field` row now uses the
# mixed `band_global_mix` reduction, which slightly improves both support-aware
# and full-grid RMSE over the earlier pure `per_band` default on the current
# `128x128` frontier probe.
# The sparse TPV exact benchmark family now also includes a limiting-case
# `tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit` row with only
# three weakly coupled exact bands but fifteen visits per band, so we can test
# whether the latent field can beat three separate coadds by pooling signal
# across forty-five total exposures without relying on dense spectral sampling.
# Experimental coherent shared-sky patch sampling is available in the fitter,
# but current large sparse TPV probes keep the older independent patch
# minibatches because the shared-sky variant improved support-region errors in
# some longer runs while clearly regressing the full-grid frontier metrics.
# That 3-band sparse TPV frontier now uses the new chunked sparse-training
# contract for `sky_lambda_feature_field`: band-balanced observation minibatches
# plus local training patches. On the hard `128x128` row this improves the field
# over the earlier full-frame fit, even though the coadd still remains better.
# The same chunked contract is now also used on the sparse TPV broad-overlap
# frontier. On the current `32x32` smoke it improves support-aware field
# accuracy relative to the old full-frame feature-field path, and at a larger
# `16`-step budget on that same smoke row it does beat the direct coadd.
# Field-fit rows now report both full-grid product metrics and support-aware
# metrics using the `coadd_bilinear` support mask as the comparison domain.
# Recommendation rows prefer `support_reconstruction_rmse` when it is available.
# Sparse frontier rows now also emit cheap region-split diagnostics:
# - supported single-overlap vs multi-overlap RMSE
# - fringe-extrapolated vs far-extrapolated RMSE
# so large sparse-mosaic failures can be localized without a separate notebook.
# They also emit a true uniform-grid metric (`uniform_reconstruction_rmse`) that
# ignores each method's native support mask, so extrapolating fields and masked
# coadds can be compared on the same target grid without ambiguity.
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 48 --steps 120 --runs 1

# Focused sparse-mosaic frontier profile with case-specific step budgets
# - `tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit` runs at 32 steps
# - `tpv_field_fit_sparse_large_mosaic_broad_denseoverlap` runs at 16 steps
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 32 --steps 8 --step-profile sparse_mosaic_frontier --runs 1 --cases tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit,tpv_field_fit_sparse_large_mosaic_broad_denseoverlap --models coadd_bilinear,sky_lambda_feature_field,sky_encoder_resmlp

# Public UPdec-Webb real-data smoke on one or more bands. This currently
# benchmarks the published product pairs directly (Drizzle iter0 vs UPDC iter10)
# rather than the raw JWST exposure-level forward model.
pixi run -e test python scripts/download_updec_webb.py --bands F277W --iterations 0,10
pixi run -e test python benchmarks/bench_updec_webb_real.py --bands F277W

# Sequentially download the public multiband references plus a manageable
# module-B public JWST exposure subset for each band into data/.
pixi run -e test python scripts/download_smacs0723_multiband_public.py --bands F090W,F150W,F200W,F277W,F356W,F444W --module b --updec-iterations 0

# First real exposure-level JWST SMACS J0723 smoke, using local public F277W
# calibrated exposures plus the published F277W iter0 drizzle product as the
# target/reference grid. Native SIP headers now run by default; `--linearize-sip`
# remains available as a fallback/debug option.
pixi run -e test python scripts/download_jwst_smacs0723_mast.py --band F277W --max-exposures 6
pixi run -e test python benchmarks/bench_map_smacs0723_real.py --crop-size 768

# Reproduction sweep against the published F277W drizzle patch with several
# torchsky map variants on the same real exposures. Native SIP now runs by
# default; `--linearize-sip` remains available as a fallback/debug option.
# If local `*_crf.fits` files are present they are preferred over `*_cal.fits`.
pixi run -e test python benchmarks/bench_map_smacs0723_reproduce.py --crop-size 512

# Same sweep, but also add true external drizzle rows on the matched CRF set.
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_reproduce.py --crop-size 512 --include-drizzle --drizzle-kernels point,square

# Direct in-tree drizzle-style reproduction on the same matched CRF set.
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_real.py --crop-size 512 --resampling-kernel drizzle_square --value-semantics surface_brightness --stacking-method weighted_mean --drizzle-pixfrac 1.0

# Real-data stacker comparison on the paper-faithful 9-visit F277Wb crop.
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_stackers.py --band F277W --module b --input-product cal --filename-contains nrcblong --crop-size 512 --crop-mode center

# Real-data multiband non-INR stacker sweep across all public UPdec-Webb bands,
# reusing the same legacy and torchsky stackers sequentially per band.
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_stackers_band_sweep.py --bands F090W,F150W,F200W,F277W,F356W,F444W --module b --input-product cal --crop-size 512 --crop-mode best_overlap

# Exposure ladder on the same real SMACS J0723 F277W target, using the current
# best quick reproduction setting (`bilinear + flux + weighted_mean`) and an
# overlap-aware crop so we can see how agreement changes as more real visits
# are added.
pixi run -e test python benchmarks/bench_map_smacs0723_ladder.py --tiers 2,4,6 --crop-size 512 --crop-mode best_overlap --resampling-kernel bilinear --value-semantics flux --stacking-method weighted_mean

# Real-data INR smoke on the same F277W target. This currently treats the
# public F277W exposures as an exact single-band field-fit problem and compares
# them against the matching coadd support region on the published drizzle grid.
pixi run -e test python benchmarks/bench_map_smacs0723_field_fit.py --crop-size 128 --crop-mode best_overlap --steps 4 --models coadd_bilinear,spectral_basis_field,sky_encoder_resmlp,sky_lambda_feature_field,operator_constrained_feature_field

# Real-data per-band INR sweep across multiple published UPdec-Webb bands,
# using local module-B public JWST exposures when present.
pixi run -e test python benchmarks/bench_map_smacs0723_field_fit_band_sweep.py --bands F090W,F150W,F200W,F277W,F356W,F444W --module b --crop-size 32 --steps 2

# Real-data joint multiband INR smoke on the currently complete public bands.
# This fits one field across several filters simultaneously, then scores each
# rendered band against its own published iter0 reference crop.
PYTHONPATH=src:benchmarks pixi run -e map-compat python benchmarks/bench_map_smacs0723_field_fit_joint.py --bands F090W,F277W,F356W,F444W --crop-size 32 --crop-strategy best_overlap --steps 4 --models coadd_bilinear,spectral_basis_field,sky_encoder_resmlp,sky_lambda_feature_field,operator_constrained_feature_field

# Focused large sparse-mosaic frontier profile
# - keeps the exact 3-band multivisit row at 32 steps
# - drops the broad dense-overlap row to a low-memory 12-step profile at 128+
# - the 128+ broad row also switches to a smaller, deterministic-quadrature
#   `sky_lambda_feature_field` contract and weighted global MSE fitting so the
#   benchmark stays runnable
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 128 --steps 6 --step-profile sparse_mosaic_large_frontier --runs 1 --cases tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit,tpv_field_fit_sparse_large_mosaic_broad_denseoverlap --models coadd_bilinear,sky_lambda_feature_field,sky_encoder_resmlp

# The unified operator-aware follow-up family can be compared on the same sparse
# frontiers. It keeps a continuous spectral scaffold, adds observability-aware
# spectral regularization, and now uses a width-gated support-refinement head so
# exact or delta-like bands stay on the pure continuous path.
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 32 --steps 8 --step-profile sparse_mosaic_frontier --runs 1 --cases tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit,tpv_field_fit_sparse_large_mosaic_broad_denseoverlap --models coadd_bilinear,sky_lambda_feature_field,operator_constrained_feature_field

# The current regime scoreboard for that unified family is recorded in:
# `benchmark_results/map_field_fit_unified_regimes.md`

# Staged stronger non-INR baseline: a coadd-initialized explicit spectral grid
# refined directly against the observations through the same forward model.
# This is currently useful on the sparse exact 3-band gate, but not yet on the
# sparse broad-overlap gate.
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 32 --steps 8 --step-profile sparse_mosaic_frontier --runs 1 --cases tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit,tpv_field_fit_sparse_large_mosaic_broad_denseoverlap --models coadd_bilinear,explicit_spectral_grid_field

# Experimental latent-field fitting run with baseline regression gate
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 24 --steps 8 --runs 2 --cases tan_field_fit --models sine_implicit,coadd_bilinear --baseline-json benchmarks/baselines/map_field_fit_smoke.json

# Focused spectral-basis exact-band sweep helper
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/diagnose_field_fit_sweep.py --size 24 --steps 32,64 --hidden-dims 96,128 --hidden-layers 2 --spatial-frequencies 6,8 --spectral-frequencies 4,6

# Convenience Pixi gate tasks for the finalized map contract
pixi run bench-map-ground-truth-gate
pixi run bench-map-multifield-gate
pixi run bench-map-field-fit-gate
pixi run bench-map-compat-gate

# Larger representative gates with active timing checks on stack, multifield, and field-fit
pixi run bench-map-ground-truth-gate-rep
pixi run bench-map-multifield-gate-rep
pixi run bench-map-field-fit-gate-rep

# Aggregate acceptance entry points
pixi run bench-map-acceptance-smoke
pixi run bench-map-acceptance

# Mapmaking compatibility run (torchsky vs optional external reprojection libraries)
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_compat.py --size 256 --n-obs 3 --runs 3

# Competitive large-union stack benchmark on the fixed 10-image rectangular case
pixi run bench-map-ground-truth-competitive-large-union

# Aggregate the 512 / 1024 / 2048 competitive large-union runs into one report
pixi run bench-map-ground-truth-competitive-large-union-aggregate

# Competitive large-union stack benchmark on the rectangular target with explicit weight maps and masks
pixi run bench-map-ground-truth-competitive-large-union-rect-masked

# Aggregate the 512x384 / 1024x768 / 2048x1536 rectangular weight+mask runs into one report
pixi run bench-map-ground-truth-competitive-large-union-rect-masked-aggregate

# Combined TAN/SIP scoreboard, quick routine profile:
# - large-union TAN truth runs at 512x384 / 1024x768
# - weighted-stack diversity rows across aligned, noisy, multires, low-overlap, and mosaic TAN cases
# - weighted-mean final-stack comparisons against fair external tools on those diverse TAN cases
# - representative mean/median method sweep on the weighted+masked TAN family
# - real SIP compatibility runs against fair external tools
pixi run bench-map-tan-sip-suite

# Combined TAN/SIP scoreboard, full graduation profile:
# - large-union TAN truth runs at 512x384 / 1024x768 / 2048x1536
# - broader weighted-stack diversity coverage including stronger multires and larger mosaic cases
# - weighted-mean final-stack comparisons against fair external tools on those diverse TAN cases
# - representative mean/median method sweep on the weighted+masked TAN family
# - real SIP compatibility runs against the extended comparator set
pixi run bench-map-tan-sip-suite-full

# Combined wide-projection mosaicking scoreboard, quick routine profile:
# - synthetic truth-backed weighted-mean stacks across all current target projection families
#   (`SIN`, `ARC`, `ZEA`, `STG`, `CEA`, `MER`, `AIT`, `MOL`)
# - equal-PSF aligned controls on representative projections so weighted-mean behavior can be judged
#   without PSF-depth coupling
# - hard-row Projection Stack Methods sweep on CEA / AIT with `weighted_mean`, `mean`, and `median`
#   under `bilinear` and `lanczos3`
# - Montage-first compatibility rows on representative cylindrical and all-sky targets
# - INR Common-PSF rows:
#   - same-grid direct-stack vs INR projected products
#   - one shared implicit blur level per case (`common_psf_sigma_pix`)
#   - coordinate-mode comparison for INR (`local_tangent` vs `sphere_xyz`)
#   - pure observation-space INR fitting, with no direct-stack prefit
#   - robust Charbonnier loss plus cosine LR schedule / gradient clipping for INR rows
#   - quick profile keeps the strongest current future-facing INR candidates
#     (`wire_implicit@sphere_xyz`, `sky_encoder_mlp@sphere_xyz`, `hashgrid_mlp@sphere_xyz`)
#   - exact-wavelength product regressions live in the unit-test surface and compare
#     `render_field_product(...)` directly against per-band stacks on the same target grid
#   - direct-vs-INR RMSE/time deltas against the best direct stack baseline
# - support-stratified truth metrics (`support_fringe_q25_*`, `support_core_q75_*`) so wide-projection
#   rows can distinguish low-support fringe errors from high-support core behavior
pixi run bench-map-projection-mosaic-suite

# Combined wide-projection mosaicking scoreboard, full expansion profile:
# - aligned, shifted-overlap, multiresolution, low-overlap, and large-union rows across the target projection set
# - equal-PSF aligned controls across representative zenithal, cylindrical, and all-sky projections
# - large hard-row Projection Stack Methods sweep across zenithal, cylindrical, and all-sky families
#   (`ARC`, `STG`, `CEA`, `MER`, `AIT`, `MOL`) with `weighted_mean`, `mean`, and `median`
#   under `bilinear` and `lanczos3`
# - Montage compatibility rows on the fair medium/large projection cases
# - medium and large INR Common-PSF rows with both direct stacks and INR coordinate-mode sweeps
pixi run bench-map-projection-mosaic-suite-full

# Exact-band field-fit support-aware representative slice
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/bench_map_field_fit.py --size 24 --steps 64 --runs 1 --cases tan_field_fit_exact_transmission,tpv_field_fit_deep_multiband_exact --models spectral_basis_field,coadd_bilinear,sky_encoder_mlp
# Exact-band field-fit JSON rows now include:
# - `band_centers_angstrom`
# - `band_reconstruction_rmse`
# - `band_flux_relative_error`
# - `support_band_reconstruction_rmse`
# - `support_band_flux_relative_error`
# - uncertainty diagnostics:
#   - `uncertainty_kind`
#   - `uncertainty_error_corr`
#   - `uncertainty_top_bottom_abs_err_ratio`
#   - `uncertainty_support_mean_sigma`
#   - `uncertainty_extrapolated_mean_sigma`

# Residual diagnostic for the exact-band field-fit gates, using the tuned per-case model defaults
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/diagnose_field_fit_residuals.py --case tpv_field_fit_deep_multiband_exact --model-kind sky_encoder_mlp
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/diagnose_field_fit_residuals.py --case tan_field_fit_exact_transmission --model-kind spectral_basis_field

# Broad-band deep-field representative slice
# These rows emulate the next-stage same-field HST/JWST-style challenge more closely:
# repeated visits per broad band, mixed scale/depth/coverage, known transmissions,
# shared PSF, and TAN-SIP / TPV source distortion.
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/bench_map_field_fit.py --size 24 --steps 64 --runs 1 --cases tan_sip_field_fit_deep_multiband_broad,tpv_field_fit_deep_multiband_broad --models coadd_bilinear,sky_encoder_bandpass_field,sky_encoder_resmlp

# Sparse larger-mosaic deep-field smoke
# These rows keep the individual visits small while growing the requested product
# field, closer to the "many HST/JWST visits on one larger mosaic" regime.
PYTHONPATH=src:benchmarks pixi run -e test python benchmarks/bench_map_field_fit.py --size 128 --steps 8 --runs 1 --cases tan_sip_field_fit_sparse_large_mosaic_broad,tpv_field_fit_sparse_large_mosaic_exact --models coadd_bilinear,sky_encoder_bandpass_field,sky_encoder_mlp

# The simulation and evaluation flow for these map truth benchmarks is documented in:
# docs/map_truth_pipeline.md

# Real-fixture compatibility smoke gate on the fixed Astropy SIP products
pixi run -e sphere-bench python scripts/run_local_sphere_bench.py benchmarks/bench_map_compat.py --size 32 --n-obs 2 --runs 1 --cases real_astropy_sip_fixture,real_astropy_sip_shifted_wcs,real_astropy_sip_mef,real_astropy_sip_mef_shifted_wcs --torchsky-kernels bilinear --reproject-kernels bilinear --drizzle-kernels point --swarp-kernels bilinear --baseline-json benchmarks/baselines/map_compat_real_smoke.json

# External-tool compatibility run in the dedicated Python-3.13 env
pixi run -e map-compat python scripts/run_local_sphere_bench.py benchmarks/bench_map_compat.py --size 256 --n-obs 3 --runs 3

# Bootstrap official source builds for external map comparators
pixi run -e map-compat python scripts/bootstrap_map_compat_tools.py --tool swarp
pixi run -e map-compat python scripts/bootstrap_map_compat_tools.py --tool montage
pixi run -e map-compat python scripts/bootstrap_map_compat_tools.py --tool casutools
# `casutools` defaults to downloading `casutools-1.0.43.tar.gz`; override with
# `CASUTOOLS_SOURCE=/path/to/tree` or `CASUTOOLS_URL=https://...tar.gz` if needed.

# WCS parity+throughput run with mixed interior/boundary sampling and p99 gates
pixi run python benchmarks/bench_wcs.py --cases TAN,SIN,ARC,AIT,MOL,HPX,CEA,MER,TAN_SIP,TPV --sample-profile mixed --n-points 200000 --runs 5 --p99-angular-error-arcsec 1e-2 --p99-inverse-pixel-error 1e-2

# WCS small-N center-projection check (bench_wcs emits center_cpp_* fields when active)
pixi run python benchmarks/bench_wcs.py --cases AIT,HPX,CEA,MER,CAR,SFL --sample-profile mixed --n-points 512 --runs 9

# Dedicated TPV inverse primitive benchmark
pixi run bench-wcs-tpv-inverse

# HEALPix parity+throughput run on CPU (strict parity gates)
pixi run python benchmarks/bench_healpix.py --device cpu --sample-profile mixed --nside 1024 --n-points 200000 --runs 5 --max-index-mismatches 0 --max-pix2ang-dra-deg 1e-10 --max-pix2ang-ddec-deg 1e-10

# HEALPix CUDA run with CPU comparison gate
pixi run python benchmarks/bench_healpix.py --device cuda --compare-cpu --sample-profile mixed --nside 1024 --n-points 200000 --runs 5 --min-cuda-speedup-vs-cpu 2.0

# HEALPix Apple GPU (MPS) run with CPU comparison (float32 thresholds)
pixi run python benchmarks/bench_healpix.py --device mps --compare-cpu --sample-profile uniform --nside 1024 --n-points 200000 --runs 5 --max-index-mismatches 10 --max-pix2ang-dra-deg 5e-4 --max-pix2ang-ddec-deg 5e-4

# Advanced HEALPix benchmark (neighbors + interpolation)
pixi run -e sphere-bench sphere-bench-healpix-advanced

# Advanced HEALPix benchmark gate (ratio thresholds vs healpy)
pixi run -e sphere-bench sphere-bench-healpix-advanced-gate

# Advanced HEALPix small-N gate (overhead-sensitive)
pixi run -e sphere-bench sphere-bench-healpix-advanced-gate-small

# Sphere-core benchmark (multi-band sampling + geometry primitives)
pixi run -e sphere-bench sphere-bench-core

# Non-convex polygon benchmark (contains/intersects/query/area)
pixi run -e sphere-bench sphere-bench-polygons

# Catalog crossmatch refinement benchmark (declination vs coarse HEALPix candidate pruning)
pixi run -e sphere-bench sphere-bench-catalog-crossmatch

# Catalog crossmatch compatibility benchmark (torchsky vs astropy SkyCoord)
pixi run -e sphere-bench sphere-bench-catalog-compat

# Spectral primitive benchmark (CPU-first scalar + spin transforms)
pixi run -e sphere-bench sphere-bench-spectral

# End-to-end table pushdown + sphere pipeline benchmark
pixi run -e sphere-bench sphere-bench-pipeline

# End-to-end table pushdown + sphere pipeline gate (ratio thresholds)
pixi run -e sphere-bench sphere-bench-pipeline-gate

# Install optional sphere benchmark comparators from official releases
pixi run -e sphere-bench sphere-bench-bootstrap

# Install only core HEALPix comparator packages (lean setup)
pixi run -e sphere-bench sphere-bench-bootstrap-core

# Sync upstream release source artifacts and extract tests/data fixtures
pixi run -e sphere-bench sphere-upstream-sync

# Replay astropy-healpix upstream test-style HEALPix cases on torchsky vs healpy
pixi run -e sphere-bench sphere-upstream-gate

# Replay upstream test functions directly (including shape semantics + hypothesis inner tests)
pixi run -e sphere-bench sphere-upstream-test-gate

# Replay/gate upstream healpy spin transform cases
pixi run -e sphere-bench sphere-upstream-spin-gate

# Optional extended spin replay matrix (higher-complexity spin-2 cases)
pixi run -e sphere-bench python benchmarks/replay_upstream_healpy_spin.py --case-set extended --runs 5

# Replay/gate upstream interpolation edge cases (pole/wrap/pixel-boundary semantics)
pixi run -e sphere-bench sphere-upstream-interp-edge-gate

# Replay/gate upstream spherical-geometry polygon fixtures/data
pixi run -e sphere-bench sphere-upstream-polygon-gate

# Cross-library sphere geometry benchmark (fails if comparator comes from local/editable/VCS install)
pixi run -e sphere-bench sphere-bench-geometry-fast

# Ecosystem benchmark snapshot against released packages (healpy/hpgeom/astropy-healpix/healpix/mhealpy)
pixi run -e sphere-bench sphere-bench-geometry-ecosystem

# Cross-library sphere geometry matrix with correctness + median ratio gates
pixi run -e sphere-bench sphere-bench-geometry-gate

# Small-N matrix/profile gate to catch overhead-sensitive regressions
pixi run -e sphere-bench sphere-bench-geometry-gate-small
```

## Sphere Benchmark Policy

- Comparator libraries must come from official released distributions (pip/uv/conda), not local repo clones.
- `bench_healpix.py` and `bench_sphere_geometry.py` now print comparator package provenance and fail by default on local/editable/VCS installs.
- `bench_sphere_matrix.py` runs `uniform/boundary/mixed` profiles and can enforce median torchsky/healpy speed-ratio floors.
- `bench_sphere_polygons.py` benchmarks non-convex polygon contains/intersects/query/area, includes medium-size `contains` threshold rows plus upstream difficult-overlap intersection rows, optionally checks parity against released `spherical-geometry`, and now emits `torchsky_recommendation` rows describing the preferred contains/intersection backend on the measured workload.
- `scripts/run_local_sphere_bench.py` is used by the polygon bench/replay tasks to editable-install the local package into the active `sphere-bench` env on demand, so those tasks can see the native `torchsky.cpp` extension instead of falling back to Python-only imports.
- `bench_catalog_crossmatch.py` benchmarks crossmatch candidate generation with parity checks between the legacy declination-pruned path and the internal coarse nested-HEALPix index, reports candidate-row / pair-count reduction ratios plus the selected coarse NSIDE, and emits a `torchsky_recommendation` row summarizing which strategy should be preferred on the measured workload.
- `bench_catalog_compat.py` benchmarks `torchsky.catalogs.crossmatch_sky` against `astropy.coordinates.SkyCoord.match_to_catalog_sky`, using both synthetic clustered catalogs and vendored real Astropy VOTable fixtures; when the native table extension is present it also runs a real-data FITS roundtrip case derived from those fixtures.
- `bench_map_image_plane.py` benchmarks the current image-plane coadd path on both synthetic observations and real Astropy SIP FITS image fixtures, including shifted-SIP reprojection, generated-MEF cases, and a synthetic rotated low-overlap TPV MEF case, and also times both the multiband compatibility wrapper and the header-derived planned spectral-cube path on a shared target grid; it reports wall time, Mpix/s, coverage fraction, mean accumulated weight, max overlap, footprint area, spectral-axis metadata, and summed spectral projection weight. Run it through `scripts/run_local_sphere_bench.py` so the active env sees the local native extension.
- `bench_map_ground_truth.py` is the mapmaking correctness harness. It does not use `reproject`, `Montage`, `drizzle`, `SWarp`, or any other package as truth. Instead it generates a latent analytic sky field, simulates calibrated observations with known exposure time / transmission / PSF / noise / mask structure, and scores methods against that ground truth with `rmse_vs_truth`, `mae_vs_truth`, `p95_abs_err_vs_truth`, `max_abs_err_vs_truth`, `signal_rms_truth`, `residual_rms`, `output_snr`, `output_snr_db`, and integrated-flux metrics (`total_flux_truth`, `total_flux_output`, `total_flux_abs_err`, `total_flux_rel_err`). Torchsky rows now form a kernel matrix across `nearest`, `bilinear`, `bicubic`, `lanczos3`, `lanczos5`, `mitchell_netravali`, and `catmull_rom`, the synthetic case set includes harder resolution-ratio sweeps plus explicit `flux`-semantics cases, and `--include-external` extends the same truth scoring to the available external backends and variants. Flux-semantics rows also record the synthetic `noise_model`; the current contract uses area-scaled flux noise for flux-per-pixel cases so distorted TPV exposures are judged under a physically consistent variance model instead of a constant-flux-noise artifact. The TPV flux set is split into a milder regression-gated case and a harder stress case that stays in the matrix as an optimization target rather than being hidden. The synthetic TPV benchmark family now uses a non-degenerate linear TPV component and sampled local-Jacobian checks so the warped cases remain locally invertible rather than benchmarking fold/branch ambiguity. In the current torchsky path, distorted `SIP`/`TPV` reprojection uses fixed 2x2 Gauss-Legendre target-pixel quadrature for `flux` products, while the default `surface_brightness` path uses a single center sample; the quadrature world grids are cached once per prepared target so distorted large-map benchmarks measure reprojection quality rather than repeated target-grid recomputation overhead. CPU fused WCS is now enabled by default for supported projections, including TPV, and can be disabled with `TORCHSKY_WCS_FUSED=0` for debugging or comparator studies. The script also emits `torchsky_recommendation` rows per case, including a practical default (`fastest_within_1pct_rmse`), the best interpolatory kernel, and the best smooth reconstruction kernel from the measured torchsky rows.
- `bench_map_ground_truth.py` now evaluates methods on the final requested product contract instead of an implicit script-local output shape. The primary metric vocabulary is standardized with the field-fit benchmark: `reconstruction_rmse`, `reconstruction_mae`, `reconstruction_p95_abs_err`, `reconstruction_max_abs_err`, `signal_to_residual_ratio`, `signal_to_residual_db`, `total_flux_relative_error`, `band_flux_relative_error_mean/max`, `coverage_fraction`, plus color-ratio errors where multiple bands exist. The script writes JSON plus a concise Markdown summary table via the shared map benchmark helpers. `benchmarks/baselines/map_ground_truth_smoke.json` is the lightweight smoke baseline; it gates correctness/SNR/flux on a small TAN case and intentionally skips time regression because the smoke-scale timing is too noisy to be meaningful on its own. `benchmarks/baselines/map_ground_truth_rep.json` is the larger representative stack baseline with active timing checks on `tan_multires_noise` and `tpv_flux_multires_mild`.
- `bench_map_multifield_truth.py` is the first dedicated continuous-field, multiwavelength mapmaking benchmark. It constructs latent spectral-spatial fields with two model families: an explicit analytic basis field and an initial frozen implicit-neural-field (`implicit_siren`) variant. Observation synthesis now uses normalized Gaussian transmission filters across each observation bandpass rather than a pure top-hat average, and the core cube stack path is evaluated on the same product-oriented metric vocabulary as the other map truth benches. The scalar throughput parameter still controls noise level under the assumption that photometric calibration is already known. The benchmark simulates overlapping observations with mixed TAN/TPV geometry, PSF width, bandpass, exposure time, and throughput, then scores the recovered final product against latent cube truth with `reconstruction_*`, SNR, flux, and per-band summaries while still carrying the older `cube_rmse_vs_truth` aliases for continuity. Like `bench_map_ground_truth.py`, it also emits `torchsky_recommendation` rows so the kernel choice is benchmark-backed rather than static. Its synthetic TPV family now matches the locally invertible benchmark policy used in the map ground-truth harness, so the continuous-field warped cases stay physically coherent. It now supports rectangular target grids via `--width/--height`, explicit large-scale rectangular cases with mixed per-observation `NAXIS1/NAXIS2`, and a deterministic ten-image stress case with random widths/heights in the 500–4000 range, mixed TAN/TPV geometry, mixed pixel scales, random shifts/rotations, and deliberate partial non-overlap. `benchmarks/baselines/map_multifield_truth_smoke.json` is the lightweight multifield baseline for `tan_multifield_cube` and skips time regression only where the smoke-scale run is too noisy; `benchmarks/baselines/map_multifield_truth_rep.json` is the larger representative multifield baseline with active timing checks on `tan_multifield_cube` and `tpv_multifield_cube`.
- `bench_map_field_fit.py` is now fully product-oriented too: direct stack baselines and fitted field models are both rendered onto the same requested final product before scoring. The current case matrix spans `TAN`, `TAN-SIP`, and `TPV`; implicit and analytic latent fields; low-overlap / mixed-depth / multiresolution / PSF-mismatch environments; exact-transmission as well as Gaussian-filter spectral targets; and both smaller multicolor and deeper same-field repeated-visit broad-band products. It also now includes sparse larger-mosaic rows where many small visits are fit onto a much larger target product, which is the current bridge from local deep-field INR success to realistic mosaic-scale latent modeling. The model matrix includes `sky_encoder_bandpass_field`, a transmission-aware sky-encoder branch that aggregates bandpass-weighted multiscale sky features before decoding, `sky_encoder_resmlp`, which is currently the strongest broad-band deep-field INR branch after tuning its architecture, `mixture_mc`-based training setup, and case-specific readout policy, and the newer `operator_constrained_feature_field`, which keeps a continuous spectral scaffold but constrains its degrees of freedom through the actual bandpass operator and a small zero-initialized bandpass residual path. Reported metrics use the same standardized product vocabulary as the stack-truth benchmark, plus `fit_time_ms`, observation-fit summaries, support-aware product metrics keyed off the `coadd_bilinear` support mask, and true uniform-grid metrics that help separate supported-pixel fidelity from global-field coherence. `benchmarks/baselines/map_field_fit_smoke.json` is the lightweight smoke baseline and intentionally skips timing on its tiny coadd helper rows. `benchmarks/baselines/map_field_fit_rep.json` is the larger representative field baseline with active timing checks on the field-fit defaults across the current representative rows.
- `diagnose_field_fit_sparse_frontier.py` is the focused scale-ladder surface for those sparse larger-mosaic rows. By default it keeps the honest four-way comparison in view: `coadd_bilinear`, the regular `sky_encoder_mlp` INR, `sky_lambda_kplanes_field`, and the newer `sky_lambda_feature_field`. It sweeps one or more sparse cases across target sizes, fit-step budgets, and selected field models, then reports full-grid and support-aware product metrics on the same final grids. This is the script to use before concluding that a sparse-mosaic INR gap is architectural rather than budget-limited.
- `bench_wcs_tpv_inverse.py` isolates the current warped-map hotspot from the full stack contract. It benchmarks only TPV `world_to_pixel` at large `N`, using Astropy-generated world coordinates from known pixel samples and scoring inverse-pixel reconstruction against the original pixel coordinates. The current matrix is `torchsky` fused TPV inverse, `torchsky` tensor-core TPV inverse with fused WCS disabled, and Astropy `all_world2pix`, across both an affine-only TPV header and a non-linear broad-field TPV header. Reported metrics are `time_ms`, `mpts_s`, `mean_inverse_pixel_error`, `p99_inverse_pixel_error`, `max_inverse_pixel_error`, and tensor-path iteration summaries where available. This is the authoritative surface for further TPV inverse kernel work; map-level TPV optimizations should be justified here before being inferred from the full mapmaking benchmark.
- `bench_map_compat.py` is the external mapmaking comparison harness. It now emits a concrete variant matrix across torchsky `nearest`/`bilinear`/`bicubic`/`lanczos3`/`lanczos5`/`mitchell_netravali`/`catmull_rom` kernels, `reproject_interp`, `dfreproject`, low-level `drizzle`, `drizzlepac`, `Montage`, `SWarp`, and `casutools imstack`, plus explicit pairwise comparison rows for every successful method pair in a case. `reproject_interp` and `dfreproject` are wired with kernel variants, `drizzle` is wired with multiple low-level kernels, and `SWarp` is wired with multiple resampling modes where the requested target grid is representable. For the pure resampling/stacking milestone, the SWarp benchmark helper now writes `FSCALASTRO_TYPE NONE` by default so hidden astrometric flux scaling does not leak calibration semantics into the comparator rows. The low-level `drizzle` backend is wired through upstream-style `drizzle.utils.calc_pixmap(...)` and `drizzle.resample.Drizzle.add_image(...)`, and includes an additional CRPIX-translation case patterned after drizzle's own `test_translated_map`. The `Montage` backend is wired through source-built official binaries and now uses `mProjectPP` automatically for the fast plane-to-plane cases it actually supports, while falling back to `mProject` for SIP/distorted inputs. The harness also includes generated MEF cases built from real Astropy SIP data plus a synthetic rotated low-overlap TPV MEF case so extension-addressed `SCI/ERR/DQ` style workflows are exercised in addition to single-HDU inputs. `SWarp` is wired through a source-built CLI path for simple equatorial non-distorted target grids and can therefore participate in the TPV-input/TAN-target synthetic case, while SIP targets are reported as explicit skips instead of being benchmarked against an inexact surrogate grid. `casutools imstack` is wired from a local source build and is only exercised on the documented CASU WCS regime (`TAN`/`ZPN`, no `SIP`/`PV` distortion); because its confidence maps are renormalized confidence images rather than direct inverse-variance weights, its comparison rows use coverage-mask semantics for weight/coverage agreement and common-support semantics for data deltas. Some comparators can still fail on the harder TPV inverse problem; those now surface as explicit `SKIPPED` rows with the upstream error text instead of aborting the matrix. The script reports both global and edge-vs-interior deltas so boundary-convention differences do not get mistaken for full-field reprojection errors, and it now reports those deltas directly between successful method variants. Missing, unsupported, or not-yet-wired comparators show up as explicit `SKIPPED` rows rather than silently disappearing, and import/ABI failures are reported verbatim when available. It also now accepts `--cases`, writes JSON plus a concise Markdown summary table, and can enforce a committed real-fixture compat baseline through `benchmarks/baselines/map_compat_real_smoke.json`. The default `sphere-bench` env stays on the main current Python stack; the dedicated `map-compat` env pins Python `3.13` so lagging external packages can still be evaluated when upstream packaging allows it. The bench env pins PyPI `drizzle==2.2.0` and `dfreproject==1.3.1` explicitly so we do not accidentally benchmark older lineages or conflate `drizzle` with `drizzlepac`.
- `run_ground_truth_competitive_large_union.py` is the heavy end-to-end competitor runner for the large union-stack cases. It compares torchsky against the current external stackers on a fixed final target product and writes incremental JSON plus Markdown artifacts while the run is in progress. Torchsky rows now default to a warmed median of three timed runs (`--torchsky-runs 3`) so the heavy warped cases have a more stable internal baseline than a single-shot wall clock; external comparator rows still use the requested `--runs` contract unless a backend requires a minimum of two runs.
- `diagnose_map_large_union_reprojection.py` is the follow-up diagnostic surface for those same large-union cases. It is intentionally narrower than `bench_map_compat.py`: instead of surveying many backends, it compares `torchsky` and `dfreproject` observation-by-observation on the same synthetic truth case, then emits pairwise reprojection deltas and stack summaries so regressions can be localized to specific warped inputs or to the post-reprojection accumulation stage.
- `diagnose_map_large_union_accumulation.py` is the companion diagnostic for the surviving TAN weighted overlap gap. It compares grouped and per-observation accumulation for both `torchsky` and `SWarp`, reports overlap-count histograms and overlap-binned residual statistics, and emits explicit data/weight/numerator hotspots so a final stack mismatch can be traced to a particular overlap regime and, when needed, to a specific contributing observation at a specific target pixel.
- `bench_map_boundary_support.py` is the decision surface for explicit bilinear edge-support policies on the large-union TAN weighted cases. It monkeypatches candidate source-boundary margins only for the benchmark run, then scores each margin against analytic truth and against grouped `SWarp` output. This is the benchmark to use before changing any production boundary-support rule in `image_plane.py`.
- The current large-union TAN edge hunt found an important comparator nuance: on the representative `2048x1536` weighted hotspot, `torchsky`, `reproject_interp`, and `dfreproject` agree on the single-observation bilinear sample while `SWarp` returns zero support. That means the surviving grouped `SWarp` mismatch is not automatically evidence that torchsky's bilinear boundary semantics are wrong; benchmark-only edge-policy experiments should be judged against synthetic truth and at least one additional comparator, not only against `SWarp`.
- `scripts/bootstrap_map_compat_tools.py` clones and builds the official upstream SWarp and Montage sources and can also stage a CASUtools build into `.benchmarks/external-tools/prefix/{swarp,montage,casutools}` from inside the `map-compat` env, either from `CASUTOOLS_SOURCE` or from the default public `casutools-1.0.43.tar.gz` URL. The CASUtools bootstrap expects `cfitsio` and `wcslib` development headers in the active `map-compat` env (or `WCSLIB_INCLUDE` / `WCSLIB_LIB` overrides). `bench_map_compat.py` checks those repo-managed install prefixes before falling back to `PATH`, so source-built external tools show up in the matrix even when Python wheels are unusable on the host architecture.
- The current `SWarp` benchmark path is intentionally narrow: it only runs when the requested target grid can be represented by SWarp's manual output-grid controls. SIP-distorted, PV-distorted, rotated, sheared, or non-equatorial target WCSes are reported as `SKIPPED` rather than benchmarked against an inexact surrogate grid.
- External mapmaking comparisons should source cases in this order:
  1. official upstream tests/fixtures from the comparator package when they exist
  2. popular pipeline scenarios that actually exercise the package in production, for example DrizzlePac/HST and JWST level-3 style resampling flows
  3. torchsky-owned synthetic stress cases used to probe scaling, boundary behavior, and cross-device performance
- `replay_upstream_spherical_geometry_polygons.py` replays polygon fixtures/data from upstream spherical-geometry test suite with correctness gates (contains + pair intersections + difficult-case nonempty parity + multi-NSIDE overlap-area/convergence checks, default ladder `128..16384`); a reduced version of this replay is also exercised from `pytest` as a regression gate.
- `sync_upstream_sphere_sources.py` downloads release artifacts for selected packages and extracts upstream `tests/` + `data/` trees with a pinned manifest.
- `replay_upstream_astropy_healpy.py` replays astropy-healpix test-style HEALPix cases (same NSIDE range and boundary example-style vectors) against torchsky/healpy with correctness + ratio gates.
- `replay_upstream_test_functions.py` executes selected upstream astropy-healpix test functions directly against torchsky adapters for closer semantic parity checks.
- `replay_upstream_healpy_interp_edges.py` replays interpolation edge cases (lon wrap, poles, edge pixels) against healpy with mismatch/error and ratio gates.
- `replay_upstream_healpy_spin.py` replays spin transform cases (`map2alm_spin`, `alm2map_spin`) against healpy with per-case error gates and median throughput ratio thresholds.
- Spin replay defaults to the current validated profile (`TORCHSKY_RING_FOURIER_CPP=1`, recurrence off, `TORCHSKY_SPIN_RING_AUTO_MIN_BYTES=32 MiB`) unless explicitly overridden in the environment.
- Override only for local development experiments with:
  - `bench_healpix.py --allow-nonrelease-healpy`
  - `bench_sphere_geometry.py --allow-nonrelease-distributions`

## Optimization History

For a concise log of attempted optimizations and outcomes (including dead ends), see:
- `docs/performance_attempts.md`

## 0.3.1 Release Snapshot

Run metadata:
- Date: `2026-03-06`
- Exhaustive output: `benchmarks_results/20260306_191113`
- Profile: `user` (`cache=10`, `handle_cache=16`, `hot_cache=10`)
- Tables included: `True`

### Astronomer Scorecard (Win Rates)

| Domain | Family | torchsky First | Win Rate | Legacy In Ranking |
|---|---|---:|---:|---:|
| fits | smart | 83/84 | 98.8% | 0 |
| fits | specialized | 165/165 | 100.0% | 0 |
| fitstable | smart | 70/70 | 100.0% | 0 |
| fitstable | specialized | 70/70 | 100.0% | 0 |
| sphere | specialized | 16/18 | 88.9% | 0 |
| wcs | smart | 121/150 | 80.7% | 75 |

### FITS I/O Representative Medians (vs `fitsio`)

| Case | torchsky median (s) | Fitsio median (s) | Speedup vs Fitsio |
|------|---------------------:|------------------:|------------------:|
| `tiny_int8_1d [read_full]` | 0.000044 | 0.000185 | 4.20x |
| `small_float32_2d [read_full]` | 0.000075 | 0.000212 | 2.82x |
| `medium_int16_3d [read_full]` | 0.000524 | 0.000782 | 1.49x |
| `large_float64_2d [read_full]` | 0.005312 | 0.005470 | 1.03x |

### Known Deficits (Target for 0.4.0)

- **Sphere (Spin Transforms)**: `map2alm_spin` and `alm2map_spin` show ~6x-7x lag vs `healpy`. These will be optimized with a custom RING-Fouier kernel.
- **WCS (Small-N)**: Some forward small-N rows still lag vs `pyast` (legacy), especially away from the center-equator projections. `bench_wcs.py` now tracks the CPU center-helper crossover explicitly through `center_cpp_*` fields for `AIT/CAR/CEA/HPX/MER/SFL`.
- **ZPN/AIT Projections**: Slight visible deficits in intermediate size tiers.

## ML Loader Snapshot (0.3.0)

Command used:

```bash
pixi run python benchmarks/bench_ml_loader.py --device cpu --shape 2048,2048 --n-files 50 --batch-size 4 --num-workers 4 --epochs 3 --repeats 5 --warm-cache
```

Results (Median of 5 runs):
- Uncompressed: `0.985x` (vs `fitsio + numpy`)
- Compressed (RICE): `1.008x` (vs `fitsio`)
