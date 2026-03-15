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
| `bench_map_compat.py` | Mapmaking compatibility benchmark comparing torchsky image-plane coadds against an explicit external-tool matrix. `reproject_interp`, `dfreproject`, low-level `drizzle`, source-built `Montage`, source-built `SWarp`, and local-source `casutools imstack` are wired where their target/grid semantics are comparable; `drizzlepac` and unsupported or backend-failing target/grid combinations appear as explicit `SKIPPED` rows with concrete reasons. |
| `bench_map_ground_truth.py` | Synthetic mapmaking correctness benchmark against known sky truth. It evaluates final requested map products on a shared `MapProductSpec`, emits machine-readable JSON plus a concise Markdown summary table, and can optionally enforce representative baseline JSON gates. |
| `bench_map_multifield_truth.py` | Synthetic multiwavelength latent-field cube benchmark. It simulates wavelength-dependent continuous sky fields with mixed bandpasses, PSFs, WCS distortions, throughput, and overlap, using both analytic fields and an initial frozen implicit-neural-field variant, then scores `coadd_image_plane_cube(...)` against latent cube truth and emits per-case torchsky kernel recommendations. |
| `bench_map_field_fit.py` | Experimental continuous-field fitting benchmark. It renders fitted field models and direct stack baselines onto the same requested final product, emits JSON plus Markdown summaries, and can optionally enforce representative baseline JSON gates. |
| `bench_wcs.py` | WCS transform throughput checks, including small-batch center-projection `default vs python-only` fields on CPU. |
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
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 48 --steps 120 --runs 1

# Experimental latent-field fitting run with baseline regression gate
pixi run -e sphere-bench python benchmarks/bench_map_field_fit.py --size 24 --steps 8 --runs 2 --cases tan_field_fit --models sine_implicit,coadd_bilinear --baseline-json benchmarks/baselines/map_field_fit_smoke.json

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

# WCS parity+throughput run with mixed interior/boundary sampling and p99 gates
pixi run python benchmarks/bench_wcs.py --cases TAN,SIN,ARC,AIT,MOL,HPX,CEA,MER,TAN_SIP,TPV --sample-profile mixed --n-points 200000 --runs 5 --p99-angular-error-arcsec 1e-2 --p99-inverse-pixel-error 1e-2

# WCS small-N center-projection check (bench_wcs emits center_cpp_* fields when active)
pixi run python benchmarks/bench_wcs.py --cases AIT,HPX,CEA,MER,CAR,SFL --sample-profile mixed --n-points 512 --runs 9

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
- `bench_map_ground_truth.py` is the mapmaking correctness harness. It does not use `reproject`, `Montage`, `drizzle`, `SWarp`, or any other package as truth. Instead it generates a latent analytic sky field, simulates calibrated observations with known exposure time / transmission / PSF / noise / mask structure, and scores methods against that ground truth with `rmse_vs_truth`, `mae_vs_truth`, `p95_abs_err_vs_truth`, `max_abs_err_vs_truth`, `signal_rms_truth`, `residual_rms`, `output_snr`, `output_snr_db`, and integrated-flux metrics (`total_flux_truth`, `total_flux_output`, `total_flux_abs_err`, `total_flux_rel_err`). Torchsky rows now form a kernel matrix across `nearest`, `bilinear`, `bicubic`, `lanczos3`, `lanczos5`, `mitchell_netravali`, and `catmull_rom`, the synthetic case set includes harder resolution-ratio sweeps plus explicit `flux`-semantics cases, and `--include-external` extends the same truth scoring to the available external backends and variants. Flux-semantics rows also record the synthetic `noise_model`; the current contract uses area-scaled flux noise for flux-per-pixel cases so distorted TPV exposures are judged under a physically consistent variance model instead of a constant-flux-noise artifact. The TPV flux set is split into a milder regression-gated case and a harder stress case that stays in the matrix as an optimization target rather than being hidden. The synthetic TPV benchmark family now uses a non-degenerate linear TPV component and sampled local-Jacobian checks so the warped cases remain locally invertible rather than benchmarking fold/branch ambiguity. In the current torchsky path, distorted `SIP`/`TPV` reprojection also uses fixed 2x2 Gauss-Legendre target-pixel quadrature instead of a single center sample, and those four subpixel world grids are cached once per prepared target so distorted large-map benchmarks measure reprojection quality rather than repeated target-grid recomputation overhead. The script also emits `torchsky_recommendation` rows per case, including a practical default (`fastest_within_1pct_rmse`), the best interpolatory kernel, and the best smooth reconstruction kernel from the measured torchsky rows.
- `bench_map_ground_truth.py` is the mapmaking correctness harness. It does not use `reproject`, `Montage`, `drizzle`, `SWarp`, or any other package as truth. Instead it generates a latent analytic sky field, simulates calibrated observations with known exposure time / transmission / PSF / noise / mask structure, and scores methods against that ground truth with `rmse_vs_truth`, `mae_vs_truth`, `p95_abs_err_vs_truth`, `max_abs_err_vs_truth`, `signal_rms_truth`, `residual_rms`, `output_snr`, `output_snr_db`, and integrated-flux metrics (`total_flux_truth`, `total_flux_output`, `total_flux_abs_err`, `total_flux_rel_err`). Torchsky rows now form a kernel matrix across `nearest`, `bilinear`, `bicubic`, `lanczos3`, `lanczos5`, `mitchell_netravali`, and `catmull_rom`, the synthetic case set includes harder resolution-ratio sweeps plus explicit `flux`-semantics cases, and `--include-external` extends the same truth scoring to the available external backends and variants. Flux-semantics rows also record the synthetic `noise_model`; the current contract uses area-scaled flux noise for flux-per-pixel cases so distorted TPV exposures are judged under a physically consistent variance model instead of a constant-flux-noise artifact. The TPV flux set is split into a milder regression-gated case and a harder stress case that stays in the matrix as an optimization target rather than being hidden. The synthetic TPV benchmark family now uses a non-degenerate linear TPV component and sampled local-Jacobian checks so the warped cases remain locally invertible rather than benchmarking fold/branch ambiguity. In the current torchsky path, distorted `SIP`/`TPV` reprojection also uses fixed 2x2 Gauss-Legendre target-pixel quadrature instead of a single center sample, and those four subpixel world grids are cached once per prepared target so distorted large-map benchmarks measure reprojection quality rather than repeated target-grid recomputation overhead. CPU fused WCS is now enabled by default for supported projections, including TPV, and can be disabled with `TORCHSKY_WCS_FUSED=0` for debugging or comparator studies. The script also emits `torchsky_recommendation` rows per case, including a practical default (`fastest_within_1pct_rmse`), the best interpolatory kernel, and the best smooth reconstruction kernel from the measured torchsky rows.
- `bench_map_ground_truth.py` now evaluates methods on the final requested product contract instead of an implicit script-local output shape. The primary metric vocabulary is standardized with the field-fit benchmark: `reconstruction_rmse`, `reconstruction_mae`, `reconstruction_p95_abs_err`, `reconstruction_max_abs_err`, `signal_to_residual_ratio`, `signal_to_residual_db`, `total_flux_relative_error`, `band_flux_relative_error_mean/max`, `coverage_fraction`, plus color-ratio errors where multiple bands exist. The script writes JSON plus a concise Markdown summary table via the shared map benchmark helpers. `benchmarks/baselines/map_ground_truth_smoke.json` is the lightweight smoke baseline; it gates correctness/SNR/flux on a small TAN case and intentionally skips time regression because the smoke-scale timing is too noisy to be meaningful on its own. `benchmarks/baselines/map_ground_truth_rep.json` is the larger representative stack baseline with active timing checks on `tan_multires_noise` and `tpv_flux_multires_mild`.
- `bench_map_multifield_truth.py` is the first dedicated continuous-field, multiwavelength mapmaking benchmark. It constructs latent spectral-spatial fields with two model families: an explicit analytic basis field and an initial frozen implicit-neural-field (`implicit_siren`) variant. Observation synthesis now uses normalized Gaussian transmission filters across each observation bandpass rather than a pure top-hat average, and the core cube stack path is evaluated on the same product-oriented metric vocabulary as the other map truth benches. The scalar throughput parameter still controls noise level under the assumption that photometric calibration is already known. The benchmark simulates overlapping observations with mixed TAN/TPV geometry, PSF width, bandpass, exposure time, and throughput, then scores the recovered final product against latent cube truth with `reconstruction_*`, SNR, flux, and per-band summaries while still carrying the older `cube_rmse_vs_truth` aliases for continuity. Like `bench_map_ground_truth.py`, it also emits `torchsky_recommendation` rows so the kernel choice is benchmark-backed rather than static. Its synthetic TPV family now matches the locally invertible benchmark policy used in the map ground-truth harness, so the continuous-field warped cases stay physically coherent. It now supports rectangular target grids via `--width/--height`, explicit large-scale rectangular cases with mixed per-observation `NAXIS1/NAXIS2`, and a deterministic ten-image stress case with random widths/heights in the 500–4000 range, mixed TAN/TPV geometry, mixed pixel scales, random shifts/rotations, and deliberate partial non-overlap. `benchmarks/baselines/map_multifield_truth_smoke.json` is the lightweight multifield baseline for `tan_multifield_cube` and skips time regression only where the smoke-scale run is too noisy; `benchmarks/baselines/map_multifield_truth_rep.json` is the larger representative multifield baseline with active timing checks on `tan_multifield_cube` and `tpv_multifield_cube`.
- `bench_map_field_fit.py` is now fully product-oriented too: direct stack baselines and fitted field models are both rendered onto the same requested final product before scoring. The current case matrix spans `TAN`, `TAN-SIP`, and `TPV`; implicit and analytic latent fields; low-overlap / mixed-depth / multiresolution / PSF-mismatch environments; and three- and four-band targets with known Gaussian transmission filters. The model matrix covers `sine_implicit` (SIREN-style), `wire_implicit` (WIRE-style), `gaussian_basis`, and direct `coadd_bilinear` / `coadd_bicubic` baselines on the same observations. Reported metrics use the same standardized product vocabulary as the stack-truth benchmark, plus `fit_time_ms` and observation-fit summaries. `benchmarks/baselines/map_field_fit_smoke.json` is the lightweight smoke baseline and intentionally skips timing on its tiny coadd helper rows. `benchmarks/baselines/map_field_fit_rep.json` is the larger representative field baseline with active timing checks on the field-fit defaults across `tan_field_fit` and `tan_sip_field_fit_multicolor`.
- `bench_map_compat.py` is the external mapmaking comparison harness. It now emits a concrete variant matrix across torchsky `nearest`/`bilinear`/`bicubic`/`lanczos3`/`lanczos5`/`mitchell_netravali`/`catmull_rom` kernels, `reproject_interp`, `dfreproject`, low-level `drizzle`, `drizzlepac`, `Montage`, `SWarp`, and `casutools imstack`, plus explicit pairwise comparison rows for every successful method pair in a case. `reproject_interp` and `dfreproject` are wired with kernel variants, `drizzle` is wired with multiple low-level kernels, and `SWarp` is wired with multiple resampling modes where the requested target grid is representable. The low-level `drizzle` backend is wired through upstream-style `drizzle.utils.calc_pixmap(...)` and `drizzle.resample.Drizzle.add_image(...)`, and includes an additional CRPIX-translation case patterned after drizzle's own `test_translated_map`. The `Montage` backend is wired through source-built official binaries and now uses `mProjectPP` automatically for the fast plane-to-plane cases it actually supports, while falling back to `mProject` for SIP/distorted inputs. The harness also includes generated MEF cases built from real Astropy SIP data plus a synthetic rotated low-overlap TPV MEF case so extension-addressed `SCI/ERR/DQ` style workflows are exercised in addition to single-HDU inputs. `SWarp` is wired through a source-built CLI path for simple equatorial non-distorted target grids and can therefore participate in the TPV-input/TAN-target synthetic case, while SIP targets are reported as explicit skips instead of being benchmarked against an inexact surrogate grid. `casutools imstack` is wired from a local source build and is only exercised on the documented CASU WCS regime (`TAN`/`ZPN`, no `SIP`/`PV` distortion); because its confidence maps are renormalized confidence images rather than direct inverse-variance weights, its comparison rows use coverage-mask semantics for weight/coverage agreement and common-support semantics for data deltas. Some comparators can still fail on the harder TPV inverse problem; those now surface as explicit `SKIPPED` rows with the upstream error text instead of aborting the matrix. The script reports both global and edge-vs-interior deltas so boundary-convention differences do not get mistaken for full-field reprojection errors, and it now reports those deltas directly between successful method variants. Missing, unsupported, or not-yet-wired comparators show up as explicit `SKIPPED` rows rather than silently disappearing, and import/ABI failures are reported verbatim when available. It also now accepts `--cases`, writes JSON plus a concise Markdown summary table, and can enforce a committed real-fixture compat baseline through `benchmarks/baselines/map_compat_real_smoke.json`. The default `sphere-bench` env stays on the main current Python stack; the dedicated `map-compat` env pins Python `3.13` so lagging external packages can still be evaluated when upstream packaging allows it. The bench env pins PyPI `drizzle==2.2.0` and `dfreproject==1.3.1` explicitly so we do not accidentally benchmark older lineages or conflate `drizzle` with `drizzlepac`.
- `scripts/bootstrap_map_compat_tools.py` clones and builds the official upstream SWarp and Montage sources and can also stage a local-source CASUtools build into `.benchmarks/external-tools/prefix/{swarp,montage,casutools}` from inside the `map-compat` env. `bench_map_compat.py` checks those repo-managed install prefixes before falling back to `PATH`, so source-built external tools show up in the matrix even when Python wheels are unusable on the host architecture.
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
