# Implementation Ledger

> Maintained low-level companion to `status.md`. This file records shipped
> implementation improvements, validation gates, and concrete follow-ups by
> subsystem. Keep `status.md` focused on high-level roadmap state; keep the old
> `../reviews/autonomous_refactor_log.md` as an archival audit trail only.

Last updated: 2026-06-18.

## Latest Local Gate

- `pixi run ci-local`
- Result: `1775 passed, 59 skipped, 63 warnings`.
- Notes: pytest exits cleanly. Existing nanobind shutdown leak diagnostics for
  `FITSFile` still print during interpreter teardown.

## Cross-Cutting API And Architecture

### Shipped

- Root `import torchsky` is a lean lazy facade.
- Public API surfaces are grouped by stability tier and architecture pillar in
  `torchsky.surfaces`.
- Specialized functionality is routed to owning namespaces:
  - `torchsky.io`
  - `torchsky.table`
  - `torchsky.wcs`
  - `torchsky.sphere`
  - `torchsky.map`
  - `torchsky.scene`
  - `torchsky.field`
  - `torchsky.viz`
- Internal call sites that did not need package-root lazy dispatch now import
  concrete domain modules directly.

### Validation

- Cold-import tests protect the lean root facade.
- API-surface tests protect tier and pillar metadata.
- `pixi run ci-local` is the standing pre-push gate.

### Follow-Ups

- Keep compatibility aliases lazy and documented, but avoid adding new root
  aliases for experimental or niche helpers.
- Keep domain docs aligned with namespace ownership.

## FITS, HDU, Table, And I/O

### Shipped

- FITS read/write implementation was moved behind `torchsky.io` and private
  `_io_engine` modules while preserving root public wrappers.
- FITS file operations now have an owning `torchsky.io.fits` namespace, with
  `torchsky.io` and root wrappers kept as compatibility/convenience aliases.
- Header/card ordering, repeated `HISTORY`/`COMMENT`, and CFITSIO-backed
  header-card writeback are active.
- Checksum write/verify helpers are active for files and HDUs.
- Table mutation validates unsupported mmap cases, rename/drop collisions, and
  missing columns before mutation.
- Compressed table writes now emit real BINTABLE extensions.
- Generated FITS replay coverage now exercises MEF image HDUs, compressed
  images, fixed-width string/boolean/numeric BINTABLEs, and Astropy VLA
  fixtures through `torchsky.io.fits`.
- Deterministic radio/VLA-style table replay now covers time, antenna pairs,
  UVW vector columns, complex visibility columns, weights, flags, metadata, and
  mutation-safe row updates.
- `bench_fitstable_io.py` now emits table mutation rows for append, update, and
  rename/drop operations in addition to read/projection/slice/filter/scan rows.
- Named FITS I/O cache policy is centralized through `IO_CACHE_SUBSYSTEMS`,
  `cache_subsystem_policy(...)`, and `clear_cache_subsystem(...)` for image
  data, table data, header metadata, WCS extraction, and all-cache clearing.
- FITS table backend selection policy has a private `_table_engine` home while
  scan/read/Arrow/mutation public behavior remains in `torchsky.table`.
- `IterableFITSDataset` consumes real JSON or newline manifests from local,
  `file://`, `http://`, or `https://` indexes.

### Validation

- FITS focused gates cover image/table read/write behavior, checksums,
  compression, mutation, cache policy, and dataset/dataloader forwarding.
- Latest relevant targeted gates included:
  - `tests/test_fits_fixture_replay_matrix.py`
  - `tests/test_io_engine_caches.py`
  - `tests/test_io.py`
  - `tests/test_table.py`
  - `tests/test_datasets.py`
  - `tests/test_dataloader.py`
  - `tests/test_frame_stypes.py::test_write_tensor_frame_stypes`
  - `benchmarks/bench_fitstable_io.py --quick --max-cases 1 --warmup 0`

### Follow-Ups

- Add larger committed benchmark baselines when representative data policy is
  settled.
- Keep Python `fitsio` out of runtime library code.

## WCS And Coordinate Systems

### Shipped

- Active celestial projection set includes common TAN/SIN/ZPN/ZEA/STG/CEA/CAR/
  MER/CYP/AIT/MOL/HPX/SFL paths plus tested distortion families.
- Alternate WCS keys, metadata surfaces, array-order helpers, slicing, uncoupled
  axis dropping/reordering, celestial extraction, and mixed pixel/world solving
  are active.
- Differentiable pixel/world transforms and local Jacobians support optimization
  and field-model use.
- Spectral WCS translations support frequency, vacuum/air wavelength,
  wavenumber, energy, redshift, velocity conventions, selected `-LOG`, the
  supported one-axis F/W/A/V compound matrix, and grism algorithms.
- Temporal WCS covers common units, scale conversion, leap-second-aware UTC,
  observation-date metadata, midpoint/duration metadata, and a fast local
  barycentric/heliocentric light-time correction, including explicit GEOCENTER
  no-ephemeris behavior, normalized reference-position aliases, and geodetic
  observatory metadata replay.
- Tabular WCS supports independent 1D `-TAB`, separable multidimensional arrays
  selected by `PVi_3`, and a forward `CoupledTabularLookupWCS` substrate for
  genuinely coupled multidimensional coordinate arrays.
- Main `WCS(header, tables=...)` now auto-loads genuinely coupled FITS `-TAB`
  coordinate arrays into forward evaluation, `TensorHDU.wcs`,
  `torchsky.get_wcs(path, hdu=...)`, differentiable Jacobians, and local
  square-grid inverse roundtrips for both all-axis non-spatial lookup WCS and
  mixed spatial WCS with coupled extra axes. The 2D square-grid inverse path
  enumerates candidate cells and fails closed when multiple distinct inverse
  solutions are valid.
- High-level spectral cube helpers now fail closed for missing `-TAB`
  coordinate tables and preserve non-uniform wavelength grids.

### Validation

- Broad WCS replay uses Astropy/WCSLIB fixtures where comparable.
- Latest focused WCS gate: `134 passed, 8 warnings`.
- Focused gates cover:
  - `tests/test_wcs_api.py`
  - `tests/test_wcs_tabular.py`
  - `tests/test_wcs_spectral.py`
  - `tests/test_wcs_tabular.py`
  - `tests/test_wcs_temporal.py`
  - `tests/test_wcs_upstream_astropy.py`
- WCS spectral/temporal benchmark smoke:
  - `pixi run bench-wcs-spectral-temporal`
  - The benchmark covers the supported F/W/A/V compound spectral matrix and
    uses bounded log-axis pixel sampling for WCSLIB-compatible timing rows.
- WCS tabular benchmark smoke:
  - `benchmarks/bench_wcs_tabular.py --grid-size 8 --n-points 32 --runs 1`
  - The benchmark covers direct `CoupledTabularLookupWCS` and main
    `WCS(header, tables=...)` forward plus ambiguity-safe inverse timing.

### Follow-Ups

- Add higher-dimensional ambiguity handling and broader upstream-style fixture
  replay for coupled multidimensional `-TAB` grids.
- Broaden FITS Paper III spectral algorithm-chain replay.
- Add ephemeris-grade Earth-orientation and apparent-place temporal corrections.
- Build reusable frame/mapping graph objects inspired by AST/GWCS while keeping
  execution torch-native.
- Add Kapteyn/PyAST/Cornish legacy replay gates where behavior is comparable.

## Sphere, HEALPix, MOC, And Sparse Maps

### Shipped

- HEALPix source ownership lives in `torchsky.sphere.healpix`; public discovery
  routes through `torchsky.sphere` and `torchsky.sphere.compat` /
  `torchsky.sphere.healpy`.
- MOC construction now supports general polygon pixelization, region inputs,
  pixelized-region roundtrips, sparse-map conversion, compact ASCII ranges, and
  direct pixel containment.
- Sparse maps enforce sorted-unique pixels with last-value-wins duplicate
  semantics.
- Sparse FITS roundtrips preserve `VALUE` dtype instead of coercing to
  `float32`.
- Sparse coverage hierarchy is built through the NESTED parent relation and is
  preserved through construction, updates, FITS reads, and valid `ud_grade(...)`
  operations.
- High-NSIDE query correctness is gated at `nside=1024` for HEALPix center
  roundtrips, neighbors, interpolation weights, sparse lookup, MOC pixel
  containment, and MOC catalog filtering without dense full-sky materialization.
- Sparse maps now expose direct coverage bridges to MOC and `PixelizedRegion`,
  plus `SparseHealpixMap.from_region(...)` for exact/native/boolean regions and
  pixelized regions without dense full-sky materialization.
- `SkyMaskPipe.combine(...)` now performs AND/OR mask algebra through sparse
  pixel sets instead of materializing dense full-sky maps.
- `torchsky.sphere.harmonics` is now the preferred harmonic-transform namespace;
  `torchsky.sphere.spectral` remains a backward-compatible import path.
- Pairwise angular distances use a stable `atan2(cross, dot)` formulation.

### Validation

- Focused native and ecosystem gates cover:
  - `tests/test_sphere_core.py`
  - `tests/test_sphere_moc.py`
  - `tests/test_sphere_sparse.py`
  - `tests/test_sphere_high_nside.py`
  - `tests/test_healsparse_upstream.py`
  - ecosystem parity tests where optional packages are available.
- Latest focused sparse/MOC/high-NSIDE gate:
  - `61 passed, 2 warnings`.
- Latest focused sparse/high-NSIDE gate:
  - `25 passed, 2 warnings`.
- Latest focused sphere gate:
  - `344 passed, 8 skipped, 10 warnings`.

### Follow-Ups

- Add larger sparse-map benchmarks and committed high-NSIDE benchmark baselines.
- Continue exact geometry work for region/MOC edges and boolean operations.
- Expand spin/SHT replacement gates.

## Map, Scene, Operators, And Inverse Problems

### Shipped

- `ObservationOperatorAdapter` can bind executable targets and route them through
  solver hooks and normal-equation probes.
- Image-plane target binding is implemented in `torchsky.map.operators.inverse`
  with exact VJP adjoints of the same projection used for prediction.
- Multi-observation image-plane bindings assemble summed normal equations over a
  shared latent target grid.
- Inverse-noise hooks handle scalar weights, weight maps, masks, and primary
  beams with shape checks and nonnegative clamping.
- Promoted scene facades delegate to the shared map-operator implementation
  instead of owning solver policy.

### Validation

- Focused gates cover:
  - `tests/test_scene_facade.py`
  - `tests/test_map_operator_bundle.py`
  - `tests/test_map_adjoints.py`
  - map compatibility and image-plane smoke tests.

### Follow-Ups

- Make `ObservationOperatorBundle` the common execution substrate across map
  products, field fitting, and hybrid scene models.
- Move more planning/windowing/I/O orchestration out of `map/image_plane.py`.
- Add exact adjoint gates for every promoted kernel before adding solver policy.

## INR, Fields, Diagnostics, And Adaptive Capacity

### Shipped

- Field-model public exposure was rationalized through presets and factories.
- Field fitting can refine anchored per-observation exposure poses.
- Field diagnostics cover residual products, weighted RMSE/MAE, residual SNR,
  prediction-vs-target sharpness, multiscale coherent residual RMSE, support
  fraction, leakage metrics, and timing/memory summaries.
- `FieldFitResult` retains the merged `ObservationOperatorBundle` used by the
  observation set and records enabled operator slots in fit diagnostics, so field
  fits expose the same observation-law substrate used by promoted map/scene paths.
- Field renderers accept an explicit per-observation `ObservationOperatorBundle`;
  `fit_field_model(...)` caches those bundles and passes them through training,
  final scoring, and diagnostics instead of repeatedly deriving operator slots.
- Fit results retain the per-observation operator-bundle sequence, and residual /
  quality helpers accept explicit bundles so post-fit quality products can reuse
  the same observation-law metadata as training.
- Residual products and quality reports now surface `psf_residual_alignment`
  for PSF-enabled observations, and aggregate scorecards account for optional
  per-observation metrics across mixed observation sets.
- Added `field_quality_acceptance_report(...)` for explicit support-aware quality
  gates over weighted RMSE, residual SNR, support fraction, effective weight, and
  optional PSF residual alignment.
- Added `render_field_fit_quality(...)` as a convenience wrapper around
  `render_field_quality(...)` that consumes `FieldFitResult` directly and uses
  stored exposure poses and per-observation operator bundles by default.
- `ContinuousFieldModel.render_to_target(...)` and
  `render_distribution_to_target(...)` forward explicit operator bundles into
  field rendering, so scene-level field rendering shares the same observation-law
  metadata path as fit and quality helpers.
- `scene.operator_bundles_for_observations(...)` builds ordered per-observation
  operator-bundle sequences for fitting, quality diagnostics, and implicit
  forward-model workflows.
- `predict_observation_mean(...)` and `SurveyDataTerm` accept per-observation
  operator bundles, and `multi_survey_charbonnier_loss(...)` forwards them into
  rendering for implicit forward-model workflows.
- `ObservationOperatorBundle` now owns ordered slot iteration and enabled-slot
  summary generation for shared diagnostics across map, scene, and field code.
- Field distribution fallback rendering no longer applies PSF/background
  operators twice when the model only has scalar forward/render methods.
- `torchsky.map.operators` lazily resolves inverse image-plane binder exports so
  solver imports can depend on `operators.bundle` without triggering an
  operators/inverse/solvers import cycle.
- `field_quality_scorecard(...)` reports by-band aggregates and exact extrema.
- Support-aware residual tile selection ranks local regions by weighted residual
  energy.
- `ResidualTileBasisField` adds bounded local correction atoms on top of a
  diffuse field from selected residual evidence.

### Validation

- Focused gates cover:
  - `tests/test_field_diagnostics.py`
  - `tests/test_field_fit_bundle.py`
  - `tests/test_map_field.py`
  - `tests/test_map_field_fit.py`
- Latest focused operator-substrate gate:
  - `pixi run -e test -- pytest tests/test_field_fit_bundle.py tests/test_map_operator_bundle.py -q`
  - `pixi run -e test -- pytest tests/test_map_field.py::test_render_field_observation_accepts_explicit_operator_bundle tests/test_map_field.py::test_render_field_observation_distribution_fallback_applies_operators_once tests/test_field_fit_bundle.py tests/test_map_operator_bundle.py::test_image_plane_target_operator_binder_exposes_exact_adjoint -q`
- Current local gate:
  - `pixi run -e test -- pytest tests/test_field_diagnostics.py tests/test_field_fit_bundle.py tests/test_map_field.py tests/test_map_scene_representations.py tests/test_scene_facade.py tests/test_implicit_forward.py tests/test_map_operator_bundle.py tests/test_docs_integrity.py -q`
  - `pixi run preflight-push`
  - `pixi run typecheck`
  - `pixi run -e test -- pytest tests/test_api.py::test_top_level_namespaces_expose_phase1_modules tests/test_api_surfaces.py tests/test_cold_import.py -q`

### Follow-Ups

- Persist richer residual products and scorecards for representative real-data
  runs.
- Add per-band leakage diagnostics.
- Keep adaptive local-capacity layers explicit and experimental until memory,
  support, and real-data gates are stronger.

## Spectral Cubes, Transforms, And ML Helpers

### Shipped

- Non-uniform spectral cube write/read supports custom-card and standard `-TAB`
  coordinate-table modes.
- `SpectralAxis` frequency-to-wavelength conversion now honors Hz/MHz/GHz units
  and exact speed-of-light constants.
- `Spectrum1D.resample(...)` performs explicit linear interpolation and
  inverse-variance propagation instead of nearest-neighbor overwrite.
- Display/ML transforms now validate parameters, use real `AsinhStretch`
  softening, avoid cached CPU tensor constants, handle explicit scalar
  normalization values on the input device/dtype, and use iterative zscale
  limits.

### Validation

- Focused gates cover:
  - `tests/test_spectral.py`
  - `tests/test_spectral_nonuniform.py`
  - `tests/test_transforms.py`
  - `tests/test_viz_plots.py`

### Follow-Ups

- Add broader spectral cube fixtures with realistic wavelength tables.
- Keep transform behavior documented as preprocessing/display helpers, not
  calibrated photometric operations.
