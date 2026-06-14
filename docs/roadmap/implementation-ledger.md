# Implementation Ledger

> Maintained low-level companion to `status.md`. This file records shipped
> implementation improvements, validation gates, and concrete follow-ups by
> subsystem. Keep `status.md` focused on high-level roadmap state; keep the old
> `../reviews/autonomous_refactor_log.md` as an archival audit trail only.

Last updated: 2026-06-14.

## Latest Local Gate

- `pixi run ci-local`
- Result: `1721 passed, 59 skipped, 63 warnings`.
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
- Header/card ordering, repeated `HISTORY`/`COMMENT`, and CFITSIO-backed
  header-card writeback are active.
- Checksum write/verify helpers are active for files and HDUs.
- Table mutation validates unsupported mmap cases, rename/drop collisions, and
  missing columns before mutation.
- Compressed table writes now emit real BINTABLE extensions.
- `IterableFITSDataset` consumes real JSON or newline manifests from local,
  `file://`, `http://`, or `https://` indexes.

### Validation

- FITS focused gates cover image/table read/write behavior, checksums,
  compression, mutation, cache policy, and dataset/dataloader forwarding.
- Latest relevant targeted gates included:
  - `tests/test_datasets.py`
  - `tests/test_dataloader.py`
  - `tests/test_frame_stypes.py::test_write_tensor_frame_stypes`

### Follow-Ups

- Broaden fixture replay against `astropy.io.fits` and Python `fitsio`.
- Add more realistic VLA/string/complex table mutation fixtures.
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
  wavenumber, energy, redshift, velocity conventions, selected `-LOG`,
  reciprocal, grism, and velocity compound algorithms.
- Temporal WCS covers common units, scale conversion, leap-second-aware UTC,
  observation-date metadata, midpoint/duration metadata, and a fast local
  barycentric/heliocentric light-time correction.
- Tabular WCS supports independent 1D `-TAB`, separable multidimensional arrays
  selected by `PVi_3`, and a forward `CoupledTabularLookupWCS` substrate for
  genuinely coupled multidimensional coordinate arrays.
- High-level spectral cube helpers now fail closed for missing `-TAB`
  coordinate tables and preserve non-uniform wavelength grids.

### Validation

- Broad WCS replay uses Astropy/WCSLIB fixtures where comparable.
- Focused gates cover:
  - `tests/test_wcs_api.py`
  - `tests/test_wcs_tabular.py`
  - `tests/test_wcs_spectral.py`
  - `tests/test_wcs_temporal.py`
  - `tests/test_wcs_upstream_astropy.py`
- WCS spectral/temporal benchmark smoke:
  - `pixi run bench-wcs-spectral-temporal`

### Follow-Ups

- Add inverse solving and main-`WCS` integration for coupled multidimensional
  `-TAB` grids.
- Broaden FITS Paper III spectral algorithm-chain replay.
- Add ephemeris-grade Earth-orientation and apparent-place temporal corrections.
- Build reusable frame/mapping graph objects inspired by AST/GWCS while keeping
  execution torch-native.
- Add Kapteyn/PyAST/Cornish legacy replay gates where behavior is comparable.

## Sphere, HEALPix, MOC, And Sparse Maps

### Shipped

- HEALPix ownership moved toward `torchsky.sphere`.
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
- Pairwise angular distances use a stable `atan2(cross, dot)` formulation.

### Validation

- Focused native and ecosystem gates cover:
  - `tests/test_sphere_core.py`
  - `tests/test_sphere_moc.py`
  - `tests/test_sphere_sparse.py`
  - `tests/test_healsparse_upstream.py`
  - ecosystem parity tests where optional packages are available.

### Follow-Ups

- Add more high-NSIDE query and large sparse-map benchmarks.
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

### Follow-Ups

- Persist richer residual products and scorecards for representative real-data
  runs.
- Add PSF residual alignment reports, per-band leakage diagnostics, and
  support-aware acceptance criteria.
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

