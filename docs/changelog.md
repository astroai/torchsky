# Changelog

All notable changes to torchsky will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Historical benchmark tables and comparative performance claims have moved to
[`torchsky-harness`](https://github.com/astroai/torchsky-harness/blob/main/docs/archive/benchmark-history.md).
They are not current package status. FITS/table I/O performance evidence is
owned by `torchfits`.

## [0.4.0] - 2026-10-18

### Architecture

- **FITS I/O ownership moved to `torchfits`**: FITS images, HDUs, headers,
  compression, checksums, file caches, tables, Arrow scans, and I/O benchmarks
  are no longer torchsky-owned surfaces.
- Torchsky's active scope is WCS, sphere/HEALPix/MOC, catalog geometry,
  observation operators, mapmaking, scenes, and field models.
- **Breaking:** the root I/O aliases and the `torchsky.io`, `torchsky.table`,
  `torchsky.hdu`, `torchsky.cache`, `torchsky.datasets`, `torchsky.dataloader`,
  `torchsky.header_parser`, and `torchsky.core` compatibility modules were
  removed. Use the public `torchfits` API at the FITS/table boundary.
- `torchfits` is the required FITS/table runtime dependency; dataset classes are
  imported from `torchfits.data`.

### Added
- **py.typed**: PEP 561 type-checker marker for downstream type inference.
- Flux-conserving spectral rebinning and wavelength-aware relativistic Doppler
  shifts with batched tensor, mask, and inverse-variance propagation.
- **Experimental native response/likelihood**: native source-node response,
  Gaussian likelihood evidence, small linear queries, evidence lifecycle guards,
  continuous source/spectral quadrature, detector moments, pixel calibration,
  profiled/calibrated nuisance precision, and the bounded Poisson–Gaussian law.
  These remain experimental and do not establish calibrated real-sky response or
  SOTA. See `docs/native-likelihood.md`.

- **Non-uniform spectral axes in `coadd_cube`**: 3D IFU cube assembly now maps
  each detector sample to the target axis with a per-interval fractional index
  instead of a single uniform `d_lambda`, so irregular axes no longer misplace
  samples into the wrong bins. Regular axes keep the previous behaviour.
  Irregular axes remain an index-space interpolation with a recorded bias
  envelope; see `docs/map.md` and `tests/test_map_cube_spectral_axis.py`.

### Changed
- **Sphere spectral operations split**: power-spectrum, pixel-window, beam,
  synthesis, and smoothing operations moved from `sphere/spectral.py` to
  `sphere/spectral_ops.py`; the SHT primitives are imported lazily so the
  module graph stays acyclic, and `spectral` re-exports the historical names.
  spectral.py drops from 4,636 to 4,031 lines.
- **Power Foam primitives split**: the 4,071-line `map/field/foam.py` now
  re-exports role modules: `foam_geometry` (distance/ownership/candidate graphs),
  `foam_solvers` (basis rendering and least-squares solves), `foam_operators`
  (bounded pixel operators), and `foam_field` (`SpectralPowerFoam` and
  adaptation). `torchsky.map.field.foam` imports are unchanged.
- **Field model classes split**: the 4,666-line `map/field/models.py` now keeps
  only re-exports; the model zoo lives in `models_foam`, `models_component`,
  `models_implicit`, `models_encoder`, and `models_skylambda` by representation
  family. The historical `torchsky.map.field.models` and `torchsky.map.field`
  imports are unchanged.
- **Map calibration helpers split**: observation calibration, variance
  accumulation, provenance-context, and area-scaling helpers moved from
  `map/image_plane.py` to `map/calibration.py`; image_plane re-exports them and
  drops from 4,180 to 3,987 lines.
- **Field model classes split**: the 27 neural field-model classes moved from
  `map/field/core.py` to `map/field/models.py`; `core` now holds only the model
  factory/dispatch and public re-exports (502 lines, down from 5,651).
  Imports of every historical class from `torchsky.map.field.core` keep
  working, and `_bandpass_bounds` stays re-exported for the spectral budget
  module.
- **Map target planning split**: the union-footprint planners and public
  `plan`/`plan_cube` family moved from `map/image_plane.py` to
  `map/planning.py`; spectral-plan helpers come from `map/spectral_projection.py`
  and image-plane metadata helpers are imported lazily. `image_plane`
  re-exports the historical names and `product.py` follows the canonical
  module. image_plane.py drops from 4,869 to 4,180 lines.
- **Field helper layer split**: the shared helpers and `_BaseFieldModel` moved
  out of the 5.6k-line `map/field/core.py` into `map/field/_core_helpers.py`
  (feature/atlas helpers, domain estimation, strainer checkpoint helpers);
  `core` re-exports every moved name so imports are unchanged. core.py drops
  from 5,651 to 5,141 lines.
- **Map sky matching split**: `detect_outliers`, `match_sky`, their private
  helpers, and the sky-matching tile resolver moved from the 5.3k-line
  `map/image_plane.py` to `map/sky_matching.py`; the image-plane module now
  re-exports the public names, so imports and the `torchsky.map` facade are
  unchanged. `image_plane.py` drops by ~450 lines.
- **WCS core split**: the 5.2k-line `wcs/core.py` module now keeps the `WCS`
  state machine while rotations/TAN projection live in `wcs/_rotation.py`,
  footprint/geometry/serialization in `wcs/_geometry.py`, and the shared
  wrap/trig helpers in `wcs/_math.py`. Public API and behaviour are unchanged.
- **AIR projection solver**: pixel->world for the Airy zenithal projection now
  uses a fixed-iteration Newton solve in `t = tan(xi)` instead of a bracketed
  regula-falsi loop with per-iteration host synchronization. On the harness WCS
  grid (20k points, CPU) it is 2.3x astropy instead of 0.53x, and the maximum
  angular parity against astropy improves from 4.1e-5 to 4.1e-7 arcsec; the
  same accuracy holds for PV2_1 in 45..120 deg.
- **Lean-code pass (native, WCS, sphere, map)**: removed the unused NumPy
  HEALPix bindings and dead native kernels (68 → 56 exported symbols), the
  pre-`torchfits` CFITSIO vendor tree and CI hooks, the merged
  `torchsky.cpp` compatibility shim, unused logging and CMake options, dead
  Astropy-mimic WCS methods, the unused `torch.func` SIP/TPV distortion path,
  and duplicated field/atlas/Fourier/diagnostic helpers. `torchsky.map.field`
  is now a lazy facade (1 field module imported instead of 28). Public API
  names were not changed.

### Removed
- **Legacy CFITSIO vendoring**: `extern/vendor.sh`, its version pin, the
  generated `extern/cfitsio` tree, the CI vendor steps, and the manifest entry
  are gone. The native extension builds only WCS/HEALPix/geometry kernels from
  `src/torchsky/cpp_src`; FITS I/O and CFITSIO belong to `torchfits`.

### Documentation and developer experience
- **Published documentation site**: Zensical + Material build from `docs/` only
  (unpublished plans, research, reviews, and archives stay outside git;
  `notes/planned.md` is the single tracked plan), with a home
  page, quick start, navigation by domain, API reference pages for every
  namespace, a strict build, an offline link checker, and a Pages workflow.
- **Documentation contracts**: every public `__all__` name must appear in the
  docs, every Python block's imports must resolve, and the published tree/nav/
  assets/links are tested (`tests/test_docs_integrity.py`,
  `tests/test_docs_snippets.py`, `tests/test_docs_coverage.py`).
- **Examples are gated and published**: `examples/test_examples.py` runs every
  standalone example as a bounded subprocess, `pixi run examples` is a CI gate,
  and the scripts are copied into the built site so their links work.

### Fixed
- **Native extension loader on PyPI torch wheels**: the compiled module was
  named `_cpp`, so `import torchsky._cpp` dlopened libtorch before torch had
  loaded it; conda environments hid this, PyPI torch wheels do not. The shared
  object is now `_cpp_ext` behind a tiny `torchsky/_cpp.py` loader that
  preloads torch and re-exports every native name (public import path
  unchanged). Found by the CANFAR headless GPU subject-install smoke.
- **FITS `-TAB` default indexing vector**: the fallback table index was
  `0..N-1`; the FITS/table-WCS convention is `1..N`, so every tabulated
  coordinate (e.g. a real `TIME-TAB` axis) was shifted by one index and the
  last sample fell off the table. Astropy/WCSLIB parity is now exact on a real
  HST `-TAB` product, and the harness gates it against astropy
  (`bench_wcs_real_headers`).
- **`-TAB` tables from `torchfits.HDUList`**: table lookup required a
  name-indexed mapping or an iterable of HDUs, and the public FITS boundary
  (`torchfits.HDUList`) exposes integer indexing and `__len__` but not
  `__iter__`. The resolver now falls back to positional lookup, so real
  `-TAB` FITS products load through the documented boundary.
- **Viz routes importing a deleted helper**: the blocking helper
  `viz/_blocking.py` had been removed by a "no importers" cleanup while sixteen
  lazy imports in the fits, hips, moc, validate, and catalog routers still
  referenced it, so those endpoints failed on first request. Restored it on
  Starlette's thread pool, installed the viz test dependencies so the server
  suite actually runs, and added a static lazy-import-target test
  (`tests/test_lazy_import_targets.py`) that resolves every function-local
  relative import against the package tree.
- **Center-native WCS inverse rotation**: `world_to_pixel` for cylindrical,
  all-sky, pseudocylindrical, and polyconic projections reused the pole-native
  rotation basis, so any header with a non-zero reference latitude
  (`CRVAL2 != 0`) inverted the native frame incorrectly. The error was invisible
  at `CRVAL2 = 0` (and in all prior comparison cases, which used the equator)
  and reached hundreds of pixels for `MER`, `CYP`, `AIT`, `HPX`, `SFL`, `BON`,
  and `PAR` at `CRVAL2 = 20 deg`. The inverse now uses the same east/north/radial
  basis as the forward transform; `tests/test_wcs_projection_matrix.py` covers
  all 28 projection codes at both reference latitudes, and the harness gained
  reference-offset cases (`MER_OFF`, `CYP_OFF`, `AIT_OFF`, `HPX_OFF`, `SFL_OFF`,
  `XPH_OFF`, `BON_OFF`, `PAR_OFF`).

### Changed
- **Leaner spectral namespace**: `torchsky.spectral` no longer re-exports the
  26 redshift/velocity helpers that `torchsky.redshift` (and
  `torchsky.log_lambda`) own. The spectral namespace now declares only its data
  containers, axis helpers, resampling, and its own Doppler shift; every
  namespace has an explicit `__all__`, checked by
  `tests/test_public_namespaces.py`.
- **Breaking: experimental field/native module paths renamed** for clarity:
  `map/field/native_physical.py` → `native_integrated_read.py`
  (`NativeIntegratedReadLikelihood`), `joint_native.py` →
  `joint_native_likelihood.py`, `neural_inverse.py` → `gauss_newton.py`
  (`gauss_newton_pcg_step`), and `map/operators/lensing.py` →
  `magnification.py`. The old module paths are gone; public entry points are
  the `torchsky.map.field` facade names.
- **Second lean-code pass (map, field, solvers, viz)**: removed the
  undocumented `learned_optimizer` and `calibration_partitions` prototypes,
  the unused `representative_pixel_area_fn` plumbing through the geometry
  kernels, the no-op `estimate_normal_operator_diagonal(max_entries)` option,
  unused parameters in cube/windowing/scanner helpers, the unused
  `/api/validate/{pid}/dr_report` route, dead static-UI blocks (catalog
  overlay controls, validation stats card, unused footer fields), and the
  abandoned Vite shell. Registered the documented `bacon` frequency-adaptive
  model kind in the factory and documented every reachable expert model kind.
- **API manifest completed**: the frozen public-surface manifest now covers all
  30 namespaces (adding spectral, spectrophotometry, redshift, log_lambda,
  extinction, cosmology, transforms, quality, and viz) instead of a subset.

### Removed
- FITS image/table/checksum implementation tests and benchmarks now owned by
  `torchfits`.
- The root `read`, `write`, `open`, `get_header`, `get_wcs`, `Header`, `Card`,
  `HDUList`, `TensorHDU`, and `TableHDU` aliases. Construct `WCS` from a
  `torchfits.read_header(...)` mapping instead.
- The `torchsky.io`, `torchsky.table`, `torchsky.hdu`, `torchsky.cache`,
  `torchsky.datasets`, `torchsky.dataloader`, and `torchsky.header_parser`
  modules.
- The private WHERE parser fallback; table filtering delegates to the public
  `torchfits.table` API.
- The optional `torchsky.frame` / `pytorch-frame` adapter; Arrow and Polars are
  the supported table interchange surfaces.
- The index-based random `RedshiftShift` image transform; use the physical
  `torchsky.spectral.relativistic_doppler_shift` API.

### Performance
- **sphere/spectral.py**: O(1) `alm_size` via closed-form formula; precision fixes in harmonic primitives (`(ell-1)**2` → explicit multiplication).
- **sphere/sparse.py**: Explicit `ratio * ratio` instead of `** 2` in `pixels_per_cov` and `_build_coverage_map` for precision consistency.
- **wcs/legacy.py**: TNX regex patterns cached via `@functools.lru_cache(maxsize=16)`.
- **wcs/sip.py**: Extracted `_make_pow_cache` and `_make_pow_stack` static methods for SIP power-cache reuse.
- **wcs/tpv.py**: `PV1_KEYS_SET`/`PV2_KEYS_SET` frozensets for O(1) header key lookup; `keys_set.intersection(header.keys())` to avoid 40 individual lookups.
- **wcs/core.py**: `PV1_KEYS_SET`/`PV2_KEYS_SET` for efficient `has_pv` check.
- **cpp_src/hardware.cpp**: macOS uses `sysctlbyname()` instead of `popen("sysctl -n ...")` for L3 cache and memory size detection.
- **datasets.py**: Import hoisting in `_process_shard` (moved `read` and `logger` imports before loop).

### Changed
- **CI**: Added `pyarrow` to build-test dependencies for table tests.
- **pixi.toml**: Added `pyarrow` to `feature.test`; added `healsparse` to `feature.sphere-bench` for HealSparse parity tests.

## [0.3.1] - 2026-03-07

### Changed
- Version synchronization across `pyproject.toml`, `pixi.toml`, and `__init__.py` (was inconsistent in 0.3.1).
- CI workflow cleanup: `ci.yml` now references `pyarrow` instead of `pytorch-frame`.
- Historical comparator environments and exhaustive parity suites were later
  moved to `torchsky-harness`.

### Fixed
- **Spectral Transforms**: Resolved minor parity bugs in `torchsky.sphere.spectral` relating to harmonic primitive precision.

Historical comparative results are archived in the
[harness benchmark history](https://github.com/astroai/torchsky-harness/blob/main/docs/archive/benchmark-history.md).

## [0.3.0] - 2026-03-06

### Added
- **Native PyTorch WCS (`torchsky.wcs`)**: A complete, pure-PyTorch reimplementation of astronomical coordinate transformations.
    - Support for core projections: `TAN`, `SIN`, `ARC`, `ZPN`, `ZEA`, `HPX`, `AIT`, `MOL`, `MER`, `CEA`, `STG`, `CYP`.
    - Support for polynomial distortions: `SIP`, `TPV`, `TNX`, `ZPX`.
    - Batch-accelerated `pixel_to_world` and `world_to_pixel` transformations.
    - Automatic initialization from FITS headers via `torchsky.get_wcs()`.
- **Spherical Geometry & HEALPix (`torchsky.sphere`)**:
    - Core HEALPix primitives: `ang2pix`, `pix2ang`, `nest2ring`, `ring2nest`, `neighbors`, `interp`.
    - **Spherical Polygons**: Support for non-convex region queries, area calculations, and `contains` checks on the unit sphere.
    - **Harmonic Transforms**: Initial support for `map2alm` and `alm2map` (scalar and spin transforms).
    - **Multi-Order Coverage (MOC)**: Basic support for `MOC` maps and filtering.
    - **HealSparse**: Unified API for `HealSparseMap` targets and sparse-to-dense conversions.
- **Spectral Data Support (`torchsky.spectral`)**:
    - `Spectrum1D`: Specialized container for 1D spectra with flux, uncertainty (ivar/error), and mask handling.
    - `DataCube`: Support for IFU (Integral Field Unit) cubes with spectral axis and spatial wcs integration.
    - `SpectralAxis`: Unit-aware spectral coordinate management (wavelength, frequency, velocity).
- **ML Loader Enhancements**: Improved `FITSDataset` and `create_fits_dataloader` with better payload auto-detection (`hdu="auto"`) and more robust multi-worker handling.

### Changed
- **WCS Resolution**: `torchsky.get_wcs` now consistently resolves to the native PyTorch implementation, removing runtime dependency on external C++ libs for coordinates.
- **Build System**: Removed all `WCSLIB` detection and linking logic. The C++ engine is now leaner, focusing exclusively on FITS I/O (vendored `cfitsio`).

### Removed
- **External Dependency (WCSLIB)**: Completely removed the C++ `wcslib` dependency, significantly reducing binary size and making wheels fully self-contained.
- **Legacy Components**: Removed the obsolete monolithic `wcs.py` and the redundant `torchsky.fits` namespace.
- **Developmental Artifacts**: Cleaned up transient test artifacts and local tool metadata.

Historical comparative results are archived in the
[harness benchmark history](https://github.com/astroai/torchsky-harness/blob/main/docs/archive/benchmark-history.md).

## [0.2.1] - 2026-02-14

### Fixed
- Routed `int32` image reads through the CFITSIO fast path, fixing a regression.
- Implemented 128 MB chunking for large file reads to prevent memory spikes during type conversion.

Table I/O performance evidence is now owned by `torchfits`.

## [0.2.0] - 2026-02-14

### Added
- Arrow-native table API in `torchsky.table`: `read()`, `scan()`, `reader()`, `dataset()`, `scanner()`.
- In-place FITS table mutation APIs: `append_rows()`, `insert_rows()`, `delete_rows()`, `update_rows()`, `insert_column()`, `replace_column()`, `rename_columns()`, `drop_columns()`.
- `TableHDURef` file-backed mutation helpers and lazy-handle workflow.
- HTML representation for `HDUList` in notebook environments.
- `hdu="auto"` / `hdu=None` payload detection in `read()` and `get_header()`.
- `get_wcs()` convenience API for file-based WCS initialization.
- ML-loader evaluation moved to `torchsky-harness`.

### Changed
- Minimum supported Python version raised to 3.11 (later raised to 3.12 in 0.4.0.dev0).
- Build system updated for vendored CFITSIO. AVX2/SSSE3 optimizations in C++ hot paths.

### Fixed
- FITS filename handling hardening and header/column-name sanitization on write paths.
- WCS/TAN GPU approximation correctness fixes.
- CI/build-system reliability fixes (build isolation, formatter compliance, dependency setup).
- Multiple interoperability and non-table regression fixes with test coverage updates.

### Performance Notes
- Historical benchmark methodology and result snapshots are archived in the
  [harness benchmark history](https://github.com/astroai/torchsky-harness/blob/main/docs/archive/benchmark-history.md).
- FITS/table I/O and data-loading performance comparisons are maintained by
  `torchfits` and `torchsky-harness`, respectively.

## [0.1.1] - 2025-12-05

Patch release. No user-visible changes.

## [0.1.0] - 2025-12-05

### Added
- FITS image and table reading with zero-copy tensor creation on CPU, CUDA, and MPS.
- Column selection, row slicing, and cutout/subset reading.
- FITS file writing for images and tables.
- Multi-HDU support with context management.
- Batch WCS coordinate transformations via wcslib (later replaced by native PyTorch in 0.3.0).
- `FITSDataset` and `IterableFITSDataset` for [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html) integration.
- GPU-accelerated astronomical transforms: ZScale, AsinhStretch, LogStretch, PowerStretch, Normalize, MinMaxScale, RobustScale.
- Data augmentation transforms: RandomCrop, CenterCrop, RandomFlip, RandomRotation, GaussianNoise, PoissonNoise.
- Spectroscopy transforms: RedshiftShift, PerturbByError.
- C++ engine with [nanobind](https://nanobind.readthedocs.io/) bindings, SIMD-optimized type conversion, multi-level caching.
- [PyArrow](https://arrow.apache.org/docs/python/) and [Pandas](https://pandas.pydata.org/) conversion utilities.
- Remote file support (HTTP/HTTPS/FTP via CFITSIO).

### Known Limitations
- Variable-length array (VLA) table columns have limited support.
- Compressed image writing supports Rice compression only.
- MPS (Apple Silicon) shows overhead for small workloads due to transfer latency.

[0.1.0]: https://github.com/astroai/torchsky/releases/tag/v0.1.0
[0.1.1]: https://github.com/astroai/torchsky/releases/tag/v0.1.1
[0.2.0]: https://github.com/astroai/torchsky/releases/tag/v0.2.0
[0.2.1]: https://github.com/astroai/torchsky/releases/tag/v0.2.1
[0.3.0]: https://github.com/astroai/torchsky/releases/tag/v0.3.0
[0.3.1]: https://github.com/astroai/torchsky/releases/tag/v0.3.1
[0.4.0]: https://github.com/astroai/torchsky/releases/tag/v0.4.0
