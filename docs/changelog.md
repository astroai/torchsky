# Changelog

All notable changes to torchsky will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0.dev0] - Unreleased

### Architecture

- **FITS I/O ownership moved to `torchfits`**: FITS images, HDUs, headers,
  compression, checksums, file caches, tables, Arrow scans, and I/O benchmarks
  are no longer torchsky-owned surfaces.
- Torchsky's active scope is WCS, sphere/HEALPix/MOC, catalog geometry,
  observation operators, mapmaking, scenes, and field models.
- Residual root I/O and `torchsky.io` / `torchsky.table` surfaces are migration
  debt pending removal or thin deprecation-only delegation.
- The supported integration baseline is `torchfits 0.8.0` or newer;
  dataset classes are imported from `torchfits.data`.

### Added
- **py.typed**: PEP 561 type-checker marker for downstream type inference.

### Removed
- FITS image/table/checksum implementation tests and benchmarks now owned by
  `torchfits`.
- The private WHERE parser fallback; table filtering delegates to the public
  `torchfits.table` API.

### Performance
- **sphere/spectral.py**: O(1) `alm_size` via closed-form formula; precision fixes in harmonic primitives (`(ell-1)**2` → explicit multiplication).
- **sphere/sparse.py**: Explicit `ratio * ratio` instead of `** 2` in `pixels_per_cov` and `_build_coverage_map` for precision consistency.
- **header_parser.py**: Fast path for `/` separator, `card.isspace() or not card`, optimized `_parse_string_value` with `find()`.
- **wcs/legacy.py**: TNX regex patterns cached via `@functools.lru_cache(maxsize=16)`.
- **wcs/sip.py**: Extracted `_make_pow_cache` and `_make_pow_stack` static methods for SIP power-cache reuse.
- **wcs/tpv.py**: `PV1_KEYS_SET`/`PV2_KEYS_SET` frozensets for O(1) header key lookup; `keys_set.intersection(header.keys())` to avoid 40 individual lookups.
- **wcs/core.py**: `PV1_KEYS_SET`/`PV2_KEYS_SET` for efficient `has_pv` check.
- **wcs/utils.py**: In-place RA wrap operations and `torch.hypot` for Newton-Raphson residuals.
- **cpp_src/hardware.cpp**: macOS uses `sysctlbyname()` instead of `popen("sysctl -n ...")` for L3 cache and memory size detection.
- **datasets.py**: Import hoisting in `_process_shard` (moved `read` and `logger` imports before loop).

### Changed
- **CI**: Added `pyarrow` to build-test dependencies for table tests.
- **pixi.toml**: Added `pyarrow` to `feature.test`; added `healsparse` to `feature.sphere-bench` for HealSparse parity tests.

## [0.3.1] - 2026-03-07

### Changed
- Version synchronization across `pyproject.toml`, `pixi.toml`, and `__init__.py` (was inconsistent in 0.3.1).
- Benchmark snapshot updated to run `20260318_083620` with improved results across all domains.
- CI workflow cleanup: `ci.yml` now references `pyarrow` instead of `pytorch-frame`.
- Default pixi environment now includes comparison libraries (`healpy`, `hpgeom`, `healsparse`, `astropy-healpix`, `spherical-geometry`) so parity tests run by default; skipped tests reduced from ~142 to ~16.

### Fixed
- **Spectral Transforms**: Resolved minor parity bugs in `torchsky.sphere.spectral` relating to harmonic primitive precision.

### Performance
- Re-baselined the exhaustive benchmark suite. `torchsky` maintains its 100% win rate against `fitsio` and improves aggregate WCS leadership (80.7% win rate vs AST/Astropy).

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
- **Standardized Benchmark Output**: Default benchmark results now consistently target the `benchmarks_results/` directory.

### Removed
- **External Dependency (WCSLIB)**: Completely removed the C++ `wcslib` dependency, significantly reducing binary size and making wheels fully self-contained.
- **Legacy Components**: Removed the obsolete monolithic `wcs.py` and the redundant `torchsky.fits` namespace.
- **Developmental Artifacts**: Cleaned up transient test artifacts and local tool metadata.

### Performance
- **FITS I/O**: Maintained 100% win rate against `fitsio` across 249 image-domain and 140 table-domain cases (typically 10x-100x faster than `astropy` for large-N payloads).
- **WCS Thruput**: Achieved 1.1x-1.3x leads in high-volume (N > 100k) forward transforms.
- **Sphere Geometry**: Achieved ~89% win rate across the HEALPix ecosystem; identified optimization targets for spin transforms in 0.3.1.

## [0.2.1] - 2026-02-14

### Fixed
- Routed `int32` image reads through the CFITSIO fast path, fixing a regression.
- Implemented 128 MB chunking for large file reads to prevent memory spikes during type conversion.

### Performance
- Table reads via memory mapping show substantial speedups for large catalogs.

## [0.2.0] - 2026-02-14

### Added
- Arrow-native table API in `torchsky.table`: `read()`, `scan()`, `reader()`, `dataset()`, `scanner()`.
- In-place FITS table mutation APIs: `append_rows()`, `insert_rows()`, `delete_rows()`, `update_rows()`, `insert_column()`, `replace_column()`, `rename_columns()`, `drop_columns()`.
- `TableHDURef` file-backed mutation helpers and lazy-handle workflow.
- HTML representation for `HDUList` in notebook environments.
- `hdu="auto"` / `hdu=None` payload detection in `read()` and `get_header()`.
- `get_wcs()` convenience API for file-based WCS initialization.
- ML loader benchmark script (`bench_ml_loader.py`).

### Changed
- Minimum supported Python version raised to 3.11 (later raised to 3.12 in 0.4.0.dev0).
- Build system updated for vendored CFITSIO. AVX2/SSSE3 optimizations in C++ hot paths.

### Fixed
- FITS filename handling hardening and header/column-name sanitization on write paths.
- WCS/TAN GPU approximation correctness fixes.
- CI/build-system reliability fixes (build isolation, formatter compliance, dependency setup).
- Multiple interoperability and non-table regression fixes with test coverage updates.

### Performance Notes
- Benchmarks were re-baselined and workflow noise was reduced (repeat control, focused suites, stable fast runs) to make before/after performance comparisons more reproducible.
- Full 0.2.0 release benchmark (with tables) shows torchsky ahead in 87/88 `read_full` rows vs `fitsio` and 87/88 vs `fitsio_torch`; only `medium_float64_2d` remains a slight loss in this snapshot.
- ML-loader benchmark configuration was aligned for fairer comparisons (persistent workers + payload HDU detection); current CPU results are near parity with run-to-run variance (uncompressed median 0.985x vs `fitsio`, compressed median 1.008x), so performance claims should use repeated runs rather than single-shot output.

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

[0.1.0]: https://github.com/sfabbro/torchsky/releases/tag/v0.1.0
[0.1.1]: https://github.com/sfabbro/torchsky/releases/tag/v0.1.1
[0.2.0]: https://github.com/sfabbro/torchsky/releases/tag/v0.2.0
[0.2.1]: https://github.com/sfabbro/torchsky/releases/tag/v0.2.1
[0.3.0]: https://github.com/sfabbro/torchsky/releases/tag/v0.3.0
[0.3.1]: https://github.com/sfabbro/torchsky/releases/tag/v0.3.1
