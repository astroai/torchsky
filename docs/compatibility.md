# Ecosystem compatibility

torchsky is **torch-native first**: primitives, kernels, and public tensor paths are designed for performance, differentiability, and scale. On top of that, the project maintains **compatibility surfaces** so workflows built on common astronomy stacks can migrate incrementally and be checked with **parity tests** against trusted libraries.

The goal is **not** to match every internal detail of upstream packages. The intent is to match **behavior users rely on** at useful API boundaries (see `AGENTS.md`).

## Reference libraries (non-exhaustive)

| Area | Typical references | torchsky entry / notes |
|------|-------------------|-------------------------|
| HEALPix pixel + harmonic API | **healpy**, **hpgeom** (strict queries), **astropy-healpix** | `torchsky.sphere.healpix`, `torchsky.sphere.compat`, `torchsky.sphere.healpy` (alias); spectral transforms are torch-native—compare to healpy in tests/benchmarks only |
| WCS / projections | **Astropy** `astropy.wcs`, FITS WCS | `torchsky.wcs.WCS`, C++/fused paths; parity tests vs Astropy where scoped |
| FITS I/O & tables | **fitsio**, **Astropy** `io.fits` / tables | [`torchfits`](https://github.com/astroai/torchfits); out of scope for torchsky |
| Spherical geometry / polygons | **spherical_geometry**, **astropy-healpix** (where applicable) | `torchsky.sphere` and `torchsky.sphere.geom`; exact regions and benchmarks vs upstream fixtures |
| Catalogs / coordinates | **Astropy** `SkyCoord`, etc. | Compatibility-oriented tests and helpers where explicitly maintained |

## Where compatibility lives

- **HEALPix native pixel API** — `torchsky.sphere.healpix` owns indexing, neighbors, interpolation, range helpers, and query primitives.
- **HEALPix / healpy-shaped API** — [`torchsky.sphere.compat`](sphere.md#compatibility) and [`torchsky.sphere.healpy`](sphere.md#compatibility) (same symbols; use `import torchsky.sphere.healpy as hp` if you prefer healpy naming).
- **Sphere native + extended geometry** — `torchsky.sphere` (regions, MOC, sparse maps, etc.), often compared to healpy/hpgeom/spherical_geometry in tests and benchmarks.
- **WCS** — `torchsky.wcs` (see tests and WCS-related benchmarks).
- **FITS / tables** — `torchfits`; torchsky accepts decoded tensors, headers,
  masks/weights, and table batches (see `io.md` and `table.md`).
- **Mapmaking** — `torchsky.map` for native map products, with external stackers used only as parity and benchmark comparators (see `map.md` and `benchmarks.md`).

## Upstream Validation Matrix

“Upstream” means reference libraries, fixture families, and public data surfaces
used to validate behavior. It does not mean torchsky depends on those packages
at runtime.

### Astropy

| Upstream surface | Source | Local coverage | Contract |
|---|---|---|---|
| WCS core fixture-heavy tests | `astropy/wcs/tests/test_wcs.py` | `tests/test_wcs_parity.py`, `tests/test_wcs_allsky.py`, `tests/test_wcs_tpv.py`, `tests/test_wcs_legacy.py`, `tests/test_wcs_zpx.py`, `benchmarks/bench_wcs.py`, `benchmarks/bench_wcs_suite.py` | parity-test, benchmark-hook |
| FITS docs/examples | `astropy.io.fits` docs | Owned and validated by `torchfits`; torchsky retains only a cross-package compute smoke | integration |

### Healpy

| Upstream surface | Source | Local coverage | Contract |
|---|---|---|---|
| HEALPix primitive tests/tutorial flows | healpy tests/tutorials | `tests/test_healpix_parity.py`, `benchmarks/bench_healpix.py`, `benchmarks/bench_healpix_advanced.py`, `benchmarks/bench_spin_matrix.py` | parity-test, benchmark-hook |

### HealSparse

| Upstream surface | Source | Local coverage | Contract |
|---|---|---|---|
| I/O lifecycle | `tests/test_io.py` | `tests/test_healsparse_upstream.py`, `benchmarks/bench_sphere_sparse.py` | adapter-test, benchmark-hook |
| Degrade semantics | `tests/test_degrade.py` | `tests/test_healsparse_upstream.py`, `benchmarks/bench_sphere_sparse.py` | adapter-test, benchmark-hook |
| Geometry and map operations | `tests/test_geom.py`, `tests/test_operations.py` | `tests/test_healsparse_upstream.py`, `benchmarks/bench_sphere_sparse.py` | adapter-test |
| Quickstart/basic interface | `basic_interface.html` | `tests/test_healsparse_upstream.py` | example-smoke |

### Fitsio / Astropy

FITS I/O parity, compression, checksum, HDU, table, and throughput matrices
live in `torchfits`. Duplicating those matrices in torchsky would blur ownership
and waste CI. Torchsky validates only that decoded data and metadata cross the
boundary without changing WCS/map/catalog semantics. The supported
cross-package baseline is `torchfits 0.8.0` or newer.

### Real-Data Reference Fixtures

| Real-data surface | Source | Local coverage | Contract |
|---|---|---|---|
| Installed scaled/VLA/WCS FITS fixtures | `astropy` package test data | `tests/test_real_data_validation.py` | adapter-test |
| Installed compressed image fixture | `fitsio` package test image | `tests/test_real_data_validation.py` | adapter-test |
| Installed HEALPix weight table | `healpy` package data | `tests/test_real_data_validation.py` | adapter-test |
| Public sample/reference FITS image | Astropy data server tutorial image | `tests/test_release_smoke.py` when network/fixture access is available | example-smoke |

If upstream replay scripts return, add their source inventory and restore the
corresponding docs/tests in the same PR. `healsparse` remains adapter-based
rather than exact API replay because torchsky intentionally exposes sparse-map
functionality through its own spherical namespace.

## Optional dependencies

Many parity and benchmark paths import **healpy**, **Astropy**, **hpgeom**, or others **only when installed**. They are used to **validate** torchsky behavior, not as the default implementation of core tensor APIs.

## Adding new compatibility

1. Prefer implementing behavior in **torch-native** primitives, then expose a **thin** compatibility wrapper with the upstream name/signature.
2. Add **tests** against the reference library or packaged fixtures when available.
3. Document the mapping here or in the domain doc (`sphere.md`, `api.md`, etc.).
