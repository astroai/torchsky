# Ecosystem compatibility

torchsky is **torch-native first**: primitives, kernels, and public tensor paths are designed for performance, differentiability, and scale. On top of that, the project maintains **compatibility surfaces** so workflows built on common astronomy stacks can migrate incrementally and be checked with **parity tests** against trusted libraries.

The goal is **not** to match every internal detail of upstream packages. The intent is to match **behavior users rely on** at useful API boundaries (see `AGENTS.md`).

## World-layer replacement matrix

A **job** is claimed only when all four exist: a native implementation with no
runtime import of the peer, oracle tests or packaged fixtures, a throughput row
versus the peer on CPU (and CUDA/MPS only where the device is claimed), and
docs that fail loud on gaps. Linux leadership is not claimed from Mac numbers.
Thin shims (`torchsky.sphere.compat`, `torchsky.wcs.WCS`) stay for migration;
this is not `import healpy` and not a public-symbol clone.

| Peer | Jobs | torchsky entry | Oracle | Status |
|---|---|---|---|---|
| healpy | Pixel API; scalar/spin SHT; synfast/anafast; `gal_cut`; RING quadrature weights | `sphere.healpix`, `sphere.harmonics`, `sphere.compat` | healpy/ducc | **partial** — fused Ishioka `map2alm_spin` leads healpy on Mac CPU; `alm2map_spin` still ~1.2–1.4× healpy there. Linux unconfirmed until CANFAR VOS artifacts exist. Weights are computed RING quadrature, not HEALPix weight FITS files. |
| hpgeom | `query_disc` / box / ellipse / strip | `sphere.healpix` | hpgeom | **claimed** for query semantics. Inclusive disc padding and `fact==4` stay documented. |
| astropy-healpix | Same pixel jobs as healpy/hpgeom | `sphere.healpix` | astropy-healpix | Oracle only; no third API. |
| healsparse | construct/update, `get_values`/`valid_pixels`, coverage, `ud_grade`, FITS `.hs`, union/intersection sums | `sphere.HealSparseMap` | healsparse | **partial** — jobs implemented and valid-pixel values match (`maxabs 0.0`). Fused CPU `get_values` matches healsparse on Mac at nside=256 / 50k nest queries (~0.20 ms median interleaved, ratio ~1.03). Coverage-block `ud_grade` (float maps, coverage nside unchanged) matches healsparse on Mac at nside=256→128 and 512→256, 10% nest, interleaved medians ~1.02× and ~1.09× (`maxabs ~4e-16`). Do not claim Linux. |
| mocpy | cone/polygon, contains, union/area, FITS/JSON/ASCII, empty/full, wrap | `sphere.MOC` | mocpy | **partial** — STMOC/time is out of scope. |
| spherical-geometry | contains, intersect, union, area | `sphere.geom` | spherical_geometry | **claimed** for contains/area. Exact backend vs spherical-geometry area atol `1e-12` steradians (`tests/test_sphere_exact_regions.py`). Approximate backend is the fast CPU path; polygons are CPU-only. |
| astropy.wcs | FITS pixel↔world, SIP/TPV/TNX/ZPX, spectral/temporal, slicing | `wcs.WCS` | astropy.wcs | **partial** — large-N CPU/CUDA first; small-N overhead under 3 ms is acceptable. Coupled `-TAB` inverse stays fail-loud unless `pixel_n_dim == world_n_dim`. MPS is float32. |
| gwcs | Calibration-time detector→sky callable | `wcs.sample_gwcs_detector_to_tangent`, `RegularGridTransform2D` | gwcs at sample time only | **partial** — runtime never imports `gwcs`. Inverse and multi-epoch frozen tables are torch operators with residual bounds. |
| astropy.coordinates / ERFA | Geometric frames, space motion | `wcs.CatalogState`, `wcs.frames` | Astropy/ERFA | **partial** — not a `SkyCoord` clone. Observed AltAz needs explicit IERS DUT1 (`wcs.observed`). |

Out of this program: ducc0 internals, lsst.sphgeom, wcslib/pyast/Kapteyn (optional WCS benches only), reproject/SWarp/drizzle, astropy.io.fits, map stackers.

## Reference libraries (non-exhaustive)

| Area | Typical references | torchsky entry / notes |
|------|-------------------|-------------------------|
| HEALPix pixel + harmonic API | **healpy**, **hpgeom** (strict queries), **astropy-healpix** | `torchsky.sphere.healpix`, `torchsky.sphere.compat`, `torchsky.sphere.healpy` (alias); spectral transforms are torch-native—compare to healpy in tests/benchmarks only |
| WCS / projections | **Astropy** `astropy.wcs`, FITS WCS | `torchsky.wcs.WCS`, C++/fused paths; parity tests vs Astropy where scoped |
| FITS I/O & tables | **fitsio**, **Astropy** `io.fits` / tables | [`torchfits`](https://github.com/astroai/torchfits); out of scope for torchsky |
| Spherical geometry / polygons | **spherical_geometry**, **astropy-healpix** (where applicable) | `torchsky.sphere` and `torchsky.sphere.geom`; exact regions and benchmarks vs upstream fixtures |
| Catalogs / coordinates | **Astropy** `SkyCoord` / ERFA | `torchsky.wcs.CatalogState` and `torchsky.wcs.frames`; see [coordinates.md](coordinates.md). Not a SkyCoord clone. |

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
cross-package baseline is `torchfits 1.0.0rc4` or newer.

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
