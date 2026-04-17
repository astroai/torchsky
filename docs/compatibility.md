# Ecosystem compatibility

torchsky is **torch-native first**: primitives, kernels, and public tensor paths are designed for performance, differentiability, and scale. On top of that, the project maintains **compatibility surfaces** so workflows built on common astronomy stacks can migrate incrementally and be checked with **parity tests** against trusted libraries.

This is **not** a goal of matching every internal detail of upstream packages. The intent is to match **behavior users rely on** at useful API boundaries (see `AGENTS.md`).

## Reference libraries (non-exhaustive)

| Area | Typical references | torchsky entry / notes |
|------|-------------------|-------------------------|
| HEALPix pixel + harmonic API | **healpy**, **hpgeom** (strict queries), **astropy-healpix** | `torchsky.sphere.compat`, `torchsky.sphere.healpy` (alias), `torchsky.wcs.healpix`; spectral transforms are torch-native—compare to healpy in tests/benchmarks only |
| WCS / projections | **Astropy** `astropy.wcs`, FITS WCS | `torchsky.wcs.WCS`, C++/fused paths; parity tests vs Astropy where scoped |
| FITS I/O & tables | **fitsio**, **Astropy** `io.fits` / tables | `torchsky.read` / `write`, table module, Arrow/Polars interop |
| Spherical geometry / polygons | **spherical_geometry**, **astropy-healpix** (where applicable) | `torchsky.sphere.geom`, exact regions, benchmarks vs upstream fixtures |
| Catalogs / coordinates | **Astropy** `SkyCoord`, etc. | Compatibility-oriented tests and helpers where explicitly maintained |

## Where compatibility lives

- **HEALPix / healpy-shaped API** — [`torchsky.sphere.compat`](sphere.md#compatibility-map) and [`torchsky.sphere.healpy`](sphere.md#compatibility-map) (same symbols; use `import torchsky.sphere.healpy as hp` if you prefer healpy naming).
- **Sphere native + extended geometry** — `torchsky.sphere` (regions, MOC, sparse maps, etc.), often compared to healpy/hpgeom/spherical_geometry in tests and benchmarks.
- **WCS** — `torchsky.wcs` (see tests and WCS-related benchmarks).
- **FITS / tables** — package root I/O and `torchsky.table` (see `docs/api.md`).

## Optional dependencies

Many parity and benchmark paths import **healpy**, **Astropy**, **hpgeom**, or others **only when installed**. They are used to **validate** torchsky behavior, not as the default implementation of core tensor APIs.

## Adding new compatibility

1. Prefer implementing behavior in **torch-native** primitives, then expose a **thin** compatibility wrapper with the upstream name/signature.
2. Add **tests** against the reference library or packaged fixtures when available.
3. Document the mapping here or in the domain doc (`sphere.md`, `api.md`, etc.).
