# Changelog

All notable changes to torchsky are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Changes after 0.5.0rc1 are listed here.

## [0.5.0rc1] - 2026-11-03

First public release of torchsky: a torch-native astronomy stack for sky
coordinates, geometry, maps, observation operators, scenes, and field models.
FITS file and table I/O is provided by the separate
[`torchfits`](https://github.com/astroai/torchfits) package, a required
dependency.

### Supported lanes
- Python 3.12, 3.13 and 3.14 (full CI suite on each, plus macOS for 3.14).
- PyTorch 2.14; torchfits 1.3.1 or newer.
- CPU, CUDA 13 and newer (PyTorch `cu130`, validated on an NVIDIA H100), and
  Apple MPS.
- Linux x86_64 and macOS 14+ arm64 wheels.

### What torchsky provides
- **WCS** (`torchsky.wcs`): FITS WCS for celestial, spectral, and time axes with
  every standard projection, SIP, TPV, TNX/ZPX, paper-IV lookup tables, `-TAB`
  axes, exact Jacobians, footprints, and astrometric fitting; batched,
  differentiable, and device-aware.
- **Coordinates** (`torchsky.coords`): `CatalogState` with proper motion,
  parallax, radial velocity, and epochs; geometric frame conversion; observed
  place (ERFA Earth ephemeris, aberration, light deflection, refraction)
  through explicit `EarthOrientation` and `Site`.
- **Sphere** (`torchsky.sphere`): HEALPix pixelization (RING and NEST),
  queries that descend the quadtree, spherical regions, MOC and sparse maps,
  spherical harmonic transforms (scalar and spin) with `lmax` up to
  `3*nside-1`, and a healpy-shaped `torchsky.sphere.compat` namespace.
- **Catalogs** (`torchsky.catalogs`): sky-indexed cone search and crossmatch with
  `best1`, `best2`, `mutual`, `symmetric`, and `all` modes, per-row errors,
  `join_rows`, exact matching, self-matching groups, and multi-catalog
  matching, with exact pair-set parity with STILTS on Gaia DR3 x 2MASS x AllWISE.
- **Mapmaking** (`torchsky.map`): planned resampling and coaddition (Lanczos,
  bilinear, nearest, exact square drizzle), weights, masks, DQ and variance
  propagation, automatic memory-budget tiling, HEALPix maps, and IFU cubes
  (`coadd_cube`) with flux-conserving spectral rebinning.
- **Spectra and photometry**: spectral axes, relativistic Doppler shifts,
  passbands and synthetic photometry, dust extinction laws, and differentiable
  FLRW cosmology.
- **Scenes and fields** (`torchsky.scene`, `torchsky.field`): observation
  operators, source-catalog scenes, continuous field models, chromatic PSF
  fields, and fitting; the native response and likelihood layer is experimental.
- **ML and tooling**: `torchsky.transforms` and `torchsky.encoders`, the
  `torchsky` command line, and `torchsky.viz` all-sky figures
  (`plot_healpix_sky`, `plot_sky_grid`) on Mollweide, Aitoff, Hammer, and polar
  projections.
- The package ships `py.typed`.

### Highlights
- 135 JWST/NIRCam exposures (566 Mpix) coadded onto a 5025 x 5025 grid: Lanczos-3
  in 221 s on CPU and 65 s on a GPU slice; exact drizzle in 149 s on CPU and
  148 s on GPU. Drizzle output matches STScI drizzle to better than 2e-6 sigma.
- A 280-CCD CFHT/MegaCam tile coadd onto 10k x 10k pixels in 711 s on CPU and
  660 s on a GPU slice with automatic tiling and no out-of-memory failures.
- SIP forward, inverse, and Jacobian use nested Horner evaluation (JWST order-5
  SIP, 4 M points: distort 15x and Newton inverse 4x faster than a power stack).
- Observed place agrees with `erfa.atco13` to 0.012 mas.

Measurement environments and full tables are on the
[Performance highlights](../performance/highlights.md) page.

### Known limitations
- The native response/likelihood layer and `torchsky.map.solvers` are
  experimental and do not establish calibrated real-sky response.
- On a 5025 x 5025 grid, CPU Lanczos-3 is slower than SWarp; the GPU run is
  faster.
- MPS float32 cannot reach half a pixel on very fine WCS grids; a
  `PrecisionWarning` is raised.
- Resampling between FK5 (J2000) and ICRS emits `CelestialFrameBiasWarning`
  (about 20-30 mas); set `RADESYS = 'ICRS'` on headers solved against Gaia.
- Irregular spectral axes in `coadd_cube` use index-space interpolation with a
  recorded bias envelope.

torchsky continues the sky-geometry part of torchfits ≤ 0.3.1; see the torchfits changelog for earlier history.
