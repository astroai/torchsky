# API Guide

This page is the API router for `torchsky`. Domain pages carry the detailed
concepts, recipes, examples, and grouped API surfaces.

For migration from common astronomy libraries, see
[compatibility.md](compatibility.md). For validation and performance, see
[benchmarks.md](benchmarks.md).

For the high-level design target and refactor rules, see
[architecture.md](architecture.md).

## Quick Paths

| Goal | Entry point | Guide |
|---|---|---|
| Read/write FITS images, HDUs, headers, cutouts | `torchfits` | [io.md](io.md) |
| Work with FITS tables and Arrow scans | `torchfits.table` | [table.md](table.md) |
| Transform pixels and sky coordinates | `torchsky.wcs.WCS` | [wcs.md](wcs.md) |
| Use catalog frames and space motion | `torchsky.wcs.CatalogState` | [coordinates.md](coordinates.md) |
| Use spherical predicates, regions, HEALPix, MOC, sparse maps, SHT | `torchsky.sphere` | [sphere.md](sphere.md) |
| Build coadds and map products | `torchsky.map` | [map.md](map.md) |
| Work with scenes and observation facades | `torchsky.scene` | [scene.md](scene.md) |
| Crossmatch catalogs | `torchsky.catalogs` | [catalogs.md](catalogs.md) |
| Work with spectra and cubes | `torchsky.spectral` | [spectral.md](spectral.md) |
| Sample-wise redshift, BERV, lnλ LSF | `torchsky.redshift`, `torchsky.log_lambda` | [spectral.md](spectral.md) |
| Synthetic photometry / passbands | `torchsky.spectrophotometry` | [spectral.md](spectral.md) |
| Dust laws (CCM89 / O'Donnell / F99 / Calzetti) | `torchsky.extinction` | [spectral.md](spectral.md) |
| Flat ΛCDM / wCDM distances | `torchsky.cosmology` | [spectral.md](spectral.md) |
| Build tensor transforms and training inputs | `torchsky.transforms` plus reader-defined datasets | [ml.md](ml.md) |
| Use field encoders | `torchsky.encoders` | [encoders.md](encoders.md) |
| Inspect FITS files from a shell | `torchfits` tools/APIs | [io.md](io.md) |
| Run the browser viewer | `torchsky.viz`, `torchsky viz` | [viz.md](viz.md), [cli.md](cli.md) |

## Stable Surface Tiers

`torchsky` exposes public surfaces in three broad tiers:

- **Core:** WCS transforms, sphere/map primitives, catalog geometry, observation
  operators, and tensor/header-based scene objects.
- **Compatibility:** APIs that intentionally mirror useful upstream behavior,
  such as healpy-shaped HEALPix access.
- **Experimental:** active modelling and workflow surfaces such as field/INR
  map models, visualization server paths, and training-facing utilities.

Surfaces are assigned to architecture pillars: `world`, `scene`, and `viz`.
FITS I/O is a package boundary, not a torchsky pillar. Use `torchsky.surfaces`
when you need programmatic API-surface metadata.

## Architecture Pillars

| Pillar | Role |
|---|---|
| `world` | WCS, frames, spherical geometry, HEALPix, MOC, sparse maps, and harmonics. |
| `scene` | Observations, operators, instruments, latent scenes, products, coadds, objectives, renderers, and fits. |
| `viz` | Browser and notebook-facing visualization. |

## Public Namespaces

- [`torchsky.wcs`](wcs.md)
- [`torchsky.sphere`](sphere.md)
- [`torchsky.map`](map.md)
- [`torchsky.scene`](scene.md)
- [`torchsky.field`](field.md)
- [`torchsky.transforms`](ml.md)
- [`torchsky.datasets`](ml.md)
- `torchsky.cache` is a compatibility alias for `torchfits.cache`.
- [`torchsky.catalogs`](catalogs.md)
- [`torchsky.spectral`](spectral.md)
- [`torchsky.spectrophotometry`](spectral.md)
- [`torchsky.extinction`](spectral.md)
- [`torchsky.cosmology`](spectral.md)
- [`torchsky.encoders`](encoders.md)
- [`torchsky.viz`](viz.md)

## Cross-Cutting Docs

- [getting-started.md](getting-started.md)
- [examples.md](examples.md)
- [compatibility.md](compatibility.md)
- [benchmarks.md](benchmarks.md)
- [contributing.md](contributing.md)
- [ml.md](ml.md)
- [cli.md](cli.md)
- [install.md](install.md)
