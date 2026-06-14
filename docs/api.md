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
| Read FITS images, HDUs, headers, cutouts | `torchsky.read`, `torchsky.open`, `torchsky.io` | [io.md](io.md) |
| Work with FITS tables and Arrow scans | `torchsky.table` | [table.md](table.md) |
| Transform pixels and sky coordinates | `torchsky.wcs.WCS` | [wcs.md](wcs.md) |
| Use coordinate conversion helpers | `torchsky.wcs` | [wcs.md](wcs.md) |
| Use spherical predicates, regions, HEALPix, MOC, sparse maps, SHT | `torchsky.sphere` | [sphere.md](sphere.md) |
| Build coadds and map products | `torchsky.map` | [map.md](map.md) |
| Work with scenes and observation facades | `torchsky.scene` | [scene.md](scene.md) |
| Crossmatch catalogs | `torchsky.catalogs` | [catalogs.md](catalogs.md) |
| Work with spectra and cubes | `torchsky.spectral` | [spectral.md](spectral.md) |
| Build PyTorch datasets/loaders | `torchsky.datasets`, `torchsky.transforms`, `torchsky.cache` | [ml.md](ml.md) |
| Use field encoders | `torchsky.encoders` | [encoders.md](encoders.md) |
| Inspect FITS files from a shell | `torchsky info`, `torchsky cat` | [cli.md](cli.md) |
| Run the browser viewer | `torchsky.viz`, `torchsky viz` | [viz.md](viz.md), [cli.md](cli.md) |

## Stable Surface Tiers

`torchsky` exposes public surfaces in three broad tiers:

- **Core:** FITS I/O, table access, WCS transforms, promoted sphere/map
  primitives, and HDU/header objects.
- **Compatibility:** APIs that intentionally mirror useful upstream behavior,
  such as healpy-shaped HEALPix access.
- **Experimental:** active modelling and workflow surfaces such as field/INR
  map models, visualization server paths, and training-facing utilities.

Surfaces are also assigned to architecture pillars: `io`, `world`, `scene`, and
`viz`. Use `torchsky.surfaces` when you need programmatic API-surface metadata.

## Architecture Pillars

| Pillar | Role |
|---|---|
| `io` | FITS images, HDUs, tables, cutouts, streaming, and cache policy. |
| `world` | WCS, frames, spherical geometry, HEALPix, MOC, sparse maps, and harmonics. |
| `scene` | Observations, operators, instruments, latent scenes, products, coadds, objectives, renderers, and fits. |
| `viz` | Browser and notebook-facing visualization. |

## Public Namespaces

- [`torchsky.io`](io.md)
- [`torchsky.table`](table.md)
- [`torchsky.wcs`](wcs.md)
- [`torchsky.sphere`](sphere.md)
- [`torchsky.map`](map.md)
- [`torchsky.scene`](scene.md)
- [`torchsky.field`](field.md)
- [`torchsky.transforms`](ml.md)
- [`torchsky.datasets`](ml.md)
- [`torchsky.cache`](ml.md)
- [`torchsky.catalogs`](catalogs.md)
- [`torchsky.spectral`](spectral.md)
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
