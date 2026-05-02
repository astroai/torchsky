# API Guide

This page is the API router for `torchsky`. Domain pages carry the detailed
concepts, recipes, examples, and grouped API surfaces.

For migration from common astronomy libraries, see
[compatibility.md](compatibility.md). For validation and performance, see
[benchmarks.md](benchmarks.md).

## Quick Paths

| Goal | Entry point | Guide |
|---|---|---|
| Read FITS images, HDUs, headers, cutouts | `torchsky.read`, `torchsky.open`, `torchsky.io` | [io.md](io.md) |
| Work with FITS tables and Arrow scans | `torchsky.table` | [table.md](table.md) |
| Transform pixels and sky coordinates | `torchsky.wcs.WCS` | [wcs.md](wcs.md) |
| Use coordinate conversion helpers | `torchsky.coords` | [coords.md](coords.md) |
| Use spherical predicates and regions | `torchsky.geom` | [geom.md](geom.md) |
| Use HEALPix, MOC, sparse maps, SHT | `torchsky.sphere` | [sphere.md](sphere.md) |
| Build coadds and map products | `torchsky.map` | [map.md](map.md) |
| Crossmatch catalogs | `torchsky.catalogs` | [catalogs.md](catalogs.md) |
| Work with spectra and cubes | `torchsky.spectral` | [spectral.md](spectral.md) |
| Build PyTorch datasets/loaders | `torchsky.FITSDataset`, `torchsky.transforms`, root loader/cache helpers | [ml.md](ml.md) |
| Use field encoders | `torchsky.encoders` | [encoders.md](encoders.md) |
| Run the browser viewer | `torchsky.viz`, `torchsky viz` | [viz.md](viz.md), [cli.md](cli.md) |

## Stable Surface Tiers

`torchsky` exposes public surfaces in three broad tiers:

- **Core:** FITS I/O, table access, WCS transforms, promoted sphere/map
  primitives, and HDU/header objects.
- **Compatibility:** APIs that intentionally mirror useful upstream behavior,
  such as healpy-shaped HEALPix access.
- **Experimental:** active modelling and workflow surfaces such as field/INR
  map models, visualization server paths, and training-facing utilities.

Use `torchsky.API_SURFACES`, `torchsky.SurfaceTier`, and
`torchsky.api_surfaces_by_tier(...)` when you need programmatic API-surface
metadata.

## Public Namespaces

- [`torchsky.io`](io.md)
- [`torchsky.table`](table.md)
- [`torchsky.wcs`](wcs.md)
- [`torchsky.coords`](coords.md)
- [`torchsky.geom`](geom.md)
- [`torchsky.sphere`](sphere.md)
- [`torchsky.map`](map.md)
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
