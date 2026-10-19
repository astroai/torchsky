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
| Use catalog frames and space motion | `torchsky.coords` / `CatalogState` | [coordinates.md](coordinates.md) |
| Crossmatch catalogs | `torchsky.catalogs` | [catalogs.md](catalogs.md) |
| Use spherical predicates, regions, HEALPix, MOC, sparse maps, SHT | `torchsky.sphere` | [sphere.md](sphere.md) |
| Build coadds and map products | `torchsky.map` | [map.md](map.md) |
| Work with scenes and observation facades | `torchsky.scene` | [scene.md](scene.md) |
| Work with spectra and cubes | `torchsky.spectral` | [spectral.md](spectral.md) |
| Sample-wise redshift, BERV, lnλ LSF | `torchsky.redshift`, `torchsky.log_lambda` | [spectral.md](spectral.md) |
| Synthetic photometry / passbands | `torchsky.spectrophotometry` | [spectral.md](spectral.md) |
| Dust laws (CCM89 / O'Donnell / F99 / Calzetti) | `torchsky.extinction` | [spectral.md](spectral.md) |
| FLRW distances, comoving geometry, and AP scaling | `torchsky.cosmology` | [spectral.md](spectral.md) |
| Build tensor transforms and training inputs | `torchsky.transforms` plus reader-defined datasets | [ml.md](ml.md) |
| Native forced photometry, blends, and likelihood evidence | `torchsky.map.likelihood`, `torchsky.map.query` | [native-likelihood.md](native-likelihood.md) |
| Native detector response, nuisances, and physical stages | `torchsky.map.operators`, `torchsky.map.field.native_response` | [native-likelihood.md](native-likelihood.md) |
| PSF pixel-response conventions | `torchsky.map.operators` | [psf-response-conventions.md](psf-response-conventions.md) |
| Use field encoders | `torchsky.encoders` | [encoders.md](encoders.md) |
| Inspect FITS files from a shell | `torchfits` tools/APIs | [io.md](io.md) |
| Run the browser viewer | `torchsky.viz`, `torchsky viz` | [viz.md](viz.md), [cli.md](cli.md) |

## Stable Surface Tiers

`torchsky` exposes public surfaces in three broad tiers:

- **Core:** WCS transforms, sphere/map primitives, catalog geometry, and
  observation operators.
- **Compatibility:** APIs that intentionally mirror useful upstream behavior,
  such as healpy-shaped HEALPix access.
- **Experimental:** scene representations/operator adapters, field/INR models,
  native likelihood/query/evidence, visualization server paths, and
  training-facing utilities.

Surfaces are assigned to architecture pillars: `world`, `scene`, and `viz`.
FITS I/O is a package boundary, not a torchsky pillar. The public contract is
enumerated by the namespaces below and checked by the package API tests.

## Architecture Pillars

| Pillar | Role |
|---|---|
| `world` | WCS, frames, spherical geometry, HEALPix, MOC, sparse maps, and harmonics. |
| `scene` | Observations, operators, instruments, latent scenes, products, coadds, objectives, renderers, and fits. |
| `viz` | Browser and notebook-facing visualization. |

## Reference Pages

- [WCS reference](api-wcs.md): `torchsky.wcs` classes, frames, tabular axes,
  motion/time helpers, spectral axes.
- [Sphere and HEALPix reference](api-sphere.md): indexing, queries, regions,
  coverage maps, harmonic transforms.
- [Map reference](api-map.md): the 31-name facade, planning and coadd entry
  points, configs and results, submodule ownership.
- [Field reference](api-field.md): build/fit/render/sample workflow, kinds and
  presets, observations, artifacts, spectral capacity.
- [Catalog reference](api-catalogs.md): catalog model, crossmatch entry points,
  streaming, and HEALPix index helpers.
- [Spectral reference](api-spectral.md): spectral axes, containers, readers,
  flux-conserving resampling, Doppler shifts, and cube writing.
- [Architecture](architecture.md): design target and pillar rules.

## Public Namespaces

- [`torchsky.wcs`](wcs.md)
- [`torchsky.coords`](coordinates.md)
- [`torchsky.sphere`](sphere.md)
- [`torchsky.map`](map.md)
- `torchsky.map.likelihood`, `torchsky.map.query`, `torchsky.map.evidence` —
  experimental native likelihood and forced-photometry surfaces
  ([native-likelihood.md](native-likelihood.md))
- [`torchsky.scene`](scene.md)
- [`torchsky.field`](field.md)
- [`torchsky.transforms`](ml.md)
- FITS I/O, tables, caches, and datasets are provided by `torchfits`; torchsky
  does not expose alias namespaces for them.
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
