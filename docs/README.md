# torchsky Documentation

Start here when browsing the repository docs. Domain guides are organized around
the public `torchsky` namespaces and combine concepts, recipes, examples,
validation, and API pointers.

## Start Here

- [Getting started](getting-started.md) - first 10 minutes through I/O, tables,
  WCS, HEALPix, and map products.
- [Architecture](architecture.md) - north-star design, public pillars, and
  refactor rules.
- [API guide](api.md) - routing table for public namespaces and domain pages.
- [Installation and source builds](install.md) - wheels, source builds, CFITSIO,
  CUDA notes, and pip-only fallback guidance.
- [Examples](examples.md) - runnable scripts grouped by domain.

## Domain Guides

- [I/O](io.md) - root FITS I/O, `torchsky.io`, HDUs, headers, cutouts,
  checksums, batch reads, and runtime caches.
- [Tables](table.md) - `torchsky.table`, Arrow scanners, Polars/DuckDB/Pandas,
  predicate pushdown, and table mutation.
- [WCS](wcs.md) - `torchsky.wcs`, transforms, spectral/time axes, frames,
  alignment, TAB axes, and compiled tensor paths.
- [Coordinates](coords.md) - `torchsky.coords`, frame and spectral conversion
  helpers.
- [Geometry](geom.md) - `torchsky.geom`, spherical predicates and regions.
- [Sphere and HEALPix](sphere.md) - `torchsky.sphere`, HEALPix, geometry,
  sparse maps, MOC, and harmonic primitives.
- [Map](map.md) - `torchsky.map`, products, kernels, value semantics, truth
  evaluation, operators, and field/INR workflows.
- [Field / INR fitting](field.md) - `FieldObservation`, `make_field_model`,
  `fit_field_model`, rendering cubes and HEALPix (experimental).
- [Implicit forward modeling](implicit_forward.md) - multi-survey losses,
  kinematics, `predict_observation_mean`, Charbonnier scaffold.
- [Catalogs](catalogs.md) - `torchsky.catalogs`, crossmatch, and streaming
  catalog datasets.
- [Spectral data](spectral.md) - `torchsky.spectral`, spectra, cubes, and
  spectral axes.
- [ML and runtime](ml.md) - datasets, dataloaders, transforms, caches, and
  buffers.
- [Encoders](encoders.md) - `torchsky.encoders` and field-feature encoders.
- [Visualization](viz.md) - `torchsky.viz` browser viewer.
- [CLI](cli.md) - `torchsky` command-line workflows.

## Compatibility And Measurement

- [Ecosystem compatibility](compatibility.md) - reference libraries, optional
  compatibility surfaces, and upstream validation coverage.
- [Benchmarks](benchmarks.md) - methodology, comparator environments, real-data
  setup, SMACS/UPdec runs, tasks, and benchmark snapshot.

## Projects

- [Hyperspectral fusion / neural fields (student onboarding)](projects/hyperspectral-fusion.md) -
  SMACS mini-cutouts, classical stack baseline, quicklook visualization, INR first fit, and library vs research split.

## Contributor Docs

- [Contributing guide](contributing.md) - setup, local gates, benchmark policy,
  and PR expectations.
- [Release checklist](release.md) - version sync, gates, artifacts, and
  publishing.
- [Changelog](changelog.md) - release notes.

## Roadmap

These files are planning and research context, not the current public API
contract.

- [Roadmap status](roadmap/status.md) - roadmap thesis plus code-grounded status.
- [Research knowledge base](roadmap/knowledge-base.md)
- [World model whitepaper plan](roadmap/world-model-whitepaper-plan.md)
- [Sky measure scene plan](roadmap/sky-measure-scene-plan.md)

## Archive

Archived docs are preserved for context and provenance. Do not treat them as
current workflow or API contracts.

- [Torchfits migration history](archive/torchfits-migration-history.md)
- [Benchmark history](archive/benchmark-history.md)
- [Development handoff snapshot](archive/development-handoff.md)
