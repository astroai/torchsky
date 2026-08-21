# torchsky Documentation

Start here when browsing the repository docs. Domain guides are organized around
the public `torchsky` namespaces and combine concepts, recipes, examples,
validation, and API pointers.

## Start Here

- [Getting started](getting-started.md) - first 10 minutes through the
  `torchfits` boundary, WCS, HEALPix, and map products.
- [Architecture](architecture.md) - north-star design, public pillars, and
  refactor rules.
- [API guide](api.md) - routing table for public namespaces and domain pages.
- [Installation and source builds](install.md) - wheels, `torchfits`
  integration, native builds, CUDA notes, and pip-only fallback guidance.
- [Examples](examples.md) - runnable scripts grouped by domain.

## Domain Guides

- [I/O boundary](io.md) - loading tensors and metadata with `torchfits`, then
  passing them into torchsky compute APIs.
- [Table boundary](table.md) - `torchfits.table` scans and tensor/Arrow batches
  consumed by `torchsky.catalogs`.
- [WCS](wcs.md) - `torchsky.wcs`, transforms, spectral/time axes, frames,
  alignment, TAB axes, and compiled tensor paths.
- [Coordinates](coordinates.md) - `torchsky.coords`, `CatalogState`, geometric
  frames, and observed place with explicit IERS DUT1.
- [Scene](scene.md) - `torchsky.scene`, promoted facade, observations, target render
  specifications, and operator adapters.
- [Sphere and HEALPix](sphere.md) - `torchsky.sphere`, HEALPix, geometry,
  sparse maps, MOC, and harmonic primitives.
- [Map](map.md) - `torchsky.map`, products, kernels, value semantics, truth
  evaluation, operators, and field/INR workflows.
- [Continuous fields](field.md) - `torchsky.field`, composable `model`, `fit`,
  `render`, and `sample` workflows (experimental).
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

- [Roadmap](roadmap/status.md) - current strategy, domain status, active queue,
  and standing gates.
- [Execution roadmap](roadmap/next.md) - ordered Torchsky map, INR, Power-Foam,
  multi-wavelength depth, and infrastructure task list.
- [Roadmap history](roadmap/history.md) - maintained record of shipped
  improvements, validation gates, and follow-ups.

## Archive

Archived docs are preserved for context and provenance. Do not treat them as
current workflow or API contracts.

- [Torchfits migration history](archive/torchfits-migration-history.md)
- [Benchmark history](archive/benchmark-history.md)
- [Development handoff snapshot](archive/development-handoff.md)
- [Roadmap knowledge base](archive/roadmap-knowledge-base.md)
- [World model whitepaper plan](archive/roadmap-world-model-whitepaper-plan.md)
- [Sky measure scene plan](archive/roadmap-sky-measure-scene-plan.md)
