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
- [Map](map.md) - `torchsky.map`, products, kernels, value semantics,
  operators, and field/INR workflows.
- [Continuous fields](field.md) - `torchsky.field`, composable `model`, `fit`,
  `render`, and `sample` workflows (experimental).
- [Implicit forward modeling](implicit_forward.md) - multi-survey losses,
  kinematics, `predict_observation_mean`, Charbonnier scaffold.
- [Native likelihood and calibration foundations](native-likelihood.md) -
  native response, Gaussian evidence, linear queries, physical detector stages,
  and the bounded Poisson–Gaussian law (experimental).
- [PSF response conventions](psf-response-conventions.md) - detector-grid
  kernel orientation and source-flux versus uniform-sky contracts.
- [Catalogs](catalogs.md) - `torchsky.catalogs`, crossmatch, and streaming
  catalog datasets. Scale covering: [catalog matching research](research/catalog_matching.md).
- [Spectral data](spectral.md) - `torchsky.spectral`, spectra, cubes, and
  spectral axes.
- [ML and runtime](ml.md) - datasets, dataloaders, transforms, caches, and
  buffers.
- [Encoders](encoders.md) - `torchsky.encoders` and field-feature encoders.
- [Visualization](viz.md) - `torchsky.viz` browser viewer.
- [CLI](cli.md) - `torchsky` command-line workflows.

## Compatibility And Measurement

- [Ecosystem compatibility](compatibility.md) - reference libraries, optional
  compatibility surfaces, and the package boundary for parity work.
- [Scientific evaluation](benchmarks.md) - evaluation ownership and how to use
  the sibling harness; detailed methodology, comparator environments,
  real-data campaigns, and scorecards live in
  [`torchsky-harness`](https://github.com/astroai/torchsky-harness).

## Projects

- [INR and Power-Foam formulation](projects/inr_foam_problem_formulation.md) -
  scientific objective and model formulation for differentiable spatial-
  spectral reconstruction.
- [Field response and discovery plan](projects/field_response_discovery_plan.md) -
  accepted native-response, calibration, and discovery plan.
- [Field response implementation status](projects/field_response_discovery_status.md) -
  bounded components, exact checks, and open scientific gates.
- [Field response amendments](projects/field_response_discovery_amendments.md) -
  revisions and clarifications to the accepted plan.
- [Research notes](research/catalog_matching.md) - focused literature context
  for catalog matching; this is not a public API contract.

## Contributor Docs

- [Contributing guide](contributing.md) - setup, local gates, benchmark policy,
  and PR expectations.
- [Release checklist](release.md) - version sync, gates, artifacts, and
  publishing.
- [Changelog](changelog.md) - release notes.

## Roadmap

These files track package architecture and implementation concerns, not
scientific campaign status or public API guarantees.

- [Roadmap](roadmap/status.md) - package ownership, implemented surfaces, and
  durable engineering priorities.
- [Execution principles](roadmap/next.md) - bounded implementation and review
  priorities for core changes.
- [Catalog matching research](research/catalog_matching.md) - exact S² covering
  vs GIS / point-cloud / ANN literature; not a public API.
- [Implementation history](roadmap/history.md) - package-level ownership and
  architecture decisions.

## Archive

Archived docs are preserved for context and provenance. Do not treat them as
current workflow or API contracts.

- [Torchfits migration history](archive/torchfits-migration-history.md)
- [Roadmap knowledge base](archive/roadmap-knowledge-base.md)
- [World model whitepaper plan](archive/roadmap-world-model-whitepaper-plan.md)
- [Sky measure scene plan](archive/roadmap-sky-measure-scene-plan.md)

Scientific campaign records, scorecards, historical evaluation, and the
student-facing SMACS workflow are maintained in
[`torchsky-harness`](https://github.com/astroai/torchsky-harness), including
its [benchmark methodology](https://github.com/astroai/torchsky-harness/blob/main/docs/benchmarks.md)
and [campaigns](https://github.com/astroai/torchsky-harness/tree/main/campaigns).
