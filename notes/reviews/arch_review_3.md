# Source-First Architecture Review - 2026-05-02

This is an independent review of the current `torchsky` source architecture against
the stated goal: a differentiable, any-device, torch-native backbone for modelling
observations of the sky jointly across instruments, wavelengths, foregrounds,
backgrounds, explicit physical components, and learned components.

The review is source-first. I used `docs/` only to cross-check intended public
positioning and existing API tier language. The main evidence is from
`src/torchsky/` and tests that exercise those source paths.

## Executive Summary

`torchsky` has the raw ingredients for the right future architecture:

- native FITS, table, WCS, HEALPix, sphere, and mapmaking primitives
- a small runtime dependency contract
- real performance and parity tests
- early observation/product/target/scene/operator concepts
- a serious field-model and differentiable rendering effort

The risk is that the package shape still reads like a fast FITS plus WCS plus
coadd library with an experimental INR layer attached. The mission wants the
opposite center of gravity: an observation-native modelling stack where I/O,
coordinates, geometry, instruments, scenes, likelihoods, rendering, and fitting
are clearly separated but composable.

The highest leverage change is not a big user-breaking rename. It is to add and
promote a small spine:

```text
Observation -> Operator -> Scene -> Target/Product -> Objective -> Fit/Render
```

The existing `MapProductSpec`, `TargetRenderSpec`, `SceneRepresentation`,
`ObservationOperatorBundle`, `ImageObservation`, and `FieldObservation` are close
to that spine, but they are split across `map`, `map.field`, `map.operators`,
and `map.scenes`, and several are marked internal or experimental. They should
become the conceptual core, with old import paths kept as compatibility shims.

## Source Inventory

Python source concentration, counted directly from `src/torchsky`:

| Lines | File | Main role |
|---:|---|---|
| 7713 | `src/torchsky/map/field/core.py` | field model zoo, rendering, training loop |
| 4112 | `src/torchsky/sphere/spectral.py` | scalar/spin spherical harmonic transforms |
| 4108 | `src/torchsky/wcs/core.py` | WCS parser, transforms, fast paths, slicing |
| 3760 | `src/torchsky/table.py` | Arrow/FITS table scan, read, write, mutation |
| 3268 | `src/torchsky/map/image_plane.py` | image-plane coadd, kernels, planning |
| 3149 | `src/torchsky/__init__.py` | public facade plus FITS I/O engine and caches |
| 2410 | `src/torchsky/map/solvers.py` | solver hooks, priors, operator builders |
| 2228 | `src/torchsky/wcs/healpix.py` | HEALPix indexing and query kernels |
| 2225 | `src/torchsky/hdu.py` | HDU, Header, TensorHDU, TableHDU |
| 1854 | `src/torchsky/sphere/geom.py` | spherical polygons and regions |

Total Python source is about 57.7k lines across 105 files. C++ source and headers
are about 16.3k lines across 17 files, with most code in `fits.cpp`,
`healpix.cpp`, `table.cpp`, `bindings.cpp`, and `wcs.cpp`.

Public export counts are already high:

- `torchsky.__all__`: 83 symbols
- `torchsky.map.__all__`: 150 symbols
- `torchsky.sphere.__all__`: 77 symbols

These counts do not include submodule public surfaces such as
`torchsky.table.read`, `torchsky.table.scan`, or `torchsky.map.operators`.

## Current Architecture, As Implemented

The package is currently organized around these centers:

- `torchsky.__init__`: public facade and primary FITS I/O implementation.
- `torchsky.io`: thin wrappers back to `torchsky.__init__`, not an independent I/O
  implementation.
- `torchsky.table`: Arrow-oriented FITS table APIs.
- `torchsky.hdu`: HDU/Header object model.
- `torchsky.wcs`: FITS WCS and coordinate transforms.
- `torchsky.sphere`: spherical geometry, sparse maps, MOC, HEALPix compatibility,
  and spherical harmonics.
- `torchsky.map`: coadds, map products, field models, scene adapters, operators,
  solvers, HEALPix/image bridges, and many low-level kernels.
- `torchsky.viz`: optional FastAPI browser viewer.
- `torchsky.cli`: one `map` workflow and one `viz` workflow.

That shape explains the current user experience: users enter through FITS I/O,
then WCS/sphere, then `map`, and only then discover continuous fields, operators,
and scenes. For the mission, the conceptual entry should be observations and
models, not file formats and resampling functions.

## What Is Strong

### Dependency Discipline

`pyproject.toml` keeps core runtime dependencies to `torch` and `numpy`, with
heavy astronomy and table ecosystem packages in extras and Pixi environments.
This is aligned with a torch-native core. Source checks also enforce no top-level
`astropy` or Python `fitsio` imports in core package files outside allowed
contexts.

### Native Numerical Foundations

`WCS`, HEALPix, spherical geometry, spherical spectral transforms, and FITS/table
I/O are real native surfaces, not wrappers around Astropy/healpy/fitsio. This is
the right base for differentiability, any-device execution, and long-term
performance control.

### Benchmark and Parity Culture

Tests such as `tests/test_perf_gate.py`, map compatibility tests, upstream parity
tests, and no-external-backend tests create useful engineering pressure. The
project already treats legacy astronomy libraries as references and comparators,
which matches the mission.

### Early Modelling Spine Exists

The most important concepts are present:

- `ImageObservation` and `FieldObservation`
- `MapProductSpec` and `MapBandpassSpec`
- `TargetRenderSpec`
- `SceneRepresentation` and adapters for continuous fields, discrete target
  images, HEALPix maps, and harmonic coefficients
- `ObservationOperatorBundle` and explicit/learned operator slots
- `fit_field_model` and product rendering functions

These should be promoted and regularized rather than discarded.

## Major Findings

### 1. The Mission Concepts Are Not The Public Spine

For a library whose long-term goal is "model any observation of the sky jointly",
the first-class concepts should be:

- observation metadata and data access
- instrument/operator model
- scene/sky representation
- target product or prediction request
- objective/likelihood
- optimizer/solver
- validation and benchmarks

In the source, those concepts are scattered:

- observations live in both `map/image_plane.py` and `map/field/types.py`
- targets live in both `map/product.py` and `map/target.py`
- scenes live in `map/scenes.py` and are mostly internal
- operators live in `map/operators`, but the base classes are metadata dataclasses,
  not executable operator protocols
- fitting is a large function in `map/field/core.py`
- solver hooks live in `map/solvers.py` and are explicitly future-facing

This makes `torchsky` feel like a set of advanced kernels rather than an
observation modelling framework.

Recommendation: define a promoted modelling spine and make current pieces adapt
to it. Keep `torchsky.read`, `torchsky.table`, `torchsky.wcs`, and
`torchsky.sphere` stable, but make new users see modelling concepts first.

### 2. Root `__init__.py` Is Doing Too Much

`src/torchsky/__init__.py` is 3149 lines and mixes:

- package exports
- cache setup and cache invalidation
- C++/CFITSIO probing
- HDU autodetection
- image read policy
- table read fallback policy
- write, insert, replace, delete
- batch reads and subset reads
- `open`, `get_header`, `get_wcs`
- lazy namespace loading

This file is currently the source of truth for core FITS I/O. `torchsky.io`
wraps back to it, so the canonical package namespace and the implementation
namespace are inverted.

Impact:

- import-time side effects are harder to reason about
- I/O policy cannot be tested or evolved as a clean subsystem
- cache invalidation spans root, table, and C++ caches manually
- public API and backend implementation churn are coupled

Recommendation: move implementation into internal modules such as
`torchsky.io.fits`, `torchsky.io.cache`, and `torchsky.io.hdu_ops`. Leave
`torchsky.read`, `torchsky.write`, and friends as stable wrappers.

### 3. Namespace Aliases Add Cognitive Load

`torchsky.coords` re-exports from `torchsky.wcs`. `torchsky.geom` re-exports
from `torchsky.sphere`. `torchsky.io` re-exports root I/O functions. The docs
list all of these as public namespaces.

This creates several ways to find the same concept:

- spherical geometry appears under `sphere`, `sphere.geom`, and `geom`
- HEALPix appears under `wcs.healpix`, `sphere`, `coords`, and compatibility
  modules
- FITS reads appear under `torchsky.read`, `torchsky.io.read`,
  `torchsky.io.read_fast`, `torchsky.read_image`, and `torchsky._fastio.read`

Aliases can be useful after a design is stable, but here they make the design
look less settled.

Recommendation: pick one canonical home for each concept and mark other homes
as compatibility aliases. Do this in metadata and docs before removing anything.

### 4. HEALPix Belongs To Sphere, Not WCS

The 2228-line HEALPix implementation is in `src/torchsky/wcs/healpix.py`, then
re-exported by `torchsky.sphere`. Other modules import it from both sides.

HEALPix is a spherical pixelization and indexing system. It should live under
`sphere` as an implementation detail or submodule, with `wcs.healpix` as a
compatibility shim if needed.

Impact:

- the `wcs` package carries non-WCS responsibilities
- `sphere` depends on implementation living outside its boundary
- users cannot infer the canonical import path

Recommendation: migrate implementation to `torchsky.sphere.healpix` or
`torchsky.sphere.pixelization.healpix`, leave `torchsky.wcs.healpix` as a shim,
and update imports gradually.

### 5. "Spectral" Means Three Different Things

There are three spectral modules:

- `torchsky/spectral.py`: FITS spectral axes, spectra, cubes
- `torchsky/wcs/spectral.py`: WCS spectral coordinate transforms
- `torchsky/sphere/spectral.py`: spherical harmonic transforms

These are all legitimate, but the shared name makes discovery difficult.

Recommendation:

- keep `torchsky.spectral` for physical spectral axes and data cubes
- rename the public concept of `sphere/spectral.py` to `sphere.harmonic` or
  `sphere.sht` over time, while preserving imports
- keep `wcs.spectral` internal-ish or document it as WCS spectral transforms only

### 6. `torchsky.map` Is A Kitchen-Sink Public Namespace

`torchsky.map.__all__` exports about 150 symbols. It mixes:

- coadd APIs
- map product specs
- field model dataclasses
- 20+ field model classes
- low-level resampling kernels
- HEALPix/image/harmonic sampling functions
- solver and prior builders
- learned operator scaffolding
- spectral-budget heuristics
- training losses

This makes the namespace powerful but not lean. A new user cannot tell which
three things are the recommended path.

Recommendation:

- split public layers without breaking imports:
  - `torchsky.map` for stable product/coadd/render entry points
  - `torchsky.scene` for scene representations
  - `torchsky.obs` or `torchsky.observations` for observation metadata
  - `torchsky.ops` for operator contracts
  - `torchsky.fit` for optimizers/objectives
  - `torchsky.map.experimental` or `torchsky.models` for field model families
- keep existing `torchsky.map.*` names as aliases while the new shape matures

### 7. Field Models Are Too Exposed As A Zoo

`map/field/core.py` contains many architecture classes:

- SIREN/WIRE/Fourier/Gaussian
- spectral basis and learned spectral basis
- sky encoder MLP/resMLP/STRAINER-style
- hash-grid and K-plane variants
- feature fields and atlas variants
- component decomposition
- operator-constrained feature fields

There is a registry and a `RECOMMENDED_FIELD_MODEL_KINDS` list, which is good.
But the public API still exposes the entire zoo and string names such as
`sky_lambda_feature_field_hetero` and `strainer_sky_encoder_mlp`.

Impact:

- users must choose among research architecture names
- API stability gets tied to model experiments
- the CLI defaults to `sine_implicit`, which is unlikely to express the intended
  next-generation default

Recommendation:

- expose a small preset API:

```python
model = torchsky.scene.field("compact", domain=domain)
model = torchsky.scene.field("spectral", domain=domain)
model = torchsky.scene.field("survey-scale", domain=domain)
```

- keep architecture class imports in an expert or experimental namespace
- move training knobs into config dataclasses instead of a 60+ keyword function
- make the recommended registry the basis for docs, CLI, and examples

### 8. Operator Abstraction Is Promising But Too Thin

`map/operators/base.py` defines operator families as dataclasses with metadata:
`GeometryOp`, `SpectralResponseOp`, `PSFOp`, `RadiometryOp`, `BackgroundOp`,
`MaskWeightOp`, and `NoiseOp`. `ObservationOperatorBundle` composes slots and
learned scaffolds can replace explicit slots.

This is the right conceptual split, but today these operators mostly declare
names and conditioning. Execution still happens through local branching in
`image_plane.py`, helper functions in `operators/execution.py`, and field render
helpers.

Recommendation: define executable operator protocols:

```python
class ObservationOperator:
    def forward(self, scene, observation, target): ...
    def adjoint(self, residual, observation, scene_state): ...
    def log_prob(self, predicted, observed, observation): ...
```

Not every operator needs every method on day one. But the protocol should make
explicit what can be differentiated, what has an adjoint, what is stochastic, and
what is learned.

### 9. Performance Work Is Real But Policy Is Entangled

The source contains serious performance engineering:

- WCS fast paths, C++ thresholds, and vectorized tensor paths
- C++ FITS and table readers
- table reader caches and row scanners
- HEALPix kernels and perf gates
- image-plane tiling for large targets
- map compatibility and benchmark surfaces

The concern is not absence of optimization. The concern is that policy,
heuristics, cache behavior, and execution are mixed into large functions.

Examples:

- root image I/O has mmap auto-selection, handle caching, raw/scale policy, table
  fallback, and batch fallback in one module
- table reads choose among `cpp_numpy`, torch row paths, Arrow filtering, and C++
  predicate pushdown inside `table.read`
- image-plane coadds build whole target buffers unless a spatial tiling option is
  explicitly selected
- operator execution is not yet represented as a reusable plan with memory and
  device budgets

Recommendation:

- introduce explicit execution plans:
  - read plan
  - projection plan
  - render plan
  - fit plan
- attach benchmark evidence and crossover thresholds to the plan/policy layer
- make memory budget, device, dtype, chunking, and cache policy inspectable
- keep exact tensor fallback paths separate from heuristic fast-path selection

### 10. Survey-Scale Data Loading Is Underdeveloped

`FITSDataset` is useful for small lists of files. `TableChunkDataset` is closer to
the survey-scale direction. But `IterableFITSDataset` is explicitly a simplified
placeholder that fabricates shard file names from an index URL.

For the stated survey-scale mission, the data layer needs first-class manifests,
partitioning, locality, streaming retries, remote object stores, and metadata
selection. This should not be left as a PyTorch `DataLoader` convenience wrapper.

Recommendation:

- build a real `ObservationManifest` / `ObservationSet` API
- support Arrow/Parquet/catalog metadata as the manifest substrate
- make file-open policy, chunk scheduling, and device transfer explicit
- keep PyTorch datasets as adapters over that substrate

### 11. Cache And Buffer Managers Are Not Integrated Enough

`cache.py` and `buffer.py` define useful managers, but source search shows buffer
managers are mostly tested directly rather than used by hot map/I/O paths.
Root I/O, `table.py`, and C++ have their own caches.

Recommendation:

- centralize cache policy around named subsystems: FITS image, FITS table,
  metadata/header, WCS grid, operator/render tiles
- expose a single stats/report API
- use buffer pools only where ownership and reuse are clear, not as a generic
  global promise

### 12. The CLI Is Not Yet A Clean Workflow Surface

The CLI has two commands: `viz` and `map`. The `map` command has 40 parser
arguments and combines coadd and INR fitting. The INR path defaults to SVO
bandpass resolution, which requires an optional ecosystem package and possible
network access.

This is not the ergonomic command-line shape for a library whose user API should
be clean and easy to adopt.

Recommendation:

```text
torchsky inspect ...
torchsky coadd ...
torchsky fit ...
torchsky render ...
torchsky validate ...
torchsky benchmark ...
torchsky viz ...
```

or, if keeping the `map` umbrella:

```text
torchsky map coadd ...
torchsky map fit-field ...
torchsky map render ...
torchsky map validate ...
```

Each command should have a short common path and an expert config-file path.

### 13. Public API Tiers Exist But Are Too Sparse

`src/torchsky/surfaces.py` is a good start. It declares core, compat, and
experimental surfaces. But it only covers selected entry points and does not
fully constrain what `__all__` exports.

Recommendation:

- expand API surface metadata to cover all public namespaces
- use it in docs generation and release checks
- require every new export to declare one of: core, compat, experimental,
  internal alias
- add a test that flags unclassified exported symbols

### 14. Some Object Model Choices Leak Implementation Details

Examples:

- `TableHDU` creates dummy tensor-frame columns when no compatible columns exist.
- `TensorHDU.to_tensor()` returns a dummy zero image when no data/file handle is
  available.
- `FieldObservation` and `ImageObservation` duplicate bandpass, background, and
  weight concepts with different data ownership models.
- `ForwardContext` exists only for field models and is passed only when forward
  signatures accept it.

Recommendation:

- fail closed instead of returning dummy data in core objects
- define one observation metadata model with image-backed and tensor-backed data
  adapters
- make context a first-class part of observation/operator execution, not a
  signature-inspection side channel

## Proposed Target Architecture

The target should preserve the current strengths while making the modelling spine
obvious.

```text
torchsky
  io
    FITS, HDU, table, Arrow, manifests, chunked readers
  coords
    WCS, frames, spectral/temporal axes, sphere, pixelizations
  obs
    Observation, ObservationSet, Survey, Exposure, Bandpass, PSF, Noise, Mask
  scene
    explicit maps, source catalogs, continuous fields, components, priors
  ops
    geometry, PSF/beam, bandpass, radiometry, background, mask, noise
  products
    Target, ProductSpec, render specs, validation metrics
  fit
    objectives, optimizers, solver configs, diagnostics
  compat
    healpy/hpgeom/astropy-shaped adapters
  viz
    optional browser/server surfaces
```

This does not require moving all files immediately. It requires creating
canonical public facades and then moving implementation behind them.

## Desired User API Shape

A clean high-level API should look like this:

```python
import torchsky as ts

obs = ts.ObservationSet.from_fits(
    paths,
    bandpass="header",
    psf="header",
    noise="weight",
)

target = ts.Product.grid_like(obs[0], bands="union")

scene = ts.Scene.field(
    preset="spectral",
    components=["sky", "background"],
)

fit = ts.fit(
    scene,
    obs,
    target=target,
    objective="gaussian",
    optimizer="adamw",
    steps=2000,
)

product = ts.render(fit.scene, target)
```

Expert users should still be able to do this:

```python
from torchsky.map import ImageObservation, coadd_image_plane
from torchsky.sphere import map2alm, alm2map
from torchsky.wcs import WCS
```

The point is not to remove low-level power. The point is to stop forcing new
users to start with low-level power.

## Naming Recommendations

Use short public nouns for stable concepts:

- `Observation`
- `ObservationSet`
- `Bandpass`
- `PSF`
- `Noise`
- `Scene`
- `Component`
- `Operator`
- `Target`
- `Product`
- `Objective`
- `FitResult`
- `Solver`

Use parameterized functions for algorithm families:

- prefer `resample(..., mode="lanczos3")` over many exported resampler names
- prefer `scene.field(preset="hashgrid")` over exposing every model class to
  most users
- prefer `operators.geometry(kind="drizzle")` over long class names in common
  workflows

Keep long technical names available in expert namespaces where they are useful
for benchmarks and papers.

## Suggested Migration Plan

### Phase 0: Declare The Spine

Duration: days.

- Write one source-aligned architecture note that names the canonical concepts:
  Observation, Operator, Scene, Target/Product, Objective, Fit/Render.
- Expand `APISurface` metadata so exports can be classified.
- Decide canonical homes:
  - FITS/image/table I/O: `torchsky.io` and `torchsky.table`
  - WCS transforms: `torchsky.wcs`
  - spherical geometry/pixelizations/harmonics: `torchsky.sphere`
  - modelling spine: new promoted facades

### Phase 1: Extract Root I/O Without User Churn

Duration: 1 to 2 weeks.

- Move root FITS read/write implementation into `torchsky.io.fits`.
- Move root Python-side cache helpers into `torchsky.io.cache`.
- Keep `torchsky.read`, `torchsky.write`, `torchsky.open`, `torchsky.get_header`,
  and `torchsky.get_wcs` as wrappers.
- Make `torchsky.io` import the implementation directly, not through root.
- Add regression tests proving public behavior and cache invalidation are
  unchanged.

This phase reduces risk without changing user-facing imports.

### Phase 2: Promote Observation/Product/Scene Facades

Duration: 2 to 4 weeks.

- Add `Observation`, `ObservationSet`, `Bandpass`, `PSF`, and `Noise` facades.
- Make `ImageObservation` and `FieldObservation` adapters or aliases where
  possible.
- Promote `TargetRenderSpec` and `MapProductSpec` into a clearer `Product` /
  `Target` public path.
- Promote `SceneRepresentation` concepts into `torchsky.scene`.
- Keep existing `torchsky.map` imports working.

The goal is a clean high-level path while leaving the current map code intact.

### Phase 3: Make Operators Executable

Duration: 4 to 8 weeks.

- Define operator protocols for forward, optional adjoint, optional log-likelihood,
  and conditioning.
- Convert existing geometry, spectral, PSF, radiometry, background, mask, and
  noise slots from metadata-only classes into thin executable adapters.
- Keep low-level image-plane helper functions as kernels under the operator
  implementations.
- Add tests that a scene rendered through an operator bundle matches existing
  `coadd_image_plane` and `render_field_observation` behavior for small cases.

### Phase 4: Split Monoliths Internally

Duration: opportunistic, after facades exist.

- Split `map/field/core.py` into model families, rendering, losses, training,
  presets, and checkpoint/pretrain helpers.
- Split `map/image_plane.py` into planning, target preparation, kernels,
  accumulation, coadd execution, and cube/band projection.
- Split `wcs/core.py` into parsing, projection dispatch, public WCS object,
  slicing, and compatibility methods if this can be done without hurting hot
  paths.
- Split `table.py` into scan/read, Arrow conversion, mutation, and backend policy.
- Split C++ bindings by domain if nanobind setup permits it.

Keep public imports stable while doing this.

### Phase 5: Rationalize Field Models

Duration: 2 to 4 weeks for the API layer, longer for research cleanup.

- Create stable presets and config dataclasses.
- Move experimental model classes behind an expert namespace.
- Make `RECOMMENDED_FIELD_MODEL_KINDS` drive CLI defaults and docs.
- Replace giant training keyword surfaces with `FitConfig`, `SamplingConfig`,
  `LossConfig`, and `OptimizerConfig`.
- Add a minimum real-data smoke for each recommended preset.

### Phase 6: Build Survey-Scale Data And Execution Plans

Duration: longer-term.

- Replace placeholder streaming datasets with real manifests.
- Add chunked `ObservationSet` iteration with locality and worker partitioning.
- Add render/projection/fit plan objects with memory budget, device, dtype,
  chunking, and cache policy.
- Add plan explainers: users should be able to print why a kernel, chunk size,
  or backend was selected.
- Tie plan policies to benchmark artifacts and perf gates.

### Phase 7: Redesign CLI Around Workflows

Duration: after facades are stable.

- Split `torchsky map` into subcommands or top-level workflow commands.
- Support config files for advanced fitting and rendering.
- Make optional network/ecosystem requirements explicit.
- Keep the common path short:

```bash
torchsky coadd out.fits in/*.fits --reference ref.fits
torchsky fit config.toml
torchsky render model.pt product.toml out.fits
```

## Near-Term Action List

1. Create `torchsky.io.fits` and move root I/O implementation behind wrappers.
2. Add API-surface classification tests for exported symbols.
3. Add a new high-level `ObservationSet` facade that can wrap current
   `ImageObservation` and `FieldObservation`.
4. Promote `TargetRenderSpec` and `SceneRepresentation` into stable or
   experimental public facades.
5. Move HEALPix implementation ownership toward `torchsky.sphere`.
6. Define field model presets and switch the CLI default away from a raw
   architecture string.
7. Replace `IterableFITSDataset` placeholder behavior with a manifest-backed
   implementation or mark it explicitly experimental and incomplete.

## What I Would Not Do First

- Do not remove `torchsky.read`, `torchsky.table`, `torchsky.wcs.WCS`, or
  `torchsky.sphere` imports. They are useful and already tested.
- Do not start with a sweeping directory rename. Add clean facades first.
- Do not hide all low-level kernels. They are valuable for benchmarks, parity,
  and expert workflows.
- Do not make learned world-model research the core API before the observation
  operator contracts are executable and testable.

## Bottom Line

The technical foundation is much stronger than the public architecture suggests.
The main problem is not missing ambition in the source. It is that the source
exposes implementation history instead of the future modelling story.

The project should keep its native FITS/WCS/sphere/map strengths, but make the
observation-modelling spine the first thing users and contributors see:

```text
observations + operators + scenes + products + objectives + fit/render
```

That is the architecture that can grow into the backbone for joint, differentiable,
any-device modelling of the observable sky.
