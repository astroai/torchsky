# Map / INR Operator Refactor Progress

This note tracks the current state of the "shared observation operators for exact mapmaking and INR reuse" program in `torchsky.map`.

It is intentionally blunt:
- what is actually implemented
- what is only scaffolded
- what is still missing for the scientific end state

## Scope

The intended end state is:
- one shared observation-model substrate for image-plane coadds, exact remappers, INR / field models, and future inverse solvers
- native torchsky implementations for the core mapmaking operators
- first-class background and learned-operator slots
- explicit separation between latent astrophysical / cosmological structure and survey / instrument / detector effects

## Current Status

Short version:

- the internal substrate is real
- exact native drizzle is real
- spherical forward and inverse operator families are real
- multiband / TEB / mixed-constraint spherical solver machinery is real
- the first explicit shared-vs-private decomposition machinery is real
- the first learned survey / instrument decomposition machinery is real
- the full latent-scene / learned-operator science program is **not** complete

## What Is Implemented

### 1. Shared internal operator substrate

Implemented under:
- [src/torchsky/map/operators/__init__.py](../src/torchsky/map/operators/__init__.py)
- [src/torchsky/map/operators/base.py](../src/torchsky/map/operators/base.py)
- [src/torchsky/map/operators/bundle.py](../src/torchsky/map/operators/bundle.py)
- [src/torchsky/map/scenes.py](../src/torchsky/map/scenes.py)
- [src/torchsky/map/target.py](../src/torchsky/map/target.py)
- [src/torchsky/map/solvers.py](../src/torchsky/map/solvers.py)

Implemented concepts:
- `ObservationOperatorBundle`
- `GeometryOp`, `SpectralResponseOp`, `PSFOp`, `RadiometryOp`, `BackgroundOp`, `MaskWeightOp`, `NoiseOp`
- `SceneRepresentation`
- `DiscreteTargetImage`
- `ContinuousFieldModel`
- target-bound operator interfaces and solver hooks

This is no longer just a design note. The internal bundle / scene / target contract exists and is exercised in tests.

### 2. Background is first-class

Implemented under:
- [src/torchsky/map/operators/background.py](../src/torchsky/map/operators/background.py)
- [src/torchsky/map/field/types.py](../src/torchsky/map/field/types.py)
- [src/torchsky/map/field/observation_builders.py](../src/torchsky/map/field/observation_builders.py)

Implemented capabilities:
- explicit sky / instrument / detector background surfaces
- `background_offset` retained only as a compatibility shim
- shared ingestion and application of structured background components
- background-aware tests and builder propagation

### 3. Native geometry families

Implemented under:
- [src/torchsky/map/operators/geometry_drizzle.py](../src/torchsky/map/operators/geometry_drizzle.py)
- [src/torchsky/map/operators/geometry_conservative.py](../src/torchsky/map/operators/geometry_conservative.py)
- [src/torchsky/map/operators/geometry_fft.py](../src/torchsky/map/operators/geometry_fft.py)
- [src/torchsky/map/operators/geometry_interp.py](../src/torchsky/map/operators/geometry_interp.py)
- [src/torchsky/map/image_plane.py](../src/torchsky/map/image_plane.py)

Implemented capabilities:
- exact native drizzle
- native conservative polygon overlap remap
- native interpolation family
- restricted FFT / affine families
- explicit `splat_*` naming for approximate forward splats

### 4. Native spectral remappers

Implemented under:
- [src/torchsky/map/spectral_affine.py](../src/torchsky/map/spectral_affine.py)
- [src/torchsky/map/spectral_nonuniform.py](../src/torchsky/map/spectral_nonuniform.py)
- [src/torchsky/map/spectral_nufft.py](../src/torchsky/map/spectral_nufft.py)

Implemented capabilities:
- `spectral_linear_affine`
- direct nonuniform spectral remap
- fast planar NUFFT-style approximation

This closes a meaningful part of the old FFT / DFT reprojection gap, but not the whole thing.

### 5. Native spherical spectral family

Implemented under:
- [src/torchsky/map/healpix_spectral.py](../src/torchsky/map/healpix_spectral.py)
- [src/torchsky/sphere/spectral.py](../src/torchsky/sphere/spectral.py)
- [src/torchsky/map/healpix_image_plane.py](../src/torchsky/map/healpix_image_plane.py)

Implemented capabilities:
- harmonic HEALPix rebandlimit / filtering
- direct scalar harmonic evaluation on arbitrary sky coordinates
- direct spin-weighted harmonic evaluation on arbitrary sky coordinates
- exact bilinear HEALPix bridge adjoint
- scalar harmonic image-to-`alm` adjoint
- spin-harmonic image-to-`(E,B)` adjoint

### 6. Spherical inverse solvers

Implemented under:
- [src/torchsky/map/solvers.py](../src/torchsky/map/solvers.py)
- examples under [examples/](../examples)

Implemented capabilities:
- HEALPix inverse solves
- scalar harmonic inverse solves
- spin-harmonic inverse solves
- multi-cutout shared-state solves
- CG / PCG on normal equations
- diagonal probing and diagonal preconditioning
- mixed operator blocks on one latent state

### 7. Prior vocabulary for spherical states

Implemented capabilities:
- identity / anchor operators
- diagonal harmonic priors
- dense covariance priors
- low-rank template priors
- Kronecker priors
- `C_ell` priors
- joint `T/E/B` cross-spectrum priors

This is implemented in [src/torchsky/map/solvers.py](../src/torchsky/map/solvers.py) and covered heavily in [tests/test_map_solvers.py](../tests/test_map_solvers.py).

### 8. Product / benchmark integration

Implemented under:
- [src/torchsky/map/product.py](../src/torchsky/map/product.py)
- [benchmarks/bench_map_ground_truth.py](../benchmarks/bench_map_ground_truth.py)
- [benchmarks/bench_map_multifield_truth.py](../benchmarks/bench_map_multifield_truth.py)

Implemented benchmark rows now include:
- image-plane coadd baselines
- field-fit / INR rows
- spherical reprojection rows
- joint spherical inverse rows
- multiband spherical inverse rows

### 9. Shared-vs-private decomposition

Implemented under:
- [src/torchsky/map/solvers.py](../src/torchsky/map/solvers.py)

Implemented capabilities:
- `build_state_component_mixture_operator(...)`
- explicit shared / survey-private / instrument-private latent state mixtures
- adjoint-correct decomposition operators
- leakage-reduction regression tests

This is the first actual mechanistic split between:
- shared astrophysical field
- survey-private residual structure

instead of only fitting one shared field.

### 10. Learned decomposition on top of the explicit split

Implemented under:
- [src/torchsky/map/solvers.py](../src/torchsky/map/solvers.py)
- [examples/example_map_harmonic_learned_survey_instrument_split.py](../examples/example_map_harmonic_learned_survey_instrument_split.py)

Implemented capabilities:
- `SurveyInstrumentComponentMixer`
- `build_conditioned_component_mixture_operator(...)`
- `HarmonicComponentDecomposition`
- learned context weights over explicit shared / survey / instrument / interaction component roles

This is the first version of "mechanistic splitting with learned parameterization" in the repo.

## What Is Still Only Partially Done

### 1. Full execution unification across all map paths

Partly done:
- product layer and many field paths use the shared substrate
- spherical solver / render paths use the shared substrate
- image-plane coadds now share observation preparation, geometry dispatch, stack accumulation, and stack finalization helpers under [src/torchsky/map/operators/execution.py](../src/torchsky/map/operators/execution.py)

Still incomplete:
- `coadd_image_plane(...)` and `coadd_image_plane_cube(...)` still retain target planning, windowing, coverage/MOC assembly, and I/O policy in [src/torchsky/map/image_plane.py](../src/torchsky/map/image_plane.py)
- operator classes are not yet a fully uniform execution layer everywhere

### 2. Learned operators as real scientific models

Partly done:
- learned operator slots and registries exist
- learned decomposition over component roles exists

Still incomplete:
- no real learned geometry model
- no real learned PSF field
- no real learned spectral-response model
- no real learned noise model
- no first-class learned detector-systematics model

### 3. Latent component decomposition beyond harmonic toy structure

Partly done:
- harmonic shared/private decomposition works
- learned survey/instrument mixing over harmonic components works

Still incomplete:
- a first train-time multicomponent field-model decomposition workflow now exists via `make_field_component_decomposition(...)` and `fit_field_component_decomposition(...)`, alongside the explicit `FieldComponentDecomposition` wrapper and background-stripped observation-space diagnostics
- no explicit survey-private versus instrument-private component families in INR space
- no joint diffuse + compact + nuisance decomposition model yet

## What Is Not Done Yet

### 1. Full scientific identifiability framework

Missing:
- stronger gauge controls for learned decomposition beyond the first exact component-scale gauge canonicalization now wired into both `HarmonicComponentDecomposition` and `FieldComponentDecomposition`
- richer anchor surfaces for standards / astrometry / calibration transfer beyond the first solve-time field anchor observation family, which now includes both `FieldComponentAnchorTerm` for rendered image / direct sample / WCS-resolved pixel anchors and `FieldCentroidAnchorTerm` for geometry-aware role centroid constraints
- robust leakage diagnostics across real survey combinations beyond the first harmonic diagnostics now available for shared-to-private leakage, private-to-shared leakage, and calibration-anchor stability

The current learned split examples already show why this matters: without gauge fixing, shared amplitude and nuisance weights can trade off.

### 2. Full survey / instrument nuisance modeling

Missing:
- learned nuisance fields, not only learned mixture weights
- detector-side learned systematics surfaces
- time-dependent nuisance modeling
- spatially varying throughput / transfer residual models
- beam / PSF mismatch learned jointly with latent components

### 3. Full package-level replacement claims

Not yet honest to claim:
- full SWarp replacement
- full Montage replacement
- full `reproject` family replacement
- full DrizzlePac pipeline replacement

Honest current claim:
- core native scientific operators exist for much of the useful mapmaking surface

### 4. Basis-independent decomposition layer

Still missing:
- wavelet / needlet / multiscale sparse component decompositions
- hybrid harmonic + pixel + template latent states as a first-class API
- direct component-separation abstractions independent of spherical harmonics

Right now the decomposition work is still harmonic-centered, even though the operator pattern is broader.

### 5. Real joint survey science surfaces

Still missing:
- explicit cross-survey decomposition benchmarks for realistic combinations such as SPHEREx + WISE + GALEX + CMB-like diffuse components
- real-data regression surfaces for shared-vs-private decomposition quality
- metrics that quantify contamination transfer across surveys and instruments

## Best Current Interpretation

The refactor is no longer "mostly planned". It is at the stage where:

- the internal architecture is real
- exact native mapmaking kernels are real
- spherical forward / inverse machinery is real
- the first decomposition machinery is real
- the first learned mechanistic split is real

But the full scientific program is still incomplete. The main remaining gap is not basic plumbing anymore. It is the hard science layer:

- stronger identifiability controls
- richer nuisance parameterizations
- realistic cross-survey decomposition
- learned operator families beyond context weights

## Recommended Next Steps

The highest-value next steps are:

1. Extend `HarmonicComponentDecomposition` into real learned nuisance component fields instead of only learned mixture weights.
2. Add explicit survey-private / instrument-private component benchmarks on synthetic multiband truth surfaces.
3. Add identifiability metrics and regression tests:
   - shared-to-private leakage
   - private-to-shared leakage
   - calibration transfer stability
4. Lift the decomposition pattern from harmonic states into INR / field representations.
5. Add real-data multi-survey experiments where the decomposition is scientifically meaningful.

## Related Files

Useful entry points:
- [src/torchsky/map/solvers.py](../src/torchsky/map/solvers.py)
- [src/torchsky/map/scenes.py](../src/torchsky/map/scenes.py)
- [src/torchsky/map/operators/](../src/torchsky/map/operators)
- [benchmarks/bench_map_ground_truth.py](../benchmarks/bench_map_ground_truth.py)
- [benchmarks/bench_map_multifield_truth.py](../benchmarks/bench_map_multifield_truth.py)
- [tests/test_map_solvers.py](../tests/test_map_solvers.py)
- [docs/mapping_parity.md](mapping_parity.md)
- [docs/examples.md](examples.md)
