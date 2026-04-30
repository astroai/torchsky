# Implicit SkyMeasure Scene Implementation Plan

## Purpose

This note lays out an implementation plan for an **implicit-first, measure-valued, multiscale sky scene** in `torchsky`: a latent sky representation designed to explain many exposures directly, rather than first collapsing them into a coadd.

The goal is an archive-native "super-stack" model that can represent point-like structure, compact structure, extended galaxies, diffuse emission, ultra-large-scale low-surface-brightness structure, unresolved populations, and time-variable or moving structure **without requiring a hard, explicit source decomposition as the primary abstraction**.

The guiding principle is:

> learn a shared latent sky state that is rendered through exposure-specific WCS, PSF, bandpass, detector, background, mask, and noise operators; introduce explicit components only as optional inductive biases, accelerators, diagnostics, or constraints.

This is broader than a pure INR, Gaussian splat model, catalog model, or coadd. Those become possible parameterizations of the latent sky state, not mandatory source taxonomies.

## Design thesis

The latent sky should be treated as a measure over direction, wavelength, and time:

$$
\mu_\theta(d\omega, d\lambda, dt), \qquad \omega=(\alpha,\delta).
$$

An exposure observes this latent scene through an exposure operator:

$$
Y_e[p] \sim \mathbb{P}_{\psi_e}\left(\cdot\mid
\int R_{e,p}(\omega,\lambda,t)\,\mu_\theta(d\omega,d\lambda,dt) + B_e[p]
\right),
$$

where $R_{e,p}$ includes WCS sampling, PSF, bandpass, exposure time, detector response, and masking, while $B_e$ contains exposure-side background and nuisance terms.

The implementation should start from the most implicit useful representation:

$$
\mu_\theta(d\omega,d\lambda,dt)
= f_\theta(\omega,\lambda,t)\,d\omega d\lambda dt
+ \mu^{\rm sparse}_\theta(d\omega,d\lambda,dt),
$$

where $f_\theta$ is a continuous latent field and $\mu^{\rm sparse}_\theta$ is optional. The sparse term should not initially be interpreted as a fixed catalog of stars/galaxies. It is an adaptive latent measure for features that are inefficient for a continuous field to represent. It may later be regularized, inspected, or converted into catalog-like summaries, but the core fit should not depend on hard semantic labels.

## Key correction: implicit first, decomposition optional

The previous version of this plan described named components such as point sources, diffuse fields, extended primitives, unresolved populations, and transients. Those are useful scientific interpretations and possible implementation modules, but they should not be the starting assumption.

A better hierarchy is:

1. **Implicit latent sky state:** the primary object being inferred.
2. **Multiscale parameterization:** the numerical basis used to represent that state.
3. **Optional adaptive atoms/primitives:** local capacity added where the implicit field is inefficient.
4. **Optional semantic views:** catalog-like or component-like interpretations derived from the fitted state.

In other words, do not begin by asking whether a photon came from "diffuse", "point", or "extended" source classes. Begin by asking whether the latent sky state, when rendered through all exposure operators, predicts the detector data with calibrated residuals and sensible uncertainty.

## Non-goals

- Do not implement a full graphics-style Power Foam or NeRF renderer.
- Do not make exact weighted Voronoi or power-diagram construction a dependency of the first milestone.
- Do not force a hard astrophysical source taxonomy into the core API.
- Do not require every fitted structure to be assigned to a named source class.
- Do not add heavy astronomy ecosystem packages to the runtime dependency set.
- Do not require `astropy`, `healpy`, `fitsio`, `pandas`, or similar packages in core import paths.
- Do not replace the existing `torchsky.map.field` path; build a scene abstraction that can reuse it.
- Do not bake detector artifacts, PSF wings, chip defects, or cosmic rays into the persistent sky representation unless explicitly modeled as sky-persistent features.

## Representation levels

### Level 1: pure implicit field

The simplest scene is a continuous latent sky field:

$$
I_\theta(\omega,\lambda,t)=F_\theta(\omega,\lambda,t).
$$

This should be the first-class path. The field can be an MLP, SIREN-like INR, multiresolution grid, hash/grid encoding, spherical encoder, spectral encoder, or pretrained trunk. It should render through the same exposure operator as every later representation.

This level tests the fundamental super-stack idea:

$$
\min_\theta \sum_e \mathcal{L}\left(Y_e,\;\mathcal{R}_e[F_\theta]\right),
$$

where $\mathcal{R}_e$ is the exposure-specific rendering operator.

### Level 2: multiscale implicit field

A single INR may be too globally coupled. A stronger implicit formulation is a multiscale latent field:

$$
I_\theta(\omega,\lambda,t)
=
F_\theta\left(
\gamma_{\rm sky}(\omega),
\gamma_\lambda(\lambda),
\gamma_t(t),
Z_0(\omega), Z_1(\omega), \ldots, Z_L(\omega)
\right),
$$

where $Z_l$ are learnable latent grids, needlet-like maps, HEALPix-like levels, or patch-local latent tensors. This keeps the representation implicit while making scale separation computationally tractable.

This level should handle much of what might otherwise be assigned to "diffuse" or "extended" components, without naming those components explicitly.

### Level 3: adaptive latent measure

Some structures are inefficient for a continuous field: very sharp cores, undersampled point-like features, blends, cosmic-ray-like residuals, or small high-frequency structures. Add an optional adaptive measure:

$$
\mu^{\rm sparse}_\theta = \sum_i a_i(\lambda,t)\,\phi_i(d\omega;\eta_i).
$$

The atoms $\phi_i$ should initially be treated as **adaptive basis functions**, not physical source labels. They can be compact kernels, bounded primitives, soft power-cell atoms, or learned local patches.

This term is useful if it improves optimization, residuals, and scalability. It should not be required for the architecture to make sense.

### Level 4: semantic projections

After fitting, optional derived views can be computed:

- catalog-like summaries;
- candidate point-source lists;
- extended-object footprints;
- LSB residual maps;
- transient or moving-source candidates;
- uncertainty and coverage diagnostics.

These are outputs or diagnostics, not the primary latent representation.

## Proposed package structure

Initial experimental namespace:

```text
torchsky.map.scene
  base.py              # implicit scene/component protocols and common dataclasses
  context.py           # exposure rendering context
  render.py            # joint scene rendering orchestration
  losses.py            # weighted exposure losses and diagnostics
  implicit.py          # implicit-first scene wrappers around existing field models
  multiscale.py        # latent grids / multiresolution implicit fields
  adaptive.py          # optional untyped adaptive atoms/primitives
  diagnostics.py       # semantic projections and residual analysis
```

Optional later submodules can provide named scientific interpretations, but the core namespace should not require them:

```text
torchsky.map.scene.views
  catalogs.py          # catalog-like summaries from latent states
  residuals.py         # LSB/confusion/transient diagnostics
  components.py        # optional semantic decomposition utilities
```

## Core abstractions

### `ImplicitSkyScene`

A scene is a latent model plus rendering and regularization hooks:

```python
class ImplicitSkyScene(nn.Module):
    def render_observation(self, observation: FieldObservation, **kw) -> Tensor: ...
    def evaluate_sky(self, ra_deg: Tensor, dec_deg: Tensor, wavelength: Tensor | float, **kw) -> Tensor: ...
    def regularization_terms(self) -> dict[str, Tensor]: ...
    def diagnostics(self) -> dict[str, object]: ...
```

The scene should support a pure field-only implementation and a hybrid implementation. The public interface should not require the caller to know whether the scene internally uses an INR, latent grid, compact atoms, or a mixture.

### `SkyLatentModule`

A lower-level module represents part of the latent state. It is not necessarily a semantic source component:

```python
class SkyLatentModule(nn.Module):
    def render_observation(self, observation: FieldObservation, **kw) -> Tensor: ...
    def regularization_terms(self) -> dict[str, Tensor]: ...
```

Examples:

- continuous INR branch;
- multiresolution latent-grid branch;
- adaptive compact atom branch;
- exposure-persistent residual branch;
- time-dependent latent branch.

### `ExposureRenderContext`

Introduce a small dataclass that carries rendering options not intrinsic to `FieldObservation`:

```python
@dataclass(frozen=True)
class ExposureRenderContext:
    num_spectral_samples: int = 8
    spectral_sampling: str = "quadrature"
    tile_size: int | None = None
    psf_mode: str = "observation"
    include_background: bool = True
    return_latent_terms: bool = False
```

This should initially wrap existing `render_field_observation` options rather than duplicate them.

## Initial implementation path

### 1. Field-only scene wrapper

Wrap the current field models as an implicit scene:

$$
I_\theta(\omega,\lambda)=F_\theta(\omega,\lambda).
$$

Responsibilities:

- reuse existing `torchsky.map.field` models and `render_field_observation`;
- support multi-band rendering and existing bandpass quadrature;
- expose a scene-level loss over many `FieldObservation` objects;
- preserve exact equivalence with the existing field rendering path for one observation.

First implementation: `FieldSkyScene(model)`.

### 2. Multiscale implicit scene

Add learnable latent grids or patch-local latent tensors feeding a compact decoder:

$$
I_\theta(\omega,\lambda)
= D_\theta\left(E(\omega,\lambda), Z_0(\omega), \ldots, Z_L(\omega)\right).
$$

This is the preferred way to represent both smooth and moderately high-frequency structure without forcing source labels.

Implementation options:

- tangent-plane multiresolution grids for small patches;
- sphere-aware or HEALPix-like levels for larger areas;
- spectral basis coefficients per spatial latent;
- pretrained trunk plus trainable local latent state.

### 3. Adaptive latent atoms

Add an optional untyped sparse term:

$$
I_\theta(\omega,\lambda)
= I^{\rm implicit}_\theta(\omega,\lambda)
+ \sum_i a_i(\lambda)\,\phi_i(\omega;\eta_i).
$$

These atoms are numerical basis functions. They should not initially be called stars, galaxies, or artifacts in the core API.

Initial kernels:

1. compact circular Gaussian;
2. compact elliptical Gaussian;
3. compact Moffat-like profile;
4. optional soft power-ownership weights for overlapping atoms.

Soft power ownership, if added, can use:

$$
q_i(x)=\|x-p_i\|^2-r_i^2,
$$

$$
o_i(x)=\frac{\exp[-q_i(x)/\tau]}{\sum_{j\in\mathcal{N}(x)}\exp[-q_j(x)/\tau]}.
$$

This gives a differentiable approximation to bounded power-cell ownership without implementing exact clipped power diagrams.

### 4. Semantic views and diagnostics

Only after the implicit and adaptive representations work, add derived views:

- compact-candidate extraction;
- persistent-vs-exposure-tied residual maps;
- catalog-like summaries;
- local morphology summaries;
- LSB residual maps;
- transient/motion candidate maps.

These should be diagnostic layers, not prerequisites for fitting.

## Rendering strategy

### Dense implicit rendering

Implicit fields and multiscale latent grids can reuse `render_field_observation` or a scene-level generalization of it.

### Sparse adaptive rendering

Adaptive atoms should render sparsely:

1. project atom centers into exposure pixel coordinates using `WCS.world_to_pixel`;
2. compute finite support radius in pixels;
3. build atom-to-tile or atom-to-pixel candidate lists;
4. evaluate kernels only inside local footprints;
5. accumulate with differentiable torch operations;
6. convolve with PSF only if the atom is not already rendered through an analytic PSF path.

Initial implementation may use simple vectorized chunking before native kernels. Add C++/CUDA only after profiling proves the bottleneck.

### PSF policy

For the first prototype:

- keep the persistent sky state PSF-free;
- apply PSF as part of observation rendering;
- use existing scalar Gaussian PSF support where available;
- document when an analytic sparse path folds the Gaussian PSF into the footprint for efficiency.

Later:

- support per-exposure spatially varying PSF specifications;
- support chromatic PSF effects;
- add parity tests against brute-force convolution for small fixtures.

## Losses and regularization

### Observation loss

Start with weighted Charbonnier or Gaussian negative log-likelihood:

$$
\mathcal{L}_{\rm obs} = \sum_e \sum_p w_{e,p}\,\rho(Y_e[p]-\hat{Y}_e[p]).
$$

Reuse existing mask and weight semantics from `FieldObservation`.

### Representation regularization

Regularize numerical behavior rather than semantic source labels:

- multiscale smoothness by level;
- spectral smoothness or low-rank spectral structure;
- total variation or Sobolev-like penalties on selected latent levels;
- atom compactness and finite-support penalties;
- atom sparsity only if an adaptive atom branch is enabled;
- overlap or redundancy penalties for atoms;
- exposure-nuisance smoothness;
- temporal smoothness for dynamic latent branches later.

Do not hide strong priors inside rendering code. Keep them explicit in `regularization_terms()`.

## Milestones

### Milestone 0: design skeleton and documentation

Deliverables:

- this plan;
- `torchsky.map.scene` placeholder package;
- base protocols/classes;
- no public API stability promise yet;
- one small synthetic example in docs or examples.

Success criteria:

- imports cleanly on CPU without optional dependencies;
- no changes to existing field rendering behavior.

### Milestone 1: field-only implicit scene

Deliverables:

- `FieldSkyScene` wrapping an existing field model;
- `render_scene_observation(scene, observation)`;
- scene loss over a list of `FieldObservation` objects.

Success criteria:

- reproduces existing `render_field_observation` output for a one-scene field model;
- tests compare scene wrapper to current field path.

### Milestone 2: multi-exposure super-stack fit

Deliverables:

- fitting loop for many exposures sharing one implicit scene;
- synthetic dithered exposures;
- residual and gradient diagnostics.

Success criteria:

- shared latent field can fit multiple dithered exposures better than fitting each exposure independently;
- masks and weights behave correctly;
- gradients propagate through WCS sampling and spectral rendering paths already supported by the field machinery.

### Milestone 3: multiscale latent field

Deliverables:

- learnable multiresolution latent grid or patch-local latent tensors;
- compact decoder over coordinates, wavelength, and latent samples;
- regularization by scale level.

Success criteria:

- improves convergence and residuals relative to a single global INR on synthetic multi-scale scenes;
- handles smooth and moderately compact structure without explicit source labels;
- memory and runtime scale plausibly with patch size.

### Milestone 4: optional adaptive atom branch

Deliverables:

- untyped compact atom basis;
- finite-support rendering;
- optional soft power-ownership prototype;
- atom regularization terms.

Success criteria:

- improves representation of sharp or undersampled structure when enabled;
- does not become mandatory for smooth-field cases;
- rendering avoids full dense atom-pixel products for moderate atom counts.

### Milestone 5: hybrid implicit super-stack prototype

Deliverables:

- scene with multiscale implicit branch plus optional adaptive atoms;
- multi-exposure fitting loop;
- diagnostics separating implicit residuals from adaptive-atom residuals without claiming physical source labels;
- example script using small realistic FITS fixtures or synthetic WCS/PSF data.

Success criteria:

- joint fit beats a fixed-grid coadd baseline on residuals and astrometric consistency;
- adaptive atoms reduce high-frequency residuals without degrading large-scale LSB recovery;
- the implicit branch remains the primary representation.

### Milestone 6: semantic projections

Deliverables:

- catalog-like summaries from fitted latent states;
- compact-candidate extraction;
- large-scale residual maps;
- persistent-vs-exposure-tied residual diagnostics.

Success criteria:

- semantic views are useful for inspection but not required to render or fit the scene;
- derived catalogs can be regenerated from the latent state.

### Milestone 7: time-dependent latent scene

Deliverables:

- time coordinate support in the implicit scene;
- linear-motion and variable-flux optional adapters;
- exposure-time-aware rendering;
- synthetic moving/variable structure tests.

Success criteria:

- time-variable structure is not forced into a static sky;
- static and dynamic latent terms separate cleanly in controlled simulations.

## Initial tests

### Unit tests

- field-only scene wrapper equals existing field renderer;
- scene loss respects masks and weights;
- multiscale latent sampling is differentiable;
- adaptive atom renderer matches brute-force dense rendering on tiny images;
- CPU and CUDA/MPS behavior is consistent where supported.

### Synthetic integration tests

- fit several dithered exposures with a pure implicit field;
- fit smooth plus compact synthetic structure without source labels;
- compare single-scale and multiscale implicit scenes;
- enable adaptive atoms and confirm high-frequency residual reduction;
- test masked pixels and chip gaps;
- test different PSFs per exposure.

### Realistic-data checks

Use small packaged or downloaded fixtures only in optional test/example paths. Do not make real-data downloads part of standard tests.

Candidate examples:

- small CFHT/HSC-like FITS cutouts;
- synthetic WCS headers with realistic dithers;
- simulated PSF and weight maps;
- later, a tiny real multi-exposure stack if redistribution is permitted.

## Benchmark surfaces

Track the following:

- render time per exposure as a function of image size and latent representation size;
- gradient step time for multi-exposure fitting;
- memory use of multiscale latent grids;
- candidate pixel evaluations for adaptive atoms when enabled;
- crossover point between dense implicit rendering and sparse adaptive rendering;
- CPU vs CUDA behavior.

Benchmark against:

- current diffuse INR-only rendering;
- fixed-grid latent image rendering;
- naive dense atom rendering;
- brute-force tiny-scene renderer for correctness only.

## API sketch

```python
from torchsky.map.scene import FieldSkyScene, render_scene_observation

scene = FieldSkyScene(field_model)
pred = render_scene_observation(scene, observation)
```

Multiscale implicit scene:

```python
from torchsky.map.scene import MultiscaleSkyScene

scene = MultiscaleSkyScene(
    domain=domain,
    levels=(32, 64, 128),
    decoder="mlp",
    spectral_basis=8,
)
```

Optional adaptive atoms:

```python
scene.enable_adaptive_atoms(
    count=1024,
    kernel="compact_gaussian",
    spectral_basis=4,
)
```

A fitting loop should accept many observations:

```python
loss = multi_observation_scene_loss(
    scene,
    observations,
    loss_kind="charbonnier",
    regularization_weights={
        "multiscale.level_smoothness": 1.0e-5,
        "adaptive_atoms.sparsity": 1.0e-4,
        "adaptive_atoms.overlap": 1.0e-3,
    },
)
```

## Open design questions

1. What is the best first multiscale implicit parameterization: latent grid, hash grid, HEALPix-like hierarchy, or pretrained trunk plus local latents?
2. Should adaptive atoms be a branch of the scene or a separate optimizer-side residual module?
3. How should atom birth/death/splitting be handled without brittle heuristics?
4. What is the cleanest sparse rendering representation on CPU, CUDA, and MPS?
5. Should WCS projection remain differentiable with respect to astrometric nuisance parameters in the first version?
6. When should exact power-cell geometry be worth implementing?
7. How should uncertainty be represented: ensembles, profile scans, conformal residual diagnostics, stochastic residual fields, or a combination?
8. How should the scene state be serialized so it becomes an archive-native product rather than an ephemeral model checkpoint?
9. How can semantic summaries be derived without making them part of the primary optimization target?

## Recommended first PR sequence

1. Add `torchsky.map.scene` skeleton with implicit-first base classes and no new dependencies.
2. Add `FieldSkyScene` and tests proving equivalence to existing field rendering.
3. Add a tiny synthetic multi-exposure scene-fitting example.
4. Add `MultiscaleSkyScene` with a simple learnable latent grid plus decoder.
5. Add diagnostics and benchmark scripts for multi-exposure fitting.
6. Add optional `AdaptiveAtomBranch` only after the pure/multiscale implicit path is working.
7. Only then consider optimized sparse kernels or exact power-cell approximations.

## Expected scientific payoff

A successful implementation would produce a latent sky state rather than only a coadd:

```text
latent_sky_state/
  implicit_field
  multiscale_latents
  optional_adaptive_atoms
  exposure_nuisance_parameters
  diagnostics
  semantic_views
```

This could support:

- exposure-native super-stacking;
- deblending and compact-structure discovery without premature catalog cuts;
- joint multi-band and multi-survey inference;
- LSB science without forcing all structure into named compact components;
- optional dynamic extensions for proper motion, transients, and Solar System objects;
- uncertainty diagnostics tied to the forward model rather than to a post-hoc coadd;
- catalog-like products as derived views rather than as the core representation.
