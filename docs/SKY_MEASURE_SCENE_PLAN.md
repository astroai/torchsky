# SkyMeasure Scene Implementation Plan

## Purpose

This note lays out an implementation plan for a **measure-valued, multiscale sky scene** in `torchsky`: a latent sky representation designed to explain many exposures directly, rather than first collapsing them into a coadd. The target is an archive-native "super-stack" model that can represent point sources, compact galaxies, extended galaxies, diffuse emission, ultra-large-scale low-surface-brightness structure, unresolved populations, and time-variable or moving sources.

The guiding principle is:

> represent the sky with components whose mathematical structure matches the phenomenon, then render all components jointly through exposure-specific WCS, PSF, bandpass, detector, background, mask, and noise operators.

This is intentionally broader than a pure INR, Gaussian splat model, catalog model, or coadd. Those become special cases inside a common forward-modeling framework.

## Design thesis

The latent sky should be treated as a measure over direction, wavelength, and time:

$$
\mu(d\omega, d\lambda, dt), \qquad \omega=(\alpha,\delta).
$$

An exposure observes this latent scene through an exposure operator:

$$
Y_e[p] \sim \mathbb{P}_{\psi_e}\left(\cdot\mid
\int R_{e,p}(\omega,\lambda,t)\,\mu(d\omega,d\lambda,dt) + B_e[p]
\right),
$$

where $R_{e,p}$ includes WCS sampling, PSF, bandpass, exposure time, detector response, and masking, while $B_e$ contains exposure-side background and nuisance terms.

A practical scene decomposition is:

$$
\mu =
\mu_{\rm ultra\text{-}large}
+ \mu_{\rm diffuse}
+ \mu_{\rm objects}
+ \mu_{\rm unresolved}
+ \mu_{\rm transient}.
$$

The first implementation should not attempt all terms at once. It should establish a minimal but extensible architecture where each component can render into an observation and contribute to a common loss.

## Non-goals

- Do not implement a full graphics-style Power Foam or NeRF renderer.
- Do not make exact weighted Voronoi or power-diagram construction a dependency of the first milestone.
- Do not add heavy astronomy ecosystem packages to the runtime dependency set.
- Do not require `astropy`, `healpy`, `fitsio`, `pandas`, or similar packages in core import paths.
- Do not replace the existing `torchsky.map.field` path; build a scene abstraction that can reuse it.
- Do not bake detector artifacts, PSF wings, chip defects, or cosmic rays into the persistent sky representation unless explicitly modeled as sky-persistent features.

## Proposed package structure

Initial experimental namespace:

```text
torchsky.map.scene
  base.py              # scene/component protocols and common dataclasses
  exposure.py          # exposure operator wrappers around existing FieldObservation paths
  render.py            # joint scene rendering orchestration
  losses.py            # weighted exposure losses and diagnostics
  components/
    diffuse.py         # wrappers for existing INR fields
    points.py          # delta-like point-source measure component
    primitives.py      # compact bounded sky primitives
    lowrank.py         # optional low-frequency map/needlet branch later
    stochastic.py      # unresolved/confusion branch later
    dynamic.py         # moving/variable sources later
```

If this proves stable, promote selected parts to public API. Until then, keep the namespace explicitly experimental or document the API as provisional.

## Core abstractions

### `SkyScene`

A scene is a collection of components plus optional metadata:

```python
class SkyScene(nn.Module):
    components: nn.ModuleDict

    def render_observation(self, observation: FieldObservation, **kw) -> Tensor: ...
    def regularization_terms(self) -> dict[str, Tensor]: ...
    def diagnostics(self) -> dict[str, object]: ...
```

The scene itself should not know survey-specific policy beyond invoking the exposure renderer. It should compose components and sum their predicted sky contribution before or during observation rendering.

### `SkyComponent`

Each component must implement a narrow rendering interface:

```python
class SkyComponent(nn.Module):
    def render_observation(self, observation: FieldObservation, **kw) -> Tensor: ...
    def regularization_terms(self) -> dict[str, Tensor]: ...
```

For components that can be sampled as a continuous field, also support:

```python
def evaluate_sky(self, ra_deg: Tensor, dec_deg: Tensor, wavelength: Tensor | float, **kw) -> Tensor: ...
```

Not all components need to support dense sky evaluation. Point-source and primitive components may be more efficiently rendered by sparse footprint accumulation.

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
    return_component_images: bool = False
```

This should initially wrap existing `render_field_observation` options rather than duplicate them.

## Component formulations

### 1. Diffuse INR component

Wrap the current field models as a scene component:

$$
I_{\rm diff}(\omega,\lambda) = F_\theta(\omega,\lambda).
$$

Responsibilities:

- reuse existing `torchsky.map.field` models and `render_field_observation`;
- support multi-band rendering and existing bandpass quadrature;
- optionally expose smoothness regularization and spectral regularization;
- serve as the baseline component for smooth sky and LSB structure.

First implementation: a thin `DiffuseFieldComponent(model)` wrapper.

### 2. Point-source measure component

Represent unresolved compact sources as atomic measures:

$$
\mu_{\rm point} = \sum_i S_i(\lambda,t)\,\delta(\omega-\omega_i).
$$

The PSF must remain in the exposure operator, not in the sky component. A point source is a delta-like sky object rendered through each exposure's WCS and PSF.

Parameters:

- `ra_deg`, `dec_deg` or local tangent coordinates;
- spectral coefficients per source;
- optional time variability coefficients;
- optional proper motion/parallax later;
- optional role/type labels for diagnostics.

First implementation:

- render each source into the target image using WCS projection and a simple differentiable Gaussian PSF approximation from `FieldObservation.psf_sigma_pix`;
- support finite tile/footprint accumulation to avoid full image-by-source `O(N_sources * N_pixels)` work;
- add tests against brute-force dense rendering for small scenes.

### 3. Compact bounded primitive component

Represent extended and semi-compact sources with bounded adaptive sky primitives:

$$
I_{\rm prim}(\omega,\lambda) = \sum_i A_i(\lambda)\,\phi_i(\omega; p_i, r_i, q_i, \eta_i).
$$

Start with compact elliptical Gaussian or compact Moffat-like kernels with soft truncation. Keep the Power-Foam/power-cell idea as an optional later refinement.

Parameters:

- center position;
- log radius or scale;
- ellipticity/axis-ratio/orientation;
- spectral coefficients;
- optional local morphology coefficients;
- optional source grouping metadata.

Initial kernel options:

1. circular bounded Gaussian;
2. elliptical bounded Gaussian;
3. compact Moffat-like profile;
4. optional soft power-ownership weights for overlapping primitives.

Soft power ownership, if added, can use:

$$
q_i(x)=\|x-p_i\|^2-r_i^2,
$$

$$
o_i(x)=\frac{\exp[-q_i(x)/\tau]}{\sum_{j\in\mathcal{N}(x)}\exp[-q_j(x)/\tau]}.
$$

This gives a differentiable approximation to bounded power-cell ownership without implementing exact clipped power diagrams.

### 4. Ultra-large-scale component

Represent sky structure larger than a tract, chip, or exposure with a low-frequency basis:

$$
I_{\rm ULS}(\omega,\lambda) = \sum_k a_k(\lambda)\,b_k(\omega),
$$

where $b_k$ may be spherical harmonics, needlets, low-resolution HEALPix pixels, or patchwise low-order modes.

First implementation should be deferred until the three-component static prototype works. For early work, a low-resolution learnable grid sampled onto images is sufficient.

### 5. Unresolved stochastic component

Represent confusion and sub-threshold populations statistically rather than as deterministic fake sources:

$$
\mu_{\rm unresolved} \sim \mathcal{P}_\varphi(P(k,\lambda), N(S,\lambda), C_\ell).
$$

This is a later milestone. Initial implementation can be a correlated residual field with learned or fixed covariance proxy, used only in diagnostics and uncertainty experiments.

### 6. Dynamic component

Add time-dependent positions and fluxes after static rendering works:

$$
\omega_i(t)=\omega_{i,0}+\boldsymbol{\mu}_i t+\varpi_i\Pi(t)+\omega_{\rm orbit}(t),
$$

$$
S_i(\lambda,t)=S_{i,0}(\lambda)+\Delta S_i(\lambda,t).
$$

This should reuse the existing kinematic direction in `torchsky.map.field.implicit_forward` rather than create a separate system.

## Rendering strategy

### Dense field rendering

Diffuse components can reuse `render_field_observation` directly.

### Sparse component rendering

Point and primitive components should render sparsely:

1. project component centers into exposure pixel coordinates using `WCS.world_to_pixel`;
2. compute finite support radius in pixels;
3. build source-to-tile or source-to-pixel candidate lists;
4. evaluate kernels only inside local footprints;
5. accumulate with differentiable torch operations;
6. convolve with PSF only if the component is not already rendered through an analytic PSF path.

Initial implementation may use simple vectorized chunking before native kernels. Add C++/CUDA only after profiling proves the bottleneck.

### PSF policy

For the first prototype:

- use existing scalar Gaussian PSF support where available;
- keep the sky component PSF-free;
- apply PSF as part of observation rendering;
- document when an analytic point-source rendering path folds the Gaussian PSF into the source footprint for efficiency.

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

### Component regularization

Add optional terms, each individually named and weighted:

- diffuse smoothness or spectral smoothness;
- primitive compactness;
- primitive flux positivity or soft positivity;
- point-source sparsity;
- overlap penalties for bounded primitives;
- background/nuisance smoothness;
- cross-band color smoothness;
- temporal smoothness for dynamic components later.

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

### Milestone 1: diffuse scene wrapper

Deliverables:

- `DiffuseFieldComponent` wrapping an existing field model;
- `SkyScene` summing one or more components;
- `render_scene_observation(scene, observation)`;
- scene loss over a list of `FieldObservation` objects.

Success criteria:

- reproduces existing `render_field_observation` output for a one-component diffuse scene;
- tests compare scene wrapper to current field path.

### Milestone 2: point-source component

Deliverables:

- `PointSourceComponent` with positions and spectral coefficients;
- differentiable projection through WCS;
- sparse or chunked footprint rendering;
- synthetic tests with known injected point sources.

Success criteria:

- source positions and fluxes can be recovered from multiple dithered synthetic exposures;
- gradients propagate to positions and fluxes;
- rendering avoids full dense source-pixel products for moderate source counts.

### Milestone 3: bounded primitive component

Deliverables:

- circular and elliptical compact primitive kernels;
- finite-support rendering;
- optional soft power-ownership prototype;
- overlap graph construction in torch;
- primitive regularization terms.

Success criteria:

- extended synthetic sources are recovered better than point-only models;
- primitive rendering is stable under dithers and masks;
- overlap graph reduces candidate evaluations relative to naive rendering.

### Milestone 4: hybrid super-stack prototype

Deliverables:

- scene with diffuse INR + point sources + bounded primitives;
- multi-exposure fitting loop;
- diagnostics separating component residuals;
- example notebook/script using small real or realistic FITS fixtures.

Success criteria:

- joint fit to multiple exposures beats a fixed-grid coadd baseline on residuals and astrometric consistency;
- point-source fluxes remain stable across dithers;
- diffuse branch does not absorb obvious point-source residuals when the point component is active.

### Milestone 5: ultra-large-scale branch

Deliverables:

- low-resolution learnable sky grid or simple harmonic/needlet-like branch;
- rendering into many observations;
- regularization for smoothness and amplitude control.

Success criteria:

- large-scale injected backgrounds or LSB components are recovered without forcing compact primitives to explain them;
- component separation diagnostics remain interpretable.

### Milestone 6: stochastic unresolved residual branch

Deliverables:

- residual field model or covariance proxy for unresolved populations;
- uncertainty diagnostics;
- tests showing reduced deterministic overfitting to sub-threshold confusion.

Success criteria:

- deterministic components do not hallucinate excessive faint sources in confusion-only simulations;
- residual diagnostics are better calibrated than point/primitive-only fits.

### Milestone 7: dynamic sources

Deliverables:

- proper-motion/linear-motion point source extension;
- time-dependent flux coefficients;
- exposure-time-aware rendering;
- synthetic moving-object and variable-source tests.

Success criteria:

- moving sources are not smeared into the static sky;
- static and dynamic components separate cleanly in controlled simulations.

## Initial tests

### Unit tests

- scene wrapper equals existing field renderer for a one-component diffuse scene;
- WCS projection gradients behave as expected for local perturbations;
- point-source renderer matches brute-force dense rendering on tiny images;
- bounded primitive renderer matches brute-force dense rendering on tiny images;
- masks and weights are respected;
- CPU and CUDA/MPS behavior is consistent where supported.

### Synthetic integration tests

- recover one point source from several dithered exposures;
- recover two blended point sources with different colors;
- recover one extended source with an elliptical primitive;
- recover diffuse background plus compact source without component leakage;
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

- render time per exposure as a function of image size and component count;
- candidate pixel evaluations before and after footprint pruning;
- memory use for source-to-tile candidate structures;
- gradient step time for multi-exposure fitting;
- crossover point between dense and sparse rendering;
- CPU vs CUDA behavior.

Benchmark against:

- current diffuse INR-only rendering;
- fixed-grid latent image rendering;
- naive dense splat rendering;
- brute-force tiny-scene renderer for correctness only.

## API sketch

```python
from torchsky.map.scene import SkyScene, render_scene_observation
from torchsky.map.scene.components import (
    DiffuseFieldComponent,
    PointSourceComponent,
    BoundedPrimitiveComponent,
)

scene = SkyScene({
    "diffuse": DiffuseFieldComponent(field_model),
    "points": PointSourceComponent(...),
    "extended": BoundedPrimitiveComponent(...),
})

pred = render_scene_observation(scene, observation)
loss = scene_observation_loss(pred, observation)
regs = scene.regularization_terms()
```

A fitting loop should accept many observations:

```python
loss = multi_observation_scene_loss(
    scene,
    observations,
    loss_kind="charbonnier",
    regularization_weights={
        "points.sparsity": 1.0e-4,
        "extended.overlap": 1.0e-3,
        "diffuse.smoothness": 1.0e-5,
    },
)
```

## Open design questions

1. Should point sources and compact primitives share a single object table, or remain separate modules?
2. Should spectral coefficients be simple per-band amplitudes first, or basis coefficients over wavelength?
3. How should component birth/death/splitting be handled without brittle heuristics?
4. What is the cleanest sparse rendering representation on CPU, CUDA, and MPS?
5. Should WCS projection remain differentiable with respect to astrometric nuisance parameters in the first version?
6. When should exact power-cell geometry be worth implementing?
7. How should uncertainty be represented: ensembles, profile scans, conformal residual diagnostics, stochastic residual fields, or a combination?
8. How should the scene state be serialized so it becomes an archive-native product rather than an ephemeral model checkpoint?

## Recommended first PR sequence

1. Add `torchsky.map.scene` skeleton with base classes and no new dependencies.
2. Add `DiffuseFieldComponent` and tests proving equivalence to existing field rendering.
3. Add a tiny synthetic multi-exposure scene-fitting example.
4. Add `PointSourceComponent` with brute-force and chunked render paths.
5. Add `BoundedPrimitiveComponent` with simple circular kernels.
6. Add scene diagnostics and benchmark scripts.
7. Only then consider optimized sparse kernels or exact power-cell approximations.

## Expected scientific payoff

A successful implementation would produce a latent sky state rather than only a coadd:

```text
latent_sky_state/
  diffuse_field
  point_sources
  extended_primitives
  ultra_large_scale_modes
  unresolved_residual_model
  exposure_nuisance_parameters
  diagnostics
```

This could support:

- exposure-native super-stacking;
- deblending without premature catalog cuts;
- joint multi-band and multi-survey inference;
- stable point-source astrometry/photometry across dithers;
- LSB science without forcing all structure into compact sources;
- dynamic-source extensions for proper motion, transients, and Solar System objects;
- uncertainty diagnostics tied to the forward model rather than to a post-hoc coadd.
