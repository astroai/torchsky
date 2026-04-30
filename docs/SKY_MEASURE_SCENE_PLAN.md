# Implicit SkyMeasure Scene Implementation Plan

## Purpose

This note lays out an implementation plan for an **implicit-first, measure-valued, multiscale sky scene** in `torchsky`: a latent sky representation designed to explain many exposures directly, rather than first collapsing them into a coadd.

The goal is an archive-native "super-stack" model that can represent point-like structure, compact structure, extended galaxies, diffuse emission, ultra-large-scale low-surface-brightness structure, unresolved populations, and time-variable or moving structure **without requiring a hard, explicit source decomposition as the primary abstraction**.

The guiding principle is:

> learn a shared latent sky state that is rendered through exposure-specific WCS, PSF, bandpass, detector, background, mask, and noise operators; introduce explicit components only as optional inductive biases, accelerators, diagnostics, or constraints.

This is broader than a pure INR, Gaussian splat model, catalog model, or coadd. Those become possible parameterizations of the latent sky state, not mandatory source taxonomies.

## Fit with the current `torchsky` codebase

This plan should extend the existing mapmaking and field-model architecture rather than create an incompatible parallel stack.

The current code already has several pieces that should remain central:

- `torchsky.map.field.FieldObservation`: in-memory exposure-like observation carrying image, WCS header, weight, mask, bandpass metadata, PSF metadata, and background metadata.
- `torchsky.map.field.FieldDomain`: coordinate normalization domain for field models.
- `torchsky.map.field.render_field_observation`: the current differentiable rendering path from a continuous field to an observation.
- `torchsky.map.field.implicit_forward`: a scaffold for shared-scene, multi-survey losses using the same rendering path.
- `torchsky.map.scenes.SceneRepresentation`: the existing scene adapter abstraction used by map-product rendering.
- `torchsky.map.scenes.ContinuousFieldModel`: the existing adapter that wraps an `nn.Module` and renders it through `render_field_observation`.
- `TargetRenderSpec` and map-product APIs: the product-level surface for comparing direct stacks and rendered field products.

Therefore, the first implementation should **not** create an unrelated `torchsky.map.scene` universe. The safer path is:

1. extend or reuse `src/torchsky/map/scenes.py` for adapter-level behavior;
2. keep field-specific rendering in `src/torchsky/map/field/`;
3. add new experimental implementations only where the current abstractions become too large;
4. keep all paths compatible with `FieldObservation`, `TargetRenderSpec`, and `render_field_observation`.

A future subpackage such as `torchsky.map.sky_measure` or `torchsky.map.scene_state` may be justified once the implementation exceeds a single adapter module, but the first PR should fit the existing names and call graph.

## Mathematical foundation

### 1. The sky as a finite radiance measure

Let

$$
\Omega = \mathbb{S}^2 \times \Lambda \times \mathcal{T}
$$

be the product of sky direction, wavelength, and time. The latent sky is a non-negative finite measure

$$
\mu_\theta \in \mathcal{M}_+(\Omega),
$$

where $\theta$ denotes trainable parameters. A measure-valued formulation is useful because it does not require the sky to be representable as an ordinary smooth image everywhere. It can include smooth emission, sharp compact structure, singular/atomic structure, and stochastic unresolved structure in one mathematical object.

The implementation should start from the absolutely continuous case,

$$
d\mu_\theta(\omega,\lambda,t)
= I_\theta(\omega,\lambda,t)\,d\sigma(\omega)\,d\lambda\,dt,
$$

where $d\sigma$ is solid angle on the sphere and $I_\theta$ is a latent specific intensity field. This keeps the first version close to the existing continuous-field machinery.

The more general measure notation remains important because later extensions may use an optional singular or atomic part,

$$
\mu_\theta
= I_\theta\,d\sigma d\lambda dt
+ \mu^{\rm sing}_\theta,
$$

without making that singular part a semantic catalog. In other words, the formalism permits atoms, but the implementation should not force all compact structure to be interpreted as named stars or galaxies.

### 2. Exposure rendering as a linear operator on the sky state

For exposure $e$ and detector pixel $p$, define a response kernel

$$
R_{e,p}(\omega,\lambda,t) \ge 0.
$$

This kernel includes, at minimum:

- projection from sky coordinates to detector coordinates through the exposure WCS;
- integration over the pixel footprint;
- the wavelength-dependent throughput or bandpass;
- the PSF, possibly spatially and chromatically varying;
- exposure time and photometric calibration;
- masks and effective support.

The noiseless prediction is

$$
m_{e,p}(\theta,\psi_e)
= \int_\Omega R_{e,p}(\omega,\lambda,t;\psi_e)\,\mu_\theta(d\omega,d\lambda,dt)
+ b_{e,p}(\psi_e),
$$

where $\psi_e$ are exposure/instrument/background nuisance parameters and $b_{e,p}$ is an exposure-side background or detector term.

In vector notation:

$$
\mathbf{m}_e = \mathcal{R}_e(\psi_e)[\mu_\theta] + \mathbf{b}_e.
$$

For an absolutely continuous field this becomes

$$
m_{e,p}
= \int R_{e,p}(\omega,\lambda,t;\psi_e)
I_\theta(\omega,\lambda,t)
\,d\sigma\,d\lambda\,dt
+ b_{e,p}.
$$

This is the mathematical basis for the existing `render_field_observation` path. The new scene-state machinery should generalize the latent representation, not replace the exposure operator concept.

### 3. Likelihood and loss

A useful default observation model is inverse-variance weighted Gaussian noise:

$$
Y_{e,p} \sim \mathcal{N}(m_{e,p},\sigma_{e,p}^2),
$$

with weight $w_{e,p}=\sigma_{e,p}^{-2}$ and mask $M_{e,p}\in\{0,1\}$. The negative log-likelihood is

$$
\mathcal{L}_{\rm NLL}(\theta,\psi)
= \frac{1}{2}\sum_{e,p} M_{e,p}w_{e,p}
\left(Y_{e,p}-m_{e,p}\right)^2
+ \frac{1}{2}\sum_{e,p} M_{e,p}\log\sigma_{e,p}^2.
$$

A robust first implementation can use the Charbonnier loss,

$$
\rho_\epsilon(r)=\sqrt{r^2+\epsilon^2}-\epsilon,
$$

$$
\mathcal{L}_{\rm obs}
= \sum_{e,p} M_{e,p}w_{e,p}\rho_\epsilon(Y_{e,p}-m_{e,p}).
$$

This is already aligned with the existing implicit-forward scaffold. The scene implementation should keep mask and weight semantics identical to `FieldObservation`.

The full optimization problem is

$$
\min_{\theta,\psi}\;
\mathcal{L}_{\rm obs}(\theta,\psi)
+ \sum_k \alpha_k\mathcal{R}_k(\theta,\psi),
$$

where $\mathcal{R}_k$ are explicit regularization terms. Regularization must be visible and named; it should not be hidden inside rendering code.

### 4. Ill-posedness and identifiability

The inverse problem is ill-posed. Many latent skies can predict the same finite set of noisy, PSF-blurred, masked exposures. The representation should therefore separate three things:

1. **data likelihood:** what the exposures actually constrain;
2. **representation prior / inductive bias:** smoothness, locality, spectral structure, compactness;
3. **semantic interpretation:** optional summaries derived after fitting.

A useful test is whether a proposed latent degree of freedom is identifiable under the exposure operators. For a perturbation $\delta\mu$, if

$$
\mathcal{R}_e[\delta\mu] \approx 0 \quad \forall e,
$$

then that mode is in or near the null space of the data. It should be regularized, constrained by external data, or omitted from the first implementation.

This also motivates multiscale representations: well-observed spatial frequencies can be fitted with more freedom, while poorly observed frequencies require stronger regularization or lower-dimensional parameterization.

### 5. Multiscale implicit parameterization

A single global INR can be elegant but too globally coupled. A stronger implicit formulation is a multiscale latent field:

$$
I_\theta(\omega,\lambda,t)
= D_\theta\left(
\gamma_\omega(\omega),
\gamma_\lambda(\lambda),
\gamma_t(t),
Z_0(\omega),Z_1(\omega),\ldots,Z_L(\omega)
\right),
$$

where:

- $\gamma_\omega,\gamma_\lambda,\gamma_t$ are coordinate encodings;
- $Z_l$ are learnable latent grids, feature planes, HEALPix-like levels, needlet-like levels, or patch-local tensors;
- $D_\theta$ is a compact decoder.

This combines INR flexibility with local capacity. It is closer to modern feature-field methods than to a monolithic MLP.

For tangent-plane patches, $Z_l$ can be image-like latent grids sampled with differentiable interpolation. For wide fields or all-sky work, $Z_l$ should eventually be sphere-aware: HEALPix-like, harmonic, or needlet-like rather than a single flat image plane.

### 6. Optional adaptive atoms as numerical basis functions

Some structures may be inefficient for a continuous field. The optional sparse/adaptive term can be written as

$$
I_\theta(\omega,\lambda,t)
= I^{\rm implicit}_\theta(\omega,\lambda,t)
+ \sum_i a_i(\lambda,t)\,\phi_i(\omega;\eta_i).
$$

The atoms $\phi_i$ are **basis functions**, not source labels. They can be used for sharp residuals, compact features, or local capacity, but the core API should not call them stars, galaxies, cosmic rays, or artifacts.

A simple compact atom is

$$
\phi_i(x)
= A_i\exp\left[-\frac{1}{2}(x-p_i)^\top \Sigma_i^{-1}(x-p_i)\right]s\left(\frac{\|x-p_i\|}{r_i}\right),
$$

where $s$ is a smooth compact-support taper. This gives finite support while preserving useful gradients.

A Power-Foam-inspired soft ownership variant can use power distances

$$
q_i(x)=\|x-p_i\|^2-r_i^2,
$$

and local soft ownership

$$
o_i(x)=\frac{\exp[-q_i(x)/\tau]}{\sum_{j\in\mathcal{N}(x)}\exp[-q_j(x)/\tau]}.
$$

This gives a differentiable approximation to bounded power-cell behavior without committing to exact clipped power diagrams in the first implementation.

### 7. Positivity and flux semantics

For surface-brightness fields, non-negativity can be enforced by parameterizing

$$
I_\theta = \operatorname{softplus}(u_\theta)
$$

or by using a signed residual field around a positive baseline when calibrated background subtraction is already applied. The choice must be explicit because signed residual maps and physical intensities have different semantics.

The implementation must preserve the existing `value_semantics` distinction in map products:

- surface brightness: values are intensities per solid angle or per pixel convention;
- flux: rendering or reprojection must account for pixel-area changes.

The scene-state API should not silently switch between these semantics.

## Key correction: implicit first, decomposition optional

A hard decomposition into named components such as point sources, diffuse fields, extended primitives, unresolved populations, and transients is useful for interpretation, but it should not be the primary abstraction.

A better hierarchy is:

1. **Implicit latent sky state:** the object being inferred.
2. **Multiscale parameterization:** the numerical basis used to represent that state.
3. **Optional adaptive atoms/primitives:** local capacity added where the implicit field is inefficient.
4. **Optional semantic views:** catalog-like or component-like interpretations derived from the fitted state.

Do not begin by asking whether a photon came from "diffuse", "point", or "extended" source classes. Begin by asking whether the latent sky state, when rendered through all exposure operators, predicts the detector data with calibrated residuals and sensible uncertainty.

## Non-goals

- Do not implement a full graphics-style Power Foam or NeRF renderer.
- Do not make exact weighted Voronoi or power-diagram construction a dependency of the first milestone.
- Do not force a hard astrophysical source taxonomy into the core API.
- Do not require every fitted structure to be assigned to a named source class.
- Do not add heavy astronomy ecosystem packages to the runtime dependency set.
- Do not require `astropy`, `healpy`, `fitsio`, `pandas`, or similar packages in core import paths.
- Do not replace the existing `torchsky.map.field` path; build on it.
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
\right).
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
- candidate compact-feature lists;
- extended-feature footprints;
- LSB residual maps;
- transient or moving-feature candidates;
- uncertainty and coverage diagnostics.

These are outputs or diagnostics, not the primary latent representation.

## Integration plan with existing modules

### Existing adapter layer

`src/torchsky/map/scenes.py` already defines the adapter pattern:

```python
class SceneRepresentation(ABC):
    def render_to_target(self, target: TargetRenderSpec, **kwargs) -> Tensor: ...
```

and wrappers such as:

- `DiscreteTargetImage`;
- `HealpixMapScene`;
- `HarmonicAlmScene`;
- `SpinHarmonicAlmScene`;
- `ContinuousFieldModel`.

The first implementation should therefore add new behavior by extending this family, not by replacing it.

Recommended first additions:

```text
src/torchsky/map/scenes.py
  ImplicitSkyState          # optional abstract base or protocol-like adapter
  FieldSkyState             # thin wrapper around existing field model path
  MultiscaleFieldSkyState   # later; may live in map/field if it is a field model
```

If the module becomes too large, split implementation details into:

```text
src/torchsky/map/scene_state/
  __init__.py
  base.py
  multiscale.py
  adaptive.py
  diagnostics.py
```

but keep `torchsky.map.scenes` as the compatibility/export surface.

### Existing field-rendering path

The first scene-state wrapper should be equivalent to:

```python
ContinuousFieldModel(model).render_to_target(target, observation=obs)
```

or directly:

```python
render_field_observation(model, obs, ...)
```

Therefore, `FieldSkyState` should initially be a thin semantic wrapper around `ContinuousFieldModel`, adding only:

- multi-observation loss helpers;
- explicit regularization hooks;
- diagnostics hooks;
- optional serialization metadata.

It should not duplicate WCS, bandpass, or PSF logic.

### Existing product surface

The product-level comparison path should remain:

```text
plan_map_product -> stack_map_product
                 -> render_field_product / render_scene_product
                 -> evaluate_map_product
```

A future `render_scene_product` should accept any `SceneRepresentation` and produce the same `MapProductResult` contract as `render_field_product`. This keeps direct stacks, field products, and scene-state products comparable.

### Naming guidance

Avoid introducing `torchsky.map.scene` immediately because the repository already has `torchsky.map.scenes`. Prefer:

- extend `torchsky.map.scenes` for adapters;
- use `torchsky.map.scene_state` only if implementation files need to be split;
- expose stable user-facing names from `torchsky.map` only after the API is proven.

## Proposed implementation abstractions

### `FieldSkyState`

A field-only scene state:

```python
@dataclass
class FieldSkyState(SceneRepresentation):
    model: nn.Module

    def render_to_target(self, target: TargetRenderSpec, **kwargs) -> Tensor: ...
    def render_observation(self, observation: FieldObservation, **kwargs) -> Tensor: ...
    def regularization_terms(self) -> dict[str, Tensor]: ...
```

This is intentionally close to `ContinuousFieldModel`. It may even subclass or contain it.

### `MultiscaleSkyState`

A scene state with trainable latent grids or feature levels plus a decoder:

```python
class MultiscaleSkyState(nn.Module):
    domain: FieldDomain
    levels: nn.ParameterList
    decoder: nn.Module

    def forward(self, ra_deg, dec_deg, wavelength, *, forward_context=None): ...
```

If it exposes the same `forward(ra, dec, wavelength)` signature as existing field models, it can be rendered by `render_field_observation` without special scene rendering. This is the preferred route: make new scene-state models compatible with the existing field renderer whenever possible.

### `AdaptiveAtomBranch`

An optional branch for compact adaptive basis functions. This should be gated and experimental.

```python
class AdaptiveAtomBranch(nn.Module):
    def render_observation(self, observation: FieldObservation, **kwargs) -> Tensor: ...
    def regularization_terms(self) -> dict[str, Tensor]: ...
```

This branch will likely require sparse footprint rendering and therefore may not fit the plain `forward(ra, dec, wavelength)` interface. It should be added only after the field-only and multiscale implicit paths are working.

### `CompositeSkyState`

A later composition wrapper:

```python
class CompositeSkyState(SceneRepresentation):
    implicit: SceneRepresentation | nn.Module
    adaptive: AdaptiveAtomBranch | None

    def render_to_target(...):
        return implicit_render + adaptive_render
```

The composition is numerical, not semantic. It should not expose fields named `stars`, `galaxies`, or `diffuse` in the core API.

## Initial implementation path

### 1. Field-only scene-state wrapper

Wrap the current field models as an implicit sky state:

$$
I_\theta(\omega,\lambda)=F_\theta(\omega,\lambda).
$$

Responsibilities:

- reuse existing `torchsky.map.field` models and `render_field_observation`;
- support multi-band rendering and existing bandpass quadrature;
- expose a scene-level loss over many `FieldObservation` objects;
- preserve exact equivalence with the existing field rendering path for one observation.

First implementation: `FieldSkyState(model)`, implemented as a very thin wrapper around `ContinuousFieldModel` or directly around `render_field_observation`.

### 2. Multi-exposure implicit super-stack loss

Add a loss helper over many observations:

$$
\mathcal{L}(\theta)
= \sum_e \mathcal{L}_e\left(Y_e,\mathcal{R}_e[F_\theta]\right).
$$

This can be implemented in terms of existing `multi_survey_charbonnier_loss` concepts, but should live close to map/scene product evaluation if it is intended as a mapmaking entry point.

### 3. Multiscale implicit field

Add learnable latent grids or patch-local latent tensors feeding a compact decoder:

$$
I_\theta(\omega,\lambda)
= D_\theta\left(E(\omega,\lambda), Z_0(\omega), \ldots, Z_L(\omega)\right).
$$

Implementation options:

- tangent-plane multiresolution grids for small patches;
- sphere-aware or HEALPix-like levels for larger areas;
- spectral basis coefficients per spatial latent;
- pretrained trunk plus trainable local latent state.

If possible, implement this as an ordinary field model under `torchsky.map.field` so it can use `render_field_observation` unchanged.

### 4. Adaptive latent atoms

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

### 5. Semantic views and diagnostics

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

Implicit fields and multiscale latent grids should reuse `render_field_observation` whenever they expose the standard field-model interface.

### Sparse adaptive rendering

Adaptive atoms should render sparsely:

1. project atom centers into exposure pixel coordinates using `WCS.world_to_pixel`;
2. compute finite support radius in pixels;
3. build atom-to-tile or atom-to-pixel candidate lists;
4. evaluate kernels only inside local footprints;
5. accumulate with differentiable torch operations;
6. convolve with PSF only if the atom is not already rendered through an analytic PSF path.

Initial implementation may use vectorized chunking before native kernels. Add C++/CUDA only after profiling proves the bottleneck.

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
- no public API stability promise yet;
- one small synthetic example in docs or examples.

Success criteria:

- no changes to existing field rendering behavior;
- no new runtime dependencies;
- design explicitly reuses `SceneRepresentation`, `ContinuousFieldModel`, `FieldObservation`, and `render_field_observation`.

### Milestone 1: field-only implicit sky state

Deliverables:

- `FieldSkyState` or equivalent wrapper around an existing field model;
- `render_scene_observation(state, observation)` or equivalent helper;
- scene-state loss over a list of `FieldObservation` objects.

Success criteria:

- reproduces existing `render_field_observation` output for a one-scene field model;
- tests compare wrapper output to current field path bitwise or within tight tolerance;
- product rendering remains compatible with `TargetRenderSpec`.

### Milestone 2: multi-exposure super-stack fit

Deliverables:

- fitting loop for many exposures sharing one implicit sky state;
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
- memory and runtime scale plausibly with patch size;
- implementation either uses the existing field model interface or clearly documents why it cannot.

### Milestone 4: optional adaptive atom branch

Deliverables:

- untyped compact atom basis;
- finite-support rendering;
- optional soft power-ownership prototype;
- atom regularization terms.

Success criteria:

- improves representation of sharp or undersampled structure when enabled;
- does not become mandatory for smooth-field cases;
- rendering avoids full dense atom-pixel products for moderate atom counts;
- no semantic source labels are required by the optimizer.

### Milestone 5: hybrid implicit super-stack prototype

Deliverables:

- scene state with multiscale implicit branch plus optional adaptive atoms;
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

- field-only scene-state wrapper equals existing field renderer;
- scene-state loss respects masks and weights;
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
- brute-force tiny-scene renderer for correctness only;
- existing direct stack products via `stack_map_product` where product-level comparison is meaningful.

## API sketch

Field-only wrapper:

```python
from torchsky.map.scenes import FieldSkyState

state = FieldSkyState(field_model)
pred = state.render_observation(observation)
```

Product rendering should remain compatible with the existing map-product surface:

```python
from torchsky.map import plan_map_product, evaluate_map_product

product = plan_map_product(observations)
pred_product = render_scene_product(state, product)
metrics = evaluate_map_product(pred_product.data, reference.data, weight=reference.weight)
```

Multiscale implicit state:

```python
from torchsky.map.field import estimate_field_domain
from torchsky.map.scenes import MultiscaleSkyState

domain = estimate_field_domain(observations)
state = MultiscaleSkyState(
    domain=domain,
    levels=(32, 64, 128),
    decoder="mlp",
    spectral_basis=8,
)
```

Optional adaptive atoms:

```python
state.enable_adaptive_atoms(
    count=1024,
    kernel="compact_gaussian",
    spectral_basis=4,
)
```

A fitting loop should accept many observations:

```python
loss = multi_observation_scene_loss(
    state,
    observations,
    loss_kind="charbonnier",
    regularization_weights={
        "multiscale.level_smoothness": 1.0e-5,
        "adaptive_atoms.sparsity": 1.0e-4,
        "adaptive_atoms.overlap": 1.0e-3,
    },
)
```

Names are provisional. The first PR should keep the export surface minimal.

## Open design questions

1. What is the best first multiscale implicit parameterization: latent grid, hash grid, HEALPix-like hierarchy, or pretrained trunk plus local latents?
2. Should adaptive atoms be a branch of the scene state or a separate optimizer-side residual module?
3. How should atom birth/death/splitting be handled without brittle heuristics?
4. What is the cleanest sparse rendering representation on CPU, CUDA, and MPS?
5. Should WCS projection remain differentiable with respect to astrometric nuisance parameters in the first version?
6. When should exact power-cell geometry be worth implementing?
7. How should uncertainty be represented: ensembles, profile scans, conformal residual diagnostics, stochastic residual fields, or a combination?
8. How should the scene state be serialized so it becomes an archive-native product rather than an ephemeral model checkpoint?
9. How can semantic summaries be derived without making them part of the primary optimization target?
10. Should `FieldSkyState` live permanently in `torchsky.map.scenes`, or should a later `scene_state` subpackage own implementation details while `scenes.py` remains the adapter surface?

## Recommended first PR sequence

1. Extend `torchsky.map.scenes` with the thinnest possible `FieldSkyState` wrapper, or document why `ContinuousFieldModel` already suffices.
2. Add tests proving equivalence to existing `render_field_observation` and product rendering paths.
3. Add a tiny synthetic multi-exposure scene-fitting example using `FieldObservation`.
4. Add a multiscale field model under `torchsky.map.field` if it can preserve the standard field-model interface.
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

## Selected references

These references are conceptual anchors and validation targets. They should not be interpreted as required runtime dependencies.

### Astronomical forward modeling, deblending, and coaddition

- Stetson, P. B. 1987, **DAOPHOT: A computer program for crowded-field stellar photometry**, PASP, 99, 191. Classic point-source modeling in crowded fields.
- Bertin, E. & Arnouts, S. 1996, **SExtractor: Software for source extraction**, A&AS, 117, 393. Catalog-oriented detection/measurement baseline.
- Fruchter, A. S. & Hook, R. N. 2002, **Drizzle: A method for the linear reconstruction of undersampled images**, PASP, 114, 144. Exposure-combination baseline for dithered imaging.
- Bertin, E. 2010, **SWarp: Resampling and co-adding FITS images together**, ASPC, 442, 435. Widely used resampling/coaddition baseline.
- Lang, D., Hogg, D. W., & Schlegel, D. J. 2016, **The Tractor: Probabilistic astronomical source detection and measurement**, AJ, 151, 36. Explicit forward modeling of astronomical scenes through images.
- Rowe, B. T. P. et al. 2015, **GalSim: The modular galaxy image simulation toolkit**, Astronomy and Computing, 10, 121. Useful reference for rendering, PSF, and image simulation semantics.
- Melchior, P. et al. 2018/2021, **scarlet: source separation in multi-band images by constrained matrix factorization**. Useful reference for multi-band deblending, constraints, and component interpretability.

### Inverse problems, regularization, and multiscale representations

- Tikhonov, A. N. & Arsenin, V. Y. 1977, **Solutions of Ill-posed Problems**. Classical regularized inverse-problem foundation.
- Mallat, S. 1999/2009, **A Wavelet Tour of Signal Processing**. Multiresolution analysis and scale-localized representations.
- Starck, J.-L., Murtagh, F., & Fadili, J. 2010, **Sparse Image and Signal Processing**. Astronomically relevant sparse/multiscale methods.
- Górski, K. M. et al. 2005, **HEALPix: A framework for high-resolution discretization and fast analysis of data distributed on the sphere**, ApJ, 622, 759. Spherical map discretization reference.
- Narcowich, F. J., Petrushev, P., & Ward, J. D. 2006; Marinucci, D. et al. 2008. Spherical needlets and localized multiscale analysis on the sphere.

### Neural fields and coordinate-based representations

- Ulyanov, D., Vedaldi, A., & Lempitsky, V. 2018, **Deep Image Prior**, CVPR. Demonstrates implicit regularization from neural parameterization.
- Sitzmann, V. et al. 2020, **Implicit Neural Representations with Periodic Activation Functions** / SIREN, NeurIPS. Periodic-coordinate neural fields.
- Tancik, M. et al. 2020, **Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains**, NeurIPS. Coordinate encodings for high-frequency signals.
- Mildenhall, B. et al. 2020, **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**, ECCV. Neural field rendering reference; conceptually useful but not an astronomy-forward-model target.
- Müller, T. et al. 2022, **Instant Neural Graphics Primitives with a Multiresolution Hash Encoding**, SIGGRAPH. Multiresolution hash-grid encoding reference.
- Fridovich-Keil, S. et al. 2023, **K-Planes: Explicit Radiance Fields in Space, Time, and Appearance**, CVPR. Factorized feature-plane reference relevant to sky-wavelength-time fields.
- Xie, Y. et al. 2022, **Neural Fields in Visual Computing and Beyond**, Computer Graphics Forum. Broad neural-field survey.

### Adaptive atoms, splats, and power-cell ideas

- Kerbl, B. et al. 2023, **3D Gaussian Splatting for Real-Time Radiance Field Rendering**, SIGGRAPH. Adaptive Gaussian primitive reference; useful for computational ideas, not a direct astronomy model.
- Aurenhammer, F. 1987, **Power diagrams: Properties, algorithms and applications**, SIAM Journal on Computing. Mathematical basis for weighted Voronoi/power cells.
- Power Foam, 2026, **Power Foam: Unifying Real-Time Differentiable Ray Tracing and Rasterization**. Recent bounded power-cell rendering idea; relevant as inspiration for optional adaptive atoms, not as a core dependency or literal implementation target.

### Stochastic processes and uncertainty

- Kingman, J. F. C. 1993, **Poisson Processes**. Foundation for point processes and random measures.
- Møller, J. & Waagepetersen, R. P. 2004, **Statistical Inference and Simulation for Spatial Point Processes**. Marked point processes and spatial statistics.
- Rasmussen, C. E. & Williams, C. K. I. 2006, **Gaussian Processes for Machine Learning**. Function-space uncertainty and covariance modeling.
- Angelopoulos, A. N. & Bates, S. 2021, **A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification**. Useful entry point for coverage diagnostics on scene residuals.
