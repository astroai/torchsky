# INR / Power-Foam audit and next primitive

Date: 2026-07-22
Scope: read-only production-code audit; this note is a proposal, not an API commitment.

## Decision

Torchsky already has a substantial continuous-field/INR stack and one experimental
residual-driven Gaussian basis wrapper. It does **not** implement Power Foam: there
is no power-distance evaluation, soft or exact power-cell ownership, compact cell
renderer, or sparse atom-to-pixel candidate structure in production code.

The next primitive should not be a speculative Foam renderer. Add
**deterministic pixel-footprint quadrature to continuous-field observation
rendering**, reusing the 2x2 Gauss-Legendre WCS sampling already used by image-plane
mapmaking. Compact adaptive atoms are exactly where pixel-centre sampling is least
scientifically defensible; fixing the observation operator benefits every INR and
any later Foam-inspired representation.

## What is shipped now

### Continuous field and scene substrate

- `src/torchsky/map/scenes.py` has `SceneRepresentation` and
  `ContinuousFieldModel`. Any compatible `nn.Module` can render to a
  `TargetRenderSpec` through the standard field observation path. The archived
  plan's proposed `FieldSkyState` would currently duplicate this adapter.
- `src/torchsky/map/field/rendering.py` performs WCS pixel-to-world sampling,
  deterministic or Monte Carlo bandpass integration, optional exposure-pose
  adjustment, model evaluation, PSF application, and background application.
  It accepts an explicit `ObservationOperatorBundle` and has spatial tiling for
  the no-PSF path.
- `src/torchsky/map/field/implicit_forward.py` has shared-model multi-survey
  prediction and Charbonnier loss, per-observation context, rest-frame spectral
  remapping, kinematics, and per-observation operator bundles. This covers much
  of archived milestones 1 and 2 without introducing a second scene hierarchy.
- `torchsky.scene` exposes the same operator-bundle substrate used by map and
  inverse paths. Image-plane observations additionally have exact VJP adjoints
  and bounded multi-exposure operator sets.

### INR representations

- The model zoo includes SIREN-like and WIRE-like implicit fields, Fourier MLPs,
  spectral-basis models, sky-coordinate encoders, multiresolution hash grids,
  sky-lambda planes/K-planes, feature fields, atlases, conditioned fields, and
  operator-constrained variants (`src/torchsky/map/field/core.py`).
- Multiscale capacity is real, not merely roadmap text: `HashGridEncoder` provides
  dense-coarse/hashed-fine levels; K-plane and sky-lambda feature models provide
  structured spatial-spectral latent representations. The concise user surface
  has `fast`, `balanced`, `spectral`, and `operator_constrained` presets, while
  `RECOMMENDED_FIELD_MODEL_KINDS` identifies the narrower benchmarked set.
- Training supports observation/patch sampling, several robust likelihoods,
  band balancing, spectral sampling and curricula, pose refinement, component
  constraints, diagnostics, and device-aware execution.
- Rendering includes image observations, arbitrary world-coordinate samples,
  cubes, HEALPix maps, and cube-via-HEALPix paths.

### Evidence and diagnostics

- Bounded runtime contracts are tested in the core field/map tests, including
  loss behavior, gradients, tiling, and device handling.
- Survey-scale throughput, stack comparisons, and real-data evaluation belong
  to [`torchsky-harness`](https://github.com/astroai/torchsky-harness); the
  harness records the exact subject revision and evaluation evidence.
- Field residual products, support-aware scorecards, multiscale residual metrics,
  PSF alignment diagnostics, and residual-tile ranking exist.

## What is Foam-like versus what is not

### Implemented prerequisites

- `select_field_residual_tiles(...)` ranks fixed detector-image tiles by exact
  weighted residual energy with explicit support filtering.
- `ResidualTileBasisField` converts selected tile centres/scales through the
  observation WCS into encoded field coordinates and adds signed Gaussian
  corrections to a diffuse base field.
- The correction is an ordinary field model, so it automatically passes through
  the existing WCS, bandpass, PSF, background, loss, and scene adapter paths.
- Atom count is bounded by caller-supplied `max_tiles`, amplitudes are initialized
  from weighted residual means, and scales can optionally be learned.

These are useful adaptive-capacity prerequisites, but they are not a Power-Foam
implementation.

### Roadmap-only / absent

- Power distances `||x-p_i||^2-r_i^2`.
- Normalized soft ownership or exact clipped power cells.
- Compact-support tapers or exact bounded-cell support.
- Sparse atom-to-tile/atom-to-pixel candidate lists and sparse accumulation.
- A composite implicit-plus-adaptive scene state with explicit regularization
  terms.
- Atom split/merge/birth/death rules, overlap penalties, or calibrated sparsity
  controls.
- Multi-exposure persistence tests that distinguish sky residuals from
  detector/exposure-specific residuals.
- Foam-specific real-data, memory-scaling, throughput, or crossover benchmarks.

There are also two important limits in the existing adaptive code:

1. `ResidualTileBasisField.residual_correction` broadcasts every queried point
   against every atom and directly sums Gaussian values. Its execution and
   intermediate storage scale as `O(number_of_queries * number_of_atoms)`; it is
   not a sparse renderer.
2. Gaussian support is mathematically infinite. The implementation has a bounded
   **atom count**, not bounded spatial support. Calling it a bounded or sparse
   field should not be interpreted as a finite-footprint guarantee.

The initializer also consumes one residual product from one observation. Without
cross-exposure agreement or an exposure-tied nuisance branch, it can promote a
detector artifact into the persistent sky state. This is a reason to keep it
experimental, as the active roadmap already does.

## One next primitive: pixel-footprint-integrated field rendering

### Mathematical contract

For a detector pixel centred at `(x_p, y_p)`, provide an opt-in 2x2
Gauss-Legendre approximation to the pixel average:

```text
offset = 1 / (2 * sqrt(3))
I_bar[p] = (1/4) * sum_{Δx,Δy in {-offset,+offset}}
    I(WCS(x_p + Δx, y_p + Δy), wavelength)
```

The spatial quadrature nests inside the existing spectral quadrature. Exposure
pose is applied to every world-coordinate sample. The resulting pixel-integrated
plane then follows the current PSF and background path; mask/weight/noise semantics
remain in the existing likelihood/operator layers.

The existing private `_quadrature_target_world_grids(...)` in
`src/torchsky/map/windowing.py` already uses these exact nodes for distorted-WCS
flux reprojection. Extract or reuse that node/weight definition rather than
inventing a second convention. The continuous-field API only needs one narrow
option, for example:

```python
render_field_observation(
    model,
    observation,
    spatial_sampling="gauss_legendre_2x2",  # default remains "center"
)
```

Do not add a new model class. Implement the four spatial samples inside the
existing tile loop so memory remains bounded; do not cache eight extra full-image
world grids for survey-sized images.

### Why this is the correct next step

- It closes a mismatch with the archived response-kernel equation, which includes
  integration over detector pixel footprints, while the current field renderer
  samples only pixel centres.
- It is representation-independent: SIREN, hash grids, K-planes, feature atlases,
  Gaussian residual atoms, and future soft power cells all benefit.
- Compact/subpixel adaptive structure is the worst case for centre sampling.
  Adding power ownership before anti-aliased observation rendering would fit
  aliases and make dither-dependent residuals look like scene capacity.
- It composes with current `FieldObservation`, WCS, exposure pose, spectral
  response, PSF/background, `ObservationOperatorBundle`, scene adapters, and
  multi-survey losses.
- It follows the active roadmap's operator-first priority and reuses existing
  torchsky map mathematics instead of widening the experimental API.

### Minimal implementation surface

- `src/torchsky/map/field/rendering.py`: opt-in spatial quadrature in
  `render_field_observation`; thread the option through the distribution fallback
  without applying operators twice.
- `src/torchsky/map/windowing.py` or one small shared operator helper: expose the
  existing 2x2 offsets and equal weights to both image-plane and field rendering.
- `src/torchsky/map/scenes.py`, `src/torchsky/map/field/implicit_forward.py`, and
  the concise `torchsky.field.render(...)` facade only need transparent keyword
  forwarding if their current forwarding is incomplete.
- One focused test file or additions to `tests/test_map_field.py`; one benchmark
  row in the existing INR/map-field harness. No dependency and no new package.

### Acceptance gates

1. `spatial_sampling="center"` is the default and preserves current results.
2. A constant field is exactly preserved by centre and 2x2 quadrature through a
   distorted WCS.
3. For a narrow analytic Gaussian shifted within a pixel, 2x2 quadrature has
   lower error than centre sampling against a much denser deterministic reference;
   repeat across several subpixel phases rather than selecting one favourable
   phase.
4. Two dithered observations of the same compact field become more mutually
   consistent under footprint integration, measured before fitting.
5. Gradients to field parameters and adaptive atom centres/scales are finite and
   agree with finite differences on a tiny case.
6. Explicit PSF/background bundles are applied once, and opt-out bundles still
   suppress those operators exactly as today.
7. Tiled and untiled results agree within a stated tolerance; peak memory is
   bounded by tile size rather than full-image area times four nodes.
8. Benchmark centre versus 2x2 sampling at small, medium, and representative
   sizes on CPU and available accelerators, reporting the expected roughly 4x
   model-query work and the measured science-error improvement.

### Explicit non-goals for this milestone

- No Power-Foam class or public `FoamScene` name.
- No exact power-diagram construction.
- No soft ownership, atom birth/death, or semantic source labels.
- No native kernel before the tensor implementation is benchmarked.
- No claim that fixed 2x2 quadrature resolves arbitrarily narrow structure; it is
  a controlled baseline whose residual phase error can justify later adaptive or
  integrated-encoding work.

After this gate is green, the smallest honest Foam-specific experiment would be a
compact partition-of-unity correction over a **caller-provided fixed atom set**,
still behind the experimental field namespace. It should not precede the
pixel-response primitive above.
