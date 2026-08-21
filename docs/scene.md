# `torchsky.scene`

`torchsky.scene` is the promoted namespace for joint sky-modelling concepts. It is currently an additive facade over the mature `torchsky.map` and `torchsky.map.field` implementation, so existing map imports keep working.

Use this namespace for the mission-level objects:

- `scene.FieldObservation` and `scene.ImageObservation`
- `scene.TargetRenderSpec`
- `scene.MapProductSpec` and `scene.MapBandpassSpec`
- `scene.SceneRepresentation` plus tensor, HEALPix, harmonic, and field-model adapters
- `scene.ObservationOperatorAdapter`, `scene.operator_bundle_for_observation(...)`, and image-plane operator-set binding
- `scene.sample_image_plane_batched_channels(...)` and
  `scene.splat_to_native(...)` for differentiable native-frame query/deposit

The intent is that users can describe an observation, a target product, and a scene representation without needing to know which lower-level map module owns the current implementation.

## Minimal Example

```python
import torch
from torchsky import scene

# Define a WCS header dictionary
header = {
    "NAXIS": 2,
    "NAXIS1": 32,
    "NAXIS2": 32,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5,
    "CRPIX2": 16.5,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CDELT1": -0.05 / 3600.0,
    "CDELT2": 0.05 / 3600.0,
}

# Construct render target spec and render a zero scene to it
target = scene.target_from_header(header, value_semantics="surface_brightness")
prediction = scene.render(torch.zeros(target.shape), target)
```

For differentiable field models, `scene.predict(...)` and `scene.predict_observation_mean(...)` call the same current field-observation path used by `torchsky.map.field`.

## Operator Adapter

`scene.operator_adapter_for_observation(...)` wraps the existing `ObservationOperatorBundle` used by map and field rendering. It exposes enabled operator names, slot checks, and the deterministic prediction path while keeping the underlying bundle available through `.bundle` for lower-level integrations.

The adapter also binds the bundle to a `TargetRenderSpec` through the same normal-equation substrate used by the map solvers:

```python
adapter = scene.operator_adapter_for_observation(observation)
operator = adapter.bind_target(
    target,
    apply_operator=my_forward_model,
    apply_adjoint=my_exact_adjoint,
    apply_noise=my_inverse_noise,
)

probe = operator.normal_equation_probe(scene_tensor, data)
result = operator.solve_normal_equation_cg(data, iterations=8)
```

Exact adjoints are explicit: pass `apply_adjoint` or a target binder for the operator family being solved. The default render path remains available for prediction-only workflows.

For image-plane inverse problems, `ObservationOperatorAdapter.bind_image_plane_target(...)` binds one `ImageObservation` to a latent target grid and uses an exact vector-Jacobian product as the adjoint of the same projection:

```python
observation = scene.image_observation("science.fits")
target = scene.target_from_header(target_header)
operator = scene.operator_adapter_for_observation(
    observation,
    resampling_kernel="bilinear",
).bind_image_plane_target(target)

result = operator.solve_normal_equation_cg(observed_image, iterations=8)
```

Scalar observation weights and explicit weight/mask/primary-beam maps feed the bound operator's inverse-noise hook. Use `scene.bind_image_plane_observations_to_target(...)` when several image-plane observations share the same latent target grid. It returns the multi-observation operator set used by the map solver substrate, so `A^T N^-1 A` and `A^T N^-1 y` are summed across observations. Set `exposure_chunk_size` to cap the number of prediction tensors retained at once; iterative solves use this matrix-free path while preserving exposure-order accumulation. `iter_apply_chunks(...)` exposes the same bounded execution for consumers that do not need all predictions materialized as one tuple.

The same bound operator can estimate posterior map variance without forming a
dense covariance matrix:

```python
variance = operator_set.estimate_posterior_variance_diagonal(
    reference_map,
    num_probes=64,
    seed=7,
    iterations=32,
    damping=1.0e-6,
)
```

This is a seeded Hutchinson estimate of
`diag((A^T N^-1 A + damping I)^-1)`. Compose a prior operator into the set to
include a non-identity prior precision. Each probe is solved independently by
matrix-free CG, so memory stays bounded by one probe and the solver workspace;
finite probe count and incomplete CG convergence both contribute estimation
error. Negative finite-probe estimates are clipped to the positive machine
floor so the returned variance remains usable; that clipping is biased near
zero and is not a substitute for increasing probes or checking CG convergence.
