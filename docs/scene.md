# `torchsky.scene`

`torchsky.scene` is an **experimental** namespace for scene representations and
operator adapters. Production map verbs live in `torchsky.map`; field
model/fit/render/sample verbs live in `torchsky.field`.

Use this namespace for:

- `scene.FieldObservation` and `scene.ImageObservation`
- `scene.TargetRenderSpec`
- `scene.ProductSpec` and `scene.BandpassSpec`
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

The same bound operator can estimate an inverse-curvature diagonal without
forming a dense covariance matrix. The historical method name is retained:

```python
inverse_curvature_diagonal = operator_set.estimate_posterior_variance_diagonal(
    reference_map,
    num_probes=64,
    seed=7,
    iterations=32,
    damping=1.0e-6,
    rtol=1.0e-5,
    atol=0.0,
)
```

This is a seeded Hutchinson estimate of
`diag((H + damping I)^-1)`, where `H = A^T N^-1 A`, for a fixed Hermitian
positive-definite system on the declared state space. Each probe is solved
independently by matrix-free CG, so memory stays bounded by one probe and the
solver workspace. Every solution is checked against the actual normal action:
`||(H + damping I)x - z|| <= max(atol, rtol ||z||)`. An inadequate or non-finite
solve raises instead of returning uncertainty. Shape, dtype and device must
agree with the reference state. Finite residual checks and a finite probe set
do not certify global identifiability; a small residual also does not bound
inverse error without conditioning information.

Finite-probe Hutchinson estimates can be negative even when the true inverse
diagonal is positive. Returned estimates preserve those signed values; clipping
would bias them and turn Monte Carlo noise into false tiny variances. Increase
the probe budget or evaluate selected functionals when that noise is too large
for the query. The raw signed estimate is not a per-pixel uncertainty guarantee.

With no regularization, inverse information has the usual fixed-model sampling
covariance interpretation only on an estimable, explicitly gauge-fixed
subspace and under the declared noise/nuisance assumptions. Damping or composing
a quadratic penalty operator changes the target to inverse penalized curvature.
For a penalty `P`, the regularized estimator's sampling covariance is
`(H + P)^-1 H (H + P)^-1`, not `(H + P)^-1`; mean transfer and possible bias
must also be considered. A Bayesian posterior interpretation requires its
declared prior and likelihood assumptions.
