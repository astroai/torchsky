# Scenes and operators

!!! warning "Experimental surface"
    `torchsky.scene` is experimental. Production map verbs live in
    [`torchsky.map`](map.md) and field model, fit, render, and sample verbs in
    [`torchsky.field`](field.md).

`torchsky.scene` lets you describe an observation, a target product, and a scene
representation without knowing which lower-level module owns the current
implementation. It provides:

- `scene.FieldObservation`, `scene.ImageObservation`;
- `scene.TargetRenderSpec`, `scene.ProductSpec`, `scene.BandpassSpec`;
- `scene.SceneRepresentation` with tensor, HEALPix, harmonic, and field-model
  adapters;
- `scene.ObservationOperatorAdapter`,
  `scene.operator_adapter_for_observation(...)`, and image-plane operator-set
  binding.

## Render A Scene

<!-- docs: run -->
```python
import torch

from torchsky import scene

header = {
    "NAXIS": 2, "NAXIS1": 32, "NAXIS2": 32,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5, "CRPIX2": 16.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CDELT1": -0.05 / 3600.0, "CDELT2": 0.05 / 3600.0,
}
target = scene.target_from_header(header, value_semantics="surface_brightness")
prediction = scene.render(torch.zeros(target.shape), target)
assert prediction.shape == target.shape
```

For differentiable field models, `scene.predict(...)` and
`scene.predict_observation_mean(...)` call the same field-observation path as
`torchsky.field`.

`scene.image_observation(path)` builds an `ImageObservation` from a file path
(read with `torchfits`) when you start from FITS rather than tensors.

## Operators With Exact Adjoints

`scene.operator_adapter_for_observation` wraps the `ObservationOperatorBundle`
used by map and field rendering. It exposes enabled operator names, slot checks,
and the deterministic prediction path, with the bundle available as `.bundle`.
Binding it to a target gives a matrix-free operator \(A\) whose adjoint is an
exact vector-Jacobian product of the same projection, so normal-equation solves
use \(A^T N^{-1} A\) without forming a matrix.

<!-- docs: run -->
```python
import torch

from torchsky import scene
from torchsky.map import ImageObservation

header = {
    "NAXIS": 2, "NAXIS1": 32, "NAXIS2": 32,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5, "CRPIX2": 16.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CDELT1": -0.05 / 3600.0, "CDELT2": 0.05 / 3600.0,
}
target = scene.target_from_header(header, value_semantics="surface_brightness")

yy, xx = torch.meshgrid(torch.arange(32.0), torch.arange(32.0), indexing="ij")
sky = torch.exp(-((xx - 16) ** 2 + (yy - 16) ** 2) / 18.0)

# An exposure offset by a fraction of a pixel from the target grid.
shifted = {**header, "CRVAL1": 150.0 + 0.4 * 0.05 / 3600.0}
observation = ImageObservation(data=sky, wcs=shifted)
operator = scene.operator_adapter_for_observation(
    observation, resampling_kernel="bilinear"
).bind_image_plane_target(target)

# Dot-product test: <A u, v> == <u, A^T v>.
u, v = torch.randn(32, 32), torch.randn(32, 32)
lhs = (operator.apply(u) * v).sum()
rhs = (u * operator.apply_adjoint(v)).sum()
torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-4)

data = operator.apply(sky)
solved = operator.solve_normal_equation_cg(data, iterations=8)
assert solved.solution.shape == sky.shape
```

Exact adjoints are explicit. Pass `apply_adjoint` or a target binder when you bring
your own forward model through `adapter.bind_target(target, apply_operator=...,
apply_adjoint=..., apply_noise=...)`; the default render path stays available for
prediction-only work. Scalar weights and explicit weight, mask, and primary-beam
maps feed the inverse-noise hook.

For several observations on one latent grid use
`scene.bind_image_plane_observations_to_target(...)`: the normal equation is
summed across observations in exposure order, and `exposure_chunk_size` caps how
many predictions are retained at once. `iter_apply_chunks(...)` exposes the same
bounded execution.

## Inverse-Curvature Diagonal

The bound operator estimates a diagonal of \((H + \epsilon I)^{-1}\), with
\(H = A^T N^{-1} A\), by seeded Hutchinson probing, each solved by matrix-free
CG:

```python
# fragment: operator_set from bind_image_plane_observations_to_target
inverse_curvature_diagonal = operator_set.estimate_posterior_variance_diagonal(
    reference_map, num_probes=64, seed=7, iterations=32, damping=1.0e-6,
)
```

Every solve is checked against the actual normal action and an inadequate or
non-finite solve raises instead of returning an uncertainty. Caveats worth
internalizing:

- The method name is a misnomer: this is inverse curvature, not automatically a
  posterior variance. A sampling-covariance reading needs an estimable,
  explicitly gauge-fixed subspace and the declared noise assumptions; with a
  penalty \(P\) the estimator covariance is \((H+P)^{-1}H(H+P)^{-1}\).
- Finite-probe estimates can be negative where the true diagonal is positive.
  They are returned signed because clipping would bias them. Increase
  `num_probes` or evaluate selected functionals if the Monte Carlo noise matters.
- A small residual does not bound inverse error without conditioning information,
  and finite probes do not certify global identifiability.

## Related

- [Native likelihood](native-likelihood.md) for Gaussian likelihood factors on
  bound operators.
- [PSF conventions](psf-response-conventions.md) for kernel orientation and
  flux semantics.
- [Fields](field.md) for continuous scene models.
