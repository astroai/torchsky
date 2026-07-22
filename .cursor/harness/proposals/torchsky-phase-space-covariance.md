# Batched phase-space Jacobian and covariance design

Status: implementation proposal, 2026-07-22. This audit made no production-code
changes.

## Decision

Add one public primitive, not a covariance class or a raw RA/Dec-covariance API:

```python
def propagate_space_motion_with_jacobian(
    lon_deg: Any,
    lat_deg: Any,
    pm_lon_coslat_mas_yr: Any,
    pm_lat_mas_yr: Any,
    parallax_mas: Any,
    radial_velocity_km_s: Any,
    *,
    source_epoch_jyear: Any,
    target_epoch_jyear: Any,
    create_graph: bool = False,
) -> tuple[
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    Tensor,
]: ...
```

The first item is exactly the existing six-output state, computed once. The
second has shape `broadcast_shape + (6, 6)`, with axes `[..., output, input]`.
Returning the propagated state with the Jacobian avoids making a survey-scale
consumer run the relativistic/light-time kernel twice.

Do not add a second public covariance helper yet. Linear covariance propagation
is already one stable PyTorch expression:

```python
cov_out = jacobian @ cov_in.to(jacobian) @ jacobian.mT
cov_out = 0.5 * (cov_out + cov_out.mT)
```

Document that expression next to the new API. Add a helper only if a second
real consumer needs common covariance validation or non-linear uncertainty
policy. This keeps distribution assumptions out of the coordinate primitive.

## Jacobian coordinates and units

The nominal state continues to use the current public units and ordering:

```text
(lon [deg], lat [deg], pm_lon_coslat [mas/yr], pm_lat [mas/yr],
 parallax [mas], radial_velocity [km/s])
```

The Jacobian acts on **local catalogue perturbations**, at both the source and
propagated directions:

```text
(d_lon_coslat [mas], d_lat [mas], d_pm_lon_coslat [mas/yr],
 d_pm_lat [mas/yr], d_parallax [mas], d_radial_velocity [km/s])
```

Here `d_lon_coslat = cos(lat) * d_lon`; it is a differential, not the global
quantity `lon * cos(lat)`. This is the convention used by Gaia astrometric
covariances and makes the positional `2 x 2` block directly usable as a local
east/north error ellipse. It also prevents a 0/360-degree wrap from inflating a
covariance. Epochs are fixed parameters, not covariance coordinates.

For an autograd Jacobian `J_raw` in the six nominal state units, let
`M = 3_600_000` mas/deg. Convert it without approximating the propagation:

```text
A_in  = diag(1 / (M cos(lat_in)), 1 / M, 1, 1, 1, 1)
B_out = diag(M cos(lat_out),      M,     1, 1, 1, 1)
J_local = B_out @ J_raw @ A_in
```

This conversion includes all perspective, basis-rotation, parallax, and radial-
velocity cross terms already present in `J_raw`.

Do not publish raw longitude covariance as the primary result. Longitude is a
chart coordinate, discontinuous at the wrap and undefined at a pole. With the
minimal scaling implementation above, reject exactly `abs(lat) == 90 deg` at
either endpoint with a contextual error: the state propagation remains valid,
but its longitude-chart Jacobian is not. Do not invent a near-pole threshold.
Near-pole states must remain supported and tested. Exact-pole covariance support
would justify a later Cartesian phase-space differential kernel, not a hidden
epsilon in this API.

## Batched autodiff implementation

The current scalar `torch.func.jacrev` test is valid, but
`vmap(jacrev(public_function))` is not a suitable implementation: the public
function contains value-dependent `bool(torch.all(...))` validation and a
value-dependent convergence fallback. Those operations graph-break or fail
under `vmap`.

Use the smallest existing repository pattern instead:

1. Broadcast and validate the batch once at the public boundary.
2. Evaluate a branch-free internal tensor core on the full batch. If the
   relativistic correction requires the 100-iteration fallback, evaluate the
   whole accepted batch with that count so the returned state and derivative
   describe the same branch.
3. Stack the six inputs as `coords[..., 6]` and the six outputs as
   `result[..., 6]`.
4. Because rows are independent, compute six vector-Jacobian products:
   `grad(result[..., i].sum(), coords)` and stack them at `dim=-2`.
   This produces per-row blocks directly, with no Python loop over catalogue
   rows and no `vmap` requirement.
5. Apply the local-coordinate matrices above. Follow the WCS Jacobian convention
   with `create_graph=False` by default; when true, avoid detaching nominal
   tensor inputs so higher derivatives remain possible.

This costs six reverse passes, is `O(N)`, preserves tensor batching, and avoids
adding a dependency or a bespoke analytic derivative. A future forward-mode
or analytic kernel should be benchmark-driven, not part of the first API.

## Shape, dtype, and device contract

- All six observables and both epochs broadcast exactly as
  `propagate_space_motion` does now. Each state output has that broadcast shape;
  the Jacobian appends `(6, 6)`.
- The Jacobian is always on the longitude input's device.
- Preserve the current frame-module contract: `_as_tensor` promotes floating
  tensor inputs to `float64`, so state and Jacobian are `float64`. Do not claim
  float32 preservation in this API while the underlying transform does not do
  it.
- CPU and CUDA float64 are supported. MPS does not provide the required
  float64 path; fail clearly or leave it unsupported rather than silently
  reducing astrometric precision.
- The caller's covariance should be finite, have trailing shape `(6, 6)`, and
  use the local units/order above. The one-line propagation naturally supports
  leading-dimension broadcasting. Validation of symmetry/PSD belongs in a
  future covariance helper or the catalogue ingestion boundary; an eigendecompose
  of every survey row should not be added to this coordinate kernel.

For Xmatch, the propagated positional ellipse is
`cov_out[..., :2, :2]` in mas-squared. Divide it by `1e6` for arcsec-squared.
Gaia's published five-parameter order must be explicitly rearranged into
Torchsky's order; radial-velocity variance/correlation policy remains an
Xmatch/catalogue concern.

## Parity and regression gate

Keep focused tests in `tests/test_wcs_frames.py`:

1. **Identity epoch:** away from a pole, state is unchanged and the local
   Jacobian is identity within float64 roundoff. This immediately catches unit
   or state-order mistakes.
2. **ERFA finite-difference parity:** for ordinary, Barnard-like, wrap-crossing,
   approaching/receding, distant positive-parallax, and `abs(lat)=89.9` states,
   central-difference `erfa.starpm`. Perturb source position in local east/north
   units and compare propagated residuals in the output local frame. Use
   wrapped longitude deltas; never subtract raw longitudes across 0/360.
3. **Astropy parity:** repeat one ordinary and one high-proper-motion case with
   `SkyCoord.apply_space_motion` and TCB Julian-year epochs, comparing both the
   nominal six-state result and deterministic `+/-` perturbation responses.
   Astropy is a state/finite-difference oracle; it has no analytic covariance API.
4. **Batch independence:** heterogeneous `(B, 6)` states and broadcast epochs
   give `J.shape == (B, 6, 6)` and equal a stack of scalar calls. Include scalar,
   `B=1`, `B>1`, and negative time intervals.
5. **Covariance algebra:** a correlated SPD covariance yields symmetric PSD
   `J C J^T`; at identical epochs it is unchanged. Add a deterministic sigma-
   point or small seeded Monte-Carlo check only to demonstrate the documented
   first-order regime, not as the sole Jacobian oracle.
6. **Charts:** crossing 360 degrees and near both poles remain finite and agree
   with ERFA/Astropy in local coordinates; exact source/output poles fail the
   Jacobian API explicitly.
7. **Autograd:** `create_graph=True` permits a finite derivative of a Jacobian
   scalar for an ordinary physical state. Existing direct differentiation of
   `propagate_space_motion` remains supported.
8. **Devices/dtypes:** CPU always; conditional CUDA float64 equality and device
   retention. Assert current float32 input promotion to float64. Do not add a
   pretend MPS parity test.
9. **Failures:** retain the existing incomplete/unphysical state checks and add
   exact-pole Jacobian failures. Validation must happen before the unchecked
   core.

Derive finite-difference steps per coordinate and commit the convergence sweep
that justifies the final tolerances. A single epsilon in mixed degrees,
mas/year, mas, and km/s is not scientifically meaningful.

## What is deliberately out of scope

- no uncertainty-distribution or Monte-Carlo API;
- no raw RA/Dec covariance promise;
- no Gaia-column parsing in Torchsky;
- no exact-pole gauge convention;
- no new dependency, dataclass, or native kernel;
- no throughput claim until state-plus-Jacobian is benchmarked at catalogue
  sizes on CPU and CUDA.

