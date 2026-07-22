# Phase-space covariance: validation matrix

## Audit findings (2026-07-22)

The existing frame coverage is a good base, but it is not yet a covariance
test suite:

- `tests/test_wcs_frames.py` checks `propagate_space_motion` against both
  ERFA `starpm` (including a high-velocity case) and `SkyCoord.apply_space_motion`
  for ordinary physical states.
- It checks one scalar `torch.func.jacrev` result only for shape and finiteness;
  it does not check its values, covariance propagation, dtype, device, longitude
  wrapping, or a batch of independent Jacobians.
- There is no phase-space-motion benchmark. Existing WCS and catalogue
  benchmarks do not exercise this operator, so no performance claim should be
  added merely by adding a covariance API.
- Astropy 8.0's `SkyCoord.apply_space_motion(new_obstime=...)` is the right
  independent mean reference, but it has no covariance-propagation API. It
  calls ERFA `pmsafe` internally and will silently use its distance handling;
  Torchsky's positive-parallax contract remains the authoritative input
  contract. Use Astropy only for Monte-Carlo output statistics of clearly
  physical samples, not as an analytic-Jacobian oracle.

### Important implementation constraint

`torch.vmap(torch.func.jacrev(single_state_fn))(states)` currently fails:

```
RuntimeError: vmap ... calling .item() / data-dependent control flow
```

The `bool(torch.all(...))` validation branches in `propagate_space_motion`
cause that failure. A public batched Jacobian/covariance path therefore needs
validation at the public boundary and a branch-free tensor core (or an
equivalent vectorized implementation). Do not promise `vmap` support until the
test below passes.

## Required API semantics before tests

The input state is the catalogue vector
`(lon_deg, lat_deg, pm_lon_coslat_mas_yr, pm_lat_mas_yr, parallax_mas,
radial_velocity_km_s)` and the input covariance is in exactly those units.
For the output covariance, do **not** publish a raw longitude/latitude
covariance as the primary scientific representation: longitude is discontinuous
at 0/360 and undefined at the poles. Define a local tangent-plane output for
the positional block, e.g. `(d_lon_coslat, d_lat, pm_lon_coslat, pm_lat,
parallax, radial_velocity)`, explicitly naming the output tangent point and
units. If a raw observable covariance helper is also offered, document it as a
chart-dependent convenience and test it only away from poles/wrap.

The covariance contract should be linearized:
`C_out = J C_in J^T`, symmetrized as `(C_out + C_out.transpose(-1, -2)) / 2`.
It should state that this is first-order propagation, not an uncertainty
distribution transform. Reject non-finite, non-symmetric, or non-PSD
covariances at the public boundary (with a small dtype-aware numerical
tolerance only if the API deliberately accepts nearly-PSD catalogue input).

## Compact test matrix

Put focused tests next to the present ones in `tests/test_wcs_frames.py`; keep
the Monte-Carlo test seeded and small enough for the normal test environment.

| Test | State / assertion | Reference and tolerance |
| --- | --- | --- |
| Identity epoch | A normal, non-polar, physical float64 state and SPD covariance, `target == source`. Output state equals input; `J == I`; `C_out == C_in`, including off-diagonal terms. | `torch.testing.assert_close`, float64 `atol=2e-11`, `rtol=2e-11` for state/J; covariance tolerance scaled by `C_in`. This catches unit/order mistakes immediately. |
| Autograd vs central difference | Same state propagated 25 yr. Compare every element of the public Jacobian to a central difference with per-coordinate perturbations chosen in native units (recommended: `[1e-5 deg, 1e-5 deg, 1e-3 mas/yr, 1e-3 mas/yr, 1e-4 mas, 1e-4 km/s]`; tune from observed convergence, never one universal epsilon). Compare longitude residuals with a wrapped difference or test the tangent output. | Float64, relative/absolute tolerances derived from the finite-difference convergence sweep and committed in the test. This validates derivatives independently of the same autograd implementation used to form `J`. |
| Linear covariance vs sampled distribution | Fixed seed; draw 10k small Gaussian perturbations from a well-conditioned correlated 6x6 covariance around a Gaia-like state, propagate all draws, form output residuals in the *same local tangent basis*, and compare empirical mean/covariance to `f(mean)`/`J C J^T`. | Use Torch samples for the actual distribution. Astropy `SkyCoord.apply_space_motion` on a deterministic 1k subset is an independent mean/output-coordinate check. Keep fractional uncertainty small enough that first-order error is expected; test normalized covariance discrepancy rather than falsely demanding bit parity. |
| Astropy parity with perturbations | 8--16 physical samples spanning ordinary star, high-PM nearby star, distant low-parallax-but-positive source, approaching/receding RV, RA close to 0/360, and `abs(dec)=89.9 deg` (not exact pole). Propagate nominal and a deterministic small ± perturbation in each coordinate. | Compare means and finite-difference response in a tangent plane to Astropy. Use `Time(..., format="jyear", scale="tcb")` for Torchsky's declared epoch convention; record Astropy's TDB conversion as an expected comparator difference, not a Torchsky bug. |
| Batch independence | `(B, 6)` states and `(B, 6, 6)` SPD covariances, with heterogeneous epochs. Assert result shapes `(B, 6, 6)`, symmetry, PSD within tolerance, and equality to stacking the scalar public call. Include `B=1` and `B=3`. | This is the regression for the current `vmap` failure and guards against accidental cross-source blocks. |
| Float32/float64 | Run the same benign batch as float32 and float64. Assert output dtype/device follows input/API contract, output finite, covariance symmetric and PSD. Compare float32 to float64 after casting only with scientifically defensible loose tolerances; do not demand Astropy-level microarcsecond parity in float32. | Use a 25 yr baseline, avoid pole/wrap. Start with `rtol~2e-4`, then tighten only after measurement. Float64 remains the ERFA-parity scientific path. |
| Longitude wrap and near-pole charts | RA 359.9999 deg crossing 0 deg and Dec ±89.999 deg. Assert local-tangent covariance stays finite, symmetric, PSD, and agrees with Monte Carlo. If raw RA/Dec covariance is exposed, assert/document that it is unavailable or ill-conditioned at exact poles. | Never assert raw RA equality by subtraction; use wrapped residuals or 3D/tangent coordinates. Exact-pole longitude must be rejected or have an explicitly defined convention. |
| CUDA (conditional) | `if torch.cuda.is_available()`: float64 and float32 batch output/J/covariance vs CPU using the same tensors. | Assert device preservation and CPU/CUDA agreement. Run only conditionally in the normal suite; do not add a fake CUDA requirement. |
| Input failures | Wrong trailing dimensions, non-finite state/covariance, non-symmetric covariance, clearly indefinite covariance, nonpositive parallax, and superluminal state. | Check contextual `ValueError`s; no partial output/fallback. |

## Benchmark recommendation

Do not add a benchmark gate in the first covariance PR: correctness and API
semantics are the risk. Once the branch-free batched core exists, add one small
`benchmarks/bench_space_motion.py` (or a row in the existing benchmark runner)
that reports, without a pass/fail claim:

- state-only throughput and `state + J + J C J^T` throughput;
- CPU float64 and CUDA float32/float64 when available;
- `N = 1, 1_000, 100_000`, fixed physical states/covariances;
- peak-memory or allocation count for the covariance path.

The baseline should be Torchsky state-only versus Torchsky covariance, not
Astropy: Astropy's Python/ERFA scalar-oriented route is a semantic reference,
not a fair survey-scale batch performance comparator.

## Minimal implementation order

1. Specify and implement the tangent-plane covariance chart plus validated
   `(B, 6)` / `(B, 6, 6)` public API.
2. Land identity, finite-difference, batch, failure, and dtype tests.
3. Add seeded Torch Monte Carlo plus small Astropy parity subset.
4. Add conditional CUDA coverage.
5. Measure before deciding whether a benchmark belongs in CI/documentation.
