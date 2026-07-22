# Angular-motion covariance audit

Status: implementation review, 2026-07-22. No production code was changed by
this audit.

## Verdict

The proposed fallback is worthwhile and the narrow API is correct:

```python
propagate_proper_motion_with_jacobian(...) \
    -> ((lon_deg, lat_deg), jacobian[..., 2, 4])
```

The Jacobian maps source
`(position_east_mas, position_north_mas, pm_lon_coslat_mas_yr,
pm_lat_mas_yr)` into destination `(position_east_mas,
position_north_mas)`. The source and destination positional axes are the local
tangent bases at their respective nominal coordinates. This is the only useful
contract for Gaia covariance transport: a raw RA derivative is discontinuous
at 0/360 degrees and singular at a pole.

Torchsky's existing angular kernel is already branch-free, broadcasted, and
differentiable. Reuse the same local-perturbation plus
`vmap(jacrev(...))` pattern as the 6D primitive; no analytic derivative, new
dependency, covariance class, or generic uncertainty API is needed.

## Scientific scope

This transports **formal random uncertainty under the same angular model used
for the mean**. It is not a complete prediction covariance when distance or RV
is unavailable. In particular:

- the model does not marginalize over an unknown RV;
- it omits perspective-acceleration uncertainty, which can matter for nearby,
  high-proper-motion stars over long baselines;
- it ignores parallax and parallax correlations on the angular path, even when
  a parallax measurement exists but the RV is missing.

The implementation is mathematically equivalent to an initially zero-radial-
motion rectilinear angular trajectory at an arbitrary distance; saying only
that no distance/RV is "imputed" must not be read as marginalization. Document
the result as an angular-model covariance, not total astrophysical uncertainty.
Do not invent an RV variance or isotropic perspective-error floor. A later,
explicit RV-prior marginalization can address that use case.

At an exact celestial pole the east/north orientation depends on the supplied
longitude gauge. The vector/tangent implementation remains finite, but the API
should promise wrap-safe and near-pole behavior, not a coordinate-invariant
ellipse at the exact pole.

## Xmatch routing precedence

Use disjoint row masks and never propagate a row twice:

| Mean-trajectory row | Covariance row | Action |
| --- | --- | --- |
| complete, finite, physical 6D | valid Gaia 5D | 6D mean and `J6[:2, :5] C5 J6.T`; RV is conditioned on its measured value |
| complete 6D | invalid 5D | 6D mean; retain reference-epoch/floor covariance |
| angular path | finite RA/Dec/PM/epoch and valid Gaia 4D | angular mean and `J4 C4 J4.T` |
| angular path | invalid PM/epoch or invalid 4D | existing angular-mean semantics; retain reference-epoch/floor covariance |
| required Torchsky Jacobian unavailable | otherwise eligible | existing `warn` fallback or `error`, independently for 6D and angular rows |

The 6D route has precedence. Do not use the angular Jacobian for a complete 6D
mean merely because its parallax covariance is bad: that would transport
uncertainty under a different model, or silently condition on exact parallax.

### Required correction to the current live Xmatch diff

`_astrometric_covariance_mas` currently produces one validity mask for the full
five-parameter covariance, and the angular route reuses that mask before
slicing `covariance[:, :4, :4]`. That is not a true submatrix fallback: a NaN
`parallax_error` or any invalid parallax correlation wrongly disables an
otherwise valid `(alpha*, delta, pmra, pmdec)` covariance.

Build/validate the 4D principal submatrix independently from exactly these
fields:

```text
ra_error, dec_error, pmra_error, pmdec_error,
ra_dec_corr, ra_pmra_corr, ra_pmdec_corr,
dec_pmra_corr, dec_pmdec_corr, pmra_pmdec_corr
```

Its order is `(alpha*, delta, pmra, pmdec)`. The configured Gaia mapping may
remain the existing complete fifteen-field mapping; only row-level 4D validity
must be independent of parallax fields. A private `include_parallax=False`
mode on the existing builder is enough.

## Minimum falsifying tests

Torchsky, in `tests/test_wcs_frames.py`:

1. Same epoch: position is unchanged and `J == [I2 | 0]` for ordinary, wrap,
   and near-pole rows.
2. Zero proper motion over `dt`: at an ordinary coordinate,
   `J == [I2 | dt*I2]` within float64 roundoff. This catches ordering and unit
   errors more directly than a generic finite test.
3. Central differences: perturb source position through the same normalized
   vector retraction (not `delta_RA / cos(dec)` near a pole), and measure output
   residuals in the nominal destination tangent basis. Cover a wrap crossing,
   both high-declination signs, and negative `dt`.
4. Batched results equal stacked scalar calls and remain finite. Keep the
   Jacobian shape assertion `(batch..., 2, 4)`.

Xmatch:

1. A mixed batch proves the routing table: valid 6D, missing RV with valid 4D,
   invalid 4D, and missing epoch; assert call masks and row order.
2. A missing-RV row with NaN parallax error/correlations but a valid 4D block
   still transports its angular covariance. This is the regression for the
   current validity-mask bug.
3. An invalid 4D correlation or non-PSD 4D block is not zero-imputed and retains
   the configured reference ellipse/floor.
4. Missing angular-Jacobian support obeys both `fallback_policy="warn"` and
   `"error"`; complete 6D rows must not fall through to the angular Jacobian.

The existing macOS direct-Python smoke could not run because duplicate
`libomp.dylib` initialization aborted before Torchsky execution; this is already
recorded in `.cursor/harness/failures/2026-07-22-phase-space-vmap-openmp.md`.
Use the repository pytest task for verification and do not enable
`KMP_DUPLICATE_LIB_OK`.
