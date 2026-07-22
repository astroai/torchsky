# Map/coadd next milestone: spherical, seam-safe union target planning

## Recommendation

Ship one bounded correctness milestone before adding another resampling kernel or
widening the CLI: replace the current RA/Dec bounding-box union planner with a
rotation-invariant spherical/TAN-plane planner.

Keep the public signatures unchanged:

- `plan_image_plane_target(..., union_footprint=True)`
- `plan_map_product(..., union_footprint=True)`
- existing `footprint_padding` and `max_footprint_pixels` behavior

This is the highest-impact safe change because every classical coadd kernel,
every product stack, the INR target renderer, and `torchsky map` rely on the
same planned target. A bad target cannot be repaired downstream by a better
resampler.

## Reproduced failures on current `main`

The implementation in `src/torchsky/map/image_plane.py` currently:

1. takes scalar minima/maxima of corner RAs and declinations;
2. estimates pixel scale from only diagonal `CD`/`PC`/`CDELT` entries; and
3. converts angular RA span directly to target pixels.

This causes three concrete failures:

- **RA seam:** two 16x16, 0.001 deg/pixel TAN images centred at RA 359.99 and
  0.01 deg planned a `(15, 359995)` target centred at RA 180 deg. The expected
  union is only tens of pixels wide.
- **High declination:** two equivalent images at Dec 80 deg separated by 0.1
  deg in RA planned `(15, 187)`. The tangent-plane separation is reduced by the
  spherical geometry; direct RA-degree division grossly overallocates width.
- **Rotated CD:** a valid 90-degree rotated CD matrix with scale 0.001 deg/pixel
  failed with `RuntimeError: Could not determine pixel scale from observation
  headers` because both CD diagonal entries are zero.

The current tests only assert that a simple low-declination union is wider than
one reference image, so none of these scientific edge cases is gated.

A separate reproduced issue is the module-global `_obs_image_cache`: overwriting
a FITS path and coadding it again returned the old pixels (`16.0` before and
after replacing a 4x4 ones image with zeros). It also retains decoded CPU/GPU
tensors across calls and defeats a strict bounded-memory claim. That should be
the next execution milestone, but fixing it correctly requires an
exposure-major tiled loader/accumulator refactor. The union planner is smaller,
publicly visible, and independently shippable now.

## Exact implementation contract

### Files and private helpers

Primary implementation:

- `src/torchsky/map/image_plane.py`
  - replace `_compute_union_footprint` and `_target_header_from_footprint` with
    one private spherical union-planning path;
  - retain the existing public `plan_image_plane_target` API;
  - do not add a runtime dependency or use Astropy in production.

Suggested private decomposition, only if it keeps the change readable:

- `_spatial_pixel_scale_deg(wcs)`: compute a rotation-invariant per-observation
  scale from `sqrt(abs(det(wcs.cd)))`, reject singular/non-finite matrices, and
  take the median across observations to preserve the current median-scale
  policy.
- `_image_boundary_world(wcs, shape, ...)`: sample pixel **edges** (coordinates
  `-0.5` through `N-0.5`), not only corner pixel centres. Subdivide edges until
  tangent-plane midpoint error is below a documented fraction of one output
  pixel; fail closed if the error bound is not reached at the maximum depth.
- `_spherical_union_center(points)`: normalize the mean Cartesian unit vector,
  yielding a seam-safe centre; reject degenerate/antipodal sets.
- `_world_to_tan_plane(points, centre)`: exact gnomonic coordinates. Reject any
  boundary point on or behind the tangent-plane horizon instead of producing a
  silently invalid single-TAN product.

The final header should be derived from min/max tangent-plane `xi, eta`, with
padding applied in that plane. Compute `NAXIS1/2` and `CRPIX1/2` so every sampled
pixel edge lies inside the output pixel-edge rectangle. Preserve the conventional
east-left orientation (`CD1_1 < 0`, `CD2_2 > 0`). Apply
`max_footprint_pixels` to the resulting shape before any target tensor is
allocated.

Do not silently fall back to the current RA/Dec box. An unrepresentable
single-TAN union should tell the user to provide `--reference` or partition the
mosaic.

### Public behavior

- Normal small TAN unions remain compatible within a one-pixel containment
  difference.
- RA seam, polar/high-declination, rotated/skewed CD/PC, SIP, and TPV footprints
  plan finite targets that contain all adaptively sampled source boundaries.
- Planning is deterministic, device independent, and uses float64 geometry.
- No public class, dependency, or command-line option is added in this
  milestone.

## Test plan

Add focused cases to `tests/test_map_image_plane.py` and one product-level
delegation case to `tests/test_map_product.py`:

1. RA 359.99/0.01 seam: target centre is near 0 deg (modulo 360), width is
   bounded, and dense source-edge samples map inside output pixel edges.
2. Dec 80 deg pair: output width follows tangent-plane rather than raw RA span;
   dense boundaries remain contained.
3. 90-degree rotated CD and a skewed CD matrix: planning succeeds and chosen
   scale matches `sqrt(abs(det(CD)))`.
4. SIP/TPV curved edges: adaptive boundary planning contains an independently
   dense edge replay, not only four corners.
5. A footprint crossing the TAN horizon fails with an actionable message.
6. `footprint_padding < 1`, non-finite/singular CD, and a non-positive
   `max_footprint_pixels` fail at the trust boundary.
7. `max_footprint_pixels` rejects the final plane-derived shape before coadd.
8. `plan_map_product(..., union_footprint=True)` exposes exactly the same target
   as `plan_image_plane_target`.

Focused verifier during development:

```text
pixi run -e test -- pytest tests/test_map_image_plane.py tests/test_map_product.py -q
```

Repository pre-push verifier remains:

```text
pixi run ci-local
```

## Benchmark and parity plan

Extend `benchmarks/bench_map_projection_mosaic_suite.py` rather than creating a
new benchmark surface. Add small/medium seam, high-declination, rotated-CD, and
distorted-edge planning cases and report:

- planning wall time;
- planned output pixel count;
- fraction of independently dense boundary samples contained;
- maximum boundary overshoot in output pixels;
- end-to-end coadd coverage/RMSE on the planned grid.

In the optional `map-compat` environment, compare target containment and pixel
count against `reproject.mosaicking.find_optimal_celestial_wcs` (parity
reference only, never a runtime path). Run the existing projection/mosaic suite
before and after to ensure no regression on ordinary TAN/SIP cases.

## Audit summary and why this wins

The map stack already has substantial strengths:

- native interpolation, drizzle, conservative and splat geometry families;
- exact adjoint/gradcheck coverage for promoted geometry operators;
- chunked drizzle/conservative pixel work;
- target-window pruning for large mosaics;
- matrix-free, exposure-chunked normal actions;
- deterministic repeat-run and external/ground-truth parity suites.

The important remaining weaknesses are:

- union planning is not spherical and can catastrophically mis-size a product;
- path coadds retain an unbounded, stale process-global observation cache;
- spatial tiling still allocates the full output and recursively reprocesses
  observations;
- median stacking materializes every projected observation;
- `MapResult.weight` is later treated as native inverse variance even though
  exact variance propagation through interpolation kernels is not represented;
- the path-oriented coadd hard-casts to float32, while differentiability is
  strongest in the lower-level bound operators rather than the promoted FITS
  workflow.

Seam-safe target planning beats the alternatives for this shipment because it:

- fixes exact scientific geometry rather than optimizing a wrong grid;
- affects coadd, INR, products, and CLI at once;
- prevents catastrophic output sizes and accidental auto-tiling/I/O;
- requires no public API growth and no dependency;
- has crisp containment and comparator gates;
- is much smaller and safer than the necessary streaming/cache refactor;
- should precede inverse-variance API changes, which need a deliberate semantic
  contract across all kernels.

After this lands, the next map execution milestone should remove
`_obs_image_cache` and make weighted/mean tiled coaddition exposure-major with
one decoded exposure resident at a time. Only after both planning and execution
are trustworthy should more INR/FOAM capacity or additional convenience CLI
surface outrank operator/noise correctness.
