# Torchsky map parity audit: SWarp, Drizzle, and Montage

Date: 2026-07-22
Torchsky snapshot: `c874fa8` (`main`)
Scope: image-plane projection, coaddition, background matching, uncertainty,
provenance, and execution. FITS parsing/writing remains owned by `torchfits`.

## Executive result

The statement that Torchsky already has “about 80%” is defensible only for the
**low-level reprojection/deposition substrate**. Torchsky already has native WCS
transforms, pull interpolation, exact point/square drizzle, polygon-overlap
remapping, masks/weights, several stack reducers, union-grid planning, and a
useful external-comparator suite.

It is not yet 80% of the **science-ready workflows** represented by the three
reference tools. The important missing work is concentrated rather than broad:

1. exact output uncertainty and data-quality provenance;
2. truly bounded, exposure-major streaming output;
3. Montage-style overlap-graph background matching;
4. blot/model-based outlier and cosmic-ray rejection;
5. curved-boundary target planning and a genuinely spherical conservative
   remapper for wide/high-latitude pixels.

These gaps matter directly to the INR/Foam roadmap. A learned multi-wavelength
model cannot make a credible “deeper than stacking” claim if the classical
baseline has approximate weights, background seams, or hidden unbounded-memory
behavior.

The implementation goal should be semantic parity at the operator and product
levels, not a clone of three legacy CLIs. The existing
`ObservationOperatorBundle`, `MapProductSpec`, and `torchsky map` surfaces are
the right architecture.

## Audit basis: primary sources only

- SWarp official project page and manual:
  [Astromatic SWarp](https://www.astromatic.net/software/swarp/) and the
  [official manual](https://raw.githubusercontent.com/astromatic/swarp/legacy_doc/prevdoc/swarp.pdf).
  The audited configuration vocabulary is from official source commit
  [`bf4f496`](https://github.com/astromatic/swarp/blob/bf4f496f18c04a8d32022b45449ef8675ab9b3da/src/preflist.h),
  including `RESAMPLING_TYPE`, `COMBINE_TYPE`, `WEIGHT_TYPE`, background,
  astrometric flux scaling, memory, and provenance settings.
- Drizzle algorithm and current implementation:
  [Fruchter & Hook](https://arxiv.org/abs/astro-ph/9808087), the
  [2025 DrizzlePac handbook](https://www.stsci.edu/files/live/sites/www/files/home/scientific-community/software/drizzlepac/_documents/drizzlepac-handbook.pdf),
  the [official STScI drizzle repository](https://github.com/spacetelescope/drizzle),
  and the released
  [`drizzle.resample.Drizzle` API](https://spacetelescope-drizzle.readthedocs.io/en/latest/api/drizzle.resample.Drizzle.html).
  The API basis is release `2.2.0` (2026-01-06), which includes squared-weight
  data propagation and DQ output in addition to science, weight, and context
  products.
- Montage official project, source, and algorithm documentation:
  [Caltech-IPAC/Montage](https://github.com/Caltech-IPAC/Montage),
  [reprojection/background/coadd algorithms](https://irsa.ipac.caltech.edu/Montage/docs/algorithms.html),
  [`mProject`](https://irsa.ipac.caltech.edu/Montage/docs/mProject.html), and
  [toolkit components](https://irsa.ipac.caltech.edu/Montage/docs/components.html).
  The source snapshot checked was official `main` commit `ac0a153` from
  2026-04-06.

## Status legend

- **Present**: native production path exists and has focused tests.
- **Partial**: a useful implementation exists, but an important semantic,
  correctness, scale, or user-product part is missing.
- **Absent**: no production equivalent was found.
- **Non-goal**: functionality should remain elsewhere or is better expressed as
  ordinary tensor composition; it should not widen `torchsky map`.

## Capability matrix

### Grid definition, WCS, and reprojection

| Capability | Reference behavior | Torchsky status | Evidence and exact gap | Closure |
| --- | --- | --- | --- | --- |
| User-specified target WCS and shape | All three tools can map to a declared output grid/header. | **Present** | `coadd_image_plane(target_header=...)`, `MapProductSpec`, and `torchsky map --reference`; tensor-native WCS remains in Torchsky. | Retain the single target-product contract. |
| Automatic union target | SWarp derives center, scale, projection, and image size; Montage has `mMakeHdr`/expanded target workflows. | **Partial** | `_plan_union_footprint_target` now uses a seam-safe spherical centre, exact TAN-plane bounds, and determinant scale, but samples only four input pixel-edge corners and always creates north-up TAN. Curved SIP/TPV boundaries can escape. | Adaptively subdivide all four pixel edges until output-plane midpoint error is bounded; fail closed at the TAN horizon. Keep reference headers for non-TAN products. |
| WCS projection breadth | SWarp accepts WCS projection codes; Montage explicitly targets broad projection/coordinate-system coverage. | **Present/partial** | Torchsky has broad native WCS projection support and can consume a reference header. Automatic planning is TAN-only and comparator coverage is concentrated on TAN/SIP/TPV. | Add wide/high-latitude SIN, AIT, CAR and seam fixtures using explicit target headers; do not add an Astropy runtime path. |
| Nearest, bilinear, Lanczos interpolation | SWarp: nearest, bilinear, Lanczos2/3/4; DrizzlePac also offers interpolation/deposition kernels. | **Present/partial** | Torchsky has nearest, bilinear, Lanczos3/5 plus cubic families. External SWarp gates cover bilinear/Lanczos3 in benchmarks, but only one tiny optional bilinear pytest is a direct SWarp gate. | Promote the existing comparator cases into versioned parity fixtures; Lanczos2/4 are low priority unless real workflows demand them. |
| Variable-pixel point/square drizzle | Drizzle maps shrunken input pixels using `pixfrac`, pixel mapping, weights, and geometric distortion. Montage `mProject -z` also shrinks input footprints. | **Present/partial** | Native `drizzle_point` and `drizzle_square` exist with chunked projection, weights, masks, flux/surface-brightness modes, gradients, and optional official-`drizzle` parity. The validator accepts any positive `pixfrac`, while the reference algorithm defines the normal physical range as 0–1; Torchsky clamps square half-size to `1e-6`, so exact `pixfrac=0` interlacing is unavailable through `drizzle_square`. | Define `0 <= pixfrac <= 1`; route exact zero to point deposition. Tighten official-reference parity on translations, rotations, scales, SIP, and edge support. |
| Other drizzle kernels | Current STScI drizzle supports point, square, turbo, Gaussian, and Lanczos-family deposition behavior. | **Partial** | Torchsky has Gaussian/Lanczos sampling elsewhere, not those forward-deposition kernels. | Add only after point/square uncertainty and DQ semantics are complete. Gaussian is useful; turbo/Lanczos deposition is P2. |
| Conservative pixel overlap | Montage projects input/output pixels on the sphere and computes spherical polygon overlap; its area image is the natural coadd weight. | **Partial** | `conservative_polygon` transforms four source-pixel corners into the target Cartesian plane and clips quadrilaterals there. It is exact for that planar four-corner model, not Montage’s spherical polygon intersection; curved pixel edges are not subdivided. | Add a spherical convex-polygon overlap primitive, with adaptive curved-edge subdivision where WCS distortion requires it. Keep the current planar kernel as a documented fast path. |
| Surface-brightness vs integrated-flux semantics | Drizzle applies scale/Jacobian rules according to units; Montage conserves input energy using sky overlap area. | **Present/partial** | Torchsky explicitly distinguishes `surface_brightness` and `flux`, computes pixel-area maps, and tests flux preservation. The path hard-casts image-plane products to float32 and has no cross-tool wide-field flux gate. | Gate constant surface brightness, aperture flux, and all-pixel integrated flux in float32 and float64 reference calculations. Preserve accumulator dtype as a later measured decision. |
| Inverse mapping / blot | Drizzle `blot` maps a combined truth/median image back to each detector for rejection and diagnostics. | **Absent** | Torchsky has formal adjoints and bound observation operators, but no product-level blot API reproducing the forward model on each source grid for a coadd workflow. An adjoint is not a blot. | Expose one source-grid prediction operation from the bound operator; use it for residuals and rejection rather than adding a separate legacy-shaped implementation. |

### Weights, combination, calibration, and uncertainty

| Capability | Reference behavior | Torchsky status | Evidence and exact gap | Closure |
| --- | --- | --- | --- | --- |
| Scalar and per-pixel weights/masks | SWarp accepts background/RMS/variance/weight maps and thresholds; Drizzle accepts weight maps; Montage accepts weight files and thresholds. | **Present/partial** | `ImageObservation` has scalar weight, weight map, mask map, and primary-beam map. Non-finite/negative inverse variance is sanitized. The CLI does not identify RMS versus variance versus inverse variance or expose a threshold contract. | Add an input weight-semantics enum at the adapter/CLI boundary and convert once to inverse variance; add an explicit rejection threshold. Do not overload a bare tensor deep in the operator. |
| Photometric scaling and additive offsets | SWarp has `FSCALE_*` and background controls; Drizzle has `iscale`; Montage `mProject -x` scales flux. | **Present** | `photometric_scale * (image - background_offset)` is implemented, and inverse variance is rescaled consistently. CLI accepts per-input values and records them in the plan JSON. | Add a real multi-exposure aperture-photometry gate; keyword discovery belongs in a thin torchfits/header adapter. |
| Weighted mean, equal mean, median | All references provide their relevant weighted/mean combination; SWarp and Montage support median. | **Present/partial** | All three methods exist and deterministic repeat tests cover them. Median materializes all projected images; equal mean ignores the user’s per-pixel weights except as support by design. | Document mean semantics explicitly. Replace dense median with tile-local/order-statistic execution only when requested. |
| Robust clipped coadd | SWarp has `CLIPPED` with amplitude and sigma controls; AstroDrizzle rejects cosmic rays through drizzle/median/blot/residual masks. | **Absent** | No production sigma-clipped/robust iterative stack. INR losses do not replace a classical robust baseline. | Implement model/blot residual rejection first, then a bounded tile-local clipped reducer. Report rejected samples and never silently mutate input masks. |
| Min/max/sum and logical/chi reducers | SWarp exposes MIN, MAX, SUM, chi variants, and logical reducers. | **Partial/non-goal** | Sum is an internal numerator operation but not a public stack mode; min/max/chi/logical modes are absent. | Add `sum` only if a science product requires it. Min/max and logical reducers are ordinary tensor reductions and should not widen the map CLI. Chi modes need a concrete use case and statistical contract. |
| Output weight semantics | Drizzle’s `out_wht` is the algorithmic accumulated weight; Montage emits area coverage; SWarp emits its configured weight product. | **Partial** | `MapResult.weight` is the accumulated denominator/support according to kernel and value semantics. It is useful, but `estimate_map_inverse_variance(mode="native_inverse_variance")` currently treats it as inverse variance without encoding covariance or the squared interpolation coefficients. | Split `accumulated_weight`, `coverage_area`, and `inverse_variance` concepts. Propagate independent input variance with squared operator coefficients. Refuse to label a support proxy as native inverse variance. |
| Variance/error propagation | Drizzle 2.2.0 accepts `data2` and produces squared-weight outputs for standard variance propagation. | **Absent in path coadd** | Matrix-free solvers can estimate posterior variance, and fields can predict variance, but the classical FITS coadd path does not propagate input variance through the linear resampling coefficients. | Make the geometry operator optionally accumulate `sum(a_i^2 var_i)` alongside `sum(a_i w_i)`. Publish both per-pixel variance and the documented correlated-noise limitation. |
| Correlated-noise accounting | Drizzle documentation explicitly warns that nonzero drops correlate output pixels and gives correction-ratio analysis. | **Absent** | No covariance/correlation kernel or aperture-noise correction accompanies classical stacks. | Emit a local noise-correlation kernel or effective correction factor for stationary affine cases; for general WCS provide a reproducible Monte Carlo estimator. Never claim independent-pixel depth from the diagonal alone. |

### Backgrounds, rejection, data quality, and provenance

| Capability | Reference behavior | Torchsky status | Evidence and exact gap | Closure |
| --- | --- | --- | --- | --- |
| Per-image mesh sky | SWarp estimates a mesh background, filters it, interpolates it, and supports manual background values. | **Present/partial** | `MeshSkyEstimator` has robust cell statistics, filtering, interpolation, self-masking, and polynomial replacement. The CLI still advertises `--sky`, but `_run_coadd` prints that it is no longer built in and asks for precomputed offsets. | Either wire the estimator before reprojection with masks or remove the misleading CLI option. First add source-preservation and crowded-field bias fixtures. |
| Cross-image background matching | Montage fits plane differences in pairwise overlaps, then solves/iterates a global correction model. | **Absent** | `CrossExposureSkyCorrector` compares full reprojected-image means and subtracts per-image sky models. It does not form an overlap graph, fit robust pairwise planes, weight equations by overlap/noise, handle disconnected components, or preserve a declared gauge. It also forces storage of every projection. | Implement a sparse overlap graph: robust weighted plane fit per edge, connected-component gauge constraints, sparse least-squares global solve, then apply corrections exposure-major. This is the highest-value missing Montage algorithm. |
| Cosmic-ray/outlier workflow | AstroDrizzle makes separate drizzles, combines a median truth, blots it back, compares residuals with derivative/noise terms, and updates masks before final drizzle. | **Absent** | Masks are accepted, but no model-predict/reject/refit loop exists. | Build on the shared source-grid prediction operator. First support supplied noise + conservative two-threshold residual masks; add derivative dilation only with an upstream fixture. |
| Context/provenance image | Drizzle context bitplanes identify every contributing input at every output pixel, including more than 32 inputs. | **Partial** | `overlap_count` records only how many inputs contributed; `observation_mocs` record coarse footprints. Neither answers which observations formed a pixel. | Add optional packed 32-bit context planes. Keep disabled by default because memory is `ceil(N/32) * H * W * 4` bytes; support tile-streamed writes. |
| DQ propagation | Current drizzle OR-combines input uint32 DQ bits into `out_dq`. SWarp has flag interpolation modes. | **Absent** | Torchsky mask maps collapse quality to contribute/do-not-contribute. Named quality sanitization exists elsewhere, but image-plane coadd emits no DQ plane. | Accept optional uint32 DQ plus a reject-bit mask; OR only contributing non-rejected bits. Preserve bit semantics in the product metadata. |
| Header/provenance merge | SWarp can copy keywords and write per-file/XML metadata; AstroDrizzle blends headers and records inputs. | **Partial** | Atomic `--plan-json` records resolved inputs, HDUs, calibrations, target, execution, and device. Output FITS currently writes only the data plane with the target header. | Treat plan JSON as canonical provenance; have torchfits write data/weight/variance/overlap/context/DQ HDUs plus a compact plan hash/input table. Do not duplicate FITS machinery in Torchsky. |

### Execution, memory, and user workflow

| Capability | Reference behavior | Torchsky status | Evidence and exact gap | Closure |
| --- | --- | --- | --- | --- |
| Bounded reprojection work | SWarp exposes RAM/VMEM/coadd buffers; Montage reads only input rows contributing to the current output row; Drizzle accepts preallocated output arrays and input bounds. | **Partial** | Individual drizzle/conservative work is chunked and projected observations use target windows. Path coadds keep a module-global `_obs_image_cache`; spatial tiling allocates the full output and recursively re-runs all observations per tile. Median and cross-exposure correction store all projections. | Replace the global cache with call-scoped loading and an exposure-major accumulator. Stream target tiles or rows to a torchfits-backed output sink. One decoded exposure and a bounded tile set should be resident. |
| Cache correctness | Reference tools reread named inputs for a run. | **Absent** | `_obs_image_cache[(path,hdu)]` has no file identity/version and can return stale data after overwrite; it also retains device tensors indefinitely. | Delete it. If a cache is later proven necessary, put it in torchfits with file identity, byte budget, explicit lifetime, and invalidation. |
| Automatic footprint pruning | Montage has metadata tables/overlap analysis; SWarp opens relevant images/rows. | **Present/partial** | Torchsky projects into target windows and can produce MOCs, but the path coadd does not construct a reusable image-to-tile schedule before execution. | During plan resolution, build tile→observation indices from exact/adaptive footprints or MOCs; record counts in plan JSON. |
| Parallel/device execution | SWarp has threads/nodes; Montage decomposes per-image stages; Torchsky targets CPU/CUDA/MPS. | **Present/partial** | Tensor kernels are cross-device and several paths are differentiable. Python loops and file orchestration remain serial; MPS float64 WCS constraints are known. | Fix algorithmic streaming first. Then overlap torchfits decode/host transfer with device compute using a bounded producer-consumer queue. Benchmark before adding native code. |
| Full output product | Mature tools emit science plus weight/area and often context/DQ/intermediates. | **Partial** | `MapResult` holds data, weight, overlap count, and optional MOCs, but `torchsky map` writes only `result.data`. | Make CLI output a deliberate MEF product through torchfits: SCI, WHT, VAR (when valid), NEXP, optional CTX/DQ, plus provenance. |
| Dry-run/reproducible execution | Legacy tools expose resolved config; SWarp can emit metadata. | **Present** | `--dry-run --plan-json` atomically records expanded inputs and execution settings. | Add planned peak-memory estimate, ancillary products, comparator/tool versions for benchmark plans, and tile schedule summary. |
| Multiband/cube products | Montage can mosaic multiple wavelengths but does not jointly infer a latent spectral field; legacy stackers are normally per band. | **Beyond parity, present/experimental** | Torchsky already has multiband/cube coadds and joint continuous fields. | Use the corrected weights/background/provenance above as the classical control for INR/Foam depth experiments. This is a Torchsky advantage, not a legacy-compat task. |
| Archive discovery/download and distributed workflow engines | Montage includes archive and grid/Pegasus-style utilities. | **Non-goal** | Data discovery belongs in Zensus and orchestration in CANFAR/lab tooling. | Keep `torchsky map` focused on resolved local/object-store inputs and reproducible plans. |

## Prioritized closure plan

### P0: correctness and scale foundations

1. **Exposure-major streaming coadd; delete `_obs_image_cache`.**
   This is prerequisite to honest “survey-scale” claims. Weighted mean and mean
   should read each HDU once per call, project only its scheduled target tiles,
   and release it. The output sink must not require full data/weight arrays in
   memory. Median remains explicitly memory-heavier until it gets a tile-local
   implementation.
2. **Separate accumulated weight, coverage, and inverse variance.**
   Add squared-coefficient variance propagation and stop treating an arbitrary
   interpolation denominator as native inverse variance. This is prerequisite
   to any depth comparison with INR/Foam.
3. **Adaptive footprint boundaries.**
   Subdivide SIP/TPV pixel edges until a target-plane error bound is met, then
   reuse the boundary for union containment and tile scheduling.
4. **Overlap-graph background solve.**
   Replace whole-image mean matching with robust pairwise overlap planes and a
   sparse globally constrained solve. Apply the solved planes during the
   exposure-major pass.

### P1: complete the scientifically important workflows

5. **Blot/source-grid prediction and iterative residual rejection.**
   Reuse the shared forward observation operator; do not make an unrelated
   AstroDrizzle wrapper.
6. **Ancillary products:** variance, NEXP, optional context bitplanes, and DQ OR
   output via torchfits MEFs.
7. **Spherical conservative overlap** for high-latitude/wide-pixel products,
   retaining the planar kernel as an explicit fast path.
8. **Tile-local robust clipped coadd** with explicit rejection logs/masks.

### P2: breadth after the foundations are gated

9. Gaussian drizzle deposition and only the additional SWarp/Drizzle kernels
   justified by benchmarked science cases.
10. Optional `sum` reducer and richer header propagation. Do not copy SWarp’s
    logical/min/max/chi catalogue without users and statistical contracts.

## Executable parity fixtures

Extend `benchmarks/bench_map_projection_mosaic_suite.py` and
`benchmarks/bench_map_tan_sip_suite.py`; do not create another top-level suite.
Small deterministic pytest counterparts belong beside the existing optional
external parity tests.

### F1 — SWarp interpolation and stack grid

- Inputs: four 96×80 TAN images of an analytic constant + linear gradient + two
  Gaussian sources; subpixel translations `(0,0)`, `(0.37,-0.21)`, rotation
  `7 deg`, and pixel-scale ratio `1.13`; heterogeneous inverse-variance maps,
  one bad-pixel stripe, and known photometric scales.
- Target: one explicit 120×112 TAN header shared by both tools.
- Matrix: nearest, bilinear, Lanczos3 × weighted mean, equal mean, median.
- Comparator: official SWarp binary, with `FSCALASTRO_TYPE=NONE` and matching
  pre-calibration as the existing benchmark already does.
- Report separately: common-interior RMSE normalized by truth dynamic range,
  edge RMSE, support Jaccard, aperture-flux error, and weight difference.
- Gate: constant-field interior error `<= 8 * eps_float32`; bilinear
  common-interior normalized RMSE `<= 1e-4`; aperture flux differs by `<= 0.2%`.
  Freeze Lanczos/edge tolerances only after one pinned-version baseline run,
  because boundary conventions differ.

### F2 — Drizzle upstream geometry matrix

- Inputs: one impulse grid and one smooth flux field on 64×64 TAN and SIP
  images; translated, rotated, and 0.83/1.19 scale targets.
- Matrix: point/square × `pixfrac={0,0.3,0.7,1}` × surface brightness/flux ×
  uniform/heterogeneous weights.
- Comparator: official `drizzle==2.2.0`, direct `calc_pixmap` + `Drizzle`.
- Gate: bit-identical support for pure integer translations; otherwise support
  Jaccard `>= 0.999`; constant-field common-support error
  `<= 2e-5`; total flux error `<= 5e-5` relative; centroid difference
  `<= 1e-3` output pixel. Report PSF second moments and weight residuals.

### F3 — Drizzle variance, context, and DQ

- Inputs: 65 aligned 16×16 exposures so three 32-bit context planes are
  required; analytic per-input variances; unique DQ bits distributed over the
  first 32 bits; masks that remove selected contributors.
- Comparator: official `drizzle==2.2.0` `out_img2`, `out_ctx`, and `out_dq`.
- Gate: exact context bits, exact DQ OR, and variance equality at
  `rtol=2e-6, atol=2e-7` for point drizzle. Square-drizzle variance is gated
  against an independent dense coefficient matrix as well as the external
  output.

### F4 — Drizzle/blot rejection

- Inputs: eight Nyquist-challenged, subpixel-dithered point-source exposures
  with seeded Gaussian noise, cosmic rays, bad columns, and one saturated core.
- Workflow: initial drizzle → median/model → source-grid prediction → residual
  mask → final drizzle.
- Comparator: official AstroDrizzle workflow on the same WCS/masks where its
  instrument-independent steps are callable.
- Gate: injected cosmic-ray recall `>= 0.995`, clean-pixel false-positive rate
  `<= 1e-3`, no rejection of the saturated core beyond the declared DQ mask,
  aperture flux bias `<= 0.2%`, and centroid shift `<= 0.01` pixel.

### F5 — Montage spherical projection

- Inputs: coarse 0.25-degree pixels at Dec 80 deg crossing RA 0, plus one
  all-sky-tail AIT target; constant surface brightness, point sources, and an
  integrated-flux image.
- Comparator: official Montage `mProject`/`mProjectPP` where applicable, using
  its area outputs.
- Gate: every finite source footprint is contained; constant surface brightness
  error `<= 1e-6` in float64 reference math; integrated flux error `<= 1e-6`;
  centroid error `<= 1e-3` target pixel; support/area Jaccard `>= 0.9999`.
  The current planar `conservative_polygon` row is expected to fail at least one
  high-latitude tolerance until the spherical kernel lands.

### F6 — Montage overlap-graph background recovery

- Inputs: a 3×3 mosaic with 20% overlaps, known per-exposure planes, masked
  sources, heterogeneous noise, and one disconnected two-image component.
- Comparator: official `mOverlaps` → `mDiffFitExec` → `mBgModel` corrections.
- Gate: recover plane coefficients modulo one gauge per connected component;
  edge-weighted overlap residual RMS within 5% of Montage and within 10% of the
  injected noise floor; preserve injected diffuse sky modes above the declared
  background polynomial order to `<= 0.5%` in integrated flux.

### F7 — Curved-footprint planner containment

- Inputs: SIP and TPV images with deliberately curved boundaries, including an
  RA seam and a high-declination case.
- Reference: independently sample every pixel edge at 4097 points and transform
  with the Torchsky WCS; compare planned size/center with SWarp and Montage as
  descriptive rows.
- Gate: 100% of dense edge samples fall inside target pixel edges with maximum
  overshoot `<= 1e-6` pixel; target area is no more than 1.15× the minimum
  axis-aligned TAN rectangle after requested padding.

### F8 — Bounded execution and cache invalidation

- Inputs: quick profile 24×512²; full profile 128×4096²; repeated paths with
  multiple HDUs; 8192² target tiled at 1024².
- Instrumentation: torchfits read-call counter, decoded bytes in flight, process
  peak RSS, CUDA peak allocation when available, and output checksum.
- Gate: each image HDU is decoded once per coadd call; decoded image residency
  is at most one exposure plus a documented prefetch slot; accumulator residency
  is bounded by configured tiles; overwriting an input between calls changes the
  result; tile sizes 512/1024/2048 are bit-identical under deterministic mode.
  Peak memory must be fitted and reported versus exposure count and target area,
  not hidden behind one machine-specific byte threshold.

### F9 — Real-product acceptance

- Reuse the existing SMACS J0723/UPdec-Webb benchmark inputs and exact published
  crop/target definitions.
- Run Torchsky bilinear, Lanczos3, drizzle square, conservative spherical, and
  the official SWarp/Drizzle/Montage comparators.
- Report source aperture flux/centroid/shape, empty-aperture correlated noise,
  overlap-seam residuals, background gradients, coverage, peak memory, and wall
  time. Do not rank on PSNR alone.
- This is the classical control for the multi-wavelength INR/Foam experiment;
  depth claims use empty-aperture noise and injected-source completeness at
  fixed false-positive rate.

## Commands and gate placement

Focused development commands:

```text
pixi run -e test -- pytest tests/test_map_image_plane.py tests/test_map_drizzle_external_parity.py tests/test_map_conservative_external_parity.py tests/test_map_swarp_optional_parity.py -q
pixi run bench-map-projection-mosaic-suite
pixi run bench-map-compat-gate
```

The `map-compat` environment must pin and record comparator versions in every
JSON artifact. Missing external tools remain explicit `SKIPPED` rows in routine
CI; one scheduled/release job must require all three references and fail on a
skip. Repository pre-push verification remains:

```text
pixi run ci-local
```

## Definition of parity complete

Torchsky may claim SWarp/Drizzle/Montage-class mapmaking when all of the
following are true:

1. F1–F8 pass with pinned official comparator versions, and F9 has a published
   artifact on at least one representative real mosaic.
2. Science, accumulated weight/coverage, propagated variance, overlap count,
   optional context, and DQ have distinct documented semantics and are emitted
   through torchfits without full-output memory residency.
3. Background correction is solved on an overlap graph and reports its gauge,
   connected components, coefficients, and residuals.
4. Drizzle point/square, blot/source-grid prediction, and rejection compose the
   same observation operators used by INR/Foam fitting.
5. The CLI can dry-run the exact plan, estimate memory, execute it, and reproduce
   the result without process-global data state.

That closure is more valuable than copying every legacy reducer or command-line
flag. It makes the classical baseline scientifically hard to beat, which is
exactly what the joint multi-wavelength INR/Foam demonstration needs.
