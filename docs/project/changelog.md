# Changelog

All notable changes to torchsky are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0rc1] - Unreleased

### Added

- `associate(blends="pairs", blend_radius_arcsec=...)`: blend-aware
  association for low-resolution right catalogues (WISE against Gaia). A
  right row may be the blend of two left rows closer than the blend radius,
  at their flux-weighted centroid: `blend_flux` fixes each pair's flux
  ratio, otherwise it is marginalised under a uniform prior (closed form).
  Single left rows and left pairs are independent sources conditioned on
  each left row contributing at most once; the blend rate is fitted by
  maximum marginal likelihood with completeness, calibration and
  background density (the prior normaliser is a monomer-dimer partition
  function, from the same belief propagation). Posteriors are Bethe
  marginals, exact on cycle-free graphs; without pair sources the result
  equals the exclusive model. `Association.blend`
  (`torchsky.catalogs.BlendAssociation`) holds the pair hypotheses,
  their posteriors and flux fractions, per-row background probabilities
  and each right row's most probable member list. On injected blends the
  fitted error scale is no longer inflated by blended positions, and more
  true members are matched confidently. Off by default; not combined with
  photometric `terms` or per-row right epochs.

- `associate_many(terms=...)`: photometric likelihood-ratio terms in N-way
  association, one tuple of `HistogramRatio` per secondary catalogue with
  the semantics of `associate` (right magnitudes, or pair colours relative
  to the primary). Each present member's factor is multiplied by its ratio.
  The ratios are learnt by the pairwise fits, then by EM from the hypothesis
  posteriors, with `exclusive=True` and `False` and with per-row-epoch
  catalogues. The best-first bound includes each candidate's ratio, so the
  residual mass stays certified. With two catalogues the result equals
  `associate` with the same terms. `max_iter` and `tol` bound the EM.

- `associate(exclusive=True)` and `associate_many(exclusive=True)` (the new
  default): one-to-one association. A right (secondary) row is the
  counterpart of at most one primary. Posteriors are the Bethe marginals of
  that model, from belief propagation on the candidate graph with
  Anderson-accelerated message updates and implicit-function gradients. They
  are exact on cycle-free candidate graphs and reduce to the independent
  posteriors where no right row is shared. In crowded fields the independent
  model (`exclusive=False`, nway's) counts a shared right row once per left row
  that sees it, which biased completeness and error calibration. The exclusive
  model also fits the right background density (rows that are not
  counterparts, with a Gamma(3/2) prior per cell), evaluated at the right
  row. `Association.density_mas2` reports it. On CPU the message sweeps use
  a torch-free native kernel (`xmatch_segment_lse_excl`, segmented
  log-sum-exp excluding self) with the torch path as reference for other
  devices and gradients.

- `torchsky.catalogs.PartitionPlan`, `healpix29` and `margin_arcsec`:
  partition primitives for survey-scale matching without I/O. Mixed-order
  nested cells (order-29 ranges, HATS `_healpix_29`) come from explicit pixels,
  a MOC or adaptive counts. Row ownership, plan-to-plan neighbour cells and
  margin row selection are sound by construction (cap bounds with the analytic
  maximum pixel radius, through the exact zone engine). Partitioned `all`,
  `best` and `match_nearest` equal the single pass across seams and poles.
- `torchsky.catalogs.associate_many` / `NWayAssociation`: N-way association
  (nway-style). For each primary, the hypotheses are every combination of a gated
  counterpart or none per catalogue, weighted by the exact Gaussian marginal
  likelihood in the primary's tangent plane (Budavari-Szalay eq. 18 with full
  covariances), completeness and density. Primaries over `budget` hypotheses are
  enumerated best-first with a certified residual mass instead of a silent cut.
  With two catalogues it reproduces `associate` exactly.
  Catalogues with per-row epochs (2MASS dates) are gated at pair epochs and
  placed relative to the primary propagated to each candidate's epoch.
- `associate` evaluates pairs at the right row's own epoch when the right
  catalogue has per-row epochs (2MASS observation dates). The left row is
  propagated with its covariance to each candidate's epoch, and the search
  radius is widened by the left row's motion over the epoch span
  (covariance bounded at the span's end points). A single median epoch lost
  Barnard's star (10 arcsec/yr) on real Gaia x 2MASS data.
- `associate(calibrate="floor" | "scale_floor")` fits a 2x2 systematic floor
  and a scale on the right catalogue's errors by posterior-weighted maximum
  likelihood, alternated with the association. Real Gaia x CatWISE matches
  showed CatWISE errors underestimated (mean d^2 2.6-5.0 instead of 2).
  `Association.right_scale` and `floor_mas2` report the fit.
  `calibrate_by=` / `calibrate_edges=` fit the terms per bin of any per-row
  variable (magnitude, or a blending count from `count_within`), because
  blended sources in crowded fields have larger errors than isolated ones.
- `HistogramRatio(..., left_values=...)`: pair-colour likelihood ratios
  (e.g. W1 - G). Both the counterpart and background colour distributions are learnt by
  EM from the candidate pairs; this separates a blend's true counterpart from
  its neighbours. Photometric terms are now fitted by EM also with a fixed
  completeness. Calibration fits stay within physical bounds and fall back to the
  previous estimate on divergence (an overflow crashed one real field).
- `torchsky.catalogs.associate`, `HistogramRatio` and `assign_one_to_one`:
  probabilistic counterparts. The pipeline is common-epoch propagation with
  covariances, then a Mahalanobis chi^2 gate with provably sound per-row search
  radii (right rows bucketed by radius class), then Sutherland-Saunders /
  Budavari-Szalay likelihood ratios against the right catalogue's density over
  its MOC footprint, then posteriors with EM completeness, and finally exact
  one-to-one assignment (vectorised star components plus the Hungarian
  algorithm). Pair offsets and covariances are parallel-transported to the pair
  midpoint, so the metric is exact at the poles. Validated by brute-force gate
  recall (field, pole, RA wrap), chi^2 calibration of true pairs, posterior
  calibration (ECE < 0.03), completeness recovery and scipy
  `linear_sum_assignment` parity.
- `torchsky.map.exposure_fit` (experimental; moved from a research prototype
  with unchanged numerics, checked bit for bit against it): `load_exposures`
  reads JWST NIRCam and HST WFC3/IR and ACS/WFC exposure files through
  `torchfits` into `FieldObservation` objects; `fit_sky(observations,
  SkyFitConfig)` fits a sky of Gaussian items and an instrument response in
  each exposure's own pixels (no resampling); `SkyFit` (save and load) and
  `predict_exposure` (cross-fitted prediction of held-out exposures and of
  exposures that were not in the fit). `SkyFit.product(bands, ...,
  observations=...)` returns a `SkyProduct`: the sky image of one or several
  bands at a stated resolution with its standard deviation, from the
  Gauss-Newton normal operator with the response held fixed (one solve for
  all bands, a time budget, circulant or posterior-draw standard deviation);
  `half_difference` compares two products from disjoint halves of the
  exposures; `SkyFit.response.kernel`, `.throughput` and `.noise_terms` read
  out the fitted response. The standard deviation is checked on synthetic
  data only: on real JWST data the Gauss-Newton solve does not converge yet.
  `ReferenceCatalogue` is a placeholder that raises `NotImplementedError`.
- `torchsky.instrument.InstrumentDescription` and `instrument_description`
  ("NIRCAM", "WFC3_IR", "ACS_WFC"): detector names and channels, pixel scales,
  aperture, data-quality policy from the bit tables, conversion of the science
  plane to MJy/sr from header cards, extension layout.
- `torchsky.catalogs.Astrometry`, `MotionPrior` and `propagate_astrometry`:
  per-row 6x6 astrometric covariances (position, proper motion, parallax,
  radial velocity; local east/north basis), built from Gaia archive columns,
  error ellipses or plain errors, validated fail-closed (unmeasured blocks are
  NaN, never zero). `propagate_astrometry` returns positions and the target-epoch
  position covariance `J C J^T` with the exact motion Jacobians, routed per
  row. `MotionPrior.from_reference` turns a reference catalogue's proper
  motions (for example Gaia) into a per-HEALPix-cell prior, with explicit
  parent-cell pooling below `min_count` stars, for rows without proper
  motion. `Catalog(astrometry=...)` carries the covariance through `to` and `take`.
- `CatalogState.propagate` / `Catalog.propagate` and `CatalogState.motion_level`:
  per-row propagation with the richest measured motion. Full space motion
  needs proper motion, parallax > 0 and radial velocity; otherwise linear
  proper motion (parallax ignored, never imputed); otherwise an explicit
  prior mean.
- `torchsky.catalogs.SkyIndex`, `match_within`, `match_nearest` and
  `count_within`: an exact, reusable sky index with the queries of astropy
  `search_around_sky` / `match_to_catalog_sky` and scipy `cKDTree`
  (`query_ball_point`, `query(k, distance_upper_bound)`,
  `return_length=True`). `match_nearest` returns the exact `k` nearest
  neighbours with or without a radius cap (doubling search radii, so crowded
  fields resolve early and sparse targets still find a distant catalogue).
  `match_within` takes per-row radii on both sides, combined by `"sum"`,
  `"quadrature"` or `"max"`. Separations are differentiable.

- `torchsky.scene.GaussianItemScene`: a differentiable sky of many anisotropic
  Gaussian items (position, two widths, position angle, K coefficients each),
  rendered in batch onto a tangent-plane grid through a WCS. Memory in the
  backward pass is bounded by `chunk_elements`, not by items x patch x K, and
  `mode="pyramid"` keeps the cost per item independent of its size. Item
  parameters use the `SourceCatalogScene` Gaussian conventions.
- `torchsky.scene.render_gaussian_items`: the functional form of the Gaussian-item
  rasteriser (pixel centres, covariances or `(major, minor, angle)`, and `(C, K)`
  fluxes in; `(K, H, W)` flux per pixel or surface brightness out; no parameters
  held, first-order differentiable in every input). `GaussianItemScene` now calls it.
  New `patch_cap` (default 128 nodes) bounds the patch of one item: items whose larger
  width is about 32 pixels or less are unchanged; wider round items go to the
  pyramid levels in `mode="direct"`, and long thin items are truncated at the cap
  (see the docstring for the size of the error).
- `torchsky.wcs.PolynomialDetectorDistortion`, `TelescopeAttitude` and
  `TelescopeFrameDetectorToSky`: detector-to-sky astrometry for exposures that
  carry no distortion in the header (JWST count-rate `_rate` files). A
  differentiable SIAF polynomial maps science pixels to the ideal frame, a
  rotation by `V3IdlYAngle` and parity maps that to the telescope frame, and an
  exact spherical attitude from `V2_REF`, `V3_REF`, `RA_REF`, `DEC_REF` and
  `ROLL_REF` maps it to the sky. The composed transform satisfies the
  `FieldObservation.detector_to_sky` protocol and also provides `world_to_pixel`
  (needed by shared-sky cropping). Coefficients are tensors; torchsky ships no
  calibration data.
- `field.field_observation_from_exposure_file` reads one calibrated exposure file
  (JWST `_cal` / `_rate`, HST `_flt` / `_flc` chips) into a `FieldObservation` through
  torchfits: validity mask and inverse-variance weight from the image, error and
  data-quality extensions under an explicit `DQPolicy`, `BUNIT`, exposure time and band
  from the headers, and an optional sky-box cut that reads only the needed region and
  records its offset in `detector_x0_pix` / `detector_y0_pix`.
- `torchsky.map.operators.NestedKernelPSF`: a PSF with a sharp core and wings out to hundreds of pixels,
  applied at a cost that grows with the logarithm of the extent instead of its area. The kernel is split
  into shells sampled at pitches 1, 2, 4, ... pixels and combined by FFT; it may vary over the detector
  through a grid of region nodes blended with tent weights. It follows the PSF conventions (impulse-response
  stamps, zero outside the image, edge loss, exact adjoint, flux conserved for source-conditioned unit-sum
  kernels) and the `ConditionedKernelPSF` call contract, so `LearnedPSFOp(model=...)` and tiled rendering
  accept it. Experimental.
- `torchsky.instrument.JWSTDataQuality`, `WFC3IRDataQuality` and `ACSWFCDataQuality`: named
  data-quality bit tables (`IntFlag`) for `DQPolicy(reject_bits=JWSTDataQuality.DO_NOT_USE | ...)`.
  The HST tables name only the bits that are certain; the docstrings list what is left out.
- `DQPolicy.strict()` and `DQPolicy.accept_all()` spell the two meanings of `reject_bits`
  (`None`: any set bit rejects; `0`: no bit rejects), now documented. Behaviour is unchanged.
- `field_observation_from_exposure_file(..., with_extras=True)` returns an `ExposureFileData`
  with the data-quality plane, `INSTRUME`, `DETECTOR` and the `AREA` plane;
  `zero_fill_unusable=False` keeps the file's science values in unusable pixels; HST ACS band
  names are read from `FILTER1` / `FILTER2`, ignoring `CLEAR*`.
- `FieldValueSemantics` and `VALUE_SEMANTICS` (in `map.field.types`): the accepted values of
  `FieldObservation.value_semantics`, with the new `"count_rate_per_pixel"`.

### Changed

- **Experimental field likelihoods renamed** so the names say what they do;
  "native" clashed with the C++ `_native`/`_cpp` layer. The 0.5.0 names still
  resolve, emit `DeprecationWarning`, and are removed in 0.7:

  | 0.5.0 | 0.6.0 |
  |---|---|
  | `map.field.native_response.NativeFieldResponse` | `map.field.source_quadrature.SourceQuadratureResponse` |
  | `map.field.native_response.bind_native_field_likelihood` | `map.field.source_quadrature.bind_source_quadrature_likelihood` |
  | `map.field.native_response.render_native_field_observation` | `map.field.source_quadrature.render_source_quadrature_observation` |
  | `map.field.native_integrated_read.NativeIntegratedReadLikelihood` (also `map.field.`) | `map.field.source_quadrature_read.SourceQuadratureReadLikelihood` |
  | `map.field.native_integrated_read.bind_native_integrated_read_likelihood` (also `map.field.`) | `map.field.source_quadrature_read.bind_source_quadrature_read_likelihood` |
  | `map.field.joint_native_likelihood.bind_joint_native_likelihood` | `map.field.shared_nuisance_likelihood.bind_shared_nuisance_likelihood` |

  `examples/native_operators.py` is now `examples/source_quadrature_operators.py`.
- **Lensing operators are opt-in.** `ConvergenceMagnification`,
  `magnify_flux`, `scale_source_sizes` and `magnification_bias` moved from
  `torchsky.map.operators` to `torchsky.map.operators.lensing`, and
  `MagnificationResidual` is now `LearnedConvergenceCorrection`. The old
  `torchsky.map.operators` and `torchsky.map.operators.magnification` names
  warn and are removed in 0.7. `ConvergenceMagnification` now defaults to
  `max_magnification=None` like the functions (it clamped at 100): the law is
  applied exactly and the linear law raises for `kappa <= -1/2`. Pass
  `max_magnification=100.0` for the old clamp.
- Celestial `WCS.pixel_to_world` / `world_to_pixel` trace as a single graph
  under `torch.compile(fullgraph=True)` on CPU and MPS. This covers every
  supported projection plus SIP and TPV. Host syncs that broke the graph are
  gone:
  - the MPS float32 precision estimate is computed once per WCS (and per
    `.to()`), not read back on every call;
  - AIT, GLS, HPX and PCO no longer branch on `valid.all()` / `.any()`;
  - projection PV constants are host floats.

  Inductor's Metal backend (PyTorch 2.14) has no `hypot`/`copysign`
  codegen, so traced MPS graphs use `sqrt(a*a + b*b)` and a sign `where`
  instead. The SIP and TPV inverses still exceed Metal's 31 kernel buffers
  under Inductor on MPS; they trace and run with `backend="eager"` there.

  While tracing, the SIP and TPV inverses run a fixed number of masked Newton
  steps that reproduce the eager stopping rules. Eager calls keep their
  early-exit loops and the native CPU TPV kernel. Eager TPV world→pixel no
  longer counts unconverged points (a device sync) unless
  `TPV.set_invert_trace(True)` is on.

  **Compiled TPV world→pixel is slower than eager on CPU.** Inside
  `torch.compile`, TPV world→pixel runs all 20 Newton iterations on every
  point, while eager float64 CPU calls use the native per-point kernel. On a
  loaded 4-thread CPU, 1e6 points took 1.3 s compiled against 0.24 s eager.
  Pixel→world, TAN and SIN get faster when compiled (2–7× on CPU, 3–8× on
  MPS).
- **No more per-PyTorch-minor lock.** The native extension no longer includes
  or links libtorch:
  - tensors enter its kernels through DLPack without a copy, and results come
    back the same way;
  - the `torch.ops.torchsky` operators are registered with `torch.library` from
    Python, with the same schemas, fake kernels and autograd as before;
  - the extension is built against the CPython stable ABI.

  One `cp312-abi3` wheel per platform therefore serves CPython 3.12 and newer
  and every PyTorch from 2.13 up. The requirement is now `torch>=2.13` with no
  upper bound (tested up to 2.15) and `torchfits>=1.3.3`, the first
  torch-free torchfits, and building from source no longer needs PyTorch. The release
  tooling (`scripts/torch_support.json`, `scripts/release_lane.py`) tracks a
  tested floor and a tested maximum instead of one torch lane per release.
- `import torchsky` (and `torchsky.sphere`, `torchsky.wcs`, `torchsky.catalogs`)
  no longer loads torchfits: the FITS readers and writers in `sphere.compat`,
  `sphere.stmoc`, the spectral weight loader and `wcs.astrometry_net` import it
  when called.
- **Scalar spherical harmonic transforms off the CPU are 100–500× faster.**
  This is the pure-torch path that CUDA, MPS and builds without the native
  extension use. On an M2 with MPS (float32), end to end:

  | nside | map2alm, before → after | alm2map, before → after |
  |---|---|---|
  | 256 | 5.0 s → 9 ms | 2.8 s → 12 ms |
  | 512 | ~25 s → 46 ms | ~23 s → 52 ms |

  At nside 512 that is on par with ducc0 on four CPU threads. How:
  - The Legendre step uses a cached Legendre table built once in float64,
    so each transform is a few batched matmuls (`sphere._sht_blocks`). The
    table goes on the GPU or MPS when it fits a memory budget: half of
    GPU memory, a quarter of MPS's recommended maximum, or
    `TORCHSKY_SHT_TABLE_BYTES`.
  - Above the budget it uses a blocked, north/south-folded recurrence that
    advances 16 steps per call and can be compiled; on MPS the native float64
    CPU kernel is used instead.
  - The polar rings, one FFT per distinct length before, share a single
    batched Bluestein (chirp-z) FFT off the CPU (`sphere._ring_bluestein`).
  - float32 results are now accurate to ~1e-7 relative, against ~1e-4 to
    1e-3 from the old float32 recurrence.
  - CPU transforms with the native extension are unchanged.
- `sphere.spherical_fourier_features` (and so `SkyCoordinateEncoder`)
  evaluates all frequencies in one broadcast `sin`/`cos`. Results are bitwise
  identical, and a forward plus backward pass is about 13% faster.
- TPV world→pixel inversion:
  - On CPU float64 inputs that need no gradient, it solves each point by
    Newton's method in the native core (`wcs_tpv_invert`), under the same
    seed, step, conditioning and tolerance contract as the torch path.
  - The torch path, still used for gradients and other devices, sweeps only
    the points still moving and evaluates the polynomial and its Jacobian as
    a single matmul over the monomials.
  - Measured against the fused C++ path removed below (4 threads): 0.39× its
    time at 10k points and 0.33× at 1M.
- Faster exclusive association. On CPU the pairwise belief-propagation
  solve (Anderson-accelerated message sweeps) is one torch-free native call,
  `xmatch_bp_pairwise_solve`, with its own thread pool and results
  independent of the thread count. `xmatch_bp_pairwise_sweep` is one sweep.
  The torch path stays the reference for other devices, for gradients and
  for `associate_many`, and agrees with the native sweep to 1e-12. With
  calibration, the calibration and gate rounds first run with the
  independent model until stable; the exclusive rounds then start from that
  calibration (and the closed-form background density), carry the messages
  of kept pairs from round to round, and stop by the same rule. The
  per-bin posterior solve is skipped when there is one calibration bin. On a
  crowded synthetic field (`calibrate="scale_floor"`, 8-core CPU), 50k
  primaries went from 13.4 s to 3.9 s (independent: 2.1 s) and 200k
  primaries from 199 s to 10.5 s (independent: 4.1 s), with 17
  belief-propagation solves instead of 238. The fixed-point iteration also
  empties its difference history on restart (it reused older differences).
- `associate` fits completeness, error calibration and (exclusive) background
  density by maximum marginal likelihood (L-BFGS on the exact or Bethe log
  likelihood) instead of alternating EM. In crowded fields EM moved the error
  scale by about 5 % per round and stopped after 8 rounds at 1.45 for a true 2.0
  on an injected bulge-like field; the direct fit recovers 1.92.
  `HistogramRatio` terms are still updated by EM between fits.
- `FieldObservation(value_semantics=...)` now raises `ValueError` for a string outside
  `VALUE_SEMANTICS` instead of accepting any non-empty string.
- `DQPolicy` reads signed `int8` / `int16` / `int32` data-quality planes as unsigned bit fields,
  so a set top bit no longer sign-extends into higher bits.
- `FieldObservation.to(dtype=...)` no longer narrows a float64 detector-to-sky module to float32;
  float32 modules still widen to float64.
- `crop_field_observation_to_shared_sky_patch` asks `world_to_pixel` for NaN (`fail_closed=False`)
  when the transform accepts it, so an exposure far from the reference patch gives `None`
  instead of a convergence error.
- In float32, `field_observation_from_exposure_file` keeps a pixel usable when `1 / err**2`
  overflows (`err` finite and positive); the weight is clipped to the largest finite value.
- Crossmatch candidates come from a declination-zone index (Gray, Szalay and
  Nieto-Santisteban 2006) shared by every matcher, replacing the strip, HEALPix
  ring and declination-window engines. On a dense 100 deg² field (2″ radius,
  0.3″ jitter, 4 CPU threads) `find="best"` takes 0.61 s at 10⁶ × 10⁶ rows
  (was 1.48 s; scipy `cKDTree` 0.82 s) and 2.8 s at 4 × 10⁶ (was 6.0 s;
  `cKDTree` 4.6 s); `find="all"` 0.89 s and 5.4 s (`cKDTree` 2.0 s and
  9.3 s). Uniform all-sky matching evaluates about one candidate per row. Pair
  sets are unchanged. `right_chunk_size` now bounds the candidate pairs held
  per batch (default 2²²).
- A torch-free native kernel (`cpp_src/crossmatch.cpp`: nanobind `ndarray`
  buffers, its own thread pool, no libtorch) runs the zone index on CPU float64
  for within-radius pairs, nearest matches, counts and exact `k` nearest
  neighbours. It walks left rows in zone order for locality and keeps the `k`
  best per row without materialising all pairs. The torch path remains the
  reference and the CUDA/MPS path; both give identical pairs and separations.
  Dense field, 4 threads: `find="best"` 0.27 s at 10⁶ and 1.3 s at 4 × 10⁶
  (scipy `cKDTree` 0.52 s and 3.0 s); `match_nearest(k=2)` 0.55 s at 10⁶
  (`cKDTree` 0.72 s), on crowded clusters 0.63 s (`cKDTree` 0.85 s).
- `crossmatch_sky(..., index=)` and `Catalog.crossmatch(..., index=)` take a
  `SkyIndex`, usable with every matcher and `find` mode; `right_index=`
  (a `HealpixIndex`) is removed. `Catalog.crossmatch` no longer builds a
  HEALPix index sized for ~64 rows per pixel, which produced ~3000
  candidates per row at 1″ and unbounded memory.
- `CatalogState.propagate_angular` carries proper motion to the new epoch
  (projected onto the new local basis, as `SkyCoord.apply_space_motion` does
  without distance), and it raises when a row changes epoch without finite proper
  motion. Before, the proper motion stayed at its old-epoch value, and NaN proper
  motion silently produced NaN positions.
- `Catalog.crossmatch(target_epoch=...)` propagates per row (see
  `Catalog.propagate`) and takes `motion_prior=`; `motion="angular"|"physical"`
  is removed. Before, `"physical"` raised for a whole catalogue as soon as one
  row had a non-positive parallax.
- Motion Jacobians are pinned against astropy central finite differences:
  proper motion only, to 1e-6 relative; full space motion, to 5e-6 relative.
  Both hold at Dec ±89.98° for a Barnard's-star path that crosses the pole.
- The torch crossmatch path (CUDA and builds without the native kernel) orders
  `find="all"` pairs per left chunk instead of with three global sorts, so
  output-sized temporaries no longer exhaust GPU memory (crowded 1e7 on a
  10.75 GB GPU ran out of memory).
- Read-only NumPy inputs (Arrow / polars `to_numpy` views) are copied before
  conversion instead of triggering torch's non-writable-array warning.
- MPS crossmatching runs on the CPU in float64 and returns results on MPS. Before,
  positions were cast to float32 on MPS, and pairs within ~0.1 arcsec of the
  threshold could be wrong (about 6% of pairs in a 0.3 arcsec crowded-field match).
- `crossmatch_self` with a per-row radius keeps a pair when
  `separation <= max(radius[i], radius[j])`; it used `radius[i]` only, so the
  result depended on row order.
- `max_pixel_radius` is the closed-form HEALPix bound (`Healpix_Base::max_pixrad`)
  instead of a three-pixel sample (same values), pinned by an exhaustive test.
- `wcs.match_approximate_wcs` uses one exact nearest-neighbour query instead of
  a per-row disc query loop; ties go to the lowest reference row. Its `index=`
  is a `SkyIndex`.
- **`torch.compile` no longer unrolls the chunk loops of the pure-torch square
  drizzle, separable (Lanczos / Keys) resampling, zone query and exclusive BP
  solve.** Each loop now runs eagerly (it is skipped by Dynamo) around one
  fixed-shape tile or step function that Dynamo compiles once and reuses; host
  checks (finite coordinates, invalid footprints, candidate counts,
  convergence) stay outside the compiled graphs, and the last tile is padded
  under compilation only. A single-chunk resample is still traced inline with
  its caller. Eager results are unchanged: bitwise for the drizzle, the sampler
  and the zone query on CPU, to rounding (~1e-15) for the BP messages. The
  native CPU kernels are untouched. Cold compile on an M2 (4 threads, inductor
  caches off, heavily loaded host): Lanczos-3 resample of 1024² from 396 s to
  41–61 s on CPU and from 114 s to 14–23 s on MPS; square drizzle of 512² from
  66 s to 15–23 s on CPU. A 4096² Lanczos-3 resample now compiles in 44 s (CPU)
  / 11 s (MPS) and runs in 4.6 s / 0.37 s instead of 27 s / 2.6 s eager. Warm
  compiled times at the smaller sizes are unchanged within the noise of that
  host.
- **`sphere.healpix.query_strip` follows healpy `query_strip`.** For
  `theta1 < theta2` it returns pixels with centre colatitude in
  `(theta1, theta2]`; for `theta1 >= theta2` it returns the two polar strips
  north of `theta2` and south of `theta1`, where it used to swap the bounds.
  Ring membership is decided with healpy's `ring_above`, so ties on a ring
  centre agree. `inclusive=True` adds one ring on each side, as healpy does,
  instead of padding by the maximum pixel radius. Checked against healpy
  1.20.1 for nside 1–256 (936 cases, including ring-centre ties). The ring
  and NESTED outputs are built from ring ranges with no per-ring Python loop.
  `tests/test_healpix_compile.py` checks that the pure-torch point operations
  compile to one graph and give exactly the eager pixels.

### Removed

- The fused C++ WCS path (`torch.ops.torchsky.wcs_*`, on by default for CPU
  float64 inputs without gradients), the other native WCS shortcuts and
  `torchsky.wcs.functional`. Every projection now runs the pure-torch path
  (TPV inversion excepted, see Changed). Results agree to rounding.
  Measured against the fused path on 4 threads:
  - 10k and 1M points: within 0.33–1.02× of its time, apart from AIT
    world→pixel at 10k (1.19×). TAN, SIN and AIT pixel→world at 1M take
    0.35–0.37× of its time.
  - 100 points: the fixed per-call cost of tensor ops adds 20–70 µs. For
    example SIN takes 0.08 ms instead of 0.05 ms, and TPV pixel→world 0.13 ms
    instead of 0.06 ms.

  Also removed: the `TORCHSKY_WCS_FUSED` / `TORCHSKY_CPP_DISPATCHER`
  environment variables, `_ops.has_wcs_dispatcher` and
  `TPV.set_cpp_invert_max_points`.
- `crossmatch_sky_from_arrow`, `crossmatch_sky_streamed`, and the Arrow /
  `streamed=` branches of `catalogs.crossmatch`: read tables with torchfits,
  pyarrow or polars and pass tensors; match batches against a prebuilt
  `SkyIndex`. `catalogs.index.nested_neighbor_rings`,
  `candidate_coarse_pixels_for_left_pixel` and `HealpixIndex.expand`
  (crossmatch-only helpers) are removed.

### Fixed

- `associate_many` raised `IndexError` when a secondary catalogue had no
  candidate within the gate. Such a catalogue is now absent from every
  hypothesis (`-1`), and the posteriors are those of the association without
  it.
- WCS validity masks now match WCSLIB (Astropy) outside each projection's
  domain:
  - pixel→world rejects plane points whose native longitude/latitude falls
    outside ±180°/±90° (WCSLIB `prjbchk`). CAR, CEA, MER, CYP, SFL, PAR,
    MOL, AIT, the conics, BON and PCO used to rotate them onto the sphere
    as valid coordinates.
  - AZP pixel→world no longer accepts points beyond the perspective horizon
    (its check used the clamped value). CYP pixel→world uses WCSLIB's closed
    form instead of a clamped Newton solve that returned ±90° outside the
    domain. XPH pixel→world applies WCSLIB's facet bounds.
  - world→pixel returns NaN for sky points that have no plane position:
    the far hemisphere of SIN (orthographic and slant), AZP and SZP overlap
    and divergence regions, ZPN colatitudes beyond the polynomial's turning
    point, and COP points behind the cone apex. They used to map to pixels
    of the visible side.
- WCS projections in float32 (MPS) lost half their digits near the reference
  point, face centres or poles. The affected formulas were `1 - cos`,
  `asin` of a value near 1, and `log(cos x)`. They now use exactly equivalent
  forms that avoid the cancellation, so errors stay at a few float32 ulps:
  - SZP pixel→world near the reference point: up to 100″ before;
  - AIR in both directions: up to 70 px before, and pixel→world rejected
    most points because its 1e-10 convergence test is out of float32 reach;
  - QSC near face centres, TSC/CSC/QSC near the poles, and HPX/XPH near the
    poles: up to 2e-3 deg before.

  HPX/XPH facet-edge tests also allow for float32 rounding of the plane
  coordinate. float64 results change by no more than rounding.
- SIP world→pixel fails closed. Points the Newton inverse cannot solve now
  return NaN, as TPV already did: targets with no pre-image, or none within
  reach in `max_iter` steps. Before, the last iterate was returned as if it
  had converged, which gave pixels hundreds of pixels off. A point counts as
  converged when its distorted position is within 1e-6 px of the target, or
  within the float32 rounding floor on MPS. Converged results are unchanged.
  Image-plane resampling treats such target pixels as outside the source
  image (no coverage), as its round-trip check already did.
- `WCS` rejects ZPN headers that WCSLIB `zpnset` rejects, raising
  `ValueError` at construction. These are polynomials with no non-zero
  `PV2_m`, and polynomials of order ≥ 2 with `PV2_1 <= 0`. Before, a header
  without ZPN terms silently became `r = w`, and a zero or negative slope at
  the pole was accepted.
- TPV world→pixel in float32 (MPS) returned NaN for points whose residual
  could not reach the fixed 1e-11 deg tolerance. Convergence is now judged
  against a per-point floor set by the dtype's rounding error. On a
  1M-point MPS run there are no NaNs, and the error is at most 0.1 px.
  float64 is unchanged.
- Exclusive association and `associate_many` tie-breaking take their int64
  minimum scatters through `device._scatter_reduce_int`, so they run on MPS
  under macOS 14. The direct int64 `scatter_reduce("amin")` raised there.
- Gradients that were NaN on PyTorch < 2.14 are finite on every version:
  - the SZP pixel→world Jacobian next to the reference point;
  - the IFU angular distance at coincident points.

  In both, a discarded branch evaluated an infinite derivative (`asin` at 1,
  `sqrt` at 0).
- `device._scatter_reduce_int` falls back to the CPU on MPS backends that
  have no int32 scatter (PyTorch < 2.14 on macOS < 15). It used to raise.
- Spherical pixel areas (`map.operators.pixel_area`) form the triangle triple product from
  edge vectors. The old corner-vector form lost about 1e-4 relative accuracy
  on 0.03″ pixels; the new one is accurate to 6e-9 against the exact TAN area.
- Field rendering (`render_field_observation`, its distribution and tiled
  variants, `predict_observation_mean`) follows the observation's device when
  the model has no parameters or buffers. It used to fall back to the default
  compute device, so a CPU observation rendered on CUDA.
- Combined device-and-dtype conversions in `catalogs` and
  `map.exposure_fit` go through `torchsky.device._to` / `_as_tensor`, avoiding
  the MPS float64 copy that returns zeros.
- `encoders.SkyCoordinateEncoder` wraps RA by whole turns around a new
  `lon_center_deg` (default 180, so in-range RA is unchanged) before its
  longitude features. Non-integer frequencies, which domain-scaled field
  encoders use, made RA and RA ± 360° encode differently and put an O(1)
  feature jump at RA = 0 inside patches that cross it; field encoders now
  centre the seam opposite the domain. `spectral_frequencies=0` works
  (`log_spectral_encoding(num_frequencies=0)` returns the log coordinate).
- `map.field.make_residual_tile_basis_field` initialised atoms with
  `prediction - data` and added them, doubling the residual; atoms now correct
  it. Tile centres honour `detector_to_sky` and are encoded in float64.
- `OperatorConstrainedFeatureField.spectral_observability_penalty` reused
  cached weights for bands with the same name and bounds but a different
  transmission table, passband or wavelength, and raised `TypeError` when
  named and unnamed bands were mixed.
- `render_field_cube` and `render_field_healpix` render float64 models in
  float64 (float32 on MPS) instead of rounding wavelengths and outputs to
  float32.
- `SpectralSkyField.penalty` is zero, not NaN, with fewer than three basis
  terms; `ChromaticPSFField.basis` keeps float64 wavelengths.
- `magnify_flux`, `scale_source_sizes`, `magnification_bias` and
  `ConvergenceMagnification` reject `1 + 2 kappa <= 0` under the unbounded
  linear law instead of returning negative flux or NaN.
- `resample_healpix` on a sparse `HealpixResult` now gives the same map and
  weight as on the dense result: it interpolates `weight * data` and `weight`
  with the plain HEALPix weights. It used to average the data over neighbours
  without their weights, and treated a cell whose value is exactly 0 as
  uncovered.
- On MPS, `blot` and the `coadd_image_plane` projections that need a world
  grid (non-TAN targets, SIP sources) map pixels to source pixels in float64
  on the host and move only the pixel coordinates, as the other projections
  already did. With a float32 world grid the samples were off by up to 1.4
  pixels at 0.04 arcsec, and the SIP round-trip check dropped covered pixels.
- `build_ifu_cube` (and `coadd_cube` with `IFUObservation`s) reports in
  `band_metadata.observation_count` the number of exposures that reach each
  spectral plane, as the other cube paths do. It was at most 1.
- `torchsky.map.product.estimate_map_inverse_variance` in its default
  `native_inverse_variance` mode returns the propagated inverse variance of a
  product that carries one. It returned the coadd weight instead, which differs
  after resampling, for relative weights and for flux products.

## [0.5.0] - 2026-11-03

First stable release. Identical to 0.5.0rc2 apart from the version number;
the changes since 0.5.0rc1 are listed under 0.5.0rc2 below.

## [0.5.0rc2] - 2026-11-03

Second release candidate for 0.5.0. Changes since 0.5.0rc1:

### Added

- `cosmology.lookback_time` and `cosmology.age` (also as `FLRW` methods), and
  `HUBBLE_TIME_GYR`.
- `weighting="energy"` for `synthetic_photometry` and
  `synthetic_photometry_bands`; `vega_spectrum` and `vega_zero_point` from the
  packaged CALSPEC `alpha_lyr_stis_012` spectrum.
- G23, F19 and GCC09 extinction laws (`law="g23"`, `"f19"`, `"gcc09"`). They
  raise outside their calibrated wavelength and R_V ranges.
- `sphere.pixwin` computes the HEALPix pixel window (T and P) for any nside up
  to 8192 and l up to 4·nside, instead of shipping nside 8/16 tables.
- `torchsky.wcs.WCSHeaderWarning` for header interpretations that go beyond
  the literal `CTYPE`.
- API reference for `gaussian_nll` and `poisson_gaussian_nll`.
- Real-data examples: MaNGA IFU cube, Legacy Surveys multi-band field fit,
  Gaia forced photometry against Tractor; the distortions guide uses real
  HST SIP and DECam TPV headers.
- More real-data examples: JWST SMACS 0723 NIRCam coadd checked against the
  MAST mosaic; Legacy Surveys g/r/i/z on sparse HEALPix maps; a MaNGA cube
  rebuilt from fibre spectra; Gaia DR3 Pleiades proper motions in the
  Galactic frame; SFD dust at Gaia positions with G23; SDSS photometric
  redshifts with the experimental encoders; SDSS/GALEX/eFEDS footprint
  algebra with MOCs. TAP and SQL query snapshots are pinned by sha256.

### Changed

- WCS: SIP coefficients without a `-SIP` `CTYPE`, and ambiguous `TAN`+`PV`
  headers, raise `ValueError`. `TAN` with SCAMP `PVi_m` (m ≥ 5) is read as TPV
  with a `WCSHeaderWarning`.
- `observed_altaz` always applies precession; TT−UTC defaults to the built-in
  leap-second table and raises outside it.
- `coadd_healpix` integrates the exact image overlap for HEALPix cells coarser
  than the kernel support, so cells larger than the image no longer drop out.
- HEALPix mapmaking applies the FK5→ICRS frame bias, with
  `CelestialFrameBiasWarning`, instead of treating FK5 as ICRS.
- Planck power-spectrum example: apodised Galactic mask, monopole and dipole
  removal, w₂ normalisation.

### Fixed

- Integer `amin`/`amax` scatters on MPS (MOC normalisation, one-to-one
  crossmatch, connected components, response bit planes) no longer need 64-bit
  Metal atomics, which some Apple GPUs lack ("Failed to create pipeline state
  object"); they reduce in int32 when values fit and on the CPU otherwise.
- Without the compiled extension (and on MPS), `query_disc`, RING
  `query_polygon` and `SphericalPolygon.pixelize` no longer scan every sky
  pixel: a torch quadtree descent selects the same pixels as the native path
  in `O(boundary cells x depth + output)`. A 1.6-arcmin polygon at NSIDE 2^12
  took 38 s and now takes 0.02 s; NSIDE 2^16 takes 0.04 s. RING polygons whose
  centroid lies outside them also no longer scan the full sky with the
  extension.
- `SparseHealpixMap` with `nside_coverage` no longer allocates a dense
  lookup block of `(nside / nside_coverage)**2` values per covered coverage
  cell when that exceeds eight slots per stored pixel; such maps look values
  up by binary search on the sorted pixels. A sparse NSIDE 2^19 coadd needed
  6 GiB per map and ran out of memory on a 10.75 GiB GPU.
- IFU drizzle with `aperture_diameter_arcsec`: spaxels that a fibre misses or
  only touches got rounding-noise weights (~1e-16) that differed between CPU
  and CUDA and switched whole spaxels on or off. Overlap fractions within the
  computation's rounding bound (64 eps) are now zero.
- MPS: moving an MPS tensor to CPU float64/complex128 in one `.to(device,
  dtype)`, `torch.as_tensor` or `.to(other)` call returns zeros in PyTorch
  2.14. Every such conversion in torchsky now moves first and casts on the
  target device; a test keeps fused conversions out of `src/`.
- WCS: TPV `PV1_3`/`PV1_4` are no longer read as `LONPOLE`/`LATPOLE`; DECam
  fields were mirrored about `CRVAL`.
- `spectral_axis_from_header` and `SpectralReader.read_data_cube` read
  `CDi_i` and `PCi_i·CDELTi` and apply `-LOG`, `-F2W`, `-V2W` and grism
  codes through the WCS; unsupported codes raise. MaNGA cubes were truncated
  at 8184 Å instead of 10354 Å.
- `SkyConfig(mode="per_exposure")` works on images no larger than the mesh.
- `Passband.to("mps")` holds the curve in float32 instead of raising.
- `MOC.read_fits` reads IVOA MOC files without `EXTNAME='MOC'` (the table in
  the first extension with `ORDERING='NUNIQ'`, as CDS publishes them); time
  MOCs and non-ICRS coverage raise.
- `build_ifu_cube` and `plan_ifu_cube_response` are vectorised: the same plans
  as before, 780k MaNGA samples in about 10–80 s instead of 30–40 min.
- `IFUObservation.aperture_diameter_arcsec` gives fibre spectra exact circular
  footprints; `drizzle` without a footprint raises for samples with no
  neighbour in their own slice, instead of borrowing one from another slice.
- `IFUSliceWCS.to` works around a PyTorch 2.14 MPS bug where an MPS float32 to
  CPU float64 copy returns zeros.
- Scalar `R_V` and `E(B-V)` in `extinction.a_lambda` are built in the
  wavelength dtype, with no float32 round-trip.
- The Horsehead example draws the MOC boundary its caption describes.
- Coadd weights with interpolation kernels (bicubic, Lanczos, Catmull-Rom,
  Mitchell-Netravali) renormalise the kernel taps that fall off an input image,
  so a constant inverse-variance map keeps its value up to the edge. Lanczos3
  gave up to 1.2× the input inverse variance next to an edge (0.98 instead of
  0.81 in the maps overview), which also over-weighted that exposure in the
  mean. The fix covers `coadd`, `coadd_cube`, `coadd_healpix` and `blot`;
  `inverse_variance` stays the exact variance of the linear estimate.
- `exact_intersection`, `exact_union` and `exact_difference` on polygons, caps
  and multipolygons return a `SphericalBooleanRegion` instead of a
  `NativeApproximateSphericalRegion` whose `area()` raised.
  `SphericalBooleanRegion.area()` is the exact area of polygon combinations,
  clipped from the great-circle edges; caps, boxes and ellipses raise
  `ValueError`. `exact_area()` and native `to_exact` regions use it.
- `SphericalMultiPolygon.area()` counts overlapping members once; it summed
  them. `SphericalCap.area()` returns the analytic cap area.

## [0.5.0rc1] - 2026-11-03

First public release of torchsky: a torch-native astronomy stack for sky
coordinates, geometry, maps, observation operators, scenes, and field models.
FITS file and table I/O is provided by the separate
[`torchfits`](https://github.com/astroai/torchfits) package, a required
dependency.

### Supported lanes
- Python 3.12, 3.13 and 3.14 (full CI suite on each, plus macOS for 3.14).
- PyTorch 2.14; torchfits 1.3.1 or newer.
- CPU, CUDA 13 and newer (PyTorch `cu130`, validated on an NVIDIA H100), and
  Apple MPS.
- Linux x86_64 and macOS 14+ arm64 wheels.

### What torchsky provides
- **WCS** (`torchsky.wcs`): FITS WCS for celestial, spectral, and time axes with
  every standard projection, SIP, TPV, TNX/ZPX, paper-IV lookup tables, `-TAB`
  axes, exact Jacobians, footprints, and astrometric fitting; batched,
  differentiable, and device-aware.
- **Coordinates** (`torchsky.coords`): `CatalogState` with proper motion,
  parallax, radial velocity, and epochs; geometric frame conversion; observed
  place (ERFA Earth ephemeris, aberration, light deflection, refraction)
  through explicit `EarthOrientation` and `Site`.
- **Sphere** (`torchsky.sphere`): HEALPix pixelization (RING and NEST),
  queries that descend the quadtree, spherical regions, MOC and sparse maps,
  spherical harmonic transforms (scalar and spin) with `lmax` up to
  `3*nside-1`, and a healpy-shaped `torchsky.sphere.compat` namespace.
- **Catalogs** (`torchsky.catalogs`): sky-indexed cone search and crossmatch with
  `best1`, `best2`, `mutual`, `symmetric`, and `all` modes, per-row errors,
  `join_rows`, exact matching, self-matching groups, and multi-catalog
  matching, with exact pair-set parity with STILTS on Gaia DR3 x 2MASS x AllWISE.
- **Mapmaking** (`torchsky.map`): planned resampling and coaddition (Lanczos,
  bilinear, nearest, exact square drizzle), weights, masks, DQ and variance
  propagation, automatic memory-budget tiling, HEALPix maps, and IFU cubes
  (`coadd_cube`) with flux-conserving spectral rebinning.
- **Spectra and photometry**: spectral axes, relativistic Doppler shifts,
  passbands and synthetic photometry, dust extinction laws, and differentiable
  FLRW cosmology.
- **Scenes and fields** (`torchsky.scene`, `torchsky.field`): observation
  operators, source-catalog scenes, continuous field models, chromatic PSF
  fields, and fitting; the native response and likelihood layer is experimental.
- **ML and tooling**: `torchsky.transforms` and `torchsky.encoders`, the
  `torchsky` command line, and `torchsky.viz` all-sky figures
  (`plot_healpix_sky`, `plot_sky_grid`) on Mollweide, Aitoff, Hammer, and polar
  projections.
- The package ships `py.typed`.

### Highlights
- 135 JWST/NIRCam exposures (566 Mpix) coadded onto a 5025 x 5025 grid: Lanczos-3
  in 221 s on CPU and 65 s on a GPU slice; exact drizzle in 149 s on CPU and
  148 s on GPU. Drizzle output matches STScI drizzle to better than 2e-6 sigma.
- A 280-CCD CFHT/MegaCam tile coadd onto 10k x 10k pixels in 711 s on CPU and
  660 s on a GPU slice with automatic tiling and no out-of-memory failures.
- SIP forward, inverse, and Jacobian use nested Horner evaluation (JWST order-5
  SIP, 4 M points: distort 15x and Newton inverse 4x faster than a power stack).
- Observed place agrees with `erfa.atco13` to 0.012 mas.

Measurement environments and full tables are on the
[Performance highlights](../performance/highlights.md) page.

### Known limitations
- The native response/likelihood layer and `torchsky.map.solvers` are
  experimental and do not establish calibrated real-sky response.
- On a 5025 x 5025 grid, CPU Lanczos-3 is slower than SWarp; the GPU run is
  faster.
- MPS float32 cannot reach half a pixel on very fine WCS grids; a
  `PrecisionWarning` is raised.
- Resampling between FK5 (J2000) and ICRS emits `CelestialFrameBiasWarning`
  (about 20-30 mas); set `RADESYS = 'ICRS'` on headers solved against Gaia.
- Irregular spectral axes in `coadd_cube` use index-space interpolation with a
  recorded bias envelope.

torchsky continues the sky-geometry part of torchfits ≤ 0.3.1; see the torchfits changelog for earlier history.
