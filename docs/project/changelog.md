# Changelog

All notable changes to torchsky are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0rc1] - Unreleased

### Added

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

### Removed

- `crossmatch_sky_from_arrow`, `crossmatch_sky_streamed`, and the Arrow /
  `streamed=` branches of `catalogs.crossmatch`: read tables with torchfits,
  pyarrow or polars and pass tensors; match batches against a prebuilt
  `SkyIndex`. `catalogs.index.nested_neighbor_rings`,
  `candidate_coarse_pixels_for_left_pixel` and `HealpixIndex.expand`
  (crossmatch-only helpers) are removed.

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
