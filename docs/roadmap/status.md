# Roadmap Status

> Roadmap and research note: this file is a time-sensitive planning/status surface, not the current public API contract. For shipped APIs, start with `../api.md`.

## Vision


### The Core Thesis
Classical mapping pipelines stage WCS, photometry, and coadds in isolation, which tends to lose coupled uncertainties and distort noise models. `torchsky` treats survey inference as **one inverse problem**: finding the latent scene and instrument law such that predicted data agree with observations.

The generative map follows the relation:
$$Y \sim \mathbb{P}_{\psi}\big(\,\cdot \mid F_{\theta},\, \zeta,\, c(t)\,\big)$$

- **$F_{\theta}$:** Continuous latent specific intensity (the sky).
- **$\zeta$:** Kinematic latents (redshift, proper motion, acceleration).
- **$c(t)$:** Observational context (pointing, platform ephemeris).
- **$\mathbb{P}_{\psi}$:** Instrument-environment-noise model (convolutions, gains, detector defects).

### The "Sky Away From Detectors" Goal
The ultimate target is a **single continuous scene** that explains multi-epoch, multi-band data without baking detector defects into the sky representation.

---

## 1. Implementation Roadmap

### Phase 1: Static Sky Fields (Current Core)
- **$F_{\theta}$:** Field-like MLPs/encoders over sky + wavelength on tangent patches or spheres.
- **$\mathcal{A}_{\psi}$:** Bandpass quadrature and PSF convolution via `render_field_observation`.
- **Status:** Implemented in `torchsky.map.field`.

### Phase 2: Pretrained Trunks (STRAINER)
- **Main Decision:** Implement transferable encoder pretraining (inspired by the *Surprising Effectiveness of Noise Pretraining* paper).
- **Mechanism:** Pretrain a shared "trunk" (e.g., `StrainerSkyEncoderMLPField`) on synthetic noise and astronomy-shaped fields, then freeze or fine-tune it for specific survey patches.
- **Objective:** Improve supported-pixel fidelity especially in sparse, heterogeneous observation regimes.
- **Status:** Phase 1 (API and format) and Phase 2 (multi-noise pretraining) are implemented in tree.

### Phase 3: Kinematics and Dynamic Skies
- **Rest-frame Modeling:** Define $F_{\theta}$ in rest-frame $\lambda_{emit}$, with explicit Doppler/redshift shifts $z \in \zeta$.
- **Transverse Motion:** Support proper motion $\boldsymbol{\mu}$, parallax $\varpi$, and sky paths $\omega(t)$ for high-precision astrometry.
- **Peculiar Velocities:** Support motion relative to a reference field (v-ref + v-pec) without assuming a specific cosmology.

### Phase 4: Hybrid Scenes and Multi-Survey Fusion
- **Composite Models:** Shared latent fields for diffuse structure combined with discrete source models for bright objects.
- **Multi-Survey Likelihood:** $\mathcal{L} = \sum_s \mathcal{L}_s$. Link Euclid, LSST, eROSITA, and CMB data through shared sky patches while using survey-specific instrument modules $\psi^{(s)}$.

---

## 2. Beyond Maps: Strategic Directions

- **Generative Physics (Improved MeanFlow):** Moving from "fitting a map" to "generative modeling of the latent physics." This allows for rigorous non-Gaussian cosmology constraints.
- **Any-to-Any Modality:** Cross-wavelength field representations as central building blocks, not side products.
- **Uncertainty Quantification:** Taking region-conditional property seriously, incorporating distributional conformal prediction (VSPS) for rigorous constraints.

---

## 3. Feasibility and Scalability

Large-scale survey fusion is not a monolithic single-job problem, but a **systems and engineering challenge**:
1. **Modular Likelihoods:** Surveys share only what the data actually constrain.
2. **Tracts and Tiling:** Processing the sky in overlapping patches and healpix tiles to cap peak memory.
3. **Multi-resolution Branches:** Separate branches for high-res imaging (arcsec) and low-res/harmonic CMB (degree-scale).
4. **Curriculum Fusion:** Starting with Gaia/LSST anchors and progressively adding spectroscopy and high-systematics data.

---

## Code-Grounded Status

This file tracks roadmap status from live code and tests. Treat narrative docs as secondary when they disagree with implementation.

Last updated: 2026-06-14.

## Current Baseline

- `pixi run ci-local` passes.
- Latest local gate result: `1718 passed, 59 skipped, 77 warnings`.
- Focused FITS gate result: `78 passed, 1 skipped`.
- Focused FITS closeout gate result: `30 passed`.
- Focused WCS API gate result: `24 passed`.
- Focused WCS broad replay result: `277 passed, 2 skipped`.
- Focused WCS spectral/grism/velocity result: `29 passed`.
- Focused WCS temporal scale/reference-position result: `15 passed`.
- Focused WCS celestial frame result: `4 passed`.
- Focused WCS alignment correction result: `6 passed`.
- Focused WCS tabular lookup result: `29 passed`.
- Focused autonomous-review hardening gates: WCS/tabular, sparse/MOC, spectral/viz, field diagnostics, API/cold import, scene/operator binding, and spectral cache clear regressions passed in targeted replay.
- Focused field diagnostics/bundle/presets/docs gate result: `45 passed`.
- Focused WCS + field diagnostics/bundle gate result: `96 passed`.
- Focused WCS API + perf gate result: `29 passed`.
- Focused WCS spectral/temporal WCSLIB benchmark result: `pixi run bench-wcs-spectral-temporal` passed at `N=200000`. Spectral parity was exact/roundoff-level (`<= 3.55e-15` world units, `<= 2.39e-12` pixel). Temporal scale parity was exact or `<= 7.28e-12` days after adding UTC leap-day fractional-offset handling. WCSLIB remains faster for the one-axis spectral benchmark; torchsky was faster for UTC/TAI/TT/GPS scale conversion in this run.
- Focused WCS grism/velocity benchmark smoke result: `pixi run bench-wcs-spectral-temporal --n-points 1000 --runs 1 --skip-temporal` passed with added `WAVE-GRI`, `WAVE-GRA`, `FREQ-GRA`, `WAVE-V2W`, `VELO-F2V`, and `FREQ-V2F` cases.
- Focused scene/operator-adapter gate result: `68 passed`.
- Public API surfaces are now grouped into `core`, `compat`, and `experimental` tiers in `torchsky.surfaces`.

## Completed In Current Slice

- API stability tiers:
  - Added `torchsky.surfaces.SurfaceTier`, `torchsky.surfaces.APISurface`, `torchsky.surfaces.API_SURFACES`, `torchsky.surfaces.get_api_surface`, and `torchsky.surfaces.api_surfaces_by_tier`.
  - Marked FITS/WCS/native map/sphere primitives as `core`, compatibility adapters as `compat`, and field/visualization/training-facing surfaces as `experimental`.
- Non-Uniform Wavelength FITS Writing & Reading:
  - Added support to `write_cube_fits` for non-uniform spectral coordinates with parameter `allow_nonuniform=True` and fallback options `"custom_cards"` or standard `-TAB` `"table"`.
  - Reconstructed non-uniform wavelength grids in `spectral_axis_from_header` and `read_data_cube` from custom header cards or coordinate tables.
  - Implemented bin-edge midpoint-based calculation in `spectral_bin_edges_from_header` for non-uniform coordinate grids.
- Matplotlib-based quick visualizers:
  - Added `plot_pseudo_rgb` and `plot_pixel_spectrum` in `torchsky.viz.plots` with lazy matplotlib/pyplot imports to prevent eager dependency loading at startup.
- Lean public API & lazy import optimization:
  - Made `import torchsky` a stdlib-only lazy facade with no eager `torch`, `numpy`, or `torchsky.cpp` import.
  - Kept root exports to core FITS/WCS calls, HDU/header objects, and lazy namespaces.
  - Moved specialized I/O, datasets/loaders, transforms, cache controls, interop helpers, API-surface metadata, and field/model helpers to their owning namespaces.
  - Reduced advertised `torchsky.map` exports to promoted coadd/product calls and result/spec objects, with implementation helpers available through explicit submodules.
  - Resolved circular import constraints by keeping internal module resolution lazy using dynamically resolved module getters inside `__getattr__` and populated `_EXPORTS`.
- Observation-operator core:
  - Added target binding, multi-target binding, solver-hook construction, and normal-equation probing to `torchsky.scene.ObservationOperatorAdapter`.
  - Kept exact adjoints explicit by requiring an `apply_adjoint` hook or target binder for solve/probe workflows, while preserving prediction-only rendering.
  - Added promoted scene-facade gates for exact adjoint identity and conjugate-gradient normal-equation recovery through the existing solver substrate.
  - Added `ObservationOperatorAdapter.bind_image_plane_target(...)` for mapping a target-grid tensor scene into one `ImageObservation` grid with an exact VJP adjoint of the same projection.
  - Added `scene.bind_image_plane_observations_to_target(...)` so several image-plane observations can share one latent target grid and assemble summed normal equations through `BoundObservationOperatorSet`.
  - Routed scalar image-observation weights and explicit weight/mask/primary-beam maps into the bound operator inverse-noise hook.
  - Moved the concrete image-plane inverse binders into `torchsky.map.operators.inverse`, with `torchsky.scene` delegating to the shared map-operator implementation.
- FITS checksum API:
  - Added file/HDU-level checksum write and verify helpers backed by the compiled CFITSIO extension.
  - Deprecated unsupported in-memory checksum claims by making those methods fail closed.
  - Added multi-HDU isolation, data-payload corruption, missing-keyword, invalid-HDU argument, compressed-HDU, and table-mutation checksum coverage.
- FITS header/card API:
  - Added public `torchsky.Card`.
  - Made `Header` maintain ordered card state across assignment, update, insert, append, remove, history, and comment operations.
  - Preserved repeated `HISTORY`/`COMMENT` cards when opening files through `HDUList.fromfile`.
  - Added CFITSIO-backed header card writeback so image/table writes preserve comments plus repeated `HISTORY`/`COMMENT`.
- FITS table mutation and compression:
  - Made explicit mmap table updates reject VLA/string/complex columns instead of silently falling through.
  - Kept `mmap="auto"` fallback behavior for unsupported table-update columns.
  - Added pre-mutation validation for rename/drop column collisions and missing columns.
  - Fixed `compress=True` table writes so table HDUs are emitted as real BINTABLE extensions rather than going through a TensorFrame placeholder path.
  - Fixed in-memory `TableHDU` construction from NumPy numeric columns.
- WCS:
  - **Status: partially complete, not replacement-complete.**
  - Implemented CYP projection forward/inverse behavior and Astropy parity coverage.
  - Re-enabled legacy polynomial, TNX, and ZPX tests that were previously skipped.
  - Fixed slant-orthographic `SIN` PV handling and invalid-domain NaN propagation against Astropy's `outside_sky.hdr` fixture.
  - Fixed ZPN coefficient indexing/units for `PV2_0..PV2_30`, including hole-domain NaNs and inverse parity against Astropy's `zpn-hole.hdr` fixture.
  - Added alternate FITS-WCS version parsing (`key="A"` .. `"Z"`) and active parity coverage against Astropy alternate headers.
  - Added GWCS/Astropy-style WCS metadata surfaces: axis counts, axis names, units, physical types, pixel/array shape, pixel bounds, correlation matrix, image footprint, and in-image masking.
  - Added Astropy-style WCS value/index helpers: `pixel_to_world_values`, `world_to_pixel_values`, `array_index_to_world_values`, and `world_to_array_index_values`, with NumPy array-order behavior.
  - Added Astropy-compatible `all_pix2world(...)` and `all_world2pix(...)` aliases for stacked-coordinate and separate-coordinate call styles with positional `origin`.
  - Added WCS slicing for positive-step array cutouts/views, updating `CRPIX` and `CD` so cropped and strided image arrays retain correct coordinates, including integer-index slicing that fixes and drops uncoupled axes.
  - Added high-impact cube/image axis reduction and subsetting helpers: `dropaxis(axis)` for uncoupled FITS-order axes, `sub(axes)` for selected/reordered uncoupled axes, and `celestial` for extracting a 2D sky WCS from spatial cubes, with explicit fail-closed behavior for coupled axes.
  - Fixed spatial cube slicing/subsetting so extra spectral/time axes keep their `CD` scale after the output header is normalized from `CDELT` to `CD`.
  - Added a Kapteyn-style `mixed_pixel_world(...)` solver for mixed pixel/world coordinate inputs, including the common 2D celestial case with one known pixel axis and one known world axis.
  - Added pure-Torch differentiable WCS entry points for INR/field-model use: `pixel_to_world_differentiable`, `world_to_pixel_differentiable`, `pixel_to_world_jacobian`, and `world_to_pixel_jacobian`.
  - Added torch-native spectral translations for frequency, vacuum wavelength, energy, air wavelength, redshift, radio velocity, optical velocity, relativistic velocity, and rest-frequency/rest-wavelength handling, with Astropy unit-equivalency parity tests where comparable.
  - Added wavenumber (`WAVN`) spectral conversion support with `cm-1`/inverse-meter unit handling and Astropy spectral-equivalency parity coverage.
  - Added validated one-axis FITS spectral `-LOG` WCS algorithm handling for `FREQ-LOG`, `WAVE-LOG`, and `WAVN-LOG`, with Astropy WCS parity coverage for forward and inverse transforms.
  - Added validated one-axis FITS spectral reciprocal compound algorithms for `WAVE-F2W` and `FREQ-W2F`, with Astropy/WCSLIB parity coverage for forward and inverse transforms plus fail-closed `NaN` behavior at reciprocal singularities.
  - Added validated one-axis FITS spectral grism and velocity compound algorithms for `WAVE-GRI`, `WAVE-GRA`, `AWAV-GRI`, `AWAV-GRA`, `FREQ-GRI`, `FREQ-GRA`, `WAVE-V2W`, `VELO-F2V`, and `FREQ-V2F`, with Astropy/WCSLIB parity coverage for forward and inverse transforms.
  - Added leap-second-aware temporal scale conversions for UTC, TAI, TT, and GPS, with Astropy Time parity tests.
  - Added a fast local Keplerian barycentric/heliocentric light-time correction for topocentric temporal WCS use, with Astropy parity tests at seconds-level accuracy; full ephemeris-grade Earth-orientation/apparent-place behavior remains open.
  - Fixed UTC leap-day fractional `TAI-UTC` handling so UTC MJD values inside leap-second days match Astropy/ERFA semantics instead of using a midnight-only step.
  - Expanded temporal unit handling to common FITS/VO sub-day units (`ns`, `us`, `ms`, `s`, `min`, `h`) plus days/years/centuries, and added temporal reference metadata surfaces for `TREFPOS`, `TREFDIR`, `PLEPHEM`, and `OBSGEO-*`.
  - Added exact temporal observation-date metadata parsing for `MJD-OBS`/`DATE-OBS`, `MJD-BEG`/`DATE-BEG`, and `MJD-END`/`DATE-END`.
  - Added temporal duration/midpoint metadata from `MJD-BEG`/`MJD-END`, `TELAPSE`, or `EXPOSURE`, plus `WCS.temporal_axis_indices` and `WCS.temporal_wcs(...)` integration for main WCS headers.
  - Added first torch-native celestial frame utilities: ICRS/Galactic rotation, FK5 Julian-equinox mean precession, WCS `RADESYS`/`RADECSYS`/`EQUINOX` metadata parsing, and Astropy parity tests at classical-precession precision.
  - Added tweakwcs-inspired weighted 2D affine correction fitting for `shift`, `rshift`, `rscale`, and `general` geometries with residuals, RMS, inlier mask, and sigma-clipping diagnostics.
  - Added distortion-free pixel-affine WCS correction helpers, including `to_header()`, `with_pixel_affine_correction(...)`, `with_affine_fit_correction(...)`, and torchsky correction metadata roundtrip.
  - Added differentiable gnomonic tangent-plane frames and sky-coordinate affine fitting so catalog corrections can be solved in a local plane and applied back to RA/Dec.
  - Added a torch-native tabular lookup substrate for FITS `-TAB`-style axes: piecewise-linear `TabularAxis`, monotonic inverse lookup, separable `TabularLookupWCS`, bounds modes, and autograd through local interpolation slopes.
  - Wired independent `CTYPEi` `-TAB` axes into `WCS(..., tabular_axes={axis_number: axis})` for forward, inverse, spatial-cube extra-axis, and differentiable linear-path use.
  - Added `tabular_axes_from_fits(header, tables)` for FITS `-TAB` metadata: `PSi_0` table selection, `PSi_1` coordinate-array columns, optional `PSi_2` indexing-vector columns, `PVi_1`/`PVi_2` `EXTVER`/`EXTLEVEL` selection, independent 1D axes, and the exact separable subset of multidimensional coordinate arrays selected by `PVi_3`.
  - Added direct independent 1D `-TAB` auto-discovery through `WCS(header, tables=...)`, `TensorHDU.wcs`, and `torchsky.get_wcs(path, hdu=...)`, with fail-closed behavior when required coordinate tables are missing.
  - Extended file-backed `torchsky.get_wcs(path, hdu=..., key="A")` / `torchsky.io.get_wcs(...)` to select alternate FITS-WCS versions directly, including alternate-key independent 1D `-TAB` table discovery.
  - Hardened `-TAB` coordinate/index column lookup to use exact names first, then unique case-insensitive matches, with explicit ambiguity errors.
  - Added fail-closed `PVi_3` handling for `-TAB`: separable multidimensional coordinate arrays are recovered into per-axis tabular transforms, while genuinely coupled arrays that vary along non-selected table dimensions are rejected explicitly.
  - Added `CoupledTabularLookupWCS` plus `coupled_tabular_wcs_from_fits(...)` as the first torch-native forward substrate for genuinely coupled multidimensional FITS `-TAB` coordinate arrays, using multilinear interpolation over explicit table-grid axes and fail-closed inverse behavior.
  - Added CI-friendly WCS performance smoke gates for common bulk TAN `pixel_to_world` and `world_to_pixel` paths, with `TORCHSKY_PERF_GATE_WCS_*` overrides.
- Sphere and tiling:
  - Added sparse-map operand validation plus union/intersection sum helpers.
  - Re-enabled the HealSparse upstream `update_values_pix` parity case.
  - Re-enabled the HealSparse sparse FITS write/read partial-map roundtrip gate.
  - Promoted MOC polygon construction onto the shared general spherical-polygon pixelizer so simple non-convex footprints no longer rely on the convex-only HEALPix polygon path.
  - Added `MOC.from_region(...)` and `MOC.to_pixelized_region(...)` to bridge exact/native/boolean torchsky spherical regions, `PixelizedRegion`, sparse maps, and NUNIQ FITS MOC coverage while preserving NESTED/RING ordering semantics explicitly.
  - Reworked `HealSparseMap.from_moc(...)` and `MOC.to_sparse_map(...)` to realize MOC coverage directly as sparse nested pixels instead of materializing a dense full-sky mask, while fixing non-empty `MOC.to_sparse_map(...)` value/sentinel handling and small-NSIDE coverage defaults.
  - Preserved sparse FITS `VALUE` column dtype on write/read for boolean, integer, and float coverage maps instead of coercing every sparse map to `float32`.
  - Added `MOC.contains_pixels(nside, pixels, nest=True)` and routed lon/lat containment through the same range-membership implementation, giving HEALPix-indexed catalogs and sparse maps a direct high-NSIDE coverage predicate.
  - Fixed `MOC.from_sparse_map(...)` to convert RING-ordered sparse-map pixels to NESTED before NUNIQ construction.
  - Enforced sorted-unique sparse-map pixel storage with last-value-wins duplicate handling in `SparseHealpixMap`, so `from_pixels(...)`, direct construction, updates, and `get_values(...)` share the same indexed lookup invariant.
  - Fixed `MOC.from_threshold(...)` to threshold actual sparse-map pixel entries directly, including RING-to-NESTED conversion, instead of reconstructing pixels from coverage-block offsets.
  - Extended `MOC.from_ascii(...)` to parse compact inclusive pixel ranges such as `3/5-7`, with descending ranges rejected.
  - Added optional `MOC.to_ascii(compact=True)` range emission for compact MOC ASCII roundtrips while preserving the existing default text form.
  - Made sparse-map coverage hierarchy explicit for direct construction, dense conversion, pixel construction, updates, and FITS reads: maps with `nside_coverage` now always materialize a derived `coverage_map`, reject invalid coverage NSIDE values, and compute RING sparse-map coverage parents through the NESTED hierarchy.
  - Preserved valid sparse-map `nside_coverage` through `ud_grade(...)`, matching HealSparse-style behavior by retaining the old coverage resolution when possible and lowering it to the output NSIDE when degrading below the old coverage map.
- Map:
  - Re-enabled ground-truth map and multi-field truth tests.
- INR and field models:
  - Added a recommended field-model set so experimental model growth has a narrower endorsed surface.
  - Added fit diagnostics for spatial/spectral leakage, gradient correlation, residual chi-square, PSF residual alignment, uncertainty support ratios, and timing/memory profiles.
  - Made field-fit final-loss evaluation use the same exposure poses as training, so bundle-adjusted results are scored in the refined frame.
  - Added synthetic STRAINER trunk pretraining timing profiles.
  - Removed a bandpass-conditioned field scalar-conversion warning in differentiable wavelength paths and moved external bandpass weights onto the wavelength tensor device.
  - Added read-time coordinate-cache invalidation for `FieldObservation` based on image shape, requested grid shape, and header card values.
  - Added `fit_field_model(..., refine_poses=True)` to create anchored per-observation `ExposurePose` modules, return them on `FieldFitResult.exposure_poses`, and report final pose parameters in diagnostics.
  - Added field reconstruction quality helpers for rendered residual maps, weighted RMSE/MAE, residual SNR, prediction-vs-target sharpness, multiscale coherent residual RMSE, support fraction, and aggregate per-observation summaries.
  - Added optional `gradient_residual_weight` training loss for data-consistent edge fidelity on rendered observation images.
  - Added structured field residual products carrying prediction/target/residual/weight/support tensors plus metrics.
  - Added opt-in band/support-aware quality scorecards with by-band aggregates and exact extrema for worst RMSE, lowest SNR, lowest support, and lowest effective weight.
  - Added support-aware residual-energy tile selection as the first evidence layer for later sparse local-capacity/adaptive-foam field components.
  - Added an explicit experimental residual-tile basis wrapper that adds bounded signed local correction atoms on top of a diffuse field from selected residual evidence.

## WCS Replacement Status

The WCS roadmap must distinguish active gates from remaining replacement claims.

### WCS Done And Active

- Core 2D FITS-WCS celestial transforms for the currently supported projection set: `TAN`, `SIN`, `ARC`, `ZPN`, `ZEA`, `STG`, `CEA`, `CAR`, `MER`, `CYP`, `AIT`, `MOL`, `HPX`, `SFL`.
- Distortion families with active tests: `SIP`, `TPV`, `TNX`, `ZPX`, plus CYP/PV projection parameters, slant `SIN`, and ZPN hole-domain behavior.
- High-dimensional linear extra axes for basic spatial + spectral cube workflows.
- Alternate FITS-WCS version keys (`A` .. `Z`) for standard suffixed WCS keywords.
- Metadata and geometry surfaces needed by downstream map/INR code: axis names, units, physical-type hints, axis-correlation matrix, pixel bounds, `footprint`, and `in_image`.
- Broad Astropy-style image interoperability helpers: `all_pix2world` / `all_world2pix`, value aliases, NumPy-order array index conversion, WCS slicing for positive-step cutouts/views plus integer-index drops of uncoupled axes, and uncoupled axis reduction / selected-axis reordering / celestial WCS extraction for cubes.
- Mixed pixel/world solving for full-pixel, full-world, and the common 2D case with one known pixel axis plus one known world axis.
- Differentiable pure-Torch WCS paths and local Jacobian extraction for optimization, INR renderers, field-model training, and multi-modal geometry encoders.
- Spectral translation primitives through frequency for `FREQ`, `WAVE`, `WAVN`, `AWAV`, `ENER`, `ZOPT`, `VRAD`, `VOPT`, `VELO`, and `BETA`, including `RESTFRQ`/`RESTFREQ`/`RESTWAV` handling and WCS convenience methods for spectral axes.
- First FITS spectral algorithm-code transforms: validated one-axis `-LOG` support for `FREQ-LOG`, `WAVE-LOG`, and `WAVN-LOG`; reciprocal `WAVE-F2W` and `FREQ-W2F`; grism `GRI`/`GRA` for wavelength/frequency/air-wavelength cases; and selected velocity compound axes including `WAVE-V2W`, `VELO-F2V`, and `FREQ-V2F`.
- Temporal scale conversion primitives for UTC/TAI/TT/GPS with a built-in IERS/NIST leap-second table current through the 2026-06-30 no-leap Bulletin C 71 window; common sub-day/year units, observation-date metadata, duration/midpoint metadata, WCS temporal-axis discovery, reference metadata, and fast Keplerian barycentric/heliocentric light-time corrections are active. Full ephemeris-grade Earth orientation, apparent place, and coupled space-time corrections remain open.
- Comparator-only WCSLIB/Astropy benchmark surface for spectral FITS-WCS algorithms and temporal scale conversions: `pixi run bench-wcs-spectral-temporal`. This is benchmark/reference infrastructure only; runtime WCS paths must remain torch-native and must never fall back to WCSLIB.
- Celestial frame primitives for ICRS/Galactic conversion and FK5 Julian-equinox mean precession, plus WCS helpers for `RADESYS`/`EQUINOX` metadata.
- First catalog-alignment correction primitives: weighted 2D affine fitting with shift / rotation+shift / rotation+scale+shift / full affine modes, differentiable gnomonic tangent-plane fitting for sky catalogs, and distortion-free direct WCS header updates from pixel-affine fits.
- First table-driven WCS support: independent one-dimensional lookup axes with differentiable interpolation, monotonic inverse lookup, explicit `WCS` integration for FITS 1-based `CTYPEi` `-TAB` axes, a parser helper for the independent 1D `PSi_0`/`PSi_1`/`PSi_2` plus `PVi_1`/`PVi_2` metadata subset, separable multidimensional coordinate-array recovery via `PVi_3`, a forward coupled-grid lookup helper for genuinely multidimensional arrays, and path/HDUList auto-discovery for the invertible per-axis subset.

### WCS Partially Done

- Spectral/time axes: spectral coordinate translation primitives, validated one-axis `-LOG`, reciprocal, selected velocity, and grism spectral WCS plus temporal scale/unit/date/duration/reference metadata and fast Keplerian light-time primitives now exist, but full FITS Paper III coupled spectral chains and full Paper IV space-time corrections are not replacement-complete.
- Kapteyn-style mixed coordinates: the common 2D celestial mixed solve is active; arbitrary N-dimensional mixtures and non-celestial coupled axes remain open.
- Axis subsetting: uncoupled axis drops, selected-axis reordering, integer-index slicing for uncoupled axes, and celestial extraction from spatial cubes are active; coupled-axis subsetting remains open.
- GWCS/AST-style composition: metadata surfaces exist, but explicit frame/mapping pipeline objects and reusable transform graph nodes are not implemented.
- FITS `-TAB` / lookup-table WCS: independent lookup-axis interpolation is active in the main `WCS` transform when axes are supplied explicitly or discovered from an HDUList/path, and the parser handles independent 1D axes plus separable multidimensional coordinate arrays selected by `PVi_3`; arbitrary coupled table-coordinate arrays now have a forward multilinear lookup substrate, but general inverse solving and main-`WCS` integration for coupled grids are not yet implemented.
- Performance: common CPU/vectorized/C++/compiled paths exist, a CI-friendly TAN forward/inverse smoke gate is active, and spectral/temporal WCSLIB comparator timing now has a focused benchmark surface. Replacement claims still need broader timing baselines across projection, distortion, batch-size, device, spectral/time algorithm, and inverse-solve regimes.
- Compatibility replay: Astropy/WCSLIB fixture coverage is active, but Kapteyn/PyAST/Cornish bridge replays are not yet formal gates.

### WCS Not Done

- Full celestial frame-system conversions beyond the first FK5/ICRS/Galactic slice: FK4/Besselian E-terms, high-precision frame bias, nutation, aberration, proper motion/parallax/radial velocity propagation, and observer-dependent apparent-place behavior comparable to Kapteyn/AST.
- Full FITS Paper III spectral WCS algorithm-code parsing beyond the validated one-axis `-LOG`, reciprocal, selected velocity, and grism cases: coupled nonlinear spectral pixel-axis transforms and broader algorithm-code replay remain open.
- Full temporal WCS reference-position corrections: Earth orientation, ephemeris-grade barycentric/heliocentric light-time corrections, apparent-place behavior, and coupled space-time corrections.
- Generalized AST/GWCS frame graphs, serialization, and conversion to/from approximating FITS SIP/TAB representations.
- Full FITS `-TAB` arbitrary coupled multidimensional coordinate arrays with inverse solving and main-`WCS` integration, plus generalized coupled table-driven distortions beyond the current forward coupled-grid substrate.
- Full tweakwcs-class correction workflows: iterative matching, richer correction metadata serialization, and distorted-WCS header updates.
- Plot-grid/formatting helpers and sexagesimal/unit-aware coordinate formatting at Kapteyn/AST breadth.

## Near-Term Next Work

1. FITS completeness:
   - FITS is closed for the current roadmap slice.
   - Future parity hardening should focus on broader astropy.io.fits/fitsio fixture replay, large-table mutation benchmarks, and additional realistic VLA/table fixtures.
2. WCS completeness:
   - Current active gates cover supported projections/distortions, slant `SIN`, ZPN hole fixtures, alternate WCS keys, axis metadata, footprint, mixed pixel/world transforms, and differentiable Jacobians.
   - Current spectral/temporal slice covers `WAVN`, `-LOG`, `WAVE-F2W`, `FREQ-W2F`, selected velocity compound axes, grism `GRI`/`GRA`, common time units, observation-date metadata, duration/midpoint metadata, WCS temporal-axis discovery, temporal reference metadata, and fast Keplerian barycentric/heliocentric light-time corrections. Next spectral step is coupled FITS Paper III algorithm-chain replay; next temporal step is ephemeris-backed Earth-orientation/apparent-place corrections and realistic observatory fixtures.
   - Use WCSLIB/Astropy for benchmark and parity comparison only. Do not add WCSLIB runtime fallback paths in `src/torchsky`.
   - Add a formal Kapteyn/PyAST/Cornish legacy-bridge replay matrix for sky-system conversions, epochs/equinoxes, mixed coordinates, alternate headers, spectral translations, and formatting/plot grid behavior where comparable.
   - Adopt the GWCS/AST design pattern of explicit transform pipelines and frame/mapping composition for future non-FITS-native WCS, but keep runtime implementation torch-native.
   - Build on the tangent-plane affine correction and direct linear-WCS update primitives with iterative catalog matching, richer correction metadata, and distorted-WCS header update helpers.
   - Build true coupled multidimensional table-coordinate inverse semantics and main-`WCS` integration on top of `CoupledTabularLookupWCS`, including realistic FITS table-array fixtures.
   - Harden high-dimensional spatial/spectral/time axes against Astropy/WCSLIB fixtures.
   - Add more fixture replay gates for mixed projection plus distortion headers.
3. Sphere and tiling:
   - Continue replacing exact-geometry gaps with robust MOC/geometry coverage; MOC now consumes general polygon and exact/native/boolean region pixelization directly.
   - Expand HealSparse metadata/header fidelity beyond the current sparse FITS roundtrip; sparse MOC realization now avoids dense-mask materialization, scalar value dtype is preserved across sparse FITS roundtrips, and coverage-map hierarchy is ordering-aware through construction, updates, FITS reads, and `ud_grade(...)`.
   - Add high-NSIDE query and spin/SHT replacement gates; MOC now has a pixel-native containment predicate for high-NSIDE indexed workflows.
4. Map architecture:
   - Make `ObservationOperatorBundle` the common execution substrate across promoted map paths.
   - Move remaining planning/windowing/I/O policy out of `map/image_plane.py`.
   - Add exact adjoint gates for promoted kernels.
5. INR and fields:
   - Keep new architectures behind explicit experimental factories.
   - Prioritize identifiability, PSF/noise/nuisance models, richer photometric metrics, residual products beyond current rendered residual maps and multiscale RMSE summaries, support masks, leakage metrics, timing baselines, and richer pose-refinement workflows around matching and priors.

## Next Execution Roadmap

This is the proposed work order after the lean public-API cleanup lands.

1. Lean API and documentation lock:
   - Keep `import torchsky` near-zero overhead and make domain namespaces the default way to discover functionality.
   - Keep root names limited to the smallest practical FITS/WCS entry points and core AST objects.
   - Move examples and docs away from root aliases so the public surface stays composable and easy to remember.
2. Observation-operator core:
   - Promote `ObservationOperatorBundle` as the common execution layer for image coadds, map products, field fitting, and later hybrid scene models.
   - Add exact adjoint/normal-equation gates for promoted operators before adding more solver policy.
   - Move residual planning, windowing, and I/O orchestration out of `map/image_plane.py` into smaller operator/product modules.
3. INR and multiband image quality:
   - Treat "best SNR and sharp image of a given sky field" as the field-model acceptance target, not just low pixel RMSE.
   - Extend the current residual/SNR/sharpness metrics into persisted residual products, support-aware scorecards, PSF residual alignment reports, and per-band leakage diagnostics.
   - Keep all promoted fit paths bandpass-aware, PSF/noise-aware, differentiable where practical, and benchmarked against direct stack baselines.
4. Adaptive foam / local capacity:
   - Build the first optional adaptive local-capacity layer on top of the diffuse field: multiscale cells, sparse local basis atoms, or feature tiles that add capacity only where residual evidence justifies it.
   - Keep this representation survey-scale: bounded memory, explicit support masks, sparse activation, and no source-label assumptions baked into the core API.
   - Gate it against synthetic multi-scale scenes, sparse larger-mosaic cases, and public multiband fixtures before presenting it as a recommended field path.
5. Real-data and scale gates:
   - Make SMACS/JWST multiband downloads, direct stacks, INR fits, and residual comparisons a repeatable smoke-to-representative ladder.
   - Add larger rectangular/mosaic benchmark rows for both direct stack and field/foam paths.
   - Track throughput, peak memory, support-aware fidelity, SNR, flux conservation, color leakage, and sharpness together so tuning cannot improve one metric while hiding a regression in another.
6. Longer WCS/sphere closure:
   - Continue WCS replacement work on coupled spectral/time/TAB cases, frame conversions, and correction workflows only where it unlocks map/field fidelity or closes explicit compatibility gates.
   - Continue sphere work on exact geometry, sparse maps, high-NSIDE queries, and SHT/spin gates as shared infrastructure for full-sky and cross-survey products.

## Standing Gates

- Pre-push: `pixi run ci-local`.
- Focused replacement gates should stay domain-specific:
  - FITS vs `astropy.io.fits` and Python `fitsio`.
  - WCS vs Astropy/WCSLIB fixture behavior.
  - Sphere vs healpy, hpgeom, healsparse, astropy-healpix, and spherical-geometry.
  - Map vs reproject, drizzle, SWarp, and Montage where installed.
