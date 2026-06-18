# Roadmap Status

> Roadmap and research note: this file is a time-sensitive planning/status surface, not the current public API contract. For shipped APIs, start with `../api.md`.
>
> Navigation: this file is the high-level roadmap and capability status. Use
> `implementation-ledger.md` for the maintained low-level record of shipped
> improvements, validation gates, and concrete follow-ups. The old
> `../reviews/autonomous_refactor_log.md` is an archive only.

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

Last updated: 2026-06-18.

## Current Baseline

- Full local gate: `pixi run ci-local` passes.
- Latest full local gate result: `1775 passed, 59 skipped, 63 warnings`.
- Latest focused WCS gate result: `134 passed, 8 warnings`.
- Latest focused sphere gate result: `344 passed, 8 skipped, 10 warnings`.
- High-level roadmap and replacement status live in this file.
- Low-level shipped-item records, focused replay counts, benchmark snippets, and concrete follow-ups live in `implementation-ledger.md`.
- The archived `../reviews/autonomous_refactor_log.md` is no longer the active work queue.
- Public API surfaces are grouped into `core`, `compat`, and `experimental` tiers in `torchsky.surfaces`.
- Root import remains a lean lazy facade; heavier functionality is discoverable through owning domain namespaces.
- Current technical center of gravity: common observation operators, support-aware field diagnostics, adaptive local-capacity field components, WCS/sphere replacement closure, and real-data/scale gates.

## Completed In Current Slice

This section is the high-level shipped-capability summary. For lower-level
test counts, benchmark details, and exact follow-up items, use
`implementation-ledger.md`.

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
  - Broadened FITS Paper III compound spectral replay to the full supported F/W/A/V matrix for `FREQ-*2F`, `WAVE-*2W`, `AWAV-*2A`, and `VELO-*2V` algorithms, with Astropy/WCSLIB forward/inverse parity gates and benchmark rows.
  - Added validated one-axis FITS spectral grism and velocity compound algorithms for `WAVE-GRI`, `WAVE-GRA`, `AWAV-GRI`, `AWAV-GRA`, `FREQ-GRI`, `FREQ-GRA`, `WAVE-V2W`, `VELO-F2V`, and `FREQ-V2F`, with Astropy/WCSLIB parity coverage for forward and inverse transforms.
  - Added leap-second-aware temporal scale conversions for UTC, TAI, TT, and GPS, with Astropy Time parity tests.
  - Added normalized temporal reference-position aliases (`TOPOCENTRE`, `GEO`, `BARYCENTRE`, `SSB`, `HELIOCENTRE`, etc.), a fast local Keplerian barycentric/heliocentric light-time correction for topocentric and geodetic observatory temporal WCS use, with Astropy parity tests at seconds-level accuracy; `GEOCENTER` is now explicit no-ephemeris behavior. Full ephemeris-grade Earth-orientation/apparent-place behavior remains open.
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
  - Wired coupled FITS `-TAB` tables into main `WCS(header, tables=...)` forward transforms, `TensorHDU.wcs`, `torchsky.get_wcs(path, hdu=...)`, differentiable Jacobians, and local square-grid inverse roundtrips for both all-axis non-spatial lookup WCS and mixed spatial WCS with coupled extra axes, including a file-backed FITS image plus WCS-TAB table-array replay gate and 2D square-grid inverse ambiguity detection.
  - Added `bench_wcs_tabular.py` / `pixi run bench-wcs-tabular` for coupled `-TAB` lookup and main-`WCS` forward plus ambiguity-safe inverse timing.
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
  - Added a high-NSIDE sphere query gate at `nside=1024` covering HEALPix center roundtrips, neighbors, interpolation weights, sparse lookup, MOC pixel containment, and MOC catalog filtering without dense full-sky materialization.
  - Added sparse-map coverage bridges: `SparseHealpixMap.to_moc(...)`, `SparseHealpixMap.to_pixelized_region(...)`, and `SparseHealpixMap.from_region(...)` now round-trip sparse maps, MOC coverage, `PixelizedRegion`, and exact spherical regions without dense full-sky masks, preserving explicit RING/NESTED ordering and scalar value dtype.
  - Reworked `SkyMaskPipe.combine(...)` to perform AND/OR mask algebra through sparse pixel sets instead of materializing dense full-sky maps, with high-NSIDE sparse coverage tests.
  - Added `torchsky.sphere.harmonics` as the preferred harmonic-transform namespace while keeping `torchsky.sphere.spectral` as a backward-compatible alias.
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
  - Field-fit results now retain the merged `ObservationOperatorBundle` for the input observations and report enabled operator slots in diagnostics, giving field fitting an inspectable bridge into the common map/scene operator substrate.
  - Field renderers now accept an explicit per-observation operator bundle, and `fit_field_model(...)` threads cached bundles through training, final scoring, and diagnostics instead of recomputing observation-law metadata on every render.
  - Fixed field-distribution fallback rendering so PSF/background operators are not applied twice when a model lacks distribution-specific forward methods.
  - Field residual and quality helpers now accept explicit per-observation operator bundles, and fit results retain the per-observation bundle sequence so post-fit quality reports can use the same observation-law metadata as training.
  - Field residual products and quality reports now include PSF residual-alignment metrics for PSF-enabled observations, with scorecard aggregation handling optional metrics across mixed observation sets.
  - Added explicit support-aware quality acceptance reports over weighted RMSE, residual SNR, support fraction, effective weight, and optional PSF residual alignment.
  - Added `render_field_fit_quality(...)` so post-fit quality summaries can consume `FieldFitResult` directly, reusing stored model, pose, and observation-operator metadata unless explicitly overridden.
  - `ContinuousFieldModel` scene rendering now forwards explicit operator bundles into field render and distribution paths, aligning scene-level rendering with the field fit/quality observation-law substrate.
  - Added `scene.operator_bundles_for_observations(...)` to construct ordered per-observation bundle sequences for fit, quality, and implicit-forward workflows.
  - The implicit forward-model shell now accepts per-observation operator bundles in `predict_observation_mean(...)` and `SurveyDataTerm`, so multi-survey loss scaffolds can use the same observation-law metadata path as rendering, fitting, and quality diagnostics.

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
- First FITS spectral algorithm-code transforms: validated one-axis `-LOG` support for `FREQ-LOG`, `WAVE-LOG`, and `WAVN-LOG`; the full supported F/W/A/V one-axis compound matrix for `FREQ-*2F`, `WAVE-*2W`, `AWAV-*2A`, and `VELO-*2V`; and grism `GRI`/`GRA` for wavelength/frequency/air-wavelength cases.
- Temporal scale conversion primitives for UTC/TAI/TT/GPS with a built-in IERS/NIST leap-second table current through the 2026-06-30 no-leap Bulletin C 71 window; common sub-day/year units, observation-date metadata, duration/midpoint metadata, WCS temporal-axis discovery, normalized reference metadata aliases, explicit GEOCENTER no-correction behavior, and fast Keplerian barycentric/heliocentric light-time corrections are active. Full ephemeris-grade Earth orientation, apparent place, and coupled space-time corrections remain open.
- Comparator-only WCSLIB/Astropy benchmark surface for spectral FITS-WCS algorithms and temporal scale conversions: `pixi run bench-wcs-spectral-temporal`. This is benchmark/reference infrastructure only; runtime WCS paths must remain torch-native and must never fall back to WCSLIB.
- Celestial frame primitives for ICRS/Galactic conversion and FK5 Julian-equinox mean precession, plus WCS helpers for `RADESYS`/`EQUINOX` metadata.
- First catalog-alignment correction primitives: weighted 2D affine fitting with shift / rotation+shift / rotation+scale+shift / full affine modes, differentiable gnomonic tangent-plane fitting for sky catalogs, and distortion-free direct WCS header updates from pixel-affine fits.
- First table-driven WCS support: independent one-dimensional lookup axes with differentiable interpolation, monotonic inverse lookup, explicit `WCS` integration for FITS 1-based `CTYPEi` `-TAB` axes, a parser helper for the independent 1D `PSi_0`/`PSi_1`/`PSi_2` plus `PVi_1`/`PVi_2` metadata subset, separable multidimensional coordinate-array recovery via `PVi_3`, a forward coupled-grid lookup helper for genuinely multidimensional arrays, path/HDUList auto-discovery for the invertible per-axis subset, and main-`WCS` forward/local-inverse integration for all-axis non-spatial coupled grids plus mixed spatial WCS with coupled extra axes.

### WCS Partially Done

- Spectral/time axes: spectral coordinate translation primitives, validated one-axis `-LOG`, supported F/W/A/V one-axis compound spectral WCS, and grism spectral WCS plus temporal scale/unit/date/duration/reference metadata and fast Keplerian light-time primitives now exist, but full FITS Paper III coupled spectral chains and full Paper IV space-time corrections are not replacement-complete.
- Kapteyn-style mixed coordinates: the common 2D celestial mixed solve is active; arbitrary N-dimensional mixtures and non-celestial coupled axes remain open.
- Axis subsetting: uncoupled axis drops, selected-axis reordering, integer-index slicing for uncoupled axes, and celestial extraction from spatial cubes are active; coupled-axis subsetting remains open.
- GWCS/AST-style composition: metadata surfaces exist, but explicit frame/mapping pipeline objects and reusable transform graph nodes are not implemented.
- FITS `-TAB` / lookup-table WCS: independent lookup-axis interpolation is active in the main `WCS` transform when axes are supplied explicitly or discovered from an HDUList/path, and the parser handles independent 1D axes plus separable multidimensional coordinate arrays selected by `PVi_3`; arbitrary coupled table-coordinate arrays now have a multilinear lookup substrate, and both all-axis non-spatial coupled grids and mixed spatial WCS with coupled extra axes are integrated into main `WCS` forward/Jacobian/local-inverse evaluation with file-backed table-array replay through `TensorHDU.wcs` and `get_wcs`. The 2D square-grid inverse path now detects multiple distinct candidate cells and fails closed, while higher-dimensional/global ambiguity handling and broader upstream-style fixture replay remain open.
- Performance: common CPU/vectorized/C++/compiled paths exist, a CI-friendly TAN forward/inverse smoke gate is active, spectral/temporal WCSLIB comparator timing has a focused benchmark surface, and coupled `-TAB` forward/inverse timing has a focused torch-native benchmark surface. Replacement claims still need broader timing baselines across projection, distortion, batch-size, device, spectral/time algorithm, and inverse-solve regimes.
- Compatibility replay: Astropy/WCSLIB fixture coverage is active, but Kapteyn/PyAST/Cornish bridge replays are not yet formal gates.

### WCS Not Done

- Full celestial frame-system conversions beyond the first FK5/ICRS/Galactic slice: FK4/Besselian E-terms, high-precision frame bias, nutation, aberration, proper motion/parallax/radial velocity propagation, and observer-dependent apparent-place behavior comparable to Kapteyn/AST.
- Full FITS Paper III spectral WCS algorithm-code parsing beyond the validated one-axis `-LOG`, reciprocal, selected velocity, and grism cases: coupled nonlinear spectral pixel-axis transforms and broader algorithm-code replay remain open.
- Full temporal WCS reference-position corrections: Earth orientation, ephemeris-grade barycentric/heliocentric light-time corrections, apparent-place behavior, and coupled space-time corrections.
- Generalized AST/GWCS frame graphs, serialization, and conversion to/from approximating FITS SIP/TAB representations.
- Full FITS `-TAB` arbitrary coupled multidimensional coordinate arrays with global ambiguity handling and generalized coupled table-driven distortions beyond the current local square-grid coupled path.
- Full tweakwcs-class correction workflows: iterative matching, richer correction metadata serialization, and distorted-WCS header updates.
- Plot-grid/formatting helpers and sexagesimal/unit-aware coordinate formatting at Kapteyn/AST breadth.

## FITS, WCS, And Sphere Consolidated Plan

This is the active tracker for FITS I/O, WCS replacement, and sphere/HEALPix
infrastructure. Update this section when a task ships, moves priority, changes
scope, or gets a new validation gate.

Status key:

- `[x]` shipped and covered by active tests or benchmark gates.
- `[~]` partially implemented; continue work before making replacement claims.
- `[ ]` open.

### FITS I/O Tracker

FITS is functionally strong for the current roadmap slice. The remaining work is
mostly hardening, scale evidence, and API ownership cleanup; it should not add
Python `fitsio` or Astropy runtime dependencies to core `src/torchsky` code.

| Status | Priority | Task | Acceptance gate / update trigger |
| --- | --- | --- | --- |
| [x] | P0 | Keep native CFITSIO-backed image, header, checksum, compression, and table read/write paths as the default core I/O substrate. | `pixi run ci-local`; FITS image/table tests; no core imports of Python `fitsio` or Astropy. |
| [x] | P0 | Preserve ordered FITS header/card behavior, repeated `HISTORY`/`COMMENT`, checksum write/verify, compressed table writes, and validated table mutation failure modes. | Existing FITS checksum/header/table mutation tests remain active. |
| [x] | P1 | Move root FITS implementation ownership behind `torchsky.io.fits` while keeping `torchsky.read`, `torchsky.open`, `torchsky.write`, `torchsky.get_header`, and `torchsky.get_wcs` stable wrappers. | API import tests plus FITS I/O tests pass; docs route new users through `torchsky.io`. |
| [x] | P1 | Build a maintained FITS fixture replay matrix against `astropy.io.fits` and Python `fitsio` for images, MEFs, compression, checksums, headers, BINTABLEs, variable-length arrays, strings, masks, and edge dtypes. | `tests/test_fits_fixture_replay_matrix.py` covers generated MEFs, compressed images, string/boolean/numeric BINTABLEs, and Astropy VLA fixtures. |
| [x] | P1 | Add large-table mutation and scan benchmarks that cover append/insert/delete/update rows, column rename/drop/replace, chunked reads, and mmap/auto fallback behavior. | `benchmarks/bench_fitstable_io.py` emits read/scan rows plus `mutation_append`, `mutation_update`, and `mutation_rename_drop` rows. |
| [x] | P1 | Add realistic radio/VLA-style table fixtures and representative large survey table fixtures. | `tests/test_fits_fixture_replay_matrix.py` includes a deterministic VLA-style visibility table with time, antenna pairs, UVW vectors, complex visibilities, weights, flags, metadata, and mutation-safe row updates. |
| [~] | P2 | Split table implementation by scan/read, Arrow conversion, mutation, and backend policy once behavior is stable. | Backend policy now lives in private `_table_engine`; full scan/read/Arrow/mutation extraction remains a structural cleanup with public table APIs unchanged. |
| [x] | P2 | Define cache policy for FITS image, FITS table, header metadata, and WCS extraction as named subsystems instead of scattered knobs. | `IO_CACHE_SUBSYSTEMS`, `cache_subsystem_policy(...)`, and `clear_cache_subsystem(...)` cover image data, table data, header metadata, WCS extraction, and all-cache invalidation with focused tests. |

### WCS Tracker

WCS remains a core replacement project. Runtime paths must stay torch-native:
Astropy, WCSLIB, Kapteyn, PyAST, Cornish, and GWCS are references, comparators,
or design targets only.

| Status | Priority | Task | Acceptance gate / update trigger |
| --- | --- | --- | --- |
| [x] | P0 | Keep current celestial projection/distortion coverage active: `TAN`, `SIN`, `ARC`, `ZPN`, `ZEA`, `STG`, `CEA`, `CAR`, `MER`, `CYP`, `AIT`, `MOL`, `HPX`, `SFL`, plus `SIP`, `TPV`, `TNX`, `ZPX`, slant `SIN`, and ZPN hole-domain behavior. | `tests/test_wcs_*.py`; parity fixtures against Astropy/WCSLIB where available. |
| [x] | P0 | Maintain Astropy-style metadata, slicing, axis dropping/subsetting, celestial extraction, mixed 2D celestial solve, and differentiable WCS/Jacobian paths. | WCS API, slicing, parity, differentiable, and perf-smoke tests remain active. |
| [~] | P0 | Finish FITS Paper III spectral WCS replacement beyond current one-axis `-LOG`, supported F/W/A/V compound, and grism cases. | Coupled spectral algorithm-chain fixtures replay against Astropy/WCSLIB without WCSLIB runtime fallback. |
| [~] | P0 | Finish FITS Paper IV temporal/reference-position behavior beyond common unit/scale/date metadata, normalized reference-position aliases, explicit TOPOCENTER/GEOCENTER handling, and fast Keplerian light-time approximation. | Observatory fixtures with ephemeris-backed barycentric/heliocentric, Earth-orientation, apparent-place, and coupled space-time checks. |
| [~] | P0 | Complete FITS `-TAB` support for arbitrary coupled multidimensional coordinate arrays: generalized ambiguity handling plus broader fixture replay on top of `CoupledTabularLookupWCS`. | All-axis non-spatial and mixed spatial+extra-axis forward/Jacobian/local-inverse `WCS(header, tables=...)`, `TensorHDU.wcs`, and `get_wcs` integration is active with a file-backed table-array fixture; 2D square-grid inverse ambiguity tests are active; higher-dimensional ambiguity and broader upstream-style fixtures remain. |
| [ ] | P1 | Formalize a legacy bridge replay matrix for Kapteyn/PyAST/Cornish behaviors: sky systems, epochs/equinoxes, mixed coordinates, alternate headers, spectral translations, and formatting/plot-grid behavior where comparable. | New replay docs plus optional benchmark/test tasks that can skip cleanly when legacy deps are absent. |
| [ ] | P1 | Add explicit transform graph/frame mapping objects inspired by AST/GWCS for non-FITS-native WCS composition while preserving torch-native execution. | Internal transform graph can represent existing linear/projection/distortion/TAB paths; no public API churn until proven. |
| [~] | P1 | Grow catalog-correction workflows from weighted affine fits into tweakwcs-class iterative matching, richer metadata serialization, and distorted-WCS header update helpers. | Synthetic and fixture-based catalog alignment tests with correction metadata roundtrip. |
| [~] | P1 | Broaden WCS performance baselines across projection, distortion, inverse solve, batch size, device, spectral/time, and TAB regimes. | `bench-wcs`, `bench-wcs-spectral-temporal`, `bench-wcs-tabular`, and targeted benchmark JSON/markdown are updated with regression thresholds where stable. |
| [ ] | P2 | Expand celestial frame conversions beyond ICRS/Galactic/FK5 mean precession: FK4/Besselian E-terms, frame bias, nutation, aberration, proper motion, parallax, radial velocity, and observer-dependent apparent place. | Astropy/Kapteyn/AST parity fixtures with explicit accuracy budgets. |
| [ ] | P2 | Add sexagesimal/unit-aware formatting, plot-grid helpers, and coordinate display breadth only after core transform parity is stronger. | Formatting tests and docs; no impact on transform hot paths. |

### Sphere, HEALPix, Sparse Map, And MOC Tracker

Sphere work supports full-sky products, cross-survey tiling, sparse coverage, and
map/field bridges. Compatibility layers should sit on top of torchsky-native
kernels, not replace them.

| Status | Priority | Task | Acceptance gate / update trigger |
| --- | --- | --- | --- |
| [x] | P0 | Keep core HEALPix primitives, MOC range membership, sparse map construction/update/FITS roundtrip, ordering conversion, and general-region pixelization tests active. | `tests/test_sphere_*.py`, `tests/test_healpix_*.py`, HealSparse/MOC parity tests. |
| [x] | P0 | Move HEALPix implementation ownership from `torchsky.wcs.healpix` toward `torchsky.sphere.healpix`, leaving compatibility shims where needed. | Source implementation lives in `torchsky.sphere.healpix`; imports and public discovery route through `torchsky.sphere`; stale `torchsky.wcs.healpix` source references are gone outside archived review docs. |
| [x] | P0 | Expand exact geometry and MOC coverage for polygons, boolean/native regions, high-NSIDE indexed catalogs, sparse maps, and NUNIQ FITS MOC exchange. | MOC, sparse, region, FITS MOC, high-NSIDE catalog, and no-dense sparse workflow gates are active in `tests/test_sphere_moc.py`, `tests/test_sphere_sparse.py`, and `tests/test_sphere_high_nside.py`. |
| [x] | P0 | Harden HealSparse-style metadata/header fidelity: coverage maps, ordering, sentinel/null behavior, dtype preservation, partial maps, and `ud_grade(...)`. | HealSparse upstream replay tests and sparse FITS roundtrip tests pass; sparse construction/update/FITS/`ud_grade(...)` preserve coverage hierarchy and ordering contracts. |
| [x] | P0 | Add formal high-NSIDE query gates for coverage predicates, interpolation, neighbors, sparse lookup, and catalog filtering. | `tests/test_sphere_high_nside.py` covers correctness at `nside=1024`; `sphere-bench-healpix-*` tasks cover high-NSIDE throughput for HEALPix/interpolation primitives. |
| [ ] | P1 | Add spin/SHT replacement gates and throughput baselines for scalar and spin transforms. | `bench_spin_matrix.py`, `bench_sphere_spectral.py`, and parity tests have stable baselines. |
| [ ] | P1 | Expand catalog crossmatch and spherical indexing gates for survey-scale table + sphere pipelines. | `bench_catalog_crossmatch.py`, `bench_catalog_compat.py`, and `bench_pipeline_table_sphere.py` have maintained baselines. |
| [x] | P1 | Define sparse-map algebra and region interoperability contracts across dense maps, sparse maps, MOC, pixelized regions, and exact regions. | Sparse `sum_union`/`sum_intersection`, MOC/sparse/pixelized-region/exact-region bridges, and sparse `SkyMaskPipe` AND/OR gates are active without dense full-sky materialization. |
| [x] | P2 | Rename or alias harmonic-specific public concepts so `sphere.spectral` is not confused with WCS spectral coordinates. | `torchsky.sphere.harmonics` is the preferred alias; `torchsky.sphere.spectral` remains import-compatible and tests assert matching exports. |
| [ ] | P2 | Add device-specific sphere gates for CPU/CUDA/MPS where kernels claim support. | Device matrix tests/benches skip cleanly when devices are unavailable. |

### Shared Validation Gates

- Pre-push remains `pixi run ci-local`.
- FITS-focused changes should run the relevant FITS/table tests plus
  `pixi run bench-fits` or `pixi run bench-fitstable` when performance or scale
  behavior changes.
- WCS/sphere replacement changes should run `pixi run wcs-sphere-acceptance`
  before making replacement or performance claims.
- Comparator packages remain optional extras/environments. They are allowed in
  tests, benchmarks, docs, and scripts, but not as default runtime imports.

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
   - Extend the current residual/SNR/sharpness metrics into persisted residual products, support-aware scorecards, and per-band leakage diagnostics.
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
