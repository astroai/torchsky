# Code-Grounded Roadmap Status

This file tracks roadmap status from live code and tests. Treat narrative docs as secondary when they disagree with implementation.

Last updated: 2026-04-30.

## Current Baseline

- `pixi run ci-local` passes.
- Latest local gate result: `1501 passed, 61 skipped, 1 xfailed`.
- Focused FITS gate result: `78 passed, 1 skipped`.
- Focused FITS closeout gate result: `30 passed`.
- Focused WCS API gate result: `18 passed`.
- Focused WCS broad replay result: `262 passed, 2 skipped`.
- Focused WCS spectral translation result: `16 passed`.
- Focused WCS temporal scale result: `10 passed`.
- Focused WCS celestial frame result: `4 passed`.
- Focused WCS alignment correction result: `5 passed`.
- Focused WCS tabular lookup result: `13 passed`.
- Focused WCS API + perf gate result: `23 passed`.
- Public API surfaces are now grouped into `core`, `compat`, and `experimental` tiers in `torchsky.surfaces`.

## Completed In Current Slice

- API stability tiers:
  - Added `torchsky.SurfaceTier`, `torchsky.APISurface`, `torchsky.API_SURFACES`, `torchsky.get_api_surface`, and `torchsky.api_surfaces_by_tier`.
  - Marked FITS/WCS/native map/sphere primitives as `core`, compatibility adapters as `compat`, and field/visualization/training-facing surfaces as `experimental`.
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
  - Added dimensionality-preserving WCS slicing for positive-step array cutouts/views, updating `CRPIX` and `CD` so cropped and strided image arrays retain correct coordinates.
  - Added high-impact cube/image axis reduction helpers: `dropaxis(axis)` for uncoupled FITS-order axes and `celestial` for extracting a 2D sky WCS from spatial cubes, with explicit fail-closed behavior for coupled axes.
  - Added a Kapteyn-style `mixed_pixel_world(...)` solver for mixed pixel/world coordinate inputs, including the common 2D celestial case with one known pixel axis and one known world axis.
  - Added pure-Torch differentiable WCS entry points for INR/field-model use: `pixel_to_world_differentiable`, `world_to_pixel_differentiable`, `pixel_to_world_jacobian`, and `world_to_pixel_jacobian`.
  - Added torch-native spectral translations for frequency, vacuum wavelength, energy, air wavelength, redshift, radio velocity, optical velocity, relativistic velocity, and rest-frequency/rest-wavelength handling, with Astropy unit-equivalency parity tests where comparable.
  - Added wavenumber (`WAVN`) spectral conversion support with `cm-1`/inverse-meter unit handling and Astropy spectral-equivalency parity coverage.
  - Added validated one-axis FITS spectral `-LOG` WCS algorithm handling for `FREQ-LOG`, `WAVE-LOG`, and `WAVN-LOG`, with Astropy WCS parity coverage for forward and inverse transforms.
  - Added leap-second-aware temporal scale conversions for UTC, TAI, TT, and GPS, with Astropy Time parity tests and explicit fail-closed behavior for ephemeris-dependent barycentric corrections.
  - Expanded temporal unit handling to common FITS/VO sub-day units (`ns`, `us`, `ms`, `s`, `min`, `h`) plus days/years/centuries, and added temporal reference metadata surfaces for `TREFPOS`, `TREFDIR`, `PLEPHEM`, and `OBSGEO-*`.
  - Added first torch-native celestial frame utilities: ICRS/Galactic rotation, FK5 Julian-equinox mean precession, WCS `RADESYS`/`RADECSYS`/`EQUINOX` metadata parsing, and Astropy parity tests at classical-precession precision.
  - Added tweakwcs-inspired weighted 2D affine correction fitting for `shift`, `rshift`, `rscale`, and `general` geometries with residuals, RMS, inlier mask, and sigma-clipping diagnostics.
  - Added differentiable gnomonic tangent-plane frames and sky-coordinate affine fitting so catalog corrections can be solved in a local plane and applied back to RA/Dec.
  - Added a torch-native tabular lookup substrate for FITS `-TAB`-style axes: piecewise-linear `TabularAxis`, monotonic inverse lookup, separable `TabularLookupWCS`, bounds modes, and autograd through local interpolation slopes.
  - Wired independent `CTYPEi` `-TAB` axes into `WCS(..., tabular_axes={axis_number: axis})` for forward, inverse, spatial-cube extra-axis, and differentiable linear-path use.
  - Added `tabular_axes_from_fits(header, tables)` for the independent 1D FITS `-TAB` metadata subset: `PSi_0` table selection, `PSi_1` coordinate-array columns, optional `PSi_2` indexing-vector columns, and explicit rejection of coupled multidimensional coordinate arrays.
  - Added CI-friendly WCS performance smoke gates for common bulk TAN `pixel_to_world` and `world_to_pixel` paths, with `TORCHSKY_PERF_GATE_WCS_*` overrides.
- Sphere and tiling:
  - Added sparse-map operand validation plus union/intersection sum helpers.
  - Re-enabled the HealSparse upstream `update_values_pix` parity case.
  - Re-enabled the HealSparse sparse FITS write/read partial-map roundtrip gate.
- Map:
  - Re-enabled ground-truth map and multi-field truth tests.
- INR and field models:
  - Added a recommended field-model set so experimental model growth has a narrower endorsed surface.

## WCS Replacement Status

The WCS roadmap must distinguish active gates from remaining replacement claims.

### WCS Done And Active

- Core 2D FITS-WCS celestial transforms for the currently supported projection set: `TAN`, `SIN`, `ARC`, `ZPN`, `ZEA`, `STG`, `CEA`, `CAR`, `MER`, `AIT`, `MOL`, `HPX`, `SFL`.
- Distortion families with active tests: `SIP`, `TPV`, `TNX`, `ZPX`, plus CYP/PV projection parameters, slant `SIN`, and ZPN hole-domain behavior.
- High-dimensional linear extra axes for basic spatial + spectral cube workflows.
- Alternate FITS-WCS version keys (`A` .. `Z`) for standard suffixed WCS keywords.
- Metadata and geometry surfaces needed by downstream map/INR code: axis names, units, physical-type hints, axis-correlation matrix, pixel bounds, `footprint`, and `in_image`.
- Broad Astropy-style image interoperability helpers: `all_pix2world` / `all_world2pix`, value aliases, NumPy-order array index conversion, dimensionality-preserving WCS slicing for positive-step cutouts/views, and uncoupled axis reduction / celestial WCS extraction for cubes.
- Mixed pixel/world solving for full-pixel, full-world, and the common 2D case with one known pixel axis plus one known world axis.
- Differentiable pure-Torch WCS paths and local Jacobian extraction for optimization, INR renderers, field-model training, and multi-modal geometry encoders.
- Spectral translation primitives through frequency for `FREQ`, `WAVE`, `WAVN`, `AWAV`, `ENER`, `ZOPT`, `VRAD`, `VOPT`, `VELO`, and `BETA`, including `RESTFRQ`/`RESTFREQ`/`RESTWAV` handling and WCS convenience methods for spectral axes.
- First FITS spectral algorithm-code transform: validated one-axis `-LOG` support for `FREQ-LOG`, `WAVE-LOG`, and `WAVN-LOG`.
- Temporal scale conversion primitives for UTC/TAI/TT/GPS with a built-in IERS/NIST leap-second table current through the 2026-06-30 no-leap Bulletin C 71 window; common sub-day/year units and reference metadata are parsed; barycentric/heliocentric corrections explicitly require a future ephemeris backend.
- Celestial frame primitives for ICRS/Galactic conversion and FK5 Julian-equinox mean precession, plus WCS helpers for `RADESYS`/`EQUINOX` metadata.
- First catalog-alignment correction primitives: weighted 2D affine fitting with shift / rotation+shift / rotation+scale+shift / full affine modes, plus differentiable gnomonic tangent-plane fitting for sky catalogs.
- First table-driven WCS support: independent one-dimensional lookup axes with differentiable interpolation, monotonic inverse lookup, explicit `WCS` integration for FITS 1-based `CTYPEi` `-TAB` axes, and a parser helper for the independent 1D `PSi_0`/`PSi_1`/`PSi_2` metadata subset.

### WCS Partially Done

- Spectral/time axes: spectral coordinate translation primitives, `-LOG` spectral WCS, and temporal scale/unit/reference metadata primitives now exist, but full FITS Paper III spectral algorithm-code WCS (`CTYPE` algorithm chains such as `F2W`, `W2F`, velocity grisms, etc.) and full Paper IV space-time corrections are not replacement-complete.
- Kapteyn-style mixed coordinates: the common 2D celestial mixed solve is active; arbitrary N-dimensional mixtures and non-celestial coupled axes remain open.
- Axis subsetting: uncoupled axis drops and celestial extraction from spatial cubes are active; arbitrary reordering, integer-index slicing, and coupled-axis subsetting are not implemented.
- GWCS/AST-style composition: metadata surfaces exist, but explicit frame/mapping pipeline objects and reusable transform graph nodes are not implemented.
- FITS `-TAB` / lookup-table WCS: independent lookup-axis interpolation is active in the main `WCS` transform when axes are supplied explicitly, and the independent 1D `PSi_0`/`PSi_1`/`PSi_2` parser helper exists; direct HDUList auto-discovery, `PVi_ma` version/level handling, multidimensional coordinate arrays, and coupled table-driven distortions are not yet implemented.
- Performance: common CPU/vectorized/C++/compiled paths exist and a CI-friendly TAN forward/inverse smoke gate is active, but replacement claims still need broader timing baselines across projection, distortion, batch-size, device, and inverse-solve regimes.
- Compatibility replay: Astropy/WCSLIB fixture coverage is active, but Kapteyn/PyAST/Cornish bridge replays are not yet formal gates.

### WCS Not Done

- Full celestial frame-system conversions beyond the first FK5/ICRS/Galactic slice: FK4/Besselian E-terms, high-precision frame bias, nutation, aberration, proper motion/parallax/radial velocity propagation, and observer-dependent apparent-place behavior comparable to Kapteyn/AST.
- Full FITS Paper III spectral WCS algorithm-code parsing beyond validated one-axis `-LOG`: compound `F2W`/`W2F`, velocity/grism algorithms, and coupled nonlinear spectral pixel-axis transforms.
- Full temporal WCS reference-position corrections: Earth orientation, barycentric/heliocentric light-time corrections, and coupled space-time corrections.
- Generalized AST/GWCS frame graphs, serialization, and conversion to/from approximating FITS SIP/TAB representations.
- Full FITS `-TAB` HDUList auto-discovery, version/level disambiguation, multidimensional coordinate arrays, and generalized coupled table-driven distortions.
- Full tweakwcs-class correction workflows: iterative matching, correction metadata serialization, and direct WCS header updates.
- Plot-grid/formatting helpers and sexagesimal/unit-aware coordinate formatting at Kapteyn/AST breadth.

## Near-Term Next Work

1. FITS completeness:
   - FITS is closed for the current roadmap slice.
   - Future parity hardening should focus on broader astropy.io.fits/fitsio fixture replay, large-table mutation benchmarks, and additional realistic VLA/table fixtures.
2. WCS completeness:
   - Current active gates cover supported projections/distortions, slant `SIN`, ZPN hole fixtures, alternate WCS keys, axis metadata, footprint, mixed pixel/world transforms, and differentiable Jacobians.
   - Current spectral/temporal slice covers `WAVN`, `-LOG`, common time units, and temporal reference metadata. Next spectral step is compound FITS Paper III algorithm-code parsing (`F2W`, `W2F`, velocity/grism algorithms) in actual WCS pixel/world transforms; next temporal step is ephemeris-backed reference-position corrections and realistic observatory fixtures.
   - Add a formal Kapteyn/PyAST/Cornish legacy-bridge replay matrix for sky-system conversions, epochs/equinoxes, mixed coordinates, alternate headers, spectral translations, and formatting/plot grid behavior where comparable.
   - Adopt the GWCS/AST design pattern of explicit transform pipelines and frame/mapping composition for future non-FITS-native WCS, but keep runtime implementation torch-native.
   - Build on the new tangent-plane affine correction primitives with iterative catalog matching, correction metadata serialization, and direct WCS header update helpers.
   - Build direct HDUList/path auto-discovery and `PVi_ma` version/level disambiguation on top of `tabular_axes_from_fits(...)`, then add coupled multidimensional table-coordinate semantics.
   - Harden high-dimensional spatial/spectral/time axes against Astropy/WCSLIB fixtures.
   - Add more fixture replay gates for mixed projection plus distortion headers.
3. Sphere and tiling:
   - Replace current exact-geometry xfail with robust MOC/geometry coverage.
   - Expand HealSparse metadata/header fidelity beyond the current sparse FITS roundtrip.
   - Add high-NSIDE query and spin/SHT replacement gates.
4. Map architecture:
   - Make `ObservationOperatorBundle` the common execution substrate across promoted map paths.
   - Move remaining planning/windowing/I/O policy out of `map/image_plane.py`.
   - Add exact adjoint gates for promoted kernels.
5. INR and fields:
   - Keep new architectures behind explicit experimental factories.
   - Prioritize identifiability, PSF/noise/nuisance models, photometric metrics, residual maps, support masks, leakage metrics, and timing baselines.

## Standing Gates

- Pre-push: `pixi run ci-local`.
- Focused replacement gates should stay domain-specific:
  - FITS vs `astropy.io.fits` and Python `fitsio`.
  - WCS vs Astropy/WCSLIB fixture behavior.
  - Sphere vs healpy, hpgeom, healsparse, astropy-healpix, and spherical-geometry.
  - Map vs reproject, drizzle, SWarp, and Montage where installed.
