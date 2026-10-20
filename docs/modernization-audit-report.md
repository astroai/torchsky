# TorchSky modernization and correctness audit report

Date: 2026-10-20
Scope: native C++, WCS, sphere/HEALPix/SHT, map operators, and field modelling
Method: static audit, direct native contract tests, independent astronomy oracles, falsifying numerical probes, concurrency/subprocess stress, gradient/adjoint tests, packaging inspection, and integrated regression suites

## Executive summary

This program began as an audit of `src/torchsky/cpp_src` and expanded through the Python WCS, sphere, map, and field layers. It found failures in four broad categories:

1. **Safety and concurrency:** unchecked raw buffers, an empty-input segfault, non-reentrant scheduling, and cache lifetime races.
2. **Scientific correctness:** invalid WCS domains returning finite coordinates, incomplete generalized HPX equations, unstable Mollweide and TPV inversion, scale-dependent spherical geometry, incorrect variance propagation, and biased likelihoods.
3. **Tensor/API contracts:** failures on scalar, empty, noncontiguous, multidimensional, gradient-tracked, device-specific, and high-order inputs.
4. **Architecture and portability:** one monolithic C++ translation unit, direct DLPack/nanobind tensor calls, unconditional AVX2/FMA compilation, CUDA-sensitive CPU builds, incomplete artifact licensing, and missing dispatcher/Meta/autograd contracts.

The work repaired the verified defects, added independent regression coverage, split native translation units, migrated production tensor kernels to PyTorch dispatcher operators, introduced FakeTensor/compile/autograd support, isolated the remaining compatibility bridge, added runtime ISA selection, and hardened wheel/sdist checks.

## Verification summary

Final integrated results:

- Native, WCS, sphere, HEALPix, dispatcher, bridge, and package tests: **1072 passed, 31 skipped**.
- Map and field tests: **1024 passed, 23 skipped**.
- Performance smoke: **4 passed**.
- `pixi run preflight-push`: passed.
- `pixi run typecheck`: **0 errors**, one pre-existing `__all__` warning.
- Wheel and sdist: content/linkage inspection passed; `twine check` passed; clean installed-wheel dispatcher smoke passed; `TORCHSKY_DISABLE_CPP=1` smoke passed.

The full `pixi run ci-local` lane was not run automatically because repository policy reserves it for explicit human opt-in. The targeted coverage above includes all changed subsystems, including map tests that are excluded from the default `pixi run test` task.

## Native C++ defects fixed

### Unsafe fused WCS buffers

The fused nanobind entry points accepted unchecked device, dtype, contiguity, shapes, caller-owned output capacity, overlap, projection names, scalar parameters, and TPV metadata before raw `double*` access.

Fixed by:

- validating CPU/float64/contiguous tensors and identical coordinate/output shapes;
- checking output overlap and capacity;
- validating finite scalar parameters and supported projection/native types;
- validating TPV index shape, dtype, lengths, coefficients, and exponent bounds;
- handling omitted TPV tensors when TPV is disabled;
- adding subprocess tests proving malformed direct calls raise rather than crash.

### Empty ring synthesis segfault and raw spectral contracts

A zero-ring synthesis read `starts_ptr[-1]`. Other exported recurrence/Fourier kernels accepted undersized ALMs, invalid bandlimits, inconsistent rings, and overflowing layouts.

Fixed by:

- explicit empty-layout behavior;
- checked ALM-size arithmetic;
- validation of `lmax`, `mmax`, spin, rank, dtype, device, phases, weights, ring starts/lengths, overlap, map extents, and integer overflow;
- direct malformed-binding and subprocess crash-safety tests.

### Non-reentrant `SpinMPool`

A process-global scheduler allowed concurrent submissions to overwrite one shared job and completion counter. Worker exceptions could terminate the process.

Fixed by:

- serializing complete submissions;
- capturing and rethrowing worker exceptions on the caller;
- clearing job state on every exit;
- safe nested-call fallback;
- bounded environment thread overrides;
- honoring PyTorch thread limits on all platforms;
- concurrent mixed-shape and timeout tests.

The existing scheduler was contained rather than replaced without benchmark evidence.

### Cache lifetime races

Several native caches returned references or raw pointers into storage that could be replaced, resized, cleared, or evicted after unlocking.

Fixed by:

- immutable `shared_ptr<const Entry>` snapshots;
- complete parameter keys;
- build-before-publish semantics;
- eviction that removes cache ownership without invalidating active callers;
- immutable published factor tables;
- concurrent mixed-key, eviction, and determinism tests.

### DLPack/autograd routing failures

`Tensor.__dlpack__()` rejects gradient-tracked tensors. TPV inverse, coordinate interpolation, and geometry predicates could select native paths and expose `BufferError`.

Fixed by:

- routing differentiable TPV/interpolation inputs through tensor-native paths;
- detaching only explicitly nondifferentiable Boolean predicates;
- replacing the old identity `echo_tensor` test with a real conversion test while the bridge remained;
- later migrating production tensor exchange to dispatcher operators.

### Scale-dependent spherical arc intersections

Unnormalized great-circle normals were compared with an absolute threshold, rejecting valid milliarcsecond-scale crossings. Coincident overlapping/touching arcs were skipped.

Fixed by:

- normalized great-circle normals;
- stable oriented/`atan2` arc membership;
- explicit zero-length, antipodal, transverse, endpoint, coincident-overlap, touch, and disjoint semantics;
- tests down to `2e-8` radians.

### Point-in-polygon epsilon changed geometry

The implementation replaced small edge denominators with epsilon, moving intersections and changing classifications.

Fixed by:

- skipping noncrossing/horizontal edges before division;
- a half-open crossing rule;
- separate boundary semantics;
- Boolean parity rather than an overflowable crossing count;
- matched Python/native behavior.

### Invalid HEALPix coordinates and queries

NaN/Inf angles, latitude outside `[-90°,90°]`, invalid query radii/vectors, and NaN cosine thresholds could produce plausible pixels or whole-sphere results.

Fixed by:

- finite/domain validation before trigonometry or integer conversion;
- exact pole and transition handling;
- radius `[0,pi]`, cosine `[-1,1]`, and finite nonzero vector contracts;
- brute-force bounded-NSIDE query parity tests.

### High-NSIDE overflow and `O(nside)` sparse setup

Native NSIDE validation lacked the order-29 ceiling, arithmetic was unchecked, and one-pixel conversions allocated NSIDE-sized Morton tables.

Fixed by:

- native `nside <= 2**29` enforcement;
- checked pixel/allocation arithmetic;
- direct constant-memory 64-bit Morton interleave/deinterleave;
- order-28/29 sparse tests and order-30 rejection.

### Direct scalar SHT overflow

Unnormalized associated-Legendre growth overflowed at moderate bandlimit; `l=m=256` produced NaN.

Fixed by:

- normalized recurrence;
- principled binary `frexp`/`ldexp` rescaling;
- fail-closed finite checks;
- SciPy/stable-recurrence parity, basis, random-ALM, equator/pole, linearity, and conjugation tests.

## Native architecture and build modernization

### Translation units

The extension previously textually included `geom.cpp`, `wcs.cpp`, and `healpix.cpp` into `bindings.cpp`.

Fixed by:

- narrow internal headers;
- four independently compiled implementation/module translation units;
- removal of `.cpp` includes and `OBJECT_DEPENDS` workarounds;
- target-scoped options, includes, definitions, and IPO behavior;
- preservation of all historical compatibility symbols.

### PyTorch dispatcher migration

Production tensor kernels now use registered `torchsky::` operators:

- functional WCS pixel/world transforms;
- 15 fixed-shape geometry, HEALPix, interpolation, and sparse operators;
- 14 spectral/SHT operators.

Added:

- CPU implementations;
- functional fresh-output WCS APIs;
- FakeTensor/Meta registrations;
- `torch.library.opcheck` coverage;
- `torch.compile(fullgraph=True)` coverage;
- explicit autograd and higher-order gradient behavior where mathematically valid;
- explicit nondifferentiability for discrete outputs;
- legacy `_cpp` compatibility adapters with deprecation behavior.

Deferred intentionally:

- `query_disc`, because output length depends on tensor values;
- two experimental basis kernels that remain profiling-only.

### Native bridge isolation

The extension previously exposed dozens of direct tensor-taking nanobind functions using DLPack and private `THPVariable_Wrap`.

Fixed by:

- migrating production call sites to `torch.ops`;
- reducing `_cpp_ext` tensor bindings to three deferred surfaces;
- routing 31 historical names through dispatcher compatibility adapters;
- enforcing that production code does not add new direct `_cpp` calls.

`torch_python` remains linked only because the three deferred bindings still use the isolated bridge.

### Portable ISA dispatch

The whole extension previously compiled with `-mavx2 -mfma` whenever the build host accepted those flags.

Fixed by:

- portable scalar baseline;
- isolated AVX2+FMA and arm64 NEON Ishioka leaves;
- once-per-process runtime ISA selection;
- safe force/diagnostic controls;
- LTO isolation for ISA leaves;
- parity tests for forced scalar/native paths;
- IPO-on and IPO-off build coverage.

Local verification ran on macOS arm64/NEON. AVX2 and Linux execution remain CI-covered rather than locally executed.

### CMake and distribution portability

Fixed by:

- imported Torch target preference with tested fallback;
- deliberate Torch ABI flags;
- LTO-safe registration objects;
- IPO-off diagnostic mode;
- complete native source/header inclusion in sdist;
- current pocketfft notices and provenance;
- wheel/sdist content, linkage, path-remapping, and installed-wheel checks;
- SPDX/license metadata and `twine check`.

A CUDA-enabled PyTorch build still requires the toolkit required by its upstream TorchConfig; the CPU-only local build does not add CUDA sources.

## WCS defects fixed

### Projection domains and conventions

Fixed:

- TAN horizon/back-hemisphere points now fail rather than being clamped into huge finite coordinates.
- ARC rejects radius beyond 180° outside roundoff.
- CEA validates lambda and inverse-sine domains before clamping.
- MER rejects invalid latitude rather than mapping both poles to finite extreme pixels.
- Zenithal handedness is consistent across standalone, fused, and Python paths.
- AIR no longer maps the native antipode to the projection origin.

### Mollweide stability and domain

Fixed by:

- bracketed monotone solving;
- analytic pole handling;
- residual and interval stopping;
- full inverse ellipse validation including the X bound;
- shared behavior across standalone, center, fused, and tensor paths.

### Generalized HPX

The implementation previously changed one scale while retaining default `H=4,K=3` facet geometry.

Fixed by centralizing validated H/K-dependent:

- facet count and spacing;
- centers and widths;
- K-parity staggering;
- equatorial transition/scale;
- pole extent and domain masks.

Parity covers default and non-default parameter pairs, seams, transitions, poles, invalid facets, and fused/Python paths.

### TPV inverse correctness

Fixed:

- metadata validation and exponent bounds;
- empty TPV identity consistency;
- finite coefficient/input/control validation;
- relative Jacobian conditioning instead of determinant flooring;
- per-point failure state;
- final forward residual enforcement;
- NaN results for singular, folded, or unconverged points;
- matched Python/native/fused semantics and inverse gradcheck.

### Celestial units

Legal non-degree celestial `CUNIT` headers were parsed but calculations assumed degrees.

Fixed by:

- normalizing celestial CRVAL/CDELT/CD to internal degrees;
- preserving declared units/numerics during serialization;
- radian/arcminute/arcsecond parity coverage.

### WCS state preservation

Slicing and affine cloning silently discarded SIP, TPV, lookup tables, and stored affine state.

Fixed by:

- exact preservation for identity operations;
- explicit fail-closed `NotImplementedError` for unsupported nonidentity SIP/CPDIS/DET2IM/LUT/existing-affine rebasing or composition;
- invariants for distortion state.

### Observed azimuth

The IAU-2006 vector path negated the east component and mirrored azimuth.

Fixed against absolute ERFA AltAz fixtures across sites/hour angles/polar motion.

### Temporal `TIMEOFFS` and ISO shapes

Fixed by:

- applying `TIMEOFFS` symmetrically in forward/inverse time conversion;
- scalar and N-D ISO handling;
- round-trip and scale tests.

### Tensor shape and NaN contracts

Fixed:

- noncontiguous N-D inverse now uses safe reshape;
- separate coordinate arguments broadcast where compatible;
- separable axes avoid unnecessary `0*NaN` contamination;
- invalid MER values return paired NaNs consistently.

### Detector astrometry and interoperability

Fixed:

- 2x2 residual grids no longer produce NaN bending energy;
- inverse tolerance is defined in output units;
- Astropy `obstime`/epoch is preserved in both directions;
- optional dependencies remain opt-in.

### Tabular inverse memory

The inverse previously built full query-by-grid distances before a specialized path that ignored them.

Fixed by:

- branching before unused seed construction;
- exact chunked nearest lookup;
- bounded pairwise memory;
- scaling, parity, round-trip, empty, noncontiguous, dtype, and device tests.

## Sphere, HEALPix, MOC, sparse, and SHT defects fixed

### Sampling and tensor shapes

Fixed:

- scalar HEALPix sampling no longer calls `reshape()` without a shape;
- integer/bool bilinear sampling uses a floating work dtype instead of truncating fractional weights;
- multimap `reorder` acts on the final pixel axis and preserves leading dimensions;
- `vec2ang` preserves leading shapes;
- vector query APIs validate their scalar/vector contract;
- zero vectors are rejected by lon/lat conversion.

### Angular-distance gradients

Pairwise distance evaluated singular diagonal square roots before replacing diagonal values, producing NaN gradients.

Fixed with stable diagonal/near-coincident treatment and gradient checks.

### Weighted iterative SHT

Weights were applied once to the input map but not to synthesis residuals in later iterations.

Fixed by forming residuals in map space and applying masks/weights within every analysis pass. Operator-equivalence tests cover ring/pixel weights and Galactic cuts.

### Spin transform and covariance contracts

Fixed:

- explicit `nside` must match spin-map length before native routing;
- materially non-PSD covariance spectra are rejected instead of silently projected to zero;
- only scale-relative roundoff negatives are clamped;
- spin output dtype/device construction is consistent, including MPS coverage.

### Sparse map invariants

Fixed:

- NSIDE and pixel bounds;
- exact valid-mask shape;
- value rank and column schema;
- single named-column access for one-dimensional values;
- arbitrary-count dtype folding;
- query-shape preservation;
- consistent multidimensional FITS output or fail-closed behavior.

### MOC/BMOC hierarchy

Fixed:

- `nest2uniq` pixel bounds;
- canonical UNIQ validation;
- rejection of aliased invalid cells;
- hierarchical parent/child normalization;
- conservative deterministic flag combination;
- area/to-MOC consistency;
- empty containment broadcasting.

### Spherical region semantics

The class named `NativeExactSphericalRegion` used center-selected finite-resolution HEALPix coverage and could report zero area for a non-empty tiny cap.

Fixed by:

- introducing `NativeApproximateSphericalRegion`;
- retaining the old name as a compatibility alias;
- analytic cap/polygon/box area and supported intersections;
- fail-closed unsupported exact operations;
- exposing center sampling only as `area_estimate()`;
- domain validation for pixelized regions, boxes, ellipses, and region operands.

## Map defects fixed

### FFT affine normalization and Nyquist handling

Zero-padding an RFFT spectrum changed inverse normalization, causing constant 1-D values to shrink by the upsample factor and 2-D values by its square.

Fixed by:

- correct Fourier interpolation normalization;
- separable even-length Nyquist handling;
- positive/negative scale coverage;
- dtype/device/gradient preservation;
- DC, impulse, sinusoid, odd/even, and variance tests.

### Double photometric calibration

HEALPix and IFU cube paths applied calibration a second time after canonical observation calibration.

Fixed by applying calibration exactly once and adding non-unit-scale tests.

### IFU cube inverse variance

The cube returned linear normalization as inverse variance and ignored squared interpolation coefficients.

Fixed by:

- separate variance accumulation using squared coefficients;
- output diagonal IVM `denominator² / variance_numerator` for declared inverse-variance inputs;
- returning `None` for generic weights;
- covariance diagnostics for sample sharing between voxels.

### Pixel solid angle

Undistorted TAN was incorrectly treated as constant area.

Fixed by:

- exact TAN Jacobian behavior;
- spherical distorted-pixel areas;
- comprehensive distortion detection;
- center/corner and wide-field tests.

### Conservative overlap caps

Small targets received radius zero and larger targets used non-conservative subsampling/tangent distance.

Fixed by:

- boundary/corner inclusion for every target size;
- exact great-circle separation;
- no-false-rejection tests for distorted and off-center targets.

### Noise calibration and sanitization

Fixed:

- one-dimensional calibration vectors follow the documented leading axis even when their length equals image width;
- trailing maps require explicit shapes;
- inverse-variance sanitization zeros NaN, Inf, zero, and negative values while preserving finite positives.

### PSF kernels

Even kernels changed output shape and invalid sums could be normalized through a clamp.

Fixed by deterministic rejection of unsupported even kernels and finite/non-near-zero normalization checks, with signal/variance/adjoint/gradient coverage.

### HEALPix image-plane device behavior

The implementation forced float64 on MPS and failed before execution.

Fixed by:

- float32 work tensors on MPS;
- float64 work paths on CPU/CUDA where supported;
- device-local WCS state;
- MPS forward/adjoint device tests.

### Solver package refactor

A concurrent split of `map/solvers.py` initially left missing names/import failures.

Fixed by:

- verifying 47 definitions are AST-equivalent;
- exporting all 53 declared symbols exactly once;
- restoring callable aliases;
- eliminating circular/missing imports;
- adding API parity coverage.

## Field defects fixed

### Vmapped rendering

The advertised vmap path entered tensor-dependent Python validation and failed.

Fixed with a stateless batched fallback/prevalidated path and vmap parity tests.

### Sample shape contracts

Hard-coded broadcast shapes broke scalars, one-dimensional distributions, and multidimensional deterministic samples.

Fixed for scalar, empty, noncontiguous, one-dimensional, and arbitrary N-D coordinates.

### Float64 rendering precision

Rendering accumulated in float32 and cast back to float64, erasing small residuals.

Fixed by choosing work dtype from model/output/requested dtype, with an MPS float32 exception.

### Predictive variance integration

The renderer assumed independent quadrature nodes, causing variance to shrink merely when numerical quadrature resolution increased.

Fixed by:

- coherent predictive-variance semantics by default;
- an explicit independent-aleatoric mode;
- quadrature-resolution invariance tests.

### Gaussian likelihood

Predictive variance and observation precision were combined incorrectly, then inverse variance was applied again as a generic weight.

Fixed by:

- total variance = predictive variance + measurement variance;
- `0.5 * (residual² / total + log(total))`;
- separate generic sample weighting;
- analytic optimum, gradient, mask, reduction, dtype, and device tests.

### Field domain validation

Zero/NaN scales and invalid coordinates propagated NaNs into models.

Fixed with frozen `FieldDomain` validation for finite centers, positive finite scales, valid declination, wavelength, and coordinate mode.

### Quality statistics

Empty support reported perfect zero error, and aggregate RMSE was averaged linearly.

Fixed by:

- explicit NaN semantics for zero effective support;
- pooling primitive weighted sums/counts before nonlinear metrics;
- correct pooled RMSE and related statistics.

### Bootstrap memory

Bootstrap validation materialized `samples x N` index/draw matrices.

Fixed with bounded deterministic chunking and reference-equivalence tests.

### Lazy exports, builders, and move semantics

Fixed:

- pseudo-module lazy exports now resolve actual modules;
- auxiliary weight/background planes preserve image dtype;
- `.to()` returns do not mutate nested modules in the source observation;
- copy/move semantics are explicit and tested.

## Why existing tests and torchsky-harness missed these defects

### Tests exercised happy paths, not contracts

Many suites tested representative valid arrays but omitted:

- empty and scalar tensors;
- noncontiguous views;
- mismatched paired shapes;
- wrong device/dtype;
- malformed layout metadata;
- NaN/Inf and out-of-domain coordinates;
- near-singular projection boundaries;
- high NSIDE and high bandlimit;
- concurrent independent calls.

The new failures appeared only under those contracts.

### Native tests were not always forcing native code

The geometry availability probe imported a removed `torchsky.cpp` namespace, so native parity tests were permanently skipped. Other tests exercised public APIs without proving which backend ran.

The new tests call dispatcher/direct compatibility surfaces explicitly and include backend-disable/force controls.

### Some parity tests compared two implementations sharing the same defect

Native and Python formulas were often translations of each other. Agreement therefore proved consistency, not correctness. Examples include projection domains, point-in-polygon epsilon behavior, and HEALPix conventions.

The new tests use independent sources where possible:

- Astropy/WCSLIB for WCS domains and generalized projections;
- ERFA for observed-frame azimuth;
- analytic geometry and high-precision invariants;
- brute-force HEALPix center membership;
- SciPy/stable recurrence for spherical harmonics;
- hand-derived variance and likelihood counterexamples.

### Existing tests checked adjoints but not scientific normalization

An operator can have a correct transpose while both forward and adjoint share the same wrong scale. This is why FFT normalization, IFU inverse variance, photometric double-scaling, and predictive variance passed strong adjoint-style tests.

New coverage checks conservation, units, variance algebra, analytic optima, and independent expected values in addition to adjoints.

### Low-order tests concealed numerical instability

The direct scalar SHT was tested around `lmax=8`; overflow appeared at `l=m=256`. HEALPix tests used moderate NSIDE; the `O(nside)` setup became pathological near order 29. Projection tests avoided poles, horizons, seams, invalid facets, and singular distortions.

New tests sweep the failure boundaries and include high-order sparse cases that remain memory bounded.

### Concurrency was not part of the earlier contract

Single-thread tests could not expose the shared `SpinMPool` job overwrite or cache reference invalidation. The new suite runs concurrent mixed-parameter transforms in subprocesses with hard timeouts and deterministic baselines.

### Device matrices were incomplete

MPS float64 failures, CUDA/device-routing behavior, and CPU-only build portability were often skipped on a single development machine. The implementation now has explicit device policies and CI/source checks, but AVX2/Linux and optional accelerator execution still depend on their platform lanes.

### Default test tasks intentionally omit map tests

Repository policy defines `pixi run test` as the main suite with `tests/test_map_*.py` excluded; map tests run in a separate matrix. This is operationally reasonable, but it means running only the default task cannot detect map FFT, calibration, variance, PSF, windowing, or field rendering failures.

This work ran the map/field corpus explicitly: **1024 passed, 23 skipped**.

### Harness benchmarks measure speed, not semantic edge cases

`torchsky-harness` owns realistic performance campaigns and large-data scorecards. A benchmark can faithfully time an algorithm that is consistently misnormalized or only wrong on invalid/singular/high-order cases. The harness did not replace bounded unit/API/oracle tests in this repository.

The appropriate split is:

- torchsky: bounded correctness, contract, adjoint, gradient, device, race, and regression tests;
- torchsky-harness: realistic throughput, memory, crossover, comparator, and campaign evidence.

### Some tests encoded or accepted incorrect behavior

Examples included accepting singleton sample-shape defects, treating normalization as inverse variance, and expecting strict behavior from a sanitizer. These tests were replaced with scientific/API invariants rather than merely updated to match the implementation.

## Deferred and intentionally unsupported work

The following were not silently claimed complete:

- `query_disc` remains outside FakeTensor/fullgraph dispatcher support because output length is data-dependent.
- Two experimental SHT basis bindings remain legacy-only.
- Their isolated bridge still requires `torch_python`; all production migrated tensor paths do not.
- Exact nonidentity rebasing/composition for SIP/CPDIS/DET2IM/LUT/existing pixel-affine state remains unsupported and raises.
- Celestial-axis permutations beyond the currently supported axis arrangement remain a documented architectural limitation.
- AVX2/Linux runtime execution is CI-covered but was not locally executed on the macOS arm64 host.
- CUDA-enabled PyTorch still follows upstream TorchConfig toolkit requirements.
- Large-scale crossover/default changes to the contained scheduler or hybrid FFT policy require same-machine, cross-platform `torchsky-harness` evidence; no unmeasured default performance claim was made.
- Potential future scaling work remains for pathological polygon candidate expansion, full detector pixmap construction, incremental HEALPix coadd reduction, Python spectral-cache unification, and indexed STMOC operations.

## Result

The code now fails closed for malformed structural inputs, returns oracle-compatible invalid-domain results, preserves tensor/device/dtype/gradient contracts across substantially broader inputs, and exposes native kernels through PyTorch’s operator system rather than an opaque direct binding surface. The added tests focus on the exact classes of failure that the previous happy-path, shared-formula, low-order, single-thread, and partial-device coverage did not exercise.
