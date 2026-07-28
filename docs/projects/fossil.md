# FOSSIL/BISOU Mission-Neutral Torchsky Track

> Distilled through 2026-07-28 from the user-authored
> `fossil_primitives_plan.md` and `fossil_science.docx`. The untracked source
> documents remain untouched.
> This file is the executable Torchsky backlog; mission configuration, telemetry,
> sensitivity assumptions, and publication estimators remain outside core.

## Scientific Objective

Use the standard primordial `mu` distortion as the precision and
signal-preservation target, then extend to dark-sector and unknown-shape
distortions. The differentiator is a discovery-preserving measurement pipeline:
known transforms remain explicit, flexible models learn only held-out-predictable
nuisance structure, and stable unexplained residuals remain exportable science
products.

The ordered discovery programme is:

1. standard `mu` injections and an open full-sky residual atlas;
2. joint monopole/anisotropy residual tomography and external cross-correlations;
3. blinded dark-sector and unknown-shape challenge spectra;
4. a falsifiable test of the standard thermalization manifold.

The reusable implementation target is broader than one mission: the same
spatial-spectral contracts should support CMB component separation and spectral
distortions, 21 cm/line-intensity foreground inference, MOS/IFU sky and
telluric removal, and static-sky separation for moving objects. Cosmology is the
first validation profile because it makes weak-signal attenuation and
foreground leakage impossible to ignore.

## Common Differentiable Separation Spine

Parametric fitting, moment expansions, PCA/NMF, neural fields, and hybrid
models must plug into one spine rather than become separate pipelines:

1. explicit spatial, spectral, detector, and time axes;
2. a differentiable response/mixing operator with an exact adjoint;
3. declared target and nuisance subspaces, including weighted protection;
4. scientifically independent cross-fit partitions;
5. paired positive/negative signal injections;
6. an exportable open residual with attenuation and leakage diagnostics.

The learned model may be highly flexible; the measurement contract and
validation cannot be implicit. In particular, a low reconstruction loss is not
evidence that a `mu`, 21 cm, line, telluric, or moving-object signal survived.

## Ownership Boundary

Torchsky owns mission-neutral tensor semantics, decoded observation sequences,
measurement operators and adjoints, full-sky spectral products, scientific
partitions, injections, nuisance interfaces, open residuals, and validation.

A FOSSIL/BISOU extension owns channel tables, instrument/scan configurations,
telemetry decoding, collaboration simulations, sensitivity assumptions, CMB and
dark-sector template libraries, and publication-specific estimators. Torchsky
must not become a spacecraft packet decoder or a closed foreground ontology.

## Existing Substrate Audit

- [x] Torch-native WCS, HEALPix, sparse maps, harmonics, and image/full-sky
  mapmaking exist.
- [x] `ObservationOperatorBundle` already separates geometry, spectral response,
  PSF, radiometry, background, masks/weights, and noise.
- [x] Matrix-free multi-observation execution, exact-adjoint gates, spectral
  fields, learned bases, residual products, and bounded execution exist.
- [x] `SpectralAxis` exists in `torchsky.spectral`; extend it rather than creating
  a parallel axis type.
- [ ] Existing operator declarations do not yet provide one typed
  domain/codomain, forward/adjoint, chunk/device protocol across domains.
- [ ] No first-class decoded detector sequence, interferogram, full-sky spectral
  product, monopole contract, scientific cross-fit manifest, or open residual
  product exists.

## Phase A — Reusable Spectral And Full-Sky Substrate

- [x] Harden `SpectralAxis` with an explicit
  frequency/wavelength/energy/wavenumber quantity, unit conversion, bin edges,
  monotonic/duplicate checks, regularity diagnostics, interval selection,
  quadrature weights, device/dtype transfer, and a FITS-WCS adapter that leaves
  FITS I/O in Torchfits.
  Evidence: the existing axis now supports ascending/descending irregular grids,
  explicit or midpoint-derived edges, positive widths, interval-preserving
  selection, transfer, and differentiable conversion among the four physical
  quantities. The existing FITS-WCS header adapter remains authoritative.
  Thirty-six focused spectral tests pass.
- [ ] Add the smallest immutable `TensorSpec`/axis-spec contract needed to reject
  shape-compatible but physically incompatible tensors. Include named axes,
  value semantics, unit, and coordinate frame; do not build a new units package.
- [ ] Promote spectral transmission to a reusable `SpectralResponse` supporting
  tabulated/analytic curves, compact response uncertainty, detector batches,
  line-spread/channel mixing, photon/energy conventions, and analytic-spectrum
  integration gates.
- [ ] Add explicit radiometric value semantics and conversions for
  `I_nu <-> I_lambda`, flux/surface brightness through pixel solid angle,
  brightness temperature, colour correction, and calibration Jacobians.
- [ ] Define a typed linear-operator protocol beneath the existing semantic
  bundle: domain/codomain specs, `forward`, `adjoint`, optional normal/JVP/VJP,
  device transfer, and chunk iteration.
- [ ] Add only the operator combinators demanded by real consumers:
  composition, sum/scale, stack, block diagonal, difference, mask, low rank,
  identity, and deterministic reduction.
- [ ] Consolidate adjoint, linearity, eager/chunk, replay, and device checks into
  one reusable operator gate without duplicating current map tests.
- [ ] Add dense and sparse `FullSkySpectralMap` products with spectral batching,
  RING/NEST conversion, explicit resolution/value semantics, per-channel
  mask/weight/coverage, beam metadata, splits/coadds, and harmonic batching.
- [ ] Make monopole, dipole, anisotropy, masked-sky estimate, and absolute
  zero-level semantics explicit. Every full-sky operation states what it
  preserves, removes, or estimates.
- [ ] Add detector/time/scan/region/instrument/calibration `CrossFitPartition`
  manifests and paired positive/negative `Injection` records with deterministic
  provenance.

### Phase A exit gate

An irregular absolute spectral product can round-trip coordinates and
radiometry, execute through typed/chunked operators with exact adjoints, form
scientifically independent splits, and replay signal injections without
mission-specific dependencies.

## Phase B — Decoded Sequences, Transforms, And Experimental FTS

- [ ] Add a decoded `ObservationSequence` carrying data, sample/detector axes,
  aligned flags, housekeeping, observation context, provenance, grouping, and
  bounded iteration. Packet bytes remain an adapter concern.
- [ ] Add the missing generic transform/differential operators only where not
  already covered: FFT/DCT, dense-reference non-uniform transform, irregular
  resampling, line-spread convolution, finite differences, pair/reference/common
  mode/on-off/scan-direction differences, and exact adjoints.
- [ ] Add experimental `DelayAxis` and `Interferogram` contracts with optical
  path, zero-path difference, sidedness, scan direction, sweep/segment IDs,
  missing samples, flags, detector context, and regularity diagnostics.
- [ ] Implement an idealized differential spectral/interferogram operator with a
  tiny dense oracle, regular and irregular OPD, finite maximum OPD, phase,
  throughput mismatch, flags, detector batches, and exact adjoint.
- [ ] Add composable ZPD estimation, alignment, apodization, phase correction,
  baseline, temporal response, scan reversal, and leakage diagnostics rather
  than a monolithic FTS pipeline.

### Phase B exit gate

Forward and adjoint FTS paths match dense fixtures for delta lines and analytic
spectra under regular, irregular, masked, single-sided, and double-sided scans.

## Phase C — Discovery-Preserving Validation

- [ ] Define distribution-agnostic `NuisanceLearner` and `NoiseLearner`
  protocols with classical and user-supplied Torch adapters; no normalized
  likelihood is required.
- [ ] Add weighted `ProtectedSubspace` projections and report target/nuisance
  overlap, effective target norm, condition number, and hard/soft protection
  consequences.
- [ ] Add an `OrthogonalTargetEstimator` result containing fold estimates,
  attenuation, effective sensitivity, nuisance-architecture sensitivity,
  partition sensitivity, and unresolved degeneracies.
- [ ] Add an `OpenResidualProduct` exportable at sequence, interferogram,
  spectrum, and full-sky levels, with observed, targeted, predictable nuisance,
  unexplained, whitened unexplained, context, and provenance fields.
- [ ] Add standard null constructors for detector, time/half-mission,
  scan-direction, sky/reference, calibration-state, odd/even sweep,
  frequency-subset, and shuffled-housekeeping differences.
- [ ] Add paired-injection target response, nonlinearity, cross-talk,
  false-positive, detection-efficiency, shift-test, and identifiability reports.

### Phase C exit gate

Every weak-target result reports science bias and attenuation, split and
architecture stability, null failures, residual covariance/autocorrelation,
unsupported regimes, deterministic provenance, throughput, and peak memory.

## Phase D — Science Benchmarks

- [ ] Recover the standard `mu ~ 2e-8` target under foreground and instrument
  misspecification using identical diagnostics for parametric/moment, PCA/NMF,
  learned-basis, continuous-field, and protected hybrid models.
- [ ] Produce a full-sky open residual atlas ranked by detector/scan/time
  reproducibility, celestial consistency, calibration stability, spectrum, sky
  morphology, and external-tracer correlation.
- [ ] Add blinded dark-sector and unknown-shape spectra withheld from model
  development; report detection efficiency by morphology and nuisance overlap,
  not only particle parameters.
- [ ] Add joint monopole/anisotropy residual modes and cross-correlations with
  CMB T/E/B, CIB, line-intensity, and large-scale-structure tracers.
- [ ] With a thermalization theorist, test robust residuals against the broad
  standard Compton/double-Compton/bremsstrahlung thermalization manifold.

## Required Benchmark Tiers

- [ ] Exact fixtures: coordinate/radiometric round trips, response integrals,
  operator adjoints, differential signs, FTS lines/blackbodies, and monopole
  contracts.
- [ ] Controlled stress: spatially varying spectral curvature, low-rank+sparse
  nuisance, correlated/non-stationary noise, missing data, response/phase/gain
  shifts, calibration drift, target overlap, and withheld morphologies.
- [ ] Real-background injections: observed multi-frequency/spectroscopic
  nuisance backgrounds so claims do not depend on one simulator family.

## Immediate Ordering Relative To The Active Torchsky Cycle

The active Foam/map operator gates in [`../roadmap/active.md`](../roadmap/active.md)
remain coupled to this track because spatial-spectral inference depends on
correct, bounded map operators. The next implementation order is:

1. introduce the minimal `TensorSpec` and typed operator adapter;
2. promote spectral response/channel mixing and gate its adjoint;
3. add weighted protected-subspace projections;
4. implement cross-fit partitions and paired injections;
5. implement full-sky spectral/monopole/open-residual products;
6. only then start decoded sequences and experimental FTS.
