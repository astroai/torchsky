# TorchSky Roadmap for FOSSIL-Ready Full-Sky Spectroscopy

**Status:** proposed technical roadmap
**Date:** July 2026
**Primary goal:** make TorchSky useful for FOSSIL and BISOU while building reusable infrastructure for full-sky, multi-frequency, spectroscopic, time-ordered, and open-discovery analyses.

## 1. Executive recommendation

TorchSky should not become a mission-specific CMB package. It should become an **observation-native operator and validation substrate** capable of representing the path

```text
latent sky / learned scene
    -> instrument response
    -> detector sequence or interferogram
    -> calibrated spectrum
    -> full-sky spectral product
    -> targeted estimate + open residual
```

The durable contribution is not a particular foreground network. It is the combination of:

1. typed physical axes and radiometric semantics;
2. executable, adjointable, chunkable observation operators;
3. first-class full-sky spectral and monopole products;
4. decoded detector-sequence and interferogram containers;
5. model-agnostic nuisance and noise learners;
6. leakage-safe cross-fitting and target-protection tools;
7. injections, null tests, shift tests, and identifiability reports;
8. an unexplained residual that remains a scientific product.

FOSSIL-specific templates, sensitivity assumptions, channel tables, and telemetry adapters should initially live in a separate research package or experimental namespace.

## 2. Architectural boundary

### TorchSky core should own

- axes, tensor specifications, units, masks, weights, flags, and provenance;
- spectral response, radiometry, beams, transforms, and differential operators;
- full-sky dense and sparse spectral maps;
- time/sample/delay sequences after mission-specific packet decoding;
- operator composition, adjoints, normal operators, and bounded execution;
- generic nuisance-learning and residual interfaces;
- cross-fitting, injections, null tests, and validation reports;
- deterministic benchmarks and trusted numerical fixtures.

### A FOSSIL/BISOU extension should own

- exact band definitions and baseline instrument configurations;
- CMB temperature, $y$, $\mu$, relativistic, recombination, and exotic templates;
- mission scan strategy and engineering metadata conventions;
- FOSSIL/BISOU file-format and telemetry adapters;
- collaboration simulations and sensitivity assumptions;
- publication-specific estimators and learned architectures.

### Explicit non-goals

- Do not make TorchSky a spacecraft packet decoder.
- Do not hard-code a complete physical foreground ontology.
- Do not require every learned nuisance or noise model to define a normalized probability distribution.
- Do not let a learned scene silently absorb known geometry, bandpass, FTS, reference, or sampling transformations.
- Do not claim unexplained residuals are automatically new physics.

## 3. Existing TorchSky foundations to preserve

The current repository already contains most of the architectural seeds:

- torch-native WCS, HEALPix, spherical geometry, harmonic transforms, and sparse maps;
- full-sky and image-plane mapmaking;
- `ObservationOperatorBundle` slots for geometry, spectral response, PSF, radiometry, background, masks/weights, and noise;
- matrix-free multi-observation execution and exact-adjoint testing;
- continuous spatial-spectral fields $F_\theta(\mathrm{RA},\mathrm{Dec},\lambda)$;
- learned spectral bases, contextual field models, and shared/private component decompositions;
- residual products, support-aware quality diagnostics, and bounded-memory execution.

The roadmap should strengthen these layers rather than create a parallel `cmb_pipeline` that bypasses them.

## 4. Design principles

1. **Axis explicitness.** Frequency, wavelength, energy, time, optical-path difference, detector, and scan axes are metadata-bearing objects, not anonymous tensor dimensions.
2. **Operator explicitness.** Known measurement transformations remain executable operators with tested adjoints.
3. **Distributionally agnostic nuisance modelling.** A nuisance learner may be predictive without being a generative probability model.
4. **Open residuals.** Stable unexplained structure is retained, versioned, and tested rather than automatically reclassified as nuisance.
5. **Identifiability before sensitivity.** A weak-signal estimate is incomplete without target attenuation, nuisance overlap, split stability, and null-test results.
6. **Raw-to-map continuity.** Detector sequences, interferograms, spectra, and maps share compatible tensor, operator, split, and provenance contracts.
7. **Full-sky correctness.** Monopoles, dipoles, partial coverage, sparse maps, beams, masks, and coordinate frames are first-class concerns.
8. **Mission-neutral core.** New primitives must have at least one credible application beyond FOSSIL.

## 5. Proposed package layout

```text
torchsky/
  axes/
    base.py
    spectral.py
    time.py
    delay.py
    detector.py

  radiometry/
    quantity.py
    conversion.py
    response.py
    calibration.py

  operators/
    protocol.py
    algebra.py
    execution.py
    spectral.py
    transform.py
    differential.py
    temporal.py
    covariance.py

  sequence/
    observation.py
    flags.py
    housekeeping.py
    grouping.py

  fullsky/
    spectral_map.py
    sparse_spectral_map.py
    monopole.py
    beam.py
    splits.py
    statistics.py

  field/
    nuisance.py
    protected.py
    crossfit.py
    open_residual.py

  validation/
    partition.py
    injection.py
    nulls.py
    identifiability.py
    shifts.py
    reports.py

  experimental/
    fts/
      interferogram.py
      operator.py
      zpd.py
      phase.py
      apodization.py

    discovery/
      residual_basis.py
      stability.py
      anomaly.py
```

Names are provisional. The important separation is between **data contracts**, **measurement operators**, **full-sky products**, **learned nuisance models**, and **scientific validation**.

---

## 6. Primitive roadmap

### P0. Tensor specifications and typed axes

#### 6.1 `TensorSpec`

A small immutable description of a tensor's meaning:

```python
@dataclass(frozen=True)
class TensorSpec:
    axes: tuple[AxisSpec, ...]
    value_semantics: str
    unit: str | None
    coordinate_frame: str | None = None
```

Required properties:

- named dimensions;
- axis lengths and optional coordinates;
- value semantics such as surface brightness, integrated flux, power, counts, or dimensionless field;
- physical unit;
- coordinate frame;
- broadcast and compatibility checks.

This avoids a recurring failure mode in scientific Python: shape-compatible tensors with incompatible meanings.

#### 6.2 `SpectralAxis`

```python
@dataclass(frozen=True)
class SpectralAxis:
    values: Tensor
    quantity: Literal["frequency", "wavelength", "energy", "wavenumber"]
    unit: str
    bin_edges: Tensor | None = None
```

Required operations:

- quantity and unit conversion;
- monotonicity and duplicate checks;
- ascending/descending orientation;
- regular versus irregular sampling detection;
- channel widths and quadrature weights;
- index selection by physical interval;
- device/dtype transfer;
- FITS-WCS adapter without moving FITS I/O back into TorchSky.

#### 6.3 `TimeAxis`, `SampleAxis`, and `DelayAxis`

`DelayAxis` represents optical-path difference and must include:

- physical values and unit;
- zero-path-difference estimate or index;
- single-sided versus double-sided acquisition;
- scan direction;
- segment and sweep identifiers;
- missing-sample mask;
- regularity diagnostics.

#### Acceptance gates

- reference round trips among frequency, wavelength, energy, and wavenumber;
- irregular and descending-axis fixtures;
- CPU/CUDA parity;
- gradient checks where conversions are differentiable;
- fail-closed dimensional semantics;
- no silent unit guessing.

#### Reuse beyond FOSSIL

IFU cubes, radio spectra, X-ray energy cubes, laboratory spectrometers, time-domain surveys, line-intensity mapping, and remote sensing.

---

### P1. Spectral responses and radiometric semantics

#### 6.4 `SpectralResponse`

Promote spectral transmission from observation-specific fields to a reusable object:

```python
@dataclass
class SpectralResponse:
    axis: SpectralAxis
    response: Tensor
    normalization: Literal[
        "none",
        "unit_integral",
        "unit_peak",
        "photon_weighted",
        "energy_weighted",
    ]
```

Required methods:

```python
response.evaluate(query_axis)
response.integrate(spectrum)
response.quadrature(n=...)
response.effective_coordinate(reference_spectrum)
response.perturb(coefficients)
```

Required features:

- batched detector responses;
- tabulated and analytic curves;
- line-spread functions;
- channel-mixing matrices;
- compact response-uncertainty bases;
- sparse or compact-support representation;
- energy-counting and photon-counting conventions.

#### 6.5 `RadiometricQuantity`

```python
@dataclass(frozen=True)
class RadiometricQuantity:
    unit: str
    semantics: Literal[
        "surface_brightness",
        "flux_density",
        "integrated_flux",
        "temperature",
        "counts",
        "power",
    ]
    convention: str | None = None
```

Implement:

- $I_\nu \leftrightarrow I_\lambda$ transformations with Jacobians;
- spectral radiance and surface-brightness conversions;
- flux versus surface-brightness conversions using pixel solid angle;
- brightness temperature and Rayleigh-Jeans temperature;
- bandpass-aware colour corrections;
- calibration Jacobians;
- optional CMB thermodynamic units in an extension.

#### Acceptance gates

- dimensional round trips;
- flux-conservation fixtures;
- reference blackbody values;
- response integration against analytic spectra;
- detector-batch and device parity;
- gradient checks through response parameters.

#### Reuse beyond FOSSIL

Cross-survey photometric calibration, synthetic photometry, IFU calibration, radio cubes, and detector-array spectroscopy.

---

### P2. Executable operator protocol and algebra

TorchSky's semantic operator bundle should remain. Add a common mathematical execution protocol beneath it.

#### 6.6 `LinearOperator`

```python
class LinearOperator(Protocol):
    domain_spec: TensorSpec
    codomain_spec: TensorSpec

    def forward(self, x: Tensor) -> Tensor: ...
    def adjoint(self, y: Tensor) -> Tensor: ...
```

Optional capabilities:

```python
operator.normal(x)
operator.diagonal_approximation()
operator.jvp(x, tangent)
operator.vjp(x, cotangent)
operator.to(device=..., dtype=...)
operator.iter_forward_chunks(x, chunk_spec=...)
```

#### 6.7 Operator combinators

- `CompositeOperator(A, B)` for $A(Bx)$;
- `SumOperator`;
- `ScaledOperator`;
- `StackOperator`;
- `BlockDiagonalOperator`;
- `DifferenceOperator`;
- `MaskedOperator`;
- `LowRankOperator`;
- `IdentityOperator`;
- deterministic reduce/accumulate operators.

#### 6.8 Operator test harness

```python
check_adjoint(operator, x, y)
check_linear(operator, x1, x2, a, b)
check_chunk_parity(operator, x)
check_device_parity(operator, x)
```

Every promoted operator must have:

- dot-product adjoint tests;
- deterministic replay;
- chunked/eager parity;
- documented dtype/device behaviour;
- explicit linear or nonlinear declaration;
- realistic peak-memory and throughput benchmarks.

#### Reuse beyond FOSSIL

Mapmaking, deconvolution, survey coaddition, interferometry, learned instrument models, and generic inverse problems.

---

### P3. Transform and differential measurement operators

#### 6.9 Generic transforms

Implement:

- Fourier transform;
- cosine transform;
- non-uniform Fourier transform;
- convolution and line-spread convolution;
- regular/irregular-axis resampling;
- spectral mixing;
- finite-difference and smoothness operators;
- optional wavelet transforms.

Each transform should have:

1. a tiny explicit-matrix reference implementation;
2. a fast regular-grid implementation;
3. a masked or non-uniform implementation where relevant;
4. an exact adjoint;
5. batched detector and sky dimensions.

#### 6.10 Differential operators

Implement:

- pair difference;
- sky-reference difference;
- common-mode subtraction;
- on/off observation difference;
- scan-direction difference;
- detector-group difference.

The difference must live in the operator graph. Post-hoc subtraction loses calibration derivatives and makes provenance ambiguous.

#### Reuse beyond FOSSIL

Chopped infrared observations, beam switching, nodding, pair-differenced polarimetry, delay-spectrum radio work, and laboratory FTS measurements.

---

### P4. Observation sequences and decoded raw-data boundary

TorchSky should accept **decoded detector samples**, not mission packet bytes.

#### 6.11 `ObservationSequence`

```python
@dataclass
class ObservationSequence:
    data: Tensor
    sample_axis: SampleAxis
    detector_axis: DetectorAxis | None
    flags: Tensor | None
    context: ObservationContext
    housekeeping: dict[str, Tensor]
    operator: LinearOperator | None = None
```

Context fields should include:

- survey and instrument;
- detector or detector group;
- observation and exposure identifiers;
- scan and scan direction;
- sky/reference state;
- calibrator state;
- time range;
- processing level and provenance.

Required operations:

- group by detector, scan, epoch, state, or flag;
- aligned slicing of signal and housekeeping;
- lazy/chunked iteration;
- deterministic split creation;
- aggregation into interferograms, spectra, or maps;
- preservation of flags and provenance.

#### Data-level recommendation

| Level | Example | TorchSky responsibility |
|---|---|---|
| L0 | packets and binary telemetry | no |
| L0.5 | decoded samples and housekeeping | adapter boundary |
| L1 | aligned timelines and OPD samples | yes |
| L1.5 | corrected interferograms | yes |
| L2 | calibrated spectra | yes |
| L3 | full-sky spectral maps | yes |

#### Reuse beyond FOSSIL

CMB time streams, radio scan data, detector arrays, repeated spectroscopy, and laboratory sequences.

---

### P5. Fourier-transform spectroscopy primitives

Keep these experimental until they have independent datasets and numerical fixtures.

#### 6.12 `Interferogram`

```python
@dataclass
class Interferogram:
    signal: Tensor
    delay_axis: DelayAxis
    detector_axis: DetectorAxis | None
    flags: Tensor | None
    context: ObservationContext
```

#### 6.13 `SpectralInterferogramOperator`

For an idealized differential FTS,

$$
I(x) = \int d\nu\, R(\nu)
\left[S_{\mathrm{sky}}(\nu)-S_{\mathrm{ref}}(\nu)\right]
\cos\!\left(\frac{2\pi \nu x}{c}+\phi(\nu,x)\right).
$$

The operator should support:

- uniform and irregular optical-path sampling;
- finite maximum OPD and instrument line shape;
- single- and double-sided scans;
- phase perturbations;
- missing and flagged samples;
- sky/reference throughput mismatch;
- batched detectors;
- exact adjoint;
- explicit dense reference path.

#### 6.14 FTS correction primitives

- zero-path-difference estimation;
- interferogram alignment;
- apodization windows;
- phase correction;
- baseline removal;
- detector temporal response;
- scan reversal;
- spectral leakage diagnostics.

These should be small composable operators rather than one monolithic pipeline.

#### Reuse beyond FOSSIL

BISOU, Herschel/SPIRE-like archival analysis, SITELLE/CONCERTO-adjacent workflows, laboratory FTS systems, and delay-domain signal processing.

---

### P6. Full-sky spectral and CMB primitives

#### 6.15 `FullSkySpectralMap`

```python
@dataclass
class FullSkySpectralMap:
    data: Tensor               # [spectral, pixel] or [spectral, component, pixel]
    spectral_axis: SpectralAxis
    nside: int
    ordering: Literal["ring", "nested"]
    frame: str
    unit: str
    mask: Tensor | None
    weight: Tensor | None
    beam: BeamModel | None
```

Required features:

- dense and sparse representations;
- RING/NEST conversion;
- resolution changes with explicit value semantics;
- beam matching and harmonic convolution;
- map splits and coadds;
- per-channel masks and coverage;
- spherical-harmonic transforms by spectral batch;
- streaming/chunking across frequency or pixel blocks.

#### 6.16 Absolute-sky and monopole conventions

Absolute spectroscopy needs an explicit separation between:

- monopole;
- dipole;
- anisotropic field;
- masked-sky estimate;
- reference/calibration offset.

Add:

```python
MonopoleEstimate
DipoleEstimate
MonopoleAnisotropyDecomposition
AbsoluteZeroLevel
```

A map operation must state whether it preserves, removes, or estimates the monopole.

#### 6.17 CMB template interface

Keep the generic primitive in TorchSky:

```python
class SpectralTemplate(Protocol):
    def evaluate(self, axis: SpectralAxis, context=None) -> Tensor: ...
```

Put actual $\Delta T$, $y$, $\mu$, relativistic, recombination, and dark-sector implementations in an extension until validated.

#### 6.18 Full-sky statistics

- per-frequency auto- and cross-spectra;
- template-amplitude maps;
- map-to-map cross-correlation;
- split cross-spectra;
- spectral covariance by sky region;
- correlations with CMB $T/E/B$, CIB, LIM, and large-scale-structure tracers;
- monopole-plus-anisotropy joint summaries.

#### Reuse beyond FOSSIL

CMB imaging, 21-cm surveys, diffuse Galactic emission, gamma/X-ray all-sky maps, and line-intensity mapping.

---

### P7. Learned nuisance and noise interfaces

The library should not prescribe one architecture.

#### 6.19 `NuisanceLearner`

```python
class NuisanceLearner(Protocol):
    def fit(self, train_data, *, context=None): ...
    def predict(self, target_data, *, context=None) -> Tensor: ...
    def representation(self, data, *, context=None) -> Tensor | None: ...
```

Adapters should cover:

- PCA and robust PCA;
- NMF;
- learned spectral bases;
- continuous fields;
- autoencoders and masked predictors;
- temporal/common-mode models;
- user-supplied PyTorch modules.

#### 6.20 `NoiseLearner`

A noise learner need not provide a likelihood. It may provide any of:

- predictable conditional mean;
- whitening transform;
- residual scale;
- empirical covariance operator;
- noise-state embedding;
- block-resampling or surrogate generator;
- anomaly/non-stationarity score.

```python
class NoiseLearner(Protocol):
    def fit(self, train_sequences): ...
    def remove_predictable(self, target_sequences): ...
    def whiten(self, residual): ...
    def diagnostics(self, residual): ...
```

#### 6.21 Operator-versus-nuisance discipline

Known transforms should remain outside the learner. A learned model may correct residual phase, response, or drift only through an explicitly named learned operator slot with conditioning and validation metadata.

#### Reuse beyond FOSSIL

Detector common modes, atmospheric subtraction, correlated spectroscopy residuals, transient backgrounds, and multi-survey fusion.

---

### P8. Protected targets, cross-fitting, and open residuals

#### 6.22 `ProtectedSubspace`

```python
protected = ProtectedSubspace(
    templates=[template_a, template_b],
    metric=weight_operator,
)
```

Capabilities:

- weighted projection;
- target/nuisance overlap;
- effective target norm after projection;
- condition number;
- hard and soft protection;
- template-bank and local tangent-space modes.

Hard projection is a tool, not a guarantee of unbiased inference. The validation layer must quantify its consequences.

#### 6.23 `CrossFitPartition`

Cross-fitting must support grouping by:

- sky region;
- detector;
- scan;
- time block;
- instrument;
- calibration state;
- combinations of these units.

The split manifest must record the actual independence or exchangeability unit. Random pixel splitting should not be the default.

#### 6.24 `OrthogonalTargetEstimator`

A generic estimator for a low-dimensional target in the presence of a learned high-dimensional nuisance:

```python
result = OrthogonalTargetEstimator(
    target=template,
    nuisance_learner=learner,
    partition=partition,
).fit(observations)
```

The result should include:

- target amplitude and fold estimates;
- target attenuation;
- target/nuisance overlap;
- effective target sensitivity;
- sensitivity to nuisance architecture;
- sensitivity to partition choice;
- unresolved degeneracies.

#### 6.25 `OpenResidualProduct`

```python
OpenResidualProduct(
    observed=...,
    targeted_prediction=...,
    predictable_nuisance=...,
    unexplained=...,
    whitened_unexplained=...,
    context=...,
    provenance=...,
)
```

The unexplained residual must remain exportable at detector, interferogram, spectrum, and full-sky levels.

#### Reuse beyond FOSSIL

Exoplanet spectra, 21-cm foreground subtraction, transient detection, weak lensing, and line searches.

---

### P9. Injection, null-test, and identifiability infrastructure

#### 6.26 Generic injections

```python
@dataclass
class Injection:
    signal: Tensor | Callable
    amplitude: Tensor
    domain: Literal["scene", "measurement", "sequence", "interferogram", "map"]
    metadata: dict
```

Required capabilities:

- positive/negative paired injections;
- global and localized injections;
- multiple amplitudes;
- before/after-operator injection;
- blinded amplitudes;
- batched GPU execution;
- deterministic replay.

#### 6.27 Target-response diagnostics

For a target template $s$ and amplitude $a$, report the symmetric response

$$
\mathcal{R}(a) =
\frac{\widehat{\alpha}(d+a s)-\widehat{\alpha}(d-a s)}{2a}.
$$

Key outputs:

- attenuation $1-\mathcal{R}$;
- nonlinearity over amplitude;
- cross-talk to other templates;
- region/detector/time dependence;
- null false-positive rate;
- detection efficiency.

#### 6.28 Null tests

Provide standard null constructors:

- detector difference;
- half-mission or time difference;
- forward/reverse scan difference;
- sky/reference-state difference;
- calibration-state difference;
- odd/even sweep difference;
- frequency-subset difference;
- shuffled-housekeeping test.

#### 6.29 `IdentifiabilityReport`

At minimum:

```text
target overlap with nuisance representation
target attenuation under injections
effective target norm and condition number
fold-to-fold instability
architecture-to-architecture instability
partition sensitivity
null-test summary
unsupported or extrapolative regimes
```

#### 6.30 Shift tests

Reusable perturbations:

- response-curve shift;
- beam shift;
- gain/offset drift;
- missing channels;
- noise-state shift;
- foreground-complexity shift;
- spatial-domain shift;
- instrument holdout.

#### Reuse beyond FOSSIL

Any scientific ML problem where a weak target can be learned away by a high-capacity nuisance model.

---

## 7. Benchmark programme

### Tier A: exact mathematical fixtures

- spectral-coordinate conversions;
- response integrals against analytic spectra;
- operator forward/adjoint identities;
- differential sign conventions;
- FTS delta-line and blackbody fixtures;
- radiometric conversion round trips;
- monopole preservation/removal contracts.

### Tier B: controlled stress tests

- low-rank plus sparse nuisance;
- spatially varying spectral curvature;
- correlated and non-stationary noise;
- missing samples and channels;
- response shifts and phase errors;
- calibration drift;
- target injections with varying overlap;
- unknown-shape injections withheld from development.

### Tier C: real-data nuisance backgrounds

Use observed multi-frequency or spectroscopic data as nuisance backgrounds and inject known test signals. The benchmark should not depend entirely on one simulated foreground family.

### Required reporting

- science error, not only reconstruction RMSE;
- target attenuation and cross-talk;
- split and architecture stability;
- null failures;
- residual covariance and autocorrelation;
- throughput and peak memory;
- transfer volume;
- deterministic checksums;
- device parity;
- data and configuration provenance.

---

## 8. Prioritized implementation backlog

### First six pull requests

#### PR 1: typed spectral axes

- `SpectralAxis`;
- quantity/unit conversion;
- regular/irregular grids;
- compatibility adapters from current field observations;
- tests and documentation.

#### PR 2: reusable spectral responses and radiometry

- `SpectralResponse`;
- quadrature;
- batched detector responses;
- response perturbation bases;
- radiometric quantity semantics.

#### PR 3: executable operator protocol

- `LinearOperator`;
- composition and stack operators;
- common adjoint test harness;
- adapters around existing map operators;
- chunk/eager parity tests.

#### PR 4: full-sky spectral map and monopole semantics

- dense and sparse spectral maps;
- channel batching;
- beam metadata;
- monopole/dipole decomposition;
- split products.

#### PR 5: scientific partitions and injections

- detector/time/scan/region partition manifests;
- paired injections;
- target-response report;
- estimator-independent result schema.

#### PR 6: observation sequences

- decoded sequence container;
- aligned flags and housekeeping;
- deterministic grouping and chunking;
- conversion hooks to maps/spectra.

### Next six pull requests

#### PR 7: generic transform and differential operators

FFT/DCT, non-uniform reference path, difference/reference operators, and exact adjoints.

#### PR 8: protected subspaces and identifiability

Target projectors, overlap metrics, attenuation reports, and condition diagnostics.

#### PR 9: nuisance/noise learner protocols

Classical and Torch adapters; no required likelihood interface.

#### PR 10: open residual products and null suite

Residuals at sequence, spectral, and full-sky levels with split comparisons.

#### PR 11: experimental FTS layer

Delay axis, interferogram object, spectral/interferogram operator, ZPD and phase primitives.

#### PR 12: FOSSIL-like end-to-end benchmark

Standard $\mu$ injection, flexible nuisance/noise baselines, map/interferogram consistency, and unknown-shape challenge injections.

---

## 9. Suggested delivery phases

### Phase A: reusable spectral/full-sky substrate

Deliver PRs 1-6. Demonstrate on existing multiband/full-sky data. No FOSSIL-specific dependency is required.

**Exit condition:** TorchSky can represent irregular frequency channels, absolute radiometric values, full-sky spectral maps, scientific splits, and signal injections with tested semantics.

### Phase B: weak-target inference and validation

Deliver PRs 7-10. Compare PCA/NMF, learned spectral bases, and continuous fields with identical target-response diagnostics.

**Exit condition:** the library can quantify whether a nuisance model improves prediction while attenuating or fabricating a weak target.

### Phase C: decoded interferogram support

Deliver PR 11 and validate against synthetic fixtures plus a public or collaboration-provided FTS dataset.

**Exit condition:** forward and adjoint FTS paths agree with dense references and preserve signal under irregular/masked scans.

### Phase D: collaboration benchmark

Deliver PR 12 in a separate experiment package, using FOSSIL/BISOU configurations and simulations.

**Exit condition:** reproducible comparison of targeted $\mu$ recovery and open unknown-shape injections across maps, spectra, and interferograms.

---

## 10. What makes this useful even without FOSSIL

The roadmap creates reusable capabilities for:

- full-sky multi-frequency surveys;
- 21-cm and line-intensity mapping;
- radio and millimetre time streams;
- IFU and hyperspectral imaging;
- laboratory Fourier-transform spectroscopy;
- cross-instrument calibration;
- learned atmospheric and detector subtraction;
- exoplanet and transient spectroscopy;
- any inverse problem where a flexible nuisance model may absorb a weak scientific signal.

The durable scientific-software identity is:

> **TorchSky provides explicit measurement operators, scalable sky representations, flexible nuisance-learning interfaces, and empirical tests of identifiability and model incompleteness.**
