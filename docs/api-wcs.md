# WCS API reference

`torchsky.wcs` implements FITS WCS transformations as torch tensor operations.
The [WCS guide](wcs.md) covers concepts and recipes; this page is the reference
for the namespaces and the promoted names.

Everything here accepts and returns `torch.Tensor` (or plain Python scalars for
frame helpers) and runs on CPU, CUDA, and Apple Silicon MPS.

---

## `torchsky.wcs.WCS`

Build a transform from a `torchfits` header mapping:

```python
from torchsky.wcs import WCS

wcs = WCS(header)
```

| Method | Contract |
|---|---|
| `pixel_to_world(*pixel_arrays)` | Pixel coordinates → world coordinates, dispatching on the axes present (celestial, spectral, time, tabular). |
| `world_to_pixel(*world_arrays)` | Inverse of `pixel_to_world` with the same axis dispatch. |
| `pixel_to_world_values(*pixel_arrays, origin=0)` | Lower-level tuple form; `origin` selects 0- or 1-based pixels. |
| `world_to_pixel_values(*world_arrays, origin=0)` | Lower-level tuple form for the inverse direction. |
| `to(device)` | Move the WCS (and any baked tables) to a device. |

Return convention: a multi-axis WCS returns one tensor per axis
(celestial, spectral, time, tabular, ...). A single-axis WCS such as a pure
spectral FITS header returns the world array directly, and the inverse accepts
that array directly. `pixel_to_world_values` always returns a tuple.

Celestial transforms support SIP, TPV, paper-IV lookup tables, and coupled
`-TAB` axes. Spectral and time axes are transforms too, not metadata.

---

## Distortion and tabular transforms

| Name | Role |
|---|---|
| `DistortionLookupTable` | FITS WCS Paper IV 2D lookup table (`CPDIS` / `DET2IM`). |
| `TabularAxis` | One-dimensional table-driven coordinate transform. |
| `TabularLookupWCS` | Separable multi-axis table-driven WCS transform. |
| `CoupledTabularLookupWCS` | Coupled multidimensional table lookup with multilinear interpolation. |
| `from_tabular(header)` | Build explicit `WCS(..., tabular_axes=...)` axes from FITS `-TAB` metadata. |
| `from_coupled_tabular(header)` | Build a forward coupled table-lookup WCS. |
| `tabular_axes_from_fits(header)` | NumPy-free axis construction helper. |
| `coupled_tabular_wcs_from_fits(header)` | Coupled-WCS construction helper. |
| `RegularGridTransform2D` | Frozen regular-grid detector transform with consistent sampled parity. |
| `DetectorToTangentTransform` | Compose a fixed table, exposure affine, and optional detector residual. |
| `DetectorToSkyTransform` | Forward-only detector-to-sky composition with a frozen attitude. |
| `FrozenDetectorEpochTables` | Time-dependent detector tables as frozen samples, not a live GWCS graph. |

GWCS sampling is a build-time step: `sample_gwcs`, `sample_gwcs_epochs`,
`sample_gwcs_detector_epochs`, and `sample_gwcs_detector_to_tangent` refine a
detector table until independent points satisfy the configured milliarcsecond
gates, then runtime never imports `gwcs`.

Validation helpers: `validate_sampled_transform`, `SampledTransformValidation`,
`GWCSGridSamplingResult`.

---

## Celestial frames and rotations

| Name | Role |
|---|---|
| `galactic_to_icrs`, `icrs_to_galactic` | Galactic ↔ ICRS. |
| `ecliptic_to_icrs`, `icrs_to_ecliptic` | Mean ecliptic of J2000 ↔ ICRS. |
| `ecliptic_of_date_to_icrs`, `icrs_to_ecliptic_of_date` | Ecliptic of date via IAU 2006 NPB. |
| `precess_fk5` | Precess FK5 equatorial coordinates between Julian equinoxes. |
| `SPEED_OF_LIGHT` | Speed of light in the units used by the velocity helpers. |
| `TabularBoundsMode` | Out-of-bounds policy for tabular axes. |
| `transform(lon, lat, *, from_frame, to_frame)` | Generic rotation/observation dispatch on `(lon, lat)` tensors. |
| `canonical_celestial_frame(name)` | Normalize FITS/AST celestial frame names. |
| `convert_celestial(lon, lat, from_frame, to_frame)` | Convert between supported celestial frames. |
| `galactocentric_to_icrs`, `icrs_to_galactocentric` | Heliocentric ICRS ↔ Galactocentric Cartesian. |

---

## Motion, time, and observation effects

| Name | Role |
|---|---|
| `propagate_proper_motion`, `propagate_proper_motion_with_jacobian` | Angular motion with analytic Jacobian. |
| `propagate_space_motion`, `propagate_space_motion_with_jacobian` | Full barycentric astrometric state with 6D Jacobian. |
| `apply_ephemeris_offset` | Comoving-frame shift for moving targets. |
| `apply_stellar_aberration` | Classical stellar aberration. |
| `apply_refraction`, `apply_dcr`, `dcr_offset` | Refraction and differential chromatic refraction on sky coordinates. |
| `observed_altaz`, `icrs_to_itrs_xyz`, `itrs_xyz_to_lonlat_deg` | Observation geometry with explicit Earth orientation (`EarthOrientation`, `Site`). |
| `heliocentric_rv_to_lsr`, `lsr_line_of_sight_kms` | LSR corrections with explicit solar UVW. |

`EarthOrientation` requires `dut1_s`; nothing guesses it.

---

## Spectral and time axes

| Name | Role |
|---|---|
| `canonical_spectral_type`, `convert_spectral` | Canonical FITS spectral coordinate types and conversions. |
| `spectral_to_frequency`, `frequency_to_spectral` | Frequency-based conversions. |
| `air_wavelength_to_vacuum`, `vacuum_wavelength_to_air` | Air/vacuum wavelength conversion. |
| `rest_frequency_from_header` | `RESTFRQ`/`FREQ`/`RESTWAV` resolution. |
| `transform_spectral` | Compatibility helper returning km/s. |
| `IFUSliceWCS` | Piecewise-disjoint detector → 3D `(ra_deg, dec_deg, wavelength)` IFU slice WCS. |
| `CoupledTabularLookupWCS`, `TabularLookupWCS` | Shared with the tabular section above. |

---

## Affine and residual corrections

| Name | Role |
|---|---|
| `fit_affine_2d`, `AffineFitResult` | Weighted 2D affine correction between point sets. |
| `fit_sky_affine`, `SkyAffineFitResult` | Local tangent-plane affine fit. |
| `TangentPlaneFrame`, `TangentPlaneAffine` | Gnomonic frame and learnable affine correction. |
| `AffineFreeResidualGrid2D` | Smooth detector-coordinate residual with constant/affine modes removed. |

---

## Typed mappings and transform graphs

| Name | Role |
|---|---|
| `UnitSpec`, `TimeSpec`, `AxisSpec`, `FrameSpec`, `DomainSpec` | Explicit units, time scales, axes, frames, tensor domains, and compatibility validation. |
| `Mapping`, `IdentityMapping`, `AffineMapping`, `UnitConversionMapping`, `CallableMapping`, `WCSMapping` | Tensor-native primitive mappings; arbitrary callables remain non-serializable adapters. |
| `SerialMapping`, `ParallelMapping`, `PortRoute`, `PortRoutingMapping` | Typed serial/parallel composition and explicit axis routing. |
| `NumericalInverseMapping`, `NumericalInverseResult`, `UnsupportedInverseError` | Bounded fail-closed numerical inverse with per-row convergence evidence. |
| `TransformEdge`, `TransformGraph`, `GeneralizedWCS` | Named frame graph, validated paths, intermediate frames, and N-D generalized WCS. |
| `AmbiguousPathError`, `UnsupportedPathError` | Explicit graph-path failures. |

## Global astrometry and coupled photometry

| Name | Role |
|---|---|
| `AstrometricObservation`, `AstrometricMatchGraph`, `AstrometryConfig` | Supplied detector centroids/covariance, associations, model blocks, gauges, and solve policy. |
| `AstrometryDiagnostics`, `AstrometrySolution` | Convergence, rank/gauge, robust weights, covariance, component, and corrected-transform results. |
| `ApproximateMatchResult`, `match_approximate_wcs(...)` | Existing-HEALPix approximate-WCS catalog association; no detection or new sky index. |
| `tangent_plane_block_residuals(...)`, `solve_astrometry(...)` | Whitened residual kernel and sparse Schur/CG supplied-catalog bundle solve. |
| `PhotometricMeasurementBlock`, `PhotometricReference`, `CoupledCalibrationGraph`, `CoupledCalibrationConfig` | Supplied flux measurements, standards, shared graph, global/local photometric blocks, and alternating calibration policy. |
| `PhotometricCalibrationDiagnostics`, `PhotometricCalibrationSolution` | Relative/absolute gauges, zeropoint, star-flat, color, variability, covariance, and holdout diagnostics. |
| `CoupledCalibrationDiagnostics`, `CoupledCalibrationSolution`, `solve_coupled_calibration(...)` | Joint astrometry/photometry/DCR convergence and result surface. |
| `photometric_log_residuals(...)` | Covariance-weighted photometric residual kernel. |

## Astrometry.net supplied-XYLS bootstrap

| Name | Role |
|---|---|
| `AstrometryNetConfig`, `AstrometryNetVerificationConfig` | Pinned executable/index, bounds, timeout, scale hints, and held-out verification policy. |
| `AstrometryNetIndexIdentity`, `AstrometryNetProvenance` | Immutable external index/executable and run provenance. |
| `AstrometryNetHypothesis`, `AstrometryNetVerification`, `AstrometryNetBootstrapResult` | Parsed hypothesis, independent unused-centroid verification, and refined bundle result. |
| `solve_with_astrometry_net(...)` | Optional bootstrap from caller-supplied XYLS followed by TorchSky bundle refinement. |
| `AstrometryNetError`, `AstrometryNetUnavailableError`, `AstrometryNetIdentityError`, `AstrometryNetExecutionError`, `AstrometryNetHypothesisRejected` | Fail-closed external bootstrap errors. |

## Gauge-safe astrometric residual models

| Name | Role |
|---|---|
| `TangentPlaneGauge`, `ResidualModeDesign`, `ResidualSupport`, `CorrectionEnvelope` | Exact gauge projection, interpretable modes, support envelope, and correction/Jacobian safety bounds. |
| `ResidualComponentInput`, `ResidualComponentResult`, `AstrometricResidualFeatures` | Supplied residual/context features and component-level results. |
| `BayesianModeState`, `BayesianModePrediction`, `fit_bayesian_modes(...)`, `predict_bayesian_modes(...)`, `update_bayesian_modes(...)`, `shadow_update_bayesian_modes(...)` | Low-rank instrument/atmosphere model, Bayesian updates, and versioned shadow promotion. |
| `SparseGPResidual`, `SparseGPResidualPrediction`, `fit_sparse_gp_residual(...)` | Torch-native sparse-GP baseline with calibrated covariance and strict support fallback. |
| `ExperimentalConditionalResidual`, `ConditionalResidualPrediction` | Experimental permutation-invariant compact-support neural residual and uncertainty head. |
| `VersionedResidualState`, `ShadowUpdateResult` | Replay/anchor identities, compatibility, no-forgetting, and rollback state. |
| `AstrometricResidualLadder`, `ResidualLadderPrediction`, `interpretable_residual_modes(...)` | Full hybrid prediction and component-wise fallback ladder. |

See also: [coordinates](coordinates.md), [astrometric calibration](astrometric_calibration.md),
and the [WCS guide](wcs.md) for distortion and `-TAB` walkthroughs.
