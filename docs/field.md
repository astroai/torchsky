# Continuous fields (`torchsky.field`)

This guide covers **experimental** spectral neural fields used to fit a continuous scene
\(F_\theta(\mathrm{RA}, \mathrm{Dec}, \lambda)\) to multi-band 2D observations.
Use the concise **`torchsky.field`** namespace for new code; the older
`torchsky.map.field` functions remain the implementation layer for map workflows.

For the scientific inverse problem shared by INRs and spectral Power Foam,
including the exact hard-cell formulation and the quantitative depth objective,
see [INR and Power-Foam formulation](field-formulation.md).

For map products and classical stacks, see [map.md](map.md). For the Charbonnier / multi-survey scaffold around field rendering, see [implicit_forward.md](implicit_forward.md).

Native stamp calibration and small likelihood/forced-photometry queries can run
without a fitted continuous field. See the [native likelihood guide](native-likelihood.md)
and [exposure example](published-examples/native_exposure_calibration.py) for their
conditional response, nuisance, rank and sampling-uncertainty contracts.

## Core workflow

1. Build one **`field.Observation`** per band (or per exposure): image tensor, FITS-like **`header`** dict (WCS), spectral limits or tabulated throughput, optional **`weight`** / **`mask`**, optional **`PSFSpec`**.
2. Call **`field.estimate_domain(observations)`** to get a **`field.Domain`** (sky center, spatial scale, wavelength center/scale) unless you set the domain by hand.
3. Construct a model explicitly with **`field.model(kind, domain, **kwargs)`**, or use the small convenience **`field.preset(name, domain, **kwargs)`** when a default recipe is enough.
4. **`field.fit(observations, model, ...)`** runs the built-in optimizer loop, or compose your own loop with **`field.render(...)`** / **`field.sample(...)`**.
5. **`field.render(...)`** produces observation predictions, cubes, or HEALPix products; **`field.sample(...)`** evaluates explicit sky-coordinate samples.

CPU and CUDA rendering preserve the WCS precision of absolute sky coordinates.
For custom models, subtract the sky centre and normalize coordinates before
casting those local features to the neural layer's dtype. Casting absolute
RA/Dec to float32 first can discard subpixel structure. Brightness tensors and
model parameters may remain float32. MPS still uses float32 sky coordinates;
fine angular structure on that device has not been verified.

Runnable references:

- [examples/example_map_field_survey_split.py](published-examples/example_map_field_survey_split.py)
- [examples/example_field_multiband_quality.py](published-examples/example_field_multiband_quality.py)
- [synthetic field-fit notebook](https://github.com/astroai/torchsky-harness/blob/main/notebooks/01_field_fit_synthetic.ipynb)
- [SMACS campaign notebooks and tools](https://github.com/astroai/torchsky-harness/tree/main/notebooks)
- [hyperspectral fusion onboarding](https://github.com/astroai/torchsky-harness/blob/main/docs/projects/hyperspectral-fusion.md)

## `field.Observation`

Defined in [`types.py`](https://github.com/astroai/torchsky/blob/main/src/torchsky/map/field/types.py). Important fields:

| Field | Role |
|-------|------|
| `image` | `[H, W]` float tensor (target patch). |
| `header` | Dict with `NAXIS*`, `CTYPE*`, `CRPIX*`, `CRVAL*`, `CD*` or `CDELT*`/`PC*` — same convention as FITS for **`torchsky.wcs.WCS.from_header`**. |
| `spectral_min`, `spectral_max` | Band limits (e.g. Å). |
| `transmission_center`, `transmission_sigma` | Gaussian bandpass summary (optional). |
| `transmission_wavelength`, `transmission_throughput` | 1D tabulated throughput (optional); quadrature integrates against this curve. |
| `weight` | Per-pixel weights (often inverse variance). |
| `mask` | Binary or soft mask (good pixels). |
| `psf_spec` | `PSFSpec.none`, `.gaussian(sigma_pix)`, or `.kernel(kernel2d)`. |
| `band_name` | Label for logging / grouping. |
| `ForwardContext` | Not on the observation — passed into render/fit when using context-conditioned models (see [implicit_forward.md](implicit_forward.md)). |

Helpers to build observation lists from IFU cubes or bin edges are exported from
**`torchsky.field`** (`field_observations_from_bin_edges`,
`make_tangent_plane_header`, …).

Observation coordinate grids are cached for repeated render/fit calls. The cache
tracks image shape, requested grid shape, and header card values, so in-place WCS
header edits or image-shape changes recompute RA/Dec grids on the next render.

## `field.Domain`

[`FieldDomain`](https://github.com/astroai/torchsky/blob/main/src/torchsky/map/field/types.py) normalizes sky + wavelength coordinates inside the network:

- **`center_ra_deg`, `center_dec_deg`** — patch center (degrees).
- **`spatial_scale_deg`** — characteristic sky extent (degrees); used to normalize tangent-plane offsets.
- **`wavelength_center`, `wavelength_scale`** — spectral normalization.
- **`coord_mode`** — `"local_tangent"` (default) or `"sphere_xyz"`.

**`field.estimate_domain(observations)`** derives these from the first observation’s `CRVAL*` / CD matrix and from spectral bounds across all observations.

## Models

Use explicit model kinds when you know the representation you want:

```python
from torchsky import field

model = field.model("sky_lambda_feature_field", domain)
```

Concrete model classes remain available under `torchsky.map.field` (for
example `torchsky.map.field.core.SineImplicitField`,
`torchsky.map.field.core.FourierMLPField`, and
`torchsky.map.field.core.SkyLambdaFeatureField`); `torchsky.field` exposes the
model/fit/render/sample composition surface and recommended presets.

Presets are a convenience layer, not the core API:

```python
model = field.preset("balanced", domain)
```

Current presets:

| Preset | Underlying kind | When to use it |
|--------|-----------------|----------------|
| `fast` | `sky_encoder_mlp` | Smoke tests and quick baselines. |
| `balanced` | `sky_lambda_feature_field` | Default multi-band field model with explicit sky-lambda features. |
| `spectral` | `hashgrid_spectral_field` | Higher-capacity spectral structure. |
| `operator_constrained` | `operator_constrained_feature_field` | Advanced experiments that attach operator-side constraints. |

Pass explicit keyword overrides to tune the selected model:

```python
model = field.preset("fast", domain, hidden_dim=64, hidden_layers=2)
```

## Expert model kinds

**`field.model(kind: FieldModelKind, domain: field.Domain, **kwargs)`** dispatches to concrete `nn.Module` classes. Kinds include:

| Kind | When to consider it |
|------|---------------------|
| `sine_implicit`, `wire_implicit` | Small-parameter implicit nets (baseline INR). |
| `fourier_mlp` | Fourier-feature MLP on encoded coords. |
| `spectral_basis_field`, `learned_spectral_basis_field` | Explicit spectral basis / learned coefficients; strong when λ sampling is structured. |
| `sky_encoder_mlp`, `sky_encoder_resmlp`, `sky_encoder_bandpass_field` | Sky-coordinate encoders + MLP; good when spatial frequency content varies with patch scale. |
| `hashgrid_mlp`, `hashgrid_resmlp`, `hashgrid_spectral_field`, `hashgrid_conditioned_field` | Hash-grid features; often good capacity/speed tradeoff for imaging patches. |
| `sky_lambda_planes_field`, `sky_lambda_kplanes_field`, `sky_lambda_feature_field`, … | K-plane / λ-atlas style spectral grids for rich multi-λ structure. |
| `bandpass_conditioned_field`, `conditioned_spectral_field` | Conditioning on band metadata. |
| `operator_constrained_feature_field` | Hooks for operator-side structure (advanced). |
| `explicit_spectral_grid_field` | Explicit multiframe spatial/spectral grid baseline for capacity comparisons. |
| `gaussian_basis` | Gaussian-basis sky model; opt-in `splat` renders additive elliptical sources. |
| `sky_lambda_feature_field_hetero` | Sky-λ feature field that also predicts per-pixel variance. |
| `sky_lambda_atlas_field`, `sky_lambda_feature_atlas_field` | Atlas-tiled fields: spatially gated expert fields that share a latent atlas. |
| `spectral_power_foam` | Sky-λ forward model on an adapted spectral Power Foam (scale-adaptive, non-neural alternative). |
| `foam_initialized_inr` | Positive INR correction initialized from a fitted Foam field. |
| `diffuse_foam_hybrid` | Low-capacity diffuse field plus a localized non-negative spectral Foam. |
| `strainer_sky_encoder_mlp` | Optional pretrained trunk path (see module docstrings). |

Exact constructor kwargs vary by kind (hidden sizes, grid resolution, Fourier frequencies, etc.). Start from **`example_map_field_survey_split.py`** and the public constructors, then tune. Scientific model comparisons and real-data tuning are documented in the [harness](https://github.com/astroai/torchsky-harness).

**Capacity hints:** **`spectral_data_budget`** / **`spectral_model_capacity_hints`** in [`spectral_budget.py`](https://github.com/astroai/torchsky/blob/main/src/torchsky/map/field/spectral_budget.py) suggest model families from your observation list.

## Spatially, spectrally, and temporally varying observation models

The experimental learned PSF and background operators use the same forward
path for ordinary neural fields and exact spectral Power Foam. Their capacity
and physically allowed scale ranges are configured independently, while the
active scales are learned: a PSF may change over tens of detector pixels,
hundreds of wavelength units, and minutes, while a sky residual may change over
a much larger spatial scale, a much finer spectral scale, and hours. Do not
share an encoder merely because both operators accept the same coordinates.

```python
import torch

from torchsky.map.operators import (
    AdaptiveSpatioSpectralTemporalEncoder,
    ConditionedBackgroundField,
    ConditionedKernelPSF,
    LearnedBackgroundOp,
    LearnedPSFOp,
    ObservationOperatorBundle,
)
from torchsky.instrument import NIRCAM_UPDEC_BANDS, nircam_imaging_band

nircam_bands = [nircam_imaging_band(name) for name in NIRCAM_UPDEC_BANDS]
initial_fwhm_pix = float(
    torch.tensor([band.empirical_psf_fwhm_pixels for band in nircam_bands]).median()
)

# Coordinates are the actual calibration-star/blank-sky sample locations,
# filter quadrature wavelengths, and exposure mid-times—not domain endpoints.
psf_coordinates = AdaptiveSpatioSpectralTemporalEncoder.from_samples(
    spatial_x=psf_star_x_pix,
    spatial_y=psf_star_y_pix,
    wavelength=psf_wavelength_angstrom,
    time_mjd=psf_time_mjd,
    spatial_bands=4,
    spectral_bands=4,
    temporal_bands=3,
)
background_coordinates = AdaptiveSpatioSpectralTemporalEncoder.from_samples(
    spatial_x=blank_sky_x_pix,
    spatial_y=blank_sky_y_pix,
    wavelength=background_wavelength_angstrom,
    time_mjd=background_time_mjd,
    spatial_bands=4,
    spectral_bands=6,
    temporal_bands=4,
)
# Profile the bank dilation and BIC-initialize gates from independent evidence.
# Residualize other axes first; marginal fits otherwise confound x, y, lambda, t.
psf_coordinates.x_axis.initialize_from_evidence(
    psf_star_x_pix, psf_coefficients_residualized_for_x
)
psf_coordinates.y_axis.initialize_from_evidence(
    psf_star_y_pix, psf_coefficients_residualized_for_y
)
bundle = ObservationOperatorBundle(
    psf=LearnedPSFOp(
        ConditionedKernelPSF(
            psf_coordinates,
            n_basis=6,
            kernel_size=21,
            init_fwhm_pix=initial_fwhm_pix,
        )
    ),
    background=LearnedBackgroundOp(
        ConditionedBackgroundField(background_coordinates),
        coordinate_frame="detector",
    ),
)
result = field.fit(
    observations,
    model,
    observation_operator_bundles=[bundle] * len(observations),
    observation_forward_contexts=contexts,  # time_mjd lives here
    operator_scale_gate_weight=1.0e-4,
    operator_scale_shift_weight=1.0e-4,
)
```

`from_samples` sets the shortest admitted period to twice the largest actual
calibration-coordinate gap and the longest to twice the sampled span. This is
a conservative marginal coverage guard patterned after regular Nyquist
sampling, not an alias-free theorem for irregular samples or a claimed physical
correlation length. It fails if fewer than three distinct samples or no ordered
identifiable interval exists. Joint spatial holes still require a separate
coverage check. An explicitly supplied `sample_spacing` may make the bound more
conservative, but cannot make it finer than the evidence coverage.

Calibration sample discovery, NIRCam exposure acquisition, frozen split policy,
and manifest extraction commands are maintained in the harness's
[calibration evidence guide](https://github.com/astroai/torchsky-harness/blob/main/docs/calibration-evidence.md).
The core API consumes validated tensors from a manifest; it does not acquire
exposures or choose calibration samples.

The harness guide documents the extraction policy, reviewed-catalog option,
source selection, and evidence-product provenance. Keep sample selection fixed
before fitting; train, validation, and test rows must remain disjoint.

The runtime split loader and operator calibration terms remain in torchsky:

```python
from pathlib import Path

from torchsky.calibration_evidence import load_operator_calibration_split
from torchsky.map.operators import BackgroundCalibrationTerm, PSFCalibrationTerm

psf_evidence = load_operator_calibration_split(
    Path("calibration-manifest.json"),
    split="train",
    role="psf",
    detector="NRCA1",
)
psf_term = PSFCalibrationTerm(model=psf_model, **psf_evidence)

background_evidence = load_operator_calibration_split(
    Path("calibration-manifest.json"),
    split="train",
    role="background",
    detector="NRCA1",
)
background_term = BackgroundCalibrationTerm(
    model=background_model,
    **background_evidence,
)
```

The loader returns detector coordinates, wavelength, exposure midpoint, and
sample weights for both roles. PSF rows also resolve hashed kernel and pixel-
weight products; background rows supply target values. When selecting one
exposure, pass both its exact `exposure_id` and `band`; the loader then uses
the exposure-specific calibration split. File-backed manifests verify product
SHA-256 values unless `verify_hashes=False` is explicitly requested.

Within these bounds, each axis learns sparse band gates and a continuous
bounded dilation. `initialize_from_evidence` jointly profiles all harmonic
bands with an intercept and trend, selects the starting dilation by weighted
residual, and initializes each gate from a full-versus-band-ablated BIC. It is
an initializer, not a result, and must use calibration/cross-fit evidence rather
than the unpartitioned science evaluation pixels.

Learned gates, feature wavelengths, gate-weighted scales, and bounds are
returned under `result.diagnostics["operator_scale_diagnostics"]`.

The gate-weighted scale is a parameter diagnostic, not an empirical correlation
length: downstream network weights can still suppress a band. Use held-out
axis-band ablations or output Jacobians before interpreting physical
variability scales.

`ConditionedKernelPSF` learns positive unit-sum basis kernels and neural convex
mixture weights at every source position, wavelength, and time. Source-side
mixing makes the non-stationary convolution flux conserving away from real
detector boundaries. Each monochromatic scene is convolved before spectral
quadrature. `ConditionedBackgroundField` is signed and additive; it may model a
residual around explicit background estimates already attached to each
observation. Detector-coordinate models use `detector_x0_pix` and
`detector_y0_pix`, which are updated automatically by training crops.

Image-shaped asymmetric PSF stamps retain their measured orientation in
rendering. Source-flux conservation does not imply a constant uniform-sky
response. See [PSF response conventions](psf-response-conventions.md) for these
invariants, detector-boundary losses and migration of old asymmetric weights.

Use evidence-space calibration terms and separate optimizer invocations to
make the staged sequence explicit:

```python
from torchsky.map.operators import (
    BackgroundCalibrationTerm,
    PSFCalibrationTerm,
    fit_operator_calibration_terms,
    set_field_training_stage,
)

psf_term = PSFCalibrationTerm(
    psf_model,
    detector_x=star_x,
    detector_y=star_y,
    wavelength=star_wavelength,
    time_mjd=star_time,
    target_kernels=measured_epsfs,
    sample_weight=star_reliability,
)
background_term = BackgroundCalibrationTerm(
    background_model,
    detector_x=blank_x,
    detector_y=blank_y,
    wavelength=blank_wavelength,
    time_mjd=blank_time,
    target_values=blank_values,
    sample_weight=blank_inverse_variance,
)

# Optional evidence-only warm start; it never sees science-scene pixels.
fit_operator_calibration_terms(
    [background_term], steps=200, lr=1e-3,
    gate_weight=1e-4, scale_shift_weight=1e-4,
)
fit_operator_calibration_terms(
    [psf_term], steps=200, lr=1e-3,
    gate_weight=1e-4, scale_shift_weight=1e-4,
)

set_field_training_stage(model, bundles, "background")
field.fit(
    observations,
    model,
    observation_operator_bundles=bundles,
    operator_calibration_terms=[background_term],
)
set_field_training_stage(model, bundles, "psf")
field.fit(
    observations,
    model,
    observation_operator_bundles=bundles,
    operator_calibration_terms=[psf_term],
)
set_field_training_stage(model, bundles, "joint")
result = field.fit(
    observations,
    model,
    observation_operator_bundles=bundles,
    operator_calibration_terms=[psf_term, background_term],
)
```

PSF terms compare positive unit-sum model kernels to normalized measured ePSFs,
so arbitrary stamp-flux scaling cannot change the target. Background terms
compare signed predicted values. Sample weights must be fixed from calibration
uncertainty or quality before the science fit; do not tune them on the withheld
evaluation exposures. `result.diagnostics["operator_calibration_diagnostics"]`
persists the final evidence-space losses. A low calibration loss is not a
held-out-prediction result.

Call `operator_calibration_score_report(validation_terms, split="validation")`
and then the same function on untouched test terms. The report includes PSF
centroid and encircled-energy errors and background bias/MAE in addition to the
optimization loss. Never label training-term diagnostics as held-out scores.

These models are deliberately constrained, but the inverse problem is still
not identifiable if scene, PSF, and background all receive unconstrained
capacity simultaneously. The recommended scientific sequence is:

1. Select the `scene` stage and anchor it with calibrated exposures and an
   explicit nominal PSF and background; retain held-out exposures and isolated
   stars.
2. Fit only a low-spatial-frequency signed background residual, with scales set
   from blank-sky structure and no compact-source capacity.
3. Fit the positive normalized PSF bases and their conditioning on isolated
   stars or counterfactual evidence stamps; validate encircled energy and
   residuals by detector position, wavelength, and time.
4. Jointly refine at a lower learning rate with priors toward the calibrated
   operators. Select capacity only from held-out exposure likelihood, injected
   flux/color recovery, and blank-sky false positives.

Value losses are supported. Distribution rendering and Gaussian NLL currently
fail closed for learned PSF/background bundles because a spatially varying PSF
requires covariance propagation with cross-basis terms; silently applying the
fixed-kernel variance rule would be incorrect.

The original `SpatioSpectralTemporalEncoder` remains available as a fixed-scale
control for ablations. Full-sky component separation must use a future
spherical harmonic/needlet adaptive encoder rather than treating RA/Dec as a
Cartesian plane across seams and poles.

## Fitting: `field.fit`

**`field.fit(observations, model, steps=200, lr=..., loss_kind=..., ...)`** supports:

- Losses: `mse`, `charbonnier`, `huber`, `gaussian_nll` (`FieldLossKind`).
- Per-step observation subsampling, patches, curricula, spectral anchors, component decomposition anchors, EMA, LR schedules, etc.

Pass **`observation_forward_contexts`** (same length as `observations`) for models that take **`forward_context`**.

Returns **`FieldFitResult`** (`model`, `loss_history`, `final_loss`,
`diagnostics`). The diagnostics dictionary records:

- `spatial_spectral_leakage` and `gradient_correlation` from local autograd probes.
- `reduced_chi2` and `psf_alignment` from rendered observation residuals.
- `quality_metrics` with weighted RMSE/MAE, residual SNR, residual bias/std,
  prediction-vs-target sharpness, multiscale residual RMSE, support fraction,
  and effective weight.
- Opt-in residual products and scorecards expose rendered predictions,
  residual maps, support masks, band-group summaries, and exact worst-observation
  extrema without changing the default scalar diagnostics.
- `uncertainty_stats` for support vs out-of-support predictive uncertainty when
  the model exposes `forward_distribution`.
- `performance_profile` with total fit time, mean step time, pixel throughput,
  and CUDA peak memory when available.

`final_loss` and diagnostics use the same exposure poses passed to the optimizer,
so bundle-adjusted fits are evaluated in the refined coordinate frame.

For sharper data-consistent fits, set **`gradient_residual_weight > 0`**. This
adds a finite-difference residual term on the rendered observation image, so it
penalizes mismatched edges after the same WCS/bandpass/PSF forward model instead
of post-hoc sharpening the latent field.

`pretrain_strainer_trunk_synthetic(...)` returns a small timing profile with
total duration, mean step time, sample throughput, and CUDA peak memory; it
returns `None` only when `steps=0`.

## Rendering

- **`field.render(model, observation, ...)`** — mean prediction on the observation’s pixel grid (bandpass integral + PSF + backgrounds via the operator bundle).

  Optional **`exposure_pose`**: an **`ExposurePose`** module applied to sky coordinates before `world_to_pixel` (see [`exposure_pose.py`](https://github.com/astroai/torchsky/blob/main/src/torchsky/map/exposure_pose.py)). Use **`None`** for fixed WCS.

  Set **`spatial_sampling="gauss_legendre_2x2"`** to render the detector-pixel
  average from four Gauss–Legendre samples instead of the default pixel-centre
  value. This is useful for compact or undersampled structure and composes with
  spectral integration, exposure pose, PSF, and background operators. Combine
  it with **`spatial_tile_hw=(h, w)`** to bound activation memory; the four
  subpixel world grids are generated one tile at a time.

- **`field.render(model, target="cube", ...)`** — samples the field on a 3D grid for export.

- **`field.render(model, target="healpix", ...)`** / **`field.render(model, target="cube_via_healpix", ...)`** — spherical products.

For cube export to FITS, use **`torchsky.spectral.write_cube_fits`** ([spectral.md](spectral.md)).

## Residual Quality

Use **`field.quality(model, observations, ...)`** after a fit to compute rendered
residual metrics across observations. For single images, **`field.metrics(pred, target, weight=..., mask=...)`**
returns weighted residual, SNR, sharpness, and multiscale residual metrics. By
default the report includes `residual_rmse_scale_1`, `_2`, and `_4`, which measure
coherent residuals after exact block averaging with the same support/weight mask.
Non-finite prediction, target, weight, or masked pixels are excluded from these
diagnostics. **`field.residual(model, observation, ...)`** returns the rendered
residual map (`prediction - image`) for quicklooks or downstream diagnostics.

For structured QA artifacts, use **`field.residual_product(...)`** on existing
prediction/target tensors or **`field.render_residual_product(model,
observation, ...)`** after rendering. The returned product carries prediction,
target, residual, weight, support mask, band name, and scalar metrics together.
Pass **`include_scorecard=True`** and/or **`include_residual_products=True`** to
**`field.quality(...)`** to include band-group aggregate metrics, exact extrema
for worst RMSE / lowest SNR / lowest support, and per-observation residual
products.

For adaptive local-capacity experiments, **`field.residual_tiles(product,
tile_hw=(h, w), ...)`** ranks fixed-size residual-product tiles by exact weighted
residual energy. This is an evidence-selection primitive for later sparse local
basis or feature-tile models: tile size, maximum tile count, and minimum support
fraction are explicit inputs, and no source labels are inferred.

To turn that evidence into an explicit local-capacity layer, wrap an existing
diffuse model with **`field.residual_tile_basis(...)`**. It builds a signed
additive `ResidualTileBasis` field whose Gaussian atoms are initialized from the
selected residual tiles and observation WCS. The number of atoms is bounded by
`max_tiles`; tile geometry comes from the WCS; amplitudes initialize from
weighted mean residuals. This remains an experimental opt-in correction layer,
not a default model path.

## Bundle adjustment / alignment (gauge-aware)

**`ExposurePose`** ([`map/exposure_pose.py`](https://github.com/astroai/torchsky/blob/main/src/torchsky/map/exposure_pose.py)) adds a small **learnable** tangent-plane residual per exposure: translation (`delta_ra_deg`, `delta_dec_deg`), in-plane rotation (`delta_theta_rad`), and log-scale (`delta_log_scale`). Defaults are zero ⇒ identity.

For the common case, pass **`refine_poses=True`** to **`field.fit(...)`** /
**`fit_field_model(...)`**. This creates one pose per observation, freezes the
first pose by default to fix the gauge, includes the rest in the optimizer, and
returns the tuple on **`FieldFitResult.exposure_poses`**. The result diagnostics
also include a compact **`pose_refinement`** block with final pose parameters.

**Gauge fixing:** multiple exposures can absorb a global shift. Typical pattern:

- Attach one **`ExposurePose`** per exposure (or only to exposures you want to refine).
- **Freeze** the pose on a reference exposure (e.g. index 0) with **`freeze_pose(pose)`** so the solve is anchored.

Document optimization loops (joint field + poses) in your training script; pass
the matching pose into **`field.render(...)`** for each observation.

## Native astrometry, structured noise, and the Foam → INR ladder

`FieldObservation.detector_to_sky` accepts a forward-only torch transform in
full-detector zero-based pixels. `sample_gwcs_detector_to_tangent(...)` samples
an external GWCS/CRDS detector→V2/V3 callable on an adaptively refined grid and
fails unless independent random, edge, and optional chip-boundary points meet
the declared p99/max mas limits. Compose the frozen table with
`TangentPlaneAffine`, an optional affine-free residual grid, and a frozen
V2/V3→sky attitude using `DetectorToSkyTransform`. Crops retain detector
origins; a custom transform is never cached while it may carry gradients.

For detector noise, store `err` or the `var_poisson`, `var_rnoise`, and
`var_flat` planes on each observation. `contamination_prior` holds soft CRF
outlier probabilities; permanent DQ failures remain in `mask`.
`fit_structured_noise_crossfit(...)` fits `StructuredDetectorNoise` only from predictions
made without the exposure being calibrated, then freezes the result. Attach it
as `StructuredNoiseOp(model, frozen=True)` and use
`loss_kind="structured_nll"` for the scene fit. This likelihood intentionally
rejects pixel/patch/context approximations because they change the amplifier
row covariance.

The transparent reconstruction ladder is:

1. fit `SpectralPowerFoamField` on native observations and check the exact
   detector-pixel handoff. Set `log_intensity_spectra=True` when Foam is the
   positive INR initializer: each cell then represents
   `exp(Legendre coefficients)` and point/exact renders share those positive
   sampled spectra. This mode currently rejects affine-in-cell spectra because
   their nonlinear polygon integral is not yet exact;
2. wrap the accepted control in `FoamInitializedINRField`, whose log-intensity
   residual is zero-initialized and remains positive;
3. keep the fixed PSF, then calibrate `ExtendedResidualKernelPSF`, a fixed
   core+wing baseline with a bounded positive, unit-sum, centroid-anchored
   residual;
4. freeze astrometry, noise, and PSF before producing the final scene.

Use `completeness_limit_magnitude(...)` for bracketed m50/m80 estimates and
`paired_bootstrap_improvement(...)` for the predeclared held-out interval.
Neither helper extrapolates a depth claim beyond the injection grid.

## Synthetic minimal example

```python
import torch
from torchsky import field

header = {
    "NAXIS": 2,
    "NAXIS1": 32,
    "NAXIS2": 32,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5,
    "CRPIX2": 16.5,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CDELT1": -0.05 / 3600.0,
    "CDELT2": 0.05 / 3600.0,
}
obs = field.Observation(
    image=torch.zeros(32, 32),
    header=header,
    spectral_min=5000.0,
    spectral_max=5000.0,
    transmission_center=5000.0,
)
domain = field.estimate_domain([obs])
model = field.preset("balanced", domain)
field.fit([obs], model, steps=50, lr=1e-2)
pred = field.render(model, obs)
```

## Multi-band / hyperspectral notes

- Each band is usually its own **`field.Observation`** with distinct spectral metadata; **`field.fit`** aggregates losses across all of them.
- Tabulated throughput (`transmission_wavelength`, `transmission_throughput`) is preferred when combining HST/JWST filters with different shapes.
- Build a classical baseline first with **`coadd_multiband`** so INR residuals can be interpreted against a conventional WCS-aligned stack. It is a comparator, not an initializer for the native-exposure production fit.
- For a drizzle-like baseline, use **`coadd_cube(...,
  resampling_kernel="drizzle_square", drizzle_pixfrac=...)`** or the
  `scripts/build_smacs0723_stack_baseline.py --resampling-kernel drizzle_square`
  helper.
- Make PNG quicklooks before and after fitting so axis flips, misregistration,
  artifacts, and tile-boundary problems are caught visually.
- For writing a dense λ cube after fitting, render with **`field.render(model, target="cube", ...)`** and **`write_cube_fits`**.
- For larger contiguous fields, keep training and rendering chunked: use
  **`training_patch_size`** / **`shared_sky_patch_sampling`** during fitting and
  **`spatial_tile_hw`** / **`spectral_chunk`** when calling **`field.render(..., target="cube")`**.

## See also

- [implicit_forward.md](implicit_forward.md) — `SurveyDataTerm`, `KinematicState`, Charbonnier scaffold.
- [spectral.md](spectral.md) — spectra and cube containers / I/O.
- [map.md](map.md) — `ImageObservation`, coadds, `render_field_product`.

## Frequency-adaptive candidates

Three experimental models implement frequency-control ideas from the INR
literature. They share the `torchsky.field` model contract but are **not**
production presets until they win an equal-budget, held-out comparison.

- `BaconField` (`bacon`): multiplicative frozen sine filters with a
  non-negative sum-of-squares readout. The declared bandwidths describe output
  angular support; latent filters receive half that budget because the
  non-negative readout doubles Fourier support.
- `SASNetField` (`sasnet`): spatially localized, frequency-adaptive model with a
  learned mask over a frozen frequency bank. It is not strictly band-limited;
  the finest frequency and hash resolution are capped from the caller-supplied
  pixel footprint and PSF support.
- `FlairField` (`flair_bla`): real-feature band-limited-layer approximation with
  Gaussian envelopes and explicit period/sigma/shift parameters.

The models are exposed by the `torchsky.map.field` factory and serialized like
other field models. Admission to a preset still requires the same
serialization/device/gradient contract, shared operator tests, and held-out
scientific recovery as the rest of the model zoo.
