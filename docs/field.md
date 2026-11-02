# Fields

!!! warning "Experimental surface"
    `torchsky.field` is experimental. Names, defaults, and serialized formats may
    change between minor releases. The geometric and mapmaking layers it builds on
    ([WCS](wcs.md), [coadds](map.md)) are stable.

A field is a continuous model of the sky, \(F_\theta(\mathrm{RA}, \mathrm{Dec}, \lambda)\),
fitted to multi-band 2D observations by gradient descent through a differentiable
forward model: sky, bandpass integral, PSF, background, and pixel sampling. Use
the concise `torchsky.field` namespace for new code. The implementation lives in
`torchsky.map.field`.

The scientific formulation (inverse problem, exact hard-cell Power Foam, the
quantitative depth objective) is in [Field formulation](field-formulation.md).
For classical stacks see [coadds and resampling](map.md). Native stamp calibration
and small likelihood queries, which need no fitted field, are in
[Native likelihood](native-likelihood.md).

## Core Workflow

1. Build one `field.FieldObservation` per band or exposure: image tensor,
   FITS-like `header` dict, spectral limits or tabulated throughput, optional
   `weight`, `mask`, and `PSFSpec`.
2. `field.estimate_domain(observations)` returns the `Domain` that normalizes sky
   and wavelength coordinates (or set it by hand).
3. Build a model with `field.model(kind, domain, **kwargs)`, or
   `field.preset(name, domain, **kwargs)` when a default recipe is enough.
4. `field.fit(observations, model, ...)` runs the built-in optimizer loop, or
   compose your own with `field.render` and `field.sample`.
5. `field.render` produces observation predictions, cubes, or HEALPix products;
   `field.sample` evaluates explicit sky samples.

<!-- docs: run -->
```python
import torch

from torchsky import field


def tangent_header(n: int = 32) -> dict:
    return {
        "NAXIS": 2, "NAXIS1": n, "NAXIS2": n,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": n / 2 + 0.5, "CRPIX2": n / 2 + 0.5,
        "CRVAL1": 150.0, "CRVAL2": 2.0,
        "CDELT1": -0.05 / 3600.0, "CDELT2": 0.05 / 3600.0,
    }


gen = torch.Generator().manual_seed(0)
yy, xx = torch.meshgrid(torch.arange(32.0), torch.arange(32.0), indexing="ij")
blob = torch.exp(-((xx - 16) ** 2 + (yy - 16) ** 2) / (2 * 3.0**2))

observations = [
    field.FieldObservation(
        image=scale * blob + 0.01 * torch.randn(32, 32, generator=gen),
        header=tangent_header(),
        spectral_min=lo,
        spectral_max=hi,
        transmission_center=(lo + hi) / 2,
        weight=torch.ones(32, 32),
        band_name=name,
    )
    for scale, lo, hi, name in ((1.0, 4500.0, 5500.0, "g"), (0.8, 5500.0, 6500.0, "r"))
]

domain = field.estimate_domain(observations)
model = field.preset("fast", domain, hidden_dim=32, hidden_layers=2)
result = field.fit(observations, model, steps=30, lr=1e-2)
assert result.final_loss < result.loss_history[0]

prediction = field.render(model, observations[0])
assert prediction.shape == (32, 32) and torch.isfinite(prediction).all()
```

CPU and CUDA rendering preserve the WCS precision of absolute sky coordinates.
For custom models, subtract the sky centre and normalize coordinates before
casting local features to the network dtype; casting absolute RA/Dec to float32
first discards subpixel structure. MPS uses float32 sky coordinates and fine
angular structure there has not been verified.

## Observations And Domain

| `FieldObservation` field | Role |
|---|---|
| `image` | `[H, W]` float tensor (target patch) |
| `header` | FITS-like dict (`NAXIS*`, `CTYPE*`, `CRPIX*`, `CRVAL*`, `CD*` or `CDELT*`/`PC*`), the same mapping `torchsky.wcs.WCS(header)` accepts |
| `spectral_min`, `spectral_max` | Band limits (for example angstrom) |
| `transmission_center`, `transmission_sigma` | Gaussian bandpass summary (optional) |
| `transmission_wavelength`, `transmission_throughput` | Tabulated throughput; quadrature integrates against it |
| `weight`, `mask` | Per-pixel weights (often inverse variance) and good-pixel mask |
| `psf_spec` | `PSFSpec.none`, `.gaussian(sigma_pix)`, or `.kernel(kernel2d)` |
| `band_name` | Label for logging and grouping |

`ForwardContext` (survey, instrument, detector, exposure id, `time_mjd`) is not
on the observation: pass it to `fit` and `render` through
`observation_forward_contexts` for context-conditioned models. Helpers such as
`field_observations_from_bin_edges` and `make_tangent_plane_header` build
observation lists from IFU cubes or bin edges. Observation coordinate grids are
cached per image shape and header values, so header edits recompute them.

`Domain` normalizes coordinates inside the network: patch centre
(`center_ra_deg`, `center_dec_deg`), `spatial_scale_deg`, `wavelength_center`,
`wavelength_scale`, and `coord_mode` (`"local_tangent"` by default, or
`"sphere_xyz"`).

## Models

Presets are a convenience layer over explicit kinds:

| Preset | Underlying kind | When to use it |
|---|---|---|
| `fast` | `sky_encoder_mlp` | Smoke tests and quick baselines |
| `balanced` | `sky_lambda_feature_field` | Default multi-band model with explicit sky-lambda features |
| `spectral` | `hashgrid_spectral_field` | Higher-capacity spectral structure |
| `operator_constrained` | `operator_constrained_feature_field` | Experiments with operator-side constraints |

Kinds accepted by `field.model` include implicit nets (`sine_implicit`,
`wire_implicit`), Fourier and hash-grid MLPs (`fourier_mlp`, `hashgrid_*`),
sky-encoder MLPs (`sky_encoder_*`), sky-lambda plane and feature fields
(`sky_lambda_*`), explicit spectral bases and grids (`spectral_basis_field`,
`explicit_spectral_grid_field`), `gaussian_basis`, and the non-neural
`spectral_power_foam` with its hybrids `foam_initialized_inr` and
`diffuse_foam_hybrid`. The authoritative list is `field.FIELD_MODEL_KINDS`, and
`field.spectral_data_budget` suggests families from your observation list. Model
comparisons on real data live in the
[harness](https://github.com/astroai/torchsky-harness).

Three frequency-adaptive candidates (`bacon`, `sasnet`, `flair_bla`) share the
model contract but are not presets until they win an equal-budget held-out
comparison. They are serialized like other models and admission requires the
same serialization, device, gradient, and operator tests.

## Fitting

`field.fit` supports `mse`, `charbonnier`, `huber`, and `gaussian_nll` losses,
per-step observation subsampling, patches and curricula, spectral anchors, EMA,
and learning-rate schedules. It returns a `FitResult` with `model`,
`loss_history`, `final_loss`, and `diagnostics`: spectral leakage and gradient
correlation probes, `reduced_chi2`, weighted quality metrics, uncertainty
statistics when the model exposes `forward_distribution`, and a performance
profile (throughput, CUDA peak memory). Diagnostics use the same refined exposure
poses passed to the optimizer. `gradient_residual_weight > 0` adds an
edge-sharpness residual evaluated after the full forward model.

For larger fields keep training and rendering chunked: `training_patch_size` and
`shared_sky_patch_sampling` while fitting, `spatial_tile_hw` and `spectral_chunk`
when rendering cubes.

### Pose refinement

`refine_poses=True` creates one learnable tangent-plane `ExposurePose` per
observation (translation, rotation, log-scale), freezes the first to fix the
gauge, and returns them on `FitResult.exposure_poses` with a `pose_refinement`
diagnostic block. Without a frozen reference, exposures can absorb a global shift.

## Rendering And Quality

- `field.render(model, observation)` returns the mean prediction on the
  observation grid: bandpass integral, PSF, and backgrounds via the operator
  bundle. `spatial_sampling="gauss_legendre_2x2"` averages four samples per pixel
  for compact structure; combine it with `spatial_tile_hw=(h, w)` to bound memory.
- `target="cube"` samples a 3D grid; `target="healpix"` and
  `target="cube_via_healpix"` produce spherical products. Write cubes with
  `torchsky.spectral.write_cube_fits` ([Spectra](spectral.md)).
- `field.quality(model, observations)` computes rendered residual metrics;
  `field.metrics(pred, target, weight=..., mask=...)` handles single images and
  reports multiscale residual RMSE after exact block averaging.
  `field.residual` returns `prediction - image`; `field.residual_product` and
  `field.render_residual_product` bundle prediction, target, weight, support, and
  metrics for QA.
- `field.residual_tiles` ranks fixed-size residual tiles by weighted energy and
  `field.residual_tile_basis` turns the selection into an explicit signed
  additive local-capacity layer. Both are experimental opt-in tools and infer no
  source labels.

## Learned Observation Operators

Learned PSF and background operators run through the same forward path for
neural fields and exact Power Foam. PSF structure (tens of pixels, hundreds of
wavelength units, minutes) and background structure (much larger in space, finer
in wavelength, hours) have independent capacity and scale ranges, so do not share
one encoder merely because both take the same coordinates.

```python
# fragment: calibration arrays come from your own evidence manifest
from torchsky.map.operators import (
    AdaptiveSpatioSpectralTemporalEncoder,
    ConditionedBackgroundField,
    ConditionedKernelPSF,
    LearnedBackgroundOp,
    LearnedPSFOp,
    ObservationOperatorBundle,
)

psf_coordinates = AdaptiveSpatioSpectralTemporalEncoder.from_samples(
    spatial_x=psf_star_x_pix, spatial_y=psf_star_y_pix,
    wavelength=psf_wavelength_angstrom, time_mjd=psf_time_mjd,
    spatial_bands=4, spectral_bands=4, temporal_bands=3,
)
bundle = ObservationOperatorBundle(
    psf=LearnedPSFOp(ConditionedKernelPSF(psf_coordinates, n_basis=6, kernel_size=21)),
    background=LearnedBackgroundOp(
        ConditionedBackgroundField(background_coordinates), coordinate_frame="detector"
    ),
)
result = field.fit(
    observations, model,
    observation_operator_bundles=[bundle] * len(observations),
    observation_forward_contexts=contexts,
)
```

`from_samples` sets the shortest admitted period to twice the largest sample gap
and the longest to twice the sampled span, a conservative marginal coverage
guard rather than an alias-free theorem; it fails with fewer than three distinct
samples. Each axis learns sparse band gates and a bounded dilation, initialized
with `initialize_from_evidence` from calibration (not science) pixels.
`result.diagnostics["operator_scale_diagnostics"]` reports learned gates and
scales; these are parameter diagnostics, not empirical correlation lengths.

`ConditionedKernelPSF` learns positive unit-sum basis kernels with neural convex
weights per position, wavelength, and time, so flux is conserved away from
detector boundaries. `ConditionedBackgroundField` is signed and additive. Value
losses are supported; distribution rendering and Gaussian NLL fail closed for
learned PSF and background bundles because a spatially varying PSF needs
covariance propagation with cross-basis terms. See
[PSF response conventions](psf-response-conventions.md) for orientation and
boundary invariants.

The recommended sequence, because scene, PSF, and background cannot all be
free at once:

1. Fit the `scene` stage with calibrated exposures and an explicit nominal PSF
   and background; keep held-out exposures and isolated stars aside.
2. Fit only a low-spatial-frequency signed background residual.
3. Fit the positive normalized PSF bases on isolated stars or counterfactual
   stamps; validate encircled energy by detector position, wavelength, and time.
4. Refine jointly at a lower learning rate with priors toward the calibrated
   operators, choosing capacity from held-out likelihood, injected flux and color
   recovery, and blank-sky false positives.

Evidence-space terms (`PSFCalibrationTerm`, `BackgroundCalibrationTerm`,
`fit_operator_calibration_terms`, `set_field_training_stage`, and the
`load_operator_calibration_split` loader) implement the staged fit. Fix sample
weights from calibration uncertainty before the science fit, never from withheld
exposures, and never report training-term diagnostics as held-out scores. The
calibration evidence workflow is documented in the harness's
[calibration guide](https://github.com/astroai/torchsky-harness/blob/main/docs/calibration-evidence.md).

## Native Astrometry, Structured Noise, And The Foam To INR Ladder

`FieldObservation.detector_to_sky` accepts a forward-only torch transform in
full-detector zero-based pixels; `sample_gwcs_detector_to_tangent` samples a
GWCS/CRDS callable on an adaptive grid and fails unless independent test points
meet declared milliarcsecond limits. For detector noise, store `err` or the
`var_poisson`, `var_rnoise`, and `var_flat` planes on each observation;
`field.fit_noise_crossfit` fits structured detector noise from predictions made
without the exposure being calibrated, then freezes it (use
`loss_kind="structured_nll"`).

The reconstruction ladder is: fit `SpectralPowerFoamField` on native
observations and check the exact detector-pixel handoff; wrap the accepted
control in `FoamInitializedINRField` (zero-initialized, positive log-intensity
residual); keep the fixed PSF, then calibrate `ExtendedResidualKernelPSF`;
finally freeze astrometry, noise, and PSF before producing the scene.
`completeness_limit_magnitude` and `paired_bootstrap_improvement` give bracketed
depth estimates and held-out intervals without extrapolating beyond the injection
grid.

## Implicit Forward Scaffold

`torchsky.map.field.implicit_forward` is a small, explicit alternative to
`field.fit` when you want to write your own training loop: a shared scene
\(F_\theta\), per-observation rendering through `field.render`, optional
spectral kinematics, and a modular loss \(\sum_s w_s L_s\). It is a module, not part
of the public `torchsky.field` namespace.

- `KinematicState(one_plus_z=...)` is a user-chosen linear map
  \(\lambda_\mathrm{obs} = (1+z)\,\lambda_\mathrm{emit}\), with no embedded
  cosmology. With `rest_frame_scene=True`, observation spectral metadata is
  scaled so a rest-frame scene is queried correctly at observed bands.
- `SurveyDataTerm(name, observations, loss_weight, ...)` groups observations with
  optional per-row kinematics and `ForwardContext`.
- `predict_observation_mean` wraps observation rendering;
  `observation_charbonnier_loss` and `multi_survey_charbonnier_loss` combine
  `mask` and `weight` into a robust Charbonnier loss summed over survey terms.

<!-- docs: run -->
```python
import torch

from torchsky import field
from torchsky.map.field.implicit_forward import SurveyDataTerm, multi_survey_charbonnier_loss

header = {
    "NAXIS": 2, "NAXIS1": 16, "NAXIS2": 16,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 8.5, "CRPIX2": 8.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CDELT1": -0.05 / 3600.0, "CDELT2": 0.05 / 3600.0,
}
observation = field.FieldObservation(
    image=torch.rand(16, 16), header=header,
    spectral_min=5000.0, spectral_max=5000.0, transmission_center=5000.0,
    weight=torch.ones(16, 16),
)
domain = field.estimate_domain([observation])
model = field.preset("fast", domain, hidden_dim=16, hidden_layers=1)

term = SurveyDataTerm(name="survey", observations=(observation,), loss_weight=1.0)
loss = multi_survey_charbonnier_loss(model, [term])
loss.backward()
assert torch.isfinite(loss)
```

## Multi-Band Notes

- Each band is its own `FieldObservation` with distinct spectral metadata;
  `field.fit` aggregates losses across all of them. Prefer tabulated throughput
  when combining filters with different shapes.
- Build a classical baseline first with `coadd_multiband` or
  `coadd_cube(..., resampling_kernel="drizzle_square")` so residuals can be read
  against a conventional WCS-aligned stack. It is a comparator, not an
  initializer for the native-exposure fit.
- Make quicklooks before and after fitting to catch axis flips,
  misregistration, and tile-boundary artifacts.

## Related

- [Scenes and operators](scene.md), [Native likelihood](native-likelihood.md),
  [PSF conventions](psf-response-conventions.md), [Encoders and ML](encoders.md).
- Examples:
  [survey split](published-examples/survey_split.py),
  [neural multiband field](published-examples/neural_field_multiband.py),
  [native operators](published-examples/native_operators.py).
- Harness notebooks:
  [synthetic field fit](https://github.com/astroai/torchsky-harness/blob/main/notebooks/01_field_fit_synthetic.ipynb)
  and [evaluation tools](https://github.com/astroai/torchsky-harness/tree/main/notebooks).
