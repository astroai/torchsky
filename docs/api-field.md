# Field API reference

`torchsky.field` is the continuous-sky modeling facade: build a differentiable
field, fit it to observations, and render or sample it. The [field guide](field.md)
covers the modeling concepts and the [implicit forward guide](implicit_forward.md)
covers losses and observation physics.

The facade exposes the workflow (`model`, `fit`, `render`, `sample`) and the
objects those calls exchange. Concrete model classes live in
`torchsky.map.field`.

---

## Build, fit, render, sample

```python
import torchsky.field as field

domain = field.estimate_domain(observations)
model = field.model("sky_encoder_resmlp", domain)
result = field.fit(model, observations)
prediction = field.render(model, observations[0])
```

| Name | Contract |
|---|---|
| `estimate_domain(observations, ...)` | Sky + wavelength normalization box from data. |
| `model(kind, domain, **kwargs)` | Build a field model by kind name. |
| `preset(name, domain, **overrides)` | Build a recommended model by preset (`fast`, `balanced`, `spectral`, `operator_constrained`). |
| `fit(model, observations, ...)` | Fit a model; returns `FitResult`. |
| `fit_hybrid(...)` | Diffuse → localized → joint staging on one operator path. |
| `fit_noise_crossfit(...)` | Exposure-noise calibration from leave-one-out predictions. |
| `fit_component_decomposition(...)` | Shared/private component fit. |
| `render(model, observation, *, target="observation")` | Render to an observation, samples, cube, HEALPix, or cube-via-HEALPix. |
| `sample(model, observation, *, ra_deg, dec_deg)` | Evaluate at explicit sky coordinates. |
| `distribution(model, observation, *, target="observation")` | Predictive mean and log-variance. |

`field.render(..., target="samples")` requires an observation for the spectral
context; `target="cube"` and `target="healpix"` use the model's own domain.

---

## Kinds and presets

| Name | Role |
|---|---|
| `FIELD_MODEL_KINDS` | Built-in model kinds (29 entries). |
| `RECOMMENDED_FIELD_MODEL_KINDS` | Current recommended subset. |
| `FIELD_MODEL_PRESET_NAMES` | `fast`, `balanced`, `spectral`, `operator_constrained`. |
| `FIELD_MODEL_PRESETS`, `FieldModelPreset` | Named recipes for users who do not need class details. |
| `get_field_model_preset(name)` | Resolve a preset. |
| `is_recommended_field_model(kind)` | Membership test for the recommended subset. |
| `FieldModelKind`, `PresetName`, `RenderTarget` | Literal types for tooling. |

Recommended kinds are `sky_lambda_feature_field`, `sky_encoder_resmlp`, and
`hashgrid_spectral_field`; the rest of the zoo remains available through
`torchsky.map.field` and string-selected `field.model(...)`.

---

## Observations and context

| Name | Role |
|---|---|
| `Observation` | One band-limited 2D map plus spectral metadata (`FieldObservation`). |
| `Domain` | Sky + wavelength normalization box (`FieldDomain`). |
| `ForwardContext` | Exposure/instrument context (PSF, spectral response, detector state). |
| `PSFSpec` | Public PSF description for observations and target products. |
| `field_observations_from_bin_edges`, `field_observations_from_bin_centers` | Build observations from binned spectra. |
| `field_observations_from_ifu_cube` | Build observations from an IFU/image-spectral cube. |
| `ifu_cube_spectral_to_leading` | Move the spectral axis to index 0. |
| `make_tangent_plane_header` | Minimal `RA---TAN`/`DEC--TAN` header for rendered products. |

---

## Results, decomposition, and quality

| Name | Role |
|---|---|
| `FitResult` | Fitted model, loss history, final loss, and diagnostics. |
| `ArtifactManifest`, `LoadedArtifact`, `save_artifact`, `load_artifact` | Model artifacts with provenance. |
| `FieldComponentDecomposition`, `component_decomposition` | Explicit shared/private decomposition over models. |
| `ResidualProduct`, `ResidualTile`, `ResidualTileBasisSpec` | Residual products, tiled energy summaries, and basis specs. |
| `residual_product`, `render_residual_product`, `residual`, `residual_tiles`, `residual_tile_basis`, `residual_tile_basis_spec` | Residual inspection and basis construction helpers. |
| `metrics`, `quality` | Weighted residual/SNR/sharpness and multiscale metrics. |

---

## Spectral capacity planning

| Name | Role |
|---|---|
| `SpectralDataBudget` | Aggregated spectral complexity from observations. |
| `spectral_data_budget(observations)` | Build a budget summary. |
| `SpectralCapacityProfile`, `SpectralModelCapacityHints` | Suggested encoder capacity from global span and finest scale. |
| `suggest_num_spectral_samples(budget)` | Recommended `num_spectral_samples`. |
| `suggest_spectral_model_capacity(budget)` | Recommended encoder width and family hints. |

See also: [encoders](encoders.md), [native likelihood](native-likelihood.md),
and [ML/runtime](ml.md) for datasets and device handling.
