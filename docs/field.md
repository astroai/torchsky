# Continuous fields (`torchsky.field`)

This guide covers **experimental** spectral neural fields used to fit a continuous scene
\(F_\theta(\mathrm{RA}, \mathrm{Dec}, \lambda)\) to multi-band 2D observations.
Use the concise **`torchsky.field`** namespace for new code; the older
`torchsky.map.field` functions remain the implementation layer for map workflows.

For map products and classical stacks, see [map.md](map.md). For the Charbonnier / multi-survey scaffold around field rendering, see [implicit_forward.md](implicit_forward.md).

## Core workflow

1. Build one **`field.Observation`** per band (or per exposure): image tensor, FITS-like **`header`** dict (WCS), spectral limits or tabulated throughput, optional **`weight`** / **`mask`**, optional **`PSFSpec`**.
2. Call **`field.estimate_domain(observations)`** to get a **`field.Domain`** (sky center, spatial scale, wavelength center/scale) unless you set the domain by hand.
3. Construct a model explicitly with **`field.model(kind, domain, **kwargs)`**, or use the small convenience **`field.preset(name, domain, **kwargs)`** when a default recipe is enough.
4. **`field.fit(observations, model, ...)`** runs the built-in optimizer loop, or compose your own loop with **`field.render(...)`** / **`field.sample(...)`**.
5. **`field.render(...)`** produces observation predictions, cubes, or HEALPix products; **`field.sample(...)`** evaluates explicit sky-coordinate samples.

Runnable references:

- [examples/example_map_field_survey_split.py](../examples/example_map_field_survey_split.py)
- [examples/notebooks/01_field_fit_synthetic.ipynb](../examples/notebooks/01_field_fit_synthetic.ipynb) (synthetic quickstart)
- [scripts/build_smacs0723_minicutout.py](../scripts/build_smacs0723_minicutout.py) (optional real JWST UPdec mini-cutouts + `manifest.json`)
- [scripts/build_smacs0723_stack_baseline.py](../scripts/build_smacs0723_stack_baseline.py) (classical multiband stack cube baseline)
- [scripts/visualize_smacs0723_products.py](../scripts/visualize_smacs0723_products.py) (PNG quicklooks for cutouts, stack cubes, and residuals)
- [docs/projects/hyperspectral-fusion.md](projects/hyperspectral-fusion.md) (student onboarding)

## `field.Observation`

Defined in [`types.py`](../src/torchsky/map/field/types.py). Important fields:

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

## `field.Domain`

[`FieldDomain`](../src/torchsky/map/field/types.py) normalizes sky + wavelength coordinates inside the network:

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

Common direct constructors are also available with shorter names, for example
**`field.Sine`**, **`field.FourierMLP`**, **`field.HashGridSpectral`**, and
**`field.SkyLambdaFeatures`**.

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
| `strainer_sky_encoder_mlp` | Optional pretrained trunk path (see module docstrings). |

Exact constructor kwargs vary by kind (hidden sizes, grid resolution, Fourier frequencies, etc.). Start from benchmarks or **`example_map_field_survey_split.py`**, then tune.

**Capacity hints:** **`spectral_data_budget`** / **`spectral_model_capacity_hints`** in [`spectral_budget.py`](../src/torchsky/map/field/spectral_budget.py) suggest model families from your observation list.

## Fitting: `field.fit`

**`field.fit(observations, model, steps=200, lr=..., loss_kind=..., ...)`** supports:

- Losses: `mse`, `charbonnier`, `huber`, `gaussian_nll` (`FieldLossKind`).
- Per-step observation subsampling, patches, curricula, spectral anchors, component decomposition anchors, EMA, LR schedules, etc.

Pass **`observation_forward_contexts`** (same length as `observations`) for models that take **`forward_context`**.

Returns **`FieldFitResult`** (`model`, `loss_history`, `final_loss`).

## Rendering

- **`field.render(model, observation, ...)`** — mean prediction on the observation’s pixel grid (bandpass integral + PSF + backgrounds via the operator bundle).

  Optional **`exposure_pose`**: an **`ExposurePose`** module applied to sky coordinates before `world_to_pixel` (see [`exposure_pose.py`](../src/torchsky/map/exposure_pose.py)). Use **`None`** for fixed WCS.

- **`field.render(model, target="cube", ...)`** — samples the field on a 3D grid for export.

- **`field.render(model, target="healpix", ...)`** / **`field.render(model, target="cube_via_healpix", ...)`** — spherical products.

For cube export to FITS, use **`torchsky.spectral.write_cube_fits`** ([spectral.md](spectral.md)).

## Bundle adjustment / alignment (gauge-aware)

**`ExposurePose`** ([`map/exposure_pose.py`](../src/torchsky/map/exposure_pose.py)) adds a small **learnable** tangent-plane residual per exposure: translation (`delta_ra_deg`, `delta_dec_deg`), in-plane rotation (`delta_theta_rad`), and log-scale (`delta_log_scale`). Defaults are zero ⇒ identity.

**Gauge fixing:** multiple exposures can absorb a global shift. Typical pattern:

- Attach one **`ExposurePose`** per exposure (or only to exposures you want to refine).
- **Freeze** the pose on a reference exposure (e.g. index 0) with **`freeze_pose(pose)`** so the solve is anchored.

Document optimization loops (joint field + poses) in your training script; pass
the matching pose into **`field.render(...)`** for each observation.

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
- Build a classical baseline first with **`coadd_image_plane_multiband`** so INR residuals can be interpreted against a conventional WCS-aligned stack.
- For a drizzle-like baseline, use **`coadd_image_plane_cube(...,
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
