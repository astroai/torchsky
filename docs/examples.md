# Examples

This page indexes runnable scripts under [examples/](../examples), grouped to
match the domain guides.

## I/O Boundary

| Script | What it demonstrates |
|---|---|
| Legacy image examples | These pre-split scripts are pending migration to `torchfits`; use the examples in the `torchfits` repository for FITS read/cutout/MEF behavior. |

## Tables

| Script | What it demonstrates |
|---|---|
| Legacy table examples | These pre-split scripts are pending migration to `torchfits`; torchsky examples should begin with decoded tensor or Arrow batches. |

## WCS

| Script | What it demonstrates |
|---|---|
| [`example_wcs_transform.py`](../examples/example_wcs_transform.py) | Pixel-to-world/world-to-pixel transforms for images, spectra, cubes, independent `WAVE-TAB` axes, grism axes, and temporal light-time corrections. |

## Sphere And HEALPix

| Script | What it demonstrates |
|---|---|
| [`example_healpix.py`](../examples/example_healpix.py) | `ang2pix`, `pix2ang`, disc queries, and map sampling. |
| [`example_spectral_sht.py`](../examples/example_spectral_sht.py) | Spherical harmonic transforms, smoothing, and spectra. |

## Map

| Script | What it demonstrates |
|---|---|
| [`example_map_healpix_inverse.py`](../examples/example_map_healpix_inverse.py) | Sphere-to-image inverse solve with HEALPix coefficients. |
| [`example_map_healpix_multi_cutout_inverse.py`](../examples/example_map_healpix_multi_cutout_inverse.py) | Joint HEALPix solve across multiple WCS cutouts. |
| [`example_map_healpix_mixed_constraints.py`](../examples/example_map_healpix_mixed_constraints.py) | Mixed-constraint HEALPix solve with all-sky anchor. |
| [`example_map_harmonic_inverse.py`](../examples/example_map_harmonic_inverse.py) | Image-to-`alm` inverse solve. |
| [`example_map_harmonic_cl_prior.py`](../examples/example_map_harmonic_cl_prior.py) | Harmonic solve with a `C_ell` prior. |
| [`example_map_harmonic_multi_cutout_inverse.py`](../examples/example_map_harmonic_multi_cutout_inverse.py) | Joint image-to-`alm` solve across cutouts. |
| [`example_map_harmonic_mixed_constraints.py`](../examples/example_map_harmonic_mixed_constraints.py) | Harmonic mixed-constraint solve. |
| [`example_map_harmonic_survey_split.py`](../examples/example_map_harmonic_survey_split.py) | Shared-versus-survey-private harmonic decomposition. |
| [`example_map_harmonic_learned_survey_instrument_split.py`](../examples/example_map_harmonic_learned_survey_instrument_split.py) | Learned survey/instrument mixer. |
| [`example_field_multiband_quality.py`](../examples/example_field_multiband_quality.py) | Tiny multiband field fit with residual SNR, sharpness, and multiscale quality metrics. |
| [`example_map_field_survey_split.py`](../examples/example_map_field_survey_split.py) | Train-time field decomposition. |
| [`example_map_teb_joint_cl_prior.py`](../examples/example_map_teb_joint_cl_prior.py) | Joint T/E/B harmonic solve. |
| [`example_map_spin_harmonic_inverse.py`](../examples/example_map_spin_harmonic_inverse.py) | Spin harmonic inverse solve for Q/U-like cutouts. |
| [`example_map_spin_harmonic_multi_cutout_inverse.py`](../examples/example_map_spin_harmonic_multi_cutout_inverse.py) | Joint spin harmonic solve across cutouts. |
| [`example_map_spin_harmonic_mixed_constraints.py`](../examples/example_map_spin_harmonic_mixed_constraints.py) | Spin mixed-constraint solve. |

## ML And Runtime

| Script | What it demonstrates |
|---|---|
| [`example_image_dataset.py`](../examples/example_image_dataset.py) | Legacy file-backed example pending conversion to a `torchfits` loader plus torchsky transforms. |

## Notebooks (field / hyperspectral)

| Notebook | What it demonstrates |
|---|---|
| [`01_field_fit_synthetic.ipynb`](../examples/notebooks/01_field_fit_synthetic.ipynb) | Synthetic single-band `fit_field_model` quickstart. |
| [`02_multiband_hyperspectral_synthetic.ipynb`](../examples/notebooks/02_multiband_hyperspectral_synthetic.ipynb) | Multi-band throughput, `render_field_cube`, `write_cube_fits`, `ExposurePose`. |
| [`03_smacs0723_real_minicutout.ipynb`](../examples/notebooks/03_smacs0723_real_minicutout.ipynb) | Real JWST UPdec cutouts; requires running `scripts/build_smacs0723_minicutout.py`. |

For the real-data path, run `scripts/build_smacs0723_minicutout.py --list-bands`
first in the Pixi `test` environment, build the mini-cutouts, then run
`scripts/build_smacs0723_stack_baseline.py` to create the classical multiband
stack baseline used for INR comparisons. Run it once with the default bilinear
kernel for debugging, then with `--resampling-kernel drizzle_square` for a
drizzle-like astronomy baseline. Run
`scripts/visualize_smacs0723_products.py` after each data product to create PNG
quicklooks before trusting a loss curve. Use explicit `--center-x` / `--center-y`
values to make adjacent reproducible tiles as the project grows beyond the first
center crop. See
[projects/hyperspectral-fusion.md](projects/hyperspectral-fusion.md) for context.

For repeatable SMACS comparisons, supply a manifest with `bands` in cube order
and `products` entries containing `method`, `path`, and either `band` (2D image)
or `bands` (cube). Set `hdu` for a non-primary science image. All FITS reads use
Torchfits. Products must already share the classical reference's WCS and units;
the report does not silently align them.

```bash
pixi run -e test -- python scripts/visualize_smacs0723_products.py \
  --manifest run/manifest.json --baseline-cube reference/classical.fits \
  --metrics-json run/metrics.json --regions-json reference/regions.json \
  --output-dir run/report
```

Metrics use `{ "method": { "bands": { "F090W": { ... } } } }`, with
`prediction_rmse`, `psf_width_pix`, `flux_bias_fraction`, `color_bias_mag`,
`runtime_s`, and `memory_mb`. Missing measurements say “not measured.”
Regions use `{ "name": [x0, y0, width, height] }` on the common reference grid.
Choose the four science regions once, before inspecting method differences.

Each method gets linear/asinh and signed image-difference panels, with fixed
per-band scales derived from the classical reference. Image differences are
not detector residuals. Later reports can reuse `--display-limits
reference/report/display_limits.json`; the reference FITS checksum must match.
Completed report directories cannot be overwritten. Six-band false color uses
Torchfits's Lupton transform with fixed channel averages and stretch; missing
coverage is gray. Resolution-matched RGB is explicitly pending until independent
PSF matching kernels are supplied. The reporting code does not infer improvement
from image sharpness or manufacture missing science measurements.

## Running Examples

```bash
pixi run -e default -- python examples/example_image.py
```

Some examples require optional dependencies such as Polars, DuckDB, Astropy, or
visualization packages. Install the relevant extra or Pixi environment for that
workflow.
