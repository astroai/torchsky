# Examples

This page indexes examples that use the retained torchsky runtime API. Scientific
campaign notebooks, real-data preparation, and comparison reports live in the
[torchsky-harness](https://github.com/astroai/torchsky-harness) repository.

## Core examples

| Example | Demonstration |
|---|---|
| [`example_wcs_transform.py`](published-examples/example_wcs_transform.py) | Pixel/world transforms, spectral and temporal WCS, and distortion handling. |
| [`example_healpix.py`](published-examples/example_healpix.py) | HEALPix coordinate conversion, disc queries, and map sampling. |
| [`example_spectral_sht.py`](published-examples/example_spectral_sht.py) | Spherical harmonic transforms, smoothing, and spectra. |
| [`example_map_healpix_inverse.py`](published-examples/example_map_healpix_inverse.py) | Sphere-to-image inverse solve with HEALPix coefficients. |
| [`example_map_harmonic_inverse.py`](published-examples/example_map_harmonic_inverse.py) | Image-to-harmonic inverse solve. |
| [`example_map_field_survey_split.py`](published-examples/example_map_field_survey_split.py) | Train-time field decomposition. |
| [`example_field_multiband_quality.py`](published-examples/example_field_multiband_quality.py) | Small multiband field fit and runtime residual diagnostics. |
| [`example_table_interop.py`](published-examples/example_table_interop.py) | Arrow table batches consumed by torchsky catalog APIs. |

Legacy file-backed image and dataset examples are being retired in favor of
`torchfits` readers followed by torchsky transforms. Use the
[`torchfits` examples](https://github.com/astroai/torchfits/tree/main/examples)
for FITS read, cutout, and table-I/O workflows.

## Scientific evaluation examples

The following notebooks and operational scripts are maintained with their
evaluation harness, where dependencies, data policy, scoring, and provenance
are documented together:

- [Synthetic field fit](https://github.com/astroai/torchsky-harness/blob/main/notebooks/01_field_fit_synthetic.ipynb)
- [Multiband hyperspectral reconstruction](https://github.com/astroai/torchsky-harness/blob/main/notebooks/02_multiband_hyperspectral_synthetic.ipynb)
- [SMACS0723 real-data mini-cutout](https://github.com/astroai/torchsky-harness/blob/main/notebooks/03_smacs0723_real_minicutout.ipynb)
- [SMACS campaign scripts](https://github.com/astroai/torchsky-harness/tree/main/scripts)
- [Hyperspectral fusion onboarding](https://github.com/astroai/torchsky-harness/blob/main/docs/projects/hyperspectral-fusion.md)
- [Calibration evidence workflow](https://github.com/astroai/torchsky-harness/blob/main/docs/calibration-evidence.md)

Run package examples from the torchsky checkout with the relevant Pixi
environment, for example:

```bash
pixi run -e default -- python examples/example_wcs_transform.py
```

Harness notebooks and scripts run from the harness checkout and require a
separately installed torchsky subject build. See the
[harness methodology](https://github.com/astroai/torchsky-harness/blob/main/docs/benchmarks.md)
for environment and subject-selection instructions.
