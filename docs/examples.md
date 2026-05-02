# Examples

This page indexes runnable scripts under [examples/](../examples), grouped to
match the domain guides.

## I/O

| Script | What it demonstrates |
|---|---|
| [`example_image.py`](../examples/example_image.py) | Read a FITS image into a tensor and inspect the header. |
| [`example_image_cutouts.py`](../examples/example_image_cutouts.py) | Extract sub-regions with `read_subset`. |
| [`example_image_cube.py`](../examples/example_image_cube.py) | Read and slice 3D image cubes. |
| [`example_image_mef.py`](../examples/example_image_mef.py) | Work with multi-extension FITS files. |

## Tables

| Script | What it demonstrates |
|---|---|
| [`example_table.py`](../examples/example_table.py) | Read binary tables into tensors and write back. |
| [`example_table_interop.py`](../examples/example_table_interop.py) | Variable-length columns and Astropy interop. |
| [`example_polars.py`](../examples/example_polars.py) | Polars and Arrow workflows. |
| [`example_table_recipes.py`](../examples/example_table_recipes.py) | Arrow scanner, Polars lazy frames, and DuckDB SQL. |

## WCS

| Script | What it demonstrates |
|---|---|
| [`example_wcs_transform.py`](../examples/example_wcs_transform.py) | Pixel-to-world and world-to-pixel transforms for images and spectra. |

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
| [`example_map_field_survey_split.py`](../examples/example_map_field_survey_split.py) | Train-time field decomposition. |
| [`example_map_teb_joint_cl_prior.py`](../examples/example_map_teb_joint_cl_prior.py) | Joint T/E/B harmonic solve. |
| [`example_map_spin_harmonic_inverse.py`](../examples/example_map_spin_harmonic_inverse.py) | Spin harmonic inverse solve for Q/U-like cutouts. |
| [`example_map_spin_harmonic_multi_cutout_inverse.py`](../examples/example_map_spin_harmonic_multi_cutout_inverse.py) | Joint spin harmonic solve across cutouts. |
| [`example_map_spin_harmonic_mixed_constraints.py`](../examples/example_map_spin_harmonic_mixed_constraints.py) | Spin mixed-constraint solve. |

## ML And Runtime

| Script | What it demonstrates |
|---|---|
| [`example_image_dataset.py`](../examples/example_image_dataset.py) | PyTorch `Dataset` for FITS images with caching and `DataLoader`. |
| [`example_ml_pipeline.py`](../examples/example_ml_pipeline.py) | End-to-end ML pipeline: loading, transforms, caching, distributed training. |

## Running Examples

```bash
pixi run -e default -- python examples/example_image.py
```

Some examples require optional dependencies such as Polars, DuckDB, Astropy, or
visualization packages. Install the relevant extra or Pixi environment for that
workflow.
