# torchsky Examples

This directory contains examples of common `torchsky` workflows.

## Images

| Script | What it demonstrates |
|---|---|
| [`example_image.py`](../examples/example_image.py) | Read a FITS image into a tensor, inspect the header |
| [`example_image_cutouts.py`](../examples/example_image_cutouts.py) | Extract sub-regions with `read_subset` (0-based pixel bounds) |
| [`example_image_cube.py`](../examples/example_image_cube.py) | Read and slice 3D data cubes (e.g. RA, Dec, velocity) |
| [`example_image_mef.py`](../examples/example_image_mef.py) | Work with multi-extension FITS files (primary + image + table HDUs) |

## Tables

| Script | What it demonstrates |
|---|---|
| [`example_table.py`](../examples/example_table.py) | Read binary tables into tensors, access columns, write back |
| [`example_table_interop.py`](../examples/example_table_interop.py) | Variable-length array columns, interop with [Astropy](https://www.astropy.org/) |
| [`example_polars.py`](../examples/example_polars.py) | Table workflows with [Polars](https://pola.rs/) and [Apache Arrow](https://arrow.apache.org/) |
| [`example_table_recipes.py`](../examples/example_table_recipes.py) | Arrow scanner, [Polars](https://pola.rs/) lazy frames, and [DuckDB](https://duckdb.org/) SQL queries on FITS tables |

## WCS

| Script | What it demonstrates |
|---|---|
| [`example_wcs_transform.py`](../examples/example_wcs_transform.py) | Pixel-to-world and world-to-pixel transforms for 2D images and 1D spectra |

## HEALPix & Spherical Harmonics

| Script | What it demonstrates |
|---|---|
| [`example_healpix.py`](../examples/example_healpix.py) | `ang2pix`, `pix2ang`, disc queries, and map sampling with `torchsky.sphere` |
| [`example_spectral_sht.py`](../examples/example_spectral_sht.py) | Spherical harmonic transforms: power spectra, `synfast`, `map2alm`, smoothing |

## ML Integration

| Script | What it demonstrates |
|---|---|
| [`example_image_dataset.py`](../examples/example_image_dataset.py) | PyTorch `Dataset` for FITS images with caching and `DataLoader` |
| [`example_ml_pipeline.py`](../examples/example_ml_pipeline.py) | End-to-end pipeline: data loading, transforms, caching, distributed training |

## Spherical Geometry & HEALPix

| Example | Description |
|---------|-------------|
| `example_healpix.py` | Coordinates to pixels, disc queries, and map sampling. |
| `example_spectral_sht.py` | Spherical Harmonic Transforms, smoothing, and GPU acceleration. |
| `example_map_healpix_inverse.py` | Native sphere↔image inverse solve using HEALPix coefficients, WCS cutouts, and the exact bilinear adjoint. |
| `example_map_healpix_multi_cutout_inverse.py` | Native joint sphere↔image inverse solve across multiple WCS cutouts that share one HEALPix sky map. |
| `example_map_healpix_mixed_constraints.py` | Native mixed-constraint HEALPix solve combining cutouts with a coarse all-sky map anchor. |
| `example_map_harmonic_inverse.py` | Native image↔alm inverse solve for low-order diffuse/background fitting on a WCS cutout. |
| `example_map_harmonic_cl_prior.py` | Native image↔alm solve with an astronomer-style `C_ell` prior and optional beam transfer. |
| `example_map_harmonic_multi_cutout_inverse.py` | Native joint image↔alm inverse solve across multiple WCS cutouts that share one smooth latent spherical component. |
| `example_map_harmonic_mixed_constraints.py` | Native mixed-constraint image↔alm solve combining image cutouts with coefficient-space anchors or priors. |
| `example_map_harmonic_survey_split.py` | Native shared-versus-survey-private harmonic decomposition showing how a split latent model can keep survey-specific structure from leaking into the shared astrophysical sky. |
| `example_map_harmonic_learned_survey_instrument_split.py` | Jointly optimizes latent harmonic components and a learned mechanistic survey/instrument mixer to separate shared sky from nuisance structure. |
| `example_map_field_survey_split.py` | Native train-time field decomposition on context-labeled observations using `make_field_component_decomposition(...)` and `fit_field_component_decomposition(...)`. |
| `example_map_teb_joint_cl_prior.py` | Native joint temperature-plus-polarization harmonic solve with a shared `T/E/B` cross-spectrum prior. |
| `example_map_spin_harmonic_inverse.py` | Native image↔(E,B) inverse solve for smooth polarized Q/U cutouts. |
| `example_map_spin_harmonic_multi_cutout_inverse.py` | Native joint image↔(E,B) inverse solve across multiple Q/U cutouts that share one smooth spherical polarization model. |
| `example_map_spin_harmonic_mixed_constraints.py` | Native mixed-constraint image↔(E,B) solve combining polarized cutouts with harmonic smoothness priors. |

## Running Examples

All examples are executable Python scripts:

```bash
pixi run python examples/example_image.py
```

Or directly:

```bash
python examples/example_image.py
```

Some examples require optional dependencies (e.g. Polars, DuckDB, Astropy). Install them with:

```bash
pip install torchsky[examples]
```
