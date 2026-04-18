# torchsky Examples

This directory contains examples of common `torchsky` workflows.

## Core Image/Table Scripts

| Example | Description |
|---------|-------------|
| `example_image.py` | Basic image reading and header access. |
| `example_table.py` | Basic table I/O operations (read, write, slice). |
| `example_table_interop.py` | Arrow/Pandas interop for table data. |
| `example_polars.py` | Polars-oriented table workflows. |

## WCS

| Example | Description |
|---------|-------------|
| `example_wcs_transform.py` | Header + coordinate transform examples for 1D/2D/3D WCS. |

## Image Handling

| Example | Description |
|---------|-------------|
| `example_image_cutouts.py` | Efficiently extracting sub-regions. |
| `example_image_cube.py` | Handling >2D data cubes. |
| `example_image_mef.py` | Working with Multi-Extension FITS (MEF) files. |
| `example_image_dataset.py` | Custom PyTorch Dataset for FITS images. |

## ML/Workflow

| Example | Description |
|---------|-------------|
| `example_ml_pipeline.py` | End-to-end ML pipeline on FITS data. |
| `example_phase2_features.py` | Feature-oriented workflow examples. |
| `example_table_recipes.py` | Practical table mutation/scan recipes. |

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
