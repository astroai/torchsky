# Spectral API reference

`torchsky.spectral` owns spectral containers and axis behaviour: 1D spectra,
spectral axes, cubes, and flux-conserving resampling. The [spectral guide](spectral.md)
covers the science workflow; adjacent namespaces own redshift, photometry,
extinction, and cosmology.

FITS access goes through `torchfits`; the containers in this namespace accept
tensors and headers.

---

## Spectral axis

```python
from torchsky.spectral import SpectralAxis

axis = SpectralAxis(values=wavelength_angstrom, unit="Angstrom", type="WAVE")
```

| Name | Contract |
|---|---|
| `SpectralAxis` | Wavelength, frequency, energy, or velocity axis with unit, type, optional rest frequency/redshift, and bin edges. |
| `canonical_spectral_type` | Canonical FITS spectral coordinate type. |
| `spectral_axis_from_header` | Build an axis from FITS header metadata (including `-TAB` tables). |
| `spectral_bin_edges_from_header` | Linear or non-linear bin edges and the tensor axis index. |
| `spectral_unit_scale` | Scale factor between equivalent units. |
| `frequency_to_spectral` | Convert a frequency to the requested coordinate type. |
| `air_wavelength_to_vacuum` | Standard-air to vacuum wavelength conversion. |

`SpectralAxis.values` is the coordinate array; `bin_edges` is filled when the
input provides edges instead of centers. Non-uniform axes are first-class; the
cube resampler honours them (see `coadd_cube` in the [map reference](api-map.md)).

---

## Containers and readers

| Name | Contract |
|---|---|
| `Spectrum1D` | 1D flux-density spectrum with optional inverse variance and mask. |
| `DataCube` | 3D cube (`spectral_axis_index` selects the spectral axis) with spatial WCS metadata. |
| `SpectralReader` | Read a 1D spectrum or a data cube through `torchfits`, returning the containers above. |
| `SpectralRebinResult` | Rebin result: `flux`, `spectral_edges`, optional `ivar`, `mask`. |

```python
from torchsky.spectral import SpectralReader

spectrum = SpectralReader.read_spectrum_1d("spectrum.fits")
cube = SpectralReader.read_data_cube("cube.fits")
```

---

## Resampling and Doppler shifts

| Name | Contract |
|---|---|
| `flux_conserving_rebin(flux, input_edges, output_edges)` | Rebin piecewise-constant flux density while preserving integrated flux; propagates inverse variance and mask. |
| `relativistic_doppler_shift(wave, flux, beta, ...)` | Signed relativistic Doppler shift of flux density. |
| `write_cube_fits(path, cube, spatial_header=..., spectral_axis=...)` | Write a 3D cube through `torchfits` with explicit spatial/spectral WCS; uniform and (optionally) non-uniform spectral axes. |

```python
import torch
from torchsky.spectral import flux_conserving_rebin

edges = torch.linspace(5000.0, 9000.0, 101, dtype=torch.float64)
flux = torch.ones(100, dtype=torch.float64)
rebinned = flux_conserving_rebin(flux, edges, torch.linspace(5000.0, 9000.0, 5, dtype=torch.float64))
```

---

## Redshift and velocity

`torchsky.redshift` owns the redshift/velocity conversions and BERV helpers.

| Group | Names |
|---|---|
| Constants and velocity kinds | `C_KM_S`, `C_M_S`, `V_KIND_BETA`, `V_KIND_KMS`, `V_KIND_REDSHIFT` |
| Redshift conversions | `one_plus_z_from_beta`, `one_plus_z_from_codes`, `one_plus_z_from_kind`, `one_plus_z_from_redshift`, `one_plus_z_from_velocity_kms`, `one_plus_z_from_zeta`, `zeta_from_beta`, `zeta_from_redshift`, `zeta_from_velocity_kms`, `beta_from_redshift` |
| Velocity helpers | `redshift_from_velocity_kms`, `velocity_kms_from_redshift`, `compose_rapidities`, `normalize_v_kind`, `convert_velocity_kind`, `require_one_plus_z` |
| Shifts and dimming | `shift_wavelength`, `shift_ln_wavelength`, `apply_redshift`, `apply_redshift_f_lambda`, `dimming_f_lambda`, `dimming_tolman_sb` |
| Earth motion | `earth_barycentric_velocity_icrs`, `los_barycentric_beta`, `zeta_berv_from_sky` |

`torchsky.log_lambda` provides the ln λ Gaussian convolution
(`convolve_log_lambda`) and resolving-power conversion (`sigma_R_kms`), sharing
`C_KM_S`.

---

## Photometry

`torchsky.spectrophotometry` integrates spectra through passbands.

| Name | Contract |
|---|---|
| `Passband`, `WavelengthUnit` | Passband definition and wavelength unit contract. |
| `PhotometryOutput`, `PhotometryResult` | Output mode and result tensor(s). |
| `throughput_from_mapping`, `evaluate_throughput` | Build a passband from a mapping and evaluate its throughput. |
| `synthetic_photometry`, `synthetic_photometry_bands` | Integrate one spectrum or a band stack. |
| `flux_to_magnitude`, `magnitude_to_flux` | Explicit-zero-point magnitude conversions. |

---

## Extinction

`torchsky.extinction` provides named laws and the generic
`A(λ) = k(λ) · A_V` interface.

| Name | Contract |
|---|---|
| `ExtinctionLaw`, `WavelengthUnit` | Law selector and wavelength unit. |
| `k_lambda`, `a_lambda`, `transmission` | Generic attenuation curve, absolute extinction, and transmitted fraction. |
| `ccm89`, `ccm89_ab`, `odonnell94`, `odonnell94_ab`, `fitzpatrick99`, `f99_a_lambda`, `calzetti`, `calzetti_a_lambda`, `calzetti_k_ebv`, `apply_extinction` | Named laws and application helpers. |

---

## Cosmology

`torchsky.cosmology` provides FLRW distances and Alcock–Paczynski scaling.

| Name | Contract |
|---|---|
| `FLRW`, `FlatWCDM`, `APScaling` | Cosmology classes and AP scaling result. |
| `PLANCK18_H0`, `PLANCK18_OM0`, `PLANCK18_W0`, `SPEED_OF_LIGHT_KM_S` | Planck 2018 defaults and constants. |
| `e_of_z`, `hubble_parameter` | Expansion rate. |
| `comoving_distance`, `transverse_comoving_distance`, `comoving_position`, `comoving_separation`, `differential_comoving_volume` | Comoving geometry. |
| `luminosity_distance`, `angular_diameter_distance`, `distance_modulus` | Observables. |
| `alcock_paczynski_scaling` | AP scaling for a radial/transverse measurement pair. |

---

See also: [spectral guide](spectral.md), [photometry encoders](photometry-encoders.md),
and [API router](api.md).
