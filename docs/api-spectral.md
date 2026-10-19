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

## Adjacent spectral namespaces

| Namespace | Ownership |
|---|---|
| `torchsky.redshift` | Velocity kinds, `one_plus_z`/`zeta` conversions, BERV helpers, shift and dimming functions. |
| `torchsky.log_lambda` | ln λ Gaussian convolution and resolving power. |
| `torchsky.spectrophotometry` | Passbands, synthetic photometry, magnitude/flux conversions. |
| `torchsky.extinction` | CCM89, O'Donnell, F99, and Calzetti extinction laws. |
| `torchsky.cosmology` | FLRW distances, comoving geometry, AP scaling, distance modulus. |

See also: [spectral guide](spectral.md), [photometry encoders](photometry-encoders.md),
and [API router](api.md).
