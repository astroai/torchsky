# `torchsky.spectral`

`torchsky.spectral` contains experimental helpers for 1D spectra, spectral axes,
and IFU-style data cubes. It complements, but does not replace, the lower-level
spectral coordinate transforms in `torchsky.wcs`.

## What This Namespace Is For

Use this namespace when you want containers and readers for spectra or cubes.
Use [wcs.md](wcs.md) for FITS-WCS spectral coordinate transforms and
[io.md](io.md) for the `torchfits` file-I/O boundary.

## Core Concepts

- `SpectralAxis` carries values plus unit/type metadata.
- `Spectrum1D` couples flux, wavelength/frequency accessors, uncertainty, and
  masks.
- `DataCube` represents spatial plus spectral arrays with helpers for slices,
  collapse, and spectrum extraction.
- `flux_conserving_rebin(...)` rebins flux density on explicit bin edges while
  propagating masks and inverse variance.
- `relativistic_doppler_shift(...)` applies signed physical radial velocities
  and optionally rebins batches onto one observed wavelength grid.
- Sample-wise redshift helpers live in `torchsky.redshift` and are re-exported
  here: `shift_wavelength`, `dimming_f_lambda` (`F_λ` densities, `1/(1+z)`),
  and `dimming_tolman_sb` (surface brightness, `(1+z)^-4`). Do not mix them.
  Rapidity helpers (`zeta_from_redshift`, `compose_rapidities`,
  `shift_ln_wavelength`) and `los_barycentric_beta` live in the same module:
  `ln λ_topo = ln λ_rest + ζ_sys + ζ_BERV`, with `v_BERV > 0` when the observer
  recedes. LOSVD and published-`R` LSF share `convolve_log_lambda` /
  `sigma_R_kms` (`σ = c / (2.355 R)`).
- Passbands and synthetic photometry are `torchsky.spectrophotometry`.
  Dust laws are `torchsky.extinction` (CCM89, O'Donnell94, Fitzpatrick 1999,
  Calzetti 2000). Distances are `torchsky.cosmology`.
- This surface is experimental; generic FITS I/O belongs to `torchfits`.

## Recipes

### Read A 1D Spectrum

```python
from torchsky.spectral import SpectralReader

spec = SpectralReader.read_spectrum_1d(
    "spec.fits",
    hdu=1,
    flux_col="FLUX",
    wave_col="WAVELENGTH",
    ivar_col="IVAR",
)
```

### Work With A Cube

```python
cube = SpectralReader.read_data_cube("cube.fits", hdu=0)
spectrum = cube.extract_spectrum(y=100, x=100)
halpha = cube.extract_slice(wavelength=6563.0, width=5)
collapsed = cube.collapse_spectral(method="mean")
```

### Rebin Or Doppler-Shift A Spectrum

```python
import torch
from torchsky.spectral import flux_conserving_rebin, relativistic_doppler_shift

edges = torch.tensor([5000.0, 5001.0, 5003.0, 5006.0])
flux = torch.tensor([1.0, 2.0, 1.0])
rebinned = flux_conserving_rebin(
    flux,
    edges,
    torch.tensor([5000.0, 5002.0, 5006.0]),
)
shifted = relativistic_doppler_shift(flux, edges, 30_000.0)
```

Flux values are densities within the supplied bins. Positive radial velocity
is receding, velocities are metres per second, and invalid or incompletely
covered output bins are marked in the returned mask rather than renormalized.

## API Surface

- `Spectrum1D`
- `DataCube`
- `SpectralAxis`
- `SpectralReader`
- `SpectralRebinResult`
- `flux_conserving_rebin(...)`
- `relativistic_doppler_shift(...)`
- `shift_wavelength(...)`, `dimming_f_lambda(...)`, `dimming_tolman_sb(...)`
- `zeta_from_redshift(...)`, `compose_rapidities(...)`, `shift_ln_wavelength(...)`
- `los_barycentric_beta(...)`
- `convolve_log_lambda(...)`, `sigma_R_kms(...)`
- `write_cube_fits(...)` — write a `(L, H, W)` tensor with a 3-axis FITS WCS
  (spectral axis `CTYPE3`); pair with `SpectralReader.read_data_cube` for round-trips.

## Related Guides

- [io.md](io.md) for loading FITS files and HDUs through `torchfits`.
- [wcs.md](wcs.md) for spectral WCS transforms.
- [map.md](map.md) for cube and multiband map products.
- [field.md](field.md) for fitting continuous spectral fields and writing cubes with
  `write_cube_fits`.
