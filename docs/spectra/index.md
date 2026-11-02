# Spectra and photometry: overview
<!-- TODO(0.5.0 docs): intro -->


!!! note "Stability"
    `torchsky.spectral` is an experimental surface for 1D spectra, spectral axes,
    and IFU-style cubes. It complements the spectral coordinate transforms in
    [`torchsky.wcs`](../coordinates/index.md); generic FITS I/O belongs to `torchfits`.

## Core Concepts

- `SpectralAxis` carries values plus unit and type metadata (`WAVE`, `FREQ`, ...)
  and converts between wavelength, frequency, energy, and wavenumber.
- `Spectrum1D` couples flux, a spectral axis, optional inverse variance, and a
  mask (`True` = good).
- `DataCube` holds spatial plus spectral arrays with slice, collapse, and
  spectrum-extraction helpers. The spectral axis is axis 0 by default.
- `flux_conserving_rebin` rebins flux density onto explicit bin edges while
  propagating masks and inverse variance.
- `relativistic_doppler_shift` applies signed physical radial velocities and can
  rebin a batch onto one observed grid.

## Neighbouring Modules

- Passbands and synthetic photometry: [Synthetic photometry](photometry.md).
- Dust laws (CCM89, O'Donnell94, Fitzpatrick 1999, Calzetti 2000):
  [Extinction](extinction.md).
- Differentiable FLRW distances and comoving geometry:
  [Cosmology](cosmology.md) and [Kinematics](kinematics.md).
- Cubes from images and IFU frames: [Spectral cubes](../maps/cubes.md).
- Spectral WCS transforms: [WCS](../coordinates/index.md).
