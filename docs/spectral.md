# `torchsky.spectral`

`torchsky.spectral` contains experimental helpers for 1D spectra, spectral axes,
and IFU-style data cubes. It complements, but does not replace, the lower-level
spectral coordinate transforms in `torchsky.wcs` and `torchsky.coords`.

## What This Namespace Is For

Use this namespace when you want containers and readers for spectra or cubes.
Use [wcs.md](wcs.md) for FITS-WCS spectral coordinate transforms and
[io.md](io.md) for generic FITS I/O.

## Core Concepts

- `SpectralAxis` carries values plus unit/type metadata.
- `Spectrum1D` couples flux, wavelength/frequency accessors, uncertainty, and
  masks.
- `DataCube` represents spatial plus spectral arrays with helpers for slices,
  collapse, and spectrum extraction.
- This surface is experimental; broad FITS I/O remains in the core I/O docs.

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

## API Surface

- `Spectrum1D`
- `DataCube`
- `SpectralAxis`
- `SpectralReader`

## Related Guides

- [io.md](io.md) for FITS files and HDUs.
- [wcs.md](wcs.md) for spectral WCS transforms.
- [map.md](map.md) for cube and multiband map products.
