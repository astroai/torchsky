# Spectral axes
<!-- TODO(0.5.0 docs): intro -->


## Spectra And Cubes In Memory

<!-- docs: run -->
```python
import torch

from torchsky.spectral import DataCube, SpectralAxis, Spectrum1D

wave = torch.linspace(5000.0, 5100.0, 101, dtype=torch.float64)
axis = SpectralAxis(values=wave, unit="Angstrom", type="WAVE")

flux = torch.exp(-0.5 * ((wave - 5050.0) / 3.0) ** 2) + 0.1
spectrum = Spectrum1D(flux=flux, spectral_axis=axis, ivar=torch.full_like(flux, 100.0))
assert spectrum.wavelength.shape == (101,)

cube = DataCube(data=torch.rand(101, 8, 8, dtype=torch.float64), spectral_axis=axis)
one = cube.extract_spectrum(y=3, x=4)
line = cube.extract_slice(wavelength=5050.0, width=5)
collapsed = cube.collapse_spectral(method="mean")
assert one.flux.shape == (101,)
assert line.shape == collapsed.shape == (8, 8)
```

With files, read through `torchfits` and construct the containers from the
decoded tensors and header; `SpectralReader.read_spectrum_1d` and
`SpectralReader.read_data_cube` do this for you, and `write_cube_fits` writes a
`(L, H, W)` tensor with a 3-axis FITS WCS (`CTYPE3` spectral):

```python
# fragment: needs FITS files on disk
from torchsky.spectral import SpectralReader

spec = SpectralReader.read_spectrum_1d(
    "spec.fits", hdu=1, flux_col="FLUX", wave_col="WAVELENGTH", ivar_col="IVAR"
)
cube = SpectralReader.read_data_cube("cube.fits", hdu=0)
```

## Rebin Or Doppler-Shift

Flux values are densities within the supplied bins. Positive radial velocity is
receding and velocities are in metres per second. Invalid or incompletely
covered output bins are marked in the returned mask rather than renormalized.

<!-- docs: run -->
```python
import torch

from torchsky.spectral import flux_conserving_rebin, relativistic_doppler_shift

edges = torch.tensor([5000.0, 5001.0, 5003.0, 5006.0])
flux = torch.tensor([1.0, 2.0, 1.0])
new_edges = torch.tensor([5000.0, 5002.0, 5006.0])

rebinned = flux_conserving_rebin(flux, edges, new_edges)
# Mean density in each new bin: (1*1 + 2*1)/2 and (2*1 + 1*3)/4.
torch.testing.assert_close(rebinned.flux, torch.tensor([1.5, 1.25]))
# The integral of flux density over the covered range is conserved.
integral_in = (flux * (edges[1:] - edges[:-1])).sum()
integral_out = (rebinned.flux * (new_edges[1:] - new_edges[:-1])).sum()
torch.testing.assert_close(integral_in, integral_out)

shifted = relativistic_doppler_shift(flux, edges, 30_000.0)
assert bool(shifted.mask.all()) and shifted.spectral_edges[0] > edges[0]
```
