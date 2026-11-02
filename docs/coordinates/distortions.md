# Distortions
<!-- TODO(0.5.0 docs): intro -->


## Distortions

SIP, TPV, TNX, and paper-IV lookup tables are part of the same object; there is
no separate distortion step. The forward, inverse, and Jacobian paths use nested
Horner evaluation.

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS

header = {
    "NAXIS": 2, "NAXIS1": 512, "NAXIS2": 512,
    "CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP",
    "CRPIX1": 256.5, "CRPIX2": 256.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CD1_1": -1e-5, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 1e-5,
    "A_ORDER": 2, "B_ORDER": 2, "A_2_0": 1e-6, "A_0_2": -5e-7, "B_1_1": 2e-6,
    "AP_ORDER": 2, "BP_ORDER": 2,
}
wcs = WCS(header)
x = torch.linspace(0.0, 511.0, 5, dtype=torch.float64)
y = x.flip(0)

ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)
assert (x_back - x).abs().max() < 1e-6 and (y_back - y).abs().max() < 1e-6
assert wcs.to_header()["CTYPE1"] == "RA---TAN-SIP"
```

## Fit A Pixel-Affine Correction

`fit_affine_2d` fits a similarity or general affine between matched pixel lists;
`WCS.with_affine` applies it to a copy of the WCS.

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS, fit_affine_2d

header = {
    "NAXIS": 2, "NAXIS1": 512, "NAXIS2": 512,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 256.5, "CRPIX2": 256.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CD1_1": -1e-5, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 1e-5,
}
wcs = WCS(header)
source = torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=torch.float64)
target = source + torch.tensor([1.5, -2.0], dtype=torch.float64)

fit = fit_affine_2d(source, target, geometry="rscale")
shifted = wcs.with_affine(fit)

# The corrected WCS assigns `wcs`'s sky position of `source` to `target` pixels.
a = shifted.pixel_to_world(target[:, 0], target[:, 1])
b = wcs.pixel_to_world(source[:, 0], source[:, 1])
torch.testing.assert_close(a[0], b[0], atol=1e-9, rtol=0.0)
torch.testing.assert_close(a[1], b[1], atol=1e-9, rtol=0.0)
```

## Lookup-Table Axes

`-TAB` axes need the decoded tables as well as the header. Read both with
`torchfits` and pass them in:

```python
# fragment: needs a FITS file with -TAB extensions
import torchfits
from torchsky.wcs import WCS

with torchfits.open("cube_with_wave_tab.fits") as hdul:
    wcs = WCS(hdul[0].header, tables=hdul)
```

One-dimensional axes use `PSi_0` (table extension), `PSi_1` (coordinate column),
optional `PSi_2` (index vector), `PVi_1` (`EXTVER`) and `PVi_2` (`EXTLEVEL`).
Missing tables or columns fail rather than falling back to a linear axis. For
coupled multidimensional tables use `coupled_tabular_wcs_from_fits`, which builds
a multilinear forward interpolator; its inverse is exact for square tables,
least-squares for overdetermined ones, and a minimum-norm particular solution
(not unique) for underdetermined ones.
