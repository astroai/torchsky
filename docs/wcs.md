# WCS

`torchsky.wcs` is a torch-native implementation of FITS-WCS: celestial
projections with distortions, spectral and temporal axes, lookup-table axes, and
the alignment helpers that sit on top. Astropy and WCSLIB are parity oracles in
tests, never runtime fallbacks.

## Core Concepts

- A `WCS` is built from a plain header mapping, the same dict `torchfits`
  returns. File decoding belongs to `torchfits`; the WCS never opens a file.
- Pixels are 0-based and celestial world values are degrees, unless a method
  says otherwise (`pixel_to_world_values(..., origin=1)` keeps the FITS
  convention).
- Dtype and device follow your tensors. `wcs.to("cuda")` moves cached tables
  with the pixels. CUDA and CPU run float64; MPS runs float32 (see
  [MPS precision](#mps-precision)).
- Invalid projection domains and singular inverses return NaN instead of
  raising, so a batch never fails because of one bad point.
- Geometry is differentiable (see [Jacobians](#jacobians-and-gradients)).

## Pixel To World And Back

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS

header = {
    "NAXIS": 2, "NAXIS1": 512, "NAXIS2": 512,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 256.5, "CRPIX2": 256.5,
    "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CD1_1": -1e-5, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 1e-5,
}
wcs = WCS(header)

x = torch.linspace(0.0, 511.0, 5, dtype=torch.float64)
y = x.flip(0)
ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)

torch.testing.assert_close(x_back, x, atol=1e-7, rtol=0.0)
torch.testing.assert_close(y_back, y, atol=1e-7, rtol=0.0)
assert wcs.footprint(shape=(512, 512)).shape == (4, 2)
```

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

## Alternate Keys, Slicing, And Compiled Paths

`WCS(header, key="A")` selects an alternate WCS. `slice`, `sub`, `dropaxis`, and
`celestial` derive new objects with the CRPIX/CD bookkeeping already updated.
`wcs.compile()` returns a tensor-path object for repeated small-batch calls, where
the transform takes an `(..., 2)` pixel tensor.

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS

header = {
    "NAXIS": 2, "NAXIS1": 64, "NAXIS2": 64,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 32.5, "CRPIX2": 32.5, "CRVAL1": 10.0, "CRVAL2": 20.0,
    "CDELT1": -1e-3, "CDELT2": 1e-3,
    "CRVAL1A": 11.0, "CRVAL2A": 21.0, "CRPIX1A": 1.0, "CRPIX2A": 1.0,
    "CTYPE1A": "RA---TAN", "CTYPE2A": "DEC--TAN",
    "CDELT1A": -1e-3, "CDELT2A": 1e-3,
}
primary, alternate = WCS(header), WCS(header, key="A")
pixel = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
assert (primary.compile().pixel_to_world(pixel) != alternate.compile().pixel_to_world(pixel)).any()

cutout = primary.slice((slice(16, 48), slice(16, 48)))
full_ra, full_dec = primary.pixel_to_world(torch.tensor([20.0]), torch.tensor([30.0]))
cut_ra, cut_dec = cutout.pixel_to_world(torch.tensor([4.0]), torch.tensor([14.0]))
torch.testing.assert_close(full_ra, cut_ra)
torch.testing.assert_close(full_dec, cut_dec)
```

## Jacobians And Gradients

`jacobian` returns the local `d(world)/d(pixel)` matrix per point, and the
`*_differentiable` variants keep the autograd graph for geometry fitting.

At the reference point (CRPIX) the native longitude is undefined for zenithal
projections (the pole) and `cot(theta)` is singular for `PCO`. There the
Jacobian is analytic rather than differenced. For zenithal projections it is
`J = W K^-1 CD`, where `K = d(xi, eta)/dt` and `W = d(lon, lat)/dt` follow
`t = 90 - theta` along two native longitudes, so `K` carries the projection's own
`R'(theta_0)`; `PCO` uses the identity native Jacobian. The inverse direction uses
the matrix inverse. Elsewhere ordinary autograd is exact at CRPIX. The Hessian
on those reference rows is not defined by this construction (zero).

`XPH` and `QSC` have a genuine kink at CRPIX: the derivative depends on the
direction of approach and is undefined there. The WCS returns one defined
subgradient, the autograd Jacobian of the branch entered along `(+1, +0.371)`
pixels, evaluated 1e-4 pixel from CRPIX (about 1e-4 relative accuracy), and the
world-to-pixel Jacobian is its exact inverse, so the two directions are consistent.

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS

header = {
    "NAXIS": 2, "NAXIS1": 64, "NAXIS2": 64,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 32.5, "CRPIX2": 32.5, "CRVAL1": 10.0, "CRVAL2": 20.0,
    "CDELT1": -1e-3, "CDELT2": 1e-3,
}
wcs = WCS(header)
pixels = torch.tensor([[10.0, 12.0], [40.0, 50.0]], dtype=torch.float64)
jac = wcs.jacobian(pixels)
assert jac.shape == (2, 2, 2)
assert torch.isfinite(jac).all()
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

## Spectral And Temporal Axes

Spectral WCS covers linear axes, `-LOG`, reciprocal compounds (`WAVE-F2W`,
`FREQ-W2F`), velocity compounds (`WAVE-V2W`), and the grism algorithms `GRI` and
`GRA`. Air wavelengths use the WCSLIB Edlen form. Temporal helpers convert
UTC/TAI/TT/GPS and apply a local Keplerian light-time correction; ephemeris-grade
apparent place is outside the runtime contract.

## Projections And Devices

Celestial projections include the Paper II zenithal, cylindrical, and conic
families, polyconic (`BON`, `PCO`), HEALPix (`HPX`, `XPH`), quadcube (`TSC`,
`CSC`, `QSC`), and `NCP` (slant `SIN`). `PV1_1` and `PV1_2` set the native
fiducial point `(phi_0, theta_0)` of Paper II section 2.5 as in WCSLIB `celset`:
they fix the Euler angles and the default `LONPOLE` (`phi_0` if `dec_0 >= theta_0`,
else `phi_0 + 180`), and the projection origin is not shifted onto the fiducial
point (WCSLIB `cel.offset = 0`). Headers with a non-default fiducial point use
the tensor path on every device (no fused C++ shortcut). `TAN` with any `PV` card
is read as SCAMP `TPV` distortion, so its `PV1_1`/`PV1_2` are distortion terms.
Fused C++ kernels cover `PCO`, `XPH`, `TSC`, `CSC`, and `QSC` on CPU; the public
methods are the only supported entry points and fall back to tensor code
elsewhere.

`CSC` reproduces WCSLIB's float32 evaluation: float32 Chan-O'Neill polynomials,
the float32 `x/45` plane coordinate with its wrap and face classification, and the
float32 face-offset sum in the forward direction (up to 1e-5 degree on faces away
from the base face). `a*b + c` is always two separately rounded operations, so
results are identical on every platform and device. A baseline x86-64 WCSLIB
build matches exactly (6e-14 degree); AArch64 WCSLIB builds fuse the multiply-add,
so against them CSC differs by one float32 ulp (about 2e-5 degree forward).

### MPS precision

MPS has no float64, so a WCS on MPS runs float32. `wcs.to("mps")` (and the first
MPS tensor passed to a WCS on another device) emits `torchsky.wcs.PrecisionWarning`
once per WCS when the expected float32 error reaches half a pixel, the point at
which the nearest pixel is no longer reliable. The estimate is one float32 ulp of
the largest world value (longitude up to 360 degrees times `cos(dec_0)`, latitude
up to 90) divided by the smallest CD-column norm in degrees per pixel, plus one
ulp of the largest pixel coordinate. At 0.2 arcsec per pixel and `dec_0 = 0` that
is 111 mas, 0.55 pixel, so the warning fires; at 1 arcsec per pixel it is 0.11
pixel and it stays silent. Use CPU or CUDA for such grids.

Identity TPV uses `PV1_1=1` and `PV2_1=1`. A header with `PV2_2=1` and no `PV2_1`
is rank-deficient and its inverse is a particular solution.

GWCS objects are sampled onto regular grid transforms at calibration time and the
runtime never imports `gwcs`. Frozen detector tables keep a consistent Jacobian
sign, including reflected detector frames.

## Where Things Live

| Need | Use |
|---|---|
| Frames, proper motion, observed place | [Coordinate frames and motion](coordinates.md) |
| Spectral axes as free functions | [Spectra](spectral.md) |
| IFU slice WCS, tabular axes | `IFUSliceWCS`, `TabularAxis`, `TabularLookupWCS` in the [reference](reference/wcs.md) |
| Footprints and pixel scales | `footprint`, `pixel_scales`, `pixel_area`, `position_angle` |
| Resampling images with a WCS | [Coadds and resampling](map.md) |

## Validation

Core tests cover bounded transform, round-trip, gradient, and device contracts.
Exhaustive Astropy/WCSLIB parity and throughput are measured in
[`torchsky-harness`](https://github.com/astroai/torchsky-harness); the summary is
in [Parity evidence](evaluation/parity.md). Related example:
[WCS tour](published-examples/wcs_tour.py).
