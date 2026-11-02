# Coordinates and WCS: overview
<!-- TODO(0.5.0 docs): intro -->


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
  [MPS precision](projections.md#mps-precision)).
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

## Spectral And Temporal Axes

Spectral WCS covers linear axes, `-LOG`, reciprocal compounds (`WAVE-F2W`,
`FREQ-W2F`), velocity compounds (`WAVE-V2W`), and the grism algorithms `GRI` and
`GRA`. Air wavelengths use the WCSLIB Edlen form. Temporal helpers convert
UTC/TAI/TT/GPS and apply a local Keplerian light-time correction; ephemeris-grade
apparent place is outside the runtime contract.

## Where Things Live

| Need | Use |
|---|---|
| Frames, proper motion, observed place | [Coordinate frames and motion](frames.md) |
| Spectral axes as free functions | [Spectra](../spectra/index.md) |
| IFU slice WCS, tabular axes | `IFUSliceWCS`, `TabularAxis`, `TabularLookupWCS` in the [reference](../reference/wcs.md) |
| Footprints and pixel scales | `footprint`, `pixel_scales`, `pixel_area`, `position_angle` |
| Resampling images with a WCS | [Coadds and resampling](../maps/index.md) |

## Validation

Core tests cover bounded transform, round-trip, gradient, and device contracts.
Exhaustive Astropy/WCSLIB parity and throughput are measured in
[`torchsky-harness`](https://github.com/astroai/torchsky-harness); the summary is
in [Parity evidence](../performance/parity.md). Related example:
[WCS tour](../published-examples/wcs_tour.py).
