# Concepts: tensors, devices, and gradients

The rules on this page hold in every area of torchsky. Read it once; the area
guides assume it. The exact contracts are collected in the
[runtime contract](../reference/runtime-contract.md).

## Tensors In, Tensors Out

Every compute path takes tensors and plain header dictionaries and returns
tensors. A FITS header is a `dict` of keyword to value, the same mapping
torchfits returns. Nothing in the core opens a file, and no object hides a NumPy
or Astropy round trip. Higher-level objects follow the same rule: a WCS holds
tensors, and a scene (observations plus a target, see
[Scenes](../models/scenes.md)) is built from tensors and headers.

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS

header = {
    "NAXIS": 2, "NAXIS1": 128, "NAXIS2": 128,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 64.5, "CRPIX2": 64.5, "CRVAL1": 10.0, "CRVAL2": -30.0,
    "CDELT1": -1e-3, "CDELT2": 1e-3,
}
wcs = WCS(header)

x = torch.arange(0.0, 128.0, 16.0, dtype=torch.float64)
ra, dec = wcs.pixel_to_world(x, x)
x_back, y_back = wcs.world_to_pixel(ra, dec)

assert isinstance(ra, torch.Tensor) and ra.device == x.device
torch.testing.assert_close(x_back, x, atol=1e-6, rtol=0.0)
```

## File Boundary

FITS files and tables belong to
[torchfits](https://github.com/astroai/torchfits), a required dependency that
installs with torchsky. torchsky contains no FITS or CFITSIO code. Read with
torchfits, then hand the tensor and the header to torchsky:

```python
# fragment: needs a FITS file on disk
import torchfits
from torchsky.wcs import WCS

image, header = torchfits.read("science.fits", hdu="auto", return_header=True)
wcs = WCS(header)
```

Catalogs follow the same rule. `torchfits.table` reads, filters, and streams
FITS tables as Arrow or tensor batches; `torchsky.catalogs` does the sky part:
indexing, partitioning, and crossmatch.

```python
# fragment: needs a FITS table on disk and reference tensors ref_ra, ref_dec
import torchfits
from torchsky.catalogs import crossmatch_sky

for batch in torchfits.table.scan("catalog.fits", columns=["RA", "DEC"], batch_size=100_000):
    matches = crossmatch_sky(batch["RA"], batch["DEC"], ref_ra, ref_dec, radius_arcsec=1.5)
```

What must survive the boundary is your responsibility and torchsky checks what
it can: tensor dtype, shape, and device; the header cards a WCS needs
(including alternate keys and `-TAB` references); the meaning of masks and
weights; spectral and time units; and row order in streamed tables. Ambiguous
metadata raises instead of being guessed. The
[torchfits documentation](https://astroai.github.io/torchfits/) has the read
and scan signatures.

## Device Follows the Input

A result lives where its inputs live. CPU tensors stay on the CPU even when a
GPU is present, and CUDA or MPS tensors stay on their device. The rules:

- An explicit `device=` argument always wins.
- Otherwise the device of the tensor inputs is used.
- Tensors on different devices without `device=` raise `ValueError`. Data is
  never moved for you.
- Only when no input carries a tensor (for example, observations given as file
  paths) does `torchsky.device.default_compute_device()` choose: the
  `TORCHSKY_DEVICE` environment variable if set, else CUDA if available, else
  CPU.

<!-- docs: run -->
```python
import torch

from torchsky.map import Observation, resample

header = {
    "NAXIS": 2, "NAXIS1": 32, "NAXIS2": 32,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5, "CRPIX2": 16.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CDELT1": -1e-4, "CDELT2": 1e-4,
}
target = {**header, "CRVAL1": 150.0005}
image = torch.ones(32, 32)

result = resample(Observation(data=image, wcs=header), target)
assert result.data.device == image.device

accelerator = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
if accelerator is not None:
    mixed = Observation(data=image, wcs=header, weight_map=torch.ones(32, 32, device=accelerator))
    try:
        resample(mixed, target)
    except ValueError as err:
        assert "different devices" in str(err)
```

## Dtypes and Precision

Sky geometry needs float64: one float32 ulp at 360° is about 0.1 arcsecond.
Images rarely do.

- **Geometry runs in float64 on CPU and CUDA.** A `WCS` holds its parameters in
  float64 and returns float64 world coordinates even for float32 pixel inputs.
  Catalog positions and HEALPix angles should be float64 too.
- **Pixel values keep their dtype.** A float32 image resamples and coadds to a
  float32 product.
- **MPS has no float64.** On Apple GPUs, geometry runs in float32. When the
  expected position error of a WCS reaches half a pixel,
  [`PrecisionWarning`](../reference/torchsky.md#torchsky.PrecisionWarning) is
  emitted once. Kernels that need float64 (exact drizzle, conservative polygon
  resampling, the spectral resamplers) raise `NotImplementedError` on MPS; run
  them on CPU or CUDA.

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS

wcs = WCS({
    "NAXIS": 2, "NAXIS1": 32, "NAXIS2": 32,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5, "CRPIX2": 16.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CDELT1": -1e-4, "CDELT2": 1e-4,
})
x = torch.tensor([0.0, 15.5, 31.0])           # float32 pixels
ra, dec = wcs.pixel_to_world(x, x)
assert x.dtype == torch.float32 and ra.dtype == torch.float64
```

## Differentiability

Autograd flows through the geometry and the resampling wherever the maths is
smooth: WCS transforms (with respect to pixel positions), interpolation and
resampling kernels (with respect to pixel values), HEALPix map
sampling, spectral rebinning, synthetic photometry, extinction, and cosmology
distances. Where plain autograd would be slow or inexact, torchsky supplies its
own backward pass: spherical harmonic transforms use their exact adjoint, and
the drizzle overlap area has an analytic derivative. Discrete outputs such as
HEALPix pixel indices and crossmatch pairs have no gradient.

`WCS.pixel_to_world_differentiable` keeps the whole graph in pure torch for
optimization loops; see [Jacobians and gradients](../coordinates/index.md#jacobians-and-gradients).

<!-- docs: run -->
```python
import torch

import torchsky.sphere as sphere
from torchsky.map import Observation, resample

header = {
    "NAXIS": 2, "NAXIS1": 32, "NAXIS2": 32,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5, "CRPIX2": 16.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CDELT1": -1e-4, "CDELT2": 1e-4,
}
image = torch.rand(32, 32, generator=torch.Generator().manual_seed(1), requires_grad=True)
shifted = resample(Observation(data=image, wcs=header), {**header, "CRVAL1": 150.0005})
shifted.data.nansum().backward()
assert image.grad.abs().sum() > 0

nside = 16
values = torch.linspace(0.0, 1.0, 12 * nside**2, dtype=torch.float64)
lon = torch.tensor([10.0, 200.0], dtype=torch.float64, requires_grad=True)
lat = torch.tensor([5.0, -40.0], dtype=torch.float64)
sphere.sample_healpix_map(values, nside, lon, lat).sum().backward()
assert torch.isfinite(lon.grad).all()
```

## Batching and Scale

Functions are batched: pass a whole pixel grid, catalog, or stack of spectra
in one call instead of looping in Python. Most coordinate functions accept any
tensor shape and return the same shape.

For inputs larger than memory, algorithms prune before they compute (HEALPix
indices, footprint cropping) and products are bounded by configuration:
chunk sizes, tiles, streamed batches, and sparse HEALPix maps. The
[scaling guide](../performance/scaling.md) lists every knob. Native C++ kernels
accelerate measured hotspots; a pure-torch path gives the same results
(set `TORCHSKY_DISABLE_CPP=1` to force it).

<!-- docs: run -->
```python
import torch

from torchsky.catalogs import crossmatch_sky_streamed

g = torch.Generator().manual_seed(0)
ref_ra = 149.0 + 2.0 * torch.rand(5000, generator=g, dtype=torch.float64)
ref_dec = 1.0 + 2.0 * torch.rand(5000, generator=g, dtype=torch.float64)

# The left catalog arrives in batches of 1000 rows; indices stay global.
batches = ((ref_ra[i:i + 1000] + 1e-5, ref_dec[i:i + 1000]) for i in range(0, 5000, 1000))
matches = crossmatch_sky_streamed(batches, ref_ra, ref_dec, radius_arcsec=1.0)
assert torch.equal(matches.left_index, matches.right_index)
```

## Shared Conventions

| Quantity | Convention |
|---|---|
| Pixel coordinates | 0-based: the centre of the first pixel is `(0, 0)`. Use `pixel_to_world_values(..., origin=1)` for FITS 1-based values. |
| Image arrays | `(..., ny, nx)`: row is `y`, column is `x`, as torchfits returns them. |
| Sky angles | Degrees for longitude and latitude; arcseconds for match radii and separations. |
| Celestial frame | ICRS unless a frame is given. Conversions are explicit calls. |
| HEALPix | RING ordering by default, `nest=True` for NESTED; `lonlat=True` means degrees, otherwise `(theta, phi)` colatitude and longitude in radians, as in healpy. |
| Headers | Plain `dict` of FITS keywords, as returned by torchfits. |
| Masks and weights | Stated by each function: a weight is inverse variance only when declared so. |

<!-- docs: run -->
```python
import torch

import torchsky.sphere as sphere
from torchsky.wcs import WCS

wcs = WCS({
    "NAXIS": 2, "NAXIS1": 32, "NAXIS2": 32,
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5, "CRPIX2": 16.5, "CRVAL1": 150.0, "CRVAL2": 2.0,
    "CDELT1": -1e-4, "CDELT2": 1e-4,
})
zero_based = wcs.pixel_to_world(torch.tensor([0.0]), torch.tensor([0.0]))
one_based = wcs.pixel_to_world_values(torch.tensor([1.0]), torch.tensor([1.0]), origin=1)
torch.testing.assert_close(zero_based, one_based)

lon, lat = torch.tensor([45.0]), torch.tensor([30.0])
ring = sphere.ang2pix(2, lon, lat, lonlat=True)
nest = sphere.ang2pix(2, lon, lat, lonlat=True, nest=True)
assert torch.equal(sphere.ring2nest(2, ring), nest)
```

## Errors Fail Fast

Invalid input raises at the call, with a message that names the problem:
an `nside` that is not a power of two, a latitude outside ±90°, mixed devices,
a float64-only kernel on MPS, or ambiguous metadata.
Batch transforms are the one deliberate exception: a point outside a
projection's valid domain returns NaN, so one bad row does not fail a million
good ones. Typed exceptions are listed in the
[exceptions reference](../reference/exceptions.md).

<!-- docs: run -->
```python
import torch

import torchsky.sphere as sphere

try:
    sphere.ang2pix(100, torch.tensor([10.0]), torch.tensor([5.0]), lonlat=True)
except ValueError as err:
    assert "power of two" in str(err)
else:
    raise AssertionError("expected ValueError")
```

## Ecosystem and Compatibility

torchsky implements its own WCS, frames, spherical geometry, catalogs, maps,
and models. Established packages are references and parity oracles in tests,
never hidden runtime backends:

| Area | Reference used to validate it |
|---|---|
| WCS and frames | Astropy (WCSLIB, ERFA) |
| HEALPix, MOC, SHT | healpy, hpgeom, healsparse, astropy-healpix, mocpy, spherical-geometry |
| Resampling and coadds | reproject, SWarp, Montage, STScI drizzle |
| Crossmatch | STILTS, Astropy `SkyCoord` |

The healpy-shaped `torchsky.sphere.compat` namespace sits on top of the native
kernels for porting existing code; see [Coming from Astropy, healpy, and
reproject](coming-from.md). Exhaustive parity and real-data validation run in
[torchsky-harness](https://github.com/astroai/torchsky-harness); the
[parity page](../performance/parity.md) summarizes them.
