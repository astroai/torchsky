# Quick start

Get from an in-memory sky image to tensor science products in a few minutes.
Every snippet on this page except the last one is self-contained and runs in CI;
none of those needs a FITS file on disk. When you are ready to read real data, the
[file boundary](concepts.md#file-boundary) shows the `torchfits` calls that produce the same tensors
and header metadata.

## 1. Install

0.5 is not on PyPI yet. Build from source or use the Pixi environment; once it is
published, `pip install torchsky` will do ([Install](install.md) has the exact
commands). `torchfits>=1.3.1` comes with torchsky and owns FITS I/O.

<!-- docs: run -->
```python
import torch
import torchsky

print(torch.__version__, torchsky.__name__)
```

## 2. Build an image and a WCS header

A `torchsky` WCS is built from a plain header mapping. The same mapping is what
`torchfits` returns, so nothing about the compute path changes when the pixels
come from a file.

<!-- docs: run -->
```python
import torch

header = {
    "NAXIS": 2,
    "NAXIS1": 256,
    "NAXIS2": 256,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 128.5,
    "CRPIX2": 128.5,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CDELT1": -0.0002777778,
    "CDELT2": 0.0002777778,
}

image = torch.randn((256, 256), dtype=torch.float32)
assert image.shape == (int(header["NAXIS2"]), int(header["NAXIS1"]))
print(image.shape, image.dtype)
```

## 3. Transform pixels and sky coordinates

<!-- docs: run -->
```python
import torch

from torchsky.wcs import WCS

header = {
    "NAXIS": 2,
    "NAXIS1": 256,
    "NAXIS2": 256,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 128.5,
    "CRPIX2": 128.5,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CDELT1": -0.0002777778,
    "CDELT2": 0.0002777778,
}

wcs = WCS(header)
x = torch.tensor([10.0, 128.5, 200.0])
y = torch.tensor([20.0, 128.5, 240.0])

ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)

torch.testing.assert_close(x_back, x.to(x_back.dtype), atol=1e-6, rtol=0.0)
torch.testing.assert_close(y_back, y.to(y_back.dtype), atol=1e-6, rtol=0.0)
print(ra.tolist(), dec.tolist())
```

Device, dtype, and differentiability follow the tensors you pass in. Distortions
(SIP, TPV), spectral and time axes, tabular axes, and non-celestial frames are
part of the same `WCS` object. Continue with the [WCS guide](wcs.md).

## 4. Index the sphere

<!-- docs: run -->
```python
import torch
import torchsky.sphere as sphere

lon = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
lat = torch.tensor([5.0, -12.0, 40.0], dtype=torch.float64)

pixels = sphere.ang2pix(1024, lon, lat, nest=True, lonlat=True)
neighbours = sphere.neighbors(1024, pixels, nest=True)
disc = sphere.query_disc(1024, torch.tensor([1.0, 0.0, 0.0]), radius=0.5, nest=True)

# pix2ang returns pixel centres: ~3.4 arcmin at nside=1024, so compare loosely.
lon_back, lat_back = sphere.pix2ang(1024, pixels, nest=True, lonlat=True)
torch.testing.assert_close(lon_back, lon.to(lon_back.dtype), atol=0.1, rtol=0.0)
assert neighbours.shape[0] == pixels.shape[0]
assert disc.numel() > 0
print(pixels.tolist(), disc.numel())
```

HEALPix indexing, neighbours, interpolation, regions, MOC, sparse maps, and
harmonic transforms live under `torchsky.sphere`. See the
[sphere guide](sphere.md).

## 5. Coadd two exposures

<!-- docs: run -->
```python
import torch

from torchsky.map import Observation, coadd, plan

header = {
    "NAXIS": 2,
    "NAXIS1": 64,
    "NAXIS2": 64,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 32.5,
    "CRPIX2": 32.5,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CDELT1": -0.001,
    "CDELT2": 0.001,
}

observations = [
    Observation(data=torch.full((64, 64), 3.0), wcs={**header, "CRVAL1": 150.0}),
    Observation(data=torch.full((64, 64), 5.0), wcs={**header, "CRVAL1": 150.02}),
]
target = plan(observations, projection="TAN")
result = coadd(observations, target, kernel="bilinear")

assert result.data.shape == target.shape
assert bool((result.weight > 0).any())
print(result.data.shape, float(result.data.mean()))
```

`plan` chooses a target grid covering the observations; `coadd` projects and
stacks them. Kernels, value semantics, masks, and outlier rejection are explicit
configuration, not hidden policy. Continue with the [map guide](map.md).

## 6. Assemble a spectral cube

<!-- docs: run -->
```python
import torch

from torchsky.map import Observation, ResampleConfig, coadd_cube

header = {
    "NAXIS": 2,
    "NAXIS1": 32,
    "NAXIS2": 32,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 16.5,
    "CRPIX2": 16.5,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CDELT1": -0.002,
    "CDELT2": 0.002,
}

bands = [
    Observation(data=torch.full((32, 32), value), wcs=header, wavelength=wavelength)
    for value, wavelength in ((1.0, 5000.0), (2.0, 6000.0), (3.0, 7000.0))
]
cube = coadd_cube(
    bands,
    wavelengths=torch.tensor([5000.0, 6000.0, 7000.0]),
    resample=ResampleConfig(spectral_kernel="linear"),
)

assert cube.data.ndim == 3
print(cube.data.shape)
```

For 3D IFU frames, pass observations with per-pixel wavelengths; `coadd_cube`
routes to the 3D path automatically. See [spectral data](spectral.md) and
[cubes](cubes.md).

## 7. Read real data

The tensors and headers above are what `torchfits` returns from a file, so the
same calls work on real images:

```python
# fragment: needs a FITS file on disk
import torch
import torchfits
from torchsky.wcs import WCS

image, header = torchfits.read("science.fits", hdu="auto", return_header=True)
ra, dec = WCS(header).pixel_to_world(torch.arange(10.0), torch.arange(10.0))
```

## 8. Where to go next

- [Concepts](concepts.md) for the file boundary, devices, and ecosystem.
- [Examples](examples.md) — every runnable script, grouped by domain.
- [Scenes and operators](scene.md) and [continuous fields](field.md) for
  modelling.
- [Coming from Astropy, healpy, and reproject](coming-from.md) for the parity boundary with Astropy,
  healpy, and other ecosystem packages.
- [Performance highlights](evaluation/highlights.md), [kernel benchmarks](evaluation/benchmarks.md),
  and [parity evidence](evaluation/parity.md) for correctness against trusted
  references.
