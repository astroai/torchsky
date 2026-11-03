# Quick start

This page is a ten-minute tour of torchsky on one real field: the globular
cluster NGC 288, imaged in the g band by the DESI Legacy Imaging Surveys (DR10,
0.262 arcsec pixels). In four steps you read an image with `torchfits`, place
stars on it with its WCS, crossmatch them against 2MASS, and resample the image
onto a new grid on a GPU when one is available.

Every block on this page runs offline and on its own. The header cards, pixel
values, and catalog rows are copied from the public files, so you can paste
any block into Python. The
[forced photometry example](../examples/index.md#forced-photometry-of-143-gaia-stars-near-ngc-288)
downloads the full 384 × 384 × 3 image and the catalog of the field, and the
other [examples](../examples/index.md) do the same for Gaia, Planck, SDSS,
Chandra, and CFHT data.

## 0. Install

```bash
pip install torchsky        # from 0.5.0; release candidates install from GitHub
```

[`torchfits`](https://github.com/astroai/torchfits), which reads and writes
FITS files, installs with it. [Install](install.md) covers release
candidates, GPU builds of PyTorch, and source builds.

## 1. Read an Image With torchfits

`torchfits.read` returns pixels as a tensor and the header as a plain dict of
cards. That dict is all torchsky needs to build a WCS. The block writes a
13 × 13 pixel cutout around the brightest star of the field to a temporary file,
so it runs anywhere, and then reads it back the way you would read any image.

<!-- docs: run -->
```python
import tempfile
from pathlib import Path

import torch
import torchfits

# Legacy Surveys DR10 g band around NGC 288 (nanomaggies per pixel), 13 x 13
# pixels around the star at RA 13.12863, Dec -26.55975 (Gaia G = 16.83).
stamp = torch.tensor([
    [0.05, 0.06, 0.08, 0.10, 0.13, 0.16, 0.16, 0.16, 0.14, 0.11, 0.09, 0.07, 0.05],
    [0.06, 0.08, 0.12, 0.17, 0.22, 0.28, 0.31, 0.30, 0.25, 0.19, 0.14, 0.10, 0.07],
    [0.07, 0.11, 0.17, 0.27, 0.40, 0.54, 0.63, 0.59, 0.48, 0.34, 0.23, 0.15, 0.09],
    [0.10, 0.15, 0.25, 0.43, 0.71, 1.04, 1.23, 1.19, 0.92, 0.60, 0.35, 0.21, 0.13],
    [0.11, 0.19, 0.34, 0.64, 1.17, 1.82, 2.23, 2.12, 1.59, 0.96, 0.53, 0.28, 0.16],
    [0.13, 0.22, 0.43, 0.85, 1.64, 2.68, 3.32, 3.07, 2.27, 1.33, 0.68, 0.34, 0.18],
    [0.14, 0.24, 0.45, 0.91, 1.81, 2.95, 3.77, 3.45, 2.58, 1.47, 0.73, 0.37, 0.20],
    [0.13, 0.22, 0.41, 0.81, 1.55, 2.52, 3.28, 3.07, 2.21, 1.27, 0.66, 0.33, 0.18],
    [0.12, 0.18, 0.32, 0.58, 1.03, 1.60, 1.97, 1.88, 1.40, 0.86, 0.48, 0.27, 0.15],
    [0.10, 0.15, 0.24, 0.38, 0.60, 0.85, 1.00, 0.96, 0.75, 0.51, 0.32, 0.20, 0.13],
    [0.09, 0.12, 0.17, 0.24, 0.33, 0.43, 0.49, 0.47, 0.40, 0.29, 0.20, 0.14, 0.09],
    [0.10, 0.11, 0.12, 0.15, 0.19, 0.23, 0.25, 0.25, 0.21, 0.17, 0.13, 0.10, 0.07],
    [0.13, 0.10, 0.10, 0.10, 0.12, 0.14, 0.14, 0.14, 0.13, 0.10, 0.08, 0.07, 0.05],
], dtype=torch.float32)
cards = {   # the DR10 cutout header, with CRPIX moved to the 13 x 13 corner
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 13.12, "CRVAL2": -26.5525,
    "CRPIX1": 113.5, "CRPIX2": 106.5, "CD1_1": -7.27777777777778e-05, "CD1_2": 0.0,
    "CD2_1": 0.0, "CD2_2": 7.27777777777778e-05,
}

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "ngc288_g.fits"
    torchfits.write(str(path), stamp, header=cards)
    image, header = torchfits.read(str(path), return_header=True)

assert image.shape == (13, 13) and header["CTYPE1"] == "RA---TAN"
print(image.dtype, image.shape, f"{float(image.sum()):.1f} nanomaggies in the cutout")
```

With the downloaded file, the call is the same:
`torchfits.read(path, hdu=0, return_header=True)`. Pass `device="cuda"` to
`torchfits.read` to place the pixels on a GPU straight away.

## 2. From Pixels to the Sky With a WCS

A [`WCS`](../reference/wcs.md#torchsky.wcs.WCS) is built from the header dict.
Here it is the header of the full 384 × 384 image, and the stars are the six
brightest of the field in the DR10 catalog, whose positions are Gaia EDR3
astrometry. Pixels are 0-based.

<!-- docs: run -->
```python
import torch

from torchsky.coords import transform
from torchsky.wcs import WCS

header = {   # Legacy Surveys DR10 cutout of NGC 288 (legacysurvey.org viewer)
    "NAXIS": 2, "NAXIS1": 384, "NAXIS2": 384, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRVAL1": 13.12, "CRVAL2": -26.5525, "CRPIX1": 192.5, "CRPIX2": 192.5,
    "CD1_1": -7.27777777777778e-05, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 7.27777777777778e-05,
}
ra = torch.tensor([13.1286345, 13.1192538, 13.1330799, 13.1248666, 13.1323514, 13.1337038],
                  dtype=torch.float64)
dec = torch.tensor([-26.5597488, -26.5647797, -26.5578989, -26.5505519, -26.5618505, -26.5574865],
                   dtype=torch.float64)

wcs = WCS(header)
x, y = wcs.world_to_pixel(ra, dec)
assert (x.round()[0], y.round()[0]) == (85.0, 92.0)   # the peak of the cutout in step 1

corners = wcs.footprint(shape=(384, 384))              # (4, 2) RA, Dec in degrees
scale = 3600 * torch.tensor(wcs.pixel_scales(), dtype=torch.float64)   # arcsec per pixel
l, b = transform(ra, dec, from_frame="ICRS", to_frame="GALACTIC")
assert torch.allclose(scale, torch.tensor(0.262, dtype=torch.float64))
assert b.max() < -89.0                                 # NGC 288 sits next to the south Galactic pole
print(torch.stack([x, y], 1).round(decimals=2))
```

The same object handles SIP and TPV distortions, spectral and time axes, and
other projections; the transforms are batched over any tensor shape and carry
gradients. [Coordinates and WCS](../coordinates/index.md) explains how.

## 3. Crossmatch Against 2MASS

A crossmatch pairs each row of one catalog with the nearest row of another
within a radius. The right-hand table here is the 2MASS Point Source Catalog
within 72 arcsec of the field centre, fetched from IRSA. 2MASS has 2 arcsec
pixels and reaches J ≈ 16 here, so only the brightest cluster stars have a
counterpart.

<!-- docs: run -->
```python
import torch

from torchsky.catalogs import crossmatch_sky

gaia_ra = torch.tensor([13.1286345, 13.1192538, 13.1330799, 13.1248666, 13.1323514, 13.1337038],
                       dtype=torch.float64)
gaia_dec = torch.tensor([-26.5597488, -26.5647797, -26.5578989, -26.5505519, -26.5618505,
                         -26.5574865], dtype=torch.float64)
twomass_ra = torch.tensor([13.124883, 13.13824, 13.135131, 13.138223, 13.119194, 13.128567,
                           13.116755, 13.12841, 13.099378], dtype=torch.float64)
twomass_dec = torch.tensor([-26.550552, -26.554075, -26.565683, -26.56237, -26.564705, -26.559723,
                            -26.569674, -26.569744, -26.550032], dtype=torch.float64)
twomass_j = torch.tensor([16.363, 15.75, 15.39, 16.239, 15.902, 15.557, 16.372, 15.65, 16.051])

matches = crossmatch_sky(gaia_ra, gaia_dec, twomass_ra, twomass_dec, radius_arcsec=1.0)
assert matches.left_index.tolist() == [0, 1, 3]
assert matches.separation_arcsec.max() < 0.4
for i, j, sep in zip(matches.left_index, matches.right_index, matches.separation_arcsec):
    print(f"star {int(i)} -> 2MASS row {int(j)}: {float(sep):.2f} arcsec, J = {float(twomass_j[j]):.2f}")
```

The result holds row indices into both inputs, so any column of either table can
be gathered with them. The same call matches millions of rows: candidates come
from sorted declination strips and only the exact great-circle test runs per
pair. [Catalogs](../catalogs/index.md) covers one-to-one matching, per-row
error radii, and streaming.

## 4. Resample the Image on a GPU

[`resample`](../reference/map.md#torchsky.map.resample) moves an image onto
another grid. The target here is centred on the star, twice as fine, and rotated
by 30°, as a second instrument might see it. `flux` semantics keep the total
flux, and the computation runs on the device of the input tensor: CUDA, Apple
MPS, or the CPU.

<!-- docs: run -->
```python
import math

import torch

from torchsky.map import Observation, resample
from torchsky.wcs import WCS

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

stamp = torch.tensor([   # the 13 x 13 cutout of step 1
    [0.05, 0.06, 0.08, 0.10, 0.13, 0.16, 0.16, 0.16, 0.14, 0.11, 0.09, 0.07, 0.05],
    [0.06, 0.08, 0.12, 0.17, 0.22, 0.28, 0.31, 0.30, 0.25, 0.19, 0.14, 0.10, 0.07],
    [0.07, 0.11, 0.17, 0.27, 0.40, 0.54, 0.63, 0.59, 0.48, 0.34, 0.23, 0.15, 0.09],
    [0.10, 0.15, 0.25, 0.43, 0.71, 1.04, 1.23, 1.19, 0.92, 0.60, 0.35, 0.21, 0.13],
    [0.11, 0.19, 0.34, 0.64, 1.17, 1.82, 2.23, 2.12, 1.59, 0.96, 0.53, 0.28, 0.16],
    [0.13, 0.22, 0.43, 0.85, 1.64, 2.68, 3.32, 3.07, 2.27, 1.33, 0.68, 0.34, 0.18],
    [0.14, 0.24, 0.45, 0.91, 1.81, 2.95, 3.77, 3.45, 2.58, 1.47, 0.73, 0.37, 0.20],
    [0.13, 0.22, 0.41, 0.81, 1.55, 2.52, 3.28, 3.07, 2.21, 1.27, 0.66, 0.33, 0.18],
    [0.12, 0.18, 0.32, 0.58, 1.03, 1.60, 1.97, 1.88, 1.40, 0.86, 0.48, 0.27, 0.15],
    [0.10, 0.15, 0.24, 0.38, 0.60, 0.85, 1.00, 0.96, 0.75, 0.51, 0.32, 0.20, 0.13],
    [0.09, 0.12, 0.17, 0.24, 0.33, 0.43, 0.49, 0.47, 0.40, 0.29, 0.20, 0.14, 0.09],
    [0.10, 0.11, 0.12, 0.15, 0.19, 0.23, 0.25, 0.25, 0.21, 0.17, 0.13, 0.10, 0.07],
    [0.13, 0.10, 0.10, 0.10, 0.12, 0.14, 0.14, 0.14, 0.13, 0.10, 0.08, 0.07, 0.05],
], device=device)
header = {
    "NAXIS": 2, "NAXIS1": 13, "NAXIS2": 13, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRVAL1": 13.12, "CRVAL2": -26.5525, "CRPIX1": 113.5, "CRPIX2": 106.5,
    "CD1_1": -7.27777777777778e-05, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 7.27777777777778e-05,
}

s, t = 0.131 / 3600, math.radians(30.0)     # 0.131 arcsec pixels, rotated by 30 deg
target = {
    "NAXIS": 2, "NAXIS1": 40, "NAXIS2": 40, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRVAL1": 13.1286345, "CRVAL2": -26.5597488, "CRPIX1": 20.5, "CRPIX2": 20.5,
    "CD1_1": -s * math.cos(t), "CD1_2": s * math.sin(t),
    "CD2_1": s * math.sin(t), "CD2_2": s * math.cos(t),
}
result = resample(Observation(data=stamp, wcs=header), target, kernel="lanczos3", semantics="flux")

assert result.data.device.type == torch.device(device).type
flux = float(result.data.nan_to_num().sum())
assert abs(flux / float(stamp.sum()) - 1.0) < 0.03          # flux kept, less the clipped border
peak = int(result.data.nan_to_num().argmax())
x, y = WCS(target).world_to_pixel(torch.tensor([13.1286345]), torch.tensor([-26.5597488]))
assert abs(peak % 40 - float(x)) <= 1.5 and abs(peak // 40 - float(y)) <= 1.5   # the star stays put
print(device, tuple(result.data.shape), f"{flux:.1f} nanomaggies")
```

`result.data`, `result.weight`, and `result.header` are tensors and header
cards; write them with `torchfits.write`. On MPS, computation is float32 and the
kernels that need float64 (exact drizzle) raise rather than lose precision.
[Images and maps](../maps/index.md) covers coadds of many exposures, drizzle,
cubes, and HEALPix maps.

## Where to Go Next

- [What torchsky is](index.md) and [concepts](concepts.md): tensors, devices,
  gradients, and the boundary with `torchfits`.
- The guide areas: [Coordinates and WCS](../coordinates/index.md),
  [Celestial sphere](../sphere/index.md), [Catalogs](../catalogs/index.md),
  [Images and maps](../maps/index.md), [Spectra and photometry](../spectra/index.md),
  and the experimental [Models](../models/index.md).
- [Examples](../examples/index.md): complete scripts on Gaia, 2MASS, AllWISE,
  Planck, SDSS, Legacy Surveys, DSS, Chandra, and CFHT data.
- [Coming from Astropy, healpy, and reproject](coming-from.md) for a mapping of
  familiar calls.
