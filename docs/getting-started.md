# Getting Started

This is the shortest path through the main `torchsky` surfaces.

## Install

```bash
pip install torchsky
```

Wheels target Python `>=3.12,<3.15` and PyTorch `>=2.0`. See
[install.md](install.md) for source builds, Pixi contributor setup, and optional
extras.

## Read FITS Data

```python
import torchsky

data, header = torchsky.read("science.fits", hdu="auto", return_header=True)
```

For details, see [io.md](io.md).

## Read A Table With Predicate Pushdown

```python
table = torchsky.table.read(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20",
)
```

For details, see [table.md](table.md).

## Use WCS

```python
wcs = torchsky.get_wcs("science.fits", hdu="auto")
ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)
```

For details, see [wcs.md](wcs.md).

## Use HEALPix

```python
import torch
import torchsky.sphere as sphere

pix = sphere.ang2pix(1024, torch.tensor([10.0]), torch.tensor([5.0]), lonlat=True)
```

For details, see [sphere.md](sphere.md).

## Build A Map Product

```python
from torchsky.map import ImageObservation, coadd_image_plane

result = coadd_image_plane([ImageObservation("science.fits")])
```

For details, see [map.md](map.md).

## Next Steps

- [examples.md](examples.md) for runnable scripts.
- [compatibility.md](compatibility.md) for migration from common astronomy
  libraries.
- [benchmarks.md](benchmarks.md) for validation and performance surfaces.
