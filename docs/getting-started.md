# Getting Started

This is the shortest path through the main `torchsky` surfaces.

## Install

```bash
pip install torchsky
```

Wheels target Python `>=3.12,<3.15` and PyTorch `>=2.0`. See [install.md](install.md) for source builds, Pixi contributor setup, and optional extras.

## Read FITS Data

```python
import torchsky

# Read image data and FITS header
data, header = torchsky.read("science.fits", hdu="auto", return_header=True)
```

For details, see [io.md](io.md).

## Read A Table With Predicate Pushdown

```python
import torchsky

# Select columns and filter rows on read
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
import torchsky

# Get WCS coordinate projection object
wcs = torchsky.get_wcs("science.fits", hdu="auto")
ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)
```

For details, see [wcs.md](wcs.md).

## Use HEALPix

```python
import torch
import torchsky.sphere as sphere

# Map sky angles (degrees) to HEALPix pixel indices
pix = sphere.ang2pix(1024, torch.tensor([10.0]), torch.tensor([5.0]), lonlat=True)
```

For details, see [sphere.md](sphere.md).

## Build A Map Product

```python
from torchsky import scene
from torchsky.map import coadd_image_plane

# Plan and run an image-plane coadd
result = coadd_image_plane([scene.image_observation("science.fits")])
```

For details, see [map.md](map.md).

## Use the Scene Facade

For high-level modeling, use the `torchsky.scene` facade:

```python
import torch
from torchsky import scene

# Build observation and target render specification
obs = scene.image_observation("science.fits")
target = scene.target_from_header(obs.header)

# Render a scene to the target grid
prediction = scene.render(torch.zeros(target.shape), target)
```

For details, see [scene.md](scene.md).

## Next Steps

- [examples.md](examples.md) for runnable scripts.
- [compatibility.md](compatibility.md) for migration from common astronomy libraries.
- [benchmarks.md](benchmarks.md) for validation and performance surfaces.
