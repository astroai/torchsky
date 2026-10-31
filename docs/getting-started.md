# Getting Started

This is the shortest path through the main `torchsky` surfaces.

## Install

The 0.4.0 line is not on PyPI yet; [install.md](install.md) covers the release
wheel, source build, and Pixi paths. Package metadata supports Python `>=3.12`
and PyTorch `>=2.14,<2.15`; native wheels enforce their compiled PyTorch ABI at
import. `torchfits>=1.3` is a required runtime dependency for FITS-backed
workflows and is installed with torchsky.

## Read FITS Data

```python
import torchfits

# torchfits owns file I/O; torchsky consumes the tensor and metadata.
data, header = torchfits.read("science.fits", hdu="auto", return_header=True)
```

For details, see [io.md](io.md).

## Read A Table With Predicate Pushdown

```python
import torchfits

# Select columns and filter rows on read
table = torchfits.table.read(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20",
)
```

For details, see [table.md](table.md).

## Use WCS

```python
from torchsky.wcs import WCS

# Build a tensor-native coordinate transform from the loaded header.
wcs = WCS(header)
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
from torchsky.map import Observation, coadd, plan

# Plan a target grid from the observation and run an image-plane coadd
observations = [Observation("science.fits")]
target = plan(observations, projection="TAN")
result = coadd(observations, target)
```

For details, see [map.md](map.md).

## Use the Scene Facade

For high-level modeling, use the `torchsky.scene` facade:

```python
import torch
import torchfits
from torchsky import scene

# Read the WCS metadata with torchfits; build the observation and target explicitly
header = dict(torchfits.read_header("science.fits", 0))
obs = scene.image_observation("science.fits", header=header)
target = scene.target_from_header(header)

# Render a scene to the target grid
prediction = scene.render(torch.zeros(target.shape), target)
```

For details, see [scene.md](scene.md).

## Next Steps

- [examples.md](examples.md) for runnable scripts.
- [compatibility.md](compatibility.md) for migration from common astronomy libraries.
- [benchmarks.md](evaluation/benchmarks.md) for validation and performance surfaces.
