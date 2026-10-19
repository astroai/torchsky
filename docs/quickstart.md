# Quick start

Get from a FITS file to tensor science products in a few minutes.

---

## 1. Install

`torchsky` is under development and is not yet on PyPI. Use a Pixi environment
in this repository (see [install.md](install.md)) or a wheel built from source:

```bash
pip install "torchsky @ file:///path/to/torchsky-0.4.0-*.whl"
```

`torchfits` owns FITS and table I/O and is a required dependency for file-based
workflows.

---

## 2. Read data with torchfits

```python
import torchfits

image, header = torchfits.read("science.fits", hdu=0, return_header=True)
table = torchfits.table.read("catalog.fits", hdu=1, columns=["RA", "DEC", "MAG_G"])
```

`torchsky` consumes the tensors and header metadata. It never opens files
itself; see the [I/O boundary](io.md) and [table boundary](table.md) guides.

---

## 3. Transform coordinates

```python
import torch
from torchsky.wcs import WCS

wcs = WCS(header)
x = torch.tensor([0.0, 10.0, 20.0])
y = torch.tensor([0.0, 5.0, 10.0])

ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)
```

The same call runs on CPU, CUDA, and MPS; pass tensors on the device you want.
Distortions (SIP, TPV), spectral/time axes, and non-celestial frames are part of
the same `WCS` object. Continue with the [WCS guide](wcs.md).

---

## 4. Work with HEALPix and spherical geometry

```python
import torch
import torchsky.sphere as sphere

lon = torch.tensor([10.0, 20.0])
lat = torch.tensor([5.0, -12.0])

pix = sphere.ang2pix(1024, lon, lat, lonlat=True)
neighbours = sphere.neighbors(1024, pix, nest=False)
counts = sphere.query_disc(1024, lon[0], lat[0], radius_deg=0.5)
```

HEALPix indexing, neighbours, interpolation, MOC, sparse maps, and harmonic
transforms live under `torchsky.sphere`. See the [sphere guide](sphere.md).

---

## 5. Coadd images

```python
from torchsky.map import Observation, coadd, plan

observations = [
    Observation("exposure_1.fits"),
    Observation("exposure_2.fits"),
]
target = plan(observations, projection="TAN")
result = coadd(observations, target)

print(result.data.shape, result.weight.shape)
```

`plan` chooses a target grid covering the observations; `coadd` projects and
stacks them. Kernels, value semantics, masks, and outlier rejection are explicit
configuration, not hidden policy. Continue with the [map guide](map.md).

---

## 6. Assemble a spectral cube

```python
import torch
from torchsky.map import ResampleConfig, coadd_cube

wavelengths = torch.linspace(1.0, 2.0, 8, dtype=torch.float64) * 1.0e4
cube = coadd_cube(
    observations,
    target,
    wavelengths=wavelengths,
    resample=ResampleConfig(spectral_kernel="drizzle"),
)
```

For 3D IFU frames, pass observations with per-pixel wavelengths; `coadd_cube`
routes to the 3D path automatically. See [spectral data](spectral.md).

---

## 7. Next steps

- [Examples](examples.md) for runnable scripts by domain.
- [Scene facade](scene.md) and [continuous fields](field.md) for modeling.
- [Compatibility](compatibility.md) for parity boundaries with Astropy, healpy,
  and other ecosystem packages.
- [Scientific evaluation](benchmarks.md) for how correctness and performance are
  measured in `torchsky-harness`.
