# torchsky

[![CI](https://github.com/astroai/torchsky/actions/workflows/ci.yml/badge.svg)](https://github.com/astroai/torchsky/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-astroai.github.io%2Ftorchsky-brightgreen)](https://astroai.github.io/torchsky/)
[![PyPI: coming with 0.5](https://img.shields.io/badge/PyPI-coming%20with%200.5-lightgrey)](https://astroai.github.io/torchsky/install/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/astroai/torchsky/blob/main/LICENSE)

Sky coordinates, spherical geometry, and mapmaking as PyTorch tensor operations.
They run on CPU, CUDA, and MPS, support autograd, and scale to survey-sized inputs.

torchsky covers FITS-WCS, HEALPix, spherical harmonic transforms, catalog
crossmatch, image coadds, spectral cubes, and observation operators. FITS reading
and writing is handled by [torchfits](https://github.com/astroai/torchfits), which
torchsky installs as a dependency.

## Install

torchsky 0.5 is not on PyPI yet. Until then, install from source (a C++ compiler
and CMake are needed for the native extension):

```bash
git clone https://github.com/astroai/torchsky.git
cd torchsky
pip install .
```

Once 0.5 is published, `pip install torchsky` will do. It supports Python 3.12 and
3.13 with PyTorch 2.14. The
[installation guide](https://astroai.github.io/torchsky/install/) covers CUDA,
Pixi, and troubleshooting.

## Quick start

```python
import torch
import torchsky.sphere as sphere
from torchsky.map import Observation, coadd, plan
from torchsky.wcs import WCS

header = {"NAXIS": 2, "NAXIS1": 64, "NAXIS2": 64,
          "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRPIX1": 32.5, "CRPIX2": 32.5,
          "CRVAL1": 150.0, "CRVAL2": 2.0, "CDELT1": -2.8e-4, "CDELT2": 2.8e-4}

ra, dec = WCS(header).pixel_to_world(torch.tensor([0.0, 63.0]), torch.tensor([0.0, 63.0]))
pix = sphere.ang2pix(1024, ra, dec, lonlat=True, nest=True)

obs = [Observation(data=torch.randn(64, 64), wcs={**header, "CRVAL1": c}) for c in (150.0, 150.01)]
result = coadd(obs, plan(obs, projection="TAN"), kernel="bilinear")
```

For real files, read them with torchfits and pass the header on:
`data, header = torchfits.read("image.fits", return_header=True)`.

## What's in the box

| Namespace | Contents |
|---|---|
| `torchsky.wcs` | FITS-WCS projections, SIP/TPV distortions, affine fitting |
| `torchsky.sphere` | HEALPix, MOC, sparse maps, polygons, spin SHT |
| `torchsky.coords` | Frame conversion, observed alt/az, proper motion |
| `torchsky.catalogs` | Crossmatch and spherical partitioning |
| `torchsky.map` | Coadds, spectral cubes, observation operators, solvers |
| `torchsky.spectral` and friends | Spectra, photometry, extinction, cosmology |
| `torchsky.scene`, `torchsky.field` | Forward models and continuous sky fields (experimental) |

## Performance

On 135 JWST/NIRCam exposures (566 Mpix) onto a 512 x 512 grid, torchsky's
Lanczos-3 coadd takes 14.3 s on 8 CPU threads and 5.2 s on a GPU slice, against
30.1 s for SWarp. Its drizzle matches STScI drizzle to better than 2e-6 sigma. A
280-CCD UNIONS tile coadd fell from 2292 s to 711 s on CPU. Hardware, versions,
and evidence are on the
[performance highlights](https://astroai.github.io/torchsky/evaluation/highlights/)
page.

## Documentation

- [Quick start](https://astroai.github.io/torchsky/quickstart/) and
  [examples](https://astroai.github.io/torchsky/examples/)
- [User guide and API reference](https://github.com/astroai/torchsky/blob/main/docs/index.md)
  (rendered at [astroai.github.io/torchsky](https://astroai.github.io/torchsky/))
- [Coming from Astropy, healpy, or reproject](https://astroai.github.io/torchsky/coming-from/)
- [Changelog](https://astroai.github.io/torchsky/changelog/)

## Contributing

Development uses [pixi](https://pixi.sh):

```bash
git clone https://github.com/astroai/torchsky.git && cd torchsky
pixi install --locked
pixi run dev              # build the native extension in place
pixi run preflight-push   # lint + compile check
pixi run test             # unit suite
```

Read [CONTRIBUTING.md](https://github.com/astroai/torchsky/blob/main/CONTRIBUTING.md)
before opening a PR. Issues labelled
[good first issue](https://github.com/astroai/torchsky/labels/good%20first%20issue)
are a good place to start.

## Citation

If torchsky helps your research, please cite it using
[CITATION.cff](https://github.com/astroai/torchsky/blob/main/CITATION.cff).

## License

[MIT](https://github.com/astroai/torchsky/blob/main/LICENSE)
