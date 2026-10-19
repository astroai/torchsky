# torchsky

[![PyPI version](https://img.shields.io/pypi/v/torchsky.svg)](https://pypi.org/project/torchsky/)
[![Wheels build](https://github.com/sfabbro/torchsky/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/sfabbro/torchsky/actions/workflows/build_wheels.yml)
[![CI status](https://github.com/sfabbro/torchsky/actions/workflows/ci.yml/badge.svg)](https://github.com/sfabbro/torchsky/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`torchsky` is a **high-performance, torch-native astronomy stack** for
survey-scale coordinates, spherical geometry, catalog indexing, mapmaking,
observation operators, and joint modelling of sky observations.

FITS image and table I/O is provided by the focused
[`torchfits`](https://github.com/astroai/torchfits) package. The boundary is
intentional: `torchfits` loads tensors and metadata; `torchsky` performs
sky-domain computation without embedding a second FITS/CFITSIO engine.
The supported integration baseline is `torchfits 1.0.0` or newer.
Torchsky does not expose alias namespaces for torchfits I/O, tables, caches, or
datasets; import those directly from `torchfits`.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Performance Highlights](#performance-highlights)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Contributor Setup](#contributor-setup)
6. [Documentation Directory](#documentation-directory)
7. [License](#license)

---

## Key Features

- 🌌 **Torch-Native WCS**: FITS-WCS transforms (`TAN`, `SIN`, `ARC`, `AIT`, etc.) and polynomial distortions (`SIP`, `TPV`, `TNX`, `ZPX`) running directly on CPU, CUDA, and MPS.
- 🌐 **Spherical Geometry & HEALPix**: Vectorized HEALPix primitives, Spherical Polygons, Multi-Order Coverage (MOC) maps, and high-performance Spherical Harmonic Transforms (SHT).
- 🗺️ **Scalable Mapmaking**: Area-aware coadds, spectral-cube stacking, matrix-free observation operator bundles, and bounded-memory execution.
- 🔭 **Survey-Scale Catalog Geometry**: Tensor-native crossmatch and spherical indexing primitives designed for candidate pruning and chunked execution.
- 🧠 **Continuous Field Models**: Experimental continuous spectral neural fields (\(F_\theta(\mathrm{RA}, \mathrm{Dec}, \lambda)\)) for implicit forward modeling.
- 🔌 **Explicit I/O Boundary**: Tensor/header/table-batch interchange with `torchfits` and other readers.

---

## Performance Highlights

Scientific performance claims, upstream parity campaigns, real-data evaluation,
and scorecards are maintained in the sibling
[`torchsky-harness`](https://github.com/astroai/torchsky-harness) repository.
Torchsky retains bounded package tests and performance smoke checks for its own
runtime paths. FITS image/table throughput belongs to `torchfits`. See the
[scientific evaluation guide](docs/benchmarks.md) for the ownership boundary.

---

## Installation

Install the pre-built wheels from PyPI:

```bash
pip install torchsky
```

Package metadata supports Python `>=3.12`, PyTorch `>=2.13`, and torchfits
`>=1.0.0` without upper caps. Native wheels still fail closed on a mismatched
compiled PyTorch ABI; build from source against a newer supported runtime when
needed. Install `torchfits` separately when a workflow reads or writes
FITS files:

```bash
pip install torchsky torchfits
```

The integration baseline is `torchfits 1.0.0` or newer. Dataset
classes are provided by `torchfits.data`; torchsky retains its own compatibility
aliases for older downstream callers.

### Optional Extras

- **`viz`**: Browser sky viewer
  ```bash
  pip install "torchsky[viz]"
  ```
- **`map`**: Extended mapmaking tools (e.g. SVO filter profiles)
  ```bash
  pip install "torchsky[map]"
  ```

---

## Quick Start

### 1. Load Data With torchfits, Compute With torchsky
```python
import torchfits
from torchsky.wcs import WCS

# File I/O belongs to torchfits.
data, header = torchfits.read(
    "science.fits",
    hdu="auto",
    device="cuda",
    return_header=True,
)

# Sky-coordinate computation belongs to torchsky.
wcs = WCS(header)
ra, dec = wcs.pixel_to_world(x, y)
```

### 2. Stream Catalog Batches
```python
# torchfits owns FITS table projection and predicate pushdown.
for batch in torchfits.table.scan(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20",
    batch_size=100_000,
):
    process_catalog_batch(batch)
```

### 3. HEALPix Operations
```python
import torch
import torchsky.sphere as sphere

lon = torch.tensor([10.0, 42.0])
lat = torch.tensor([-1.0, 5.0])

# Coordinate to HEALPix pixel mapping
pix = sphere.ang2pix(1024, lon, lat, lonlat=True, nest=True)
lon2, lat2 = sphere.pix2ang(1024, pix, lonlat=True, nest=True)
```

---

## Contributor Setup

Contributors should use `pixi` for environment management:

```bash
git clone https://github.com/sfabbro/torchsky.git
cd torchsky
pixi install
pixi run dev            # Rebuild native extensions
pixi run verify         # Verify torchsky import smoke
```

### Common Developer Commands
- Run test suite: `pixi run test`
- Fast syntax/lint check (git pre-push hook): `pixi run preflight-push`
- Typecheck: `pixi run typecheck`
- Optional local copy of GitHub `build-test`: `pixi run ci-local`

For more information, see the [Contributing Guide](docs/contributing.md).

---

## Documentation Directory

### Getting Started & Installation
- [Documentation](docs/index.md) &mdash; Entry point for guides, API reference, and examples.
- [Quick Start](docs/quickstart.md) &mdash; From a FITS file to science products in a few minutes.
- [Getting Started Guide](docs/getting-started.md) &mdash; 10-minute introduction to the stack.
- [Installation Guide](docs/install.md) &mdash; torchfits integration and native build configuration.

### Core Modules
- [I/O Boundary Guide](docs/io.md) &mdash; `torchfits` integration and tensor/metadata contracts.
- [Table Boundary Guide](docs/table.md) &mdash; `torchfits.table` and torchsky catalog batches.
- [WCS Guide](docs/wcs.md) &mdash; Coordinate projections and distortions.
- [Sphere & HEALPix Guide](docs/sphere.md) &mdash; Spherical geometry and SHT.
- [Mapmaking Guide](docs/map.md) &mdash; Image-plane coadds and product targets.
- [Scene Facade Guide](docs/scene.md) &mdash; High-level observation and rendering facade.

### Advanced Modeling & Research
- [Continuous Fields Guide](docs/field.md) &mdash; Neural fields and spectral mapping.
- [Implicit Forward Modeling Guide](docs/implicit_forward.md) &mdash; Loss formulations and kinematic states.
- [Catalogs Guide](docs/catalogs.md) &mdash; Sky crossmatching and partition streaming.
- [Spectral Data Guide](docs/spectral.md) &mdash; IFU cube handling and 1D spectrum containers.
- [ML & Runtime Guide](docs/ml.md) &mdash; Dataset, DataLoader, and cache tuning.
- [Encoders Guide](docs/encoders.md) &mdash; Feature encoders for spatial-spectral modeling.

### Platform & Tooling
- [CLI Guide](docs/cli.md) &mdash; Command-line query, print, and viz tools.
- [Visualization Guide](docs/viz.md) &mdash; Browser sky/HiPS server.
- [Ecosystem Compatibility Matrix](docs/compatibility.md) &mdash; AST, astropy, healpy interop boundaries.
- [Benchmarks Guide](docs/benchmarks.md) &mdash; Command lines, runs, and correctness gates.
- [Contributing Guidelines](docs/contributing.md) &mdash; PR policy and code rules.
- [Release Checklist](docs/release.md) &mdash; Step-by-step maintainer runbook.
- [Agent & Maintainer Operating Guide](AGENTS.md) &mdash; Engineering guidelines for agents.
- [Changelog](docs/changelog.md) &mdash; Release versions history.

---

## License

`torchsky` is licensed under the [MIT License](LICENSE).
