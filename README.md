# torchsky

[![Wheels build](https://github.com/astroai/torchsky/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/astroai/torchsky/actions/workflows/build_wheels.yml)
[![CI status](https://github.com/astroai/torchsky/actions/workflows/ci.yml/badge.svg)](https://github.com/astroai/torchsky/actions/workflows/ci.yml)
[![Docs](https://github.com/astroai/torchsky/actions/workflows/docs.yml/badge.svg)](https://github.com/astroai/torchsky/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Pre-release:** 0.4.0 is the current development line and is not yet on
> PyPI. It ships when the correctness, performance, API, documentation, and
> harness goals in the [roadmap](docs/roadmap.md) are met.

`torchsky` is a **high-performance, torch-native astronomy stack** for
survey-scale coordinates, spherical geometry, catalog indexing, mapmaking,
observation operators, and joint modelling of sky observations.

FITS image and table I/O is provided by the focused
[`torchfits`](https://github.com/astroai/torchfits) package. The boundary is
intentional: `torchfits` loads tensors and metadata; `torchsky` performs
sky-domain computation without embedding a second FITS/CFITSIO engine.
The supported integration baseline is `torchfits 1.3.1` or newer (same PyTorch 2.14 lane).
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
[scientific evaluation guide](docs/evaluation/benchmarks.md) for the ownership boundary.

Current release-candidate evidence (same-machine, reference comparisons):

- **WCS**: 38 cases against astropy (30 projections plus reference-latitude-offset
  cases). Angular parity max `6.1e-7 asec` (AIR formula floor `4.1e-7`; offset
  cases `<=1.5e-8 asec`), pole/seam-excluded sky round trip `1.28e-6 asec`,
  pixel round trip `1.27e-6 px`; forward `0.46–2.65x` and inverse `0.88–4.32x`
  astropy throughput.
- **HEALPix**: 12 operations against healpy at nside 512, zero index/angle
  mismatches (max `1.14e-13 deg`), throughput `1.02–2.19x`.
- **Sphere and spin SHT**: parity gates pass against healpy; throughput ratios
  are recorded, not gated.
- **Catalogs**: crossmatch against astropy SkyCoord, max separation difference
  `1.13e-10 asec`.
- **GPU**: 17 WCS cases on CUDA (including the reference-offset cases) all OK at
  max `8.2e-3 asec` angular parity.

---

## Installation

Torchsky is not yet published on PyPI; install a repository release wheel or
build from source in the meantime:

```bash
# Wheel built from this repository (see docs/release.md)
pip install "torchsky @ file:///path/to/torchsky-0.4.0-cp*-*.whl"
```

Package metadata supports Python `>=3.12`, PyTorch `>=2.14,<2.15`, and torchfits
`>=1.0.0`. Native wheels fail closed on a mismatched compiled PyTorch ABI.
Install `torchfits` alongside torchsky when a workflow reads or writes
FITS files:

```bash
pip install torchfits
```

The integration baseline is `torchfits 1.3.1` or newer. Dataset and DataLoader
classes are provided by `torchfits.data`; torchsky does not alias them.

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
git clone https://github.com/astroai/torchsky.git
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
- [Examples](docs/examples.md) &mdash; Runnable scripts, gated by `pixi run examples`.

### API Reference
- [API reference](docs/reference/index.md) &mdash; Generated signature-level reference for every public namespace.

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
- [Benchmarks Guide](docs/evaluation/benchmarks.md) &mdash; Command lines, runs, and correctness gates.
- [Contributing Guidelines](docs/contributing.md) &mdash; PR policy and code rules.
- [Release Checklist](docs/release.md) &mdash; Step-by-step maintainer runbook.
- [Agent & Maintainer Operating Guide](AGENTS.md) &mdash; Engineering guidelines for agents.
- [Changelog](docs/changelog.md) &mdash; Release versions history.

---

## License

`torchsky` is licensed under the [MIT License](LICENSE).
