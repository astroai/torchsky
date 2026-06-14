# torchsky

[![PyPI version](https://img.shields.io/pypi/v/torchsky.svg)](https://pypi.org/project/torchsky/)
[![Wheels build](https://github.com/sfabbro/torchsky/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/sfabbro/torchsky/actions/workflows/build_wheels.yml)
[![CI status](https://github.com/sfabbro/torchsky/actions/workflows/ci.yml/badge.svg)](https://github.com/sfabbro/torchsky/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-GPL--2.0-blue.svg)](LICENSE)

`torchsky` is a **high-performance, torch-native astronomy stack** designed for survey-scale joint modelling of sky observations. It provides optimized implementations of FITS I/O, FITS binary tables, WCS coordinate transforms, spherical geometry, HEALPix workflows, and mapmaking.

The project is built for workflows where tensor placement (CPU/CUDA/MPS), differentiability, and high-throughput data pipelines matter. It replaces legacy astronomy stacks in performance-critical paths while maintaining clean compatibility layers.

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

- ⚡ **Zero-Copy FITS I/O**: Direct-to-tensor data loading (image and binary table HDUs) backed by a compiled C++/nanobind engine.
- 🌌 **Torch-Native WCS**: FITS-WCS transforms (`TAN`, `SIN`, `ARC`, `AIT`, etc.) and polynomial distortions (`SIP`, `TPV`, `TNX`, `ZPX`) running directly on CPU, CUDA, and MPS.
- 🌐 **Spherical Geometry & HEALPix**: Vectorized HEALPix primitives, Spherical Polygons, Multi-Order Coverage (MOC) maps, and high-performance Spherical Harmonic Transforms (SHT).
- 🗺️ **Scalable Mapmaking**: Area-aware coadds, spectral-cube stacking, and observation operator bundles.
- 🧠 **Continuous Field Models**: Experimental continuous spectral neural fields (\(F_\theta(\mathrm{RA}, \mathrm{Dec}, \lambda)\)) for implicit forward modeling.
- 💾 **Dataframe Interoperability**: Seamless integration with Arrow, Polars, DuckDB, and Pandas.

---

## Performance Highlights

`torchsky` is built for survey-scale workloads. On comparable data scales, it ranks first on throughput against standard Python astronomy libraries:
- **FITS I/O**: Maintains a **100% win rate** on throughput against `fitsio` and `astropy` for large datasets.
- **WCS Transforms**: Pure-PyTorch execution delivers **1.1x–1.3x speedups** in high-volume ($N > 100\text{k}$) forward and inverse transformations.
- **HEALPix Primitives**: OpenMP-parallelized Ring FFT and Legendre recurrence block kernels out-perform pure-Python alternatives.

For detailed benchmarks and methodology, see the [Benchmarks Guide](docs/benchmarks.md).

---

## Installation

Install the pre-built wheels from PyPI:

```bash
pip install torchsky
```

Pre-built wheels target Python `>=3.12,<3.15` and PyTorch `>=2.0` on Linux and macOS. Core runtime dependencies are minimal: only `torch` and `numpy`.

### Optional Extras

- **`viz`**: Browser FITS / sky viewer
  ```bash
  pip install "torchsky[viz]"
  ```
- **`map`**: Extended mapmaking tools (e.g. SVO filter profiles)
  ```bash
  pip install "torchsky[map]"
  ```
- **`bench-all`**: All comparator libraries for testing and parity benchmarks
  ```bash
  pip install "torchsky[bench-all]"
  ```

---

## Quick Start

### 1. Read FITS Image Data to GPU
```python
import torchsky

# Load an image directly into PyTorch on a requested device
data, header = torchsky.read(
    "science.fits",
    hdu="auto",
    device="cuda",
    return_header=True,
)

print(data.shape, data.dtype)
print(header.get("OBJECT"))
```

### 2. Read Sub-region Cutouts (Lazy I/O)
```python
# Extract stamp cutouts without loading the full image into memory
stamp = torchsky.io.read_subset(
    "mosaic.fits",
    hdu=0,
    x1=1000,
    y1=1000,
    x2=1256,
    y2=1256,
)
```

### 3. FITS Table Scan with Predicate Pushdown
```python
# Filter and project columns before loading into Python
table = torchsky.table.read(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20",
)
```

### 4. Differentiable WCS Coordinate Transform
```python
# Transform sky coordinates and back using torch-native WCS
wcs = torchsky.get_wcs("science.fits", hdu="auto")

ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)
```

### 5. HEALPix Operations
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
./extern/vendor.sh      # Download CFITSIO source dependency
pixi run dev            # Rebuild native extensions
pixi run verify         # Verify torchsky import smoke
```

### Common Developer Commands
- Run test suite: `pixi run test`
- Fast syntax/lint check: `pixi run preflight-push`
- Typecheck: `pixi run typecheck`
- Full pre-push local suite: `pixi run ci-local`

For more information, see the [Contributing Guide](docs/contributing.md).

---

## Documentation Directory

### Getting Started & Installation
- [Documentation Index](docs/README.md) &mdash; Entry point for repository docs.
- [Getting Started Guide](docs/getting-started.md) &mdash; 10-minute introduction to the stack.
- [Installation Guide](docs/install.md) &mdash; CFITSIO and CUDA build configurations.

### Core Modules
- [I/O Guide](docs/io.md) &mdash; Unified readers, caching, and checksums.
- [Tables Guide](docs/table.md) &mdash; Arrow scanners and table mutations.
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
- [Visualization Guide](docs/viz.md) &mdash; Browser FITS/HiPS server.
- [Ecosystem Compatibility Matrix](docs/compatibility.md) &mdash; AST, astropy, healpy interop boundaries.
- [Benchmarks Guide](docs/benchmarks.md) &mdash; Command lines, runs, and correctness gates.
- [Contributing Guidelines](docs/contributing.md) &mdash; PR policy and code rules.
- [Release Checklist](docs/release.md) &mdash; Step-by-step maintainer runbook.
- [Agent & Maintainer Operating Guide](AGENTS.md) &mdash; Engineering guidelines for agents.
- [Changelog](docs/changelog.md) &mdash; Release versions history.

---

## License

`torchsky` is licensed under the [GNU GPL v2.0-only](LICENSE).
