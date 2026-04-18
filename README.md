# torchsky

[![PyPI](https://img.shields.io/pypi/v/torchsky)](https://pypi.org/project/torchsky/)
[![Wheels](https://github.com/sfabbro/torchsky/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/sfabbro/torchsky/actions/workflows/build_wheels.yml)
[![CI](https://github.com/sfabbro/torchsky/actions/workflows/ci.yml/badge.svg)](https://github.com/sfabbro/torchsky/actions/workflows/ci.yml)

Torch-native sky data, coordinates, geometry, mapmaking, and catalog workflows.

## Features

- **Canonical Namespaces**: `torchsky.io`, `torchsky.coords`, and `torchsky.geom` define the primary surface.
- **Native PyTorch Integration**: Read FITS images and tables directly into `torch.Tensor` with zero-copy efficiency, supporting CPU, CUDA, and MPS.
- **Coordinate-Aware Building Blocks**: New `torchsky.encoders` modules provide harmonic sky encodings, spectral encodings, and SIREN-style implicit fields.
- **Mapmaking Bootstrap**: `torchsky map` provides the first image-plane-first, differentiable coaddition baseline.
- **Catalog Refinement Hooks**: `torchsky.catalogs` adds tensor-native crossmatching primitives and iterable streaming adapters.

## Install

From PyPI:

```bash
pip install torchsky
```

Building from source (editable install, CUDA/conda notes, vendoring): see [`docs/install.md`](docs/install.md).

## Quick Examples

### GPU-Accelerated Image Loading

Read science images directly to GPU memory without intermediate copies:

```python
import torchsky

# Load directly to CUDA device (or 'mps' on Mac)
data, header = torchsky.read(
    "science.fits",
    hdu=0,
    device='cuda',
    return_header=True
)
print(data.shape, data.dtype)  # torch.Size([4096, 4096]), torch.float32
```

### Efficient Catalog Filtering and Streaming

Filter million-row tables at the C++ level before loading into Python:

```python
# Read only standard stars brighter than mag 20
table = torchsky.table.read(
    "catalog.fits",
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20.0 AND CLASS_STAR > 0.9"
)

# Stream massive catalogs in batches
for batch in torchsky.table.scan("survey.fits", batch_size=50_000):
   process(batch)
```

## Performance Snapshot (v0.3.0 Readiness)

torchsky is designed for high-throughput astronomical workflows, delivering **10x-100x speedups** over `astropy` for large arrays and tables. In the 0.3.0 exhaustive suite, it maintains a strong lead in FITS image and table I/O (1.5x-4x median speedups vs `fitsio`) while achieving parity or better in the HEALPix sphere domain (~89% win rate). While Batch WCS throughput (N > 100k) excels, small-N coordinate transforms and complex spin-harmonic operations (sphere) remain current targets for the 0.3.x optimization roadmap.

- **Current Benchmarks**: [`docs/benchmarks.md`](docs/benchmarks.md)
- **Historical Metrics & Gates**: [`docs/BENCHMARK_HISTORY.md`](docs/BENCHMARK_HISTORY.md)

## Documentation

- **User/API Reference**: [`docs/api.md`](docs/api.md)
- **Agent Operating Guide**: [`docs/AGENTS.md`](docs/AGENTS.md)
- **Research & Knowledge Base**: [`docs/RESEARCH_KNOWLEDGE_BASE.md`](docs/RESEARCH_KNOWLEDGE_BASE.md)
- **Vision & Roadmap**: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- **World-Model Whitepaper Plan**: [`docs/WORLD_MODEL_WHITEPAPER_PLAN.md`](docs/WORLD_MODEL_WHITEPAPER_PLAN.md)
- **Compatibility (healpy, Astropy, ...)**: [`docs/compatibility.md`](docs/compatibility.md)
- **Install & Dev Environment**: [`docs/install.md`](docs/install.md)
- **Migration & History**: [`docs/HISTORY.md`](docs/HISTORY.md)
- **Changelog**: [`docs/changelog.md`](docs/changelog.md)

## License

GPL-2.0
