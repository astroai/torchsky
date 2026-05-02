# torchsky

[![PyPI](https://img.shields.io/pypi/v/torchsky)](https://pypi.org/project/torchsky/)
[![Wheels](https://github.com/sfabbro/torchsky/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/sfabbro/torchsky/actions/workflows/build_wheels.yml)
[![CI](https://github.com/sfabbro/torchsky/actions/workflows/ci.yml/badge.svg)](https://github.com/sfabbro/torchsky/actions/workflows/ci.yml)

`torchsky` is a torch-native astronomy stack for FITS I/O, FITS tables,
WCS transforms, spherical geometry, HEALPix workflows, and mapmaking.

The project is designed for survey-scale workloads where tensors, device
placement, differentiability, and throughput matter. Existing astronomy
libraries such as Astropy, fitsio, healpy, hpgeom, healsparse, reproject, and
drizzle are used as references, parity baselines, or optional compatibility
targets. They are not required imports for the core runtime path.

## Status

`torchsky` is alpha software. The current package version is `0.4.0.dev0`.

The declared stable surfaces are:

- `torchsky.read`, `torchsky.write`, and `torchsky.open` for FITS images,
  tables, headers, checksums, HDUs, cutouts, and streaming table reads.
- `torchsky.table` for Arrow-oriented FITS table access and interop with
  Polars, pandas, DuckDB, and Parquet when optional dependencies are installed.
- `torchsky.wcs.WCS` for torch-native FITS-WCS transforms on CPU, CUDA, and MPS
  where the underlying tensor operations support the device.
- `torchsky.sphere` for HEALPix, spherical geometry, sparse maps, MOC helpers,
  and spherical harmonic primitives.
- `torchsky.map.coadd_image_plane` and `torchsky.map.stack_map_product` for
  promoted mapmaking paths.

Compatibility surfaces such as `torchsky.sphere.compat` and
`torchsky.sphere.healpy` intentionally mirror parts of upstream APIs while
delegating to torchsky-native primitives. Field/INR modelling under
`torchsky.map.field` and the browser visualization server under `torchsky.viz`
are active but experimental.

Additional public namespaces such as `torchsky.coords`, `torchsky.geom`,
`torchsky.catalogs`, `torchsky.spectral`, and `torchsky.encoders` are documented
in [docs/api.md](docs/api.md).

## Install

For users installing a released wheel:

```bash
pip install torchsky
```

Wheels target Python `>=3.12,<3.15` and PyTorch `>=2.0`. The core runtime is
intentionally small: PyTorch, NumPy, and the compiled extension. Heavy astronomy
ecosystem packages live behind extras or Pixi environments.

Useful extras:

```bash
pip install "torchsky[viz]"       # browser FITS / sky viewer
pip install "torchsky[map]"       # SVO filter-profile support for map CLI paths
pip install "torchsky[bench-all]" # comparator libraries for benchmark/parity work
```

Source-build details, CFITSIO notes, CUDA build notes, and pip-only fallback
instructions are in [docs/install.md](docs/install.md).

## Contributor Setup

New collaborators should use Pixi. Do not rely on whatever `python` or
`python3` happens to be on PATH for project work.

```bash
git clone https://github.com/sfabbro/torchsky.git
cd torchsky
pixi install
./extern/vendor.sh
pixi run dev
pixi run verify
```

Day-to-day commands:

```bash
pixi run test                         # full pytest suite in the test env
pixi run preflight-push               # ruff + compileall, fast local gate
pixi run typecheck                    # pyright in the test env
pixi run ci-local                     # preflight + typecheck + tests
pixi run -e test -- pytest tests/test_wcs_core.py -q
```

Before pushing, run:

```bash
pixi run ci-local
```

That mirrors the local pre-push contract: Ruff, compileall, pyright, and the
pytest suite. CI should be the final verifier, not the first syntax/import
check.

## Quick Start

Read an image, keep the FITS header, and place the tensor on a requested device:

```python
import torchsky

data, header = torchsky.read(
    "science.fits",
    hdu="auto",
    device="cuda",
    return_header=True,
)

print(data.shape, data.dtype)
print(header.get("OBJECT"))
```

Read a cutout without loading the full image:

```python
stamp = torchsky.read_subset(
    "mosaic.fits",
    hdu=0,
    x1=1000,
    y1=1000,
    x2=1256,
    y2=1256,
)
```

Filter a FITS table with predicate pushdown:

```python
table = torchsky.table.read(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    where="MAG_G < 20",
)
```

Stream a FITS table in chunks:

```python
for batch in torchsky.stream_table(
    "catalog.fits",
    hdu=1,
    columns=["RA", "DEC", "MAG_G"],
    chunk_rows=100_000,
):
    ra = batch["RA"]
    dec = batch["DEC"]
```

Use torch-native WCS transforms:

```python
wcs = torchsky.get_wcs("science.fits", hdu="auto")

ra, dec = wcs.pixel_to_world(x, y)
x_back, y_back = wcs.world_to_pixel(ra, dec)
```

Use HEALPix and spherical primitives:

```python
import torch
import torchsky.sphere as sphere

lon = torch.tensor([10.0, 42.0])
lat = torch.tensor([-1.0, 5.0])

pix = sphere.ang2pix(1024, lon, lat, lonlat=True, nest=True)
lon2, lat2 = sphere.pix2ang(1024, pix, lonlat=True, nest=True)
```

## Repository Map

```text
src/torchsky/             Python package
src/torchsky/cpp_src/     C++ extension: nanobind + CFITSIO-backed kernels
src/torchsky/wcs/         Torch-native WCS projections, distortions, frames
src/torchsky/sphere/      HEALPix, spherical geometry, sparse maps, SHT
src/torchsky/map/         Image-plane, HEALPix, field, and product mapmaking
src/torchsky/viz/         Optional browser visualization server
tests/                    Unit, integration, parity, and performance smoke tests
benchmarks/               Benchmark, parity, and upstream-derived validation runners
examples/                 Runnable usage examples
docs/                     API, install, benchmark, compatibility, roadmap docs
extern/                   Vendored native dependency bootstrap
```

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [API guide](docs/api.md)
- [Installation and source builds](docs/install.md)
- [Examples](docs/examples.md)
- [I/O](docs/io.md)
- [Tables](docs/table.md)
- [WCS](docs/wcs.md)
- [Coordinates](docs/coords.md)
- [Geometry](docs/geom.md)
- [Sphere and HEALPix API](docs/sphere.md)
- [Map](docs/map.md)
- [Catalogs](docs/catalogs.md)
- [Spectral data](docs/spectral.md)
- [ML and runtime](docs/ml.md)
- [Encoders](docs/encoders.md)
- [Visualization](docs/viz.md)
- [CLI](docs/cli.md)
- [Ecosystem compatibility](docs/compatibility.md)
- [Benchmarks](docs/benchmarks.md)
- [Roadmap status](docs/roadmap/status.md)
- [Contributing guide](docs/contributing.md)
- [Release checklist](docs/release.md)
- [Agent and maintainer operating guide](AGENTS.md)
- [Changelog](docs/changelog.md)

## Engineering Expectations

Keep core runtime dependencies small. Optional astronomy stacks belong in tests,
benchmarks, docs, examples, or extras unless there is an explicit design
decision to promote them.

For performance-sensitive changes, include the bottleneck, the scaling story,
correctness coverage, and benchmark evidence. Prefer better algorithms,
structured indexing, chunking, streaming, or tensor-native execution before
adding native code.

For compatibility work, implement behavior in torchsky-native primitives first,
then expose thin compatibility wrappers. Validate against upstream libraries or
fixtures when available.

## License

[GPL-2.0-only](LICENSE)
