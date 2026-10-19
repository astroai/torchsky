# Installation

`torchsky` can be installed via pip or built from source.

## Standard Installation (Pip)

The easiest way to install `torchsky` is using `pip`:

```bash
pip install torchsky
```

Pre-built wheels are available for Linux (x86_64, aarch64) and macOS (x86_64,
arm64). Torchsky does not own or bundle CFITSIO. Install
[`torchfits`](https://github.com/astroai/torchfits) for FITS file and table I/O:

```bash
pip install torchsky torchfits
```

The supported integration baseline is `torchfits 1.0.0` or newer. Torchsky
keeps thin compatibility import paths for one migration release.

Package metadata requires Python **>=3.12** and
[PyTorch](https://pytorch.org/) **>=2.13**, with no upper dependency caps.
The reproducible contributor lane pins Python 3.13.14, PyTorch 2.13.0, and
torchfits 1.0.0. Native extensions still enforce their compiled PyTorch ABI.

## Source Build

To build `torchsky` from source, you'll need a C++ compiler and CMake.

### Prerequisites

- Python **3.12+** (matches `requires-python` in `pyproject.toml`)
- C++17 compatible compiler (GCC, Clang, MSVC)
- CMake 3.15+ for the native project, with 3.21+ recommended for the packaged build environment and current PyTorch/CUDA toolchains
- Ninja (recommended)
- **Python build backend** (see below): `scikit-build-core`, `nanobind`, plus runtime deps `torch` and `numpy` (versions in `pyproject.toml`)

#### Build-backend packages (pip without pixi)

The package uses **scikit-build-core** as the build backend. With default `pip install -e .`, pip creates an isolated build environment and installs those requirements automatically.

If you install **without** build isolation (`--no-build-isolation`), you must have the backend and build-time deps available in the **same** environment as the interpreter:

```bash
python -m pip install "scikit-build-core>=1.0.0" "nanobind>=2.10.2" "torch>=2.13" "numpy>=1.20.0"
```

Exact pins match `[build-system].requires` in `pyproject.toml`.

Optional extras (from `pyproject.toml`):

- **`map`**: `astropy` and `astroquery` for ``torchsky map --method inr`` with the default
  ``--inr-spectral-source=svo`` (SVO filter profiles). Install with
  `pip install 'torchsky[map]'` when using PyPI wheels, or install those packages in your env.

`torchfits` is an integration package, not a hidden backend. In-memory
tensor/header workflows do not require it.

On macOS, install Xcode Command Line Tools if not present:

```bash
xcode-select --install
```

Clone the repository and build:

```bash
git clone https://github.com/astroai/torchsky.git
cd torchsky
pip install -e .        # editable install
```

For a non-editable install, run:

```bash
pip install .
```

The build uses [scikit-build-core](https://scikit-build-core.readthedocs.io/) and [nanobind](https://nanobind.readthedocs.io/) for the C++ extension.

### FITS support

CFITSIO build configuration, vendoring, and troubleshooting belong to
`torchfits`. Torchsky's native extension is limited to torchsky-owned geometry
and compute kernels.

#### Pip only (no Pixi / no conda env manager)

The commands above are enough when pip can supply an isolated build env. If you already have **PyTorch from conda** (or another layout where system headers and `site-packages/torch` can disagree), prefer:

```bash
python -m pip install --no-build-isolation -e .
```

after installing `scikit-build-core` and `nanobind` into that environment (see **Build-backend packages**). Default isolation can pull a second Torch into `/tmp/...` and the C++ build may see **two different Torch include trees**, which fails with conflicting declarations.

Torchsky normalizes conda's split PyTorch header layout when `TorchConfig.cmake`
publishes `include/torch/csrc/api/include` separately from the parent `include`
root. The build still fails closed if the corresponding autograd headers are
not installed.

If your conda base is **not writable**, use a virtual environment, `pip install --user`, or install with a conda tool you control; otherwise installation may fail when copying artifacts into `site-packages`.

#### CUDA-enabled PyTorch

If `torch.version.cuda` is set, a source or editable build must find a **full
CUDA toolkit** compatible with that PyTorch build (headers, `nvcc`, NVRTC).
Runtime-only CUDA packages are not enough: PyTorch's own CMake package loads its
CUDA configuration before Torchsky's CPU-only extension target is defined. On
conda, `nvcc` is often present while **`CUDA_HOME` is unset**, which breaks
`find_package(Torch)`. Point CMake at your toolkit root before building, for
example:

```bash
export CUDA_HOME="${CONDA_PREFIX}"   # conda base or active env prefix
# optional, helps some layouts:
export CUDAToolkit_ROOT="${CUDA_HOME}"
python -m pip install --no-build-isolation -e .
```

Adjust `CUDA_HOME` if your toolkit lives elsewhere (e.g. `/usr/local/cuda`).
CPU-only PyTorch builds do not need this.

A Torchsky wheel built against CPU-only PyTorch may be binary-compatible with
an exact-version CUDA PyTorch runtime, but that combination is distribution- and
ABI-sensitive. It is not a supported workaround for conda/Pixi or CANFAR until
the exact build wheel and target runtime pass native import plus a first WCS
construction/transform smoke. Use a full matching toolkit for source/editable
installs in those environments.

If an otherwise supported environment crashes while loading the optional native
accelerator, set `TORCHSKY_DISABLE_CPP=1` before importing Torchsky. This keeps
the Torch-native WCS and sphere implementations available while disabling only
the C++ fast paths. Treat this as a safe diagnostic/fallback, and do not publish
native performance claims from such a run.

### Troubleshooting

Install PyTorch with CUDA or MPS support for torchsky tensor computation:

- **Compiler Errors?**
  Verify your C++ compiler supports C++17. On macOS, ensure Xcode Command Line Tools are installed (`xcode-select --install`).

- **`BackendUnavailable: Cannot import 'scikit_build_core.build'`** with `--no-build-isolation`?
  Install `scikit-build-core` (and `nanobind`) into the active environment, then retry.

- **Torch header conflicts** (`torch/types.h` redeclarations, two include roots)?
  Use `--no-build-isolation` so the build uses a single Torch install (common with conda PyTorch).

- **CMake / PyTorch: CUDA or `CUDA::nvrtc` errors** with a GPU build of PyTorch?
  Set `CUDA_HOME` (and optionally `CUDAToolkit_ROOT`) to a full toolkit layout;
  ensure headers, `nvcc`, and NVRTC are installed. An NVRTC or CUDA runtime
  package by itself is insufficient.

- **Permission denied** installing into conda `site-packages`?
  Use a venv, `pip install --user`, or install with a writable prefix.
