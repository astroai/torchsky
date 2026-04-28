# Installation

`torchsky` can be installed via pip or built from source.

## Standard Installation (Pip)

The easiest way to install `torchsky` is using `pip`:

```bash
pip install torchsky
```

Pre-built wheels are available for Linux (x86_64, aarch64) and macOS (x86_64, arm64). The wheels bundle [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/) — no system library needed.

Requires Python 3.10+ and [PyTorch](https://pytorch.org/) 2.0+.

To build `torchsky` from source, you'll need a C++ compiler and CMake.

### Prerequisites

- Python **3.12+** and **below 3.15** (matches `requires-python` in `pyproject.toml`)
- C++17 compatible compiler (GCC, Clang, MSVC)
- CMake 3.15+ (as in the top-level `CMakeLists.txt`; use a recent release when building against current PyTorch/CUDA)
- Ninja (recommended)
- **Python build backend** (see below): `scikit-build-core`, `nanobind`, plus runtime deps `torch` and `numpy` (versions in `pyproject.toml`)

#### Build-backend packages (pip without pixi)

The package uses **scikit-build-core** as the build backend. With default `pip install -e .`, pip creates an isolated build environment and installs those requirements automatically.

If you install **without** build isolation (`--no-build-isolation`), you must have the backend and build-time deps available in the **same** environment as the interpreter:

```bash
python -m pip install "scikit-build-core>=0.11.6" "nanobind>=2.10.2" "torch>=2.0.0" "numpy>=1.20.0"
```

Exact pins match `[build-system].requires` in `pyproject.toml`.

Optional extras (from `pyproject.toml`):

- **`map`**: `astropy` and `astroquery` for ``torchsky map --method inr`` with the default
  ``--inr-spectral-source=svo`` (SVO filter profiles). Install with
  `pip install 'torchsky[map]'` when using PyPI wheels, or install those packages in your env.

On macOS, install Xcode Command Line Tools if not present:

```bash
xcode-select --install
```

Clone the repository and prepare vendored sources:

```bash
git clone https://github.com/sfabbro/torchsky.git
cd torchsky
./extern/vendor.sh      # download vendored CFITSIO sources
pip install -e .        # editable install
```

`torchsky` vendors its C dependency (`cfitsio`) to simplify installation.
For a non-editable install, run:

```bash
pip install .
```

The build uses [scikit-build-core](https://scikit-build-core.readthedocs.io/) and [nanobind](https://nanobind.readthedocs.io/) for the C++ extension.

### Vendored CFITSIO

Editable install (typical development):

```bash
./extern/vendor.sh --cfitsio-version cfitsio-4.6.2
pip install -e . --no-build-isolation
```

System CFITSIO (disable vendored build):

```bash
pip install . --config-settings=cmake.args="-DTORCHSKY_USE_VENDORED_CFITSIO=OFF"
```

#### Pip only (no Pixi / no conda env manager)

The commands above are enough when pip can supply an isolated build env. If you already have **PyTorch from conda** (or another layout where system headers and `site-packages/torch` can disagree), prefer:

```bash
python -m pip install --no-build-isolation -e .
```

after installing `scikit-build-core` and `nanobind` into that environment (see **Build-backend packages**). Default isolation can pull a second Torch into `/tmp/...` and the C++ build may see **two different Torch include trees**, which fails with conflicting declarations.

If your conda base is **not writable**, use a virtual environment, `pip install --user`, or install with a conda tool you control; otherwise installation may fail when copying artifacts into `site-packages`.

#### CUDA-enabled PyTorch

If `torch.version.cuda` is set, CMake must find a **CUDA toolkit** compatible with your PyTorch build (headers, `nvcc`, NVRTC). On conda, `nvcc` is often present while **`CUDA_HOME` is unset**, which breaks `find_package(Torch)`. Point CMake at your toolkit root before building, for example:

```bash
export CUDA_HOME="${CONDA_PREFIX}"   # conda base or active env prefix
# optional, helps some layouts:
export CUDAToolkit_ROOT="${CUDA_HOME}"
python -m pip install --no-build-isolation -e .
```

Adjust `CUDA_HOME` if your toolkit lives elsewhere (e.g. `/usr/local/cuda`). CPU-only PyTorch wheels do not need this.

### Troubleshooting

torchsky reads FITS data on the CPU and places the resulting tensor on the requested device. No GPU-specific build steps are required — just install PyTorch with CUDA or MPS support:

- **Compiler Errors?**
  Verify your C++ compiler supports C++17. On macOS, ensure Xcode Command Line Tools are installed (`xcode-select --install`).

- **`BackendUnavailable: Cannot import 'scikit_build_core.build'`** with `--no-build-isolation`?
  Install `scikit-build-core` (and `nanobind`) into the active environment, then retry.

- **Torch header conflicts** (`torch/types.h` redeclarations, two include roots)?
  Use `--no-build-isolation` so the build uses a single Torch install (common with conda PyTorch).

- **CMake / PyTorch: CUDA or `CUDA::nvrtc` errors** with a GPU build of PyTorch?
  Set `CUDA_HOME` (and optionally `CUDAToolkit_ROOT`) to a full toolkit layout; ensure NVRTC is installed (conda: `cuda-nvrtc` / toolkit metapackage as appropriate for your channel).

- **Permission denied** installing into conda `site-packages`?
  Use a venv, `pip install --user`, or install with a writable prefix.
