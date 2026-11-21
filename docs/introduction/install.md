# Installation


## Release status

!!! note "Stable releases on PyPI, release candidates on GitHub"
    0.5.0 is the first stable release, and stable releases are on PyPI.
    Release candidates such as 0.5.0rc2 are distributed only as
    [GitHub pre-releases](https://github.com/astroai/torchsky/releases) and are
    never uploaded to PyPI; install one from its pre-release wheels or from
    source.

```bash
pip install torchsky
```

`torchsky` does not own or bundle CFITSIO.
[`torchfits`](https://github.com/astroai/torchfits) is a required runtime
dependency for FITS file and table I/O and installs together with torchsky.

```bash
pip install "torchsky @ file:///path/to/torchsky-0.5.0-*.whl"   # a wheel you built
```

### Supported versions (0.6)

| | Supported and tested |
|---|---|
| Python | 3.12, 3.13, 3.14 (CI runs the full suite on each) |
| PyTorch | 2.13 or newer (`>=2.13`, no upper bound; tested up to 2.15) |
| GPU | Any CUDA build of a supported PyTorch, Apple MPS (float64-only paths raise) |
| OS / architecture | Linux x86_64 (manylinux_2_28), macOS 14+ arm64 |
| torchfits | 1.3.3 or newer |

The native extension is CPU-only C++ that neither includes nor links libtorch:
tensors cross into it through DLPack without a copy, and its operators are
registered with `torch.library` from Python. It also uses the CPython stable ABI.
So one `cp312-abi3` wheel per platform serves CPython 3.12 and newer, CPU and
every CUDA build, and every PyTorch from the floor up. GPU work runs through
PyTorch. Other platforms build from the sdist.

Release wheels target Linux x86_64 (manylinux_2_28) and macOS arm64 (macOS 14.0
or newer); building from source works on the platforms in `pyproject.toml`.
Package metadata requires Python >=3.12, [PyTorch](https://pytorch.org/) >=2.13
and torchfits 1.3.3 or newer. CI tests the floor (pixi environment
`test-torch-min`) and the newest PyTorch available; `scripts/torch_support.json`
records both.

Known issue: with PyTorch 2.15 on macOS 14, MPS matrix products on
transposed inputs abort the process inside PyTorch (`-[MPSNDArray
arrayViewWithShape:strides:]: unrecognized selector`). On that macOS, use
PyTorch 2.14 for MPS work, or run on the CPU.

Optional extras: `torchsky[viz]` (browser sky viewer and matplotlib figures;
`import torchsky` never needs it) and `torchsky[map]` (SVO filter profiles for
`torchsky map --method field`).

## Source Build

To build `torchsky` from source, you'll need a C++ compiler and CMake.

### Prerequisites

- Python **3.12+** (matches `requires-python` in `pyproject.toml`)
- C++20 compatible compiler (GCC or Clang on the supported release platforms)
- CMake 3.15+ for the native project, with 3.21+ recommended for the packaged build environment and current PyTorch/CUDA toolchains
- Ninja (recommended)
- **Python build backend** (see below): `scikit-build-core` and `nanobind`; the build needs no PyTorch

#### Build-backend packages (pip without pixi)

The package uses **scikit-build-core** as the build backend. With default `pip install -e .`, pip creates an isolated build environment and installs those requirements automatically.

If you install **without** build isolation (`--no-build-isolation`), you must have the backend and build-time deps available in the **same** environment as the interpreter:

```bash
python -m pip install "scikit-build-core>=1.0.0" "nanobind>=2.10.2"
python -m pip install "torch>=2.13" "numpy>=1.20.0"   # runtime
```

Exact pins match `[build-system].requires` in `pyproject.toml`.

Optional extras (from `pyproject.toml`):

- **`map`**: `astropy` and `astroquery` for SVO filter profiles used by
  ``torchsky map --method field --field-spectral-source=svo`` (the default source).
  Install with `pip install 'torchsky[map]'` when using PyPI wheels, or install
  those packages in your env.

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

#### PyTorch is not a build dependency

The native extension does not include PyTorch headers or link libtorch, so a
source or editable build needs only a C++20 compiler, CMake, `scikit-build-core`
and `nanobind`. It works the same next to CPU, CUDA and MPS builds of PyTorch,
from pip, conda or Pixi, and no CUDA toolkit is needed. `--no-build-isolation`
is only useful to reuse the backend already installed in your environment.

If an otherwise supported environment crashes while loading the optional native
accelerator, set `TORCHSKY_DISABLE_CPP=1` before importing Torchsky. This keeps
the torch implementations available and disables only the C++ fast paths
(HEALPix, spherical geometry and harmonics, crossmatch, drizzle; WCS has
none). Treat this as a safe diagnostic/fallback, and do not publish native
performance claims from such a run.

### Troubleshooting

Install PyTorch with CUDA or MPS support for torchsky tensor computation:

- **Compiler Errors?**
  Verify your C++ compiler supports C++20. On macOS, ensure Xcode Command Line Tools are installed (`xcode-select --install`).

- **`BackendUnavailable: Cannot import 'scikit_build_core.build'`** with `--no-build-isolation`?
  Install `scikit-build-core` (and `nanobind`) into the active environment, then retry.

- **Permission denied** installing into conda `site-packages`?
  Use a venv or install into a writable prefix.
