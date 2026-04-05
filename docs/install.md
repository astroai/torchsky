# Installation Guide

`torchsky` can be installed via pip or built from source.

## Standard Installation (Pip)

The easiest way to install `torchsky` is using `pip`:

```bash
pip install torchsky
```

This will download and install the latest stable version from PyPI.

## Building from Source

To build `torchsky` from source, you'll need a C++ compiler and CMake.

### Prerequisites

- Python 3.10+ (matches `requires-python` in `pyproject.toml`)
- C++17 compatible compiler (GCC, Clang, MSVC)
- CMake 3.21+
- Ninja (recommended)

### Steps

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/sfabbro/torchsky.git
    cd torchsky
    ```

2.  **Vendor Dependencies:**

    `torchsky` vendors its C dependency (`cfitsio`) to simplify installation. Run the vendoring script to prepare the source tree:

    ```bash
    ./extern/vendor.sh
    ```

    This script downloads the required version of `cfitsio` and places it in the `extern/` directory.
    It resolves latest tags by default; pin version with `--cfitsio-version` for reproducible builds.

3.  **Install:**

    You can then install the package using `pip` in editable mode for development:

    ```bash
    pip install -e .
    ```

    Or build a release version:

    ```bash
    pip install .
    ```

### Troubleshooting

- **Missing `cfitsio`?**
  Ensure you ran `./extern/vendor.sh` before installing. The build system expects these sources to be present in `extern/`.

- **Compiler Errors?**
  Verify your C++ compiler supports C++17. On macOS, ensure Xcode Command Line Tools are installed (`xcode-select --install`).
