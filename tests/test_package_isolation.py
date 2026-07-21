"""Package-isolation gate: torchsky must not own FITS I/O implementation.

All FITS I/O is owned by ``torchfits``. torchsky imports from ``torchfits``
at runtime — that is the intended contract. This gate detects regression to
in-tree FITS implementation code.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tomllib


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "torchsky"
REPO_ROOT = PACKAGE_ROOT.parents[1]


def test_native_torch_abi_contract_is_consistent() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    pixi = tomllib.loads((REPO_ROOT / "pixi.toml").read_text(encoding="utf-8"))

    assert "scikit-build-core>=1.0.0" in pyproject["build-system"]["requires"]
    assert "torch>=2.10,<2.11" in pyproject["build-system"]["requires"]
    assert "torch>=2.10,<2.11" in pyproject["project"]["dependencies"]
    assert any(x.startswith("torchfits>=") for x in pyproject["project"]["dependencies"])
    for section in ("build-dependencies", "host-dependencies", "run-dependencies"):
        assert pixi["package"][section]["pytorch"] == ">=2.10,<2.11"
    assert pixi["dependencies"]["pytorch"] == ">=2.10,<2.11"
    assert pixi["feature"]["prod"]["pypi-dependencies"]["torchfits"].startswith(">=")
    install_docs = (REPO_ROOT / "docs" / "install.md").read_text(encoding="utf-8")
    assert '"torch>=2.10,<2.11"' in install_docs

    cmake = (PACKAGE_ROOT / "cpp_src" / "CMakeLists.txt").read_text(encoding="utf-8")
    bindings = (PACKAGE_ROOT / "cpp_src" / "bindings.cpp").read_text(encoding="utf-8")
    assert "TORCHSKY_BUILD_TORCH_VERSION" in cmake
    assert 'TORCHSKY_TORCH_ABI="2.10"' in cmake
    assert "matching_abi" in bindings


def test_cuda_source_build_requires_a_real_toolkit() -> None:
    cmake = (PACKAGE_ROOT / "cpp_src" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert "CUDAToolkit_FOUND" in cmake
    assert "CUDAToolkit_NVCC_EXECUTABLE" in cmake
    assert "TARGET CUDA::nvrtc" in cmake
    assert "Runtime-only CUDA libraries are" in cmake
    assert "add_library(CUDA::" not in cmake


def test_conda_torch_include_root_is_normalized() -> None:
    cmake = (PACKAGE_ROOT / "cpp_src" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert 'DIR MATCHES "^(.+)/torch/csrc/api/include/?$"' in cmake
    assert "torch/csrc/autograd/autograd.h" in cmake
    assert "list(REMOVE_DUPLICATES TORCH_INCLUDE_DIRS_FILTERED)" in cmake


def test_macos_source_wheels_target_the_portable_arm64_baseline() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    overrides = pyproject["tool"]["scikit-build"]["overrides"]
    macos = next(
        override
        for override in overrides
        if override["if"]["platform-system"] == "^darwin"
    )

    assert macos["cmake"]["define"]["CMAKE_OSX_DEPLOYMENT_TARGET"] == "11.0"


def test_native_extension_rejects_mismatched_torch_runtime() -> None:
    script = """
import torch
torch.__version__ = "2.11.0"
import torchsky._cpp
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "built for PyTorch 2.10.x but found PyTorch 2.11.0" in result.stderr


def test_native_extension_constructs_and_uses_wcs_in_subprocess() -> None:
    script = """
import torch
import torchsky._cpp
from torchsky.wcs import WCS

header = {
    "NAXIS": 2,
    "NAXIS1": 16,
    "NAXIS2": 16,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 8.0,
    "CRPIX2": 8.0,
    "CRVAL1": 180.0,
    "CRVAL2": 0.0,
    "CD1_1": -0.01,
    "CD1_2": 0.0,
    "CD2_1": 0.0,
    "CD2_2": 0.01,
}
world = WCS(header).pixel_to_world(
    torch.tensor([[7.0, 7.0]], dtype=torch.float64)
)
torch.testing.assert_close(
    world,
    torch.tensor([[180.0, 0.0]], dtype=torch.float64),
    atol=1e-9,
    rtol=0.0,
)
"""
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_compute_modules_do_not_load_torchfits_cpp() -> None:
    """Loading an accelerator must not also load the unrelated FITS extension."""
    native_modules = (
        PACKAGE_ROOT / "wcs" / "core.py",
        PACKAGE_ROOT / "wcs" / "sip.py",
        PACKAGE_ROOT / "wcs" / "tpv.py",
        PACKAGE_ROOT / "sphere" / "healpix.py",
        PACKAGE_ROOT / "sphere" / "geom.py",
        PACKAGE_ROOT / "sphere" / "spectral.py",
    )
    for path in native_modules:
        source = path.read_text(encoding="utf-8")
        assert "import torchsky.cpp" not in source
        assert "from .. import cpp" not in source


def test_no_cfitsio_or_zlib_in_cmake() -> None:
    """Native build must not link CFITSIO or zlib."""
    cmake = (PACKAGE_ROOT / "cpp_src" / "CMakeLists.txt").read_text()
    lowered = cmake.lower()
    assert "cfitsio" not in lowered, "CFITSIO must not be linked in torchsky C++"
    assert "zlib" not in lowered, "zlib must not be linked in torchsky C++"
    assert "fits.cpp" not in cmake
    assert "table.cpp" not in cmake
    assert "compression.cpp" not in cmake
    assert "cache.cpp" not in cmake


def test_no_fits_implementation_in_python_source() -> None:
    """No Python FITS implementation code (IO engine, table engine, legacy backends).

    Thin re-export wrappers delegating to `torchfits` are allowed; actual
    FITS header/table/HDU implementation logic is not.
    """
    forbidden_patterns = (
        " _io_engine",
        " _table_engine",
        "class FITSFile",
        "class FITSHeader",
        "class FITSBinaryTable",
    )

    for py_file in PACKAGE_ROOT.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern in text:
                rel = py_file.relative_to(PACKAGE_ROOT)
                raise AssertionError(
                    f"FITS implementation code found in {rel}: "
                    f"contains forbidden pattern '{pattern}'"
                )


def test_no_torch_frame_surface_or_dependency() -> None:
    """Arrow/Polars are the table boundary; torch_frame is not supported."""
    assert not (PACKAGE_ROOT / "frame.py").exists()
    io_facade = (PACKAGE_ROOT / "io" / "__init__.py").read_text(encoding="utf-8")
    for name in ("read_tensor_frame", "to_tensor_frame", "write_tensor_frame"):
        assert name not in io_facade
    for manifest in ("pyproject.toml", "pixi.toml"):
        text = (REPO_ROOT / manifest).read_text(encoding="utf-8").lower()
        assert "pytorch-frame" not in text
