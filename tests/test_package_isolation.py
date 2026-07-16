"""Package-isolation gate: torchsky must not own FITS I/O implementation.

All FITS I/O is owned by ``torchfits``. torchsky imports from ``torchfits``
at runtime — that is the intended contract. This gate detects regression to
in-tree FITS implementation code.
"""

from __future__ import annotations

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
