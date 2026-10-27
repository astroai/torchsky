"""Package-isolation gate: torchsky must not own FITS I/O implementation.

All FITS I/O is owned by ``torchfits``. torchsky imports from ``torchfits``
at runtime — that is the intended contract. This gate detects regression to
in-tree FITS implementation code.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "torchsky"
REPO_ROOT = PACKAGE_ROOT.parents[1]


def test_native_calls_are_confined_to_dispatch_or_legacy_allowlist() -> None:
    found: dict[str, set[str]] = {}
    pattern = re.compile(r"\b_cpp\.([A-Za-z_][A-Za-z0-9_]*)")
    for path in PACKAGE_ROOT.rglob("*.py"):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        names = set(pattern.findall(path.read_text(encoding="utf-8")))
        if names:
            found[relative] = names
    assert found == {}


def test_fits_facade_is_absent_and_tensor_operators_do_not_require_native():
    script = """
import sys
sys.modules['torchsky._cpp'] = None
import importlib.util
import torchfits
from torchsky.map.operators import LowRankChromaticPSF
assert importlib.util.find_spec('torchsky.io') is None
assert importlib.util.find_spec('torchsky.table') is None
assert callable(torchfits.read)
assert LowRankChromaticPSF(grid_side=2, spectral_degree=1,
                          base_sigma_pix=0.8, kernel_size=5) is not None
"""
    subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, check=True)


def test_native_torch_abi_contract_is_consistent() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    pixi = tomllib.loads((REPO_ROOT / "pixi.toml").read_text(encoding="utf-8"))

    assert "scikit-build-core>=1.0.0" in pyproject["build-system"]["requires"]
    assert pyproject["project"]["requires-python"] == ">=3.12"
    assert "torch>=2.13,<2.14" in pyproject["build-system"]["requires"]
    assert "torch>=2.13,<2.14" in pyproject["project"]["dependencies"]
    assert "torchfits>=1.0.0" in pyproject["project"]["dependencies"]
    for section in ("build-dependencies", "host-dependencies", "run-dependencies"):
        assert pixi["package"][section]["pytorch"] == ">=2.13,<2.14"
    assert pixi["package"]["run-dependencies"]["numpy"] == ">=1.20"
    assert pixi["dependencies"]["pytorch"] == "==2.13.0"
    assert pixi["feature"]["prod"]["pypi-dependencies"]["torchfits"] == "==1.0.0"
    assert pixi["feature"]["py312"]["dependencies"]["python"] == "3.12.*"
    assert pixi["feature"]["py313"]["dependencies"]["python"] == "3.13.*"
    install_docs = (REPO_ROOT / "docs" / "install.md").read_text(encoding="utf-8")
    assert '"torch>=2.13,<2.14"' in install_docs

    cmake = (PACKAGE_ROOT / "cpp_src" / "CMakeLists.txt").read_text(encoding="utf-8")
    bindings = (PACKAGE_ROOT / "cpp_src" / "bindings.cpp").read_text(encoding="utf-8")
    assert "TORCHSKY_BUILD_TORCH_VERSION" in cmake
    assert 'TORCHSKY_TORCH_ABI="${TORCHSKY_BUILD_TORCH_ABI}"' in cmake
    assert "cxx_std_20" in cmake
    assert "TORCHSKY_RUNTIME_PREFIX" in cmake
    assert "Prioritizing libtorch headers" in cmake
    assert "matching_abi" in bindings


def test_native_extension_uses_separate_translation_units() -> None:
    cpp_root = PACKAGE_ROOT / "cpp_src"
    cmake = (cpp_root / "CMakeLists.txt").read_text(encoding="utf-8")
    bindings = (cpp_root / "bindings.cpp").read_text(encoding="utf-8")

    for source in ("bindings.cpp", "geom.cpp", "healpix.cpp", "wcs.cpp"):
        assert cmake.count(source) == 1
    assert "OBJECT_DEPENDS" not in cmake
    assert '#include "geom.cpp"' not in bindings
    assert '#include "healpix.cpp"' not in bindings
    assert '#include "wcs.cpp"' not in bindings
    for name in ("geom", "healpix", "wcs"):
        header = cpp_root / "include" / "torchsky" / f"{name}.h"
        implementation = (cpp_root / f"{name}.cpp").read_text(encoding="utf-8")
        assert header.is_file()
        assert f'#include "torchsky/{name}.h"' in implementation
    assert '#include "torchsky/healpix.h"' in bindings
    assert '#include "torchsky/fixed_operators.h"' in bindings
    assert '#include "torchsky/wcs_operators.h"' in bindings


def test_native_extension_keeps_optimized_isa_out_of_baseline_sources() -> None:
    cpp_root = PACKAGE_ROOT / "cpp_src"
    cmake = (cpp_root / "CMakeLists.txt").read_text(encoding="utf-8")
    bindings = (cpp_root / "bindings.cpp").read_text(encoding="utf-8")

    assert "list(APPEND TORCHSKY_COMPILE_OPTIONS -mavx2 -mfma)" not in cmake
    assert "torchsky_ishioka_avx2" in cmake
    assert "target_compile_options(torchsky_ishioka_avx2 PRIVATE -O3 -mavx2 -mfma)" in cmake
    assert "$<TARGET_OBJECTS:torchsky_ishioka_avx2>" in cmake
    assert "torchsky_ishioka_neon" in cmake
    assert "TORCHSKY_ENABLE_IPO" in cmake
    assert "ensure_spectral_operator_schemas_linked" in bindings
    assert "ensure_spectral_cpu_operators_linked" in bindings


def test_native_extension_uses_supported_torch_linkage() -> None:
    cmake = (PACKAGE_ROOT / "cpp_src" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert "if(TARGET torch)" in cmake
    assert "target_link_libraries(_cpp_ext PRIVATE torch)" in cmake
    assert "elseif(TORCH_LIBRARIES)" in cmake
    assert "${TORCH_LIBRARIES}" in cmake


def test_cuda_source_build_requires_a_real_toolkit() -> None:
    cmake = (PACKAGE_ROOT / "cpp_src" / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "CUDAToolkit_FOUND" in cmake
    assert "CUDAToolkit_NVCC_EXECUTABLE" in cmake
    assert "TARGET CUDA::nvrtc" in cmake
    assert "Runtime-only CUDA libraries are" in cmake
    assert "add_library(CUDA::" not in cmake


def test_conda_torch_include_root_is_normalized() -> None:
    cmake = (PACKAGE_ROOT / "cpp_src" / "CMakeLists.txt").read_text(encoding="utf-8")

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
torch.__version__ = "2.14.0"
import torchsky._cpp
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "built for PyTorch 2.13.x but found PyTorch 2.14.0" in result.stderr


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
        assert "torchfits.cpp" not in source
        assert "from .. import cpp" not in source


def test_wcs_can_disable_optional_native_extension_in_subprocess() -> None:
    script = """
import sys

import torch
import torchsky._native as native
from torchsky import _ops
from torchsky.wcs import WCS

assert native.cpp is None
assert "torchsky._cpp" not in sys.modules
assert not _ops.has_wcs_dispatcher()
wcs = WCS({
    "NAXIS": 2,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 1.0,
    "CRPIX2": 1.0,
    "CRVAL1": 150.0,
    "CRVAL2": 2.0,
    "CDELT1": -0.0001,
    "CDELT2": 0.0001,
})
ra, dec = wcs.all_pix2world(
    torch.tensor([0.0], dtype=torch.float64),
    torch.tensor([0.0], dtype=torch.float64),
    0,
)
torch.testing.assert_close(ra, torch.tensor([150.0], dtype=torch.float64))
torch.testing.assert_close(dec, torch.tensor([2.0], dtype=torch.float64))
"""
    env = {**os.environ, "TORCHSKY_DISABLE_CPP": "1"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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


def test_no_legacy_merged_cpp_shim() -> None:
    """``torchsky.cpp`` was a compatibility merge of torchfits.cpp and _cpp."""
    import importlib.util

    assert importlib.util.find_spec("torchsky.cpp") is None
    assert not (PACKAGE_ROOT / "cpp.py").exists()
    assert not (PACKAGE_ROOT / "cpp.pyi").exists()


def test_no_legacy_cfitsio_vendoring() -> None:
    """The pre-torchfits CFITSIO vendor tree and CI steps must stay gone."""
    repo = PACKAGE_ROOT.parents[1]
    assert not (repo / "extern" / "vendor.sh").exists()
    assert not (repo / "extern" / "cfitsio").exists()
    manifest = (repo / "MANIFEST.in").read_text()
    assert "extern/" not in manifest
    for workflow in (repo / ".github" / "workflows").glob("*.yml"):
        text = workflow.read_text()
        assert "extern/vendor.sh" not in text, workflow


def test_pocketfft_notices_are_included_in_source_distributions() -> None:
    repo = PACKAGE_ROOT.parents[1]
    manifest = (repo / "MANIFEST.in").read_text()
    license_text = (
        PACKAGE_ROOT / "cpp_src" / "third_party" / "pocketfft.LICENSE.md"
    ).read_text()

    assert "recursive-include src/torchsky/cpp_src" in manifest
    assert "*.md" in manifest
    assert "Copyright (C) 2010-2024 Max-Planck-Society" in license_text
    assert "Copyright (C) 2019-2020 Peter Bell" in license_text
    assert "Copyright (C) 2024 Cris Luengo" in license_text


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
    for name in ("io", "table.py", "hdu.py", "datasets.py", "dataloader.py"):
        assert not (PACKAGE_ROOT / name).exists()
    for manifest in ("pyproject.toml", "pixi.toml"):
        text = (REPO_ROOT / manifest).read_text(encoding="utf-8").lower()
        assert "pytorch-frame" not in text
