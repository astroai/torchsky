"""Package-isolation gate: torchsky must not own FITS I/O implementation.

All FITS I/O is owned by ``torchfits``. torchsky imports from ``torchfits``
at runtime — that is the intended contract. This gate detects regression to
in-tree FITS implementation code.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "torchsky"


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
