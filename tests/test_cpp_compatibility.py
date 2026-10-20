from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import torchsky._cpp as cpp


PACKAGE_ROOT = Path(cpp.__file__).parent
DEFERRED_CALL_SITES = {
    Path("sphere/healpix.py"),
    Path("sphere/spectral.py"),
}


def test_compatibility_symbol_inventory_is_complete_and_disjoint() -> None:
    migrated = set(cpp.MIGRATED_COMPAT_SYMBOLS)
    python_compat = set(cpp.PYTHON_COMPAT_SYMBOLS)
    deferred = set(cpp.DEFERRED_NATIVE_SYMBOLS)
    assert migrated
    assert deferred == {
        "healpix_query_disc_torch_cpu",
        "_healpix_spin_map2alm_from_basis_cpu",
        "_healpix_spin_alm2map_from_basis_cpu",
    }
    assert migrated.isdisjoint(python_compat | deferred)
    assert python_compat.isdisjoint(deferred)
    assert set(cpp.__all__) == migrated | python_compat | deferred | {
        "selected_cpu_isa"
    }
    assert all(hasattr(cpp, name) for name in cpp.__all__)


def test_extension_exports_only_deferred_tensor_bindings() -> None:
    exported = {name for name in dir(cpp._ext) if not name.startswith("__")}
    assert exported == set(cpp.DEFERRED_NATIVE_SYMBOLS) | {"selected_cpu_isa"}
    bindings = (PACKAGE_ROOT / "cpp_src" / "bindings.cpp").read_text(encoding="utf-8")
    assert bindings.count("m.def(") == len(cpp.DEFERRED_NATIVE_SYMBOLS) + 1
    assert all(f'"{name}"' in bindings for name in cpp.DEFERRED_NATIVE_SYMBOLS)
    assert all(f'"{name}"' not in bindings for name in cpp.MIGRATED_COMPAT_SYMBOLS)
    assert all(f'"{name}"' not in bindings for name in cpp.PYTHON_COMPAT_SYMBOLS)


def test_fixed_old_name_warns_and_dispatches_with_parity() -> None:
    pixels = torch.tensor([0, 1, 2], dtype=torch.int64)
    expected = torch.ops.torchsky.healpix_ring2nest(1, pixels)
    with pytest.warns(DeprecationWarning, match="torchsky._ops"):
        actual = cpp.healpix_ring2nest_torch_cpu(1, pixels)
    torch.testing.assert_close(actual, expected)


def test_spectral_old_name_preserves_historical_shape_and_warns() -> None:
    starts = torch.tensor([0], dtype=torch.int64)
    lengths = torch.tensor([1], dtype=torch.int64)
    row = torch.ones(1, dtype=torch.complex128)
    expected = torch.stack(
        torch.ops.torchsky.spectral_ring_fourier_modes_spin_conj(
            row, starts, lengths, 0
        )
    )
    with pytest.warns(DeprecationWarning, match="torch.ops.torchsky"):
        actual = cpp._healpix_ring_fourier_modes_spin_conj_cpu(row, starts, lengths, 0)
    torch.testing.assert_close(actual, expected)


def test_wcs_old_name_preserves_out_parameter_contract_and_warns() -> None:
    x = torch.tensor([0.0, 1.0], dtype=torch.float64)
    y = torch.tensor([0.0, -1.0], dtype=torch.float64)
    ra = torch.empty_like(x)
    dec = torch.empty_like(y)
    idx = torch.empty((0, 3), dtype=torch.int64)
    coeff = torch.empty(0, dtype=torch.float64)
    args = (
        x,
        y,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        "CAR",
        1.0,
        4.0,
        3.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        "center",
        0.0,
        0.0,
        False,
        idx,
        coeff,
        idx,
        coeff,
    )
    expected = torch.ops.torchsky.wcs_pixel_to_world(*args)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        result = cpp.wcs_pixel_to_world_fused_cpu(
            x,
            y,
            ra,
            dec,
            *args[2:],
        )
    assert result is None
    torch.testing.assert_close(ra, expected[0], equal_nan=True)
    torch.testing.assert_close(dec, expected[1], equal_nan=True)


def test_loader_reload_and_disable_cpp_subprocess_contract() -> None:
    assert importlib.reload(cpp).DEFERRED_NATIVE_SYMBOLS
    script = """
import sys
import torchsky._native as native
assert native.cpp is None
assert 'torchsky._cpp' not in sys.modules
assert 'torchsky._cpp_ext' not in sys.modules
from torchsky import _ops
assert not _ops.has_fixed_dispatcher()
assert not _ops.has_spectral_dispatcher()
assert not _ops.has_wcs_dispatcher()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PACKAGE_ROOT.parents[1],
        env={**os.environ, "TORCHSKY_DISABLE_CPP": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_production_direct_cpp_calls_are_limited_to_deferred_bindings() -> None:
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        relative = path.relative_to(PACKAGE_ROOT)
        if relative in {Path("_cpp.py"), Path("_native.py"), Path("_ops.py")}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=os.fspath(path))
        imports_cpp = any(
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.endswith("_native")
            and any(
                alias.name == "cpp" and alias.asname == "_cpp" for alias in node.names
            )
            for node in ast.walk(tree)
        )
        direct_names = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "_cpp"
        }
        if imports_cpp or direct_names:
            if relative not in DEFERRED_CALL_SITES:
                offenders.append(os.fspath(relative))
                continue
            allowed = (
                {"healpix_query_disc_torch_cpu"}
                if relative == Path("sphere/healpix.py")
                else {
                    "_healpix_spin_map2alm_from_basis_cpu",
                    "_healpix_spin_alm2map_from_basis_cpu",
                }
            )
            if not direct_names <= allowed:
                offenders.append(os.fspath(relative))
    assert offenders == []
