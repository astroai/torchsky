from __future__ import annotations

import platform
from pathlib import Path
import subprocess

import pytest

import torchsky._cpp as cpp


def test_tensor_bridge_is_isolated_to_deferred_bindings() -> None:
    root = Path(cpp.__file__).parent / "cpp_src"
    bindings = (root / "bindings.cpp").read_text(encoding="utf-8")
    bridge = (root / "torch_compat.h").read_text(encoding="utf-8")
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "echo_tensor" not in bindings
    assert "python_to_tensor" in bindings
    assert "tensor_to_python" in bindings
    assert "query_disc and experimental basis" in bridge
    assert "query_disc and experimental basis" in cmake
    assert "THPVariable_Wrap" in bridge
    assert "fromDLPack" in bridge
    assert "TORCH_PYTHON_LIBRARY" in cmake


@pytest.mark.skipif(platform.system() == "Windows", reason="no portable DLL lister")
def test_deferred_bridge_dynamic_dependency_is_explicit() -> None:
    extension = Path(cpp._ext.__file__)
    command = (
        ["otool", "-L", str(extension)]
        if platform.system() == "Darwin"
        else ["ldd", str(extension)]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "torch_python" in result.stdout
