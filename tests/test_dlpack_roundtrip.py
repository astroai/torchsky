from __future__ import annotations

import platform
from pathlib import Path
import subprocess

import pytest

import torchsky._cpp as cpp


def test_native_extension_has_no_python_tensor_bridge() -> None:
    root = Path(cpp.__file__).parent / "cpp_src"
    bindings = (root / "bindings.cpp").read_text(encoding="utf-8")
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for pattern in ("*.cpp", "*.h")
        for path in root.rglob(pattern)
    )
    assert "echo_tensor" not in bindings
    assert not (root / "torch_compat.h").exists()
    assert "python_to_tensor" not in sources
    assert "tensor_to_python" not in sources
    assert "THPVariable_Wrap" not in sources
    assert "fromDLPack" not in sources
    assert "TORCH_PYTHON_LIBRARY" not in cmake
    assert "torch_python" not in cmake


@pytest.mark.skipif(platform.system() == "Windows", reason="no portable DLL lister")
def test_native_extension_does_not_link_torch_python() -> None:
    extension = Path(cpp._ext.__file__)
    command = (
        ["otool", "-L", str(extension)]
        if platform.system() == "Darwin"
        else ["ldd", str(extension)]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "torch_python" not in result.stdout
