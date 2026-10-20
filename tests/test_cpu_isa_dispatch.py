from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import pytest


SCRIPT = """
import json
import torch
from torchsky import _cpp

torch.manual_seed(90210)
lmax = 8
nalm = (lmax + 1) * (lmax + 2) // 2
alms = torch.complex(
    torch.randn(2, nalm, dtype=torch.float64),
    torch.randn(2, nalm, dtype=torch.float64),
)
theta = torch.linspace(0.1, torch.pi - 0.1, 15, dtype=torch.float64)
out = torch.ops.torchsky.spectral_spin_interpolate_recurrence(
    alms, theta, lmax, lmax, 2
)
print(json.dumps({
    "isa": _cpp.selected_cpu_isa(),
    "real": out.real.reshape(-1).tolist(),
    "imag": out.imag.reshape(-1).tolist(),
}))
"""


def _run(isa: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env={**os.environ, "TORCHSKY_CPU_ISA": isa, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_forced_scalar_and_native_ishioka_parity() -> None:
    scalar = _run("scalar")
    native = _run("native")
    assert scalar.returncode == 0, scalar.stderr
    assert native.returncode == 0, native.stderr
    scalar_payload = json.loads(scalar.stdout)
    native_payload = json.loads(native.stdout)
    assert scalar_payload["isa"] == "scalar"
    assert native_payload["isa"] in {"scalar", "avx2+fma", "neon"}
    assert native_payload["real"] == pytest.approx(
        scalar_payload["real"], rel=2e-14, abs=2e-14
    )
    assert native_payload["imag"] == pytest.approx(
        scalar_payload["imag"], rel=2e-14, abs=2e-14
    )


def test_forced_unsupported_isa_is_rejected() -> None:
    unsupported = "neon" if platform.machine().lower() in {"x86_64", "amd64"} else "avx2"
    result = _run(unsupported)
    assert result.returncode != 0
    assert "is not supported by this build and CPU/OS" in result.stderr


@pytest.mark.skipif(
    platform.machine().lower() not in {"x86_64", "amd64"},
    reason="x86_64 binary instruction inspection",
)
def test_scalar_leaf_has_no_avx_instructions() -> None:
    objdump = shutil.which("objdump")
    if objdump is None:
        pytest.skip("objdump unavailable")
    from torchsky import _cpp_ext

    binary = Path(_cpp_ext.__file__).resolve()
    result = subprocess.run(
        [objdump, "-d", "--demangle", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(result.stderr)
    section = result.stdout.split("torchsky::ishioka_k_step_scalar", 1)[1]
    section = section.split("\n\n", 1)[0].lower()
    assert "ymm" not in section
    assert "vfmadd" not in section
