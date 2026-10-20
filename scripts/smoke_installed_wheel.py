from __future__ import annotations

import os
import subprocess
import sys

import torch
import torchsky
from torchsky import _cpp
from torchsky._ops import has_wcs_dispatcher


if not has_wcs_dispatcher():
    raise AssertionError("installed wheel did not register WCS dispatcher operators")
if not torch._C._dispatch_has_kernel_for_dispatch_key(
    "torchsky::wcs_pixel_to_world", "CPU"
):
    raise AssertionError("installed wheel did not register the CPU WCS dispatcher kernel")
if _cpp.selected_cpu_isa() not in {"scalar", "avx2+fma", "neon"}:
    raise AssertionError("installed wheel reported an invalid CPU ISA")

scalar = subprocess.run(
    [
        sys.executable,
        "-c",
        "from torchsky import _cpp; assert _cpp.selected_cpu_isa() == 'scalar'",
    ],
    env={**os.environ, "TORCHSKY_CPU_ISA": "scalar"},
    capture_output=True,
    text=True,
    check=False,
)
if scalar.returncode != 0:
    raise AssertionError(scalar.stdout + scalar.stderr)

disabled = subprocess.run(
    [
        sys.executable,
        "-c",
        "import sys; import torchsky._native as n; from torchsky._ops import has_wcs_dispatcher; assert n.cpp is None; assert not has_wcs_dispatcher(); assert 'torchsky._cpp_ext' not in sys.modules",
    ],
    env={**os.environ, "TORCHSKY_DISABLE_CPP": "1"},
    capture_output=True,
    text=True,
    check=False,
)
if disabled.returncode != 0:
    raise AssertionError(disabled.stdout + disabled.stderr)

print("installed wheel smoke passed:", torchsky.__file__)
