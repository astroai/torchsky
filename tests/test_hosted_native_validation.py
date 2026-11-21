from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "native-portability.yml"
RUNTIME_SCRIPT = REPO_ROOT / "scripts" / "validate_native_runtime.py"


def _runtime_module():
    spec = importlib.util.spec_from_file_location(
        "validate_native_runtime", RUNTIME_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_portability_workflow_covers_hosted_cpu_isa_contracts() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")

    # Ubuntu 26.04 ships x86-64-v3 packages; emulation and sanitizer jobs stay on 24.04.
    assert source.count("runs-on: ubuntu-24.04") == 2
    assert "runs-on: macos-15" in source
    assert 'test "$(uname -m)" = arm64' in source
    assert "CMAKE_ARGS: -DTORCHSKY_ENABLE_IPO=OFF" in source
    assert "CMAKE_BUILD_PARALLEL_LEVEL: '2'" in source
    assert "--candidate-isa avx2 --expected-candidate-isa avx2+fma" in source
    assert "--candidate-isa neon --expected-candidate-isa neon" in source
    assert "qemu-x86_64 -cpu Nehalem" in source
    assert "TORCHSKY_CPU_ISA=auto" in source
    assert "probe --expected-isa scalar" in source
    assert "inspect-scalar-object" in source
    assert "ishioka_scalar.cpp.o" in source
    assert source.count("timeout-minutes:") == 4
    assert "cancel-in-progress: true" in source
    assert source.count("cache: pip") == 3
    # setup-pixi rejects cache: true together with run-install: false.
    assert "cache: true" not in source
    assert "run-install: false" in source


def test_native_portability_workflow_builds_and_smokes_real_extensions() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "torch.version.cuda is None" in source
    assert "python -m build --wheel --no-isolation" in source
    assert "scripts/check_distribution.py --wheel" in source
    assert source.count("'twine>=5'") == 2
    assert source.count("scripts/smoke_installed_wheel.py") == 2
    assert source.count("scripts/validate_native_runtime.py") >= 5
    assert "continue-on-error" not in source
    assert "TORCHSKY_DISABLE_CPP" not in source


def test_cuda_hosted_job_provisions_toolkit_and_reports_runner_scope() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    pixi = (REPO_ROOT / "pixi.toml").read_text(encoding="utf-8")

    assert "linux-cuda-torch-native:" in workflow
    assert "pixi install --frozen -e cuda-bench" in workflow
    assert "diagnose_cuda_build" not in workflow
    assert "pip install -e . --no-build-isolation -v" in workflow
    assert "Standard GitHub-hosted runners have no NVIDIA GPU" in workflow
    support = json.loads(
        (REPO_ROOT / "scripts" / "torch_support.json").read_text(encoding="utf-8")
    )
    assert f'torch = ">={support["torch_min"]}"' in pixi


def test_runtime_parity_comparison_rejects_numerical_drift() -> None:
    module = _runtime_module()

    module._assert_close([1.0, 2.0], [1.0, 2.0 + 1e-15])
    with pytest.raises(AssertionError, match=r"payload\[1\] differs"):
        module._assert_close([1.0, 2.0], [1.0, 2.1])
