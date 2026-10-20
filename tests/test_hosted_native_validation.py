from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "native-portability.yml"
RUNTIME_SCRIPT = REPO_ROOT / "scripts" / "validate_native_runtime.py"
CUDA_DIAGNOSTIC = REPO_ROOT / "scripts" / "diagnose_cuda_build.py"


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

    assert "runs-on: ubuntu-latest" in source
    assert "runs-on: macos-14" in source
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
    assert source.count("timeout-minutes:") == 3
    assert "cancel-in-progress: true" in source
    assert source.count("cache: pip") == 2
    assert "cache: true" in source
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
    assert "CUDA_HOME:" in workflow
    assert "CUDAToolkit_ROOT:" in workflow
    assert "diagnose_cuda_build.py --require-cuda-torch" in workflow
    assert "pip install -e . --no-build-isolation -v" in workflow
    assert "Standard GitHub-hosted runners have no NVIDIA GPU" in workflow
    assert 'cuda-nvcc = "13.0.*"' in pixi
    assert 'cuda-nvrtc-dev = "13.0.*"' in pixi
    assert 'torch = "==2.13.0"' in pixi


def test_runtime_parity_comparison_rejects_numerical_drift() -> None:
    module = _runtime_module()

    module._assert_close([1.0, 2.0], [1.0, 2.0 + 1e-15])
    with pytest.raises(AssertionError, match=r"payload\[1\] differs"):
        module._assert_close([1.0, 2.0], [1.0, 2.1])


def test_cuda_diagnostic_explains_cpu_only_upstream_torch() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"CUDA_HOME", "CUDAToolkit_ROOT"}
    }
    result = subprocess.run(
        [sys.executable, str(CUDA_DIAGNOSTIC), "--require-cuda-torch"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "installed upstream PyTorch wheel is CPU-only" in result.stderr
    payload = json.loads(result.stdout)
    assert payload["torch_cuda"] is None
    assert payload["cuda_available"] is False
