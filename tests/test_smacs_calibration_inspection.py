from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))


def _inspection_module() -> ModuleType:
    path = BENCHMARKS / "export_smacs0723_calibration_inspection.py"
    spec = importlib.util.spec_from_file_location("smacs_calibration_inspection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import inspection module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inspection_selects_worst_residuals_and_writes_png(tmp_path: Path) -> None:
    inspection = _inspection_module()
    accepted = [
        {
            "status": "accepted",
            "split": "test",
            "band": "F150W",
            "source_id": f"accepted-{index}",
            "exposure": "exposure.fits",
            "pruned_residual": float(index),
            "pruned_absolute_error": float(index),
        }
        for index in range(4)
    ]
    rejected = [
        {
            "status": "rejected",
            "split": "test",
            "band": "F150W",
            "source_id": "rejected-0",
            "exposure": "exposure.fits",
            "reason": "insufficient_support",
        }
    ]
    selected = inspection._selected_rows(accepted, rejected, per_group=2)
    assert [row["source_id"] for row in selected[:2]] == [
        "accepted-3",
        "accepted-2",
    ]
    assert selected[-1]["source_id"] == "rejected-0"

    output = tmp_path / "sheet.png"
    inspection._write_contact_sheet(
        output,
        [(selected[0], torch.arange(25, dtype=torch.float32).reshape(5, 5))],
    )
    assert output.stat().st_size > 0


def test_canfar_inspection_uses_visualization_environment() -> None:
    runner = (ROOT / "scripts" / "run_canfar_adass_depth.sh").read_text(
        encoding="utf-8"
    )
    inspection = runner.split("  calibration-inspect)", 1)[1].split("  foam)", 1)[0]
    assert inspection.count("pixi_run bench \\\n") == 2


def test_canfar_full_mosaic_parity_is_strict_and_torchfits_only() -> None:
    runner = (ROOT / "scripts" / "run_canfar_adass_depth.sh").read_text(
        encoding="utf-8"
    )
    parity = runner.split("  mosaic-parity)", 1)[1].split("  native-score)", 1)[0]

    assert "--full-mosaic-parity" in parity
    assert "--expected-exposures" in parity
    assert "bootstrap_map_compat_tools.py" in parity
    assert "--force-source" in parity
    assert '--jobs "${TORCHSKY_SPIN_THREADS:-1}"' in parity
    assert "parity_env=${TORCHSKY_MAP_COMPAT_ENV:-map-compat}" in parity
    assert parity.count('pixi_run "${parity_env}"') == 5
    assert "torch.cuda.is_available()" in parity
    assert "torch.cuda.get_device_name()" in parity
    assert "import torchfits" in parity
    assert "astropy.io" not in parity
    assert "native_build_env=${TORCHSKY_MAP_COMPAT_ENV:-map-compat}" in runner
    assert 'ensure_native_build "${native_build_env}"' in runner


def test_canfar_depth_runner_accepts_authorized_skaha_compute_images() -> None:
    runner = (ROOT / "scripts" / "run_canfar_adass_depth.sh").read_text(
        encoding="utf-8"
    )

    assert "skaha/base-notebook:latest" in runner
    assert "skaha/astroflow-cuda:25.03" in runner


def test_canfar_depth_runner_replaces_stale_pixi() -> None:
    runner = (ROOT / "scripts" / "run_canfar_adass_depth.sh").read_text(
        encoding="utf-8"
    )

    assert "pixi_version=${TORCHSKY_PIXI_VERSION:-0.78.0}" in runner
    assert '"$(pixi --version 2>/dev/null)" != "pixi ${pixi_version}"' in runner
    assert 'export PIXI_VERSION="${pixi_version}"' in runner


def test_map_compat_cuda_environment_uses_cuda_and_official_comparators() -> None:
    pixi = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
    environment = pixi["environments"]["map-compat-cuda"]

    assert environment["no-default-feature"] is True
    assert set(environment["features"]) == {"cuda", "map-compat"}


def test_canfar_runtime_is_selected_after_each_environment_activation(tmp_path):
    runner = (ROOT / "scripts/run_canfar_adass_depth.sh").read_text()
    function = (
        "pixi_run() {" + runner.split("pixi_run() {", 1)[1].split("\n}", 1)[0] + "\n}"
    )
    # Simulate activation overwriting the inherited library path. The command
    # must still see the runtime for its own environment, not the previous one.
    mock = """
pixi() {
  while [[ "$1" != "--" ]]; do shift; done
  shift
  LD_LIBRARY_PATH=/wrong/activation PYTHONNOUSERSITE=0 "$@"
}
source_root=$1
shift
pixi_run "$@"
"""
    inherited = dict(os.environ, LD_LIBRARY_PATH="/existing/runtime")
    probe = "import json, os; print(json.dumps([os.environ['LD_LIBRARY_PATH'], os.environ['PYTHONNOUSERSITE']]))"
    for environment in ("default", "bench", "smacs-cuda", "map-compat-cuda"):
        result = subprocess.run(
            [
                "bash",
                "-c",
                function + mock,
                "runtime-check",
                str(tmp_path),
                environment,
                sys.executable,
                "-c",
                probe,
            ],
            check=True,
            capture_output=True,
            text=True,
            env=inherited,
        )
        assert json.loads(result.stdout) == [
            f"{tmp_path}/.pixi/envs/{environment}/lib:/existing/runtime",
            "1",
        ]
    assert runner.count("pixi run --manifest-path") == 1
