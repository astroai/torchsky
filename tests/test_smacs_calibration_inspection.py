from __future__ import annotations

import importlib.util
import sys
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
    assert inspection.count('-e bench -- \\\n') == 2


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
    assert "import torchfits" in parity
    assert "astropy.io" not in parity
