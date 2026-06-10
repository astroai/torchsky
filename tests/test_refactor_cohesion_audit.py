from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _audit_module():
    path = ROOT / "scripts" / "audit_refactor_cohesion.py"
    spec = importlib.util.spec_from_file_location("audit_refactor_cohesion", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_refactor_cohesion_audit_has_no_hard_failures() -> None:
    audit = _audit_module()
    results = audit.run_audit(ROOT)
    failures = [result for result in results if result.status == "fail"]
    assert not failures
    assert audit._exit_code(results, strict_warnings=False) == 0


def test_refactor_cohesion_audit_keeps_large_file_pressure_visible() -> None:
    audit = _audit_module()
    results = audit.run_audit(ROOT)
    warnings = [result for result in results if result.status == "warn"]
    assert any("map/image_plane.py" in result.name for result in warnings)
    assert any("map/field/core.py" in result.name for result in warnings)
    assert audit._exit_code(results, strict_warnings=True) == 1
