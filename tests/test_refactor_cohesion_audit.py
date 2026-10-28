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


def test_refactor_cohesion_audit_checks_structural_boundaries_not_line_counts() -> None:
    audit = _audit_module()
    results = audit.run_audit(ROOT)
    warnings = [result for result in results if result.status == "warn"]
    assert not warnings
    assert any(
        result.name == "cohesion:src/torchsky/map/field/core.py"
        and result.status == "pass"
        for result in results
    )
    assert any(
        result.name == "cohesion:src/torchsky/map/image_plane_coadd.py"
        and result.status == "pass"
        for result in results
    )
    assert audit._exit_code(results, strict_warnings=True) == 0
