import json
from pathlib import Path
import sys

import pytest
import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from score_smacs0723_native_classical import (  # noqa: E402
    METHODS,
    _audit_manifest_support,
    _root,
    _select_method,
)


def _validation(completeness: float, faintest: float, *, reached: bool = True):
    return {
        "target_purity_reached": reached,
        "pooled": {"completeness": completeness, "purity": 0.95},
        "completeness_by_ab_magnitude": {
            "29": {"completeness": 1.0},
            "30": {"completeness": faintest},
        },
    }


def test_native_selection_rule_and_root_pattern() -> None:
    winner, rows = _select_method(
        {
            METHODS[0]: _validation(0.8, 0.5),
            METHODS[1]: _validation(0.8, 0.4),
        }
    )
    assert winner == METHODS[0]
    assert rows[0]["faintest_completeness"] == 0.5
    assert _root("/runs/{band}", "F090W") == Path("/runs/F090W")


def test_native_support_audit_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "batch_id": "validation-000",
                "target_xy": [[1.2, 1.1], [3.0, 3.0]],
            }
        )
    )
    support = torch.zeros((4, 4), dtype=torch.bool)
    support[1, 1] = True
    with pytest.raises(ValueError, match="outside the native common support"):
        _audit_manifest_support((manifest,), support)
