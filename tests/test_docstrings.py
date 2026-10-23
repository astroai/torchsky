"""Public docstring coverage may only improve.

Runs the audit in ``scripts/docstring_audit.py`` against the committed baseline.
A new public name with no summary, or a Tier-1 name that loses a required
section, fails here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docstring_ratchet_holds() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [sys.executable, "scripts/docstring_audit.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
