"""Gate for the runnable example scripts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples" / "test_examples.py"


def test_every_example_script_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"examples/test_examples.py failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
    )


def test_runner_discovers_only_standalone_scripts() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import runpy; runpy.run_path(" + repr(str(RUNNER)) + ", run_name='not_main')"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
