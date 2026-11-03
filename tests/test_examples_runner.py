"""Gate for the runnable example scripts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "examples" / "test_examples.py"


def test_every_example_script_runs() -> None:
    # Offline by default: real-data examples run only from the local sample cache
    # and skip (exit 77) otherwise. Set TORCHSKY_EXAMPLE_NETWORK=1 to fetch data.
    env = dict(os.environ)
    if env.get("TORCHSKY_EXAMPLE_NETWORK") != "1":
        env.setdefault("TORCHSKY_EXAMPLE_FAST", "1")
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"examples/test_examples.py failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
    )


def test_runner_discovers_only_standalone_scripts() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy; runpy.run_path("
            + repr(str(RUNNER))
            + ", run_name='not_main')",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
