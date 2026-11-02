#!/usr/bin/env python
"""Smoke runner for every example script in ``examples/``.

Each example is a standalone script that builds its own synthetic inputs, so
the runner can execute them one by one in a bounded subprocess. It is the gate
behind ``pixi run examples`` and ``tests/test_examples_runner.py``.

Run a subset with positional names::

    python examples/test_examples.py wcs_tour.py healpix_and_moc.py

An example that exits with status 77 (``SKIP``) has an opt-in data
requirement that is not met (see ``real_data_cutout.py``); it is reported as
SKIP and does not fail the gate. Examples under ``examples/applications/`` need
external data or checkpoints and are not run here.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# Per-example timeout (seconds). Every example is designed to finish in under
# 30 s on CPU; the margin covers loaded CI runners. Raise it only with a
# measured reason.
DEFAULT_TIMEOUT = 120
SKIP = 77
TIMEOUTS: dict[str, int] = {}

EXCLUDE = {
    "test_examples.py",
    "__init__.py",
}


def discover() -> list[str]:
    return sorted(
        path.name
        for path in SCRIPT_DIR.glob("*.py")
        if path.name not in EXCLUDE and not path.name.startswith("_")
    )


def python_cmd() -> list[str]:
    if os.environ.get("PIXI_ENVIRONMENT_NAME"):
        return [sys.executable]
    if shutil.which("pixi"):
        return ["pixi", "run", "python"]
    return [sys.executable]


def run_example(name: str) -> tuple[str, str]:
    """Return ``("PASS" | "SKIP" | "FAIL", detail)``."""
    path = SCRIPT_DIR / name
    if not path.is_file():
        return "FAIL", f"file not found: {path}"
    timeout = TIMEOUTS.get(name, DEFAULT_TIMEOUT)
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    try:
        result = subprocess.run(
            [*python_cmd(), str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "FAIL", f"timeout after {timeout}s"
    if result.returncode == 0:
        return "PASS", ""
    if result.returncode == SKIP:
        return "SKIP", (result.stdout or "").strip()
    output = (result.stderr or "") + (result.stdout or "")
    return "FAIL", output[-1500:]


def main(argv: list[str]) -> int:
    if not __debug__:
        print("refusing to run with assertions disabled (-O / PYTHONOPTIMIZE)")
        return 1
    names = argv[1:] or discover()
    failures = skipped = 0
    for name in names:
        start = time.perf_counter()
        status, detail = run_example(name)
        print(f"{status}  {name}  ({time.perf_counter() - start:.1f} s)")
        if status == "FAIL":
            failures += 1
            print(detail)
        elif status == "SKIP":
            skipped += 1
            print(f"      {detail}")
    print(
        f"\n{len(names) - failures - skipped}/{len(names)} examples passed, {skipped} skipped"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
