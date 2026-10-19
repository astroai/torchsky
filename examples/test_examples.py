#!/usr/bin/env python
"""Smoke runner for every example script in ``examples/``.

Each example is a standalone script that builds its own synthetic inputs, so
the runner can execute them one by one in a bounded subprocess. It is the gate
behind ``pixi run examples`` and ``tests/test_examples_runner.py``.

Run a subset with positional names::

    python examples/test_examples.py example_wcs_transform.py example_healpix.py
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# Per-example timeout (seconds). The default covers the harmonic and field
# examples; raise it only with a measured reason.
DEFAULT_TIMEOUT = 240
TIMEOUTS: dict[str, int] = {}

EXCLUDE = {
    "test_examples.py",
    "__init__.py",
    # CLI pipeline that requires an external catalogue release, memberships,
    # and a checkpoint; it is documented in docs/photometry-encoders.md and
    # exercised through the package tests instead of as a smoke script.
    "catalogue_photoz.py",
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


def run_example(name: str) -> tuple[bool, str]:
    path = SCRIPT_DIR / name
    if not path.is_file():
        return False, f"file not found: {path}"
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
        return False, f"timeout after {timeout}s"
    if result.returncode == 0:
        return True, ""
    output = (result.stderr or "") + (result.stdout or "")
    return False, output[-1500:]


def main(argv: list[str]) -> int:
    if not __debug__:
        print("refusing to run with assertions disabled (-O / PYTHONOPTIMIZE)")
        return 1
    names = argv[1:] or discover()
    failures = 0
    for name in names:
        ok, detail = run_example(name)
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failures += 1
            print(detail)
    print(f"\n{len(names) - failures}/{len(names)} examples passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
