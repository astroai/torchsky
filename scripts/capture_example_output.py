#!/usr/bin/env python3
"""Capture the stdout of selected example scripts for the generated index.

The committed excerpts in ``docs/assets/example-output/`` are refreshed
deliberately, not during the docs build: examples are deterministic and cheap,
but running all of them at build time would make documentation depend on a
working scientific stack.

    pixi run docs-example-output -- example_healpix.py example_wcs_transform.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
OUTPUTS = ROOT / "docs" / "assets" / "example-output"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", nargs="+", help="Script file names under examples/")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for name in args.examples:
        path = EXAMPLES / name
        if not path.is_file():
            print(f"[fail] no such example: {name}")
            return 1
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=env,
        )
        if result.returncode != 0:
            print(f"[fail] {name}: exit {result.returncode}")
            print((result.stderr or "")[-1000:])
            return 1
        target = OUTPUTS / f"{path.stem}.txt"
        target.write_text(result.stdout, encoding="utf-8")
        print(f"[ok] {name} -> {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
