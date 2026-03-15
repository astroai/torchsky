#!/usr/bin/env python3
"""Ensure local torchsky native extension is available before running a bench script."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _has_local_cpp() -> bool:
    try:
        importlib.import_module("torchsky")
        importlib.import_module("torchsky.cpp")
    except Exception:
        return False
    return True


def _ensure_local_cpp() -> None:
    if _has_local_cpp():
        return
    cmd = [sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation"]
    print("Installing local torchsky editable build into the active sphere-bench env")
    subprocess.check_call(cmd, cwd=_repo_root())
    if not _has_local_cpp():
        raise RuntimeError("torchsky.cpp still unavailable after editable install")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", help="Benchmark or replay script to execute")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    _ensure_local_cpp()
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = (_repo_root() / script_path).resolve()
    cmd = [sys.executable, str(script_path), *args.script_args]
    completed = subprocess.run(cmd, cwd=_repo_root(), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
