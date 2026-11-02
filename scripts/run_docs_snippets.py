#!/usr/bin/env python3
"""Execute self-contained documentation snippets.

A fenced ``python`` block preceded by the exact line ``<!-- docs: run -->`` is a
runnable snippet. Each block runs in its own subprocess with ``src`` on the
import path, so a snippet that only works because an earlier block defined a
name fails here instead of on a reader's machine.

Blocks must be cheap, deterministic, and self-contained: build synthetic
inputs, assert the result, and print a short line. Shell fences are not run.

    pixi run docs-snippets
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MARKER = "<!-- docs: run -->"
FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
DEFAULT_TIMEOUT = 120.0
EXCLUDE_DIRS = {"published-examples"}


def discover() -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for page in sorted(DOCS.rglob("*.md")):
        if any(part in EXCLUDE_DIRS for part in page.relative_to(DOCS).parts):
            continue
        text = page.read_text(encoding="utf-8")
        matches = list(FENCE.finditer(text))
        for index, match in enumerate(matches):
            prefix = text[: match.start()].rstrip().splitlines()
            if not prefix or prefix[-1].strip() != MARKER:
                continue
            label = f"{page.relative_to(ROOT)}#{index}"
            blocks.append((label, textwrap.dedent(match.group(1))))
    return blocks


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.pop("PYTHONOPTIMIZE", None)
    return env


def run_block(label: str, source: str, timeout: float) -> tuple[bool, str]:
    # Snippets run in a scratch directory so files they write never land in the repo.
    with tempfile.TemporaryDirectory(prefix="torchsky-docs-snippet-") as workdir:
        return _run_in(label, source, timeout, workdir)


def _run_in(label: str, source: str, timeout: float, workdir: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", source],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_environment(),
        )
    except subprocess.TimeoutExpired:
        return False, f"{label}: timeout after {timeout:.0f}s"
    if result.returncode == 0:
        return True, ""
    output = (result.stderr or "") + (result.stdout or "")
    return False, f"{label}:\n{textwrap.indent(output[-2500:], '    ')}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--filter", default="", help="Only run labels containing this text"
    )
    args = parser.parse_args(argv)

    blocks = [(label, source) for label, source in discover() if args.filter in label]
    if not blocks:
        print("No runnable snippets found.")
        return 1
    failures: list[str] = []
    for label, source in blocks:
        ok, message = run_block(label, source, args.timeout)
        status = "ok" if ok else "FAIL"
        print(f"[{status}] {label}")
        if not ok:
            failures.append(message)
    print(f"\n{len(blocks) - len(failures)}/{len(blocks)} runnable snippets passed.")
    if failures:
        for failure in failures:
            print(f"\n{failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
