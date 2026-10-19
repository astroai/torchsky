#!/usr/bin/env python3
"""Minimal environment check: torch, torchsky, native extension, versions."""

from __future__ import annotations

import importlib.metadata
import sys


def main() -> int:
    try:
        import torch
    except ImportError as e:
        print("FAIL: torch not importable:", e, file=sys.stderr)
        return 1

    try:
        import torchsky
        import torchsky._cpp as cpp  # noqa: F401
    except ImportError as e:
        print("FAIL: torchsky or torchsky._cpp not importable:", e, file=sys.stderr)
        return 1

    try:
        ver = importlib.metadata.version("torchsky")
    except importlib.metadata.PackageNotFoundError:
        ver = getattr(torchsky, "__version__", "?")

    print("torch", torch.__version__)
    print("torchsky", ver)
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
