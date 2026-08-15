#!/usr/bin/env python3
"""Print importable benchmark comparator versions (JSON) for the active interpreter."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _try_version(dist_name: str, import_name: str | None = None) -> dict[str, Any]:
    mod_name = import_name or dist_name.replace("-", "_")
    try:
        m = importlib.import_module(mod_name)
    except Exception as exc:
        return {
            "name": dist_name,
            "import": mod_name,
            "status": "missing",
            "error": str(exc),
        }
    ver = getattr(m, "__version__", None)
    if ver is None and hasattr(m, "version"):
        ver = getattr(m.version, "version", None)
    return {
        "name": dist_name,
        "import": mod_name,
        "status": "ok",
        "version": str(ver) if ver is not None else "unknown",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Write JSON to this path (default: stdout only)",
    )
    args = p.parse_args()

    comparators: list[tuple[str, str]] = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("astropy", "astropy"),
        ("healpy", "healpy"),
        ("hpgeom", "hpgeom"),
        ("astropy_healpix", "astropy_healpix"),
        ("healsparse", "healsparse"),
        ("mocpy", "mocpy"),
        ("spherical_geometry", "spherical_geometry"),
        ("healpix", "healpix"),
        ("mhealpy", "mhealpy"),
        ("reproject", "reproject"),
        ("drizzle", "drizzle"),
        ("MontagePy", "MontagePy"),
        ("starlink_ast", "starlink.Ast"),
        ("kapteyn", "kapteyn"),
    ]

    pkgs = [_try_version(name, imp) for name, imp in comparators]

    sha = ""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(_REPO_ROOT)
        ).strip()
    except Exception:
        pass

    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "git_sha": sha,
        "TORCHSKY_env": {
            k: os.environ[k] for k in sorted(os.environ) if k.startswith("TORCHSKY_")
        },
        "comparators": pkgs,
    }

    try:
        import torchsky  # noqa: WPS433

        out["torchsky"] = getattr(torchsky, "__version__", "dev")
    except Exception as exc:
        out["torchsky"] = f"missing:{exc}"

    text = json.dumps(out, indent=2) + "\n"
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
