#!/usr/bin/env python3
"""Verify that every public name rendered a reference anchor.

mkdocstrings silently skips members it cannot resolve (for example a class with
no docstring of its own). The manifest says a name is public, so the built site
must contain its anchor. Run after ``pixi run docs-build``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DOCS = ROOT / "docs"
MANIFEST = DOCS / "api-manifest.json"
REGISTRY = DOCS / "reference" / "_registry.toml"


def _read_registry() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(REGISTRY.read_text(encoding="utf-8"))


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry = _read_registry()
    pages = {entry["name"]: entry["slug"] for entry in registry["namespace"]}
    module_pages = {
        entry["name"]: entry["slug"] for entry in registry.get("module", [])
    }

    missing: list[str] = []
    checked = 0
    for namespace, names in manifest.items():
        slug = pages[namespace]
        html_path = SITE / "reference" / slug / "index.html"
        if not html_path.is_file():
            missing.append(f"{namespace}: built page missing ({html_path})")
            continue
        html = html_path.read_text(encoding="utf-8", errors="replace")
        anchors = set(re.findall(r'id="([^"]+)"', html))
        for name in names:
            full = name if name == namespace else f"{namespace}.{name}"
            checked += 1
            if namespace == "torchsky" and full in pages:
                # Namespace names are documented by their own reference pages.
                continue
            if full in module_pages:
                # Public submodules have their own generated pages.
                continue
            if full not in anchors:
                missing.append(f"{full} (expected on reference/{slug}.md)")
    for namespace, slug in module_pages.items():
        html_path = SITE / "reference" / slug / "index.html"
        if not html_path.is_file():
            missing.append(f"{namespace}: built page missing ({html_path})")
        elif f'id="{namespace}"' not in html_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            missing.append(
                f"{namespace} (module anchor missing on reference/{slug}.md)"
            )

    if missing:
        print(
            f"[FAIL] {len(missing)} public name(s) have no rendered reference anchor:"
        )
        for item in missing:
            print(f"  - {item}")
        return 1
    print(f"[OK] {checked} public names rendered reference anchors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
