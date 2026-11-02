"""Every public name is in the reference; Tier-1 names are used in a guide.

The reference pages are generated from ``__all__`` and checked structurally in
``test_docs_reference.py``, so this test no longer greps for names anywhere in
``docs/``. It enforces the stronger, useful contract: the names users are most
likely to reach for are explained in prose, not only listed.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TIER1 = DOCS / "data" / "tier1.txt"


def _tier1_names() -> list[str]:
    names: list[str] = []
    for line in TIER1.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def _guide_text() -> str:
    pages = [path for path in DOCS.glob("*.md") if not path.name.startswith("api-")]
    return "\n".join(path.read_text(encoding="utf-8") for path in pages)


def test_tier1_names_appear_in_at_least_one_guide() -> None:
    text = _guide_text()
    missing: list[str] = []
    for full in _tier1_names():
        short = full.rsplit(".", 1)[-1]
        if not re.search(rf"\b{re.escape(short)}\b", text):
            missing.append(full)
    assert not missing, f"Tier-1 names missing from the guides: {missing}"


def test_tier1_list_is_not_stale() -> None:
    import json

    manifest = json.loads((DOCS / "api-manifest.json").read_text(encoding="utf-8"))
    known = {
        f"{namespace}.{name}" for namespace, names in manifest.items() for name in names
    }
    stale = sorted(name for name in _tier1_names() if name not in known)
    assert not stale, f"Tier-1 list contains unknown public names: {stale}"
