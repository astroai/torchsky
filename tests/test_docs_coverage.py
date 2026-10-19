"""Every declared public name must be discoverable in the documentation.

The namespace surface is explicit (`__all__` per namespace); the docs must not
silently lag it. This is a textual check: each name has to appear somewhere in
`docs/`, normally in its guide or API reference page. It does not assert that
the mention is a good explanation, only that removing a name from the docs is a
test failure.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import torchsky

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _docs_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in DOCS.glob("*.md"))


@pytest.mark.parametrize("namespace", sorted(torchsky._NAMESPACES))
def test_every_public_name_is_documented(namespace: str) -> None:
    module = importlib.import_module(f"torchsky.{namespace}")
    names = tuple(getattr(module, "__all__", ()))
    text = _docs_text()
    missing = [name for name in names if name not in text]
    assert not missing, f"torchsky.{namespace} names missing from docs/: {missing}"
