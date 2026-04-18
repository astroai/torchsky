"""Strict filesystem containment for viz HTTP routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException


def resolve_under_root(root: Path, rel_path: str) -> Path:
    """
    Return ``(root / rel_path).resolve()`` only if the result stays under ``root``.

    Uses path semantics (not string prefix) so a root like ``/data`` cannot be
    spoofed by a sibling such as ``/data_exfil`` (``str.startswith`` bypass).
    """
    root_r = root.resolve()
    p = (root_r / rel_path).resolve()
    try:
        p.relative_to(root_r)
    except ValueError:
        raise HTTPException(status_code=403, detail="Path outside root directory")
    return p
