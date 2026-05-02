from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_upstream_parity_manifest_references_existing_local_paths() -> None:
    matrix_path = ROOT / "docs" / "compatibility.md"

    assert matrix_path.exists()

    text = matrix_path.read_text(encoding="utf-8")
    for heading in ("### Astropy", "### Healpy", "### HealSparse", "### Fitsio"):
        assert heading in text
