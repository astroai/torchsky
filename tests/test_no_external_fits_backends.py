from __future__ import annotations

import ast
from pathlib import Path


# Production src must not import these peers. Tests and benchmarks are oracles.
_FORBIDDEN = {
    "astropy",
    "fitsio",
    "healpy",
    "hpgeom",
    "healsparse",
    "mocpy",
    "astropy_healpix",
    "spherical_geometry",
    "gwcs",
}

# Opt-in extras / oracle wrappers that are never imported by default torchsky paths.
_ALLOWED_BY_FILE = {
    "astropy_interop.py": {"astropy"},
    "geom.py": {"spherical_geometry"},
}


def _iter_imported_tops(tree: ast.AST) -> list[str]:
    tops: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                tops.append(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            tops.append(node.module.split(".", 1)[0])
    return tops


def test_src_does_not_use_peer_backends_except_opt_in_extras() -> None:
    pkg_root = Path(__file__).resolve().parents[1] / "src" / "torchsky"
    violations: list[str] = []
    for py_file in pkg_root.rglob("*.py"):
        if "viz" in py_file.parts:
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        allowed = _ALLOWED_BY_FILE.get(py_file.name, set())
        for top in _iter_imported_tops(tree):
            if top in _FORBIDDEN and top not in allowed:
                violations.append(f"{py_file}: import {top}")
    assert not violations, "Forbidden peer backend imports:\n" + "\n".join(violations)
