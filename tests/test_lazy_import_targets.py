"""Guard against function-local imports of modules that no longer exist.

The viz `_blocking` helper was once deleted while sixteen lazy imports still
referenced it; every route failed only when hit, and the viz server tests were
skipped because the optional dependencies were absent from the test env. This
static check resolves every relative import made inside a function (and inside
optional-dependency guards) against the package tree.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "torchsky"


def _module_keys() -> tuple[set[str], dict[str, Path]]:
    keys: set[str] = set()
    paths: dict[str, Path] = {}
    for path in PACKAGE.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(PACKAGE).with_suffix("")
        name = ".".join(rel.parts)
        is_package = path.name == "__init__.py"
        if is_package:
            name = name[:-9] if name.endswith(".__init__") else ""
        key = f"torchsky.{name}" if name else "torchsky"
        keys.add(key)
        paths[key] = path
    return keys, paths


def _resolve(module: str, package: str, level: int) -> str:
    base = package.split(".") if package else []
    if level > 1:
        base = base[: max(0, len(base) - (level - 1))]
    return ".".join([part for part in base if part] + ([module] if module else []))


def test_function_local_relative_imports_resolve() -> None:
    keys, _ = _module_keys()
    missing: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(PACKAGE).with_suffix("")
        name = ".".join(rel.parts)
        is_package = path.name == "__init__.py"
        if is_package:
            name = name[:-9] if name.endswith(".__init__") else ""
        module_key = f"torchsky.{name}" if name else "torchsky"
        package = module_key if is_package else module_key.rsplit(".", 1)[0]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.ImportFrom) and node.level:
                    target = _resolve(node.module or "", package, node.level)
                    if target not in keys:
                        missing.append(f"{path}:{node.lineno} -> {target}")
    assert not missing, "lazy imports reference missing modules:\n" + "\n".join(missing)
