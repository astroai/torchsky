"""Validate the Python shown in the documentation against the real package.

Documentation code blocks are executed for their *import statements only*: that
catches renamed or removed public names before a reader does, without needing
the synthetic data a full example would build. Blocks that import an optional
third-party package are skipped when it is not installed.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
OPTIONAL_MODULES = {
    "polars",
    "pyarrow",
    "duckdb",
    "fastapi",
    "uvicorn",
    "matplotlib",
    "healpy",
    "astropy",
    "reproject",
}


def _blocks() -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for page in sorted(DOCS.glob("*.md")):
        for index, match in enumerate(FENCE.findall(page.read_text(encoding="utf-8"))):
            blocks.append((f"{page.name}:{index}", textwrap.dedent(match)))
    return blocks


def _import_nodes(source: str) -> list[ast.stmt]:
    tree = ast.parse(source)
    nodes: list[ast.stmt] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
    return nodes


@pytest.mark.parametrize(
    ("label", "source"), _blocks(), ids=[label for label, _ in _blocks()]
)
def test_doc_imports_resolve(label: str, source: str) -> None:
    blocks = [_import_nodes(source)]
    namespace: dict[str, object] = {}
    for nodes in blocks:
        for node in nodes:
            if isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
            elif isinstance(node, ast.Import):
                top = node.names[0].name.split(".")[0]
            else:
                continue
            if top in OPTIONAL_MODULES:
                pytest.skip(f"{label} imports optional module {top}")
    for nodes in blocks:
        module = ast.Module(body=nodes, type_ignores=[])
        try:
            exec(compile(module, label, "exec"), namespace)  # noqa: S102 - docs imports only
        except ImportError as exc:  # pragma: no cover - explicit failure path
            pytest.fail(f"{label}: documentation import does not resolve: {exc}")
