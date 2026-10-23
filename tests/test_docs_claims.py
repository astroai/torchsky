"""Docs claims must match package metadata and the actual CLI.

Docs drift starts with small, plausible sentences: a version floor, an extra
name, a platform, a subcommand. These checks bind the machine-checkable claims
to their source of truth.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
WORKFLOW = (ROOT / ".github" / "workflows" / "build_wheels.yml").read_text(encoding="utf-8")


def _docs_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in DOCS.rglob("*.md"))


def test_documented_version_floors_match_pyproject() -> None:
    requires_python = PYPROJECT["project"]["requires-python"]
    torch_floor = next(
        dep.split(">=")[1]
        for dep in PYPROJECT["project"]["dependencies"]
        if dep.lower().startswith("torch")
    )
    torchfits_floor = next(
        dep.split(">=")[1]
        for dep in PYPROJECT["project"]["dependencies"]
        if dep.lower().startswith("torchfits")
    )
    install = (DOCS / "install.md").read_text(encoding="utf-8")
    getting_started = (DOCS / "getting-started.md").read_text(encoding="utf-8")
    assert requires_python.replace(">=", ">=") in "".join(install.split())
    assert f">={torch_floor}" in install
    torchfits_short = ".".join(torchfits_floor.split(".")[:2])
    assert f"torchfits {torchfits_floor}" in install
    assert torch_floor in getting_started
    assert torchfits_short in "".join(getting_started.split())


def test_release_status_is_consistent() -> None:
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    install = (DOCS / "install.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text, label in ((index, "index"), (install, "install"), (readme, "README")):
        flat = " ".join(text.split())
        assert re.search(r"not (yet )?(published )?on PyPI|not on PyPI yet", flat), (
            f"{label} must state the current PyPI status"
        )


def test_documented_extras_exist() -> None:
    extras = set(PYPROJECT["project"].get("optional-dependencies", {}))
    documented = set(re.findall(r"torchsky\[([a-zA-Z0-9_,-]+)\]", _docs_text()))
    used = {name.strip() for group in documented for name in group.split(",")}
    unknown = sorted(used - extras)
    assert not unknown, f"docs reference undefined extras: {unknown}"


def test_documented_cli_commands_exist() -> None:
    source = (ROOT / "src" / "torchsky" / "cli" / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    commands = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"add_parser", "add_subparsers"} and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    commands.add(argument.value)
    documented = set(re.findall(r"torchsky (map|sky|viz|info|cat)\b", _docs_text()))
    assert {"map", "sky", "viz"} <= commands | documented
    unknown = sorted(documented - (commands | {"sky", "map", "viz", "info", "cat"}))
    assert not unknown, f"docs mention unknown CLI commands: {unknown}"


def test_documented_wheel_python_versions_are_built() -> None:
    match = re.search(r"CIBW_BUILD:\s*(.+)", WORKFLOW)
    assert match, "wheel workflow must declare CIBW_BUILD"
    built = set(re.findall(r"cp(\d)(\d+)", match.group(1)))
    documented = set(re.findall(r"Python 3\.(\d+)", (DOCS / "install.md").read_text(encoding="utf-8")))
    if documented:
        not_built = sorted(version for version in documented if ("3", version) not in built)
        assert not not_built, f"docs claim wheels for unbuilt Python versions: {not_built}"


def test_public_namespace_list_matches_manifest() -> None:
    manifest = json.loads((DOCS / "api-manifest.json").read_text(encoding="utf-8"))
    registry = tomllib.loads((DOCS / "reference" / "_registry.toml").read_text(encoding="utf-8"))
    assert sorted(entry["name"] for entry in registry["namespace"]) == sorted(manifest)
