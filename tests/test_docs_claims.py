"""Docs claims must match package metadata and the actual CLI.

Docs drift starts with small, plausible sentences: a version floor, an extra
name, a platform, a subcommand. These checks bind the machine-checkable claims
to their source of truth.
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

from torchsky.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
WORKFLOW = (ROOT / ".github" / "workflows" / "build_wheels.yml").read_text(
    encoding="utf-8"
)


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
    install = (DOCS / "introduction" / "install.md").read_text(encoding="utf-8")
    assert requires_python.replace(">=", ">=") in "".join(install.split())
    assert f">={torch_floor}" in install
    torchfits_short = ".".join(torchfits_floor.split(".")[:2])
    assert f"torchfits {torchfits_floor}" in install
    assert torch_floor in install
    assert torchfits_short in "".join(install.split())


def test_release_status_is_consistent() -> None:
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    install = (DOCS / "introduction" / "install.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text, label in ((index, "index"), (install, "install"), (readme, "README")):
        flat = " ".join(text.split())
        assert re.search(r"GitHub pre-release", flat, re.IGNORECASE), (
            f"{label} must state that 0.5.0rc1 is a GitHub pre-release"
        )
        assert re.search(
            r"not (uploaded )?(to |on )PyPI|never uploaded to PyPI", flat
        ), f"{label} must state the current PyPI status"


def test_documented_extras_exist() -> None:
    extras = set(PYPROJECT["project"].get("optional-dependencies", {}))
    documented = set(re.findall(r"torchsky\[([a-zA-Z0-9_,-]+)\]", _docs_text()))
    used = {name.strip() for group in documented for name in group.split(",")}
    unknown = sorted(used - extras)
    assert not unknown, f"docs reference undefined extras: {unknown}"


def _parser_inventory(parser: argparse.ArgumentParser) -> set[str]:
    inventory = {parser.prog}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                if not name.startswith("_"):
                    inventory.update(_parser_inventory(child))
    return inventory


def test_generated_cli_reference_matches_parser_inventory() -> None:
    reference = (DOCS / "reference" / "cli.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^## `([^`]+)`$", reference, flags=re.MULTILINE))
    assert documented == _parser_inventory(build_parser())


def test_documented_wheel_python_versions_are_built() -> None:
    match = re.search(r"CIBW_BUILD:\s*(.+)", WORKFLOW)
    assert match, "wheel workflow must declare CIBW_BUILD"
    built = set(re.findall(r"cp(\d)(\d+)", match.group(1)))
    documented = set(
        re.findall(
            r"Python 3\.(\d+)",
            (DOCS / "introduction" / "install.md").read_text(encoding="utf-8"),
        )
    )
    if documented:
        not_built = sorted(
            version for version in documented if ("3", version) not in built
        )
        assert not not_built, (
            f"docs claim wheels for unbuilt Python versions: {not_built}"
        )


def test_public_namespace_list_matches_manifest() -> None:
    manifest = json.loads((DOCS / "api-manifest.json").read_text(encoding="utf-8"))
    registry = tomllib.loads(
        (DOCS / "reference" / "_registry.toml").read_text(encoding="utf-8")
    )
    assert sorted(entry["name"] for entry in registry["namespace"]) == sorted(manifest)
