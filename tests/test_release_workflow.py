from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _runtime_version() -> str:
    tree = ast.parse((ROOT / "src" / "torchsky" / "__init__.py").read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return str(ast.literal_eval(node.value))
    raise AssertionError("torchsky.__version__ assignment not found")


def test_release_versions_are_synchronized() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pixi = tomllib.loads((ROOT / "pixi.toml").read_text())
    assert pyproject["project"]["version"] == pixi["package"]["version"]
    assert pyproject["project"]["version"] == _runtime_version()


def test_exactly_one_tag_workflow_publishes_to_pypi() -> None:
    publishers = []
    for path in WORKFLOWS.glob("*.yml"):
        text = path.read_text()
        if "7245906fb61fa33a13554f9d5fb856fe61758086" in text:
            publishers.append(path.name)
    assert publishers == ["build_wheels.yml"]


def test_no_local_task_can_publish_distributions() -> None:
    pixi = (ROOT / "pixi.toml").read_text()
    assert "twine upload" not in pixi
    assert "release-manual" not in pixi


def test_wheel_workflow_matches_supported_artifact_matrix() -> None:
    workflow = (WORKFLOWS / "build_wheels.yml").read_text()
    assert "CIBW_BUILD: cp312-* cp313-*" in workflow
    assert "arch: x86_64" in workflow
    assert "arch: arm64" in workflow
    assert "CIBW_ARCHS: ${{ matrix.arch }}" in workflow
    assert "PYPI_API_TOKEN" not in workflow
    assert "needs: upload_pypi" in workflow
    assert "Validate version sources and tag" in workflow
    assert "Build and smoke-test from source distribution" in workflow
    assert "pixi-version: v0.82.0" in workflow
    refs = re.findall(r"uses:\s+\S+@([^\s]+)", workflow)
    assert refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in refs)
