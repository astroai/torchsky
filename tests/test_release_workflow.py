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
    assert "CIBW_BUILD: cp312-* cp313-* cp314-*" in workflow
    assert "arch: x86_64" in workflow
    assert "arch: arm64" in workflow
    assert "CIBW_ARCHS: ${{ matrix.arch }}" in workflow
    assert "CIBW_ENVIRONMENT_MACOS: MACOSX_DEPLOYMENT_TARGET=14.0" in workflow
    assert "PYPI_API_TOKEN" not in workflow
    assert "Build and smoke-test from source distribution" in workflow
    assert "pixi-version: v0.82.0" in workflow
    refs = re.findall(r"uses:\s+\S+@([^\s]+)", workflow)
    assert refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in refs)


def test_wheel_workflow_gates_publication() -> None:
    workflow = (WORKFLOWS / "build_wheels.yml").read_text()
    jobs = tomllib_free_job_names(workflow)
    assert jobs == [
        "release_gate",
        "tests",
        "map_tests",
        "build_wheels",
        "build_sdist",
        "validate_release",
        "upload_pypi",
        "github_release",
    ]
    upload = workflow[
        workflow.index("  upload_pypi:") : workflow.index("  github_release:")
    ]
    assert "github.event_name == 'workflow_dispatch'" in upload
    assert "inputs.publish" in upload
    assert "release_kind == 'stable'" in upload
    assert "github.repository == 'astroai/torchsky'" in upload
    assert "environment: pypi" in upload
    assert "id-token: write" in upload
    assert "--verify-manifest dist/RELEASE-MANIFEST.sha256" in upload
    assert "needs: [release_gate, validate_release]" in upload
    release = workflow[workflow.index("  github_release:") :]
    assert "release_kind == 'rc'" in release
    assert (
        "prerelease: ${{ needs.release_gate.outputs.release_kind == 'rc' }}" in release
    )
    assert "id-token" not in release
    gate = workflow[workflow.index("  release_gate:") : workflow.index("  tests:")]
    for script in (
        "check_release_tag.py",
        "check_release_rc.py",
        "check_release_run.py",
        "release_lane.py --check",
    ):
        assert script in gate
    assert "cancel-in-progress: false" in workflow
    assert "tags:\n      - 'v*'" in workflow


def tomllib_free_job_names(workflow: str) -> list[str]:
    body = workflow[workflow.index("\njobs:\n") + len("\njobs:\n") :]
    return re.findall(r"^  ([a-z_]+):$", body, flags=re.MULTILINE)
