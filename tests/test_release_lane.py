"""Release-lane contract: version sources, torch pins, workflows, rc gating."""

from __future__ import annotations

from email.parser import Parser
import io
from pathlib import Path
import re
import sys
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_release_rc as rc_gate  # noqa: E402
import check_release_run as release_run  # noqa: E402
import check_release_tag as release_tag  # noqa: E402
import release_lane as lane  # noqa: E402
import release_notes  # noqa: E402
import validate_release_sdist as release_sdist  # noqa: E402
import validate_release_wheels as release_wheels  # noqa: E402

WORKFLOWS = ROOT / ".github" / "workflows"


def test_committed_tree_matches_the_lane_map() -> None:
    assert lane.check() == 0
    assert lane.current_lane() == "2.14"
    assert lane.torch_spec("2.14") == ">=2.14,<2.15"


def test_every_version_source_agrees_with_the_lane_base() -> None:
    base = lane.load_lanes()["2.14"]["torchsky_version"]
    assert lane.strip_prerelease(lane.committed_version()) == base
    for path in (lane.PYPROJECT, lane.PIXI, lane.INIT, lane.CITATION):
        assert lane.committed_version() in path.read_text(encoding="utf-8")


def test_render_prerelease_and_finalize() -> None:
    base = lane.load_lanes()["2.14"]["torchsky_version"]
    rc = lane.render("2.14", None, prerelease="rc3")
    assert f'version = "{base}rc3"' in rc[lane.PYPROJECT]
    assert f'__version__ = "{base}rc3"' in rc[lane.INIT]
    assert f"version: {base}rc3" in rc[lane.CITATION]
    assert f'version = "{base}"' in lane.render("2.14", None)[lane.PYPROJECT]
    with pytest.raises(SystemExit):
        lane.render("2.14", "9.9.9")
    with pytest.raises(SystemExit):
        lane.render("2.99", None)


def test_workflows_do_not_hard_code_torch_pins() -> None:
    assert lane.workflow_problems() == []
    for name in ("build_wheels.yml", "native-portability.yml"):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "release_lane.py --print-pins" in text
        assert "${TORCH_PIN}" in text


def test_release_tag_kinds() -> None:
    assert release_tag.release_kind("v0.5.0rc1", "push") == "rc"
    assert release_tag.release_kind("v0.5.0", "push") == "stable"
    assert release_tag.release_kind("v0.5.0", "workflow_dispatch") == "stable"
    for tag, event in (
        ("v0.5.0rc1", "workflow_dispatch"),
        ("v0.5.0b1", "push"),
        ("v0.5.0.dev1", "push"),
        ("0.5.0", "push"),
    ):
        with pytest.raises(ValueError):
            release_tag.release_kind(tag, event)


def test_rc_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    assert rc_gate.split_version("0.5.0rc1") == ("0.5.0", "rc1")
    tags = ["v0.5.0rc10", "v0.5.0rc2", "v0.4.0", "v0.5.0"]
    assert rc_gate.rc_tags("0.5.0", tags) == ["v0.5.0rc2", "v0.5.0rc10"]
    monkeypatch.setattr(rc_gate, "git_tags", lambda: [])
    assert rc_gate.main(["--version", "0.5.0rc1"]) == 0
    assert rc_gate.main(["--version", "0.5.0"]) == 1
    assert "src/torchsky/__init__.py" in rc_gate.ALLOWED_STABLE_CHANGES
    assert "CITATION.cff" in rc_gate.ALLOWED_STABLE_CHANGES


def test_release_run_requires_green_push_run_at_the_rc_sha() -> None:
    run = {
        "event": "push",
        "conclusion": "success",
        "headBranch": "v0.5.0rc1",
        "headSha": "abc",
    }
    assert release_run.has_successful_run([run], "v0.5.0rc1", "abc")
    assert not release_run.has_successful_run([run], "v0.5.0rc1", "def")
    assert not release_run.has_successful_run(
        [{**run, "conclusion": "failure"}], "v0.5.0rc1", "abc"
    )
    assert not release_run.has_successful_run(
        [{**run, "event": "workflow_dispatch"}], "v0.5.0rc1", "abc"
    )


def test_release_notes_parse_torchsky_headings() -> None:
    text = (
        "## [Unreleased]\n\nunstamped\n\n"
        "## [0.5.0rc1] - Unreleased\n\nrc body\n\n"
        "## [0.3.1] - 2026-03-07\n\nold\n"
    )
    assert release_notes.extract(text, "0.5.0rc1") == ("[0.5.0rc1]", "rc body")
    assert release_notes.extract(text, "0.3.1") == ("[0.3.1]", "old")
    assert release_notes.extract(text, "0.5.0") == ("Unreleased", "unstamped")


def test_committed_changelog_has_the_rc_section_and_no_duplicate_headings() -> None:
    text = (ROOT / "docs" / "project" / "changelog.md").read_text(encoding="utf-8")
    label, body = release_notes.extract(text, lane.committed_version())
    assert label == f"[{lane.committed_version()}]"
    headings = re.findall(r"^### (.+)$", body, flags=re.MULTILINE)
    assert len(headings) == len(set(headings))
    assert "## [0.4.0]" not in text


def _wheel(directory: Path, python: str, platform: str, version: str) -> Path:
    path = directory / f"torchsky-{version}-{python}-{python}-{platform}.whl"
    dist_info = f"torchsky-{version}.dist-info"
    metadata = (
        "Metadata-Version: 2.4\nName: torchsky\n"
        f"Version: {version}\nRequires-Python: >=3.12\n"
        "Requires-Dist: torch<2.15,>=2.14\n"
        "Requires-Dist: numpy>=1.20.0\n"
        "Requires-Dist: torchfits>=1.3.1\n"
    )
    tags = "".join(f"Tag: {python}-{python}-{tag}\n" for tag in platform.split("."))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", f"Wheel-Version: 1.0\n{tags}")
    return path


def _matrix(directory: Path, version: str) -> list[Path]:
    out = []
    for python in ("cp312", "cp313", "cp314"):
        out.append(_wheel(directory, python, "manylinux_2_28_x86_64", version))
        out.append(_wheel(directory, python, "macosx_14_0_arm64", version))
    return out


def test_wheel_matrix_is_exact_and_manifest_detects_tampering(tmp_path: Path) -> None:
    wheels = _matrix(tmp_path, "0.5.0rc1")
    found, problems = release_wheels.validate(tmp_path, "0.5.0rc1")
    assert problems == [] and len(found) == 6
    manifest = tmp_path / "MANIFEST.sha256"
    release_wheels.write_manifest(found, manifest)
    assert release_wheels.verify_manifest(found, manifest) == []
    wheels[0].write_bytes(wheels[0].read_bytes() + b"x")
    assert release_wheels.verify_manifest(found, manifest)


@pytest.mark.parametrize(
    "drop", ["cp314-cp314-macosx_14_0_arm64", "cp312-cp312-manylinux_2_28_x86_64"]
)
def test_wheel_matrix_rejects_missing_cells(tmp_path: Path, drop: str) -> None:
    for wheel in _matrix(tmp_path, "0.5.0rc1"):
        if drop in wheel.name:
            wheel.unlink()
    _, problems = release_wheels.validate(tmp_path, "0.5.0rc1")
    assert any("expected exactly 6" in p or "mismatch" in p for p in problems)


def test_wheel_rejects_old_macos_tag_and_wrong_metadata(tmp_path: Path) -> None:
    _matrix(tmp_path, "0.5.0rc1")
    _wheel(tmp_path, "cp313", "macosx_11_0_arm64", "0.5.0rc1")
    _, problems = release_wheels.validate(tmp_path, "0.5.0rc1")
    assert any("unexpected platform tag macosx_11_0_arm64" in p for p in problems)
    _, problems = release_wheels.validate(tmp_path, "0.5.0")
    assert any("filename version" in p for p in problems)


def test_wheel_requirements_are_checked(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, "cp312", "manylinux_2_28_x86_64", "0.5.0rc1")
    with zipfile.ZipFile(wheel) as archive:
        meta = archive.read("torchsky-0.5.0rc1.dist-info/METADATA").decode()
    message = Parser().parsestr(meta)
    assert release_wheels._requirement(message, "torch") == "<2.15,>=2.14"
    assert release_wheels._requirement(message, "torchfits") == ">=1.3.1"
    assert release_wheels._requirement(message, "torchvision") is None


def _sdist(path: Path, version: str, extra: dict[str, bytes]) -> None:
    root = f"torchsky-{version}"
    files = {name: b"x" for name in release_sdist.REQUIRED}
    files["PKG-INFO"] = (
        f"Metadata-Version: 2.4\nName: torchsky\nVersion: {version}\n".encode()
    )
    files.update(extra)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_sdist_validation(tmp_path: Path) -> None:
    good = tmp_path / "torchsky-0.5.0rc1.tar.gz"
    _sdist(good, "0.5.0rc1", {})
    assert release_sdist.validate(good, "0.5.0rc1") == []
    assert release_sdist.validate(good, "0.5.0")
    for bad_name in (".github/workflows/x.yml", "extern/cfitsio/a.c", "a/b.so"):
        bad = tmp_path / "bad" / "torchsky-0.5.0rc1.tar.gz"
        bad.parent.mkdir(exist_ok=True)
        _sdist(bad, "0.5.0rc1", {bad_name: b"x"})
        assert release_sdist.validate(bad, "0.5.0rc1"), bad_name
