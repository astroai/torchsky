from __future__ import annotations

import importlib.util
from pathlib import Path
import tarfile
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_distribution.py"
SPEC = importlib.util.spec_from_file_location("check_distribution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check_distribution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_distribution)


def _write_wheel(path: Path, additions: dict[str, bytes] | None = None) -> None:
    payloads = {
        "torchsky/__init__.py": b"",
        "torchsky/_cpp_ext.so": b"native",
        "torchsky-0.4.0.dist-info/licenses/LICENSE": b"MIT",
        "torchsky-0.4.0.dist-info/licenses/src/torchsky/cpp_src/third_party/pocketfft.LICENSE.md": b"BSD",
    }
    payloads.update(additions or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)


def _write_sdist(path: Path) -> None:
    root = "torchsky-0.4.0"
    names = check_distribution._expected_cpp_sources(REPO_ROOT)
    names.update(
        {
            "LICENSE",
            "src/torchsky/cpp_src/third_party/pocketfft.LICENSE.md",
            "src/torchsky/cpp_src/third_party/README.md",
        }
    )
    staging = path.parent / root
    for name in names:
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"source")
    with tarfile.open(path, "w:gz") as archive:
        archive.add(staging, arcname=root)


def test_distribution_checker_accepts_complete_artifacts(tmp_path: Path) -> None:
    wheel = tmp_path / "torchsky.whl"
    sdist = tmp_path / "torchsky.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)

    check_distribution.check_wheel(wheel, inspect_native=False)
    check_distribution.check_sdist(sdist, project_root=REPO_ROOT)


@pytest.mark.parametrize(
    ("name", "payload", "message"),
    [
        ("torchsky/lib/libtorch_cpu.so", b"", "bundles a Torch library"),
        ("torchsky/_cpp_ext.so", b"/home/runner/work/torchsky", "build path"),
        ("torchsky/_cpp_ext.so", b"linked to libcfitsio", "CFITSIO marker"),
    ],
)
def test_wheel_rejects_forbidden_native_content(
    tmp_path: Path, name: str, payload: bytes, message: str
) -> None:
    wheel = tmp_path / "torchsky.whl"
    _write_wheel(wheel, {name: payload})

    with pytest.raises(AssertionError, match=message):
        check_distribution.check_wheel(wheel, inspect_native=False)


def test_sdist_rejects_missing_cpp_input(tmp_path: Path) -> None:
    sdist = tmp_path / "torchsky.tar.gz"
    _write_sdist(sdist)
    missing = next(iter(check_distribution._expected_cpp_sources(REPO_ROOT)))
    rewritten = tmp_path / "rewritten.tar.gz"
    with (
        tarfile.open(sdist, "r:gz") as source,
        tarfile.open(rewritten, "w:gz") as target,
    ):
        for member in source.getmembers():
            if member.name.endswith(missing):
                continue
            if member.isfile():
                target.addfile(member, source.extractfile(member))
            else:
                target.addfile(member)

    with pytest.raises(AssertionError, match=r"missing C\+\+ build inputs"):
        check_distribution.check_sdist(rewritten, project_root=REPO_ROOT)
