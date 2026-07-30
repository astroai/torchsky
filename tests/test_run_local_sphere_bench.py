from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_local_sphere_bench  # noqa: E402


def test_clean_editable_install_is_verified_in_fresh_interpreter(monkeypatch) -> None:
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setattr(run_local_sphere_bench, "_has_local_cpp", lambda: False)
    monkeypatch.setattr(
        run_local_sphere_bench.subprocess,
        "check_call",
        lambda cmd, cwd: calls.append((cmd, cwd)),
    )
    monkeypatch.setattr(
        run_local_sphere_bench.subprocess,
        "call",
        lambda cmd, cwd: calls.append((cmd, cwd)) or 0,
    )

    run_local_sphere_bench._ensure_local_cpp()

    assert calls[0][0][1:] == [
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        "--no-build-isolation",
    ]
    assert calls[1][0] == [
        sys.executable,
        "-c",
        "import torchsky; import torchsky.cpp",
    ]
    assert calls[0][1] == calls[1][1] == run_local_sphere_bench._repo_root()
