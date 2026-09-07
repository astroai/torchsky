from __future__ import annotations

from pathlib import Path
import sys
import subprocess
from unittest.mock import patch

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from bootstrap_map_compat_tools import _run, _swarp_build_env, main  # noqa: E402


def test_failed_configure_preserves_diagnostic_and_original_error(tmp_path, capsys):
    source = tmp_path / "swarp"
    source.mkdir()
    diagnostic = "checking for ffopen... no\nld: missing dependency\n"
    (source / "config.log").write_text(diagnostic)
    artifact_root = tmp_path / "persistent"
    error = subprocess.CalledProcessError(1, ["./configure"])
    with patch("bootstrap_map_compat_tools.subprocess.check_call", side_effect=error):
        with pytest.raises(subprocess.CalledProcessError) as caught:
            _run(["./configure"], cwd=source, env={"ARTIFACT_ROOT": str(artifact_root)})
    assert caught.value is error
    assert diagnostic in capsys.readouterr().out
    assert (artifact_root / "swarp-config.log").read_text() == diagnostic


def test_successful_configure_does_not_publish_stale_log(tmp_path, capsys):
    (tmp_path / "config.log").write_text("stale failure")
    with patch("bootstrap_map_compat_tools.subprocess.check_call"):
        _run(["./configure"], cwd=tmp_path, env={})
    assert "stale failure" not in capsys.readouterr().out


def test_swarp_build_env_exposes_conda_shared_dependencies() -> None:
    prefix = Path("/opt/pixi/env")

    with patch(
        "bootstrap_map_compat_tools.shutil.which", return_value="/usr/bin/cc"
    ) as which:
        env = _swarp_build_env(
            prefix,
            base={
                "CC": "/opt/conda/bin/gcc",
                "CPP": "x86_64-conda-linux-gnu-cpp",
                "LD": "x86_64-conda-linux-gnu-ld",
                "COMPILER_PATH": "/opt/conda/bin",
                "host_alias": "x86_64-conda-linux-gnu",
                "LD_LIBRARY_PATH": "/system/lib",
                "PKG_CONFIG_PATH": "/system/pkgconfig",
            },
        )

    which.assert_called_once_with("cc", path="/usr/bin:/bin")
    assert env["CC"] == "/usr/bin/cc -B/usr/bin/"
    assert env["CPP"] == "/usr/bin/cc -E"
    assert env["LD"] == "/usr/bin/ld"
    assert "COMPILER_PATH" not in env and "host_alias" not in env
    assert env["CPPFLAGS"] == "-I/opt/pixi/env/include"
    assert "-L/opt/pixi/env/lib" in env["LDFLAGS"]
    assert "-Wl,-rpath-link,/opt/pixi/env/lib" in env["LDFLAGS"]
    assert env["LD_LIBRARY_PATH"] == "/opt/pixi/env/lib:/system/lib"
    assert env["PKG_CONFIG_PATH"] == "/opt/pixi/env/lib/pkgconfig:/system/pkgconfig"


def test_bootstrap_reuses_environment_swarp(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["bootstrap", "--tool", "swarp"])
    with (
        patch(
            "bootstrap_map_compat_tools.shutil.which",
            return_value="/opt/pixi/bin/swarp",
        ),
        patch("bootstrap_map_compat_tools._ensure_repo") as ensure_repo,
    ):
        assert main() == 0

    ensure_repo.assert_not_called()
    assert "using environment binary at /opt/pixi/bin/swarp" in capsys.readouterr().out
