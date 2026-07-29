from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from bootstrap_map_compat_tools import _swarp_build_env  # noqa: E402


def test_swarp_build_env_exposes_conda_shared_dependencies() -> None:
    prefix = Path("/opt/pixi/env")

    with patch(
        "bootstrap_map_compat_tools.shutil.which", return_value="/usr/bin/cc"
    ) as which:
        env = _swarp_build_env(
            prefix,
            base={
                "CC": "/opt/conda/bin/gcc",
                "LD_LIBRARY_PATH": "/system/lib",
                "PKG_CONFIG_PATH": "/system/pkgconfig",
            },
        )

    which.assert_called_once_with("cc", path="/usr/bin:/bin")
    assert env["CC"] == "/usr/bin/cc"
    assert env["CPPFLAGS"] == "-I/opt/pixi/env/include"
    assert "-L/opt/pixi/env/lib" in env["LDFLAGS"]
    assert "-Wl,-rpath-link,/opt/pixi/env/lib" in env["LDFLAGS"]
    assert env["LD_LIBRARY_PATH"] == "/opt/pixi/env/lib:/system/lib"
    assert (
        env["PKG_CONFIG_PATH"]
        == "/opt/pixi/env/lib/pkgconfig:/system/pkgconfig"
    )
