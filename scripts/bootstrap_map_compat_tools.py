#!/usr/bin/env python3
"""Clone and build external mapmaking comparators from official source repos."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


REPOS = {
    "swarp": "https://github.com/astromatic/swarp.git",
    "montage": "https://github.com/Caltech-IPAC/Montage.git",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _src_root() -> Path:
    return _repo_root() / ".benchmarks" / "external-tools" / "src"


def _prefix_root() -> Path:
    return _repo_root() / ".benchmarks" / "external-tools" / "prefix"


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd, env=env)


def _run_allow_failure(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> int:
    print("+", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    return int(completed.returncode)


def _ensure_repo(tool: str) -> Path:
    if tool == "casutools":
        source_override = os.environ.get("CASUTOOLS_SOURCE")
        source_dir = Path(source_override).expanduser() if source_override else Path.home() / "src" / "casutools" / "casutools-1.0.43"
        if not source_dir.exists():
            raise RuntimeError(
                "casutools source is not available; set CASUTOOLS_SOURCE "
                "or place the source tree at ~/src/casutools/casutools-1.0.43"
            )
        src_dir = _src_root() / "casutools"
        if src_dir.exists():
            return src_dir
        src_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, src_dir)
        return src_dir
    src_dir = _src_root() / ("Montage" if tool == "montage" else tool)
    if src_dir.exists():
        return src_dir
    src_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", REPOS[tool], str(src_dir)], cwd=_repo_root())
    return src_dir


def _prefix(tool: str) -> Path:
    out = _prefix_root() / tool
    out.mkdir(parents=True, exist_ok=True)
    return out


def _conda_prefix() -> Path:
    value = os.environ.get("CONDA_PREFIX")
    if not value:
        raise RuntimeError("CONDA_PREFIX is not set; run this script inside a Pixi env")
    return Path(value)


def _build_swarp(src_dir: Path, prefix: Path, *, jobs: int) -> None:
    if (prefix / "bin" / "swarp").exists():
        print(f"swarp already built at {prefix / 'bin' / 'swarp'}")
        return
    env = dict(os.environ)
    conda_prefix = _conda_prefix()
    env["CPPFLAGS"] = f"-I{conda_prefix / 'include'}"
    env["LDFLAGS"] = f"-L{conda_prefix / 'lib'}"
    _run(["./autogen.sh"], cwd=src_dir, env=env)
    _run(
        [
            "./configure",
            f"--prefix={prefix}",
            f"--with-cfitsio-incdir={conda_prefix / 'include'}",
            f"--with-cfitsio-libdir={conda_prefix / 'lib'}",
        ],
        cwd=src_dir,
        env=env,
    )
    _run(["make", f"-j{jobs}"], cwd=src_dir, env=env)
    _run(["make", "install"], cwd=src_dir, env=env)


def _build_montage(src_dir: Path, prefix: Path, *, jobs: int) -> None:
    if (prefix / "bin" / "mProject").exists():
        print(f"montage already built at {prefix / 'bin' / 'mProject'}")
        return
    built_bin = src_dir / "bin"
    if (built_bin / "mProject").exists():
        target_bin = prefix / "bin"
        target_bin.mkdir(parents=True, exist_ok=True)
        for name in ("mProject", "mProjectPP", "mGetHdr", "mMakeHdr"):
            candidate = built_bin / name
            if candidate.exists():
                shutil.copy2(candidate, target_bin / name)
        return
    env = dict(os.environ)
    env["MAKEFLAGS"] = f"-j{jobs}"
    _run(["./configure", f"--prefix={prefix}"], cwd=src_dir, env=env)
    make_rc = _run_allow_failure(["make", f"-j{jobs}"], cwd=src_dir, env=env)
    mproject = built_bin / "mProject"
    if not mproject.exists():
        raise RuntimeError(f"Montage build failed before mProject was produced (make rc={make_rc})")
    target_bin = prefix / "bin"
    target_bin.mkdir(parents=True, exist_ok=True)
    for name in ("mProject", "mProjectPP", "mGetHdr", "mMakeHdr"):
        candidate = built_bin / name
        if candidate.exists():
            shutil.copy2(candidate, target_bin / name)


def _build_casutools(src_dir: Path, prefix: Path, *, jobs: int) -> None:
    if (prefix / "bin" / "mosaic").exists():
        print(f"casutools already built at {prefix / 'bin' / 'mosaic'}")
        return
    shim_lib = _prefix_root() / "shims" / "casutools" / "lib"
    shim_lib.mkdir(parents=True, exist_ok=True)
    cfitsio_dylib = Path("/opt/homebrew/lib/libcfitsio.dylib")
    if not cfitsio_dylib.exists():
        raise RuntimeError("expected /opt/homebrew/lib/libcfitsio.dylib for local casutools build")
    for name in ("libcfitsio.a", "libcfitsio.dylib"):
        target = shim_lib / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(cfitsio_dylib)

    wcslib_include = Path("/opt/homebrew/Cellar/wcslib/8.5/include")
    wcslib_lib = Path("/opt/homebrew/lib")
    cfitsio_include = Path("/opt/homebrew/include")
    if not wcslib_include.exists() or not wcslib_lib.exists() or not cfitsio_include.exists():
        raise RuntimeError("expected Homebrew wcslib/cfitsio headers and libraries for local casutools build")

    _run(
        [
            "./configure",
            f"--prefix={prefix}",
            f"--with-cfitsio-includes={cfitsio_include}",
            f"--with-cfitsio-libs={shim_lib}",
            f"--with-wcs-includes={wcslib_include}",
            f"--with-wcs-libs={wcslib_lib}",
        ],
        cwd=src_dir,
    )
    _run(["make", f"-j{jobs}"], cwd=src_dir)
    _run(["make", "install"], cwd=src_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool",
        choices=("swarp", "montage", "casutools", "all"),
        default="all",
        help="Tool to bootstrap",
    )
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    args = parser.parse_args()

    tools = ("swarp", "montage", "casutools") if args.tool == "all" else (args.tool,)
    for tool in tools:
        src_dir = _ensure_repo(tool)
        prefix = _prefix(tool)
        if tool == "swarp":
            _build_swarp(src_dir, prefix, jobs=args.jobs)
        elif tool == "montage":
            _build_montage(src_dir, prefix, jobs=args.jobs)
        else:
            _build_casutools(src_dir, prefix, jobs=args.jobs)
        print(f"{tool}: installed under {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
