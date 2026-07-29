#!/usr/bin/env python3
"""Clone and build external mapmaking comparators from official source repos."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


REPOS = {
    "swarp": "https://github.com/astromatic/swarp.git",
    "montage": "https://github.com/Caltech-IPAC/Montage.git",
}
DEFAULT_CASUTOOLS_URL = (
    "https://www.canfar.net/storage/vault/file/sfabbro/casutools-1.0.43.tar.gz"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _src_root() -> Path:
    return _repo_root() / ".benchmarks" / "external-tools" / "src"


def _prefix_root() -> Path:
    return _repo_root() / ".benchmarks" / "external-tools" / "prefix"


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd, env=env)


def _run_allow_failure(
    cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> int:
    print("+", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    return int(completed.returncode)


def _download_and_extract_tarball(url: str, *, tool: str) -> Path:
    src_root = _src_root()
    src_root.mkdir(parents=True, exist_ok=True)
    target_dir = src_root / tool
    if target_dir.exists():
        return target_dir
    with tempfile.TemporaryDirectory(prefix=f"torchsky_{tool}_download_") as tmp:
        archive_path = Path(tmp) / f"{tool}.tar.gz"
        print(f"+ download {url} -> {archive_path}")
        with urllib.request.urlopen(url) as response, archive_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        extract_dir = Path(tmp) / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(extract_dir, filter="data")
        entries = [item for item in extract_dir.iterdir() if item.name != "__MACOSX"]
        if not entries:
            raise RuntimeError(f"{tool} archive from {url} was empty")
        if len(entries) == 1 and entries[0].is_dir():
            source_dir = entries[0]
        else:
            source_dir = extract_dir
        shutil.copytree(source_dir, target_dir)
    return target_dir


def _ensure_repo(tool: str) -> Path:
    if tool == "casutools":
        source_override = os.environ.get("CASUTOOLS_SOURCE")
        src_dir = _src_root() / "casutools"
        if src_dir.exists():
            return src_dir
        if source_override:
            source_dir = Path(source_override).expanduser()
            if not source_dir.exists():
                raise RuntimeError(f"CASUTOOLS_SOURCE={source_dir} does not exist")
            src_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, src_dir)
            return src_dir
        url = os.environ.get("CASUTOOLS_URL", DEFAULT_CASUTOOLS_URL)
        return _download_and_extract_tarball(url, tool="casutools")
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


def _swarp_build_env(
    conda_prefix: Path,
    *,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Expose Pixi's CFITSIO and its shared dependencies to Autoconf."""

    env = dict(os.environ if base is None else base)
    include_dir = conda_prefix / "include"
    lib_dir = conda_prefix / "lib"
    env["CPPFLAGS"] = f"-I{include_dir}"
    env["LDFLAGS"] = (
        f"-L{lib_dir} -Wl,-rpath,{lib_dir} -Wl,-rpath-link,{lib_dir}"
    )
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(lib_dir), env.get("LD_LIBRARY_PATH", "")) if value
    )
    env["PKG_CONFIG_PATH"] = ":".join(
        value
        for value in (str(lib_dir / "pkgconfig"), env.get("PKG_CONFIG_PATH", ""))
        if value
    )
    system_cc = shutil.which("cc", path="/usr/bin:/bin")
    if system_cc:
        # ponytail: SWarp's Autoconf build is Unix-only; prefer the host linker
        # over a base-image Conda cross-toolchain with an incompatible glibc sysroot.
        env["CC"] = system_cc
    return env


def _build_swarp(src_dir: Path, prefix: Path, *, jobs: int) -> None:
    if (prefix / "bin" / "swarp").exists():
        print(f"swarp already built at {prefix / 'bin' / 'swarp'}")
        return
    conda_prefix = _conda_prefix()
    env = _swarp_build_env(conda_prefix)
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
        raise RuntimeError(
            f"Montage build failed before mProject was produced (make rc={make_rc})"
        )
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
    conda_prefix = _conda_prefix()
    shim_lib = _prefix_root() / "shims" / "casutools" / "lib"
    shim_lib.mkdir(parents=True, exist_ok=True)
    cfitsio_include = conda_prefix / "include"
    cfitsio_lib_dir = conda_prefix / "lib"
    cfitsio_dylib = cfitsio_lib_dir / "libcfitsio.dylib"
    if not cfitsio_dylib.exists():
        raise RuntimeError(
            f"expected {cfitsio_dylib} in the active Pixi env for casutools build"
        )
    for name in ("libcfitsio.a", "libcfitsio.dylib"):
        target = shim_lib / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(cfitsio_dylib)

    wcslib_include = Path(
        os.environ.get("WCSLIB_INCLUDE", str(conda_prefix / "include"))
    )
    wcslib_lib = Path(os.environ.get("WCSLIB_LIB", str(conda_prefix / "lib")))
    wcslib_header = wcslib_include / "wcs.h"
    if not wcslib_header.exists():
        raise RuntimeError(
            "wcslib headers were not found in the active Pixi env; "
            "install wcslib there or set WCSLIB_INCLUDE/WCSLIB_LIB"
        )
    if not wcslib_lib.exists() or not cfitsio_include.exists():
        raise RuntimeError(
            "expected wcslib/cfitsio headers and libraries for casutools build"
        )

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
