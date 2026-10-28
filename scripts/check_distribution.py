from __future__ import annotations

import argparse
from collections.abc import Iterable
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile


NATIVE_SUFFIXES = (".so", ".dylib", ".pyd")
TORCH_LIBRARY_RE = re.compile(
    r"^(?:lib)?(?:torch(?:_python|_cpu|_cuda|_global_deps)?|c10(?:_cuda)?)(?:\.|$)",
    re.IGNORECASE,
)
STALE_CPP_NAMES = {
    "cache.cpp",
    "compression.cpp",
    "fits.cpp",
    "hdu.cpp",
    "table.cpp",
}
GENERIC_BUILD_PREFIXES = (
    b"/home/runner/work/",
    b"/private/var/folders/",
    b"/tmp/",
    b"/Users/",
)


def _normalized_name(name: str, *, strip_first: bool) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise AssertionError(f"unsafe archive path: {name}")
    parts = path.parts[1:] if strip_first else path.parts
    return PurePosixPath(*parts).as_posix()


def _wheel_payloads(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {
            _normalized_name(info.filename, strip_first=False): archive.read(info)
            for info in archive.infolist()
            if not info.is_dir()
        }


def _sdist_payloads(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r:*") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        roots = {PurePosixPath(member.name).parts[0] for member in files}
        if len(roots) != 1:
            raise AssertionError(f"sdist must have one root directory: {sorted(roots)}")
        payloads: dict[str, bytes] = {}
        for member in files:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise AssertionError(f"cannot read sdist member: {member.name}")
            payloads[_normalized_name(member.name, strip_first=True)] = extracted.read()
        return payloads


def _assert_licenses(names: Iterable[str]) -> None:
    names = tuple(names)
    if not any(PurePosixPath(name).name == "LICENSE" for name in names):
        raise AssertionError("distribution is missing the torchsky LICENSE")
    if not any(name.endswith("pocketfft.LICENSE.md") for name in names):
        raise AssertionError("distribution is missing the pocketfft license")


def _assert_no_stale_native_content(payloads: dict[str, bytes]) -> None:
    for name, payload in payloads.items():
        lowered = name.lower()
        basename = PurePosixPath(lowered).name
        if "extern/cfitsio" in lowered or "libcfitsio" in basename:
            raise AssertionError(f"stale CFITSIO content in distribution: {name}")
        if "/cpp_src/" in f"/{lowered}" and basename in STALE_CPP_NAMES:
            raise AssertionError(f"stale FITS implementation source in distribution: {name}")
        if name.endswith(NATIVE_SUFFIXES) and b"cfitsio" in payload.lower():
            raise AssertionError(f"native extension contains a CFITSIO marker: {name}")


def _assert_no_bundled_torch(names: Iterable[str]) -> None:
    for name in names:
        basename = PurePosixPath(name).name
        if TORCH_LIBRARY_RE.match(basename):
            raise AssertionError(f"wheel bundles a Torch library: {name}")


def _forbidden_prefixes(extra: Iterable[str] = ()) -> tuple[bytes, ...]:
    values = [*extra, os.getcwd(), sys.prefix]
    values.extend(
        value
        for key in ("BUILD_PREFIX", "PREFIX", "RUNNER_TEMP", "GITHUB_WORKSPACE")
        if (value := os.environ.get(key))
    )
    prefixes = {
        os.path.realpath(value).encode()
        for value in values
        if value and os.path.isabs(value) and len(value) > 3
    }
    return tuple(sorted(prefixes, key=len, reverse=True))


def _assert_no_build_prefixes(
    native_payloads: dict[str, bytes], extra_prefixes: Iterable[str] = ()
) -> None:
    exact = _forbidden_prefixes(extra_prefixes)
    for name, payload in native_payloads.items():
        for prefix in exact:
            if prefix in payload:
                raise AssertionError(
                    f"native extension contains absolute build prefix {prefix.decode(errors='replace')!r}: {name}"
                )
        for prefix in GENERIC_BUILD_PREFIXES:
            if prefix in payload:
                raise AssertionError(
                    f"native extension contains absolute build path marker {prefix.decode()!r}: {name}"
                )
        if re.search(rb"[A-Za-z]:\\(?:Users|Temp|runneradmin)\\", payload):
            raise AssertionError(f"native extension contains a Windows build path: {name}")


def _inspect_linux_dependencies(native: Path) -> None:
    readelf = subprocess.run(
        ["readelf", "-d", native], capture_output=True, text=True, check=False
    )
    if readelf.returncode != 0:
        return
    if re.search(r"(?:RPATH|RUNPATH).*\[(?:/Users/|/home/runner/|/tmp/|/private/)", readelf.stdout):
        raise AssertionError(f"native extension has an absolute build RPATH:\n{readelf.stdout}")
    ldd = subprocess.run(
        ["ldd", native], capture_output=True, text=True, check=False
    )
    if ldd.returncode != 0:
        return
    unresolved = []
    for line in ldd.stdout.splitlines():
        if "=> not found" not in line:
            continue
        library = line.strip().split()[0]
        if not TORCH_LIBRARY_RE.match(library):
            unresolved.append(library)
    if unresolved:
        raise AssertionError(
            f"unexpected unresolved native libraries: {', '.join(sorted(unresolved))}"
        )


def _inspect_macos_dependencies(native: Path) -> None:
    result = subprocess.run(
        ["otool", "-L", native], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return
    unexpected = []
    for line in result.stdout.splitlines()[1:]:
        dependency = line.strip().split(" (", 1)[0]
        basename = PurePosixPath(dependency).name
        if dependency.startswith(("/System/Library/", "/usr/lib/")):
            continue
        if dependency.startswith("@rpath/") and TORCH_LIBRARY_RE.match(basename):
            continue
        if dependency.startswith(("@loader_path/", "@executable_path/")):
            continue
        unexpected.append(dependency)
    if unexpected:
        raise AssertionError(
            "unexpected non-system native libraries: " + ", ".join(unexpected)
        )


def _inspect_native_payloads(native_payloads: dict[str, bytes]) -> None:
    inspector = None
    if platform.system() == "Linux":
        inspector = _inspect_linux_dependencies
    elif platform.system() == "Darwin":
        inspector = _inspect_macos_dependencies
    if inspector is None:
        return
    with tempfile.TemporaryDirectory(prefix="torchsky-dist-check-") as directory:
        for index, (name, payload) in enumerate(native_payloads.items()):
            suffix = PurePosixPath(name).suffix
            native = Path(directory) / f"native-{index}{suffix}"
            native.write_bytes(payload)
            native.chmod(0o755)
            inspector(native)


def check_wheel(
    path: Path,
    *,
    inspect_native: bool = True,
    extra_prefixes: Iterable[str] = (),
) -> None:
    payloads = _wheel_payloads(path)
    names = tuple(payloads)
    if not any(name.startswith("torchsky/") and name.endswith(".py") for name in names):
        raise AssertionError(f"wheel contains no torchsky Python files: {path}")
    native_payloads = {
        name: payload
        for name, payload in payloads.items()
        if PurePosixPath(name).name.startswith("_cpp_ext")
        and name.endswith(NATIVE_SUFFIXES)
    }
    if not native_payloads:
        raise AssertionError(f"wheel contains no _cpp_ext native extension: {path}")
    _assert_licenses(names)
    _assert_no_bundled_torch(names)
    _assert_no_stale_native_content(payloads)
    _assert_no_build_prefixes(native_payloads, extra_prefixes)
    if inspect_native:
        _inspect_native_payloads(native_payloads)


def _expected_cpp_sources(project_root: Path) -> set[str]:
    cpp_root = project_root / "src" / "torchsky" / "cpp_src"
    suffixes = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".in"}
    expected = {
        path.relative_to(project_root).as_posix()
        for path in cpp_root.rglob("*")
        if path.is_file() and (path.suffix in suffixes or path.name == "CMakeLists.txt")
    }
    if not expected:
        raise AssertionError(f"no C++ sources found under {cpp_root}")
    return expected


def check_sdist(path: Path, *, project_root: Path) -> None:
    payloads = _sdist_payloads(path)
    names = set(payloads)
    missing = sorted(_expected_cpp_sources(project_root) - names)
    if missing:
        raise AssertionError("sdist is missing C++ build inputs: " + ", ".join(missing))
    _assert_licenses(names)
    _assert_no_stale_native_content(payloads)
    provenance = "src/torchsky/cpp_src/third_party/README.md"
    if provenance not in names:
        raise AssertionError("sdist is missing pocketfft provenance README")


def _twine_check(paths: list[Path]) -> None:
    subprocess.run(
        [sys.executable, "-m", "twine", "check", *map(str, paths)], check=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", action="append", type=Path, default=[])
    parser.add_argument("--sdist", action="append", type=Path, default=[])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--forbid-prefix", action="append", default=[])
    parser.add_argument("--skip-native-inspection", action="store_true")
    parser.add_argument("--skip-twine", action="store_true")
    args = parser.parse_args()
    paths = [*args.wheel, *args.sdist]
    if not paths:
        parser.error("at least one --wheel or --sdist is required")
    for path in args.wheel:
        check_wheel(
            path,
            inspect_native=not args.skip_native_inspection,
            extra_prefixes=args.forbid_prefix,
        )
    for path in args.sdist:
        check_sdist(path, project_root=args.project_root.resolve())
    if not args.skip_twine:
        _twine_check(paths)
    print("distribution checks passed:", *(str(path) for path in paths))


if __name__ == "__main__":
    main()
