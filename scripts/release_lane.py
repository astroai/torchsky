#!/usr/bin/env python3
"""Render or check torchsky's supported-PyTorch range across the repo.

torchsky's native core does not include or link libtorch: kernels take raw
buffers through DLPack and operators are registered from Python with
``torch.library``. One abi3 wheel per platform therefore serves every PyTorch
from the tested floor up, with no upper cap. ``scripts/torch_support.json``
holds:

- ``torchsky_version``: the release being prepared (plain ``X.Y.Z``);
- ``torch_min``: the oldest PyTorch minor CI tests (the runtime floor);
- ``torch_tested``: the newest PyTorch minor CI tests;
- ``torchfits``: the torchfits requirement.

This script renders or checks:

- ``pyproject.toml``: version, ``torch>=<min>`` (no build-time torch) and the
  torchfits requirement;
- ``pixi.toml``: package version, ``pytorch >=<min>`` in the dev features, and
  the ``torch-min`` feature pinned to ``<min>.*``;
- ``src/torchsky/__init__.py`` and ``CITATION.cff``: the version.

It fails when a GitHub workflow hard-codes a torch pin: workflows read the
pins from ``--print-pins``.

Usage::

    release_lane.py --check                  verify committed files match the support map
    release_lane.py --print-pins             print KEY=VALUE pins (CI: >> "$GITHUB_ENV")
    release_lane.py --apply                  render the plain release version
    release_lane.py --apply --prerelease rc1 render a release candidate (0.6.0rc1)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_FILE = ROOT / "scripts" / "torch_support.json"
PYPROJECT = ROOT / "pyproject.toml"
PIXI = ROOT / "pixi.toml"
INIT = ROOT / "src" / "torchsky" / "__init__.py"
CITATION = ROOT / "CITATION.cff"
WORKFLOWS = ROOT / ".github" / "workflows"

_VERSION_LINE_RE = re.compile(r'^(version = ")([^"]+)(")$', re.MULTILINE)
_INIT_VERSION_RE = re.compile(r'^(__version__ = ")([^"]+)(")$', re.MULTILINE)
_CITATION_VERSION_RE = re.compile(r"^(version: )(\S+)$", re.MULTILINE)
_PYPROJECT_TORCH_RE = re.compile(r'^(\s*")torch>=[^"]*(",?)$', re.MULTILINE)
_PYPROJECT_TORCHFITS_RE = re.compile(r'^(\s*")torchfits[^"]*(",?)$', re.MULTILINE)
_PIXI_PYTORCH_RE = re.compile(r'^(pytorch = ")>=[^"]*(")$', re.MULTILINE)
_PIXI_PYTORCH_MIN_RE = re.compile(r'^(pytorch = ")\d+\.\d+\.\*(")$', re.MULTILINE)
_PIXI_TORCH_RE = re.compile(r'^(torch = ")>=[^"]*(")$', re.MULTILINE)
_WORKFLOW_TORCH_RE = re.compile(r"torch\s*(?:==|>=|<=|~=|<)\s*\d")
_PRERELEASE_RE = re.compile(r"(?:rc|a|b)\d+$")
_MINOR_RE = re.compile(r"\d+\.\d+")


def load_support() -> dict[str, str]:
    data = json.loads(SUPPORT_FILE.read_text(encoding="utf-8"))
    if not re.fullmatch(r"\d+\.\d+\.\d+", data.get("torchsky_version", "")):
        raise SystemExit("torch_support.json: bad torchsky_version")
    for key in ("torch_min", "torch_tested"):
        if not _MINOR_RE.fullmatch(data.get(key, "")):
            raise SystemExit(f"torch_support.json: {key} must be a minor such as 2.14")
    if minor_key(data["torch_min"]) > minor_key(data["torch_tested"]):
        raise SystemExit("torch_support.json: torch_min is newer than torch_tested")
    if not re.fullmatch(r">=\d+\.\d+\.\d+", data.get("torchfits", "")):
        raise SystemExit("torch_support.json: bad torchfits spec")
    return data


def minor_key(minor: str) -> tuple[int, int]:
    major, rest = minor.split(".")
    return int(major), int(rest)


def strip_prerelease(version: str) -> str:
    return _PRERELEASE_RE.sub("", version)


def committed_version() -> str:
    match = _VERSION_LINE_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit("pyproject.toml has no version field")
    return match.group(2)


def _replace(text: str, pattern: re.Pattern[str], repl: str, what: str, count: int = 1):
    updated, found = pattern.subn(repl, text)
    if found != count:
        raise SystemExit(f"expected {count} match(es) for {what}; found {found}")
    return updated


def render(version: str | None, prerelease: str | None = None) -> dict[Path, str]:
    spec = load_support()
    base = spec["torchsky_version"]
    if prerelease is not None:
        version = f"{base}{prerelease}"
    elif version is None:
        version = base
    elif strip_prerelease(version) != base:
        raise SystemExit(
            f"torch_support.json prepares torchsky {base}, not {version!r}"
        )
    floor = spec["torch_min"]

    rendered: dict[Path, str] = {}
    py = PYPROJECT.read_text(encoding="utf-8")
    py = _replace(py, _VERSION_LINE_RE, rf"\g<1>{version}\g<3>", "pyproject version")
    py = _replace(
        py, _PYPROJECT_TORCH_RE, rf"\g<1>torch>={floor}\g<2>", "runtime torch"
    )
    py = _replace(
        py,
        _PYPROJECT_TORCHFITS_RE,
        rf"\g<1>torchfits{spec['torchfits']}\g<2>",
        "torchfits requirement",
    )
    rendered[PYPROJECT] = py

    pixi = PIXI.read_text(encoding="utf-8")
    pixi = _replace(pixi, _VERSION_LINE_RE, rf"\g<1>{version}\g<3>", "pixi version")
    pixi = _replace(pixi, _PIXI_PYTORCH_RE, rf"\g<1>>={floor}\g<2>", "pixi pytorch", 2)
    pixi = _replace(
        pixi, _PIXI_PYTORCH_MIN_RE, rf"\g<1>{floor}.*\g<2>", "pixi torch-min"
    )
    pixi = _replace(pixi, _PIXI_TORCH_RE, rf"\g<1>>={floor}\g<2>", "pixi PyPI torch")
    rendered[PIXI] = pixi

    init = INIT.read_text(encoding="utf-8")
    rendered[INIT] = _replace(
        init, _INIT_VERSION_RE, rf"\g<1>{version}\g<3>", "__init__ version"
    )
    cff = CITATION.read_text(encoding="utf-8")
    rendered[CITATION] = _replace(
        cff, _CITATION_VERSION_RE, rf"\g<1>{version}", "CITATION version"
    )
    return rendered


def workflow_problems() -> list[str]:
    problems = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _WORKFLOW_TORCH_RE.search(line):
                problems.append(
                    f"{path.name}:{number}: hard-coded torch pin; "
                    "use release_lane.py --print-pins"
                )
    return problems


def check() -> int:
    try:
        spec = load_support()
        expected = render(committed_version())
    except SystemExit as exc:
        print(f"[FAIL] {exc}", flush=True)
        return 1
    print(
        f"[INFO] torchsky {committed_version()}, torch >={spec['torch_min']} "
        f"(tested {spec['torch_min']}..{spec['torch_tested']})",
        flush=True,
    )
    failed = False
    for path, want in expected.items():
        if path.read_text(encoding="utf-8") == want:
            print(f"[ OK ] {path.name}", flush=True)
        else:
            print(f"[FAIL] {path.name} drifted from torch_support.json", flush=True)
            failed = True
    for problem in workflow_problems():
        print(f"[FAIL] {problem}", flush=True)
        failed = True
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-pins", action="store_true")
    parser.add_argument("--prerelease", help="PEP 440 suffix such as rc1")
    args = parser.parse_args()
    if args.prerelease is not None and not _PRERELEASE_RE.fullmatch(args.prerelease):
        parser.error(f"--prerelease {args.prerelease!r} is not rcN/aN/bN")

    if args.check:
        return check()
    if args.print_pins:
        with contextlib.redirect_stdout(sys.stderr):
            status = check()
        if status != 0:
            return 1
        spec = load_support()
        print(f"VERSION={committed_version()}")
        print(f"TORCH_PIN=>={spec['torch_min']}")
        print(f"TORCH_MIN_PIN=={spec['torch_min']}.*")
        print(f"TORCH_TESTED_PIN=={spec['torch_tested']}.*")
        print(f"TORCHFITS_PIN={spec['torchfits']}")
        return 0
    if not args.apply:
        spec = load_support()
        print(f"torch >={spec['torch_min']} (tested up to {spec['torch_tested']})")
        return 0
    for path, text in render(None, prerelease=args.prerelease).items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
