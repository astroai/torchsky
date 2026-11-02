#!/usr/bin/env python3
"""Render or check the PyTorch release-lane pins across the repo.

torchsky wheels carry a compiled extension that is ABI-matched to a single
PyTorch minor, the "lane" (PyTorch has no stable C++ ABI across minors).
``scripts/torch_lanes.json`` maps every supported torch minor to the torchsky
release made for it and the ``torchfits`` requirement of that release.

This script renders or checks the lane pins in:

- ``pyproject.toml``: project version, the ``torch`` build requirement, the
  ``torch`` and ``torchfits`` runtime requirements,
- ``pixi.toml``: package version and the ``pytorch`` / ``torch`` pins,
- ``src/torchsky/__init__.py``: ``__version__``,
- ``CITATION.cff``: ``version``.

It also fails when a GitHub workflow hard-codes a torch pin: workflows read the
pins from ``--print-pins``.

Usage::

    release_lane.py --check                  verify committed files match the lane map
    release_lane.py --print-pins             print KEY=VALUE pins (CI: >> "$GITHUB_ENV")
    release_lane.py --apply                  finalize the newest lane to its plain version
    release_lane.py --apply --prerelease rc1 render a release candidate (0.5.0rc1)
    release_lane.py                          print the lane the tree tracks

``--check`` accepts the lane base version with an optional PEP 440 prerelease
suffix (``0.5.0rc1``). ``--apply`` without ``--prerelease`` finalizes to the
plain map version.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANES_FILE = ROOT / "scripts" / "torch_lanes.json"
PYPROJECT = ROOT / "pyproject.toml"
PIXI = ROOT / "pixi.toml"
INIT = ROOT / "src" / "torchsky" / "__init__.py"
CITATION = ROOT / "CITATION.cff"
WORKFLOWS = ROOT / ".github" / "workflows"

_VERSION_LINE_RE = re.compile(r'^(version = ")([^"]+)(")$', re.MULTILINE)
_INIT_VERSION_RE = re.compile(r'^(__version__ = ")([^"]+)(")$', re.MULTILINE)
_CITATION_VERSION_RE = re.compile(r"^(version: )(\S+)$", re.MULTILINE)
_PYPROJECT_TORCH_RE = re.compile(
    r'^(\s*")torch>=\d+\.\d+,<\d+\.\d+(",?)$', re.MULTILINE
)
_PYPROJECT_TORCHFITS_RE = re.compile(r'^(\s*")torchfits[^"]*(",?)$', re.MULTILINE)
_PIXI_PYTORCH_RE = re.compile(r'^(pytorch = ")>=\d+\.\d+,<\d+\.\d+(")$', re.MULTILINE)
_PIXI_PYTORCH_STAR_RE = re.compile(r'^(pytorch = ")\d+\.\d+\.\*(")$', re.MULTILINE)
_PIXI_TORCH_EXACT_RE = re.compile(r'^(torch = ")==\d+\.\d+\.\d+(")$', re.MULTILINE)
_WORKFLOW_TORCH_RE = re.compile(r"torch\s*(?:==|>=|<=|~=|<)\s*\d")
_PRERELEASE_RE = re.compile(r"(?:rc|a|b)\d+$")


def load_lanes() -> dict[str, dict[str, str]]:
    data = json.loads(LANES_FILE.read_text(encoding="utf-8"))
    for lane, spec in data.items():
        if not re.fullmatch(r"\d+\.\d+", lane):
            raise SystemExit(f"torch_lanes.json: invalid lane {lane!r}")
        if not re.fullmatch(r"\d+\.\d+\.\d+", spec.get("torchsky_version", "")):
            raise SystemExit(f"torch_lanes.json: bad torchsky_version for lane {lane}")
        if not re.fullmatch(r">=\d+\.\d+\.\d+", spec.get("torchfits", "")):
            raise SystemExit(f"torch_lanes.json: bad torchfits spec for lane {lane}")
    return data


def next_lane(lane: str) -> str:
    major, minor = lane.split(".")
    return f"{major}.{int(minor) + 1}"


def torch_spec(lane: str) -> str:
    return f">={lane},<{next_lane(lane)}"


def strip_prerelease(version: str) -> str:
    return _PRERELEASE_RE.sub("", version)


def newest_lane(lanes: dict[str, dict[str, str]]) -> str:
    return max(lanes, key=lambda lane: tuple(int(p) for p in lane.split(".")))


def committed_version() -> str:
    match = _VERSION_LINE_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit("pyproject.toml has no version field")
    return match.group(2)


def current_lane() -> str:
    base = strip_prerelease(committed_version())
    for lane, spec in load_lanes().items():
        if spec["torchsky_version"] == base:
            return lane
    raise SystemExit(f"version {base!r} is not a torchsky release-lane version")


def _replace(text: str, pattern: re.Pattern[str], repl: str, what: str, count: int = 1):
    updated, found = pattern.subn(repl, text)
    if found != count:
        raise SystemExit(f"expected {count} match(es) for {what}; found {found}")
    return updated


def render(lane: str, version: str | None, prerelease: str | None = None):
    lanes = load_lanes()
    if lane not in lanes:
        raise SystemExit(f"lane {lane!r} is not in torch_lanes.json")
    spec = lanes[lane]
    base = spec["torchsky_version"]
    if prerelease is not None:
        version = f"{base}{prerelease}"
    elif version is None:
        version = base
    elif strip_prerelease(version) != base:
        raise SystemExit(f"lane {lane} releases as torchsky {base}, not {version!r}")
    pin = torch_spec(lane)

    rendered: dict[Path, str] = {}
    py = PYPROJECT.read_text(encoding="utf-8")
    py = _replace(py, _VERSION_LINE_RE, rf"\g<1>{version}\g<3>", "pyproject version")
    py = _replace(py, _PYPROJECT_TORCH_RE, rf"\g<1>torch{pin}\g<2>", "torch pins", 2)
    py = _replace(
        py,
        _PYPROJECT_TORCHFITS_RE,
        rf"\g<1>torchfits{spec['torchfits']}\g<2>",
        "torchfits requirement",
    )
    rendered[PYPROJECT] = py

    pixi = PIXI.read_text(encoding="utf-8")
    pixi = _replace(pixi, _VERSION_LINE_RE, rf"\g<1>{version}\g<3>", "pixi version")
    pixi = _replace(pixi, _PIXI_PYTORCH_RE, rf"\g<1>{pin}\g<2>", "pixi pytorch", 3)
    pixi = _replace(
        pixi, _PIXI_PYTORCH_STAR_RE, rf"\g<1>{lane}.*\g<2>", "pixi pytorch star"
    )
    pixi = _replace(
        pixi, _PIXI_TORCH_EXACT_RE, rf"\g<1>=={lane}.0\g<2>", "pixi exact torch"
    )
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
        lane = current_lane()
        expected = render(lane, committed_version())
    except SystemExit as exc:
        print(f"[FAIL] {exc}", flush=True)
        return 1
    print(f"[INFO] torch lane {lane}, torchsky {committed_version()}", flush=True)
    failed = False
    for path, want in expected.items():
        if path.read_text(encoding="utf-8") == want:
            print(f"[ OK ] {path.name}", flush=True)
        else:
            print(f"[FAIL] {path.name} drifted from the {lane} lane", flush=True)
            failed = True
    for problem in workflow_problems():
        print(f"[FAIL] {problem}", flush=True)
        failed = True
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", help="torch minor lane (default: newest)")
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
        lane = current_lane()
        print(f"LANE={lane}")
        print(f"VERSION={committed_version()}")
        print(f"TORCH_PIN={torch_spec(lane)}")
        print(f"TORCHFITS_PIN={load_lanes()[lane]['torchfits']}")
        return 0
    if not args.apply:
        print(f"current lane: {current_lane()}")
        return 0
    lane = args.lane or newest_lane(load_lanes())
    rendered = render(lane, None, prerelease=args.prerelease)
    for path, text in rendered.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
