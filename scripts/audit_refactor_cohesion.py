#!/usr/bin/env python3
"""Lightweight cohesion audit for refactor and benchmark wiring.

This script is intentionally cheap enough to run in the default Pixi
environment. It checks wiring and documentation for the current refactor
surface; it does not run heavy benchmark gates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import tomllib
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_TASKS: Mapping[str, tuple[str, ...]] = {
    "bench-map-acceptance-smoke": (
        "bench-map-ground-truth-gate",
        "bench-map-multifield-gate",
        "bench-map-field-fit-gate",
        "bench-map-compat-gate",
    ),
    "bench-map-ground-truth-gate": ("bench_map_ground_truth.py",),
    "bench-map-multifield-gate": ("bench_map_multifield_truth.py",),
    "bench-map-field-fit-gate": ("bench_map_field_fit.py",),
    "bench-map-compat-gate": ("bench_map_compat.py",),
    "bench-wcs-tpv-inverse": ("bench_wcs_tpv_inverse.py",),
    "sphere-bench-geom-exact-contains-gate": (
        "bench_sphere_geom_exact_contains.py",
    ),
    "audit-refactor-cohesion": ("scripts/audit_refactor_cohesion.py",),
}

REQUIRED_FILES = (
    "docs/architecture.md",
    "docs/benchmarks.md",
    "docs/map.md",
    "docs/field.md",
    "docs/scene.md",
    "docs/reviews/autonomous_refactor_log.md",
    "docs/roadmap/status.md",
    "benchmarks/bench_map_ground_truth.py",
    "benchmarks/bench_map_multifield_truth.py",
    "benchmarks/bench_map_field_fit.py",
    "benchmarks/bench_map_compat.py",
    "benchmarks/baselines/map_ground_truth_smoke.json",
    "benchmarks/baselines/map_multifield_truth_smoke.json",
    "benchmarks/baselines/map_field_fit_smoke.json",
    "benchmarks/baselines/map_compat_real_smoke.json",
)

DOC_REQUIRED_PHRASES: Mapping[str, tuple[str, ...]] = {
    "docs/benchmarks.md": (
        "pixi run audit-refactor-cohesion",
        "pixi run bench-map-acceptance-smoke",
        "bench_map_ground_truth.py",
        "bench_map_multifield_truth.py",
        "bench_map_field_fit.py",
    ),
    "docs/map.md": (
        "ObservationOperatorBundle",
        "LearnedInstrumentOperator",
        "ConstrainedKernelPSF",
    ),
    "docs/field.md": (
        "make_field_model_preset",
        "operator_constrained",
    ),
    "docs/architecture.md": (
        "scene",
        "world",
        "io",
    ),
}

REQUIRED_SOURCE_EXPORTS: Mapping[str, tuple[str, ...]] = {
    "src/torchsky/__init__.py": ("API_SURFACES", "api_surfaces_by_pillar"),
    "src/torchsky/map/__init__.py": (
        "make_field_model_preset",
        "ObservationOperatorBundle",
        "coadd_image_plane",
    ),
    "src/torchsky/map/operators/__init__.py": (
        "is_plain_tan_header",
        "source_bbox_on_target",
        "LearnedInstrumentOperator",
        "ObservationOperatorBundle",
        "wcs_has_distortion",
    ),
    "src/torchsky/scene/__init__.py": (
        "SceneRepresentation",
        "TargetRenderSpec",
    ),
}

WARN_LINE_LIMITS = {
    "src/torchsky/map/image_plane.py": 3_000,
    "src/torchsky/map/field/core.py": 6_000,
}


@dataclass(frozen=True)
class AuditResult:
    name: str
    status: str
    detail: str


def _load_toml(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _tasks_from_pixi_toml(root: Path) -> dict[str, str]:
    path = root / "pixi.toml"
    if not path.exists():
        return {}
    return dict(_load_toml(path).get("tasks", {}))


def _tasks_from_pyproject(root: Path) -> dict[str, str]:
    path = root / "pyproject.toml"
    if not path.exists():
        return {}
    data = _load_toml(path)
    return dict(data.get("tool", {}).get("pixi", {}).get("tasks", {}))


def check_tasks(root: Path) -> list[AuditResult]:
    results: list[AuditResult] = []
    task_sources = {
        "pixi.toml": _tasks_from_pixi_toml(root),
        "pyproject.toml": _tasks_from_pyproject(root),
    }
    for source_name, tasks in task_sources.items():
        for task, fragments in REQUIRED_TASKS.items():
            command = tasks.get(task)
            if command is None:
                results.append(
                    AuditResult(
                        f"{source_name}:{task}",
                        "fail",
                        "task is missing",
                    )
                )
                continue
            missing_fragments = [fragment for fragment in fragments if fragment not in command]
            if missing_fragments:
                results.append(
                    AuditResult(
                        f"{source_name}:{task}",
                        "fail",
                        f"missing command fragments: {missing_fragments}",
                    )
                )
                continue
            results.append(AuditResult(f"{source_name}:{task}", "pass", command))
    return results


def check_files(root: Path) -> list[AuditResult]:
    results: list[AuditResult] = []
    for relpath in REQUIRED_FILES:
        path = root / relpath
        status = "pass" if path.exists() else "fail"
        detail = "exists" if path.exists() else "missing"
        results.append(AuditResult(f"file:{relpath}", status, detail))
    return results


def check_docs(root: Path) -> list[AuditResult]:
    results: list[AuditResult] = []
    for relpath, phrases in DOC_REQUIRED_PHRASES.items():
        path = root / relpath
        if not path.exists():
            results.append(AuditResult(f"doc:{relpath}", "fail", "missing"))
            continue
        text = path.read_text(encoding="utf-8")
        missing = [phrase for phrase in phrases if phrase not in text]
        if missing:
            results.append(
                AuditResult(
                    f"doc:{relpath}",
                    "fail",
                    f"missing phrases: {missing}",
                )
            )
        else:
            results.append(AuditResult(f"doc:{relpath}", "pass", "phrases present"))
    return results


def check_source_exports(root: Path) -> list[AuditResult]:
    results: list[AuditResult] = []
    for relpath, attrs in REQUIRED_SOURCE_EXPORTS.items():
        path = root / relpath
        if not path.exists():
            results.append(AuditResult(f"exports:{relpath}", "fail", "missing"))
            continue
        text = path.read_text(encoding="utf-8")
        missing = [attr for attr in attrs if attr not in text]
        if missing:
            results.append(
                AuditResult(
                    f"exports:{relpath}",
                    "fail",
                    f"missing exported names: {missing}",
                )
            )
        else:
            results.append(AuditResult(f"exports:{relpath}", "pass", "names present"))
    return results


def check_line_limits(root: Path) -> list[AuditResult]:
    results: list[AuditResult] = []
    for relpath, limit in WARN_LINE_LIMITS.items():
        path = root / relpath
        if not path.exists():
            results.append(AuditResult(f"cohesion:{relpath}", "fail", "missing"))
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        status = "warn" if line_count > limit else "pass"
        detail = f"{line_count} lines; warning threshold is {limit}"
        results.append(AuditResult(f"cohesion:{relpath}", status, detail))
    return results


def run_audit(root: Path = ROOT) -> list[AuditResult]:
    return [
        *check_tasks(root),
        *check_files(root),
        *check_docs(root),
        *check_source_exports(root),
        *check_line_limits(root),
    ]


def _exit_code(results: list[AuditResult], *, strict_warnings: bool) -> int:
    failing = [result for result in results if result.status == "fail"]
    warning = [result for result in results if result.status == "warn"]
    if failing or (strict_warnings and warning):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="treat cohesion warnings, such as large files, as failures",
    )
    args = parser.parse_args(argv)

    results = run_audit(ROOT)
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        for result in results:
            print(f"{result.status.upper():4} {result.name}: {result.detail}")
        failures = sum(1 for result in results if result.status == "fail")
        warnings = sum(1 for result in results if result.status == "warn")
        print(f"summary: {failures} failures, {warnings} warnings")
    return _exit_code(results, strict_warnings=args.strict_warnings)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
