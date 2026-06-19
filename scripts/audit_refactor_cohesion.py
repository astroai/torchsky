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
    "sphere-bench-geom-exact-contains-gate": ("bench_sphere_geom_exact_contains.py",),
    "audit-refactor-cohesion": ("scripts/audit_refactor_cohesion.py",),
}

REQUIRED_FILES = (
    "docs/architecture.md",
    "docs/benchmarks.md",
    "docs/map.md",
    "docs/field.md",
    "docs/scene.md",
    "docs/reviews/autonomous_refactor_log.md",
    "docs/roadmap/history.md",
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
        "field.fit",
        "field.render",
        "field.sample",
        "operator_constrained",
    ),
    "docs/architecture.md": (
        "scene",
        "world",
        "io",
    ),
}

REQUIRED_SOURCE_EXPORTS: Mapping[str, tuple[str, ...]] = {
    "src/torchsky/surfaces.py": ("API_SURFACES", "api_surfaces_by_pillar"),
    "src/torchsky/field/__init__.py": (
        "fit",
        "render",
        "sample",
        "model",
        "preset",
        "Domain",
        "Observation",
    ),
    "src/torchsky/map/field/bandpass.py": (
        "bandpass_quadrature",
        "bandpass_response",
        "mixture_mc_samples",
    ),
    "src/torchsky/map/field/observation_ops.py": (
        "coordinate_grid",
        "sample_observation_indices",
        "sample_observation_pixels",
        "crop_field_observation",
    ),
    "src/torchsky/map/field/forwarding.py": (
        "call_scalar_field_forward",
        "call_render_bandpass",
        "type_method_accepts_kwarg",
    ),
    "src/torchsky/map/field/hashgrid.py": (
        "HashGridEncoder",
        "hash_grid_smoothstep",
        "sample_feature_plane",
    ),
    "src/torchsky/map/field/networks.py": (
        "ResidualMLPBlock",
        "ResidualStack",
        "silu_mlp_sequential",
    ),
    "src/torchsky/map/field/sky_encoding.py": (
        "make_sky_coordinate_encoder",
        "sky_encoder_output_dim",
    ),
    "src/torchsky/map/field/rendering.py": (
        "render_field_observation",
        "render_field_cube",
        "evaluate_field_coordinates_vmapped",
    ),
    "src/torchsky/map/field/losses.py": ("_field_residual_loss",),
    "src/torchsky/map/field/noise.py": ("_synthetic_noise_target",),
    "src/torchsky/map/field/training.py": (
        "fit_field_model",
        "fit_field_component_decomposition",
        "_noise_pretrain_field_model",
    ),
    "src/torchsky/map/__init__.py": (
        "make_field_model_preset",
        "ObservationOperatorBundle",
        "coadd_image_plane",
    ),
    "src/torchsky/map/operators/__init__.py": (
        "MAP_RESAMPLING_KERNELS",
        "MAP_STACKING_METHODS",
        "MAP_VALUE_SEMANTICS",
        "headers_equivalent_for_direct_copy",
        "is_plain_tan_header",
        "normalized_header_for_shape",
        "pixel_area_sr_map",
        "representative_pixel_area_sr",
        "sample_image_plane_batched_channels",
        "source_bbox_on_target",
        "target_shape_from_header",
        "infer_linear_affine_pixel_map",
        "LearnedInstrumentOperator",
        "ObservationOperatorBundle",
        "validate_image_plane_2d",
        "validate_map_resampling_kernel",
        "validate_map_stacking_method",
        "validate_map_value_semantics",
        "wcs_has_distortion",
        "wrapped_delta_ra_deg",
    ),
    "src/torchsky/scene/__init__.py": (
        "SceneRepresentation",
        "TargetRenderSpec",
    ),
}

FORBIDDEN_SOURCE_PHRASES: Mapping[str, tuple[str, ...]] = {
    "src/torchsky/map/field/core.py": (
        "def _call_scalar_field_forward",
        "def _call_render_bandpass",
        "def _type_method_accepts_kwarg",
        "def _bandpass_quadrature",
        "def _bandpass_response",
        "def _mixture_mc_samples",
        "def _sample_observation_indices",
        "def _sample_observation_pixels",
        "def _crop_field_observation",
        "def _coordinate_grid",
        "def _hash_grid_smoothstep",
        "def _sample_feature_plane",
        "class _HashGridEncoder",
        "class _ResidualMLPBlock",
        "class _ResidualStack",
        "def _make_sky_coordinate_encoder",
        "def _sky_encoder_output_dim",
        "def _silu_mlp_sequential",
        "def _reset_silu_mlp_linears",
        "def _reset_logflux_readout_linear",
        "def render_field_observation",
        "def render_field_cube",
        "def evaluate_field_coordinates_vmapped",
        "def _field_residual_loss",
        "def _noise_pretrain_field_model",
        "def fit_field_model",
        "def fit_field_component_decomposition",
    ),
    "src/torchsky/map/image_plane.py": (
        "def _sample_image_plane_batched_channels",
        "def _infer_linear_affine_pixel_map",
        "def _headers_equivalent_for_direct_copy",
        "def _target_shape_from_header",
    ),
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
            missing_fragments = [
                fragment for fragment in fragments if fragment not in command
            ]
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


def check_source_boundaries(root: Path) -> list[AuditResult]:
    results: list[AuditResult] = []
    for relpath, phrases in FORBIDDEN_SOURCE_PHRASES.items():
        path = root / relpath
        if not path.exists():
            results.append(AuditResult(f"cohesion:{relpath}", "fail", "missing"))
            continue
        text = path.read_text(encoding="utf-8")
        present = [phrase for phrase in phrases if phrase in text]
        if present:
            results.append(
                AuditResult(
                    f"cohesion:{relpath}",
                    "fail",
                    f"extracted helpers moved back into core module: {present}",
                )
            )
        else:
            results.append(
                AuditResult(
                    f"cohesion:{relpath}",
                    "pass",
                    "extracted helper boundaries preserved",
                )
            )
    return results


def run_audit(root: Path = ROOT) -> list[AuditResult]:
    return [
        *check_tasks(root),
        *check_files(root),
        *check_docs(root),
        *check_source_exports(root),
        *check_source_boundaries(root),
    ]


def _exit_code(results: list[AuditResult], *, strict_warnings: bool) -> int:
    failing = [result for result in results if result.status == "fail"]
    warning = [result for result in results if result.status == "warn"]
    if failing or (strict_warnings and warning):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="treat any audit warnings as failures",
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
