#!/usr/bin/env python3
"""Run the current TAN/SIP mapmaking comparator suite."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_compat as compat_bench  # noqa: E402
import bench_map_ground_truth as truth_bench  # noqa: E402
from torchsky.map import ImageObservation, coadd_image_plane, plan_image_plane_target  # noqa: E402


DEFAULT_METHODS = (
    "torchsky:bilinear",
    "dfreproject:bilinear",
    "drizzle:point",
    "drizzle:square",
    "montage",
    "swarp:bilinear",
)

DEFAULT_SIP_CASES = (
    "real_astropy_sip_fixture",
    "real_astropy_sip_shifted_wcs",
    "real_astropy_sip_mef",
    "real_astropy_sip_mef_shifted_wcs",
)


@dataclass(frozen=True)
class _SuiteProfile:
    tan_extents: tuple[tuple[int, int], ...]
    tan_methods: tuple[str, ...]
    sip_methods: tuple[str, ...]
    diversity_cases: tuple[str, ...]
    diversity_methods: tuple[str, ...]
    method_sweep_case: str
    method_sweep_size: int
    runs: int
    torchsky_runs: int


@dataclass(frozen=True)
class _DiversityCase:
    case: str
    size: int
    width: int | None = None
    height: int | None = None


_PROFILES = {
    "quick": _SuiteProfile(
        tan_extents=((512, 384), (1024, 768)),
        tan_methods=("torchsky:bilinear", "dfreproject:bilinear", "swarp:bilinear"),
        sip_methods=("dfreproject:bilinear", "drizzle:point", "montage"),
        diversity_cases=(
            "tan_aligned_low_noise@256",
            "tan_aligned_high_noise@256",
            "tan_multires_noise@256",
            "tan_low_overlap_exptime_transmission@256",
            "tan_random10_large_union@512",
            "tan_random10_large_union_rect_weighted_masked_tan@512x384",
        ),
        diversity_methods=(
            "torchsky:bilinear",
            "torchsky:lanczos3",
            "swarp:bilinear",
            "swarp:lanczos3",
        ),
        method_sweep_case="tan_random10_large_union_rect_weighted_masked_tan",
        method_sweep_size=512,
        runs=1,
        torchsky_runs=2,
    ),
    "full": _SuiteProfile(
        tan_extents=((512, 384), (1024, 768), (2048, 1536)),
        tan_methods=DEFAULT_METHODS,
        sip_methods=DEFAULT_METHODS,
        diversity_cases=(
            "tan_aligned_low_noise@256",
            "tan_aligned_high_noise@256",
            "tan_multires_noise@256",
            "tan_multires_extreme@256",
            "tan_low_overlap_exptime_transmission@256",
            "tan_random10_large_union@1024",
            "tan_random10_large_union_rect_weighted_masked_tan@1024x768",
        ),
        diversity_methods=(
            "torchsky:bilinear",
            "torchsky:lanczos3",
            "swarp:bilinear",
            "swarp:lanczos3",
        ),
        method_sweep_case="tan_random10_large_union_rect_weighted_masked_tan",
        method_sweep_size=1024,
        runs=1,
        torchsky_runs=3,
    ),
}


def _resolve_profile(name: str) -> _SuiteProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"unknown profile {name!r}") from exc


def _parse_method_specs(text: str) -> list[tuple[str, str | None]]:
    methods: list[tuple[str, str | None]] = []
    for item in text.split(","):
        spec = item.strip()
        if not spec:
            continue
        if ":" in spec:
            library, variant = spec.split(":", 1)
            methods.append((library.strip(), variant.strip() or None))
        else:
            methods.append((spec, None))
    if not methods:
        raise ValueError("at least one method is required")
    return methods


def _parse_extent_specs(text: str) -> list[tuple[int, int]]:
    extents: list[tuple[int, int]] = []
    for item in text.split(","):
        spec = item.strip().lower()
        if not spec:
            continue
        if "x" not in spec:
            raise ValueError(f"extent must be WIDTHxHEIGHT, got {item!r}")
        width_text, height_text = spec.split("x", 1)
        width = int(width_text)
        height = int(height_text)
        if width <= 0 or height <= 0:
            raise ValueError(f"extent must be positive, got {item!r}")
        extents.append((width, height))
    if not extents:
        raise ValueError("at least one extent is required")
    return extents


def _parse_diversity_case_specs(text: str) -> list[_DiversityCase]:
    cases: list[_DiversityCase] = []
    for item in text.split(","):
        spec = item.strip()
        if not spec:
            continue
        if "@" not in spec:
            raise ValueError(
                f"diversity case must be CASE@SIZE or CASE@WIDTHxHEIGHT, got {item!r}"
            )
        case_name, size_spec = spec.split("@", 1)
        case_name = case_name.strip()
        size_spec = size_spec.strip().lower()
        if not case_name:
            raise ValueError(f"diversity case name is required, got {item!r}")
        if "x" in size_spec:
            width_text, height_text = size_spec.split("x", 1)
            width = int(width_text)
            height = int(height_text)
            if width <= 0 or height <= 0:
                raise ValueError(f"diversity extent must be positive, got {item!r}")
            cases.append(
                _DiversityCase(
                    case=case_name, size=max(width, height), width=width, height=height
                )
            )
        else:
            size = int(size_spec)
            if size <= 0:
                raise ValueError(f"diversity size must be positive, got {item!r}")
            cases.append(_DiversityCase(case=case_name, size=size))
    if not cases:
        raise ValueError("at least one diversity case is required")
    return cases


def _format_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_table(
    rows: list[dict[str, object]],
    *,
    headers: list[str],
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_format_value(row.get(key)) for key in headers) + " |"
        )
    return "\n".join(lines) + "\n"


def _format_markdown(rows: list[dict[str, object]], *, profile: str) -> str:
    tan_rows = [row for row in rows if row.get("suite") == "tan_truth"]
    tan_diversity_rows = [
        row for row in rows if row.get("suite") == "tan_stack_diversity"
    ]
    tan_method_rows = [row for row in rows if row.get("suite") == "tan_stack_methods"]
    sip_rows = [row for row in rows if row.get("suite") == "sip_real"]

    tan_headers = [
        "suite",
        "case",
        "n_obs",
        "library",
        "variant",
        "status",
        "target_width",
        "target_height",
        "time_ms",
        "rmse_vs_truth",
        "mae_vs_truth",
        "p95_abs_err_vs_truth",
        "max_abs_err_vs_truth",
        "signal_to_residual_db",
        "total_flux_relative_error",
        "mask_affected_rmse_vs_truth",
        "mask_clean_rmse_vs_truth",
        "coverage_frac",
        "reason",
    ]
    sip_headers = [
        "suite",
        "case",
        "n_obs",
        "library",
        "variant",
        "status",
        "target_width",
        "target_height",
        "time_ms",
        "max_abs_data_diff_vs_torchsky",
        "max_abs_weight_diff_vs_torchsky",
        "reason",
    ]
    tan_method_headers = [
        "suite",
        "case",
        "n_obs",
        "library",
        "stacking_method",
        "variant",
        "status",
        "target_width",
        "target_height",
        "time_ms",
        "rmse_vs_truth",
        "mae_vs_truth",
        "p95_abs_err_vs_truth",
        "max_abs_err_vs_truth",
        "signal_to_residual_db",
        "total_flux_relative_error",
        "mask_affected_rmse_vs_truth",
        "mask_clean_rmse_vs_truth",
        "coverage_frac",
        "reason",
    ]
    tan_diversity_headers = [
        "suite",
        "case",
        "n_obs",
        "library",
        "variant",
        "status",
        "target_width",
        "target_height",
        "min_input_width",
        "max_input_width",
        "min_input_height",
        "max_input_height",
        "n_weight_maps",
        "n_mask_maps",
        "time_ms",
        "rmse_vs_truth",
        "mae_vs_truth",
        "signal_to_residual_db",
        "total_flux_relative_error",
        "mask_affected_rmse_vs_truth",
        "mask_clean_rmse_vs_truth",
        "reason",
    ]
    parts = [
        "# TAN/SIP Scoreboard",
        "",
        f"Profile: `{profile}`",
        "",
        "## TAN Truth",
        "",
        _format_table(tan_rows, headers=tan_headers).rstrip(),
        "",
        "## TAN Stack Diversity",
        "",
        _format_table(tan_diversity_rows, headers=tan_diversity_headers).rstrip(),
        "",
        "## TAN Stack Methods",
        "",
        _format_table(tan_method_rows, headers=tan_method_headers).rstrip(),
        "",
        "## Real SIP Compatibility",
        "",
        _format_table(sip_rows, headers=sip_headers).rstrip(),
        "",
    ]
    return "\n".join(parts)


def _write_outputs(
    rows: list[dict[str, object]],
    *,
    profile: str,
    json_out: Path,
    markdown_out: Path,
) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(_format_markdown(rows, profile=profile), encoding="utf-8")


def _base_truth_row(
    row: dict[str, Any],
    *,
    suite: str,
    case_label: str,
) -> dict[str, object]:
    out = dict(row)
    out["suite"] = suite
    out["case"] = case_label
    out.setdefault("status", "OK")
    return out


def _truth_case_common_metadata(
    observations: list[truth_bench.SimulatedObservation],
    *,
    case_spec: truth_bench._CaseSpec,
    target_header: dict[str, Any],
    effective_size: int,
) -> dict[str, object]:
    return {
        "projection": case_spec.projection,
        "description": case_spec.description,
        "value_semantics": case_spec.value_semantics,
        "noise_model": case_spec.noise_model,
        "size": effective_size,
        "n_obs": len(observations),
        "target_width": int(target_header["NAXIS1"]),
        "target_height": int(target_header["NAXIS2"]),
        "min_input_width": min(int(item.header["NAXIS1"]) for item in observations),
        "max_input_width": max(int(item.header["NAXIS1"]) for item in observations),
        "min_input_height": min(int(item.header["NAXIS2"]) for item in observations),
        "max_input_height": max(int(item.header["NAXIS2"]) for item in observations),
        "goal": "reconstruct_union_field_on_target_grid",
        "product_geometry": (
            "rectangular"
            if int(target_header["NAXIS1"]) != int(target_header["NAXIS2"])
            else "square"
        ),
        "n_warped_inputs": truth_bench._count_warped_inputs(observations),
        "n_weight_maps": sum(
            1 for item in observations if item.observation.weight_path is not None
        ),
        "n_mask_maps": sum(
            1 for item in observations if item.observation.mask_path is not None
        ),
    }


def _run_tan_truth_suite(
    tmpdir: Path,
    *,
    case: str,
    extents: list[tuple[int, int]],
    methods: list[tuple[str, str | None]],
    runs: int,
    torchsky_runs: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for width, height in extents:
        effective_size = max(width, height)
        case_label = f"{case}@{width}x{height}"
        observations, target_header, truth = truth_bench.make_case(
            tmpdir,
            case=case,
            size=effective_size,
            seed=seed,
            width=width,
            height=height,
        )
        case_spec = truth_bench._case_spec(case, size=effective_size)
        region_masks = truth_bench._target_mask_region_masks(
            observations, target_header=target_header
        )
        obs_list = [item.observation for item in observations]
        baseline_data = truth
        baseline_weight = torch.ones_like(truth, dtype=torch.float32)
        for library, variant in methods:
            if library == "torchsky":
                row = truth_bench.evaluate_torchsky_case(
                    tmpdir,
                    case=case,
                    size=effective_size,
                    width=width,
                    height=height,
                    seed=seed,
                    runs=max(runs, torchsky_runs),
                    resampling_kernel=variant or "bilinear",
                )
            else:
                row, data, weight = compat_bench._comparator_row(
                    library,
                    case_label,
                    obs_list,
                    target_header=target_header,
                    runs=max(2, runs),
                    torchsky_row={
                        "library": "torchsky",
                        "variant": "bilinear",
                        "time_ms": 1.0,
                    },
                    torchsky_data=baseline_data,
                    torchsky_weight=baseline_weight,
                    variant=variant,
                )
                for key in list(row):
                    if key.endswith("_vs_torchsky") or key.startswith(
                        "speed_ratio_torchsky_vs_"
                    ):
                        row.pop(key)
                row.update(
                    {
                        "method": row.get("method", library),
                        "variant": row.get("variant", variant or "default"),
                        **_truth_case_common_metadata(
                            observations,
                            case_spec=case_spec,
                            target_header=target_header,
                            effective_size=effective_size,
                        ),
                    }
                )
                if (
                    row.get("status") == "OK"
                    and data is not None
                    and weight is not None
                ):
                    row.update(
                        truth_bench._truth_metrics(
                            data.to(dtype=torch.float32),
                            weight.to(dtype=torch.float32),
                            truth,
                            target_header=target_header,
                            value_semantics=case_spec.value_semantics,
                            region_masks=region_masks,
                        )
                    )
            rows.append(_base_truth_row(row, suite="tan_truth", case_label=case_label))
    return rows


def _sip_case_observations(
    tmpdir: Path, case: str, *, size: int, n_obs: int
) -> list[ImageObservation]:
    if case == "real_astropy_sip_fixture":
        return compat_bench._prepare_real_case(tmpdir)
    if case == "real_astropy_sip_shifted_wcs":
        return compat_bench._prepare_real_shifted_sip_case(tmpdir)
    if case == "real_astropy_sip_mef":
        return compat_bench._prepare_real_sip_mef_case(tmpdir, shifted=False)
    if case == "real_astropy_sip_mef_shifted_wcs":
        return compat_bench._prepare_real_sip_mef_case(tmpdir, shifted=True)
    raise ValueError(f"unknown SIP case {case!r}")


def _run_sip_real_suite(
    tmpdir: Path,
    *,
    cases: list[str],
    methods: list[tuple[str, str | None]],
    runs: int,
    size: int,
    n_obs: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        observations = _sip_case_observations(tmpdir, case, size=size, n_obs=n_obs)
        target = plan_image_plane_target(observations, reference_index=0)
        torchsky_row, torchsky_data, torchsky_weight = compat_bench._torchsky_row(
            case,
            observations,
            target_header=target.header,
            runs=runs,
            resampling_kernel="bilinear",
        )
        rows.append(
            {
                **torchsky_row,
                "suite": "sip_real",
                "target_width": int(target.header["NAXIS1"]),
                "target_height": int(target.header["NAXIS2"]),
            }
        )
        for library, variant in methods:
            if library == "torchsky":
                continue
            row, _data, _weight = compat_bench._comparator_row(
                library,
                case,
                observations,
                target_header=target.header,
                runs=max(2, runs),
                torchsky_row=torchsky_row,
                torchsky_data=torchsky_data,
                torchsky_weight=torchsky_weight,
                variant=variant,
            )
            row["suite"] = "sip_real"
            row["target_width"] = int(target.header["NAXIS1"])
            row["target_height"] = int(target.header["NAXIS2"])
            rows.append(row)
    return rows


def _run_tan_stack_diversity_suite(
    tmpdir: Path,
    *,
    cases: list[_DiversityCase],
    methods: list[tuple[str, str | None]],
    runs: int,
    torchsky_runs: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_spec_item in cases:
        observations, target_header, truth = truth_bench.make_case(
            tmpdir,
            case=case_spec_item.case,
            size=case_spec_item.size,
            seed=seed,
            width=case_spec_item.width,
            height=case_spec_item.height,
        )
        case_spec = truth_bench._case_spec(
            case_spec_item.case, size=case_spec_item.size
        )
        region_masks = truth_bench._target_mask_region_masks(
            observations,
            target_header=target_header,
        )
        obs_list = [item.observation for item in observations]
        if case_spec_item.width is not None and case_spec_item.height is not None:
            case_label = (
                f"{case_spec_item.case}@{case_spec_item.width}x{case_spec_item.height}"
            )
        else:
            case_label = f"{case_spec_item.case}@{case_spec_item.size}"
        common = _truth_case_common_metadata(
            observations,
            case_spec=case_spec,
            target_header=target_header,
            effective_size=case_spec_item.size,
        )
        baseline_data = truth
        baseline_weight = torch.ones_like(truth, dtype=torch.float32)
        for library, variant in methods:
            if library == "torchsky":
                row = truth_bench.evaluate_torchsky_case(
                    tmpdir,
                    case=case_spec_item.case,
                    size=case_spec_item.size,
                    width=case_spec_item.width,
                    height=case_spec_item.height,
                    seed=seed,
                    runs=max(runs, torchsky_runs),
                    resampling_kernel=variant or "bilinear",
                )
                row.update(
                    {
                        "library": "torchsky",
                        "variant": variant or "bilinear",
                        "stacking_method": "weighted_mean",
                        **common,
                    }
                )
            else:
                row, data, weight = compat_bench._comparator_row(
                    library,
                    case_label,
                    obs_list,
                    target_header=target_header,
                    runs=max(2, runs),
                    torchsky_row={
                        "library": "torchsky",
                        "variant": "bilinear",
                        "time_ms": 1.0,
                    },
                    torchsky_data=baseline_data,
                    torchsky_weight=baseline_weight,
                    variant=variant,
                )
                for key in list(row):
                    if key.endswith("_vs_torchsky") or key.startswith(
                        "speed_ratio_torchsky_vs_"
                    ):
                        row.pop(key)
                row.update(
                    {
                        "method": row.get("method", library),
                        "variant": row.get("variant", variant or "default"),
                        "stacking_method": "weighted_mean",
                        **common,
                    }
                )
                if (
                    row.get("status") == "OK"
                    and data is not None
                    and weight is not None
                ):
                    row.update(
                        truth_bench._truth_metrics(
                            data.to(dtype=torch.float32),
                            weight.to(dtype=torch.float32),
                            truth,
                            target_header=target_header,
                            value_semantics=case_spec.value_semantics,
                            region_masks=region_masks,
                        )
                    )
            rows.append(
                _base_truth_row(
                    row,
                    suite="tan_stack_diversity",
                    case_label=case_label,
                )
            )
    return rows


def _run_tan_stack_method_suite(
    tmpdir: Path,
    *,
    case: str,
    size: int,
    runs: int,
    torchsky_runs: int,
    seed: int,
) -> list[dict[str, object]]:
    observations, target_header, truth = truth_bench.make_case(
        tmpdir,
        case=case,
        size=size,
        seed=seed,
    )
    case_spec = truth_bench._case_spec(case, size=size)
    region_masks = truth_bench._target_mask_region_masks(
        observations, target_header=target_header
    )
    obs_list = [item.observation for item in observations]
    case_label = f"{case}@{size}"
    rows: list[dict[str, object]] = []

    for kernel in ("bilinear", "lanczos3"):
        for stacking_method in ("weighted_mean", "mean", "median"):
            dt = truth_bench._time_many(
                lambda: coadd_image_plane(
                    obs_list,
                    target_header=target_header,
                    resampling_kernel=kernel,
                    value_semantics=case_spec.value_semantics,
                    stacking_method=stacking_method,
                ),
                max(runs, torchsky_runs),
            )
            result = coadd_image_plane(
                obs_list,
                target_header=target_header,
                resampling_kernel=kernel,
                value_semantics=case_spec.value_semantics,
                stacking_method=stacking_method,
            )
            row = {
                "library": "torchsky",
                "method": "coadd",
                "variant": kernel,
                "stacking_method": stacking_method,
                "status": "OK",
                "projection": case_spec.projection,
                "description": case_spec.description,
                "value_semantics": case_spec.value_semantics,
                "noise_model": case_spec.noise_model,
                "size": size,
                "n_obs": len(observations),
                "target_width": int(target_header["NAXIS1"]),
                "target_height": int(target_header["NAXIS2"]),
                "time_ms": dt * 1000.0,
                "mpix_s": (float(result.data.numel()) / dt) / 1e6,
                "interpolation_kernel": kernel,
                "goal": "reconstruct_union_field_on_target_grid",
            }
            row.update(
                truth_bench._truth_metrics(
                    result.data.to(dtype=torch.float32),
                    result.weight.to(dtype=torch.float32),
                    truth,
                    target_header=target_header,
                    value_semantics=case_spec.value_semantics,
                    region_masks=region_masks,
                )
            )
            rows.append(
                _base_truth_row(row, suite="tan_stack_methods", case_label=case_label)
            )

        for stacking_method, combine_type in (
            ("weighted_mean", "SUM(num/denom)"),
            ("mean", "AVERAGE"),
            ("median", "MEDIAN"),
        ):
            try:
                if stacking_method == "weighted_mean":
                    dt = truth_bench._time_many(
                        lambda: compat_bench._coadd_with_swarp(
                            obs_list,
                            target_header=target_header,
                            resampling_type=kernel.upper(),
                        ),
                        max(2, runs),
                    )
                    data_np, weight_np = compat_bench._coadd_with_swarp(
                        obs_list,
                        target_header=target_header,
                        resampling_type=kernel.upper(),
                    )
                else:
                    dt = truth_bench._time_many(
                        lambda: compat_bench._coadd_with_swarp_direct_stack(
                            obs_list,
                            target_header=target_header,
                            resampling_type=kernel.upper(),
                            combine_type=combine_type,
                        ),
                        max(2, runs),
                    )
                    data_np, weight_np = compat_bench._coadd_with_swarp_direct_stack(
                        obs_list,
                        target_header=target_header,
                        resampling_type=kernel.upper(),
                        combine_type=combine_type,
                    )
                data = torch.from_numpy(data_np).to(dtype=torch.float32)
                weight = torch.from_numpy(weight_np).to(dtype=torch.float32)
                row = {
                    "library": "swarp",
                    "method": "swarp",
                    "variant": kernel,
                    "stacking_method": stacking_method,
                    "combine_type": combine_type,
                    "status": "OK",
                    "projection": case_spec.projection,
                    "description": case_spec.description,
                    "value_semantics": case_spec.value_semantics,
                    "noise_model": case_spec.noise_model,
                    "size": size,
                    "n_obs": len(observations),
                    "target_width": int(target_header["NAXIS1"]),
                    "target_height": int(target_header["NAXIS2"]),
                    "time_ms": dt * 1000.0,
                    "mpix_s": (float(data.numel()) / dt) / 1e6,
                    "interpolation_kernel": kernel,
                    "goal": "reconstruct_union_field_on_target_grid",
                }
                row.update(
                    truth_bench._truth_metrics(
                        data,
                        weight,
                        truth,
                        target_header=target_header,
                        value_semantics=case_spec.value_semantics,
                        region_masks=region_masks,
                    )
                )
            except Exception as exc:
                row = {
                    "library": "swarp",
                    "method": "swarp",
                    "variant": kernel,
                    "stacking_method": stacking_method,
                    "combine_type": combine_type,
                    "status": "SKIPPED",
                    "projection": case_spec.projection,
                    "description": case_spec.description,
                    "value_semantics": case_spec.value_semantics,
                    "noise_model": case_spec.noise_model,
                    "size": size,
                    "n_obs": len(observations),
                    "target_width": int(target_header["NAXIS1"]),
                    "target_height": int(target_header["NAXIS2"]),
                    "reason": f"SWarp stack-method sweep failed: {exc}",
                }
            rows.append(
                _base_truth_row(row, suite="tan_stack_methods", case_label=case_label)
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(_PROFILES),
        default="quick",
        help="Benchmark workload profile: quick routine scoreboard or full graduation run",
    )
    parser.add_argument(
        "--tan-case",
        default="tan_random10_large_union_rect_weighted_masked_tan",
        help="Ground-truth TAN case to run",
    )
    parser.add_argument(
        "--tan-extents",
        default=None,
        help="Comma-separated WIDTHxHEIGHT list for TAN truth runs; defaults to the selected profile",
    )
    parser.add_argument(
        "--sip-cases",
        default=",".join(DEFAULT_SIP_CASES),
        help="Comma-separated real SIP compat cases",
    )
    parser.add_argument(
        "--diversity-cases",
        default=None,
        help="Comma-separated CASE@SIZE or CASE@WIDTHxHEIGHT list for TAN weighted-stack diversity rows",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Deprecated global library[:variant] override applied to both TAN and SIP sections",
    )
    parser.add_argument(
        "--tan-methods",
        default=None,
        help="Comma-separated library[:variant] list for TAN truth rows; defaults to the selected profile",
    )
    parser.add_argument(
        "--sip-methods",
        default=None,
        help="Comma-separated library[:variant] list for SIP compat rows; defaults to the selected profile",
    )
    parser.add_argument(
        "--diversity-methods",
        default=None,
        help="Comma-separated library[:variant] list for TAN weighted-stack diversity rows; defaults to the selected profile",
    )
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--torchsky-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--method-sweep-case",
        default=None,
        help="TAN truth case used for the stack-method kernel sweep; defaults to the selected profile",
    )
    parser.add_argument(
        "--method-sweep-size",
        type=int,
        default=None,
        help="Representative size for the stack-method kernel sweep; defaults to the selected profile",
    )
    parser.add_argument("--compat-size", type=int, default=256)
    parser.add_argument("--compat-n-obs", type=int, default=3)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/map_tan_sip_suite.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("benchmark_results/map_tan_sip_suite.md"),
    )
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)
    tan_extents = (
        _parse_extent_specs(args.tan_extents)
        if args.tan_extents is not None
        else list(profile.tan_extents)
    )
    if args.methods is not None:
        tan_methods = _parse_method_specs(args.methods)
        sip_methods = _parse_method_specs(args.methods)
        diversity_methods = _parse_method_specs(args.methods)
    else:
        tan_methods = (
            _parse_method_specs(args.tan_methods)
            if args.tan_methods
            else _parse_method_specs(",".join(profile.tan_methods))
        )
        sip_methods = (
            _parse_method_specs(args.sip_methods)
            if args.sip_methods
            else _parse_method_specs(",".join(profile.sip_methods))
        )
        diversity_methods = (
            _parse_method_specs(args.diversity_methods)
            if args.diversity_methods
            else _parse_method_specs(",".join(profile.diversity_methods))
        )
    sip_cases = [item.strip() for item in args.sip_cases.split(",") if item.strip()]
    diversity_cases = (
        _parse_diversity_case_specs(args.diversity_cases)
        if args.diversity_cases is not None
        else _parse_diversity_case_specs(",".join(profile.diversity_cases))
    )
    runs = profile.runs if args.runs is None else int(args.runs)
    torchsky_runs = (
        profile.torchsky_runs if args.torchsky_runs is None else int(args.torchsky_runs)
    )
    method_sweep_case = (
        profile.method_sweep_case
        if args.method_sweep_case is None
        else str(args.method_sweep_case)
    )
    method_sweep_size = (
        profile.method_sweep_size
        if args.method_sweep_size is None
        else int(args.method_sweep_size)
    )

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="torchsky_map_tan_sip_suite_") as tmp:
        tmpdir = Path(tmp)
        rows.extend(
            _run_tan_truth_suite(
                tmpdir,
                case=args.tan_case,
                extents=tan_extents,
                methods=tan_methods,
                runs=runs,
                torchsky_runs=torchsky_runs,
                seed=args.seed,
            )
        )
        rows.extend(
            _run_tan_stack_diversity_suite(
                tmpdir,
                cases=diversity_cases,
                methods=diversity_methods,
                runs=runs,
                torchsky_runs=torchsky_runs,
                seed=args.seed,
            )
        )
        rows.extend(
            _run_tan_stack_method_suite(
                tmpdir,
                case=method_sweep_case,
                size=method_sweep_size,
                runs=runs,
                torchsky_runs=torchsky_runs,
                seed=args.seed,
            )
        )
        rows.extend(
            _run_sip_real_suite(
                tmpdir,
                cases=sip_cases,
                methods=sip_methods,
                runs=runs,
                size=args.compat_size,
                n_obs=args.compat_n_obs,
            )
        )

    _write_outputs(
        rows,
        profile=args.profile,
        json_out=args.json_out,
        markdown_out=args.markdown_out,
    )
    print(args.markdown_out.read_text(encoding="utf-8"))
    print(f"JSON: {args.json_out}")
    print(f"Markdown: {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
