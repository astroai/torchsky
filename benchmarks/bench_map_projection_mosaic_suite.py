#!/usr/bin/env python3
"""Run the wide-projection mosaicking scoreboard beyond TAN/SIP."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_compat as compat_bench  # noqa: E402
import bench_map_field_fit as field_fit_bench  # noqa: E402
import bench_map_ground_truth as truth_bench  # noqa: E402
import torchsky  # noqa: E402
from torchsky.map import (  # noqa: E402
    FieldObservation,
    ImageObservation,
    estimate_field_domain,
    evaluate_map_product,
    fit_field_model,
    make_field_model,
    plan_map_product,
    render_field_cube,
    render_field_product,
    stack_map_product,
)


DEFAULT_TRUTH_METHODS = (
    "torchsky:bilinear",
    "torchsky:lanczos3",
)

DEFAULT_METHOD_SWEEP_KERNELS = (
    "bilinear",
    "lanczos3",
)

DEFAULT_STACK_METHODS = (
    "weighted_mean",
    "mean",
    "median",
)

DEFAULT_MONTAGE_METHODS = (
    "torchsky:bilinear",
    "torchsky:lanczos3",
    "montage",
)

DEFAULT_STACK_VS_INR_MODELS = (
    "coadd_bilinear",
    "coadd_lanczos3",
    "wire_implicit@sphere_xyz",
    "sky_encoder_mlp@sphere_xyz",
    "hashgrid_mlp@sphere_xyz",
)

_ZENITHAL_PROJECTIONS = {"SIN", "ARC", "ZEA", "STG"}
_CYLINDRICAL_PROJECTIONS = {"CEA", "MER"}
_ALLSKY_PROJECTIONS = {"AIT", "MOL"}
_PROJECTIONS = ("SIN", "ARC", "ZEA", "STG", "CEA", "MER", "AIT", "MOL")
_SOURCE_CASES = {
    "aligned": "tan_aligned_low_noise",
    "shifted_overlap": "tan_multires_noise",
    "multires": "tan_multires_extreme",
    "low_overlap": "tan_low_overlap_exptime_transmission",
    "large_union": "tan_random10_large_union",
    "large_union_masked": "tan_random10_large_union_rect_weighted_masked_tan",
}


@dataclass(frozen=True)
class _ProjectionCase:
    projection: str
    source_case: str
    size: int
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class _StackVsINRCase:
    projection: str
    field_case: str
    size: int
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class _SuiteProfile:
    truth_cases: tuple[str, ...]
    equal_psf_cases: tuple[str, ...]
    method_sweep_cases: tuple[str, ...]
    method_sweep_kernels: tuple[str, ...]
    method_sweep_methods: tuple[str, ...]
    montage_cases: tuple[str, ...]
    stack_vs_inr_cases: tuple[str, ...]
    truth_methods: tuple[str, ...]
    montage_methods: tuple[str, ...]
    stack_vs_inr_models: tuple[str, ...]
    runs: int
    torchsky_runs: int
    field_steps: int
    field_runs: int


def _format_projection_case_spec(
    projection: str,
    source_case: str,
    *,
    size: int,
    width: int | None = None,
    height: int | None = None,
) -> str:
    if width is not None and height is not None:
        extent = f"{width}x{height}"
    else:
        extent = str(size)
    return f"{projection}:{source_case}@{extent}"


def _full_projection_case_specs() -> tuple[str, ...]:
    cases: list[str] = []
    for projection in _PROJECTIONS:
        if projection in _ZENITHAL_PROJECTIONS:
            medium_width = None
            medium_height = None
            medium_size = 768
            large_width = None
            large_height = None
            large_size = 1536
        else:
            medium_width = 1024
            medium_height = 640
            medium_size = 1024
            large_width = 1536
            large_height = 768
            large_size = 1536
        cases.extend(
            (
                _format_projection_case_spec(
                    projection,
                    _SOURCE_CASES["aligned"],
                    size=medium_size,
                    width=medium_width,
                    height=medium_height,
                ),
                _format_projection_case_spec(
                    projection,
                    _SOURCE_CASES["shifted_overlap"],
                    size=medium_size,
                    width=medium_width,
                    height=medium_height,
                ),
                _format_projection_case_spec(
                    projection,
                    _SOURCE_CASES["multires"],
                    size=medium_size,
                    width=medium_width,
                    height=medium_height,
                ),
                _format_projection_case_spec(
                    projection,
                    _SOURCE_CASES["low_overlap"],
                    size=medium_size,
                    width=medium_width,
                    height=medium_height,
                ),
                _format_projection_case_spec(
                    projection,
                    _SOURCE_CASES["large_union_masked"]
                    if projection in _CYLINDRICAL_PROJECTIONS | _ALLSKY_PROJECTIONS
                    else _SOURCE_CASES["large_union"],
                    size=large_size,
                    width=large_width,
                    height=large_height,
                ),
            )
        )
    return tuple(cases)


_PROFILES = {
    "quick": _SuiteProfile(
        truth_cases=(
            "SIN:tan_aligned_low_noise@768",
            "ARC:tan_multires_noise@768",
            "ZEA:tan_low_overlap_exptime_transmission@768",
            "STG:tan_random10_large_union@768",
            "CEA:tan_random10_large_union_rect_weighted_masked_tan@1024x640",
            "MER:tan_multires_noise@1024x640",
            "AIT:tan_random10_large_union@1024x512",
            "MOL:tan_random10_large_union@1024x512",
        ),
        equal_psf_cases=(
            "SIN:tan_aligned_low_noise@768",
            "ARC:tan_aligned_low_noise@768",
            "AIT:tan_aligned_low_noise@1024x512",
        ),
        method_sweep_cases=(
            "CEA:tan_random10_large_union_rect_weighted_masked_tan@1024x640",
            "AIT:tan_random10_large_union@1024x512",
        ),
        method_sweep_kernels=DEFAULT_METHOD_SWEEP_KERNELS,
        method_sweep_methods=DEFAULT_STACK_METHODS,
        montage_cases=(
            "CEA:tan_multires_noise@1024x640",
            "AIT:tan_multires_noise@1024x512",
        ),
        stack_vs_inr_cases=(
            "CEA:tan_field_fit@768x512",
            "AIT:tan_sip_field_fit_multicolor@768x512",
        ),
        truth_methods=DEFAULT_TRUTH_METHODS,
        montage_methods=DEFAULT_MONTAGE_METHODS,
        stack_vs_inr_models=(
            "coadd_bilinear",
            "coadd_lanczos3",
            "wire_implicit@sphere_xyz",
            "sky_encoder_mlp@sphere_xyz",
            "hashgrid_mlp@sphere_xyz",
        ),
        runs=1,
        torchsky_runs=2,
        field_steps=8,
        field_runs=1,
    ),
    "full": _SuiteProfile(
        truth_cases=_full_projection_case_specs(),
        equal_psf_cases=(
            "SIN:tan_aligned_low_noise@768",
            "ARC:tan_aligned_low_noise@768",
            "STG:tan_aligned_low_noise@768",
            "CEA:tan_aligned_low_noise@1024x640",
            "MER:tan_aligned_low_noise@1024x640",
            "AIT:tan_aligned_low_noise@1024x512",
            "MOL:tan_aligned_low_noise@1024x512",
        ),
        method_sweep_cases=(
            "ARC:tan_multires_noise@768",
            "STG:tan_random10_large_union@1536",
            "CEA:tan_random10_large_union_rect_weighted_masked_tan@1536x768",
            "MER:tan_random10_large_union_rect_weighted_masked_tan@1536x768",
            "AIT:tan_random10_large_union@1536x768",
            "MOL:tan_random10_large_union@1536x768",
        ),
        method_sweep_kernels=DEFAULT_METHOD_SWEEP_KERNELS,
        method_sweep_methods=DEFAULT_STACK_METHODS,
        montage_cases=(
            "SIN:tan_multires_noise@768",
            "ARC:tan_multires_extreme@768",
            "ZEA:tan_low_overlap_exptime_transmission@768",
            "STG:tan_random10_large_union@1536",
            "CEA:tan_random10_large_union_rect_weighted_masked_tan@1024x640",
            "MER:tan_multires_noise@1024x640",
            "AIT:tan_random10_large_union@1536x768",
            "MOL:tan_random10_large_union@1536x768",
        ),
        stack_vs_inr_cases=(
            "CEA:tan_field_fit_low_overlap_depth@1024x640",
            "AIT:tan_sip_field_fit_multicolor@1024x512",
        ),
        truth_methods=DEFAULT_TRUTH_METHODS,
        montage_methods=DEFAULT_MONTAGE_METHODS,
        stack_vs_inr_models=(
            "coadd_bilinear",
            "coadd_lanczos3",
            "wire_implicit@sphere_xyz",
            "fourier_mlp@local_tangent",
            "fourier_mlp@sphere_xyz",
            "sky_encoder_mlp@sphere_xyz",
            "hashgrid_mlp@sphere_xyz",
        ),
        runs=1,
        torchsky_runs=3,
        field_steps=36,
        field_runs=1,
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


def _field_fit_kwargs_for_model(model_kind: str) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "lr": 5.0e-3,
        "num_spectral_samples": 9,
        "loss_kind": "charbonnier",
        "lr_schedule": "cosine",
        "grad_clip_norm": 1.0,
    }
    if model_kind == "wire_implicit":
        kwargs["lr"] = 4.0e-3
    elif model_kind == "fourier_mlp":
        kwargs["lr"] = 3.0e-3
        kwargs["grad_clip_norm"] = 0.8
    elif model_kind == "sky_encoder_mlp":
        kwargs["lr"] = 3.0e-3
        kwargs["grad_clip_norm"] = 0.75
    elif model_kind == "hashgrid_mlp":
        kwargs["lr"] = 2.0e-3
        kwargs["grad_clip_norm"] = 0.5
    return kwargs


def _parse_projection_case_specs(text: str) -> list[_ProjectionCase]:
    cases: list[_ProjectionCase] = []
    for item in text.split(","):
        spec = item.strip()
        if not spec:
            continue
        if ":" not in spec or "@" not in spec:
            raise ValueError(
                f"projection case must be PROJ:CASE@SIZE or PROJ:CASE@WIDTHxHEIGHT, got {item!r}"
            )
        projection, rest = spec.split(":", 1)
        source_case, extent_spec = rest.split("@", 1)
        projection = projection.strip().upper()
        source_case = source_case.strip()
        extent_spec = extent_spec.strip().lower()
        if projection not in _PROJECTIONS:
            raise ValueError(f"unsupported projection {projection!r}")
        if not source_case:
            raise ValueError(f"source case is required, got {item!r}")
        if "x" in extent_spec:
            width_text, height_text = extent_spec.split("x", 1)
            width = int(width_text)
            height = int(height_text)
            if width <= 0 or height <= 0:
                raise ValueError(f"extent must be positive, got {item!r}")
            cases.append(
                _ProjectionCase(
                    projection=projection,
                    source_case=source_case,
                    size=max(width, height),
                    width=width,
                    height=height,
                )
            )
        else:
            size = int(extent_spec)
            if size <= 0:
                raise ValueError(f"size must be positive, got {item!r}")
            cases.append(
                _ProjectionCase(
                    projection=projection,
                    source_case=source_case,
                    size=size,
                )
            )
    if not cases:
        raise ValueError("at least one projection case is required")
    return cases


def _parse_stack_vs_inr_case_specs(text: str) -> list[_StackVsINRCase]:
    parsed = _parse_projection_case_specs(text)
    return [
        _StackVsINRCase(
            projection=item.projection,
            field_case=item.source_case,
            size=item.size,
            width=item.width,
            height=item.height,
        )
        for item in parsed
    ]


def _parse_name_list(text: str, *, label: str) -> list[str]:
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise ValueError(f"at least one {label} is required")
    return items


def _parse_inr_model_spec(text: str) -> tuple[str, str | None]:
    spec = text.strip()
    if not spec:
        raise ValueError("INR model spec must not be empty")
    if "@" not in spec:
        return spec, None
    model_kind, coord_mode = spec.split("@", 1)
    model_kind = model_kind.strip()
    coord_mode = coord_mode.strip()
    if not model_kind or not coord_mode:
        raise ValueError(f"invalid INR model spec {text!r}")
    return model_kind, coord_mode


def _format_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_table(rows: list[dict[str, object]], *, headers: list[str]) -> str:
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
    truth_rows = [row for row in rows if row.get("suite") == "projection_truth"]
    equal_psf_rows = [row for row in rows if row.get("suite") == "projection_equal_psf"]
    method_rows = [
        row for row in rows if row.get("suite") == "projection_stack_methods"
    ]
    montage_rows = [row for row in rows if row.get("suite") == "montage_compat"]
    inr_rows = [row for row in rows if row.get("suite") == "stack_vs_inr"]

    truth_headers = [
        "suite",
        "case",
        "projection",
        "source_case",
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
        "coverage_frac",
        "support_fringe_q25_rmse_vs_truth",
        "support_core_q75_rmse_vs_truth",
        "reason",
    ]
    method_headers = [
        "suite",
        "case",
        "projection",
        "source_case",
        "n_obs",
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
        "coverage_frac",
        "support_fringe_q25_rmse_vs_truth",
        "support_core_q75_rmse_vs_truth",
        "reason",
    ]
    equal_psf_headers = [
        "suite",
        "case",
        "projection",
        "source_case",
        "n_obs",
        "stacking_method",
        "variant",
        "status",
        "target_width",
        "target_height",
        "control_psf_sigma_pix",
        "rmse_vs_truth",
        "mae_vs_truth",
        "coverage_frac",
        "reason",
    ]
    montage_headers = [
        "suite",
        "case",
        "projection",
        "source_case",
        "n_obs",
        "library",
        "variant",
        "status",
        "target_width",
        "target_height",
        "time_ms",
        "rmse_vs_truth",
        "mae_vs_truth",
        "rmse_data_diff_vs_torchsky",
        "mae_data_diff_vs_torchsky",
        "comparison_support_frac_vs_torchsky",
        "max_abs_data_diff_vs_torchsky",
        "max_abs_weight_diff_vs_torchsky",
        "comparison_weight_mode",
        "comparison_data_domain",
        "coverage_frac",
        "reason",
    ]
    inr_headers = [
        "suite",
        "case",
        "projection",
        "field_case",
        "n_obs",
        "num_bands",
        "model_kind",
        "field_coord_mode",
        "type",
        "status",
        "target_width",
        "target_height",
        "common_psf_sigma_pix",
        "observation_psf_mode",
        "evaluation_psf_mode",
        "fit_schedule",
        "fit_time_ms",
        "reconstruction_rmse",
        "reconstruction_mae",
        "reconstruction_p95_abs_err",
        "reconstruction_max_abs_err",
        "best_direct_model_kind",
        "best_direct_reconstruction_rmse",
        "rmse_delta_vs_best_direct",
        "fit_time_ratio_vs_best_direct",
        "best_inr_model_kind",
        "best_inr_reconstruction_rmse",
        "rmse_delta_vs_best_inr",
        "signal_to_residual_db",
        "total_flux_relative_error",
        "coverage_fraction",
        "reason",
    ]
    parts = [
        "# Projection And Mosaicking Scoreboard",
        "",
        f"Profile: `{profile}`",
        "",
        "## Projection Truth",
        "",
        _format_table(truth_rows, headers=truth_headers).rstrip(),
        "",
        "## Projection Equal-PSF Controls",
        "",
        _format_table(equal_psf_rows, headers=equal_psf_headers).rstrip(),
        "",
        "## Projection Stack Methods",
        "",
        _format_table(method_rows, headers=method_headers).rstrip(),
        "",
        "## Montage Compatibility",
        "",
        _format_table(montage_rows, headers=montage_headers).rstrip(),
        "",
        "## INR Common-PSF",
        "",
        _format_table(inr_rows, headers=inr_headers).rstrip(),
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


def _extent_label(width: int | None, height: int | None, size: int) -> str:
    if width is not None and height is not None:
        return f"{width}x{height}"
    return str(size)


def _case_label(
    projection: str,
    source_case: str,
    *,
    size: int,
    width: int | None,
    height: int | None,
) -> str:
    return f"{projection}:{source_case}@{_extent_label(width, height, size)}"


def _support_region_masks(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    positive = weight > 0.0
    if not bool(positive.any()):
        empty = torch.zeros_like(positive, dtype=torch.bool)
        return {
            "support_fringe_q25": empty,
            "support_core_q75": empty,
        }
    positive_weight = weight[positive]
    q25 = float(torch.quantile(positive_weight, 0.25).item())
    q75 = float(torch.quantile(positive_weight, 0.75).item())
    return {
        "support_fringe_q25": positive & (weight <= q25),
        "support_core_q75": positive & (weight >= q75),
    }


def _projection_scale_deg(projection: str, source_case: str) -> float:
    if source_case in {
        "tan_random10_large_union",
        "tan_random10_large_union_rect_weighted_masked_tan",
    }:
        if projection in _ALLSKY_PROJECTIONS:
            return 0.05
        if projection in _CYLINDRICAL_PROJECTIONS:
            return 0.025
        return 0.009
    if source_case == "tan_low_overlap_exptime_transmission":
        if projection in _ALLSKY_PROJECTIONS:
            return 0.03
        if projection in _CYLINDRICAL_PROJECTIONS:
            return 0.015
        return 0.0055
    if projection in _ALLSKY_PROJECTIONS:
        return 0.025
    if projection in _CYLINDRICAL_PROJECTIONS:
        return 0.012
    return 0.0048


def _projection_target_header(
    projection: str,
    *,
    width: int,
    height: int,
    crval1: float,
    crval2: float,
    scale_deg: float,
) -> dict[str, Any]:
    if projection in _ZENITHAL_PROJECTIONS:
        cd11, cd12, cd21, cd22 = truth_bench._rotated_cd(scale_deg, 0.0)
    else:
        cd11 = -float(scale_deg)
        cd12 = 0.0
        cd21 = 0.0
        cd22 = float(scale_deg)
    header = {
        "NAXIS": 2,
        "NAXIS1": int(width),
        "NAXIS2": int(height),
        "CTYPE1": f"RA---{projection}",
        "CTYPE2": f"DEC--{projection}",
        "CRPIX1": float(width) / 2.0,
        "CRPIX2": float(height) / 2.0,
        "CRVAL1": float(crval1),
        "CRVAL2": float(crval2),
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CD1_1": float(cd11),
        "CD1_2": float(cd12),
        "CD2_1": float(cd21),
        "CD2_2": float(cd22),
    }
    if projection == "CEA":
        header["PV2_1"] = 1.0
    return header


def _common_projection_metadata(
    projection_case: _ProjectionCase,
    observations: list[truth_bench.SimulatedObservation],
    *,
    case_spec: truth_bench._CaseSpec,
    target_header: dict[str, Any],
) -> dict[str, object]:
    return {
        "projection": projection_case.projection,
        "source_case": projection_case.source_case,
        "source_projection": case_spec.projection,
        "description": case_spec.description,
        "value_semantics": case_spec.value_semantics,
        "noise_model": case_spec.noise_model,
        "size": projection_case.size,
        "n_obs": len(observations),
        "target_width": int(target_header["NAXIS1"]),
        "target_height": int(target_header["NAXIS2"]),
        "min_input_width": min(int(item.header["NAXIS1"]) for item in observations),
        "max_input_width": max(int(item.header["NAXIS1"]) for item in observations),
        "min_input_height": min(int(item.header["NAXIS2"]) for item in observations),
        "max_input_height": max(int(item.header["NAXIS2"]) for item in observations),
        "goal": "reconstruct_union_field_on_requested_projection_grid",
        "product_geometry": (
            "rectangular"
            if int(target_header["NAXIS1"]) != int(target_header["NAXIS2"])
            else "square"
        ),
        "n_weight_maps": sum(
            1 for item in observations if item.observation.weight_path is not None
        ),
        "n_mask_maps": sum(
            1 for item in observations if item.observation.mask_path is not None
        ),
    }


def _prepare_projection_truth_case(
    tmpdir: Path,
    *,
    projection_case: _ProjectionCase,
    seed: int,
) -> tuple[
    list[truth_bench.SimulatedObservation],
    dict[str, Any],
    torch.Tensor,
    truth_bench._CaseSpec,
    dict[str, torch.Tensor],
]:
    observations, default_target_header, _default_truth = truth_bench.make_case(
        tmpdir,
        case=projection_case.source_case,
        size=projection_case.size,
        seed=seed,
        width=projection_case.width,
        height=projection_case.height,
    )
    case_spec = truth_bench._case_spec(
        projection_case.source_case, size=projection_case.size
    )
    target_width = (
        int(projection_case.width)
        if projection_case.width is not None
        else int(projection_case.size)
    )
    target_height = (
        int(projection_case.height)
        if projection_case.height is not None
        else int(projection_case.size)
    )
    target_header = _projection_target_header(
        projection_case.projection,
        width=target_width,
        height=target_height,
        crval1=float(default_target_header["CRVAL1"]),
        crval2=float(default_target_header["CRVAL2"]),
        scale_deg=_projection_scale_deg(
            projection_case.projection, projection_case.source_case
        ),
    )
    truth = truth_bench.build_truth_image(
        target_header,
        value_semantics=case_spec.value_semantics,
    )
    region_masks = truth_bench._target_mask_region_masks(
        observations, target_header=target_header
    )
    return observations, target_header, truth, case_spec, region_masks


def _run_projection_truth_suite(
    tmpdir: Path,
    *,
    cases: list[_ProjectionCase],
    methods: list[tuple[str, str | None]],
    runs: int,
    torchsky_runs: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_item in cases:
        observations, target_header, truth, case_spec, region_masks = (
            _prepare_projection_truth_case(
                tmpdir,
                projection_case=case_item,
                seed=seed,
            )
        )
        obs_list = [item.observation for item in observations]
        case_label = _case_label(
            case_item.projection,
            case_item.source_case,
            size=case_item.size,
            width=case_item.width,
            height=case_item.height,
        )
        common = _common_projection_metadata(
            case_item,
            observations,
            case_spec=case_spec,
            target_header=target_header,
        )
        for library, variant in methods:
            if library != "torchsky":
                row = {
                    "suite": "projection_truth",
                    "case": case_label,
                    "library": library,
                    "variant": variant or "default",
                    "status": "SKIPPED",
                    "reason": "Projection Truth keeps the synthetic truth section focused on torchsky kernels only",
                    **common,
                }
                rows.append(row)
                continue
            product = plan_map_product(
                obs_list,
                header_overrides=target_header,
                value_semantics=case_spec.value_semantics,
            )
            kernel = variant or "bilinear"
            dt = truth_bench._time_many(
                lambda: stack_map_product(
                    obs_list,
                    product,
                    resampling_kernel=kernel,
                ),
                max(runs, torchsky_runs),
            )
            result = stack_map_product(
                obs_list,
                product,
                resampling_kernel=kernel,
            )
            row = {
                "suite": "projection_truth",
                "case": case_label,
                "library": "torchsky",
                "method": "coadd",
                "variant": kernel,
                "status": "OK",
                "time_ms": dt * 1000.0,
                "mpix_s": (float(result.data.numel()) / dt) / 1.0e6,
                "stacking_method": "weighted_mean",
                "interpolation_kernel": kernel,
                **common,
            }
            row.update(
                truth_bench._truth_metrics(
                    result.data.squeeze(0).to(dtype=torch.float32),
                    result.weight.squeeze(0).to(dtype=torch.float32),
                    truth,
                    target_header=target_header,
                    value_semantics=case_spec.value_semantics,
                    region_masks={
                        **region_masks,
                        **_support_region_masks(
                            result.weight.squeeze(0).to(dtype=torch.float32)
                        ),
                    },
                )
            )
            rows.append(row)
    return rows


def _prepare_equal_psf_control_case(
    tmpdir: Path,
    *,
    projection_case: _ProjectionCase,
    seed: int,
) -> tuple[
    list[truth_bench.SimulatedObservation],
    dict[str, Any],
    torch.Tensor,
    truth_bench._CaseSpec,
    float,
]:
    tmpdir.mkdir(parents=True, exist_ok=True)
    case_spec = truth_bench._case_spec(
        projection_case.source_case, size=projection_case.size
    )
    psf_values = [float(psf_sigma) for _, _, _, psf_sigma, _ in case_spec.specs]
    control_psf_sigma_pix = float(
        torch.median(torch.tensor(psf_values, dtype=torch.float32)).item()
    )
    observations: list[truth_bench.SimulatedObservation] = []
    for idx, (header, exptime, transmission, _psf_sigma, gap) in enumerate(
        case_spec.specs
    ):
        observations.append(
            truth_bench._simulate_observation(
                tmpdir,
                name=f"equal_psf_{projection_case.projection.lower()}_{projection_case.source_case}_{idx}",
                header=header,
                exposure_time=exptime,
                transmission=transmission,
                psf_sigma_pix=control_psf_sigma_pix,
                base_noise_flux=case_spec.base_noise_flux,
                seed=seed + idx,
                value_semantics=case_spec.value_semantics,
                noise_model=case_spec.noise_model,
                flux_area_norm_sr=None,
                gap=gap,
                explicit_mask=case_spec.explicit_masks,
                weight_gradient=case_spec.weight_gradient,
            )
        )
    target_width = (
        int(projection_case.width)
        if projection_case.width is not None
        else int(projection_case.size)
    )
    target_height = (
        int(projection_case.height)
        if projection_case.height is not None
        else int(projection_case.size)
    )
    target_header = _projection_target_header(
        projection_case.projection,
        width=target_width,
        height=target_height,
        crval1=12.0,
        crval2=-3.0,
        scale_deg=_projection_scale_deg(
            projection_case.projection, projection_case.source_case
        ),
    )
    truth = truth_bench.build_truth_image(
        target_header,
        value_semantics=case_spec.value_semantics,
    )
    return observations, target_header, truth, case_spec, control_psf_sigma_pix


def _run_projection_equal_psf_suite(
    tmpdir: Path,
    *,
    cases: list[_ProjectionCase],
    kernels: list[str],
    stacking_methods: list[str],
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_item in cases:
        observations, target_header, truth, case_spec, control_psf_sigma_pix = (
            _prepare_equal_psf_control_case(
                tmpdir
                / f"equal_psf_{case_item.projection.lower()}_{case_item.source_case}_{_extent_label(case_item.width, case_item.height, case_item.size)}",
                projection_case=case_item,
                seed=seed,
            )
        )
        obs_list = [item.observation for item in observations]
        product = plan_map_product(
            obs_list,
            header_overrides=target_header,
            value_semantics=case_spec.value_semantics,
        )
        common = _common_projection_metadata(
            case_item,
            observations,
            case_spec=case_spec,
            target_header=target_header,
        )
        case_label = _case_label(
            case_item.projection,
            case_item.source_case,
            size=case_item.size,
            width=case_item.width,
            height=case_item.height,
        )
        for kernel in kernels:
            for stacking_method in stacking_methods:
                result = stack_map_product(
                    obs_list,
                    product,
                    resampling_kernel=kernel,
                    stacking_method=stacking_method,
                )
                weight = result.weight.squeeze(0).to(dtype=torch.float32)
                row = {
                    "suite": "projection_equal_psf",
                    "case": case_label,
                    "library": "torchsky",
                    "method": "coadd",
                    "variant": kernel,
                    "stacking_method": stacking_method,
                    "status": "OK",
                    "control_psf_sigma_pix": control_psf_sigma_pix,
                    **common,
                }
                row.update(
                    truth_bench._truth_metrics(
                        result.data.squeeze(0).to(dtype=torch.float32),
                        weight,
                        truth,
                        target_header=target_header,
                        value_semantics=case_spec.value_semantics,
                    )
                )
                rows.append(row)
    return rows


def _run_projection_stack_method_suite(
    tmpdir: Path,
    *,
    cases: list[_ProjectionCase],
    kernels: list[str],
    stacking_methods: list[str],
    runs: int,
    torchsky_runs: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_item in cases:
        observations, target_header, truth, case_spec, region_masks = (
            _prepare_projection_truth_case(
                tmpdir,
                projection_case=case_item,
                seed=seed,
            )
        )
        obs_list = [item.observation for item in observations]
        case_label = _case_label(
            case_item.projection,
            case_item.source_case,
            size=case_item.size,
            width=case_item.width,
            height=case_item.height,
        )
        common = _common_projection_metadata(
            case_item,
            observations,
            case_spec=case_spec,
            target_header=target_header,
        )
        product = plan_map_product(
            obs_list,
            header_overrides=target_header,
            value_semantics=case_spec.value_semantics,
        )
        for kernel in kernels:
            for stacking_method in stacking_methods:
                try:
                    dt = truth_bench._time_many(
                        lambda: stack_map_product(
                            obs_list,
                            product,
                            resampling_kernel=kernel,
                            stacking_method=stacking_method,
                        ),
                        max(runs, torchsky_runs),
                    )
                    result = stack_map_product(
                        obs_list,
                        product,
                        resampling_kernel=kernel,
                        stacking_method=stacking_method,
                    )
                    row = {
                        "suite": "projection_stack_methods",
                        "case": case_label,
                        "library": "torchsky",
                        "method": "coadd",
                        "variant": kernel,
                        "stacking_method": stacking_method,
                        "status": "OK",
                        "time_ms": dt * 1000.0,
                        "mpix_s": (float(result.data.numel()) / dt) / 1.0e6,
                        "interpolation_kernel": kernel,
                        **common,
                    }
                    weight = result.weight.squeeze(0).to(dtype=torch.float32)
                    row.update(
                        truth_bench._truth_metrics(
                            result.data.squeeze(0).to(dtype=torch.float32),
                            weight,
                            truth,
                            target_header=target_header,
                            value_semantics=case_spec.value_semantics,
                            region_masks={
                                **region_masks,
                                **_support_region_masks(weight),
                            },
                        )
                    )
                except Exception as exc:
                    row = {
                        "suite": "projection_stack_methods",
                        "case": case_label,
                        "library": "torchsky",
                        "method": "coadd",
                        "variant": kernel,
                        "stacking_method": stacking_method,
                        "status": "SKIPPED",
                        "reason": f"projection stack-method sweep failed: {exc}",
                        **common,
                    }
                rows.append(row)
    return rows


def _run_montage_compat_suite(
    tmpdir: Path,
    *,
    cases: list[_ProjectionCase],
    methods: list[tuple[str, str | None]],
    runs: int,
    torchsky_runs: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_item in cases:
        observations, target_header, truth, case_spec, region_masks = (
            _prepare_projection_truth_case(
                tmpdir,
                projection_case=case_item,
                seed=seed,
            )
        )
        obs_list = [item.observation for item in observations]
        case_label = _case_label(
            case_item.projection,
            case_item.source_case,
            size=case_item.size,
            width=case_item.width,
            height=case_item.height,
        )
        common = _common_projection_metadata(
            case_item,
            observations,
            case_spec=case_spec,
            target_header=target_header,
        )
        torchsky_outputs: dict[
            str, tuple[dict[str, Any], torch.Tensor, torch.Tensor]
        ] = {}
        for library, variant in methods:
            if library == "torchsky":
                kernel = variant or "bilinear"
                product = plan_map_product(
                    obs_list,
                    header_overrides=target_header,
                    value_semantics=case_spec.value_semantics,
                )
                dt = truth_bench._time_many(
                    lambda: stack_map_product(
                        obs_list,
                        product,
                        resampling_kernel=kernel,
                    ),
                    max(runs, torchsky_runs),
                )
                result = stack_map_product(
                    obs_list,
                    product,
                    resampling_kernel=kernel,
                )
                data = result.data.squeeze(0).to(dtype=torch.float32)
                weight = result.weight.squeeze(0).to(dtype=torch.float32)
                row = {
                    "suite": "montage_compat",
                    "case": case_label,
                    "library": "torchsky",
                    "method": "coadd",
                    "variant": kernel,
                    "status": "OK",
                    "time_ms": dt * 1000.0,
                    "mpix_s": (float(result.data.numel()) / dt) / 1.0e6,
                    "stacking_method": "weighted_mean",
                    **common,
                }
                row.update(
                    truth_bench._truth_metrics(
                        data,
                        weight,
                        truth,
                        target_header=target_header,
                        value_semantics=case_spec.value_semantics,
                        region_masks={
                            **region_masks,
                            **_support_region_masks(weight),
                        },
                    )
                )
                rows.append(row)
                torchsky_outputs[kernel] = (row, data, weight)
                continue
            if library != "montage":
                continue
            baseline = torchsky_outputs.get("bilinear")
            if baseline is None:
                baseline_product = plan_map_product(
                    obs_list,
                    header_overrides=target_header,
                    value_semantics=case_spec.value_semantics,
                )
                baseline_result = stack_map_product(
                    obs_list,
                    baseline_product,
                    resampling_kernel="bilinear",
                )
                baseline = (
                    {
                        "library": "torchsky",
                        "variant": "bilinear",
                        "time_ms": 1.0,
                    },
                    baseline_result.data.squeeze(0).to(dtype=torch.float32),
                    baseline_result.weight.squeeze(0).to(dtype=torch.float32),
                )
            row, data, weight = compat_bench._comparator_row(
                "montage",
                case_label,
                obs_list,
                target_header=target_header,
                runs=max(2, runs),
                torchsky_row=baseline[0],
                torchsky_data=baseline[1],
                torchsky_weight=baseline[2],
                variant=None,
            )
            row.update(
                {
                    "suite": "montage_compat",
                    "variant": row.get("variant", "montage"),
                    **common,
                }
            )
            if row.get("status") == "OK" and data is not None and weight is not None:
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
            rows.append(row)
    return rows


def _evaluate_stack_vs_inr_case(
    *,
    tmpdir: Path,
    case_item: _StackVsINRCase,
    model_kind_spec: str,
    seed: int,
    steps: int,
    runs: int,
) -> dict[str, object]:
    model_kind, coord_mode = _parse_inr_model_spec(model_kind_spec)
    case_spec = field_fit_bench._build_case_specs(case_item.size)[case_item.field_case]
    observations, image_observations, _default_truth_cube = (
        field_fit_bench._simulate_observations(
            case_spec,
            seed=seed,
            tmpdir=tmpdir,
        )
    )
    target_width = (
        int(case_item.width) if case_item.width is not None else int(case_item.size)
    )
    target_height = (
        int(case_item.height) if case_item.height is not None else int(case_item.size)
    )
    target_header = _projection_target_header(
        case_item.projection,
        width=target_width,
        height=target_height,
        crval1=float(case_spec.target_header["CRVAL1"]),
        crval2=float(case_spec.target_header["CRVAL2"]),
        scale_deg=_projection_scale_deg(case_item.projection, "tan_multires_noise"),
    )
    truth_model = field_fit_bench._make_truth_model(case_spec, seed=seed)
    truth_cube = render_field_cube(
        truth_model,
        target_header=target_header,
        spectral_axis=case_spec.spectral_axis,
    ).detach()
    common_psf_sigma_pix = max(
        float(observation.psf_sigma_pix) for observation in observations
    )

    def _gaussian_blur_plane(image: torch.Tensor, sigma_pix: float) -> torch.Tensor:
        if sigma_pix <= 0.0:
            return image
        radius = max(1, int(math.ceil(3.0 * float(sigma_pix))))
        coords = torch.arange(
            -radius,
            radius + 1,
            dtype=image.dtype,
            device=image.device,
        )
        kernel = torch.exp(-0.5 * torch.square(coords / float(sigma_pix)))
        kernel = kernel / torch.clamp(kernel.sum(), min=1.0e-8)
        out = image.unsqueeze(0).unsqueeze(0)
        out = F.conv2d(out, kernel.view(1, 1, 1, -1), padding=(0, radius))
        out = F.conv2d(out, kernel.view(1, 1, -1, 1), padding=(radius, 0))
        return out.squeeze(0).squeeze(0)

    def _gaussian_blur_cube_local(cube: torch.Tensor, sigma_pix: float) -> torch.Tensor:
        if sigma_pix <= 0.0:
            return cube
        return torch.stack(
            [_gaussian_blur_plane(plane, sigma_pix) for plane in cube],
            dim=0,
        )

    def _homogenize_common_psf(
        observations_in: list[FieldObservation],
        image_observations_in: list[ImageObservation],
    ) -> tuple[list[FieldObservation], list[ImageObservation]]:
        common_field_observations: list[FieldObservation] = []
        common_image_observations: list[ImageObservation] = []
        for idx, (observation, image_observation) in enumerate(
            zip(observations_in, image_observations_in, strict=True)
        ):
            sigma_now = float(observation.psf_sigma_pix)
            sigma_extra = math.sqrt(max(common_psf_sigma_pix**2 - sigma_now**2, 0.0))
            image = observation.image.to(dtype=torch.float32)
            if sigma_extra > 0.0:
                image = _gaussian_blur_plane(image, sigma_extra)
            common_field_observations.append(
                replace(
                    observation,
                    image=image.to(dtype=observation.image.dtype),
                    psf_sigma_pix=common_psf_sigma_pix,
                )
            )
            image_path = tmpdir / f"common_psf_{idx}.fits"
            torchsky.write(
                str(image_path),
                image.to(dtype=torch.float32),
                header=observation.header,
                overwrite=True,
            )
            common_image_observations.append(
                replace(
                    image_observation,
                    path=str(image_path),
                )
            )
        return common_field_observations, common_image_observations

    observations, image_observations = _homogenize_common_psf(
        observations,
        image_observations,
    )
    truth_cube = _gaussian_blur_cube_local(
        truth_cube.to(dtype=torch.float32),
        common_psf_sigma_pix,
    )
    product = plan_map_product(
        image_observations,
        header_overrides=target_header,
        spectral_axis=case_spec.spectral_axis,
        spectral_bin_edges=case_spec.spectral_bin_edges,
    )
    if model_kind.startswith("coadd_"):
        kernel = model_kind.split("_", 1)[1]
        fit_times: list[float] = []
        product_pred = None
        stack_map_product(
            image_observations,
            product,
            resampling_kernel=kernel,
        )
        for _ in range(int(runs)):
            start = time.perf_counter()
            product_pred = stack_map_product(
                image_observations,
                product,
                resampling_kernel=kernel,
            )
            fit_times.append((time.perf_counter() - start) * 1000.0)
        assert product_pred is not None
        result_type = "direct_stack"
    else:
        domain = estimate_field_domain(
            observations,
            coord_mode="local_tangent" if coord_mode is None else coord_mode,
        )
        fit_kwargs = _field_fit_kwargs_for_model(model_kind)
        fit_times = []
        result = None
        warmup_model = make_field_model(model_kind, domain)
        fit_field_model(
            observations,
            warmup_model,
            steps=steps,
            **fit_kwargs,
        )
        for run_idx in range(int(runs)):
            torch.manual_seed(int(seed) + 1000 + run_idx)
            model = make_field_model(model_kind, domain)
            start = time.perf_counter()
            result = fit_field_model(
                observations,
                model,
                steps=steps,
                **fit_kwargs,
            )
            fit_times.append((time.perf_counter() - start) * 1000.0)
        assert result is not None
        product_pred = render_field_product(result.model, product)
        product_pred = replace(
            product_pred,
            data=_gaussian_blur_cube_local(
                product_pred.data.to(dtype=torch.float32),
                common_psf_sigma_pix,
            ),
        )
        result_type = "field_fit"
    metrics = evaluate_map_product(product_pred, truth_cube)
    fit_times_sorted = sorted(float(value) for value in fit_times)
    fit_time_ms = fit_times_sorted[len(fit_times_sorted) // 2]
    return {
        "suite": "stack_vs_inr",
        "case": _case_label(
            case_item.projection,
            case_item.field_case,
            size=case_item.size,
            width=case_item.width,
            height=case_item.height,
        ),
        "projection": case_item.projection,
        "field_case": case_item.field_case,
        "status": "OK",
        "library": "torchsky",
        "type": result_type,
        "model_kind": model_kind,
        "field_coord_mode": coord_mode if coord_mode is not None else "n/a",
        "fit_time_ms": fit_time_ms,
        "num_bands": int(case_spec.spectral_axis.values.numel()),
        "n_obs": len(observations),
        "target_width": int(target_header["NAXIS1"]),
        "target_height": int(target_header["NAXIS2"]),
        "product_comparison": "same_requested_final_grid",
        "common_psf_sigma_pix": float(common_psf_sigma_pix),
        "observation_psf_mode": "implicit_shared",
        "evaluation_psf_mode": "common_shared",
        "fit_schedule": "pure_observation_fit"
        if result_type == "field_fit"
        else "direct_stack",
        "fit_loss_kind": (
            fit_kwargs["loss_kind"] if result_type == "field_fit" else "n/a"
        ),
        "fit_lr": fit_kwargs["lr"] if result_type == "field_fit" else None,
        "fit_lr_schedule": (
            fit_kwargs["lr_schedule"] if result_type == "field_fit" else "n/a"
        ),
        "fit_grad_clip_norm": (
            fit_kwargs["grad_clip_norm"] if result_type == "field_fit" else None
        ),
        "truth_model": case_spec.truth_model,
        "source_projection": case_spec.projection,
        **metrics,
    }


def _run_stack_vs_inr_suite(
    tmpdir: Path,
    *,
    cases: list[_StackVsINRCase],
    models: list[str],
    steps: int,
    runs: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, case_item in enumerate(cases):
        case_tmpdir = tmpdir / f"stack_vs_inr_{idx}"
        case_tmpdir.mkdir(parents=True, exist_ok=True)
        for model_kind in models:
            parsed_model_kind, parsed_coord_mode = _parse_inr_model_spec(model_kind)
            try:
                row = _evaluate_stack_vs_inr_case(
                    tmpdir=case_tmpdir,
                    case_item=case_item,
                    model_kind_spec=model_kind,
                    seed=seed,
                    steps=steps,
                    runs=runs,
                )
            except Exception as exc:
                row = {
                    "suite": "stack_vs_inr",
                    "case": _case_label(
                        case_item.projection,
                        case_item.field_case,
                        size=case_item.size,
                        width=case_item.width,
                        height=case_item.height,
                    ),
                    "projection": case_item.projection,
                    "field_case": case_item.field_case,
                    "library": "torchsky",
                    "type": "field_fit"
                    if not model_kind.startswith("coadd_")
                    else "direct_stack",
                    "model_kind": parsed_model_kind,
                    "field_coord_mode": parsed_coord_mode
                    if parsed_coord_mode is not None
                    else "n/a",
                    "status": "SKIPPED",
                    "reason": f"stack-vs-INR evaluation failed: {exc}",
                }
            rows.append(row)
    return rows


def _annotate_stack_vs_inr_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("suite") != "stack_vs_inr":
            continue
        grouped.setdefault(
            (str(row.get("case")), str(row.get("projection"))),
            [],
        ).append(row)

    for case_rows in grouped.values():
        ok_rows = [row for row in case_rows if row.get("status") == "OK"]
        direct_rows = [row for row in ok_rows if row.get("type") == "direct_stack"]
        inr_rows = [row for row in ok_rows if row.get("type") == "field_fit"]
        best_direct = min(
            direct_rows,
            key=lambda row: float(row.get("reconstruction_rmse", float("inf"))),
            default=None,
        )
        best_inr = min(
            inr_rows,
            key=lambda row: float(row.get("reconstruction_rmse", float("inf"))),
            default=None,
        )
        best_direct_rmse = (
            float(best_direct.get("reconstruction_rmse", float("inf")))
            if best_direct is not None
            else None
        )
        best_direct_time_ms = (
            float(best_direct.get("fit_time_ms", float("inf")))
            if best_direct is not None
            else None
        )
        best_inr_rmse = (
            float(best_inr.get("reconstruction_rmse", float("inf")))
            if best_inr is not None
            else None
        )
        for row in case_rows:
            row["best_direct_model_kind"] = (
                str(best_direct.get("model_kind")) if best_direct is not None else None
            )
            row["best_direct_reconstruction_rmse"] = best_direct_rmse
            row["best_inr_model_kind"] = (
                str(best_inr.get("model_kind")) if best_inr is not None else None
            )
            row["best_inr_reconstruction_rmse"] = best_inr_rmse
            if row.get("status") == "OK":
                row_rmse = float(row.get("reconstruction_rmse", float("inf")))
                row["rmse_delta_vs_best_direct"] = (
                    row_rmse - best_direct_rmse
                    if best_direct_rmse is not None and math.isfinite(best_direct_rmse)
                    else None
                )
                row["rmse_delta_vs_best_inr"] = (
                    row_rmse - best_inr_rmse
                    if best_inr_rmse is not None and math.isfinite(best_inr_rmse)
                    else None
                )
                row_time_ms = float(row.get("fit_time_ms", float("inf")))
                row["fit_time_ratio_vs_best_direct"] = (
                    row_time_ms / max(best_direct_time_ms, 1e-12)
                    if best_direct_time_ms is not None
                    and math.isfinite(best_direct_time_ms)
                    else None
                )
            else:
                row["rmse_delta_vs_best_direct"] = None
                row["rmse_delta_vs_best_inr"] = None
                row["fit_time_ratio_vs_best_direct"] = None
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(_PROFILES),
        default="quick",
        help="Benchmark workload profile: quick routine scoreboard or full expansion run",
    )
    parser.add_argument(
        "--truth-cases",
        default=None,
        help="Comma-separated PROJ:CASE@SIZE or PROJ:CASE@WIDTHxHEIGHT list for the Projection Truth section",
    )
    parser.add_argument(
        "--equal-psf-cases",
        default=None,
        help="Comma-separated PROJ:CASE@SIZE or PROJ:CASE@WIDTHxHEIGHT list for the Projection Equal-PSF Controls section",
    )
    parser.add_argument(
        "--method-sweep-cases",
        default=None,
        help="Comma-separated PROJ:CASE@SIZE or PROJ:CASE@WIDTHxHEIGHT list for the Projection Stack Methods section",
    )
    parser.add_argument(
        "--montage-cases",
        default=None,
        help="Comma-separated PROJ:CASE@SIZE or PROJ:CASE@WIDTHxHEIGHT list for the Montage Compatibility section",
    )
    parser.add_argument(
        "--stack-vs-inr-cases",
        default=None,
        help="Comma-separated PROJ:CASE@SIZE or PROJ:CASE@WIDTHxHEIGHT list for the Stack vs INR section",
    )
    parser.add_argument(
        "--truth-methods",
        default=None,
        help="Comma-separated library[:variant] list for the Projection Truth section",
    )
    parser.add_argument(
        "--method-sweep-kernels",
        default=None,
        help="Comma-separated torchsky kernels for the Projection Stack Methods section",
    )
    parser.add_argument(
        "--method-sweep-methods",
        default=None,
        help="Comma-separated stacking methods for the Projection Stack Methods section",
    )
    parser.add_argument(
        "--montage-methods",
        default=None,
        help="Comma-separated library[:variant] list for the Montage Compatibility section",
    )
    parser.add_argument(
        "--stack-vs-inr-models",
        default=None,
        help="Comma-separated model kinds for Stack vs INR rows",
    )
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--torchsky-runs", type=int, default=None)
    parser.add_argument("--field-steps", type=int, default=None)
    parser.add_argument("--field-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/map_projection_mosaic_suite.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("benchmark_results/map_projection_mosaic_suite.md"),
    )
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)
    truth_cases = (
        _parse_projection_case_specs(args.truth_cases)
        if args.truth_cases is not None
        else _parse_projection_case_specs(",".join(profile.truth_cases))
    )
    equal_psf_cases = (
        _parse_projection_case_specs(args.equal_psf_cases)
        if args.equal_psf_cases is not None
        else _parse_projection_case_specs(",".join(profile.equal_psf_cases))
    )
    method_sweep_cases = (
        _parse_projection_case_specs(args.method_sweep_cases)
        if args.method_sweep_cases is not None
        else _parse_projection_case_specs(",".join(profile.method_sweep_cases))
    )
    montage_cases = (
        _parse_projection_case_specs(args.montage_cases)
        if args.montage_cases is not None
        else _parse_projection_case_specs(",".join(profile.montage_cases))
    )
    stack_vs_inr_cases = (
        _parse_stack_vs_inr_case_specs(args.stack_vs_inr_cases)
        if args.stack_vs_inr_cases is not None
        else _parse_stack_vs_inr_case_specs(",".join(profile.stack_vs_inr_cases))
    )
    truth_methods = (
        _parse_method_specs(args.truth_methods)
        if args.truth_methods is not None
        else _parse_method_specs(",".join(profile.truth_methods))
    )
    method_sweep_kernels = (
        _parse_name_list(args.method_sweep_kernels, label="method-sweep kernel")
        if args.method_sweep_kernels is not None
        else list(profile.method_sweep_kernels)
    )
    method_sweep_methods = (
        _parse_name_list(
            args.method_sweep_methods, label="method-sweep stacking method"
        )
        if args.method_sweep_methods is not None
        else list(profile.method_sweep_methods)
    )
    montage_methods = (
        _parse_method_specs(args.montage_methods)
        if args.montage_methods is not None
        else _parse_method_specs(",".join(profile.montage_methods))
    )
    stack_vs_inr_models = [
        item.strip()
        for item in (
            args.stack_vs_inr_models
            if args.stack_vs_inr_models is not None
            else ",".join(profile.stack_vs_inr_models)
        ).split(",")
        if item.strip()
    ]
    if not stack_vs_inr_models:
        raise ValueError("at least one stack-vs-INR model is required")
    runs = profile.runs if args.runs is None else int(args.runs)
    torchsky_runs = (
        profile.torchsky_runs if args.torchsky_runs is None else int(args.torchsky_runs)
    )
    field_steps = (
        profile.field_steps if args.field_steps is None else int(args.field_steps)
    )
    field_runs = profile.field_runs if args.field_runs is None else int(args.field_runs)

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="torchsky_projection_mosaic_suite_") as tmp:
        tmpdir = Path(tmp)
        rows.extend(
            _run_projection_truth_suite(
                tmpdir,
                cases=truth_cases,
                methods=truth_methods,
                runs=runs,
                torchsky_runs=torchsky_runs,
                seed=args.seed,
            )
        )
        rows.extend(
            _run_projection_equal_psf_suite(
                tmpdir,
                cases=equal_psf_cases,
                kernels=list(profile.method_sweep_kernels),
                stacking_methods=list(profile.method_sweep_methods),
                seed=args.seed,
            )
        )
        rows.extend(
            _run_projection_stack_method_suite(
                tmpdir,
                cases=method_sweep_cases,
                kernels=method_sweep_kernels,
                stacking_methods=method_sweep_methods,
                runs=runs,
                torchsky_runs=torchsky_runs,
                seed=args.seed,
            )
        )
        rows.extend(
            _run_montage_compat_suite(
                tmpdir,
                cases=montage_cases,
                methods=montage_methods,
                runs=runs,
                torchsky_runs=torchsky_runs,
                seed=args.seed,
            )
        )
        rows.extend(
            _run_stack_vs_inr_suite(
                tmpdir,
                cases=stack_vs_inr_cases,
                models=stack_vs_inr_models,
                steps=field_steps,
                runs=field_runs,
                seed=args.seed,
            )
        )
        rows = _annotate_stack_vs_inr_rows(rows)

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
