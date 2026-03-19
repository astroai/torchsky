#!/usr/bin/env python3
"""Diagnose residual structure for exact-band field-fit benchmarks."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_field_fit as field_bench  # noqa: E402
from torchsky.map import (  # noqa: E402
    SkyEncoderMLPField,
    SpectralBasisField,
    estimate_field_domain,
    fit_field_model,
    plan_map_product,
    render_field_observation,
    render_field_product,
    stack_map_product,
)


def _masked_rmse(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(torch.sqrt(torch.mean(torch.square(values[mask]))).item())


def _masked_mae(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(torch.mean(torch.abs(values[mask])).item())


def _overlap_bins(
    residual: torch.Tensor,
    overlap_count: torch.Tensor,
    support: torch.Tensor,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    max_overlap = int(overlap_count.max().item()) if overlap_count.numel() > 0 else 0
    for level in range(1, max_overlap + 1):
        mask = support & (overlap_count == level)
        if bool(mask.any()):
            out[f"eq_{level}"] = {
                "count": float(mask.sum().item()),
                "rmse": _masked_rmse(residual, mask),
                "mae": _masked_mae(residual, mask),
            }
    for level in range(2, max_overlap + 1):
        mask = support & (overlap_count >= level)
        if bool(mask.any()):
            out[f"ge_{level}"] = {
                "count": float(mask.sum().item()),
                "rmse": _masked_rmse(residual, mask),
                "mae": _masked_mae(residual, mask),
            }
    return out


def _band_summaries(
    *,
    truth_cube: torch.Tensor,
    coadd_cube: torch.Tensor,
    field_cube: torch.Tensor,
    overlap_count: torch.Tensor,
    weight: torch.Tensor,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for band_idx in range(int(truth_cube.shape[0])):
        truth = truth_cube[band_idx].to(dtype=torch.float32)
        coadd = coadd_cube[band_idx].to(dtype=torch.float32)
        field = field_cube[band_idx].to(dtype=torch.float32)
        overlap = overlap_count[band_idx].to(dtype=torch.int32)
        support = weight[band_idx] > 0.0
        coadd_resid = coadd - truth
        field_resid = field - truth
        improvement = torch.where(
            support,
            torch.abs(field_resid) - torch.abs(coadd_resid),
            torch.full_like(coadd_resid, float("-inf")),
        )
        degrade = torch.where(
            support,
            torch.abs(coadd_resid) - torch.abs(field_resid),
            torch.full_like(coadd_resid, float("-inf")),
        )
        worst_field_idx = int(torch.argmax(improvement).item())
        best_field_idx = int(torch.argmax(degrade).item())
        width = int(truth.shape[1])
        fy, fx = divmod(worst_field_idx, width)
        by, bx = divmod(best_field_idx, width)
        summaries.append(
            {
                "band_index": band_idx,
                "coadd_rmse": _masked_rmse(coadd_resid, support),
                "field_rmse": _masked_rmse(field_resid, support),
                "coadd_mae": _masked_mae(coadd_resid, support),
                "field_mae": _masked_mae(field_resid, support),
                "coadd_by_overlap": _overlap_bins(coadd_resid, overlap, support),
                "field_by_overlap": _overlap_bins(field_resid, overlap, support),
                "worst_field_pixel": {
                    "x": fx,
                    "y": fy,
                    "overlap_count": int(overlap[fy, fx].item()),
                    "truth": float(truth[fy, fx].item()),
                    "coadd": float(coadd[fy, fx].item()),
                    "field": float(field[fy, fx].item()),
                    "coadd_abs_residual": float(torch.abs(coadd_resid[fy, fx]).item()),
                    "field_abs_residual": float(torch.abs(field_resid[fy, fx]).item()),
                },
                "best_field_pixel": {
                    "x": bx,
                    "y": by,
                    "overlap_count": int(overlap[by, bx].item()),
                    "truth": float(truth[by, bx].item()),
                    "coadd": float(coadd[by, bx].item()),
                    "field": float(field[by, bx].item()),
                    "coadd_abs_residual": float(torch.abs(coadd_resid[by, bx]).item()),
                    "field_abs_residual": float(torch.abs(field_resid[by, bx]).item()),
                },
            }
        )
    return summaries


def _observation_summaries(
    *,
    observations: list[Any],
    model: torch.nn.Module,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, observation in enumerate(observations):
        pred = render_field_observation(model, observation, num_spectral_samples=9)
        residual = pred - observation.image.to(device=pred.device, dtype=pred.dtype)
        weight = (
            observation.weight.to(device=pred.device, dtype=pred.dtype)
            if observation.weight is not None
            else torch.ones_like(pred)
        )
        if observation.mask is not None:
            support = observation.mask.to(device=pred.device) > 0
        else:
            support = weight > 0.0
        out.append(
            {
                "observation_index": idx,
                "band_name": observation.band_name,
                "wavelength": observation.wavelength,
                "spectral_min": observation.spectral_min,
                "spectral_max": observation.spectral_max,
                "rmse": _masked_rmse(residual, support),
                "mae": _masked_mae(residual, support),
                "mean_weight": float(weight[support].mean().item()) if bool(support.any()) else 0.0,
            }
        )
    return out


def _format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Field-Fit Residual Diagnostic",
        "",
        f"Case: `{summary['case']}`",
        "",
        f"Field model: `{summary['field_config']['model_kind']}`",
        "",
        f"Field config: `{summary['field_config']['model_kwargs']}`, steps `{summary['field_config']['steps']}`",
        "",
        "## Product Metrics",
        "",
        "| model | rmse | support rmse | mae | p95 | flux err |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| coadd_bilinear | {summary['coadd_metrics']['reconstruction_rmse']:.6f} | {summary['coadd_support_metrics']['reconstruction_rmse']:.6f} | {summary['coadd_metrics']['reconstruction_mae']:.6f} | {summary['coadd_metrics']['reconstruction_p95_abs_err']:.6f} | {summary['coadd_metrics']['total_flux_relative_error']:.6f} |",
        f"| {summary['field_config']['model_kind']} | {summary['field_metrics']['reconstruction_rmse']:.6f} | {summary['field_support_metrics']['reconstruction_rmse']:.6f} | {summary['field_metrics']['reconstruction_mae']:.6f} | {summary['field_metrics']['reconstruction_p95_abs_err']:.6f} | {summary['field_metrics']['total_flux_relative_error']:.6f} |",
        "",
        "## Bands",
        "",
    ]
    for band in summary["bands"]:
        lines.extend(
            [
                f"### Band {band['band_index']}",
                "",
                f"- Coadd RMSE/MAE: `{band['coadd_rmse']:.6f}` / `{band['coadd_mae']:.6f}`",
                f"- Field RMSE/MAE: `{band['field_rmse']:.6f}` / `{band['field_mae']:.6f}`",
                f"- Worst field pixel: `{band['worst_field_pixel']}`",
                f"- Best field pixel: `{band['best_field_pixel']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Observation Residuals",
            "",
            "| obs | band | wavelength | rmse | mae | mean_weight |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["observations"]:
        band = row["band_name"] if row["band_name"] is not None else "-"
        wave = row["wavelength"] if row["wavelength"] is not None else row["spectral_min"]
        lines.append(
            f"| {row['observation_index']} | {band} | {float(wave):.1f} | {row['rmse']:.6f} | {row['mae']:.6f} | {row['mean_weight']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("tan_field_fit_exact_transmission", "tpv_field_fit_deep_multiband_exact"),
        required=True,
    )
    parser.add_argument(
        "--model-kind",
        choices=("spectral_basis_field", "sky_encoder_mlp"),
        default="spectral_basis_field",
    )
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--hidden-layers", type=int, default=None)
    parser.add_argument("--spatial-frequencies", type=int, default=None)
    parser.add_argument("--spectral-frequencies", type=int, default=None)
    parser.add_argument("--num-fourier-frequencies", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=Path("/tmp/torchsky_field_fit_residuals.json"))
    parser.add_argument("--markdown-out", type=Path, default=Path("/tmp/torchsky_field_fit_residuals.md"))
    args = parser.parse_args()

    spec = field_bench._build_case_specs(args.size)[args.case]
    with tempfile.TemporaryDirectory(prefix="torchsky-field-fit-resid-") as tmpdir_str:
        observations, image_observations, truth_cube = field_bench._simulate_observations(
            spec,
            seed=args.seed,
            tmpdir=Path(tmpdir_str),
        )
        product = plan_map_product(
            image_observations,
            header_overrides=spec.target_header,
            spectral_axis=spec.spectral_axis,
            spectral_bin_edges=spec.spectral_bin_edges,
        )
        coadd_product = stack_map_product(
            image_observations,
            product,
            resampling_kernel="bilinear",
        )
        domain = estimate_field_domain(observations)
        model_kwargs = field_bench._field_model_kwargs_for_case(args.model_kind, case=args.case)
        if args.hidden_dim is not None:
            model_kwargs["hidden_dim"] = int(args.hidden_dim)
        if args.hidden_layers is not None:
            model_kwargs["hidden_layers"] = int(args.hidden_layers)
        if args.spatial_frequencies is not None:
            model_kwargs["num_spatial_frequencies"] = int(args.spatial_frequencies)
        if args.spectral_frequencies is not None:
            if args.model_kind == "sky_encoder_mlp":
                model_kwargs["spectral_frequencies"] = int(args.spectral_frequencies)
            else:
                model_kwargs["num_spectral_frequencies"] = int(args.spectral_frequencies)
        if args.num_fourier_frequencies is not None:
            model_kwargs["num_fourier_frequencies"] = int(args.num_fourier_frequencies)
        default_model_kwargs = field_bench._field_model_kwargs_for_case(
            args.model_kind,
            case=args.case,
        )
        if model_kwargs == default_model_kwargs:
            model = field_bench._make_field_model_for_case(
                args.model_kind,
                domain,
                case=args.case,
            )
        elif args.model_kind == "spectral_basis_field":
            model = SpectralBasisField(domain, **model_kwargs)
        else:
            model = SkyEncoderMLPField(domain, **model_kwargs)
        fit_kwargs = field_bench._field_fit_kwargs_for_model(args.model_kind, case=args.case)
        torch.manual_seed(int(args.seed) + 999)
        warmup_model = field_bench._make_field_model_for_case(
            args.model_kind,
            domain,
            case=args.case,
        )
        fit_field_model(
            observations,
            warmup_model,
            steps=int(args.steps),
            **fit_kwargs,
        )
        torch.manual_seed(int(args.seed) + 1000)
        fit_field_model(
            observations,
            model,
            steps=int(args.steps),
            **fit_kwargs,
        )
        field_product = render_field_product(model, product)
        coadd_support_metrics = field_bench.evaluate_map_product(
            coadd_product.data,
            truth_cube,
            weight=coadd_product.weight,
            header=coadd_product.header,
            value_semantics=coadd_product.product.value_semantics,
        )
        field_support_metrics = field_bench.evaluate_map_product(
            field_product.data,
            truth_cube,
            weight=coadd_product.weight,
            header=field_product.header,
            value_semantics=field_product.product.value_semantics,
        )
        summary = {
            "case": args.case,
            "field_config": {
                "model_kind": args.model_kind,
                "steps": int(args.steps),
                "model_kwargs": model_kwargs,
            },
            "coadd_metrics": field_bench.evaluate_map_product(coadd_product, truth_cube),
            "coadd_support_metrics": coadd_support_metrics,
            "field_metrics": field_bench.evaluate_map_product(field_product, truth_cube),
            "field_support_metrics": field_support_metrics,
            "bands": _band_summaries(
                truth_cube=truth_cube.to(dtype=torch.float32),
                coadd_cube=coadd_product.data.to(dtype=torch.float32),
                field_cube=field_product.data.to(dtype=torch.float32),
                overlap_count=coadd_product.overlap_count.to(dtype=torch.int32),
                weight=coadd_product.weight.to(dtype=torch.float32),
            ),
            "observations": _observation_summaries(
                observations=observations,
                model=model,
            ),
        }
    args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(_format_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
