#!/usr/bin/env python3
"""Diagnose residual tail differences between stack methods on wide-projection cases."""

from __future__ import annotations

import argparse
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

import bench_map_projection_mosaic_suite as projection_suite  # noqa: E402
from torchsky.map import plan_map_product, stack_map_product  # noqa: E402


def _support_mask(weight: torch.Tensor) -> torch.Tensor:
    return weight > 0.0


def _masked_rmse(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(torch.sqrt(torch.mean(values[mask].square())).item())


def _masked_mae(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(torch.mean(values[mask].abs()).item())


def _overlap_map(
    observations: list[Any],
    product: Any,
    *,
    kernel: str,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    overlap = None
    singles: list[tuple[torch.Tensor, torch.Tensor]] = []
    for obs in observations:
        single = stack_map_product(
            [obs],
            product,
            resampling_kernel=kernel,
            stacking_method="weighted_mean",
        )
        data = single.data.squeeze(0).to(dtype=torch.float32)
        weight = single.weight.squeeze(0).to(dtype=torch.float32)
        singles.append((data, weight))
        support = (weight > 0.0).to(dtype=torch.int32)
        overlap = support if overlap is None else overlap + support
    if overlap is None:
        raise RuntimeError("at least one observation is required")
    return overlap, singles


def _pairwise_overlap_bins(
    left_resid: torch.Tensor,
    right_resid: torch.Tensor,
    overlap: torch.Tensor,
    support: torch.Tensor,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    max_overlap = int(overlap.max().item()) if overlap.numel() > 0 else 0
    for level in range(1, max_overlap + 1):
        mask = support & (overlap == level)
        if bool(mask.any()):
            out[f"eq_{level}"] = {
                "count": float(mask.sum().item()),
                "left_rmse": _masked_rmse(left_resid, mask),
                "right_rmse": _masked_rmse(right_resid, mask),
                "left_mae": _masked_mae(left_resid, mask),
                "right_mae": _masked_mae(right_resid, mask),
            }
    return out


def _worst_pixel_summary(
    *,
    truth: torch.Tensor,
    left_data: torch.Tensor,
    right_data: torch.Tensor,
    left_resid_abs: torch.Tensor,
    right_resid_abs: torch.Tensor,
    support: torch.Tensor,
    overlap: torch.Tensor,
    singles: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    improvement = torch.where(
        support,
        left_resid_abs - right_resid_abs,
        torch.full_like(left_resid_abs, float("-inf")),
    )
    flat_idx = int(torch.argmax(improvement).item())
    y = flat_idx // int(improvement.shape[1])
    x = flat_idx % int(improvement.shape[1])
    contributors: list[dict[str, float | int]] = []
    for idx, (data, weight) in enumerate(singles):
        if weight[y, x] > 0.0:
            contributors.append(
                {
                    "observation_index": idx,
                    "value": float(data[y, x].item()),
                    "weight": float(weight[y, x].item()),
                }
            )
    return {
        "x": x,
        "y": y,
        "overlap_count": int(overlap[y, x].item()),
        "truth": float(truth[y, x].item()),
        "left_value": float(left_data[y, x].item()),
        "right_value": float(right_data[y, x].item()),
        "left_abs_residual": float(left_resid_abs[y, x].item()),
        "right_abs_residual": float(right_resid_abs[y, x].item()),
        "improvement": float(improvement[y, x].item()),
        "contributors": contributors,
    }


def _prepare_case(
    tmpdir: Path,
    *,
    case_spec: projection_suite._ProjectionCase,
    psf_mode: str,
    seed: int,
) -> tuple[list[Any], dict[str, Any], torch.Tensor, Any]:
    if psf_mode == "equalized":
        observations, target_header, truth, case_meta, _control_psf = (
            projection_suite._prepare_equal_psf_control_case(
                tmpdir,
                projection_case=case_spec,
                seed=seed,
            )
        )
        return observations, target_header, truth, case_meta
    observations, target_header, truth, case_meta, _region_masks = (
        projection_suite._prepare_projection_truth_case(
            tmpdir,
            projection_case=case_spec,
            seed=seed,
        )
    )
    return observations, target_header, truth, case_meta


def _format_markdown(summary: dict[str, Any]) -> str:
    methods = summary["methods"]
    overlap_bins = summary["comparison"]["overlap_bins"]
    worst = summary["comparison"]["worst_pixel"]
    lines = [
        "# Projection Tail Diagnostic",
        "",
        f"Case: `{summary['case']}`",
        "",
        f"PSF mode: `{summary['psf_mode']}`",
        "",
        f"Kernel: `{summary['kernel']}`",
        "",
        f"Comparison: `{summary['left_method']}` vs `{summary['right_method']}`",
        "",
        "## Methods",
        "",
        "| method | rmse_vs_truth | mae_vs_truth | coverage_frac |",
        "| --- | --- | --- | --- |",
    ]
    for method, metrics in methods.items():
        lines.append(
            f"| {method} | {metrics['rmse_vs_truth']:.8f} | {metrics['mae_vs_truth']:.8f} | {metrics['coverage_frac']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Overlap Bins",
            "",
            "| overlap | count | left_rmse | right_rmse | left_mae | right_mae |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name, stats in overlap_bins.items():
        lines.append(
            f"| {name} | {stats['count']:.0f} | {stats['left_rmse']:.8f} | {stats['right_rmse']:.8f} | {stats['left_mae']:.8f} | {stats['right_mae']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Worst Pixel",
            "",
            f"- Pixel: `({worst['x']}, {worst['y']})`, overlap `{worst['overlap_count']}`",
            f"- Truth: `{worst['truth']:.8f}`",
            f"- Left/right: `{worst['left_value']:.8f}` / `{worst['right_value']:.8f}`",
            f"- Abs residuals: `{worst['left_abs_residual']:.8f}` / `{worst['right_abs_residual']:.8f}`",
            f"- Improvement: `{worst['improvement']:.8f}`",
            "",
            "Contributors:",
        ]
    )
    for item in worst["contributors"]:
        lines.append(
            f"- obs `{item['observation_index']}`: value `{item['value']:.8f}`, weight `{item['weight']:.8f}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", required=True, help="PROJ:CASE@SIZE or PROJ:CASE@WIDTHxHEIGHT"
    )
    parser.add_argument(
        "--psf-mode",
        choices=("heterogeneous", "equalized"),
        default="heterogeneous",
        help="Use the original heterogeneous observations or an equal-PSF control rebuild",
    )
    parser.add_argument("--kernel", default="bilinear")
    parser.add_argument("--methods", default="weighted_mean,mean,median")
    parser.add_argument("--left-method", default="mean")
    parser.add_argument("--right-method", default="median")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("/tmp/torchsky_projection_tail_diag.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("/tmp/torchsky_projection_tail_diag.md"),
    )
    args = parser.parse_args()

    case_item = projection_suite._parse_projection_case_specs(args.case)[0]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if args.left_method not in methods or args.right_method not in methods:
        raise ValueError("left/right methods must be included in --methods")

    with tempfile.TemporaryDirectory(prefix="torchsky_projection_tail_diag_") as tmp:
        tmpdir = Path(tmp)
        observations, target_header, truth, case_meta = _prepare_case(
            tmpdir,
            case_spec=case_item,
            psf_mode=args.psf_mode,
            seed=args.seed,
        )
        obs_list = [item.observation for item in observations]
        product = plan_map_product(
            obs_list,
            header_overrides=target_header,
            value_semantics=case_meta.value_semantics,
        )
        overlap, singles = _overlap_map(obs_list, product, kernel=args.kernel)
        results: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        method_metrics: dict[str, dict[str, float]] = {}
        for method in methods:
            result = stack_map_product(
                obs_list,
                product,
                resampling_kernel=args.kernel,
                stacking_method=method,
            )
            data = result.data.squeeze(0).to(dtype=torch.float32)
            weight = result.weight.squeeze(0).to(dtype=torch.float32)
            resid = data - truth
            mask = _support_mask(weight)
            results[method] = (data, weight, resid)
            method_metrics[method] = {
                "rmse_vs_truth": _masked_rmse(resid, mask),
                "mae_vs_truth": _masked_mae(resid, mask),
                "coverage_frac": float(mask.to(dtype=torch.float32).mean().item()),
            }

        left_data, left_weight, left_resid = results[args.left_method]
        right_data, right_weight, right_resid = results[args.right_method]
        support = _support_mask(left_weight) & _support_mask(right_weight)
        summary = {
            "case": args.case,
            "psf_mode": args.psf_mode,
            "kernel": args.kernel,
            "left_method": args.left_method,
            "right_method": args.right_method,
            "methods": method_metrics,
            "comparison": {
                "right_better_fraction": (
                    float(
                        ((right_resid.abs() + 1e-8) < left_resid.abs())[support]
                        .to(dtype=torch.float32)
                        .mean()
                        .item()
                    )
                    if bool(support.any())
                    else 0.0
                ),
                "overlap_bins": _pairwise_overlap_bins(
                    left_resid.abs(),
                    right_resid.abs(),
                    overlap,
                    support,
                ),
                "worst_pixel": _worst_pixel_summary(
                    truth=truth,
                    left_data=left_data,
                    right_data=right_data,
                    left_resid_abs=left_resid.abs(),
                    right_resid_abs=right_resid.abs(),
                    support=support,
                    overlap=overlap,
                    singles=singles,
                ),
            },
        }
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        args.markdown_out.write_text(_format_markdown(summary), encoding="utf-8")
        print(args.markdown_out.read_text(encoding="utf-8"))
        print(f"JSON: {args.json_out}")
        print(f"Markdown: {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
