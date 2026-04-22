#!/usr/bin/env python3
"""Run selected competitive methods on the large-union ground-truth case."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_ground_truth as truth_bench  # noqa: E402
import bench_map_compat as compat_bench  # noqa: E402


DEFAULT_METHODS = (
    "torchsky:bilinear",
    "reproject_interp:bilinear",
    "dfreproject:bilinear",
    "drizzle:point",
    "montage",
    "swarp:bilinear",
    "casutools",
)


def _effective_case_size(size: int, width: int | None, height: int | None) -> int:
    return int(max(size, width or 0, height or 0))


def _normalize_row(
    row: dict[str, object], *, library: str | None = None
) -> dict[str, object]:
    method = str(row.get("method", library or "unknown"))
    if method == "coadd":
        method = "torchsky"
    normalized = dict(row)
    normalized["method"] = method
    normalized["library"] = method
    normalized["status"] = str(normalized.get("status") or "OK")
    if normalized.get("variant") is None:
        normalized["variant"] = "default"
    return normalized


def _method_key(row: dict[str, object]) -> str:
    variant = row.get("variant")
    if variant is None or variant == "default":
        return str(row["library"])
    return f"{row['library']}:{variant}"


def _format_markdown(rows: list[dict[str, object]]) -> str:
    headers = [
        "library",
        "variant",
        "status",
        "product_geometry",
        "n_warped_inputs",
        "n_weight_maps",
        "n_mask_maps",
        "resampling_method",
        "stacking_method",
        "time_ms",
        "reconstruction_rmse",
        "reconstruction_mae",
        "reconstruction_p95_abs_err",
        "reconstruction_max_abs_err",
        "signal_to_residual_db",
        "total_flux_relative_error",
        "coverage_fraction",
        "zero_support_fraction",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for key in headers:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif value is None:
                values.append("n/a")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _write_outputs(
    rows: list[dict[str, object]],
    *,
    json_out: Path,
    markdown_out: Path,
) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(_format_markdown(rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default="tan_random10_large_union",
        help="Ground-truth case name",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--torchsky-runs",
        type=int,
        default=3,
        help="Median timing runs for torchsky rows; kept separate so heavy competitive cases have a stable torchsky baseline.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated library[:variant] selection list",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/map_ground_truth_competitive_large_union.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("benchmark_results/map_ground_truth_competitive_large_union.md"),
    )
    args = parser.parse_args()
    effective_size = _effective_case_size(args.size, args.width, args.height)

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    out_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix="torchsky_ground_truth_competitive_"
    ) as tmp:
        tmpdir = Path(tmp)
        observations, target_header, truth = truth_bench.make_case(
            tmpdir,
            case=args.case,
            size=effective_size,
            seed=args.seed,
            width=args.width,
            height=args.height,
        )
        case_spec = truth_bench._case_spec(args.case, size=effective_size)
        obs_list = [item.observation for item in observations]
        baseline_data = truth
        baseline_weight = torch.ones_like(truth, dtype=torch.float32)
        for method_spec in methods:
            print(f"RUNNING {method_spec}", flush=True)
            if ":" in method_spec:
                library, variant = method_spec.split(":", 1)
            else:
                library, variant = method_spec, None
            if library == "torchsky":
                row = _normalize_row(
                    truth_bench.evaluate_torchsky_case(
                        tmpdir,
                        case=args.case,
                        size=effective_size,
                        width=args.width,
                        height=args.height,
                        seed=args.seed,
                        runs=max(args.runs, args.torchsky_runs),
                        resampling_kernel=variant or "bilinear",
                    ),
                    library=library,
                )
            else:
                row, data, weight = compat_bench._comparator_row(
                    library,
                    args.case,
                    obs_list,
                    target_header=target_header,
                    runs=max(2, args.runs),
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
                        "projection": case_spec.projection,
                        "description": case_spec.description,
                        "value_semantics": case_spec.value_semantics,
                        "noise_model": case_spec.noise_model,
                        "size": effective_size,
                        "n_obs": len(observations),
                        "target_width": int(target_header["NAXIS1"]),
                        "target_height": int(target_header["NAXIS2"]),
                        "min_input_width": min(
                            int(item.header["NAXIS1"]) for item in observations
                        ),
                        "max_input_width": max(
                            int(item.header["NAXIS1"]) for item in observations
                        ),
                        "min_input_height": min(
                            int(item.header["NAXIS2"]) for item in observations
                        ),
                        "max_input_height": max(
                            int(item.header["NAXIS2"]) for item in observations
                        ),
                        "goal": "reconstruct_union_field_on_target_grid",
                        "product_geometry": (
                            "rectangular"
                            if int(target_header["NAXIS1"])
                            != int(target_header["NAXIS2"])
                            else "square"
                        ),
                        "n_warped_inputs": truth_bench._count_warped_inputs(
                            observations
                        ),
                        "n_weight_maps": sum(
                            1
                            for item in observations
                            if item.observation.weight_path is not None
                        ),
                        "n_mask_maps": sum(
                            1
                            for item in observations
                            if item.observation.mask_path is not None
                        ),
                        "mean_zero_weight_fraction_input": float(
                            sum(item.zero_weight_fraction for item in observations)
                            / max(len(observations), 1)
                        ),
                        "mean_explicit_mask_fraction_input": float(
                            sum(item.explicit_mask_fraction for item in observations)
                            / max(len(observations), 1)
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
                            value_semantics="surface_brightness",
                        )
                    )
                row = _normalize_row(row, library=library)
            out_rows.append(row)
            _write_outputs(
                out_rows, json_out=args.json_out, markdown_out=args.markdown_out
            )

    _write_outputs(out_rows, json_out=args.json_out, markdown_out=args.markdown_out)
    print(args.json_out)
    print(args.markdown_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
