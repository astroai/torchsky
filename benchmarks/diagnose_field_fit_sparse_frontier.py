#!/usr/bin/env python3
"""Support-aware sparse-mosaic frontier sweeps for field-fit models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import bench_map_field_fit as field_bench  # noqa: E402


def _parse_csv_ints(text: str) -> list[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return [int(value) for value in values]


def _parse_csv_strings(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| case | size | steps | model | rmse | support_rmse | fit_time_ms | flux_err |\n"
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |\n"
    )
    lines = [
        "| {case} | {size} | {steps} | {model_kind} | {reconstruction_rmse:.6f} | "
        "{support_reconstruction_rmse:.6f} | {fit_time_ms:.3f} | {total_flux_relative_error:.6f} |".format(
            **row
        )
        for row in rows
    ]
    return header + "\n".join(lines) + ("\n" if lines else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=_parse_csv_strings,
        default=[
            "tan_sip_field_fit_sparse_large_mosaic_broad",
            "tpv_field_fit_sparse_large_mosaic_exact",
        ],
        help="Comma-separated sparse field-fit benchmark cases.",
    )
    parser.add_argument(
        "--models",
        type=_parse_csv_strings,
        default=[
            "coadd_bilinear",
            "sky_encoder_mlp",
            "sky_lambda_kplanes_field",
            "sky_lambda_feature_field",
        ],
        help="Comma-separated model kinds.",
    )
    parser.add_argument(
        "--sizes",
        type=_parse_csv_ints,
        default=[64, 128],
        help="Comma-separated target sizes.",
    )
    parser.add_argument(
        "--steps-list",
        type=_parse_csv_ints,
        default=[8, 16, 24],
        help="Comma-separated fit step counts.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for case in args.cases:
        for size in args.sizes:
            for steps in args.steps_list:
                for model_kind in args.models:
                    if model_kind.startswith("coadd_"):
                        eval_steps = 1
                    else:
                        eval_steps = int(steps)
                    row = field_bench.evaluate_case(
                        case=case,
                        model_kind=model_kind,
                        size=int(size),
                        seed=int(args.seed),
                        steps=eval_steps,
                        runs=int(args.runs),
                    )
                    row["size"] = int(size)
                    row["steps"] = eval_steps
                    rows.append(row)
    rows.sort(
        key=lambda row: (
            row["case"],
            int(row["size"]),
            int(row["steps"]),
            float(row["support_reconstruction_rmse"]),
            float(row["fit_time_ms"]),
        )
    )
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.write_text(_markdown_table(rows), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
