#!/usr/bin/env python3
"""Aggregate large-union competitive benchmark runs across target sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "time_ms",
    "reconstruction_rmse",
    "reconstruction_mae",
    "reconstruction_p95_abs_err",
    "reconstruction_max_abs_err",
    "signal_to_residual_db",
    "total_flux_relative_error",
    "coverage_fraction",
    "zero_support_fraction",
)


def _normalize_method(row: dict[str, object]) -> str:
    method = str(row.get("method") or row.get("library") or "unknown")
    if method == "coadd":
        method = "torchsky"
    return method


def _normalize_variant(row: dict[str, object]) -> str:
    variant = row.get("variant")
    if variant is None:
        return "default"
    return str(variant)


def _load_rows(path: Path) -> list[dict[str, object]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path} does not contain a row list")
    rows: list[dict[str, object]] = []
    for row in raw:
        normalized = dict(row)
        normalized["method"] = _normalize_method(normalized)
        normalized["variant"] = _normalize_variant(normalized)
        normalized["status"] = str(normalized.get("status") or "OK")
        rows.append(normalized)
    return rows


def _size_label(path: Path, rows: list[dict[str, object]]) -> str:
    if rows and rows[0].get("target_width") and rows[0].get("target_height"):
        return f"{int(rows[0]['target_width'])}x{int(rows[0]['target_height'])}"
    stem = path.stem
    if stem.endswith("_1024"):
        return "1024x1024"
    if stem.endswith("_2048"):
        return "2048x2048"
    return "512x512"


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _best_row(rows: list[dict[str, object]], metric: str, *, lower_is_better: bool) -> dict[str, object] | None:
    ok_rows = [row for row in rows if row.get("status") == "OK" and isinstance(row.get(metric), (int, float))]
    if not ok_rows:
        return None
    return min(ok_rows, key=lambda row: float(row[metric])) if lower_is_better else max(
        ok_rows,
        key=lambda row: float(row[metric]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            Path("benchmark_results/map_ground_truth_competitive_large_union.json"),
            Path("benchmark_results/map_ground_truth_competitive_large_union_1024.json"),
            Path("benchmark_results/map_ground_truth_competitive_large_union_2048.json"),
        ],
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/map_ground_truth_competitive_large_union_aggregate.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("benchmark_results/map_ground_truth_competitive_large_union_aggregate.md"),
    )
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, object]]] = {}
    size_rows: dict[str, list[dict[str, object]]] = {}
    for path in args.inputs:
        rows = _load_rows(path)
        label = _size_label(path, rows)
        size_rows[label] = rows
        for row in rows:
            key = f"{row['method']}:{row['variant']}"
            grouped.setdefault(key, []).append({"size_label": label, **row})

    aggregate_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        method, variant = key.split(":", 1)
        row_map = {str(entry["size_label"]): entry for entry in grouped[key]}
        aggregate: dict[str, object] = {
            "method": method,
            "variant": variant,
            "sizes": sorted(row_map),
        }
        for size in sorted(row_map):
            entry = row_map[size]
            aggregate[f"status_{size}"] = entry.get("status")
            aggregate[f"reason_{size}"] = entry.get("reason")
            for metric in METRICS:
                aggregate[f"{metric}_{size}"] = entry.get(metric)
        aggregate_rows.append(aggregate)

    summaries: dict[str, dict[str, object]] = {}
    for size, rows in size_rows.items():
        fastest = _best_row(rows, "time_ms", lower_is_better=True)
        best_rmse = _best_row(rows, "reconstruction_rmse", lower_is_better=True)
        best_snr = _best_row(rows, "signal_to_residual_db", lower_is_better=False)
        best_flux = _best_row(rows, "total_flux_relative_error", lower_is_better=True)
        summaries[size] = {
            "fastest": fastest and f"{fastest['method']}:{fastest['variant']}",
            "best_accuracy": best_rmse and f"{best_rmse['method']}:{best_rmse['variant']}",
            "best_snr": best_snr and f"{best_snr['method']}:{best_snr['variant']}",
            "best_flux": best_flux and f"{best_flux['method']}:{best_flux['variant']}",
        }

    payload = {
        "inputs": [str(path) for path in args.inputs],
        "sizes": sorted(size_rows),
        "rows": aggregate_rows,
        "summaries": summaries,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines: list[str] = ["# Large Union Competitive Aggregate", ""]
    md_lines.append("## Per-Size Winners")
    md_lines.append("")
    winner_rows: list[list[object]] = []
    for size in sorted(summaries):
        summary = summaries[size]
        winner_rows.append(
            [
                size,
                summary["fastest"],
                summary["best_accuracy"],
                summary["best_snr"],
                summary["best_flux"],
            ]
        )
    md_lines.append(
        _table(
            ["size", "fastest", "best_accuracy", "best_snr", "best_flux"],
            winner_rows,
        )
    )
    for size in sorted(size_rows):
        rows = size_rows[size]
        md_lines.append(f"## {size}")
        md_lines.append("")
        size_table_rows: list[list[object]] = []
        for row in rows:
            size_table_rows.append(
                [
                    row["method"],
                    row["variant"],
                    row["status"],
                    row.get("product_geometry"),
                    row.get("n_warped_inputs"),
                    row.get("n_weight_maps"),
                    row.get("n_mask_maps"),
                    row.get("resampling_method"),
                    row.get("stacking_method"),
                    row.get("time_ms"),
                    row.get("reconstruction_rmse"),
                    row.get("reconstruction_mae"),
                    row.get("reconstruction_p95_abs_err"),
                    row.get("reconstruction_max_abs_err"),
                    row.get("signal_to_residual_db"),
                    row.get("total_flux_relative_error"),
                    row.get("coverage_fraction"),
                    row.get("zero_support_fraction"),
                    row.get("reason"),
                ]
            )
        md_lines.append(
            _table(
                [
                    "method",
                    "variant",
                    "status",
                    "geometry",
                    "n_warped_inputs",
                    "n_weight_maps",
                    "n_mask_maps",
                    "resampling_method",
                    "stacking_method",
                    "time_ms",
                    "rmse",
                    "mae",
                    "p95_abs_err",
                    "max_abs_err",
                    "signal_to_residual_db",
                    "flux_rel_err",
                    "coverage",
                    "zero_support",
                    "reason",
                ],
                size_table_rows,
            )
        )

    md_lines.append("## Cross-Size Summary")
    md_lines.append("")
    ordered_sizes = sorted(size_rows)
    summary_headers: list[str] = ["method", "variant"]
    for size in ordered_sizes:
        summary_headers.extend(
            [
                f"status_{size}",
                f"time_ms_{size}",
                f"rmse_{size}",
            ]
        )
    summary_rows: list[list[object]] = []
    for row in aggregate_rows:
        out_row: list[object] = [row["method"], row["variant"]]
        for size in ordered_sizes:
            out_row.extend(
                [
                    row.get(f"status_{size}"),
                    row.get(f"time_ms_{size}"),
                    row.get(f"reconstruction_rmse_{size}"),
                ]
            )
        summary_rows.append(out_row)
    md_lines.append(_table(summary_headers, summary_rows))

    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    print(args.json_out)
    print(args.markdown_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
