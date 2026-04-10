#!/usr/bin/env python3
"""Sequential real-data non-INR stacker sweep across multiple SMACS J0723 bands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from bench_map_smacs0723_real import (  # noqa: E402
    DATA,
    _band_spec,
    _discover_exposure_paths,
    _reference_path_for_band,
)

DEFAULT_JSON_OUT = DATA / "benchmark_results" / "smacs0723_stackers_band_sweep.json"
DEFAULT_MARKDOWN_OUT = DATA / "benchmark_results" / "smacs0723_stackers_band_sweep.md"
from bench_map_smacs0723_stackers import _build_method_specs, run_stacker_benchmark  # noqa: E402


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _format_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SMACS J0723 Multiband Stacker Sweep",
        "",
        "| band | variant | family | status | n exp | support rmse | uniform rmse | flux ratio | time ms |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["status"] != "OK":
            lines.append(
                f"| {row['band']} | {row['variant']} | {row['family']} | {row['status']} | "
                f"{row.get('n_exposures', 0)} | nan | nan | nan | nan |"
            )
            continue
        lines.append(
            f"| {row['band']} | {row['variant']} | {row['family']} | OK | "
            f"{row['n_exposures']} | {row['support_rmse']:.6f} | {row['uniform_rmse']:.6f} | "
            f"{row['support_flux_ratio']:.4f} | {row['time_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _recommended_filename_contains(band: str, module: str) -> str:
    band_upper = str(band).upper()
    module_lower = str(module).lower()
    if band_upper in {"F277W", "F356W", "F444W"} and module_lower in {"a", "b"}:
        return f"nrc{module_lower}long"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", default="F090W,F150W,F200W,F277W,F356W,F444W")
    parser.add_argument("--module", default="b")
    parser.add_argument("--input-product", choices=("auto", "crf", "cal"), default="cal")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--crop-mode", choices=("center", "best_overlap"), default="best_overlap")
    parser.add_argument("--overlap-step", type=int, default=8)
    parser.add_argument("--linearize-sip", action="store_true")
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated stacker method names; defaults to all non-INR methods",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip bands that do not yet have both a local reference product and local exposures",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    args = parser.parse_args()

    bands = [band.upper() for band in _parse_csv(args.bands)]
    for band in bands:
        _band_spec(band)

    specs = _build_method_specs(include_inr=False)
    if args.methods.strip():
        requested = {part.strip() for part in args.methods.split(",") if part.strip()}
        specs = [spec for spec in specs if spec["name"] in requested]
        if not specs:
            raise ValueError("none of the requested methods matched the available non-INR specs")

    rows: list[dict[str, Any]] = []
    for band in bands:
        reference_path = _reference_path_for_band(band)
        filename_contains = _recommended_filename_contains(band, args.module)
        exposure_paths = _discover_exposure_paths(
            DATA / "jwst_smacs0723",
            input_product=args.input_product,
            band=band,
            module=args.module,
            filename_contains=filename_contains or None,
        )
        if not reference_path.exists() or not exposure_paths:
            if args.skip_missing:
                continue
            rows.append(
                {
                    "band": band,
                    "module": args.module,
                    "variant": "inventory",
                    "family": "meta",
                    "status": "MISSING",
                    "message": (
                        f"reference_exists={reference_path.exists()} n_exposures={len(exposure_paths)}"
                    ),
                    "n_exposures": len(exposure_paths),
                    "reference_path": str(reference_path),
                }
            )
            continue
        band_rows = run_stacker_benchmark(
            band=band,
            reference_path=reference_path,
            exposure_paths=exposure_paths,
            specs=specs,
            crop_size=args.crop_size,
            crop_mode=args.crop_mode,
            overlap_step=args.overlap_step,
            input_product=args.input_product,
            filename_contains=filename_contains,
            linearize_sip=args.linearize_sip,
            inr_steps=0,
        )
        for row in band_rows:
            row["band"] = band
            row["module"] = args.module
            row["reference_path"] = str(reference_path)
        rows.extend(band_rows)

    ok_rows = [row for row in rows if row["status"] == "OK"]
    rest_rows = [row for row in rows if row["status"] != "OK"]
    ok_rows.sort(key=lambda row: (str(row.get("band", "")), float(row["support_rmse"])))
    rest_rows.sort(key=lambda row: (str(row.get("band", "")), str(row.get("variant", ""))))
    rows = ok_rows + rest_rows
    payload = {"rows": rows}
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_format_markdown(rows), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
