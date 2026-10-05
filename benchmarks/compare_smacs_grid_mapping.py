"""Compare a matched legacy-vs-footprint SMACS grid mapping experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MODEL = "explicit_spectral_grid_field"
MATCHED_EXPERIMENT_FIELDS = (
    "input_product",
    "exposure_dir",
    "bands",
    "module",
    "psf_model",
    "background_model",
    "background_reference",
    "independent_frozen_calibration",
    "calibration_family",
    "measured_passband_dir",
    "optimizer_steps",
    "seed",
    "patt_nums",
    "target_header_path",
    "operator_calibration_manifest",
    "operator_calibration_fit_report",
    "grid_psf_margin_sigma",
)
LOWER_IS_BETTER = (
    "gaussian_nll_per_pixel",
    "chi2_per_pixel",
    "source_contrast_rmse",
    "source_rmse",
    "support_rmse",
    "aligned_rmse",
)
CLOSER_TO_ONE_IS_BETTER = (
    "source_contrast_slope",
    "support_flux_ratio",
    "source_flux_ratio",
)
INFORMATIONAL = (
    "fit_time_ms",
    "peak_accelerator_memory_bytes",
    "grid_target_intervals_x",
    "grid_target_intervals_y",
    "grid_mean_cell_arcsec_x",
    "grid_mean_cell_arcsec_y",
)


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows", []):
        if row.get("model") != MODEL:
            continue
        band = str(row.get("band", ""))
        if not band or band in rows:
            raise ValueError(f"invalid or duplicate {MODEL} row for band {band!r}")
        if row.get("status") != "OK":
            raise ValueError(f"{MODEL}/{band} did not complete successfully")
        rows[band] = row
    if not rows:
        raise ValueError(f"no successful {MODEL} rows")
    return rows


def _validation_scores(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scoring = payload.get("experiment", {}).get("validation_detector_scoring", {})
    scores = scoring.get("scores_by_model", {}).get(MODEL, [])
    result: dict[str, dict[str, Any]] = {}
    for row in scores:
        band = str(row.get("band", ""))
        if not band or band in result:
            raise ValueError(f"invalid or duplicate validation score for band {band!r}")
        result[band] = row
    return result


def _mapping(rows: dict[str, dict[str, Any]]) -> tuple[str, Any]:
    mappings = {str(row.get("grid_spatial_mapping")) for row in rows.values()}
    bounds = {
        json.dumps(row.get("grid_spatial_bounds_local_deg"), sort_keys=True)
        for row in rows.values()
    }
    if len(mappings) != 1 or len(bounds) != 1:
        raise ValueError("grid mapping diagnostics differ across bands")
    return mappings.pop(), json.loads(bounds.pop())


def _classify(control: float, treatment: float, direction: str) -> str:
    if math.isclose(control, treatment, rel_tol=0.0, abs_tol=1e-12):
        return "unchanged"
    if direction == "lower":
        return "improved" if treatment < control else "worse"
    if direction == "closer_to_one":
        return (
            "improved"
            if abs(treatment - 1.0) < abs(control - 1.0)
            else "worse"
        )
    return "informational"


def _delta(control: float, treatment: float, direction: str) -> dict[str, Any]:
    difference = treatment - control
    return {
        "control": control,
        "treatment": treatment,
        "delta": difference,
        "relative_delta": difference / abs(control) if control != 0.0 else None,
        "preferred_direction": direction,
        "classification": _classify(control, treatment, direction),
    }


def _audit_matched_configuration(
    control: dict[str, Any],
    treatment: dict[str, Any],
    *,
    allow_spatial_resolution_change: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    control_experiment = control.get("experiment", {})
    treatment_experiment = treatment.get("experiment", {})
    matched: dict[str, Any] = {}
    for field in MATCHED_EXPERIMENT_FIELDS:
        if control_experiment.get(field) != treatment_experiment.get(field):
            raise ValueError(f"experiment field differs: {field}")
        matched[field] = control_experiment.get(field)
    if control_experiment.get("grid_fit_target_footprint") is not False:
        raise ValueError("control must use legacy grid mapping")
    if treatment_experiment.get("grid_fit_target_footprint") is not True:
        raise ValueError("treatment must use target-footprint grid mapping")

    control_config = control.get("effective_model_configs", {}).get(MODEL, {})
    treatment_config = treatment.get("effective_model_configs", {}).get(MODEL, {})
    control_model = dict(control_config.get("model_kwargs", {}))
    treatment_model = dict(treatment_config.get("model_kwargs", {}))
    control_bounds = control_model.pop("spatial_bounds_local_deg", None)
    treatment_bounds = treatment_model.pop("spatial_bounds_local_deg", None)
    if control_bounds is not None:
        raise ValueError("control model must not set spatial_bounds_local_deg")
    if treatment_bounds is None:
        raise ValueError("treatment model must set spatial_bounds_local_deg")
    allowed_differences: dict[str, Any] = {}
    if allow_spatial_resolution_change:
        control_spatial_resolution = control_model.pop("spatial_resolution", None)
        treatment_spatial_resolution = treatment_model.pop("spatial_resolution", None)
        if control_spatial_resolution != treatment_spatial_resolution:
            allowed_differences["model_kwargs.spatial_resolution"] = {
                "control": control_spatial_resolution,
                "treatment": treatment_spatial_resolution,
            }
    if control_model != treatment_model:
        raise ValueError("model kwargs differ beyond spatial_bounds_local_deg")
    if control_config.get("fit_kwargs") != treatment_config.get("fit_kwargs"):
        raise ValueError("fit kwargs differ")
    return matched, allowed_differences


def compare_runs(
    control: dict[str, Any],
    treatment: dict[str, Any],
    *,
    allow_spatial_resolution_change: bool = False,
) -> dict[str, Any]:
    """Return a conservative paired screening report for two benchmark payloads."""
    matched, allowed_differences = _audit_matched_configuration(
        control,
        treatment,
        allow_spatial_resolution_change=allow_spatial_resolution_change,
    )
    control_rows = _rows(control)
    treatment_rows = _rows(treatment)
    if control_rows.keys() != treatment_rows.keys():
        raise ValueError("completed bands differ")
    control_validation = _validation_scores(control)
    treatment_validation = _validation_scores(treatment)
    if not control_validation or control_validation.keys() != control_rows.keys():
        raise ValueError("control validation-score bands are incomplete")
    if treatment_validation.keys() != treatment_rows.keys():
        raise ValueError("treatment validation-score bands are incomplete")
    if control_validation.keys() != treatment_validation.keys():
        raise ValueError("validation-score bands differ")
    control_mapping, control_bounds = _mapping(control_rows)
    treatment_mapping, treatment_bounds = _mapping(treatment_rows)
    if control_mapping != "legacy_tanh_domain":
        raise ValueError("control must use legacy grid mapping")
    if treatment_mapping != "target_footprint_linear":
        raise ValueError("treatment must use target-footprint grid mapping")

    bands: dict[str, dict[str, Any]] = {}
    for band in sorted(control_rows):
        control_row = control_rows[band]
        treatment_row = treatment_rows[band]
        metrics: dict[str, Any] = {}
        for name in LOWER_IS_BETTER:
            source_control = (
                control_validation[band] if name in control_validation[band] else control_row
            )
            source_treatment = (
                treatment_validation[band]
                if name in treatment_validation[band]
                else treatment_row
            )
            metrics[name] = _delta(
                _finite_number(source_control[name], label=f"control {band} {name}"),
                _finite_number(source_treatment[name], label=f"treatment {band} {name}"),
                "lower",
            )
        for name in CLOSER_TO_ONE_IS_BETTER:
            metrics[name] = _delta(
                _finite_number(control_row[name], label=f"control {band} {name}"),
                _finite_number(treatment_row[name], label=f"treatment {band} {name}"),
                "closer_to_one",
            )
        for name in INFORMATIONAL:
            metrics[name] = _delta(
                _finite_number(control_row[name], label=f"control {band} {name}"),
                _finite_number(treatment_row[name], label=f"treatment {band} {name}"),
                "informational",
            )
        bands[band] = metrics

    return {
        "schema_version": 1,
        "profile": "smacs-grid-mapping-paired-screen",
        "claim_status": "screening_only_no_depth_or_resolution_claim",
        "model": MODEL,
        "matched_experiment": matched,
        "configuration_provenance": {
            "allow_spatial_resolution_change": allow_spatial_resolution_change,
            "allowed_differences": allowed_differences,
        },
        "control_mapping": control_mapping,
        "treatment_mapping": treatment_mapping,
        "control_spatial_bounds_local_deg": control_bounds,
        "treatment_spatial_bounds_local_deg": treatment_bounds,
        "bands": bands,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SMACS grid mapping paired screen",
        "",
        "This is a controlled reconstruction screen, not a depth or effective-resolution claim.",
        "",
        f"- control: `{report['control_mapping']}`",
        f"- treatment: `{report['treatment_mapping']}`",
        "",
        "| band | metric | control | treatment | delta | classification |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for band, metrics in report["bands"].items():
        for metric, row in metrics.items():
            lines.append(
                f"| {band} | {metric} | {row['control']:.6g} | "
                f"{row['treatment']:.6g} | {row['delta']:.6g} | "
                f"{row['classification']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument(
        "--allow-spatial-resolution-change",
        action="store_true",
        help="Allow model_kwargs.spatial_resolution to differ between runs.",
    )
    args = parser.parse_args()
    control = json.loads(args.control.read_text(encoding="utf-8"))
    treatment = json.loads(args.treatment.read_text(encoding="utf-8"))
    report = compare_runs(
        control,
        treatment,
        allow_spatial_resolution_change=args.allow_spatial_resolution_change,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
