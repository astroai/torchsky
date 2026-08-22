"""Generic model-selection scorer over (scene model x operator arm) cells.

Loops the already-generic detection/validation builders from
``bench_multiband_depth`` across product cells laid out as
``<input-root>/<module>/<batch>/products/<psf_mode>/<band>/<model>/`` and
selects the winning cell on validation batches only (pre-registered gate:
selection on validation, test scored once afterwards). The report carries a
``psf_mode`` axis; a learned-PSF cell cannot win unless it also beats the
analytic arm's held-out calibration gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torchsky

from bench_multiband_depth import (
    _updec_classical_detection_artifact,
    _updec_classical_validation_artifact,
    _updec_control_union_artifact,
    common_support_mask,
)

DEFAULT_BANDS = ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")
BATCHES = ("blank", "validation-000", "validation-001", "validation-002")
BOOTSTRAP_DRAWS = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_methods(spec: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        psf_mode, _, model_kind = part.partition("/")
        if not psf_mode or not model_kind:
            raise ValueError(f"method cell must be psf_mode/model_kind: {part!r}")
        pairs.append((psf_mode, model_kind))
    if not pairs:
        raise ValueError("SCORE_METHODS resolved to an empty method list")
    return pairs


def _audit_manifest_support(
    manifest_paths: tuple[Path, ...],
    support: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in manifest_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        xy = torch.as_tensor(payload["target_xy"], dtype=torch.float64).reshape(-1, 2)
        rounded = torch.round(xy).to(torch.int64)
        inside = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < support.shape[1])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < support.shape[0])
        )
        accepted = torch.zeros(xy.shape[0], dtype=torch.bool)
        accepted[inside] = support[
            rounded[inside, 1],
            rounded[inside, 0],
        ]
        rows.append(
            {
                "batch_id": str(payload["batch_id"]),
                "manifest": str(path.resolve()),
                "manifest_sha256": _sha256(path),
                "source_count": int(xy.shape[0]),
                "supported_source_count": int(accepted.sum()),
            }
        )
        if not bool(accepted.all()):
            raise ValueError(
                f"{path}: {int((~accepted).sum())} frozen injections fall "
                "outside the common support"
            )
    return rows


def _stage_method_products(
    input_root: Path,
    module: str,
    bands: tuple[str, ...],
    psf_mode: str,
    model_kind: str,
    work_root: Path,
) -> Path:
    """Symlink prediction/support products into the classical builder layout."""
    method = f"{psf_mode}__{model_kind}"
    for batch in BATCHES:
        for band in bands:
            source = (
                input_root
                / module
                / batch
                / "products"
                / psf_mode
                / band
                / model_kind
            )
            prediction = source / "prediction.fits"
            support_mask = source / "support_mask.fits"
            if not prediction.is_file() or not support_mask.is_file():
                raise FileNotFoundError(
                    f"missing products for {psf_mode}/{model_kind}: {source}"
                )
            target = work_root / batch / band / "products"
            target.mkdir(parents=True, exist_ok=True)
            sci = target / f"{method}_sci.fits"
            wht = target / f"{method}_wht.fits"
            for link, source_path in ((sci, prediction), (wht, support_mask)):
                if link.is_symlink():
                    if link.resolve() != source_path.resolve():
                        raise ValueError(f"staged link mismatch: {link}")
                elif not link.exists():
                    os.symlink(source_path.resolve(), link)
    return work_root


def _magnitude_bins(completeness: dict[str, Any]) -> tuple[list[float], list[float]]:
    magnitudes = sorted(float(key) for key in completeness)
    values = [
        float(completeness[f"{magnitude:g}"]["completeness"])
        for magnitude in magnitudes
    ]
    return magnitudes, values


def _m50(magnitudes: list[float], completeness: list[float]) -> float:
    """Interpolated magnitude at 50% completeness (NaN if never reached)."""
    for index, value in enumerate(completeness):
        if value >= 0.5:
            if index == 0:
                return magnitudes[0]
            low_m, low_c = magnitudes[index - 1], completeness[index - 1]
            high_m, high_c = magnitudes[index], value
            fraction = (0.5 - low_c) / max(high_c - low_c, 1e-12)
            return low_m + fraction * (high_m - low_m)
    return float("nan")


def _m50_bootstrap_ci(
    batches: list[dict[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 1729,
) -> dict[str, float]:
    """Cluster bootstrap over validation batches for the m50 CI."""
    generator = torch.Generator().manual_seed(seed)
    estimates = []
    count = len(batches)
    for _ in range(draws):
        indices = torch.randint(0, count, (count,), generator=generator).tolist()
        counts: dict[float, list[int]] = {}
        for index in indices:
            for magnitude, pair in batches[index].items():
                cell = counts.setdefault(float(magnitude), [0, 0])
                cell[0] += int(pair["truth_count"])
                cell[1] += int(pair["matched_count"])
        magnitudes = sorted(counts)
        completeness = [
            counts[magnitude][1] / max(counts[magnitude][0], 1)
            for magnitude in magnitudes
        ]
        estimates.append(_m50(magnitudes, completeness))
    valid = torch.tensor([v for v in estimates if math.isfinite(v)])
    if valid.numel() < draws // 2:
        return {
            "m50": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "finite_draws": int(valid.numel()),
        }
    lower = torch.quantile(valid, 0.025).item()
    upper = torch.quantile(valid, 0.975).item()
    return {
        "m50": float(valid.median()),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "finite_draws": int(valid.numel()),
    }


def _heldout_gate(
    psf_mode: str,
    calibration_fit_report: Path | None,
) -> bool:
    """Learned arms must beat the analytic arm on held-out calibration NLL."""
    if psf_mode != "learned_lowrank":
        return True
    if calibration_fit_report is None:
        return False
    payload = json.loads(calibration_fit_report.read_text(encoding="utf-8"))
    for comparison in payload.get("held_out_comparisons", []):
        if str(comparison.get("role")) != "psf":
            continue
        validation = comparison.get("validation", {})
        improvement = float(validation.get("adaptive_improvement_fraction", 0.0))
        if improvement <= 0.0:
            return False
    return True


def _select(
    rows: list[dict[str, Any]],
    *,
    calibration_fit_report: Path | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    eligible = [
        row
        for row in rows
        if row["target_purity_reached"]
        and math.isfinite(row["m50_ci"]["m50"])
        and _heldout_gate(row["psf_mode"], calibration_fit_report)
    ]
    if not eligible:
        return None, rows
    winner = max(eligible, key=lambda row: row["m50_ci"]["m50"])
    return winner, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--methods",
        required=True,
        help="CSV of psf_mode/model_kind cells, e.g. "
        "fitted_gaussian/spectral_power_foam,learned_lowrank/spectral_power_foam",
    )
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--module", default="b")
    parser.add_argument("--bands", default=",".join(DEFAULT_BANDS))
    parser.add_argument("--effective-psf-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--detection-threshold", type=float, default=5.0)
    parser.add_argument("--target-purity", type=float, default=0.95)
    parser.add_argument(
        "--calibration-fit-report",
        type=Path,
        default=None,
        help="Operator calibration fit report gating learned-PSF cells",
    )
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    validation_root = args.validation_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bands = tuple(b.strip().upper() for b in str(args.bands).split(",") if b.strip())
    methods = _parse_methods(args.methods)
    calibration_fit_report = (
        args.calibration_fit_report.expanduser().resolve()
        if args.calibration_fit_report is not None
        else None
    )
    manifests = tuple(
        (validation_root / batch / "manifest.json").resolve() for batch in BATCHES[1:]
    )

    # Common support across every staged cell, from the support masks.
    staged: dict[tuple[str, str], Path] = {}
    weight_rows = []
    weights = []
    target_header: dict[str, Any] | None = None
    probe = torch.tensor(
        [[0.0, 0.0], [511.0, 0.0], [0.0, 511.0], [511.0, 511.0], [255.5, 255.5]],
        dtype=torch.float64,
    )
    target_world: torch.Tensor | None = None
    for psf_mode, model_kind in methods:
        method = f"{psf_mode}__{model_kind}"
        work_root = out_dir / "inputs" / method
        staged[(psf_mode, model_kind)] = _stage_method_products(
            input_root, args.module, bands, psf_mode, model_kind, work_root
        )
        for batch in BATCHES:
            for band in bands:
                path = work_root / batch / band / "products" / f"{method}_wht.fits"
                weight, header = torchsky.read(str(path), return_header=True)
                world = torchsky.WCS(dict(header)).pixel_to_world(probe)
                if target_world is None:
                    target_world = world
                    target_header = dict(header)
                elif not torch.allclose(
                    world, target_world, rtol=0.0, atol=1e-9
                ):
                    raise ValueError(f"{path}: target WCS differs across cells")
                weights.append((weight > 0).to(weight.dtype))
                weight_rows.append({"path": str(path), "sha256": _sha256(path)})
    support = common_support_mask(
        weights,
        threshold=0.5,
        erode_radius_px=9,
        min_footprint_fraction=0.99,
    )
    if not bool(support.any()):
        raise ValueError("model cells share no common eroded support")
    assert target_header is not None
    support_path = out_dir / "model_common_support.fits"
    torchsky.write(
        str(support_path),
        support.to(torch.uint8),
        header=target_header,
        overwrite=True,
    )
    manifest_audit = _audit_manifest_support(manifests, support)

    # Effective PSFs: shared across cells so resolution is matched by
    # construction; the operator arm differs through the predictions.
    effective_psf_root = (
        args.effective_psf_root.expanduser().resolve()
        if getattr(args, "effective_psf_root", None)
        else validation_root / "effective_psfs"
    )

    controls: dict[str, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}
    for psf_mode, model_kind in methods:
        method = f"{psf_mode}__{model_kind}"
        method_root = out_dir / method
        method_root.mkdir(parents=True, exist_ok=True)
        control = _updec_classical_detection_artifact(
            product_root=staged[(psf_mode, model_kind)] / "blank",
            effective_psf_root=effective_psf_root,
            support_mask_path=support_path,
            bands=bands,
            method=method,
            detection_threshold=args.detection_threshold,
            background_block_size=64,
            statistic_out=method_root / "blank_statistic.fits",
        )
        control_path = method_root / "blank_detection.json"
        control_path.write_text(json.dumps(control, indent=2) + "\n")
        controls[method] = control

        detection_paths = []
        for batch in BATCHES[1:]:
            detection = _updec_classical_detection_artifact(
                product_root=staged[(psf_mode, model_kind)] / batch,
                effective_psf_root=effective_psf_root,
                support_mask_path=support_path,
                bands=bands,
                method=method,
                detection_threshold=args.detection_threshold,
                background_block_size=64,
                statistic_out=method_root / f"{batch}_statistic.fits",
                whitening_path=control_path,
            )
            detection_path = method_root / f"{batch}_detection.json"
            detection_path.write_text(json.dumps(detection, indent=2) + "\n")
            detection_paths.append(detection_path)
        validation = _updec_classical_validation_artifact(
            detection_json_paths=tuple(detection_paths),
            manifest_paths=manifests,
            control_json_path=control_path,
            control_radius_px=6.0,
            match_radius_px=2.0,
            target_purity=args.target_purity,
        )
        validation_path = method_root / "validation.json"
        validation_path.write_text(json.dumps(validation, indent=2) + "\n")
        validations[method] = validation

    union = _updec_control_union_artifact(
        control_json_paths=tuple(
            out_dir / f"{psf}__{model}" / "blank_detection.json"
            for psf, model in methods
        ),
        dedup_radius_px=6.0,
    )
    union_path = out_dir / "model_control_union.json"
    union_path.write_text(json.dumps(union, indent=2) + "\n")

    per_method = []
    for psf_mode, model_kind in methods:
        method = f"{psf_mode}__{model_kind}"
        validation = validations[method]
        # Cluster bootstrap over the builders' per-batch completeness curves;
        # pooling would collapse every draw to the same m50 (zero-width CI).
        batch_curves = [
            {
                float(key): {
                    "truth_count": value["truth_count"],
                    "matched_count": value["matched_count"],
                }
                for key, value in row["completeness_by_ab_magnitude"].items()
            }
            for row in validation["batches"]
        ]
        m50_ci = _m50_bootstrap_ci(batch_curves, seed=args.seed)
        per_method.append(
            {
                "method": method,
                "psf_mode": psf_mode,
                "model_kind": model_kind,
                "target_purity_reached": bool(validation["target_purity_reached"]),
                "pooled_completeness": float(validation["pooled"]["completeness"]),
                "pooled_purity": float(validation["pooled"]["purity"]),
                "m50_ci": m50_ci,
                "heldout_gate_pass": _heldout_gate(
                    psf_mode, calibration_fit_report
                ),
            }
        )
    winner, per_method = _select(
        per_method, calibration_fit_report=calibration_fit_report
    )

    summary = {
        "schema_version": 1,
        "profile": "updec-adass-model-selection",
        "depth_claim": False,
        "input_root": str(input_root),
        "module": str(args.module),
        "bands": list(bands),
        "batches": list(BATCHES),
        "methods": [f"{psf}/{model}" for psf, model in methods],
        "selected": (
            {
                "method": winner["method"],
                "psf_mode": winner["psf_mode"],
                "model_kind": winner["model_kind"],
            }
            if winner is not None
            else None
        ),
        "selection_rule": (
            "validation-batches-only: purity gate, held-out calibration gate "
            "for learned PSF arms, then largest bootstrap-median m50"
        ),
        "per_method": per_method,
        "target_purity": args.target_purity,
        "detection_threshold": args.detection_threshold,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "support": {
            "path": str(support_path),
            "sha256": _sha256(support_path),
            "pixel_count": int(support.sum()),
            "weight_rows": weight_rows,
        },
        "manifest_audit": manifest_audit,
        "effective_psf_root": str(effective_psf_root),
        "calibration_fit_report": (
            str(calibration_fit_report) if calibration_fit_report else None
        ),
        "control_union": {
            "path": str(union_path),
            "sha256": _sha256(union_path),
            "union_count": union["union_count"],
        },
        "validation": validations,
    }
    summary_path = out_dir / "model_selection.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    lines = [
        "# Model selection (validation batches only)",
        "",
        f"- profile: `{summary['profile']}` (depth_claim=false)",
        f"- module: {args.module}",
        f"- cells: {', '.join(summary['methods'])}",
        (
            f"- selected: **{summary['selected']['method']}**"
            if summary["selected"]
            else "- selected: none (no cell passed the gates)"
        ),
        "",
        "| method | psf_mode | purity gate | m50 | 95% CI | heldout gate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in per_method:
        ci = row["m50_ci"]
        lines.append(
            f"| {row['method']} | {row['psf_mode']} "
            f"| {row['target_purity_reached']} "
            f"| {ci['m50']:.3f} "
            f"| [{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}] "
            f"| {row['heldout_gate_pass']} |"
        )
    (out_dir / "model_selection.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
