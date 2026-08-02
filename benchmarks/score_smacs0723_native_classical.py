"""Score the completed native Torchsky Drizzle validation fanout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
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

BANDS = ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")
METHODS = (
    "torchsky_drizzle_square_pf0p800",
    "torchsky_drizzle_square_pf1p000",
)
BATCHES = ("blank", "validation-000", "validation-001", "validation-002")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root(pattern: str, band: str) -> Path:
    try:
        return Path(pattern.format(band=band)).expanduser().resolve()
    except KeyError as exc:
        raise ValueError("input root pattern must use only the {band} field") from exc


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
        row = {
            "batch_id": str(payload["batch_id"]),
            "manifest": str(path.resolve()),
            "manifest_sha256": _sha256(path),
            "source_count": int(xy.shape[0]),
            "supported_source_count": int(accepted.sum()),
        }
        rows.append(row)
        if not bool(accepted.all()):
            raise ValueError(
                f"{path}: {int((~accepted).sum())} frozen injections fall outside "
                "the native common support; regenerate manifests before scoring"
            )
    return rows


def _select_method(
    validations: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for priority, method in enumerate(METHODS):
        result = validations[method]
        by_magnitude = result["completeness_by_ab_magnitude"]
        faintest = max(float(value) for value in by_magnitude)
        row = {
            "method": method,
            "target_purity_reached": bool(result["target_purity_reached"]),
            "pooled_completeness": float(result["pooled"]["completeness"]),
            "pooled_purity": float(result["pooled"]["purity"]),
            "faintest_ab_magnitude": faintest,
            "faintest_completeness": float(
                by_magnitude[f"{faintest:g}"]["completeness"]
            ),
            "tie_break_priority": priority,
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["target_purity_reached"],
            row["pooled_completeness"],
            row["faintest_completeness"],
            -row["tie_break_priority"],
        ),
        reverse=True,
    )
    if not rows[0]["target_purity_reached"]:
        raise ValueError("neither native Drizzle candidate reached target purity")
    return str(rows[0]["method"]), rows


def _audit_native_batch(
    source_root: Path,
    *,
    band: str,
    batch: str,
) -> dict[str, Any]:
    batch_root = source_root / batch / band
    benchmark_path = batch_root / "stackers.json"
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    by_method = {str(row.get("name")): row for row in payload.get("rows", [])}
    failed = [
        method
        for method in METHODS
        if method not in by_method or by_method[method].get("status") != "OK"
    ]
    if failed:
        raise ValueError(
            f"{benchmark_path}: native methods are not complete: {failed}"
        )
    products = []
    for method in METHODS:
        for suffix in ("sci", "wht"):
            path = batch_root / "products" / f"{method}_{suffix}.fits"
            if not path.is_file():
                raise FileNotFoundError(path)
            products.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    return {
        "batch": batch,
        "stackers_json": str(benchmark_path.resolve()),
        "stackers_json_sha256": _sha256(benchmark_path),
        "products": products,
    }


def _link_fanout(pattern: str, work_root: Path) -> list[dict[str, Any]]:
    rows = []
    for band in BANDS:
        source_root = _root(pattern, band)
        log = (source_root / "session.log").read_text(encoding="utf-8")
        match = re.search(r"^requested_commit=([0-9a-f]{40})$", log, re.MULTILINE)
        if match is None:
            raise ValueError(f"{source_root}: exact source commit is missing")
        batch_audit = [
            _audit_native_batch(source_root, band=band, batch=batch)
            for batch in BATCHES
        ]
        for batch in BATCHES:
            source = source_root / batch / band / "products"
            if not source.is_dir():
                raise FileNotFoundError(source)
            target = work_root / "inputs" / batch / band / "products"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise ValueError(f"stale input link: {target}")
            elif target.exists():
                raise ValueError(f"refusing to replace existing input: {target}")
            else:
                os.symlink(source, target)
        rows.append(
            {
                "band": band,
                "source_root": str(source_root),
                "source_commit": match.group(1),
                "session_log_sha256": _sha256(source_root / "session.log"),
                "batch_audit": batch_audit,
            }
        )
    commits = {row["source_commit"] for row in rows}
    if len(commits) != 1:
        raise ValueError(f"fanout used multiple source commits: {sorted(commits)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root-pattern", required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--effective-psf-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--detection-threshold", type=float, default=5.0)
    parser.add_argument("--target-purity", type=float, default=0.95)
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fanout_rows = _link_fanout(args.input_root_pattern, out_dir)
    input_root = out_dir / "inputs"
    validation_root = args.validation_root.expanduser().resolve()
    effective_psf_root = args.effective_psf_root.expanduser().resolve()
    manifests = tuple(
        (validation_root / batch / "manifest.json").resolve() for batch in BATCHES[1:]
    )

    weights = []
    target_header: dict[str, Any] | None = None
    target_world: torch.Tensor | None = None
    probe = torch.tensor(
        [[0.0, 0.0], [511.0, 0.0], [0.0, 511.0], [511.0, 511.0], [255.5, 255.5]],
        dtype=torch.float64,
    )
    weight_rows = []
    for band in BANDS:
        for method in METHODS:
            path = input_root / "blank" / band / "products" / f"{method}_wht.fits"
            weight, header = torchsky.read(str(path), return_header=True)
            if weight.shape != (512, 512):
                raise ValueError(f"{path}: expected a 512-square target")
            world = torchsky.WCS(dict(header)).pixel_to_world(probe)
            if target_world is None:
                target_world = world
                target_header = dict(header)
            elif not torch.allclose(world, target_world, rtol=0.0, atol=1e-9):
                raise ValueError(f"{path}: target WCS differs from the fanout")
            weights.append(weight)
            weight_rows.append({"path": str(path), "sha256": _sha256(path)})
    support = common_support_mask(
        weights,
        threshold=0.0,
        erode_radius_px=9,
        min_footprint_fraction=0.99,
    )
    if not bool(support.any()):
        raise ValueError("native candidates have no common eroded support")
    assert target_header is not None
    support_path = out_dir / "native_common_support.fits"
    torchsky.write(
        str(support_path),
        support.to(torch.uint8),
        header=target_header,
        overwrite=True,
    )
    manifest_audit = _audit_manifest_support(manifests, support)

    controls: dict[str, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        method_root = out_dir / method
        control = _updec_classical_detection_artifact(
            product_root=input_root / "blank",
            effective_psf_root=effective_psf_root,
            support_mask_path=support_path,
            bands=BANDS,
            method=method,
            detection_threshold=args.detection_threshold,
            background_block_size=64,
            statistic_out=method_root / "blank_statistic.fits",
        )
        control_path = method_root / "blank_detection.json"
        control_path.parent.mkdir(parents=True, exist_ok=True)
        control_path.write_text(json.dumps(control, indent=2) + "\n")
        controls[method] = control

        detection_paths = []
        for batch in BATCHES[1:]:
            detection = _updec_classical_detection_artifact(
                product_root=input_root / batch,
                effective_psf_root=effective_psf_root,
                support_mask_path=support_path,
                bands=BANDS,
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
            out_dir / method / "blank_detection.json" for method in METHODS
        ),
        dedup_radius_px=6.0,
    )
    union_path = out_dir / "native_control_union.json"
    union_path.write_text(json.dumps(union, indent=2) + "\n")
    winner, selection_rows = _select_method(validations)

    source_commit = fanout_rows[0]["source_commit"]
    scoring_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    summary = {
        "schema_version": 1,
        "profile": "updec-adass-native-classical-selection",
        "depth_claim": False,
        "source_commit": source_commit,
        "scoring_commit": scoring_commit,
        "bands": list(BANDS),
        "methods": list(METHODS),
        "batches": list(BATCHES),
        "selection_rule": (
            "target purity reached, pooled completeness, faintest-bin "
            "completeness, then predeclared pixfrac=0.8 priority"
        ),
        "selected_method": winner,
        "selection_rows": selection_rows,
        "target_purity": args.target_purity,
        "detection_threshold": args.detection_threshold,
        "support": {
            "path": str(support_path),
            "sha256": _sha256(support_path),
            "pixel_count": int(support.sum()),
            "fraction": float(support.to(torch.float64).mean()),
            "erode_radius_px": 9,
            "min_footprint_fraction": 0.99,
            "weight_rows": weight_rows,
        },
        "manifest_audit": manifest_audit,
        "effective_psf_root": str(effective_psf_root),
        "effective_psf_sha256": {
            f"{band}/{method}": _sha256(
                effective_psf_root / f"{band}_{method}_effective_psf.fits"
            )
            for band in BANDS
            for method in METHODS
        },
        "fanout": fanout_rows,
        "control_union": {
            "path": str(union_path),
            "sha256": _sha256(union_path),
            "input_detection_count": union["input_detection_count"],
            "union_count": union["union_count"],
        },
        "validation": validations,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchfits", "torchsky")
        },
    }
    summary_path = out_dir / "adass_updec_classical_config.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
