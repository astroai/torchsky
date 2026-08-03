"""Score frozen SMACS J0723 spectral-Foam validation products."""

from __future__ import annotations

import argparse
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
    common_support_mask,
    effective_psf_metrics,
    stack_effective_psf,
)
from score_smacs0723_native_classical import _audit_manifest_support, _sha256

BANDS = ("F090W", "F150W")
BATCHES = ("blank", "validation-000", "validation-001", "validation-002")
METHOD = "spectral_power_foam"


def _session_commit(root: Path) -> str:
    log = (root / "session.log").read_text(encoding="utf-8")
    match = re.search(r"^requested_commit=([0-9a-f]{40})$", log, re.MULTILINE)
    if match is None:
        raise ValueError(f"{root}: exact source commit is missing")
    return match.group(1)


def _foam_product(root: Path, batch: str, band: str, name: str) -> Path:
    return root / batch / "products" / band.lower() / METHOD / f"{name}.fits"


def _audit_foam_batch(root: Path, batch: str) -> dict[str, Any]:
    benchmark_path = root / batch / "shared-foam.json"
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    rows = {
        str(row.get("band")): row
        for row in payload.get("rows", [])
        if row.get("model") == METHOD
    }
    failed = [
        band for band in BANDS if band not in rows or rows[band].get("status") != "OK"
    ]
    if failed:
        raise ValueError(f"{benchmark_path}: Foam bands are not complete: {failed}")
    products = []
    for band in BANDS:
        for name in ("prediction", "support_mask"):
            path = _foam_product(root, batch, band, name)
            if not path.is_file():
                raise FileNotFoundError(path)
            products.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    return {
        "batch": batch,
        "benchmark_json": str(benchmark_path.resolve()),
        "benchmark_json_sha256": _sha256(benchmark_path),
        "products": products,
    }


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f"stale input link: {target}")
    elif target.exists():
        raise ValueError(f"refusing to replace existing input: {target}")
    else:
        os.symlink(source.resolve(), target)


def _stage_inputs(
    blank_root: Path,
    injection_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    roots = {"blank": blank_root} | {
        batch: injection_root for batch in BATCHES[1:]
    }
    audits = []
    for batch, root in roots.items():
        audits.append(_audit_foam_batch(root, batch))
        for band in BANDS:
            product_root = work_root / "inputs" / batch / band / "products"
            _link(
                _foam_product(root, batch, band, "prediction"),
                product_root / f"{METHOD}_sci.fits",
            )
            # ponytail: Foam currently exports binary support, so blank-response
            # whitening supplies the empirical noise scale; replace this link
            # when the model exports propagated spatial inverse variance.
            _link(
                _foam_product(root, batch, band, "support_mask"),
                product_root / f"{METHOD}_wht.fits",
            )
    return {
        "blank_root": str(blank_root),
        "blank_commit": _session_commit(blank_root),
        "injection_root": str(injection_root),
        "injection_commit": _session_commit(injection_root),
        "batch_audit": audits,
    }


def _build_effective_psfs(
    input_root: Path,
    manifest_path: Path,
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_xy = torch.as_tensor(manifest["target_xy"], dtype=torch.float64).reshape(
        -1, 2
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rows = {}
    for band in BANDS:
        blank = torchsky.read(
            str(input_root / "blank" / band / "products" / f"{METHOD}_sci.fits")
        )
        injected = torchsky.read(
            str(
                input_root
                / "validation-000"
                / band
                / "products"
                / f"{METHOD}_sci.fits"
            )
        )
        if blank.shape != injected.shape:
            raise ValueError(f"blank and injected Foam products differ for {band}")
        psf = stack_effective_psf(
            injected - blank,
            source_xy.to(device=blank.device, dtype=blank.dtype),
            radius_px=9,
        )
        path = output_root / f"{band}_{METHOD}_effective_psf.fits"
        torchsky.write(str(path), psf, overwrite=True)
        rows[band] = {
            **effective_psf_metrics(psf),
            "path": str(path),
            "sha256": _sha256(path),
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blank-root", type=Path, required=True)
    parser.add_argument("--injection-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--detection-threshold", type=float, default=5.0)
    parser.add_argument("--target-purity", type=float, default=0.95)
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    blank_root = args.blank_root.expanduser().resolve()
    injection_root = args.injection_root.expanduser().resolve()
    fanout = _stage_inputs(blank_root, injection_root, out_dir)
    input_root = out_dir / "inputs"
    validation_root = args.validation_root.expanduser().resolve()
    manifests = tuple(
        (validation_root / batch / "manifest.json").resolve() for batch in BATCHES[1:]
    )

    support_inputs = [
        torchsky.read(
            str(
                input_root
                / batch
                / band
                / "products"
                / f"{METHOD}_wht.fits"
            )
        )
        for batch in BATCHES
        for band in BANDS
    ]
    support = common_support_mask(
        support_inputs,
        threshold=0.0,
        erode_radius_px=9,
        min_footprint_fraction=0.99,
    )
    if not bool(support.any()):
        raise ValueError("Foam products have no common eroded support")
    _, target_header = torchsky.read(
        str(input_root / "blank" / BANDS[0] / "products" / f"{METHOD}_sci.fits"),
        return_header=True,
    )
    support_path = out_dir / "foam_common_support.fits"
    torchsky.write(
        str(support_path),
        support.to(torch.uint8),
        header=dict(target_header),
        overwrite=True,
    )
    manifest_audit = _audit_manifest_support(manifests, support)
    psf_root = out_dir / "effective_psfs"
    psf_rows = _build_effective_psfs(input_root, manifests[0], psf_root)

    method_root = out_dir / METHOD
    control = _updec_classical_detection_artifact(
        product_root=input_root / "blank",
        effective_psf_root=psf_root,
        support_mask_path=support_path,
        bands=BANDS,
        method=METHOD,
        detection_threshold=args.detection_threshold,
        background_block_size=64,
        statistic_out=method_root / "blank_statistic.fits",
    )
    control_path = method_root / "blank_detection.json"
    control_path.parent.mkdir(parents=True, exist_ok=True)
    control_path.write_text(json.dumps(control, indent=2) + "\n", encoding="utf-8")

    detection_paths = []
    for batch in BATCHES[1:]:
        detection = _updec_classical_detection_artifact(
            product_root=input_root / batch,
            effective_psf_root=psf_root,
            support_mask_path=support_path,
            bands=BANDS,
            method=METHOD,
            detection_threshold=args.detection_threshold,
            background_block_size=64,
            statistic_out=method_root / f"{batch}_statistic.fits",
            whitening_path=control_path,
        )
        path = method_root / f"{batch}_detection.json"
        path.write_text(json.dumps(detection, indent=2) + "\n", encoding="utf-8")
        detection_paths.append(path)
    validation = _updec_classical_validation_artifact(
        detection_json_paths=tuple(detection_paths),
        manifest_paths=manifests,
        control_json_path=control_path,
        control_radius_px=6.0,
        match_radius_px=2.0,
        target_purity=args.target_purity,
    )
    validation_path = method_root / "validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    scoring_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    summary = {
        "schema_version": 1,
        "profile": "updec-adass-spectral-foam-validation",
        "depth_claim": False,
        "scoring_commit": scoring_commit,
        "bands": list(BANDS),
        "method": METHOD,
        "batches": list(BATCHES),
        "target_purity": args.target_purity,
        "detection_threshold": args.detection_threshold,
        "support": {
            "path": str(support_path),
            "sha256": _sha256(support_path),
            "pixel_count": int(support.sum()),
            "fraction": float(support.to(torch.float64).mean()),
        },
        "manifest_audit": manifest_audit,
        "effective_psf": psf_rows,
        "fanout": fanout,
        "control": control,
        "validation": validation,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchfits", "torchsky")
        },
    }
    summary_path = out_dir / "adass_updec_foam_validation.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
