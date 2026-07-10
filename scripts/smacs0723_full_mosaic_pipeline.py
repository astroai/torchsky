#!/usr/bin/env python3
"""Opinionated full-mosaic SMACS J0723 pipeline.

Downloads the public module-b JWST exposures and published UPdec-Webb reference
stacks, builds full-grid torchsky mosaics on the published grids, compares those
coadds to the full published stacks, and optionally fits one full-grid INR per band.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks"
SCRIPTS = ROOT / "scripts"
for p in (SRC, BENCH, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torchsky  # noqa: E402
from bench_map_field_fit import _field_model_kwargs_for_case  # noqa: E402
from bench_map_smacs0723_field_fit import (  # noqa: E402
    REAL_CASE,
    _fit_with_timing,
    _real_fit_kwargs,
    _robust_intensity_scale,
    _scaled_field_observation,
)
from bench_map_smacs0723_real import (  # noqa: E402
    _discover_exposure_paths,
    _field_observation_kwargs_for_band,
    _metric_mask,
    _prepare_aux_files,
    _read_ext,
    _reference_path_for_band,
    _sci_header,
    _valid_data_mask,
)
from download_jwst_smacs0723_mast import (  # noqa: E402
    download_product as download_jwst_product,
    select_products,
)
from download_smacs0723_multiband_public import ALL_BANDS  # noqa: E402
from download_updec_webb import download_spec as download_updec_spec  # noqa: E402
from download_updec_webb import resolve_specs  # noqa: E402
from torchsky.map import (  # noqa: E402
    FieldObservation,
    ImageObservation,
    coadd_image_plane,
    estimate_field_domain,
    make_field_model,
    render_field_observation,
)


def _parse_csv(value: str) -> list[str]:
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def _has_local_cpp() -> bool:
    try:
        importlib.import_module("torchsky")
        importlib.import_module("torchsky.cpp")
    except Exception:
        return False
    return True


def _ensure_local_cpp() -> None:
    if _has_local_cpp():
        return
    cmd = [sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation"]
    print("Installing local torchsky editable build into the active env")
    subprocess.check_call(cmd, cwd=ROOT)
    if not _has_local_cpp():
        raise RuntimeError("torchsky.cpp still unavailable after editable install")


def _band_filename_contains(band: str, module: str) -> str | None:
    if str(module).lower() not in {"a", "b"}:
        return None
    if str(band).upper() in {"F277W", "F356W", "F444W"}:
        return f"nrc{str(module).lower()}long"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--bands", default=",".join(ALL_BANDS))
    parser.add_argument("--module", default="b")
    parser.add_argument(
        "--exposure-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing local exposures. If omitted, defaults to "
            "<root>/jwst_smacs0723. For UPDC-preprocessed inputs, use "
            "data/updc_inputs/F277W/completed_dqfix."
        ),
    )
    parser.add_argument(
        "--input-product", choices=("auto", "crf", "cal", "updc"), default="cal"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for TorchSky coadd/fit, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-coadd", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--background-offset",
        choices=("none", "median", "sigma_clip_median"),
        default="none",
        help=(
            "Per-individual-exposure background subtraction before coadd. "
            "'median' subtracts median(SCI[valid]); "
            "'sigma_clip_median' iteratively clips out bright sources first."
        ),
    )
    parser.add_argument(
        "--background-sigma",
        type=float,
        default=2.5,
        help="Sigma threshold used by --background-offset sigma_clip_median.",
    )
    parser.add_argument(
        "--background-maxiters",
        type=int,
        default=5,
        help="Maximum clipping iterations for sigma-clipped background estimate.",
    )
    parser.add_argument("--train-model", default="spectral_basis_field")
    parser.add_argument("--train-steps", type=int, default=12)
    parser.add_argument("--train-runs", type=int, default=1)
    parser.add_argument("--updec-attempts", type=int, default=12)
    parser.add_argument("--updec-timeout-s", type=float, default=240.0)
    parser.add_argument("--updec-backoff-s", type=float, default=3.0)
    parser.add_argument(
        "--output-label",
        default="",
        help=(
            "Optional label inserted into the full mosaic filename, e.g. "
            "'raw_cal' -> F277W_raw_cal_torchsky_full_mosaic.fits."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _download_all(
    *,
    bands: list[str],
    module: str,
    jwst_dir: Path,
    updec_dir: Path,
    subgroup: str,
    overwrite: bool,
    updec_attempts: int,
    updec_timeout_s: float,
    updec_backoff_s: float,
) -> None:
    for spec in resolve_specs(bands, [0]):
        path = download_updec_spec(
            spec,
            updec_dir,
            overwrite=overwrite,
            attempts=updec_attempts,
            timeout_s=updec_timeout_s,
            backoff_s=updec_backoff_s,
        )
        print(f"downloaded reference {path}")
    for band in bands:
        products = select_products(
            band=band,
            subgroup=subgroup,
            max_exposures=100,
            module=module,
        )
        for product in products:
            path = download_jwst_product(product, jwst_dir, overwrite=overwrite)
            print(f"downloaded exposure {path}")


def _full_target_header(reference_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    reference = _read_ext(reference_path, "SCI").astype(np.float32)
    header = dict(_sci_header(reference_path))
    header["NAXIS1"] = int(reference.shape[1])
    header["NAXIS2"] = int(reference.shape[0])
    return reference, header


def _sigma_clip_median_numpy(
    values: np.ndarray,
    *,
    sigma: float,
    maxiters: int,
) -> float:
    """Robust median used for per-individual-exposure background estimation."""
    x = np.asarray(values, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    for _ in range(max(int(maxiters), 0)):
        med = float(np.median(x))
        std = float(np.std(x))
        if not np.isfinite(std) or std <= 0.0:
            break
        keep = np.abs(x - med) < float(sigma) * std
        if int(keep.sum()) == int(x.size):
            break
        if int(keep.sum()) == 0:
            break
        x = x[keep]
    return float(np.median(x))


def _estimate_exposure_background_offset(
    exposure_path: Path,
    *,
    mode: str,
    sigma: float,
    maxiters: int,
) -> float:
    """Estimate one scalar sky/background offset for one individual exposure."""
    mode = str(mode).lower()
    if mode == "none":
        return 0.0

    science = _read_ext(exposure_path, "SCI")
    err = _read_ext(exposure_path, "ERR")
    valid = _valid_data_mask(exposure_path, err)
    values = science[valid & np.isfinite(science)]

    if values.size == 0:
        return 0.0
    if mode == "median":
        return float(np.median(values))
    if mode == "sigma_clip_median":
        return _sigma_clip_median_numpy(
            values,
            sigma=float(sigma),
            maxiters=int(maxiters),
        )
    raise ValueError(f"unknown background offset mode: {mode}")



def _coadd_full_band(
    *,
    band: str,
    module: str,
    input_product: str,
    jwst_dir: Path,
    updec_dir: Path,
    mosaic_dir: Path,
    metric_dir: Path,
    background_offset_mode: str,
    background_sigma: float,
    background_maxiters: int,
    output_label: str,
    device: str | None,
) -> dict[str, Any]:
    reference_path = updec_dir / _reference_path_for_band(band).name
    if not reference_path.exists():
        raise FileNotFoundError(
            f"missing published reference for {band}: {reference_path}. "
            "Run without --skip-download first."
        )
    reference, target_header = _full_target_header(reference_path)
    exposure_paths = _discover_exposure_paths(
        jwst_dir,
        input_product=input_product,
        band=band,
        module=module,
    )
    if not exposure_paths:
        raise FileNotFoundError(f"no exposures found for {band} under {jwst_dir}")

    tmp_root = jwst_dir.parent / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"torchsky_full_coadd_{band}_", dir=str(tmp_root)
    ) as tmp:
        tmpdir = Path(tmp)
        observations: list[ImageObservation] = []
        background_offsets: list[float] = []
        for idx, exposure_path in enumerate(exposure_paths, start=1):
            science_path, weight_path, mask_path = _prepare_aux_files(
                exposure_path,
                tmpdir,
                linearize_sip=False,
            )
            bg_offset = _estimate_exposure_background_offset(
                exposure_path,
                mode=background_offset_mode,
                sigma=float(background_sigma),
                maxiters=int(background_maxiters),
            )
            background_offsets.append(float(bg_offset))
            print(
                f"[background] {idx}/{len(exposure_paths)} "
                f"{exposure_path.name}: offset={bg_offset:.8g}",
                flush=True,
            )
            observations.append(
                ImageObservation(
                    str(science_path),
                    hdu=0,
                    weight_path=str(weight_path),
                    weight_hdu=0,
                    mask_path=str(mask_path),
                    mask_hdu=0,
                    background_offset=float(bg_offset),
                )
            )
        print(f"Running TorchSky coadd on device={device or 'auto'}")
        result = coadd_image_plane(
            observations,
            target_header=target_header,
            resampling_kernel="drizzle_square",
            value_semantics="surface_brightness",
            stacking_method="weighted_mean",
            drizzle_pixfrac=1.0,
            device=device,
        )

    pred = result.data.detach().cpu().numpy().astype(np.float32)
    support = result.weight.detach().cpu().numpy() > 0
    mask = _metric_mask(pred, reference, support)
    diff = pred[mask] - reference[mask]
    abs_diff = np.abs(diff)
    frac_abs_diff_le_5 = (
        float(np.mean(abs_diff <= 5.0)) if diff.size else float("nan")
    )
    abs_diff_p99 = (
        float(np.percentile(abs_diff, 99.0)) if diff.size else float("nan")
    )
    passes_99pct_abs_le_5 = (
        bool(frac_abs_diff_le_5 >= 0.99) if diff.size else False
    )

    mosaic_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    label = str(output_label).strip()
    if label:
        output_path = mosaic_dir / f"{band}_{label}_torchsky_full_mosaic.fits"
    else:
        output_path = mosaic_dir / f"{band}_torchsky_full_mosaic.fits"
    torchsky.write(str(output_path), result.data, header=target_header, overwrite=True)

    row = {
        "band": band,
        "method": "coadd",
        "reference_path": str(reference_path),
        "output_path": str(output_path),
        "n_exposures": len(exposure_paths),
        "shape": [int(reference.shape[0]), int(reference.shape[1])],
        "resampling_kernel": "drizzle_square",
        "value_semantics": "surface_brightness",
        "stacking_method": "weighted_mean",
        "drizzle_pixfrac": 1.0,
        "support_rmse": float(np.sqrt(np.mean(diff * diff)))
        if diff.size
        else float("nan"),
        "support_mae": float(np.mean(abs_diff)) if diff.size else float("nan"),
        "frac_abs_diff_le_5": frac_abs_diff_le_5,
        "abs_diff_p99": abs_diff_p99,
        "passes_99pct_abs_le_5": passes_99pct_abs_le_5,
        "support_flux_ratio": (
            float(np.sum(pred[mask]) / np.sum(reference[mask]))
            if diff.size
            else float("nan")
        ),
        "coverage_frac": float(mask.mean()),
        "max_overlap": int(result.overlap_count.max().item())
        if result.overlap_count is not None
        else 0,
        "background_offset_mode": background_offset_mode,
        "background_sigma": float(background_sigma),
        "background_maxiters": int(background_maxiters),
        "background_offsets": background_offsets,
    }
    metric_path = metric_dir / f"{band}_coadd_compare.json"
    metric_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))
    return row


def _fit_full_band(
    *,
    band: str,
    module: str,
    input_product: str,
    jwst_dir: Path,
    updec_dir: Path,
    fit_dir: Path,
    metric_dir: Path,
    model_kind: str,
    steps: int,
    runs: int,
) -> dict[str, Any]:
    reference_path = updec_dir / _reference_path_for_band(band).name
    if not reference_path.exists():
        raise FileNotFoundError(
            f"missing published reference for {band}: {reference_path}. "
            "Run without --skip-download first."
        )
    reference, target_header = _full_target_header(reference_path)
    exposure_paths = _discover_exposure_paths(
        jwst_dir,
        input_product=input_product,
        band=band,
        module=module,
        filename_contains=_band_filename_contains(band, module),
    )
    if not exposure_paths:
        raise FileNotFoundError(f"no exposures found for {band} under {jwst_dir}")

    tmp_root = jwst_dir.parent / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"torchsky_full_fit_{band}_", dir=str(tmp_root)
    ) as tmp:
        tmpdir = Path(tmp)
        image_observations: list[ImageObservation] = []
        raw_field_inputs: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
        ] = []
        for exposure_path in exposure_paths:
            science_path, weight_path, mask_path = _prepare_aux_files(
                exposure_path,
                tmpdir,
                linearize_sip=False,
            )
            image_observations.append(
                ImageObservation(
                    str(science_path),
                    hdu=0,
                    weight_path=str(weight_path),
                    weight_hdu=0,
                    mask_path=str(mask_path),
                    mask_hdu=0,
                )
            )
            science = _read_ext(exposure_path, "SCI")
            err = _read_ext(exposure_path, "ERR")
            # Original logic used ERR + DQ directly.
            # Route through helper so UPDC-preprocessed inputs can also use UPDCMASK.
            valid = _valid_data_mask(exposure_path, err)
            weight = np.zeros_like(err, dtype=np.float32)
            weight[valid] = 1.0 / np.square(err[valid])
            raw_field_inputs.append(
                (
                    science,
                    weight,
                    valid.astype(np.float32, copy=False),
                    dict(_sci_header(exposure_path)),
                )
            )

        field_observations: list[FieldObservation] = []
        for image, weight, mask, header in raw_field_inputs:
            field_observations.append(
                FieldObservation(
                    image=torch.from_numpy(image),
                    header=header,
                    weight=torch.from_numpy(weight),
                    mask=torch.from_numpy(mask),
                    **_field_observation_kwargs_for_band(
                        band, observation_mode="bandpass"
                    ),
                )
            )

        coadd_reference = coadd_image_plane(
            image_observations,
            target_header=target_header,
            resampling_kernel="bilinear",
            value_semantics="flux",
            stacking_method="weighted_mean",
        )
        support_mask = coadd_reference.weight.detach().cpu().numpy() > 0

        domain = estimate_field_domain(field_observations, coord_mode="sphere_xyz")
        intensity_scale = _robust_intensity_scale(raw_field_inputs)
        normalized = [
            _scaled_field_observation(obs, intensity_scale=intensity_scale)
            for obs in field_observations
        ]
        model_kwargs = _field_model_kwargs_for_case(
            model_kind,
            case=REAL_CASE,
            size=max(int(reference.shape[0]), int(reference.shape[1])),
        )
        fit_kwargs = _real_fit_kwargs(
            model_kind,
            crop_size=max(int(reference.shape[0]), int(reference.shape[1])),
            steps=int(steps),
            observation_mode="bandpass",
            num_observations=len(normalized),
        )
        model = make_field_model(model_kind, domain, **model_kwargs)
        result, fit_time_ms = _fit_with_timing(
            normalized,
            model,
            steps=int(steps),
            runs=int(runs),
            fit_kwargs=fit_kwargs,
        )
        target_observation = FieldObservation(
            image=torch.zeros(reference.shape, dtype=torch.float32),
            header=target_header,
            weight=torch.ones(reference.shape, dtype=torch.float32),
            **_field_observation_kwargs_for_band(band, observation_mode="bandpass"),
        )
        render_n = int(fit_kwargs.get("num_spectral_samples", 9))
        with torch.no_grad():
            pred_tensor = render_field_observation(
                result.model,
                target_observation,
                num_spectral_samples=render_n,
                spatial_tile_hw=(512, 512),
            )
        pred = pred_tensor.detach().cpu().numpy().astype(np.float32) * float(
            intensity_scale
        )

    fit_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    output_path = fit_dir / f"{band}_{model_kind}_full_prediction.fits"
    torchsky.write(str(output_path), pred, header=target_header, overwrite=True)

    support_metric_mask = _metric_mask(pred, reference, support_mask)
    support_diff = pred[support_metric_mask] - reference[support_metric_mask]
    uniform_mask = np.isfinite(pred) & np.isfinite(reference)
    uniform_diff = pred[uniform_mask] - reference[uniform_mask]

    row = {
        "band": band,
        "method": "train",
        "model": model_kind,
        "reference_path": str(reference_path),
        "output_path": str(output_path),
        "n_exposures": len(exposure_paths),
        "shape": [int(reference.shape[0]), int(reference.shape[1])],
        "steps": int(steps),
        "fit_time_ms": float(fit_time_ms),
        "support_rmse": (
            float(np.sqrt(np.mean(support_diff * support_diff)))
            if support_diff.size
            else float("nan")
        ),
        "support_mae": float(np.mean(np.abs(support_diff)))
        if support_diff.size
        else float("nan"),
        "support_flux_ratio": (
            float(
                np.sum(pred[support_metric_mask])
                / np.sum(reference[support_metric_mask])
            )
            if support_diff.size
            else float("nan")
        ),
        "support_frac": float(support_metric_mask.mean()),
        "uniform_rmse": (
            float(np.sqrt(np.mean(uniform_diff * uniform_diff)))
            if uniform_diff.size
            else float("nan")
        ),
        "uniform_mae": float(np.mean(np.abs(uniform_diff)))
        if uniform_diff.size
        else float("nan"),
    }
    metric_path = metric_dir / f"{band}_{model_kind}_fit_compare.json"
    metric_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))
    return row


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        _ensure_local_cpp()
        bands = _parse_csv(args.bands)
        root = args.root
        jwst_dir = (
            args.exposure_dir.expanduser().resolve()
            if args.exposure_dir is not None
            else root / "jwst_smacs0723"
        )
        updec_dir = root / "updec_webb"
        mosaic_dir = root / "full_mosaics"
        fit_dir = root / "full_inr"
        metric_dir = root / "metrics"
        for path in (jwst_dir, updec_dir, mosaic_dir, fit_dir, metric_dir):
            path.mkdir(parents=True, exist_ok=True)

        input_product = str(args.input_product).lower()
        if input_product == "updc" and not args.skip_download:
            raise SystemExit(
                "--input-product updc expects corrected exposures already produced "
                "by myscripts; pass --skip-download."
            )
        subgroup = "CRF" if input_product == "crf" else "CAL"

        if not args.skip_download:
            _download_all(
                bands=bands,
                module=str(args.module),
                jwst_dir=jwst_dir,
                updec_dir=updec_dir,
                subgroup=subgroup,
                overwrite=bool(args.overwrite),
                updec_attempts=int(args.updec_attempts),
                updec_timeout_s=float(args.updec_timeout_s),
                updec_backoff_s=float(args.updec_backoff_s),
            )

        summary: dict[str, list[dict[str, Any]]] = {"coadd": [], "train": []}
        if not args.skip_coadd:
            for band in bands:
                summary["coadd"].append(
                    _coadd_full_band(
                        band=band,
                        module=str(args.module),
                        input_product=str(args.input_product),
                        jwst_dir=jwst_dir,
                        updec_dir=updec_dir,
                        mosaic_dir=mosaic_dir,
                        metric_dir=metric_dir,
                        background_offset_mode=str(args.background_offset),
                        background_sigma=float(args.background_sigma),
                        background_maxiters=int(args.background_maxiters),
                        output_label=str(args.output_label),
                        device=args.device,
                    )
                )
        if not args.skip_train:
            for band in bands:
                summary["train"].append(
                    _fit_full_band(
                        band=band,
                        module=str(args.module),
                        input_product=str(args.input_product),
                        jwst_dir=jwst_dir,
                        updec_dir=updec_dir,
                        fit_dir=fit_dir,
                        metric_dir=metric_dir,
                        model_kind=str(args.train_model),
                        steps=int(args.train_steps),
                        runs=int(args.train_runs),
                    )
                )

        summary_path = metric_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {summary_path}")
        return 0
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
