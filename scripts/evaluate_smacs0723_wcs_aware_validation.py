#!/usr/bin/env python3
"""Evaluate a trained WCS-aware SMACS J0723 INR on fixed held-out pixels.

The training loader first draws a fixed pool of at most
``max_pixels_per_exposure`` valid pixels from every exposure. This evaluator
replays that exact draw from the checkpoint seed, then samples validation
pixels only from the complementary valid-pixel set. The resulting pixels were
not members of the training pools used by the completed run.

Outputs include overall, per-band, and per-exposure residual summaries;
absolute whitened-residual quantiles; fractions within 1, 2, 3, and 5 sigma;
and an optional compressed residual sample file for later plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchsky.map.wcs_aware_inr import (  # noqa: E402
    UnionCanvas,
    WCSAwareMultiWavelengthINR,
)

ABS_Z_QUANTILES = (0.50, 0.68, 0.90, 0.95, 0.99)
SIGMA_THRESHOLDS = (1.0, 2.0, 3.0, 5.0)


def _canvas_from_checkpoint(saved: dict[str, Any]) -> UnionCanvas:
    header = dict(saved["union_header"])
    shape = tuple(int(value) for value in saved["union_shape"])
    cd = np.array(
        [
            [float(header["CD1_1"]), float(header.get("CD1_2", 0.0))],
            [float(header.get("CD2_1", 0.0)), float(header["CD2_2"])],
        ],
        dtype=np.float64,
    )
    determinant = float(np.linalg.det(cd))
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("checkpoint union WCS has a singular CD matrix")
    return UnionCanvas(
        header=header,
        shape=(shape[0], shape[1]),
        pixel_scale_arcsec=math.sqrt(abs(determinant)) * 3600.0,
    )


def _read_headers(paths: Sequence[str]) -> list[dict[str, Any]]:
    headers: list[dict[str, Any]] = []
    for text in paths:
        path = Path(text)
        with fits.open(path, memmap=False) as hdul:
            if "SCI" not in hdul:
                raise ValueError(f"{path} has no SCI extension")
            image = hdul["SCI"].data
            if image is None or image.ndim != 2:
                raise ValueError(f"{path} has an invalid SCI array")
            header = dict(hdul["SCI"].header)
            header["NAXIS1"] = int(image.shape[1])
            header["NAXIS2"] = int(image.shape[0])
            headers.append(header)
    return headers


def _build_model(
    saved: dict[str, Any],
    headers: Sequence[dict[str, Any]],
    canvas: UnionCanvas,
) -> WCSAwareMultiWavelengthINR:
    args = dict(saved.get("args", {}))
    checkpoint_path = str(saved.get("checkpoint_path", ""))
    encoding = str(args.get("position_encoding", "fourier"))
    if checkpoint_path:
        parent = Path(checkpoint_path).parent.name.lower()
        if parent in {"fourier", "spherical"}:
            encoding = parent
    model = WCSAwareMultiWavelengthINR(
        headers,
        canvas,
        encoding=encoding,
        spatial_frequencies=int(args.get("spatial_frequencies", 10)),
        spherical_min_frequency=float(args.get("spherical_min_frequency", 1.0)),
        spherical_alignment_rank=int(args.get("spherical_alignment_rank", 8)),
        spectral_frequencies=int(args.get("spectral_frequencies", 6)),
        hidden_dim=int(args.get("hidden_dim", 256)),
        latent_dim=int(args.get("latent_dim", 128)),
        trunk_depth=int(args.get("trunk_depth", 4)),
        decoder_depth=int(args.get("decoder_depth", 3)),
        anchor_exposure=int(args.get("anchor_exposure", 0)),
        learn_background=not bool(args.get("no_background", False)),
        output_activation=str(args.get("output_activation", "softplus")),
    )
    model.load_state_dict(saved["model"], strict=True)
    return model


def _valid_arrays(
    path: Path,
    *,
    dq_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(path, memmap=False) as hdul:
        if "SCI" not in hdul or "ERR" not in hdul:
            raise ValueError(f"{path} must contain SCI and ERR")
        image = np.asarray(hdul["SCI"].data, dtype=np.float32)
        err = np.asarray(hdul["ERR"].data, dtype=np.float32)
        if image.ndim != 2 or err.shape != image.shape:
            raise ValueError(f"invalid SCI/ERR shapes in {path}")
        valid = np.isfinite(image) & np.isfinite(err) & (err > 0.0)
        if "DQ" in hdul:
            dq = np.asarray(hdul["DQ"].data)
            if dq.shape != image.shape:
                raise ValueError(f"DQ shape does not match SCI in {path}")
            if dq_mode == "strict":
                valid &= dq == 0
            elif dq_mode == "do_not_use":
                valid &= (dq.astype(np.uint64) & np.uint64(1)) == 0
            else:
                raise ValueError("dq_mode must be strict or do_not_use")
        if "UPDCMASK" in hdul:
            extra = np.asarray(hdul["UPDCMASK"].data)
            if extra.shape != image.shape:
                raise ValueError(f"UPDCMASK shape does not match SCI in {path}")
            valid &= np.isfinite(extra) & (extra == 0)
    return image, err, np.flatnonzero(valid.reshape(-1))


def _replay_training_pool(
    valid_flat: np.ndarray,
    *,
    max_pixels: int,
    training_rng: np.random.Generator,
) -> np.ndarray:
    if max_pixels > 0 and valid_flat.size > max_pixels:
        return np.asarray(
            training_rng.choice(valid_flat, size=max_pixels, replace=False),
            dtype=np.int64,
        )
    return np.asarray(valid_flat, dtype=np.int64)


def _held_out_indices(
    valid_flat: np.ndarray,
    training_flat: np.ndarray,
    *,
    count: int,
    seed: int,
    exposure_index: int,
    total_pixels: int,
) -> tuple[np.ndarray, int]:
    in_training = np.zeros(total_pixels, dtype=bool)
    in_training[training_flat] = True
    candidates = valid_flat[~in_training[valid_flat]]
    candidate_count = int(candidates.size)
    if candidate_count == 0:
        raise RuntimeError(
            "no held-out pixels remain; the training pool used every valid pixel"
        )
    n = candidate_count if count <= 0 else min(int(count), candidate_count)
    validation_rng = np.random.default_rng(int(seed) + 104729 * int(exposure_index))
    chosen = (
        validation_rng.choice(candidates, size=n, replace=False)
        if n < candidate_count
        else candidates
    )
    return np.asarray(chosen, dtype=np.int64), candidate_count


def _huber_mean(z: np.ndarray, delta: float) -> float:
    absolute = np.abs(z.astype(np.float64, copy=False))
    rho = np.where(
        absolute <= delta,
        0.5 * absolute * absolute,
        delta * (absolute - 0.5 * delta),
    )
    return float(rho.mean())


def _summarize(
    residual_normalized: np.ndarray,
    z: np.ndarray,
    *,
    intensity_scale: float,
    delta: float,
) -> dict[str, float | int]:
    residual = residual_normalized.astype(np.float64, copy=False)
    whitened = z.astype(np.float64, copy=False)
    absolute_z = np.abs(whitened)
    output: dict[str, float | int] = {
        "count": int(residual.size),
        "huber_loss": _huber_mean(whitened, delta),
        "raw_bias_normalized": float(residual.mean()),
        "raw_mae_normalized": float(np.abs(residual).mean()),
        "raw_rmse_normalized": float(np.sqrt(np.mean(residual * residual))),
        "raw_bias_physical": float(residual.mean() * intensity_scale),
        "raw_mae_physical": float(np.abs(residual).mean() * intensity_scale),
        "raw_rmse_physical": float(
            np.sqrt(np.mean(residual * residual)) * intensity_scale
        ),
        "whitened_bias": float(whitened.mean()),
        "whitened_mae": float(absolute_z.mean()),
        "whitened_rmse": float(np.sqrt(np.mean(whitened * whitened))),
    }
    for quantile in ABS_Z_QUANTILES:
        key = f"abs_z_q{int(round(100 * quantile)):02d}"
        output[key] = float(np.quantile(absolute_z, quantile))
    for threshold in SIGMA_THRESHOLDS:
        key = f"fraction_abs_z_le_{int(threshold)}"
        output[key] = float(np.mean(absolute_z <= threshold))
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _concat(items: Iterable[np.ndarray]) -> np.ndarray:
    arrays = list(items)
    if not arrays:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(arrays)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-pixels-per-exposure", type=int, default=5000)
    parser.add_argument("--validation-seed", type=int, default=20260717)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--no-save-residual-samples", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent / "validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = dict(saved.get("args", {}))
    psf_mode = str(checkpoint_args.get("psf_mode", "none"))
    if psf_mode != "none":
        raise SystemExit(
            "this evaluator currently supports no-PSF checkpoints only; "
            f"checkpoint psf_mode={psf_mode!r}"
        )

    exposure_paths = [str(value) for value in saved["exposure_paths"]]
    exposure_bands = [str(value).upper() for value in saved["exposure_bands"]]
    if len(exposure_paths) != len(exposure_bands):
        raise ValueError("checkpoint exposure path/band counts differ")

    headers = _read_headers(exposure_paths)
    canvas = _canvas_from_checkpoint(saved)
    saved["checkpoint_path"] = str(checkpoint_path)
    model = _build_model(saved, headers, canvas).to(device).eval()
    wavelengths = torch.as_tensor(saved["bandpass_wavelength"]).float().to(device)
    weights = torch.as_tensor(saved["bandpass_weight"]).float().to(device)

    intensity_scale = float(saved["intensity_scale"])
    training_seed = int(checkpoint_args.get("seed", 17))
    max_training_pixels = int(
        checkpoint_args.get("max_pixels_per_exposure", 200000)
    )
    dq_mode = str(checkpoint_args.get("dq_mode", "strict"))
    delta = float(checkpoint_args.get("robust_delta", 3.0))
    training_rng = np.random.default_rng(training_seed)

    per_exposure_rows: list[dict[str, Any]] = []
    residuals_by_band: dict[str, list[np.ndarray]] = defaultdict(list)
    z_by_band: dict[str, list[np.ndarray]] = defaultdict(list)
    all_residuals: list[np.ndarray] = []
    all_z: list[np.ndarray] = []

    sample_exposure: list[np.ndarray] = []
    sample_band: list[np.ndarray] = []
    sample_x: list[np.ndarray] = []
    sample_y: list[np.ndarray] = []
    sample_target: list[np.ndarray] = []
    sample_prediction: list[np.ndarray] = []
    sample_residual: list[np.ndarray] = []
    sample_z: list[np.ndarray] = []
    band_names = list(dict.fromkeys(exposure_bands))
    band_to_id = {band: index for index, band in enumerate(band_names)}

    with torch.inference_mode():
        for exposure_index, (path_text, band) in enumerate(
            zip(exposure_paths, exposure_bands)
        ):
            path = Path(path_text)
            image, err, valid_flat = _valid_arrays(path, dq_mode=dq_mode)
            training_flat = _replay_training_pool(
                valid_flat,
                max_pixels=max_training_pixels,
                training_rng=training_rng,
            )
            validation_flat, candidate_count = _held_out_indices(
                valid_flat,
                training_flat,
                count=args.validation_pixels_per_exposure,
                seed=args.validation_seed,
                exposure_index=exposure_index,
                total_pixels=image.size,
            )
            y, x = np.unravel_index(validation_flat, image.shape)
            target_normalized = (
                image.reshape(-1)[validation_flat].astype(np.float32) / intensity_scale
            )
            err_normalized = (
                err.reshape(-1)[validation_flat].astype(np.float32) / intensity_scale
            )
            uv = np.stack((x, y), axis=-1).astype(np.float32)

            predictions: list[np.ndarray] = []
            geometry_valid: list[np.ndarray] = []
            for start in range(0, validation_flat.size, args.batch_size):
                stop = min(start + args.batch_size, validation_flat.size)
                uv_batch = torch.from_numpy(uv[start:stop]).to(device)
                exposure_batch = torch.full(
                    (stop - start,),
                    exposure_index,
                    dtype=torch.long,
                    device=device,
                )
                prediction, valid_geometry = model.predict_observation(
                    uv_batch,
                    exposure_batch,
                    wavelengths,
                    weights,
                    psf=None,
                )
                predictions.append(prediction.float().cpu().numpy())
                geometry_valid.append(valid_geometry.cpu().numpy().astype(bool))

            prediction_normalized = np.concatenate(predictions)
            valid_geometry = np.concatenate(geometry_valid)
            invalid_geometry_count = int((~valid_geometry).sum())
            if not bool(valid_geometry.any()):
                raise RuntimeError(f"all held-out pixels failed WCS geometry for {path}")

            x = x[valid_geometry]
            y = y[valid_geometry]
            target_normalized = target_normalized[valid_geometry]
            err_normalized = err_normalized[valid_geometry]
            prediction_normalized = prediction_normalized[valid_geometry]
            residual_normalized = prediction_normalized - target_normalized
            z = residual_normalized / err_normalized
            finite = np.isfinite(residual_normalized) & np.isfinite(z)
            residual_normalized = residual_normalized[finite].astype(np.float32)
            z = z[finite].astype(np.float32)
            x = x[finite]
            y = y[finite]
            target_normalized = target_normalized[finite]
            prediction_normalized = prediction_normalized[finite]

            metrics = _summarize(
                residual_normalized,
                z,
                intensity_scale=intensity_scale,
                delta=delta,
            )
            row: dict[str, Any] = {
                "exposure_index": exposure_index,
                "band": band,
                "path": str(path),
                "valid_pixel_count": int(valid_flat.size),
                "training_pool_count": int(training_flat.size),
                "held_out_candidate_count": candidate_count,
                "requested_validation_count": int(validation_flat.size),
                "geometry_invalid_count": invalid_geometry_count,
                **metrics,
            }
            per_exposure_rows.append(row)
            residuals_by_band[band].append(residual_normalized)
            z_by_band[band].append(z)
            all_residuals.append(residual_normalized)
            all_z.append(z)

            if not args.no_save_residual_samples:
                sample_exposure.append(
                    np.full(z.size, exposure_index, dtype=np.int16)
                )
                sample_band.append(
                    np.full(z.size, band_to_id[band], dtype=np.uint8)
                )
                sample_x.append(x.astype(np.int32, copy=False))
                sample_y.append(y.astype(np.int32, copy=False))
                sample_target.append(
                    (target_normalized * intensity_scale).astype(np.float32)
                )
                sample_prediction.append(
                    (prediction_normalized * intensity_scale).astype(np.float32)
                )
                sample_residual.append(
                    (residual_normalized * intensity_scale).astype(np.float32)
                )
                sample_z.append(z)

            print(
                json.dumps(
                    {
                        "exposure": exposure_index,
                        "band": band,
                        "count": metrics["count"],
                        "huber_loss": metrics["huber_loss"],
                        "whitened_mae": metrics["whitened_mae"],
                        "whitened_rmse": metrics["whitened_rmse"],
                        "abs_z_q95": metrics["abs_z_q95"],
                    }
                ),
                flush=True,
            )

    per_band_rows: list[dict[str, Any]] = []
    per_band_json: dict[str, dict[str, float | int]] = {}
    for band in band_names:
        residual = _concat(residuals_by_band[band])
        z = _concat(z_by_band[band])
        metrics = _summarize(
            residual,
            z,
            intensity_scale=intensity_scale,
            delta=delta,
        )
        per_band_rows.append({"band": band, **metrics})
        per_band_json[band] = metrics

    overall = _summarize(
        _concat(all_residuals),
        _concat(all_z),
        intensity_scale=intensity_scale,
        delta=delta,
    )

    per_exposure_path = output_dir / "validation_per_exposure.csv"
    per_band_path = output_dir / "validation_per_band.csv"
    summary_path = output_dir / "validation_summary.json"
    residual_path = output_dir / "validation_residual_samples.npz"
    _write_csv(per_exposure_path, per_exposure_rows)
    _write_csv(per_band_path, per_band_rows)

    residual_file: str | None = None
    if not args.no_save_residual_samples:
        np.savez_compressed(
            residual_path,
            exposure_index=_concat(sample_exposure),
            band_id=_concat(sample_band),
            band_names=np.asarray(band_names),
            x=_concat(sample_x),
            y=_concat(sample_y),
            target_physical=_concat(sample_target),
            prediction_physical=_concat(sample_prediction),
            residual_physical=_concat(sample_residual),
            z=_concat(sample_z),
        )
        residual_file = str(residual_path)

    summary = {
        "format": "torchsky-wcs-aware-validation-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(saved.get("step", 0)),
        "validation_kind": (
            "deterministic held-out valid pixels from the complement of each "
            "exposure's replayed training pool"
        ),
        "interpretation_caveat": (
            "These pixels were excluded from the stochastic training pools, but "
            "they come from the same exposures and neighboring sky as training; "
            "this is a fixed residual evaluation, not a spatial-region holdout."
        ),
        "selection": {
            "training_seed": training_seed,
            "validation_seed": int(args.validation_seed),
            "max_training_pixels_per_exposure": max_training_pixels,
            "validation_pixels_per_exposure": int(
                args.validation_pixels_per_exposure
            ),
            "dq_mode": dq_mode,
            "robust_delta_sigma": delta,
            "exposure_count": len(exposure_paths),
            "bands": band_names,
        },
        "intensity_scale": intensity_scale,
        "overall": overall,
        "per_band": per_band_json,
        "outputs": {
            "per_exposure_csv": str(per_exposure_path),
            "per_band_csv": str(per_band_path),
            "residual_samples_npz": residual_file,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "overall": overall}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
