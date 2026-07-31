#!/usr/bin/env python3
"""Run the faint-source pipeline with dense native residual FITS snapshots.

All training, sampling, PSF, rendering, and detector logic is inherited from
``smacs0723_wcs_aware_faint_source_pipeline.py``.  Only residual diagnostics
change: two deterministic representative exposures per band are evaluated on
all valid native pixels and saved as detector-shaped FITS images.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch
from astropy.io import fits
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import scripts.smacs0723_wcs_aware_faint_source_pipeline as pipeline  # noqa: E402

base = pipeline.base
training_mod = pipeline.training_mod

_REPRESENTATIVES_PER_BAND = 2
_DQ_MODE = "strict"
_REPRESENTATIVES: list[tuple[int, int]] | None = None
_POOL_SIGNATURE: tuple[tuple[str, str, str], ...] | None = None
_ORIGINAL_SAVE_PIPELINE_CHECKPOINT = base.save_pipeline_checkpoint


def _diagnostic_metadata() -> dict[str, Any]:
    options = pipeline._OPTIONS
    return {
        "enabled": bool(options is not None and options.enabled),
        "mode": "dense_native_frame_fits",
        "directory": (
            None if pipeline._RESIDUAL_ROOT is None else str(pipeline._RESIDUAL_ROOT)
        ),
        "index": (
            None
            if pipeline._RESIDUAL_ROOT is None
            else str(pipeline._RESIDUAL_ROOT / "index.json")
        ),
        "every_global_steps": None if options is None else options.every_steps,
        "representative_exposures_per_band": _REPRESENTATIVES_PER_BAND,
        "selection": "path-sorted one-third and two-thirds exposures per band",
        "full_native_valid_pixels": True,
        "dq_mode": _DQ_MODE,
        "batch_pixels": None if options is None else options.batch_pixels,
        "residual_sign": "prediction_minus_native_observation",
        "snapshots": list(pipeline._SNAPSHOT_RECORDS),
    }


def _parse_residual_options(
    argv: Sequence[str],
) -> tuple[pipeline.ResidualOptions, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--save-residual-every-steps", type=int, default=0)
    parser.add_argument("--residual-batch-pixels", type=int, default=4096)
    # Backward-compatible consumption of the old sparse-residual option.
    parser.add_argument(
        "--residual-samples-per-exposure",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    namespace, remaining = parser.parse_known_args(list(argv))
    if namespace.save_residual_every_steps < 0:
        raise ValueError("save-residual-every-steps must be non-negative")
    if namespace.residual_batch_pixels <= 0:
        raise ValueError("residual-batch-pixels must be positive")
    return (
        pipeline.ResidualOptions(
            every_steps=namespace.save_residual_every_steps,
            samples_per_exposure=_REPRESENTATIVES_PER_BAND,
            batch_pixels=namespace.residual_batch_pixels,
        ),
        remaining,
    )


def _write_residual_index() -> None:
    options = pipeline._OPTIONS
    root = pipeline._RESIDUAL_ROOT
    if options is None or not options.enabled or root is None:
        return
    pipeline._ORIGINAL_WRITE_JSON(
        root / "index.json",
        {
            "definition": (
                "FITS SCI = model prediction - native observation in physical "
                "SCI units; invalid and geometry-invalid pixels are NaN."
            ),
            "frequency": {
                "unit": "global_optimizer_step",
                "every_steps": options.every_steps,
            },
            **_diagnostic_metadata(),
        },
    )


def _patched_write_json(path: Path, payload: dict[str, Any]) -> None:
    output = payload
    if path.name in {"pipeline_manifest.json", "pipeline_summary.json"}:
        output = copy.deepcopy(payload)
        output["native_residual_diagnostics"] = _diagnostic_metadata()
    pipeline._ORIGINAL_WRITE_JSON(path, output)


def _patched_save_pipeline_checkpoint(*args: Any, **kwargs: Any) -> Any:
    configuration = kwargs.get("configuration")
    if isinstance(configuration, dict):
        configuration["native_residual_diagnostics"] = _diagnostic_metadata()
    return _ORIGINAL_SAVE_PIPELINE_CHECKPOINT(*args, **kwargs)


def _representative_exposures(pools: Sequence[Any]) -> list[tuple[int, int]]:
    """Return fixed ``(exposure_index, rank_within_band)`` pairs."""

    global _REPRESENTATIVES, _POOL_SIGNATURE
    signature = tuple(
        (str(pool.path), str(pool.band), str(pool.detector)) for pool in pools
    )
    if _REPRESENTATIVES is not None:
        if signature != _POOL_SIGNATURE:
            raise RuntimeError("cached representative exposures do not match pools")
        return _REPRESENTATIVES

    grouped: dict[str, list[int]] = {}
    for index, pool in enumerate(pools):
        grouped.setdefault(str(pool.band), []).append(index)

    selected: list[tuple[int, int]] = []
    for band in sorted(grouped):
        candidates = sorted(grouped[band], key=lambda i: str(pools[i].path))
        count = min(_REPRESENTATIVES_PER_BAND, len(candidates))
        positions = [
            min(len(candidates) - 1, int((rank + 1) * len(candidates) / (count + 1)))
            for rank in range(count)
        ]
        if len(set(positions)) != count:
            positions = list(range(count))
        selected.extend(
            (candidates[position], rank)
            for rank, position in enumerate(positions, start=1)
        )

    _POOL_SIGNATURE = signature
    _REPRESENTATIVES = selected
    return selected


def _load_native_frame(
    pool: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header, fits.Header]:
    with fits.open(pool.path, memmap=False) as hdul:
        if "SCI" not in hdul or "ERR" not in hdul:
            raise ValueError(f"{pool.path} must contain SCI and ERR")
        image = np.array(hdul["SCI"].data, dtype=np.float32, copy=True)
        err = np.array(hdul["ERR"].data, dtype=np.float32, copy=True)
        if image.ndim != 2 or err.shape != image.shape:
            raise ValueError(f"invalid SCI/ERR shapes in {pool.path}")
        if tuple(image.shape) != tuple(pool.shape):
            raise ValueError(f"native shape changed for {pool.path}")
        valid = training_mod._valid_pixel_mask(
            hdul,
            image,
            err,
            dq_mode=_DQ_MODE,
        )
        primary_header = hdul[0].header.copy()
        science_header = hdul["SCI"].header.copy()
    return image, err, valid, primary_header, science_header


def _moments(
    count: int, total: float, square: float, absolute: float
) -> dict[str, Any]:
    if count == 0:
        return {"valid_pixels": 0}
    mean = total / count
    variance = max(square / count - mean * mean, 0.0)
    return {
        "valid_pixels": int(count),
        "residual_mean": float(mean),
        "residual_std": float(np.sqrt(variance)),
        "rmse": float(np.sqrt(square / count)),
        "mae": float(absolute / count),
    }


def _write_residual_fits(
    output: Path,
    residual: np.ndarray,
    primary_header: fits.Header,
    science_header: fits.Header,
    *,
    pool: Any,
    exposure_index: int,
    stage: str,
    global_step: int,
    operator: str,
    native_valid: int,
    geometry_valid: int,
) -> None:
    primary = primary_header.copy()
    primary["FILENAME"] = output.name
    primary.add_history("SCI contains model prediction minus native observation.")

    header = science_header.copy()
    header["EXTNAME"] = "SCI"
    header["BTYPE"] = "PREDICTION_MINUS_OBSERVATION"
    header["RESIDSIG"] = ("PRED-OBS", "Residual sign")
    header["MODELSTG"] = (stage, "Snapshot training stage")
    header["GLOBSTEP"] = (int(global_step), "Global optimizer step")
    header["EXPIDX"] = (int(exposure_index), "Model exposure index")
    header["NATVALID"] = (int(native_valid), "Native mask-valid pixels")
    header["GEOMVAL"] = (int(geometry_valid), "Finite predicted pixels")
    header["OPERATOR"] = (operator, "Observation operator")
    header["SRCFILE"] = (Path(pool.path).name, "Native source filename")

    fits.HDUList(
        [
            fits.PrimaryHDU(header=primary),
            fits.ImageHDU(residual.astype(np.float32, copy=False), header, name="SCI"),
        ]
    ).writeto(output, overwrite=False, checksum=True)


def _save_dense_snapshot(
    *,
    model: Any,
    pools: Sequence[Any],
    wavelength: Tensor,
    weight: Tensor,
    device: torch.device,
    intensity_scale: float,
    seed: int,
    stage: str,
    global_step: int,
    core_wing_plan: base.CoreWingPlan | None,
    device_core_wing: base.DeviceCoreWingPlan | None,
    wing_samples: int,
    offset_chunk: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    options = pipeline._OPTIONS
    root = pipeline._RESIDUAL_ROOT
    if options is None or not options.enabled or root is None:
        raise RuntimeError("dense residual diagnostics were not initialized")
    if stage == "finetune" and (
        core_wing_plan is None or device_core_wing is None
    ):
        raise ValueError("fine-tune residuals require a core/wing PSF plan")

    name = f"step_{global_step:06d}_{stage}"
    directory = root / name
    directory.mkdir(parents=True, exist_ok=False)
    representatives = _representative_exposures(pools)
    wing_generator = torch.Generator(device="cpu").manual_seed(seed + 1707)
    wavelengths = wavelength.to(device)
    weights = weight.to(device)
    operator = (
        "bandpass_only_no_PSF"
        if stage == "pretrain"
        else "exact_core_plus_fixed_seed_sampled_wings"
    )

    products: list[dict[str, Any]] = []
    aggregate = [0, 0.0, 0.0, 0.0]
    was_training = bool(model.training)
    model.eval()

    try:
        with torch.inference_mode():
            for exposure_index, rank in representatives:
                pool = pools[exposure_index]
                image, err_image, mask, primary, science = _load_native_frame(pool)
                native_indices = np.flatnonzero(mask.reshape(-1))
                residual = np.full(image.shape, np.nan, dtype=np.float32)
                image_flat = image.reshape(-1)
                err_flat = err_image.reshape(-1)
                residual_flat = residual.reshape(-1)
                stats = [0, 0.0, 0.0, 0.0]

                for start in range(0, native_indices.size, options.batch_pixels):
                    flat = native_indices[start : start + options.batch_pixels]
                    y, x = np.unravel_index(flat, image.shape)
                    uv = torch.from_numpy(
                        np.stack((x, y), axis=-1).astype(np.float32)
                    ).to(device)
                    observed = (
                        torch.from_numpy(image_flat[flat].copy()).to(device)
                        / intensity_scale
                    )
                    err = (
                        torch.from_numpy(err_flat[flat].copy()).to(device)
                        / intensity_scale
                    )
                    exposure_id = torch.full(
                        (flat.size,), exposure_index, dtype=torch.long, device=device
                    )

                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        if stage == "pretrain":
                            prediction, geometry = model.predict_observation(
                                uv, exposure_id, wavelengths, weights, psf=None
                            )
                        else:
                            prediction, geometry = base.predict_core_wing_observation(
                                model,
                                uv,
                                exposure_id,
                                wavelengths,
                                weights,
                                plan=core_wing_plan,
                                device_plan=device_core_wing,
                                wing_samples=wing_samples,
                                wing_generator=wing_generator,
                                offset_chunk=offset_chunk,
                                checkpoint_chunks=False,
                            )

                    valid = (
                        geometry
                        & torch.isfinite(prediction)
                        & torch.isfinite(observed)
                        & torch.isfinite(err)
                        & (err > 0)
                    )
                    if not bool(valid.any()):
                        continue

                    valid_cpu = valid.detach().cpu().numpy().astype(bool, copy=False)
                    values = (
                        (prediction[valid] - observed[valid])
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        * intensity_scale
                    ).astype(np.float32, copy=False)
                    residual_flat[flat[valid_cpu]] = values
                    values64 = values.astype(np.float64, copy=False)
                    chunk_stats = (
                        int(values.size),
                        float(values64.sum()),
                        float(np.square(values64).sum()),
                        float(np.abs(values64).sum()),
                    )
                    for index, value in enumerate(chunk_stats):
                        stats[index] += value
                        aggregate[index] += value

                output = directory / (
                    f"{exposure_index:03d}_{pool.band}_{pool.detector}_"
                    f"{Path(pool.path).stem}_native_residual.fits"
                )
                _write_residual_fits(
                    output,
                    residual,
                    primary,
                    science,
                    pool=pool,
                    exposure_index=exposure_index,
                    stage=stage,
                    global_step=global_step,
                    operator=operator,
                    native_valid=int(native_indices.size),
                    geometry_valid=int(stats[0]),
                )
                products.append(
                    {
                        "exposure_index": int(exposure_index),
                        "within_band_rank": int(rank),
                        "source": str(pool.path),
                        "band": str(pool.band),
                        "detector": str(pool.detector),
                        "shape": list(image.shape),
                        "native_valid_pixels": int(native_indices.size),
                        "metrics": _moments(*stats),
                        "fits": str(output),
                    }
                )
                del image, err_image, mask, residual
    finally:
        if was_training:
            model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "snapshot_name": name,
        "stage": stage,
        "global_step": int(global_step),
        "saved_utc": base.now_utc(),
        "mode": "dense_native_frame_fits",
        "residual_sign": "prediction_minus_native_observation",
        "physical_scale_applied": float(intensity_scale),
        "operator": operator,
        "dq_mode": _DQ_MODE,
        "diagnostic_batch_pixels": int(options.batch_pixels),
        "diagnostic_wing_seed": seed + 1707 if stage == "finetune" else None,
        "aggregate": _moments(*aggregate),
        "products": products,
        "directory": str(directory),
    }
    metadata_path = directory / "native_residual_snapshot.json"
    metadata["metadata"] = str(metadata_path)
    pipeline._ORIGINAL_WRITE_JSON(metadata_path, metadata)
    pipeline._SNAPSHOT_RECORDS.append(metadata)
    _write_residual_index()
    print(
        json.dumps(
            {
                "native_residual_snapshot": name,
                "fits_count": len(products),
                "aggregate": metadata["aggregate"],
                "directory": str(directory),
            },
            indent=2,
        ),
        flush=True,
    )
    return metadata


def _read_dq_mode(argv: Sequence[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    namespace, _ = parser.parse_known_args(list(argv))
    payload = json.loads(
        namespace.baseline_manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    mode = str(payload.get("args", {}).get("dq_mode", "strict"))
    if mode not in {"strict", "do_not_use"}:
        raise ValueError(f"unsupported baseline dq_mode {mode!r}")
    return mode


def main() -> int:
    global _DQ_MODE
    _DQ_MODE = _read_dq_mode(sys.argv[1:])
    pipeline.parse_residual_options = _parse_residual_options
    pipeline.write_residual_index = _write_residual_index
    pipeline.patched_write_json = _patched_write_json
    pipeline.save_native_residual_snapshot = _save_dense_snapshot
    base.save_pipeline_checkpoint = _patched_save_pipeline_checkpoint
    return int(pipeline.main())


if __name__ == "__main__":
    raise SystemExit(main())
