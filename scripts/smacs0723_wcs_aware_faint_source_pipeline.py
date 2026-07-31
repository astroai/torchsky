#!/usr/bin/env python3
"""Faint-source pipeline with optional fixed native-pixel residual snapshots.

This remains the public runnable entry point. The staged training/rendering
implementation lives in the adjacent private core module so residual diagnostics
can be added without maintaining a second runnable launcher.

Residual snapshots are disabled by default. Pass, for example,
``--save-residual-every-steps 2000`` to save comparable native-observation
residuals every 2000 global optimizer steps across stage 1 and stage 2.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from astropy.table import Table
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import scripts._smacs0723_wcs_aware_faint_source_pipeline_core as base  # noqa: E402
import torchsky.map.wcs_aware_training as training_mod  # noqa: E402


class ResidualOptions:
    def __init__(
        self,
        *,
        every_steps: int,
        samples_per_exposure: int,
        batch_pixels: int,
    ) -> None:
        self.every_steps = int(every_steps)
        self.samples_per_exposure = int(samples_per_exposure)
        self.batch_pixels = int(batch_pixels)

    @property
    def enabled(self) -> bool:
        return self.every_steps > 0


_OPTIONS: ResidualOptions | None = None
_RESIDUAL_ROOT: Path | None = None
_FIXED_SAMPLE_INDICES: list[Tensor] | None = None
_SNAPSHOT_RECORDS: list[dict[str, Any]] = []

_ORIGINAL_PREPARE_OUTPUT_DIRECTORY = base.prepare_output_directory
_ORIGINAL_WRITE_JSON = base.write_json


def residual_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--save-residual-every-steps",
        type=int,
        default=0,
        help=(
            "Save fixed native-pixel residual diagnostics every N global "
            "optimizer steps across both training stages. 0 disables; use "
            "2000 for snapshots at global steps 2000, 4000, ..."
        ),
    )
    parser.add_argument(
        "--residual-samples-per-exposure",
        type=int,
        default=512,
        help="Fixed native pixels per exposure used in every residual snapshot.",
    )
    parser.add_argument(
        "--residual-batch-pixels",
        type=int,
        default=256,
        help="Maximum diagnostic native pixels rendered at once.",
    )
    return parser


def parse_residual_options(argv: Sequence[str]) -> tuple[ResidualOptions, list[str]]:
    namespace, remaining = residual_parser().parse_known_args(list(argv))
    if namespace.save_residual_every_steps < 0:
        raise ValueError("save-residual-every-steps must be non-negative")
    if namespace.residual_samples_per_exposure <= 0:
        raise ValueError("residual-samples-per-exposure must be positive")
    if namespace.residual_batch_pixels <= 0:
        raise ValueError("residual-batch-pixels must be positive")
    return (
        ResidualOptions(
            every_steps=namespace.save_residual_every_steps,
            samples_per_exposure=namespace.residual_samples_per_exposure,
            batch_pixels=namespace.residual_batch_pixels,
        ),
        remaining,
    )


def write_residual_index() -> None:
    if _OPTIONS is None or not _OPTIONS.enabled or _RESIDUAL_ROOT is None:
        return
    _ORIGINAL_WRITE_JSON(
        _RESIDUAL_ROOT / "index.json",
        {
            "definition": (
                "Residual = model native-observation prediction minus the fixed "
                "native CRF/CAL SCI sample."
            ),
            "frequency": {
                "unit": "global_optimizer_step",
                "every_steps": _OPTIONS.every_steps,
            },
            "fixed_samples_per_exposure": _OPTIONS.samples_per_exposure,
            "diagnostic_batch_pixels": _OPTIONS.batch_pixels,
            "snapshots": _SNAPSHOT_RECORDS,
        },
    )


def prepare_output_directory(output_dir: Path, *, overwrite: bool) -> dict[str, Path]:
    global _RESIDUAL_ROOT
    paths = _ORIGINAL_PREPARE_OUTPUT_DIRECTORY(output_dir, overwrite=overwrite)
    if _OPTIONS is None or not _OPTIONS.enabled:
        return paths

    residual_root = output_dir / "native_residuals"
    if residual_root.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing native residuals: {residual_root}. "
            "Use a fresh output directory or pass --overwrite."
        )
    if overwrite and residual_root.exists():
        shutil.rmtree(residual_root)
    residual_root.mkdir(parents=True, exist_ok=False)
    paths["native_residuals"] = residual_root
    _RESIDUAL_ROOT = residual_root
    write_residual_index()
    return paths


def patched_write_json(path: Path, payload: dict[str, Any]) -> None:
    augmented = payload
    if path.name in {"pipeline_manifest.json", "pipeline_summary.json"}:
        augmented = copy.deepcopy(payload)
        augmented["native_residual_diagnostics"] = {
            "enabled": bool(_OPTIONS is not None and _OPTIONS.enabled),
            "directory": None if _RESIDUAL_ROOT is None else str(_RESIDUAL_ROOT),
            "index": (
                None
                if _RESIDUAL_ROOT is None
                else str(_RESIDUAL_ROOT / "index.json")
            ),
            "every_global_steps": (
                None if _OPTIONS is None else _OPTIONS.every_steps
            ),
            "samples_per_exposure": (
                None if _OPTIONS is None else _OPTIONS.samples_per_exposure
            ),
            "batch_pixels": None if _OPTIONS is None else _OPTIONS.batch_pixels,
            "fixed_sample_set": True,
            "snapshots": list(_SNAPSHOT_RECORDS),
        }
    _ORIGINAL_WRITE_JSON(path, augmented)


def fixed_sample_indices(
    pools: Sequence[Any],
    *,
    samples_per_exposure: int,
    seed: int,
) -> list[Tensor]:
    global _FIXED_SAMPLE_INDICES
    if _FIXED_SAMPLE_INDICES is not None:
        if len(_FIXED_SAMPLE_INDICES) != len(pools):
            raise RuntimeError("cached residual sample set does not match exposure count")
        return _FIXED_SAMPLE_INDICES

    rng = np.random.default_rng(seed + 909)
    indices: list[Tensor] = []
    for pool in pools:
        count = min(int(samples_per_exposure), int(pool.size))
        selected = rng.choice(pool.size, size=count, replace=False).astype(np.int64)
        indices.append(torch.from_numpy(selected))
    _FIXED_SAMPLE_INDICES = indices
    return indices


def aggregate_metrics(residual: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    if residual.size == 0:
        return {"valid_samples": 0}
    absolute_z = np.abs(z)
    return {
        "valid_samples": int(residual.size),
        "residual_mean": float(np.mean(residual)),
        "residual_median": float(np.median(residual)),
        "residual_std": float(np.std(residual)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "abs_z_q50": float(np.quantile(absolute_z, 0.50)),
        "abs_z_q68": float(np.quantile(absolute_z, 0.68)),
        "abs_z_q90": float(np.quantile(absolute_z, 0.90)),
        "abs_z_q95": float(np.quantile(absolute_z, 0.95)),
        "abs_z_q99": float(np.quantile(absolute_z, 0.99)),
    }


def save_native_residual_snapshot(
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
    if _OPTIONS is None or not _OPTIONS.enabled or _RESIDUAL_ROOT is None:
        raise RuntimeError("residual diagnostics were not initialized")
    if stage == "finetune" and (
        core_wing_plan is None or device_core_wing is None
    ):
        raise ValueError("fine-tune residuals require a core/wing PSF plan")

    snapshot_name = f"step_{global_step:06d}_{stage}"
    snapshot_dir = _RESIDUAL_ROOT / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    sample_indices = fixed_sample_indices(
        pools,
        samples_per_exposure=_OPTIONS.samples_per_exposure,
        seed=seed,
    )

    # Resetting this generator for every snapshot makes the stage-2 diagnostic
    # wing draws identical across snapshots, so differences reflect the model.
    diagnostic_wing_generator = torch.Generator(device="cpu").manual_seed(seed + 1707)
    wavelengths_device = wavelength.to(device)
    weights_device = weight.to(device)

    summary_rows: list[dict[str, Any]] = []
    sample_columns: dict[str, list[np.ndarray]] = {
        "exposure_index": [],
        "band": [],
        "detector": [],
        "pool_index": [],
        "u_native": [],
        "v_native": [],
        "observed": [],
        "prediction": [],
        "err": [],
        "residual_prediction_minus_observed": [],
        "z": [],
    }
    all_residual: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    was_training = bool(model.training)
    model.eval()
    operator = "bandpass_only_no_PSF"

    try:
        with torch.inference_mode():
            for exposure_index, (pool, selected) in enumerate(
                zip(pools, sample_indices, strict=True)
            ):
                residual_parts: list[np.ndarray] = []
                z_parts: list[np.ndarray] = []
                valid_count = 0

                for start in range(0, selected.numel(), _OPTIONS.batch_pixels):
                    stop = min(start + _OPTIONS.batch_pixels, selected.numel())
                    chosen = selected[start:stop]
                    uv = pool.uv[chosen].to(device)
                    observed = pool.value[chosen].to(device)
                    err = pool.err[chosen].to(device)
                    exposure_id = torch.full(
                        (chosen.numel(),),
                        exposure_index,
                        dtype=torch.long,
                        device=device,
                    )

                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        if stage == "pretrain":
                            prediction, geometry_valid = model.predict_observation(
                                uv,
                                exposure_id,
                                wavelengths_device,
                                weights_device,
                                psf=None,
                            )
                            operator = "bandpass_only_no_PSF"
                        elif stage == "finetune":
                            prediction, geometry_valid = base.predict_core_wing_observation(
                                model,
                                uv,
                                exposure_id,
                                wavelengths_device,
                                weights_device,
                                plan=core_wing_plan,
                                device_plan=device_core_wing,
                                wing_samples=wing_samples,
                                wing_generator=diagnostic_wing_generator,
                                offset_chunk=offset_chunk,
                                checkpoint_chunks=False,
                            )
                            operator = "exact_core_plus_fixed_seed_sampled_wings"
                        else:
                            raise ValueError(f"unsupported diagnostic stage {stage!r}")

                    valid = (
                        geometry_valid
                        & torch.isfinite(prediction)
                        & torch.isfinite(observed)
                        & torch.isfinite(err)
                        & (err > 0)
                    )
                    if not bool(valid.any()):
                        continue

                    valid_count += int(valid.sum())
                    residual_normalized = prediction[valid] - observed[valid]
                    z = residual_normalized / err[valid]
                    residual_physical = (
                        residual_normalized.detach().float().cpu().numpy()
                        * intensity_scale
                    )
                    z_numpy = z.detach().float().cpu().numpy()
                    prediction_physical = (
                        prediction[valid].detach().float().cpu().numpy()
                        * intensity_scale
                    )
                    observed_physical = (
                        observed[valid].detach().float().cpu().numpy()
                        * intensity_scale
                    )
                    err_physical = (
                        err[valid].detach().float().cpu().numpy()
                        * intensity_scale
                    )
                    uv_numpy = uv[valid].detach().float().cpu().numpy()
                    chosen_numpy = chosen[valid.detach().cpu()].numpy()
                    count = residual_physical.size

                    residual_parts.append(residual_physical)
                    z_parts.append(z_numpy)
                    all_residual.append(residual_physical)
                    all_z.append(z_numpy)
                    sample_columns["exposure_index"].append(
                        np.full(count, exposure_index, dtype=np.int32)
                    )
                    sample_columns["band"].append(
                        np.full(count, str(pool.band), dtype="U16")
                    )
                    sample_columns["detector"].append(
                        np.full(count, str(pool.detector), dtype="U32")
                    )
                    sample_columns["pool_index"].append(chosen_numpy.astype(np.int64))
                    sample_columns["u_native"].append(uv_numpy[:, 0])
                    sample_columns["v_native"].append(uv_numpy[:, 1])
                    sample_columns["observed"].append(observed_physical)
                    sample_columns["prediction"].append(prediction_physical)
                    sample_columns["err"].append(err_physical)
                    sample_columns["residual_prediction_minus_observed"].append(
                        residual_physical
                    )
                    sample_columns["z"].append(z_numpy)

                exposure_residual = (
                    np.concatenate(residual_parts)
                    if residual_parts
                    else np.empty(0, dtype=np.float32)
                )
                exposure_z = (
                    np.concatenate(z_parts)
                    if z_parts
                    else np.empty(0, dtype=np.float32)
                )
                summary_rows.append(
                    {
                        "exposure_index": exposure_index,
                        "path": str(pool.path),
                        "band": str(pool.band),
                        "detector": str(pool.detector),
                        "sampled": int(selected.numel()),
                        "valid": valid_count,
                        "valid_fraction": (
                            float(valid_count / selected.numel())
                            if selected.numel()
                            else 0.0
                        ),
                        **aggregate_metrics(exposure_residual, exposure_z),
                    }
                )
    finally:
        if was_training:
            model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sample_table_path = snapshot_dir / "native_residual_samples.ecsv"
    summary_table_path = snapshot_dir / "native_residual_summary.ecsv"
    if any(sample_columns.values()):
        table_payload = {
            name: np.concatenate(parts) if parts else np.array([])
            for name, parts in sample_columns.items()
        }
        Table(table_payload).write(
            sample_table_path,
            format="ascii.ecsv",
            overwrite=False,
        )
    else:
        Table().write(sample_table_path, format="ascii.ecsv", overwrite=False)
    Table(rows=summary_rows).write(
        summary_table_path,
        format="ascii.ecsv",
        overwrite=False,
    )

    aggregate_residual = (
        np.concatenate(all_residual)
        if all_residual
        else np.empty(0, dtype=np.float32)
    )
    aggregate_z = np.concatenate(all_z) if all_z else np.empty(0, dtype=np.float32)
    metadata = {
        "snapshot_name": snapshot_name,
        "stage": stage,
        "global_step": int(global_step),
        "saved_utc": base.now_utc(),
        "residual_sign": "prediction_minus_native_observation",
        "physical_scale_applied": float(intensity_scale),
        "operator": operator,
        "fixed_native_sample_set": True,
        "samples_per_exposure_requested": _OPTIONS.samples_per_exposure,
        "diagnostic_batch_pixels": _OPTIONS.batch_pixels,
        "diagnostic_sample_seed": seed + 909,
        "diagnostic_wing_seed": seed + 1707 if stage == "finetune" else None,
        "fine_tune_wing_samples": wing_samples if stage == "finetune" else 0,
        "fine_tune_offset_chunk": offset_chunk if stage == "finetune" else None,
        "aggregate": aggregate_metrics(aggregate_residual, aggregate_z),
        "sample_table": str(sample_table_path),
        "summary_table": str(summary_table_path),
        "directory": str(snapshot_dir),
    }
    metadata_path = snapshot_dir / "native_residual_snapshot.json"
    metadata["metadata"] = str(metadata_path)
    _ORIGINAL_WRITE_JSON(metadata_path, metadata)
    _SNAPSHOT_RECORDS.append(metadata)
    write_residual_index()
    print(
        json.dumps(
            {
                "native_residual_snapshot": snapshot_name,
                "stage": stage,
                "global_step": global_step,
                "aggregate": metadata["aggregate"],
                "directory": str(snapshot_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return metadata


def run_training_stage(
    *,
    stage: str,
    model: Any,
    pools: Sequence[Any],
    wavelength: Tensor,
    weight: Tensor,
    device: torch.device,
    steps: int,
    global_step_start: int,
    batch_size: int | Sequence[int],
    data_generator: torch.Generator,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    amp_enabled: bool,
    gradient_clip: float,
    robust_loss: str,
    robust_delta: float,
    max_relative_weight: float,
    alignment_regularization: float,
    background_regularization: float,
    log_every: int,
    checkpoint_every: int,
    checkpoint_path: Path,
    intensity_scale: float,
    canvas: Any,
    checkpoint_configuration: dict[str, Any],
    loss_handle: Any,
    core_wing_plan: base.CoreWingPlan | None = None,
    device_core_wing: base.DeviceCoreWingPlan | None = None,
    wing_samples: int = 0,
    wing_generator: torch.Generator | None = None,
    offset_chunk: int = 1,
    checkpoint_chunks: bool = True,
) -> base.TrainingStageResult:
    if steps == 0:
        return base.TrainingStageResult(
            stage=stage,
            steps=0,
            first_global_step=global_step_start,
            last_global_step=global_step_start,
            seconds=0.0,
            records=[],
            checkpoint=str(checkpoint_path),
        )
    if stage == "finetune" and (
        core_wing_plan is None or device_core_wing is None or wing_generator is None
    ):
        raise ValueError("fine-tuning requires a core/wing plan and RNG")

    if _OPTIONS is not None and _OPTIONS.enabled:
        checkpoint_configuration["native_residual_diagnostics"] = {
            "enabled": True,
            "directory": None if _RESIDUAL_ROOT is None else str(_RESIDUAL_ROOT),
            "every_global_steps": _OPTIONS.every_steps,
            "samples_per_exposure": _OPTIONS.samples_per_exposure,
            "batch_pixels": _OPTIONS.batch_pixels,
            "fixed_sample_seed": int(checkpoint_configuration["seed"]) + 909,
            "fixed_finetune_wing_seed": int(checkpoint_configuration["seed"]) + 1707,
        }

    model.train()
    model.to(device)
    wavelengths_device = wavelength.to(device=device)
    weights_device = weight.to(device=device)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    seed = int(checkpoint_configuration["seed"])

    for stage_step in range(1, steps + 1):
        global_step = global_step_start + stage_step
        uv, exposure_id, target, err, mask = training_mod.sample_balanced_batch(
            pools,
            batch_size=batch_size,
            generator=data_generator,
        )
        uv = uv.to(device)
        exposure_id = exposure_id.to(device)
        target = target.to(device)
        err = err.to(device)
        mask = mask.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            if stage == "pretrain":
                prediction, geometry_valid = model.predict_observation(
                    uv,
                    exposure_id,
                    wavelengths_device,
                    weights_device,
                    psf=None,
                )
            elif stage == "finetune":
                prediction, geometry_valid = base.predict_core_wing_observation(
                    model,
                    uv,
                    exposure_id,
                    wavelengths_device,
                    weights_device,
                    plan=core_wing_plan,
                    device_plan=device_core_wing,
                    wing_samples=wing_samples,
                    wing_generator=wing_generator,
                    offset_chunk=offset_chunk,
                    checkpoint_chunks=checkpoint_chunks,
                )
            else:
                raise ValueError(f"unsupported training stage {stage!r}")

            data_loss, diagnostics = base.weighted_robust_loss(
                prediction,
                target,
                err,
                mask & geometry_valid,
                kind=robust_loss,
                delta=robust_delta,
                max_relative_weight=max_relative_weight,
            )
            alignment_reg = model.alignment_regularization()
            background_reg = model.background.square().mean()
            loss = (
                data_loss
                + alignment_regularization * alignment_reg
                + background_regularization * background_reg
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        step_loss = float(loss.detach())
        loss_row = {
            "stage": stage,
            "stage_step": stage_step,
            "global_step": global_step,
            "loss": step_loss,
        }
        loss_handle.write(json.dumps(loss_row) + "\n")
        if stage_step % min(log_every, 100) == 0 or stage_step == steps:
            loss_handle.flush()

        if stage_step == 1 or stage_step % log_every == 0 or stage_step == steps:
            record = {
                **loss_row,
                "data_loss": float(data_loss.detach()),
                "rmse": float(diagnostics["rmse"]),
                "mae": float(diagnostics["mae"]),
                "whitened_rmse": float(
                    diagnostics.get("whitened_rmse", torch.tensor(float("nan")))
                ),
                "whitened_mae": float(
                    diagnostics.get("whitened_mae", torch.tensor(float("nan")))
                ),
                "valid_fraction": float(diagnostics["valid_fraction"]),
                "alignment_regularization": float(alignment_reg.detach()),
                "background_regularization": float(background_reg.detach()),
                "seconds": time.perf_counter() - started,
                "learning_rates": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
            }
            records.append(record)
            print(json.dumps(record), flush=True)

        if checkpoint_every > 0 and stage_step % checkpoint_every == 0:
            base.save_pipeline_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                stage=stage,
                stage_step=stage_step,
                global_step=global_step,
                intensity_scale=intensity_scale,
                canvas=canvas,
                pools=pools,
                wavelength=wavelength,
                weight=weight,
                configuration=checkpoint_configuration,
                records=records,
            )

        if (
            _OPTIONS is not None
            and _OPTIONS.enabled
            and global_step % _OPTIONS.every_steps == 0
        ):
            save_native_residual_snapshot(
                model=model,
                pools=pools,
                wavelength=wavelength,
                weight=weight,
                device=device,
                intensity_scale=intensity_scale,
                seed=seed,
                stage=stage,
                global_step=global_step,
                core_wing_plan=core_wing_plan,
                device_core_wing=device_core_wing,
                wing_samples=wing_samples,
                offset_chunk=offset_chunk,
                amp_enabled=amp_enabled,
            )

    final_global_step = global_step_start + steps
    base.save_pipeline_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        stage=stage,
        stage_step=steps,
        global_step=final_global_step,
        intensity_scale=intensity_scale,
        canvas=canvas,
        pools=pools,
        wavelength=wavelength,
        weight=weight,
        configuration=checkpoint_configuration,
        records=records,
    )
    return base.TrainingStageResult(
        stage=stage,
        steps=steps,
        first_global_step=global_step_start + 1,
        last_global_step=final_global_step,
        seconds=time.perf_counter() - started,
        records=records,
        checkpoint=str(checkpoint_path),
    )


def main() -> int:
    global _OPTIONS
    options, remaining = parse_residual_options(sys.argv[1:])
    _OPTIONS = options
    sys.argv = [sys.argv[0], *remaining]
    base.prepare_output_directory = prepare_output_directory
    base.write_json = patched_write_json
    base.run_training_stage = run_training_stage
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
