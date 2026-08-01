#!/usr/bin/env python3
"""Dense native-residual pipeline with a method-agnostic learnable PSF.

Differences from ``smacs0723_wcs_aware_faint_source_pipeline_dense_native_residual.py``:
(1) stage 2 uses a differentiable low-rank PSF field in the observation forward;
(2) rendered products are not convolved again in post-processing; and
(3) background coefficients and PSF-size diagnostics are saved through training.
Stage 1 remains the existing no-PSF pretrain.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from astropy.io import fits
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for search_path in (ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import scripts.smacs0723_wcs_aware_faint_source_pipeline as pipeline  # noqa: E402
import scripts.smacs0723_wcs_aware_faint_source_pipeline_dense_native_residual as dense  # noqa: E402
from torchsky.map.wcs_aware_calibration_types import PSFCalibration  # noqa: E402
from torchsky.map.wcs_aware_inr import PSFKernelBank  # noqa: E402

base = pipeline.base
training_mod = pipeline.training_mod
OriginalModel = base.WCSAwareLiteratureINR
original_dense_snapshot = dense._save_dense_snapshot
_ORIGINAL_VALIDATE_ARGUMENTS = base.validate_arguments


@dataclass(frozen=True)
class Options:
    kernel_size: int
    rank: int
    exposure_latent_dim: int
    hidden_dim: int
    initial_width_pixels: float
    max_correction: float
    learning_rate: float
    min_fwhm_pixels: float
    max_fwhm_pixels: float
    centroid_weight: float
    edge_weight: float
    size_weight: float
    wavelength_weight: float
    latent_weight: float
    basis_weight: float


OPTIONS: Options | None = None
SAVED_PSF: set[tuple[str, int]] = set()


def options() -> Options:
    if OPTIONS is None:
        raise RuntimeError("learnable PSF options are not initialized")
    return OPTIONS


def parse_options(argv: Sequence[str]) -> tuple[Options, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--learnable-psf-kernel-size", type=int, default=9)
    parser.add_argument("--learnable-psf-rank", type=int, default=8)
    parser.add_argument("--learnable-psf-exposure-latent-dim", type=int, default=2)
    parser.add_argument("--learnable-psf-hidden-dim", type=int, default=32)
    parser.add_argument("--learnable-psf-initial-width-pixels", type=float, default=1.0)
    parser.add_argument("--learnable-psf-max-correction", type=float, default=0.5)
    parser.add_argument("--learnable-psf-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--learnable-psf-min-fwhm-pixels", type=float, default=0.5)
    parser.add_argument("--learnable-psf-max-fwhm-pixels", type=float, default=6.0)
    parser.add_argument("--learnable-psf-centroid-weight", type=float, default=10.0)
    parser.add_argument("--learnable-psf-edge-weight", type=float, default=1.0)
    parser.add_argument("--learnable-psf-size-weight", type=float, default=1.0)
    parser.add_argument("--learnable-psf-wavelength-weight", type=float, default=1.0e-2)
    parser.add_argument("--learnable-psf-latent-weight", type=float, default=1.0e-3)
    parser.add_argument("--learnable-psf-basis-weight", type=float, default=1.0e-4)
    ns, remaining = parser.parse_known_args(list(argv))
    if ns.learnable_psf_kernel_size <= 0 or ns.learnable_psf_kernel_size % 2 != 1:
        raise ValueError("learnable-psf-kernel-size must be a positive odd integer")
    if ns.learnable_psf_rank <= 0 or ns.learnable_psf_hidden_dim <= 0:
        raise ValueError("learnable PSF rank and hidden dimension must be positive")
    if ns.learnable_psf_exposure_latent_dim < 0:
        raise ValueError("learnable PSF exposure latent dimension must be non-negative")
    if ns.learnable_psf_initial_width_pixels <= 0 or ns.learnable_psf_max_correction <= 0:
        raise ValueError("learnable PSF initial width and max correction must be positive")
    if ns.learnable_psf_learning_rate <= 0:
        raise ValueError("learnable PSF learning rate must be positive")
    if not 0 < ns.learnable_psf_min_fwhm_pixels < ns.learnable_psf_max_fwhm_pixels:
        raise ValueError("learnable PSF FWHM bounds are invalid")
    weights = [
        ns.learnable_psf_centroid_weight,
        ns.learnable_psf_edge_weight,
        ns.learnable_psf_size_weight,
        ns.learnable_psf_wavelength_weight,
        ns.learnable_psf_latent_weight,
        ns.learnable_psf_basis_weight,
    ]
    if any(value < 0 for value in weights):
        raise ValueError("learnable PSF regularization weights must be non-negative")
    return Options(
        kernel_size=ns.learnable_psf_kernel_size,
        rank=ns.learnable_psf_rank,
        exposure_latent_dim=ns.learnable_psf_exposure_latent_dim,
        hidden_dim=ns.learnable_psf_hidden_dim,
        initial_width_pixels=ns.learnable_psf_initial_width_pixels,
        max_correction=ns.learnable_psf_max_correction,
        learning_rate=ns.learnable_psf_learning_rate,
        min_fwhm_pixels=ns.learnable_psf_min_fwhm_pixels,
        max_fwhm_pixels=ns.learnable_psf_max_fwhm_pixels,
        centroid_weight=ns.learnable_psf_centroid_weight,
        edge_weight=ns.learnable_psf_edge_weight,
        size_weight=ns.learnable_psf_size_weight,
        wavelength_weight=ns.learnable_psf_wavelength_weight,
        latent_weight=ns.learnable_psf_latent_weight,
        basis_weight=ns.learnable_psf_basis_weight,
    ), remaining


def validate_arguments(args: argparse.Namespace) -> None:
    """Reuse all core checks except the fixed-PSF mode requirement."""

    original_mode = args.psf_mode
    original_final = args.final_psf_convolution
    try:
        # The legacy core uses psf_mode only as a proxy for whether stage 2 has
        # a PSF. This launcher supplies its PSF as a model submodule instead.
        args.psf_mode = "gaussian"
        args.final_psf_convolution = False
        _ORIGINAL_VALIDATE_ARGUMENTS(args)
    finally:
        args.psf_mode = original_mode
        args.final_psf_convolution = original_final


class LearnablePSFField(nn.Module):
    """Source-independent low-rank kernel field conditioned on wavelength/exposure."""

    def __init__(self, num_exposures: int, wave_min: float, wave_max: float) -> None:
        super().__init__()
        cfg = options()
        self.num_exposures = int(num_exposures)
        self.kernel_size = cfg.kernel_size
        self.rank = cfg.rank
        self.latent_dim = cfg.exposure_latent_dim
        self.max_correction = cfg.max_correction
        self.min_fwhm = cfg.min_fwhm_pixels
        self.max_fwhm = cfg.max_fwhm_pixels
        r = self.kernel_size // 2
        axis = torch.arange(-r, r + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        rr2 = xx.square() + yy.square()
        self.base_logits = nn.Parameter(
            -rr2 / (2.0 * cfg.initial_width_pixels**2)
        )
        self.basis = nn.Parameter(
            0.02 * torch.randn(self.rank, self.kernel_size, self.kernel_size)
        )
        self.exposure_latent = nn.Parameter(
            torch.zeros(self.num_exposures, self.latent_dim)
        )
        self.coefficient_network = nn.Sequential(
            nn.Linear(1 + self.latent_dim, cfg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.hidden_dim, self.rank),
        )
        last = self.coefficient_network[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)
        self.register_buffer("radius", torch.sqrt(rr2))
        self.register_buffer("log_wave_min", torch.tensor(math.log(wave_min)))
        self.register_buffer("log_wave_max", torch.tensor(math.log(wave_max)))

    def coefficients(self, wavelength: Tensor) -> Tensor:
        logw = torch.log(wavelength.clamp_min(1.0e-12)).float()
        span = self.log_wave_max - self.log_wave_min
        normalized = (
            torch.zeros_like(logw)
            if float(span.abs().detach().cpu()) < 1.0e-12
            else 2.0 * (logw - self.log_wave_min) / span - 1.0
        )
        if self.latent_dim:
            latent = self.exposure_latent[:, None].expand(-1, wavelength.shape[1], -1)
            features = torch.cat((normalized[..., None], latent), dim=-1)
        else:
            features = normalized[..., None]
        return torch.tanh(self.coefficient_network(features))

    def forward(self, wavelength: Tensor) -> Tensor:
        basis = self.basis - self.basis.mean(dim=(-1, -2), keepdim=True)
        basis = basis / basis.square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-6)
        correction = torch.einsum("elr,rxy->elxy", self.coefficients(wavelength), basis)
        correction = correction / math.sqrt(float(self.rank))
        logits = self.base_logits[None, None] + self.max_correction * correction
        flat = torch.softmax(logits.flatten(-2).float(), dim=-1)
        return flat.reshape(*logits.shape)

    def moments(self, kernel: Tensor) -> dict[str, Tensor]:
        mx = (kernel * self.xx).sum((-1, -2))
        my = (kernel * self.yy).sum((-1, -2))
        vx = (kernel * (self.xx - mx[..., None, None]).square()).sum((-1, -2))
        vy = (kernel * (self.yy - my[..., None, None]).square()).sum((-1, -2))
        sigma = torch.sqrt((0.5 * (vx + vy)).clamp_min(1e-12))
        border = torch.zeros_like(kernel, dtype=torch.bool)
        border[..., 0, :] = border[..., -1, :] = True
        border[..., :, 0] = border[..., :, -1] = True
        return {
            "centroid_x": mx,
            "centroid_y": my,
            "fwhm": 2.354820045 * sigma,
            "edge_mass": torch.where(border, kernel, 0.0).sum((-1, -2)),
        }

    def regularization(self, wavelength: Tensor) -> dict[str, Tensor]:
        cfg = options()
        kernel = self(wavelength)
        moment = self.moments(kernel)
        centroid = (moment["centroid_x"].square() + moment["centroid_y"].square()).mean()
        edge = moment["edge_mass"].square().mean()
        fwhm = moment["fwhm"]
        size = (
            torch.relu(self.min_fwhm - fwhm).square()
            + torch.relu(fwhm - self.max_fwhm).square()
        ).mean()
        coefficient = self.coefficients(wavelength)
        wave = (
            (coefficient[:, 1:] - coefficient[:, :-1]).square().mean()
            if coefficient.shape[1] > 1 else kernel.new_zeros(())
        )
        latent = (
            self.exposure_latent.square().mean()
            if self.exposure_latent.numel() else kernel.new_zeros(())
        )
        basis = self.basis.square().mean()
        total = (
            cfg.centroid_weight * centroid + cfg.edge_weight * edge
            + cfg.size_weight * size + cfg.wavelength_weight * wave
            + cfg.latent_weight * latent + cfg.basis_weight * basis
        )
        return {
            "total": total,
            "centroid": centroid,
            "edge": edge,
            "size": size,
            "wavelength": wave,
            "latent": latent,
            "basis": basis,
        }


class LearnablePSFModel(OriginalModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        wave_min = float(kwargs["wavelength_min"])
        wave_max = float(kwargs["wavelength_max"])
        super().__init__(*args, **kwargs)
        self.learnable_psf = LearnablePSFField(self.num_exposures, wave_min, wave_max)


def parameter_groups(
    model: LearnablePSFModel,
    *, learning_rate: float,
    alignment_learning_rate: float,
    background_learning_rate: float,
) -> list[dict[str, Any]]:
    psf = list(model.learnable_psf.parameters())
    alignment = list(model.feature_alignment.parameters())
    background = [model.background] if model.background.requires_grad else []
    excluded = {id(p) for p in [*psf, *alignment, *background]}
    scene = [p for p in model.parameters() if p.requires_grad and id(p) not in excluded]
    groups = [
        {"params": scene, "lr": learning_rate, "name": "scene"},
        {"params": psf, "lr": options().learning_rate, "name": "learnable_psf"},
    ]
    if alignment:
        groups.append({"params": alignment, "lr": alignment_learning_rate, "name": "alignment"})
    if background:
        groups.append({"params": background, "lr": background_learning_rate, "name": "background"})
    return groups


def predict_learnable_psf(
    model: LearnablePSFModel,
    uv: Tensor,
    exposure_id: Tensor,
    bandpass_wavelength: Tensor,
    bandpass_weight: Tensor,
    *,
    plan: Any = None,
    device_plan: Any = None,
    wing_samples: int = 0,
    wing_generator: Any = None,
    offset_chunk: int = 16,
    checkpoint_chunks: bool = True,
) -> tuple[Tensor, Tensor]:
    del plan, device_plan, wing_samples, wing_generator
    all_wave, all_weight = model._validate_bandpass_tensors(
        bandpass_wavelength, bandpass_weight, num_exposures=model.num_exposures
    )
    all_wave = all_wave.to(uv.device)
    all_weight = all_weight.to(uv.device)
    kernel = model.learnable_psf(all_wave)
    wave = all_wave[exposure_id]
    spectral_weight = all_weight[exposure_id]
    _, _, center_valid = model.geometry.native_to_union(uv, exposure_id)
    r = model.learnable_psf.kernel_size // 2
    axis = torch.arange(-r, r + 1, device=uv.device, dtype=uv.dtype)
    dy, dx = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack((dx.flatten(), dy.flatten()), -1)
    total_offsets = offsets.shape[0]
    batch_count = int(uv.shape[0])
    prediction = torch.zeros(batch_count, device=uv.device, dtype=uv.dtype)
    for sample in range(wave.shape[1]):
        kernel_weight = kernel[:, sample].reshape(model.num_exposures, -1)[exposure_id]
        convolved = torch.zeros_like(prediction)
        for start in range(0, total_offsets, max(1, offset_chunk)):
            stop = min(start + max(1, offset_chunk), total_offsets)
            local_offsets = offsets[start:stop]
            local_weight = kernel_weight[:, start:stop]
            chunk_count = int(local_offsets.shape[0])
            query = (uv[:, None] + local_offsets[None]).reshape(-1, 2)
            query_exposure = exposure_id[:, None].expand(-1, chunk_count).reshape(-1)
            query_wave = wave[:, sample, None].expand(-1, chunk_count).reshape(-1)

            def evaluate(
                q: Tensor,
                eid: Tensor,
                w: Tensor,
                kw: Tensor,
                fixed_batch_count: int = batch_count,
                fixed_chunk_count: int = chunk_count,
            ) -> Tensor:
                latent, valid = model.spatial_latent(q, eid)
                scene = model.decode(latent, w, valid).reshape(
                    fixed_batch_count, fixed_chunk_count
                )
                return (scene * kw.to(scene.dtype)).sum(-1)

            if checkpoint_chunks and model.training and torch.is_grad_enabled():
                convolved += checkpoint(
                    evaluate, query, query_exposure, query_wave, local_weight,
                    use_reentrant=False,
                )
            else:
                convolved += evaluate(query, query_exposure, query_wave, local_weight)
        prediction += spectral_weight[:, sample].to(convolved.dtype) * convolved
    return prediction + model.background[exposure_id], center_valid


def dummy_calibration(
    exposures: Sequence[Any], bandpass_wavelength: Tensor, **kwargs: Any
) -> PSFCalibration:
    del kwargs
    kernel = torch.ones(len(exposures), bandpass_wavelength.shape[1], 1, 1)
    bank = PSFKernelBank(kernel, bandpass_wavelength.detach().cpu().float())
    metadata = [{"mode": "learnable_model_submodule_placeholder"} for _ in exposures]
    return PSFCalibration(bank, metadata)


def groups_by_band(pools: Sequence[Any]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for index, pool in enumerate(pools):
        result.setdefault(str(pool.band), []).append(index)
    return result


def quantiles(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    value = value[np.isfinite(value)]
    if not value.size:
        return {key: float("nan") for key in ("min", "q10", "q50", "q90", "max")}
    return {
        "min": float(value.min()), "q10": float(np.quantile(value, 0.1)),
        "q50": float(np.quantile(value, 0.5)), "q90": float(np.quantile(value, 0.9)),
        "max": float(value.max()),
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()


def psf_arrays(model: LearnablePSFModel, wavelength: Tensor) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        device = next(model.parameters()).device
        kernel_device = model.learnable_psf(wavelength.to(device)).detach().float()
        moment_device = model.learnable_psf.moments(kernel_device)
        kernel = kernel_device.cpu()
        moment = {
            key: value.detach().cpu().float()
            for key, value in moment_device.items()
        }
    kernel_np = kernel.numpy()
    radius = model.learnable_psf.radius.detach().cpu().numpy().reshape(-1)
    order = np.argsort(radius)
    sorted_radius = radius[order]
    r50 = np.empty(kernel_np.shape[:2], dtype=np.float32)
    r90 = np.empty_like(r50)
    for e in range(kernel_np.shape[0]):
        for j in range(kernel_np.shape[1]):
            cumulative = np.cumsum(kernel_np[e, j].reshape(-1)[order])
            for target, output in ((0.5, r50), (0.9, r90)):
                k = min(int(np.searchsorted(cumulative, target)), len(sorted_radius) - 1)
                output[e, j] = sorted_radius[k]
    return {
        "kernel": kernel_np,
        "fwhm": moment["fwhm"].numpy(),
        "edge_mass": moment["edge_mass"].numpy(),
        "centroid_x": moment["centroid_x"].numpy(),
        "centroid_y": moment["centroid_y"].numpy(),
        "r50": r50,
        "r90": r90,
    }


def psf_summary(arrays: dict[str, np.ndarray], pools: Sequence[Any], stage: str, step: int) -> dict[str, Any]:
    def summary(indices: Any) -> dict[str, Any]:
        return {
            "fwhm_equivalent_pixels": quantiles(arrays["fwhm"][indices]),
            "r50_pixels": quantiles(arrays["r50"][indices]),
            "r90_pixels": quantiles(arrays["r90"][indices]),
            "edge_mass": quantiles(arrays["edge_mass"][indices]),
            "centroid_radius_pixels": quantiles(
                np.hypot(arrays["centroid_x"][indices], arrays["centroid_y"][indices])
            ),
        }
    return {
        "saved_utc": base.now_utc(), "stage": stage, "global_step": int(step),
        "support_size_pixels": options().kernel_size, "basis_rank": options().rank,
        "overall": summary(np.s_[:]),
        "by_band": {band: summary(indices) for band, indices in groups_by_band(pools).items()},
    }


def save_background(
    model: LearnablePSFModel, pools: Sequence[Any], intensity_scale: float,
    stage: str, stage_step: int, global_step: int, root: Path,
) -> dict[str, Any]:
    normalized = model.background.detach().cpu().float().numpy()
    physical = normalized * intensity_scale
    payload = {
        "saved_utc": base.now_utc(), "stage": stage,
        "stage_step": int(stage_step), "global_step": int(global_step),
        "normalized_summary": quantiles(normalized), "physical_summary": quantiles(physical),
        "physical_by_band": {
            band: quantiles(physical[index]) for band, index in groups_by_band(pools).items()
        },
        "exposures": [
            {
                "exposure_index": i, "path": str(pool.path), "band": str(pool.band),
                "detector": str(pool.detector),
                "background_normalized": float(normalized[i]),
                "background_physical": float(physical[i]),
            }
            for i, pool in enumerate(pools)
        ],
    }
    append_jsonl(root / "logs/background_coefficients.jsonl", payload)
    return payload


def save_psf_metrics(
    model: LearnablePSFModel, pools: Sequence[Any], wavelength: Tensor,
    stage: str, global_step: int, root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = psf_arrays(model, wavelength)
    summary = psf_summary(arrays, pools, stage, global_step)
    append_jsonl(root / "logs/learnable_psf_metrics.jsonl", summary)
    return summary, arrays


def save_psf_snapshot(
    model: LearnablePSFModel, pools: Sequence[Any], wavelength: Tensor, weight: Tensor,
    stage: str, global_step: int, root: Path,
    arrays: dict[str, np.ndarray] | None = None,
) -> None:
    key = (stage, int(global_step))
    if key in SAVED_PSF:
        return
    arrays = psf_arrays(model, wavelength) if arrays is None else arrays
    directory = root / f"learnable_psf_snapshots/step_{global_step:06d}_{stage}"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "learnable_psf_bank.npz", **arrays,
        wavelength=wavelength.cpu().numpy(), bandpass_weight=weight.cpu().numpy(),
        exposure_path=np.asarray([str(pool.path) for pool in pools]),
        exposure_band=np.asarray([str(pool.band) for pool in pools]),
        exposure_detector=np.asarray([str(pool.detector) for pool in pools]),
    )
    weight_np = weight.cpu().numpy()
    for band, indices in groups_by_band(pools).items():
        local_weight = weight_np[indices]
        local_weight /= np.clip(local_weight.sum(1, keepdims=True), 1e-20, None)
        effective = (arrays["kernel"][indices] * local_weight[..., None, None]).sum(1).mean(0)
        hdu = fits.PrimaryHDU(effective.astype(np.float32))
        hdu.header["BAND"] = band
        hdu.header["PSFMODE"] = "LEARNED_BASIS"
        hdu.header["GLOBSTEP"] = int(global_step)
        hdu.header["KERNSIZE"] = options().kernel_size
        hdu.writeto(directory / f"{band}_mean_learned_psf.fits", overwrite=True, checksum=True)
    summary = psf_summary(arrays, pools, stage, global_step)
    summary["bank"] = str(directory / "learnable_psf_bank.npz")
    pipeline._ORIGINAL_WRITE_JSON(directory / "learnable_psf_snapshot.json", summary)
    SAVED_PSF.add(key)
    print(json.dumps({
        "learnable_psf_snapshot": directory.name,
        "fwhm_q50": summary["overall"]["fwhm_equivalent_pixels"]["q50"],
        "r90_q50": summary["overall"]["r90_pixels"]["q50"],
    }), flush=True)


def metadata(root: Path) -> dict[str, Any]:
    cfg = options()
    return {
        "model": "method_agnostic_low_rank_conditional_kernel",
        "stage1_operator": "bandpass_only_no_PSF",
        "stage2_operator": "exact_learnable_PSF_in_forward",
        "postprocess_psf_convolution": False,
        "support_size_pixels": cfg.kernel_size, "basis_rank": cfg.rank,
        "exposure_latent_dim": cfg.exposure_latent_dim,
        "non_negative": True, "unit_mass": True, "source_dependent": False,
        "min_fwhm_pixels": cfg.min_fwhm_pixels,
        "max_fwhm_pixels": cfg.max_fwhm_pixels,
        "learning_rate": cfg.learning_rate,
        "background_log": str(root / "logs/background_coefficients.jsonl"),
        "psf_metric_log": str(root / "logs/learnable_psf_metrics.jsonl"),
        "psf_snapshots": str(root / "learnable_psf_snapshots"),
    }


def run_training_stage(
    *, stage: str, model: LearnablePSFModel, pools: Sequence[Any], wavelength: Tensor,
    weight: Tensor, device: torch.device, steps: int, global_step_start: int,
    batch_size: int | Sequence[int], data_generator: torch.Generator,
    optimizer: torch.optim.Optimizer, scaler: Any, amp_enabled: bool,
    gradient_clip: float, robust_loss: str, robust_delta: float,
    max_relative_weight: float, alignment_regularization: float,
    background_regularization: float, log_every: int, checkpoint_every: int,
    checkpoint_path: Path, intensity_scale: float, canvas: Any,
    checkpoint_configuration: dict[str, Any], loss_handle: Any,
    core_wing_plan: Any = None, device_core_wing: Any = None,
    wing_samples: int = 0, wing_generator: Any = None, offset_chunk: int = 1,
    checkpoint_chunks: bool = True,
) -> Any:
    del wing_samples, wing_generator
    if stage == "finetune":
        for group in optimizer.param_groups:
            if group.get("name") == "learnable_psf":
                group["lr"] = options().learning_rate
    if steps == 0:
        return base.TrainingStageResult(stage, 0, global_step_start, global_step_start, 0.0, [], str(checkpoint_path))
    root = checkpoint_path.parent.parent
    checkpoint_configuration["learnable_psf"] = metadata(root)
    model.train().to(device)
    wave_device, weight_device = wavelength.to(device), weight.to(device)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    seed = int(checkpoint_configuration["seed"])
    zero_names = ("total", "centroid", "edge", "size", "wavelength", "latent", "basis")

    for stage_step in range(1, steps + 1):
        global_step = global_step_start + stage_step
        uv, exposure_id, target, err, mask = training_mod.sample_balanced_batch(
            pools, batch_size=batch_size, generator=data_generator
        )
        uv, exposure_id = uv.to(device), exposure_id.to(device)
        target, err, mask = target.to(device), err.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            if stage == "pretrain":
                prediction, geometry = model.predict_observation(
                    uv, exposure_id, wave_device, weight_device, psf=None
                )
                psf_reg = {name: torch.zeros((), device=device) for name in zero_names}
            elif stage == "finetune":
                prediction, geometry = predict_learnable_psf(
                    model, uv, exposure_id, wave_device, weight_device,
                    plan=core_wing_plan, device_plan=device_core_wing,
                    offset_chunk=offset_chunk, checkpoint_chunks=checkpoint_chunks,
                )
                psf_reg = model.learnable_psf.regularization(wave_device)
            else:
                raise ValueError(f"unsupported stage {stage!r}")
            data_loss, diagnostics = base.weighted_robust_loss(
                prediction, target, err, mask & geometry,
                kind=robust_loss, delta=robust_delta,
                max_relative_weight=max_relative_weight,
            )
            alignment_reg = model.alignment_regularization()
            background_reg = model.background.square().mean()
            loss = (
                data_loss + alignment_regularization * alignment_reg
                + background_regularization * background_reg + psf_reg["total"]
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        loss_row = {
            "stage": stage, "stage_step": stage_step,
            "global_step": global_step, "loss": float(loss.detach()),
        }
        loss_handle.write(json.dumps(loss_row) + "\n")
        if stage_step % min(log_every, 100) == 0 or stage_step == steps:
            loss_handle.flush()
        should_log = stage_step == 1 or stage_step % log_every == 0 or stage_step == steps
        arrays = None
        if should_log:
            background = save_background(
                model, pools, intensity_scale, stage, stage_step, global_step, root
            )
            psf_info, arrays = save_psf_metrics(model, pools, wavelength, stage, global_step, root)
            record = {
                **loss_row, "data_loss": float(data_loss.detach()),
                "rmse": float(diagnostics["rmse"]), "mae": float(diagnostics["mae"]),
                "whitened_rmse": float(diagnostics.get("whitened_rmse", torch.tensor(float("nan")))),
                "whitened_mae": float(diagnostics.get("whitened_mae", torch.tensor(float("nan")))),
                "valid_fraction": float(diagnostics["valid_fraction"]),
                "alignment_regularization": float(alignment_reg.detach()),
                "background_regularization": float(background_reg.detach()),
                "background_physical_q50": background["physical_summary"]["q50"],
                "psf_regularization": {key: float(value.detach()) for key, value in psf_reg.items()},
                "psf_fwhm_pixels_q50": psf_info["overall"]["fwhm_equivalent_pixels"]["q50"],
                "psf_r90_pixels_q50": psf_info["overall"]["r90_pixels"]["q50"],
                "seconds": time.perf_counter() - started,
                "learning_rates": [
                    {"name": str(group.get("name", i)), "value": float(group["lr"])}
                    for i, group in enumerate(optimizer.param_groups)
                ],
            }
            records.append(record)
            print(json.dumps(record), flush=True)
        if checkpoint_every > 0 and stage_step % checkpoint_every == 0:
            base.save_pipeline_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer, scaler=scaler,
                stage=stage, stage_step=stage_step, global_step=global_step,
                intensity_scale=intensity_scale, canvas=canvas, pools=pools,
                wavelength=wavelength, weight=weight,
                configuration=checkpoint_configuration, records=records,
            )
            save_psf_snapshot(model, pools, wavelength, weight, stage, global_step, root, arrays)
        if (
            pipeline._OPTIONS is not None and pipeline._OPTIONS.enabled
            and global_step % pipeline._OPTIONS.every_steps == 0
        ):
            pipeline.save_native_residual_snapshot(
                model=model, pools=pools, wavelength=wavelength, weight=weight,
                device=device, intensity_scale=intensity_scale, seed=seed,
                stage=stage, global_step=global_step,
                core_wing_plan=core_wing_plan, device_core_wing=device_core_wing,
                wing_samples=0, offset_chunk=offset_chunk, amp_enabled=amp_enabled,
            )
            save_psf_snapshot(model, pools, wavelength, weight, stage, global_step, root, arrays)

    final_step = global_step_start + steps
    base.save_pipeline_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer, scaler=scaler,
        stage=stage, stage_step=steps, global_step=final_step,
        intensity_scale=intensity_scale, canvas=canvas, pools=pools,
        wavelength=wavelength, weight=weight,
        configuration=checkpoint_configuration, records=records,
    )
    save_psf_snapshot(model, pools, wavelength, weight, stage, final_step, root)
    return base.TrainingStageResult(
        stage=stage, steps=steps, first_global_step=global_step_start + 1,
        last_global_step=final_step, seconds=time.perf_counter() - started,
        records=records, checkpoint=str(checkpoint_path),
    )


def patched_dense_snapshot(**kwargs: Any) -> dict[str, Any]:
    result = original_dense_snapshot(**kwargs)
    if kwargs["stage"] == "finetune":
        result["operator"] = "bandpass_with_exact_learnable_PSF_convolution"
        result["diagnostic_wing_seed"] = None
        result["learnable_psf"] = {
            "support_size_pixels": options().kernel_size,
            "basis_rank": options().rank,
        }
        pipeline._ORIGINAL_WRITE_JSON(Path(result["metadata"]), result)
        for product in result["products"]:
            with fits.open(product["fits"], mode="update", memmap=False) as hdul:
                hdul["SCI"].header["OPERATOR"] = "LEARNED_PSF"
                hdul.flush()
        dense._write_residual_index()
    return result


def patched_write_json(path: Path, payload: dict[str, Any]) -> None:
    output = payload
    if path.name in {"pipeline_manifest.json", "pipeline_summary.json"}:
        output = copy.deepcopy(payload)
        output["native_residual_diagnostics"] = dense._diagnostic_metadata()
        output["learnable_psf"] = metadata(path.parent)
        output["background_coefficient_tracking"] = {
            "enabled": True,
            "path": str(path.parent / "logs/background_coefficients.jsonl"),
        }
        if isinstance(output.get("stages"), dict):
            stage2 = dict(output["stages"].get("stage2_finetune", {}))
            for key in ("plan", "vectorized_offsets_by_wavelength", "estimated_query_reduction_vs_full_psf"):
                stage2.pop(key, None)
            stage2.update({
                "operator": "exact method-agnostic learnable PSF in forward pass",
                "psf_model": "low_rank_conditional_kernel",
            })
            output["stages"]["stage2_finetune"] = stage2
            output["stages"]["post_training_final_psf_convolution"] = {
                "enabled": False,
                "operator": "deleted; bandpass integration only",
                "is_training_stage": False,
            }
        output["psf"] = {
            "mode": "learnable_method_agnostic",
            "support_size": options().kernel_size,
            "basis_rank": options().rank,
            "external_psf_method_used": False,
        }
        if "final_psf_convolution" in output:
            output["final_psf_convolution"] = {
                "enabled": False,
                "is_training_stage": False,
                "operator_order": "bandpass integration only",
            }
    pipeline._ORIGINAL_WRITE_JSON(path, output)


def remove_option(argv: list[str], name: str, takes_value: bool) -> list[str]:
    result: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == name:
            i += 2 if takes_value else 1
        elif takes_value and argv[i].startswith(name + "="):
            i += 1
        else:
            result.append(argv[i])
            i += 1
    return result


def force_core_flags(argv: Sequence[str]) -> list[str]:
    result = list(argv)
    for name, takes_value in (
        ("--psf-mode", True),
        ("--psf-size", True),
        ("--psf-exact-core-mass", True),
        ("--psf-wing-samples", True),
        ("--final-psf-convolution", False),
        ("--no-final-psf-convolution", False),
    ):
        result = remove_option(result, name, takes_value)
    return [
        *result,
        "--psf-mode", "none",
        "--psf-size", "1",
        "--psf-exact-core-mass", "1.0",
        "--psf-wing-samples", "0",
        "--no-final-psf-convolution",
    ]


def require_finetune(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--finetune-steps", type=int, default=0)
    ns, _ = parser.parse_known_args(list(argv))
    if ns.finetune_steps <= 0:
        raise ValueError("this launcher requires --finetune-steps > 0")


def main() -> int:
    global OPTIONS
    OPTIONS, remaining = parse_options(sys.argv[1:])
    require_finetune(remaining)
    sys.argv = [sys.argv[0], *force_core_flags(remaining)]
    base.validate_arguments = validate_arguments
    base.WCSAwareLiteratureINR = LearnablePSFModel
    base.parameter_groups = parameter_groups
    base.build_psf_calibration = dummy_calibration
    base.predict_core_wing_observation = predict_learnable_psf
    pipeline.run_training_stage = run_training_stage
    dense._save_dense_snapshot = patched_dense_snapshot
    dense._patched_write_json = patched_write_json
    return int(dense.main())


if __name__ == "__main__":
    raise SystemExit(main())
