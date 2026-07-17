"""Dataset, training, and rendering helpers for the WCS-aware INR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from astropy.io import fits
from torch import Tensor

from ..wcs import WCS
from .wcs_aware_calibration import ExposureCalibrationInput, normalize_detector
from .wcs_aware_inr import (
    PSFKernelBank,
    UnionCanvas,
    WCSAwareMultiWavelengthINR,
    weighted_robust_loss,
)


@dataclass
class ExposurePool:
    path: Path
    band: str
    detector: str
    header: dict[str, Any]
    shape: tuple[int, int]
    uv: Tensor
    value: Tensor
    err: Tensor

    @property
    def size(self) -> int:
        return int(self.value.numel())

    def calibration_input(self) -> ExposureCalibrationInput:
        return ExposureCalibrationInput(
            path=self.path,
            band=self.band,
            detector=self.detector,
            header=self.header,
            shape=self.shape,
        )


@dataclass
class TrainingRecord:
    step: int
    loss: float
    data_loss: float
    rmse: float
    mae: float
    alignment_regularization: float
    background_regularization: float
    seconds: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _product_patterns(product: str) -> tuple[str, ...]:
    return {
        "crf": ("*_crf.fits",),
        "cal": ("*_cal.fits",),
        "updc": (
            "*_cal_1overf_completed_dqfix.fits",
            "*_cal_1overf_completed.fits",
            "*_cal_*completed*dqfix*.fits",
        ),
        "auto": ("*_crf.fits", "*_cal.fits"),
    }[product]


def _read_band_detector(path: Path) -> tuple[str | None, str]:
    try:
        with fits.open(path, memmap=True) as hdul:
            primary = hdul[0].header
            science = hdul["SCI"].header if "SCI" in hdul else primary
            band: str | None = None
            for key in ("FILTER", "PUPIL"):
                value = str(science.get(key) or primary.get(key) or "").strip().upper()
                if value.startswith("F") and value.endswith("W"):
                    band = value
                    break
            detector = normalize_detector(
                science.get("DETECTOR") or primary.get("DETECTOR")
            )
            return band, detector
    except (OSError, KeyError, ValueError, TypeError):
        return None, ""


def _detector_matches(detector: str, *, band: str, module: str) -> bool:
    if module == "any":
        return True
    detector = normalize_detector(detector)
    prefix = f"NRC{module.upper()}"
    if band in {"F277W", "F356W", "F444W"}:
        return detector == f"{prefix}5"
    return detector in {f"{prefix}{index}" for index in range(1, 5)}


def discover_exposures(
    exposure_dir: Path,
    *,
    input_product: str,
    bands: Sequence[str],
    module: str,
    max_exposures_per_band: int | None,
) -> list[tuple[Path, str, str]]:
    if not exposure_dir.exists():
        raise FileNotFoundError(exposure_dir)
    paths: list[Path] = []
    for pattern in _product_patterns(input_product):
        paths.extend(exposure_dir.rglob(pattern))
        if input_product == "auto" and paths and pattern == "*_crf.fits":
            break
    paths = sorted(dict.fromkeys(path.resolve() for path in paths))
    counts = {band: 0 for band in bands}
    selected: list[tuple[Path, str, str]] = []
    for path in paths:
        band, detector = _read_band_detector(path)
        if band not in counts:
            continue
        if not _detector_matches(detector, band=band, module=module):
            continue
        if max_exposures_per_band is not None and counts[band] >= max_exposures_per_band:
            continue
        selected.append((path, band, detector))
        counts[band] += 1
    missing = [band for band, count in counts.items() if count == 0]
    if missing:
        raise FileNotFoundError("no matching exposures for: " + ", ".join(missing))
    return selected


def _valid_pixel_mask(
    hdul: fits.HDUList,
    image: np.ndarray,
    err: np.ndarray,
    *,
    dq_mode: str,
) -> np.ndarray:
    valid = np.isfinite(image) & np.isfinite(err) & (err > 0.0)
    if "DQ" in hdul:
        dq = np.asarray(hdul["DQ"].data)
        if dq.shape != image.shape:
            raise ValueError("DQ shape does not match SCI")
        if dq_mode == "strict":
            valid &= dq == 0
        elif dq_mode == "do_not_use":
            valid &= (dq.astype(np.uint64) & np.uint64(1)) == 0
        else:
            raise ValueError("dq_mode must be strict or do_not_use")
    if "UPDCMASK" in hdul:
        extra = np.asarray(hdul["UPDCMASK"].data)
        if extra.shape != image.shape:
            raise ValueError("UPDCMASK shape does not match SCI")
        valid &= np.isfinite(extra) & (extra == 0)
    return valid


def load_exposure_pool(
    path: Path,
    band: str,
    detector: str,
    *,
    max_pixels: int,
    dq_mode: str,
    rng: np.random.Generator,
) -> ExposurePool:
    with fits.open(path, memmap=False) as hdul:
        if "SCI" not in hdul or "ERR" not in hdul:
            raise ValueError(f"{path} must contain SCI and ERR")
        image = np.asarray(hdul["SCI"].data, dtype=np.float32)
        err = np.asarray(hdul["ERR"].data, dtype=np.float32)
        if image.ndim != 2 or err.shape != image.shape:
            raise ValueError(f"invalid SCI/ERR shapes in {path}")
        valid = np.flatnonzero(
            _valid_pixel_mask(hdul, image, err, dq_mode=dq_mode).reshape(-1)
        )
        if valid.size == 0:
            raise ValueError(f"no valid pixels in {path}")
        chosen = (
            rng.choice(valid, size=max_pixels, replace=False)
            if max_pixels > 0 and valid.size > max_pixels
            else valid
        )
        y, x = np.unravel_index(np.asarray(chosen, dtype=np.int64), image.shape)
        header = dict(hdul["SCI"].header)
        header["NAXIS1"] = int(image.shape[1])
        header["NAXIS2"] = int(image.shape[0])
        return ExposurePool(
            path=path,
            band=band,
            detector=detector,
            header=header,
            shape=(int(image.shape[0]), int(image.shape[1])),
            uv=torch.from_numpy(np.stack((x, y), -1).astype(np.float32)),
            value=torch.from_numpy(image.reshape(-1)[chosen].copy()),
            err=torch.from_numpy(err.reshape(-1)[chosen].copy()),
        )


def robust_intensity_scale(pools: Iterable[ExposurePool]) -> float:
    pieces: list[np.ndarray] = []
    for pool in pools:
        values = pool.value.numpy()
        if values.size > 50000:
            values = values[:: max(1, values.size // 50000)]
        pieces.append(values[np.isfinite(values)])
    merged = np.concatenate(pieces)
    median = float(np.median(merged))
    mad = float(np.median(np.abs(merged - median)))
    p95 = float(np.percentile(np.abs(merged), 95.0))
    return max(1.4826 * mad, p95, 1.0e-6)


def normalize_pools(pools: Sequence[ExposurePool], scale: float) -> list[ExposurePool]:
    return [
        ExposurePool(
            path=pool.path,
            band=pool.band,
            detector=pool.detector,
            header=pool.header,
            shape=pool.shape,
            uv=pool.uv,
            value=pool.value / scale,
            err=pool.err / scale,
        )
        for pool in pools
    ]


def sample_balanced_batch(
    pools: Sequence[ExposurePool],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if batch_size < len(pools):
        raise ValueError("batch_size must be at least the number of exposures")
    each = math.ceil(batch_size / len(pools))
    uv, exposure_id, value, err = [], [], [], []
    for index, pool in enumerate(pools):
        chosen = torch.randint(0, pool.size, (each,), generator=generator)
        uv.append(pool.uv[chosen])
        value.append(pool.value[chosen])
        err.append(pool.err[chosen])
        exposure_id.append(torch.full((each,), index, dtype=torch.long))
    uv_t = torch.cat(uv)
    exposure_t = torch.cat(exposure_id)
    value_t = torch.cat(value)
    err_t = torch.cat(err)
    if uv_t.shape[0] > batch_size:
        order = torch.randperm(uv_t.shape[0], generator=generator)[:batch_size]
        uv_t, exposure_t = uv_t[order], exposure_t[order]
        value_t, err_t = value_t[order], err_t[order]
    return uv_t, exposure_t, value_t, err_t, torch.ones_like(value_t, dtype=torch.bool)


def build_union_coverage_mask(
    headers: Sequence[dict[str, Any]],
    canvas: UnionCanvas,
    *,
    device: torch.device,
    tile_rows: int,
) -> Tensor:
    target = WCS(canvas.header).to(device)
    sources = [WCS(header).to(device) for header in headers]
    shapes = [
        (int(header["NAXIS2"]), int(header["NAXIS1"])) for header in headers
    ]
    height, width = canvas.shape
    coverage = torch.zeros((height, width), dtype=torch.bool, device=device)
    x = torch.arange(width, device=device, dtype=torch.float64)
    for row0 in range(0, height, tile_rows):
        row1 = min(row0 + tile_rows, height)
        y = torch.arange(row0, row1, device=device, dtype=torch.float64)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        ra, dec = target.pixel_to_world(xx.reshape(-1), yy.reshape(-1))
        tile = torch.zeros(ra.shape, dtype=torch.bool, device=device)
        for source, (source_h, source_w) in zip(sources, shapes):
            u, v = source.world_to_pixel(ra, dec)
            tile |= (
                torch.isfinite(u)
                & torch.isfinite(v)
                & (u >= -0.5)
                & (u <= source_w - 0.5)
                & (v >= -0.5)
                & (v <= source_h - 0.5)
            )
        coverage[row0:row1] = tile.reshape(row1 - row0, width)
    return coverage


def _parameter_groups(
    model: WCSAwareMultiWavelengthINR,
    *,
    learning_rate: float,
    alignment_learning_rate: float,
    background_learning_rate: float,
) -> list[dict[str, Any]]:
    alignment = list(model.feature_alignment.parameters())
    background = [model.background] if model.background.requires_grad else []
    excluded = {id(parameter) for parameter in alignment + background}
    main = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    ]
    groups: list[dict[str, Any]] = [
        {"params": main, "lr": learning_rate},
        {"params": alignment, "lr": alignment_learning_rate},
    ]
    if background:
        groups.append({"params": background, "lr": background_learning_rate})
    return groups


def save_checkpoint(
    path: Path,
    *,
    model: WCSAwareMultiWavelengthINR,
    optimizer: torch.optim.Optimizer,
    step: int,
    intensity_scale: float,
    canvas: UnionCanvas,
    pools: Sequence[ExposurePool],
    bandpass_wavelength: Tensor,
    bandpass_weight: Tensor,
    args: dict[str, Any],
) -> None:
    torch.save(
        {
            "format": "torchsky-wcs-aware-multiband-inr-v2",
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "intensity_scale": intensity_scale,
            "union_header": canvas.header,
            "union_shape": canvas.shape,
            "exposure_paths": [str(pool.path) for pool in pools],
            "exposure_bands": [pool.band for pool in pools],
            "bandpass_wavelength": bandpass_wavelength,
            "bandpass_weight": bandpass_weight,
            "args": args,
        },
        path,
    )


def train_model(
    model: WCSAwareMultiWavelengthINR,
    pools: Sequence[ExposurePool],
    bandpass_wavelength: Tensor,
    bandpass_weight: Tensor,
    *,
    psf: PSFKernelBank | None,
    device: torch.device,
    steps: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    alignment_learning_rate: float,
    background_learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    robust_loss: str,
    robust_delta: float,
    max_relative_weight: float,
    alignment_regularization: float,
    background_regularization: float,
    psf_offset_chunk: int,
    checkpoint_psf_chunks: bool,
    amp: bool,
    log_every: int,
    checkpoint_every: int,
    checkpoint_path: Path,
    intensity_scale: float,
    canvas: UnionCanvas,
    checkpoint_args: dict[str, Any],
    resume: Path | None = None,
) -> list[TrainingRecord]:
    model.to(device)
    wavelengths = bandpass_wavelength.to(device)
    weights = bandpass_weight.to(device)
    psf_device = psf.to(device) if psf is not None else None
    optimizer = torch.optim.AdamW(
        _parameter_groups(
            model,
            learning_rate=learning_rate,
            alignment_learning_rate=alignment_learning_rate,
            background_learning_rate=background_learning_rate,
        ),
        weight_decay=weight_decay,
    )
    start_step = 0
    if resume is not None:
        saved = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved.get("step", 0))
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    amp_enabled = device.type == "cuda" and amp
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.amp.GradScaler(enabled=amp_enabled)
    records: list[TrainingRecord] = []
    started = time.perf_counter()
    model.train()
    for step in range(start_step + 1, steps + 1):
        uv, exposure_id, target, err, mask = sample_balanced_batch(
            pools, batch_size=batch_size, generator=generator
        )
        uv, exposure_id = uv.to(device), exposure_id.to(device)
        target, err, mask = target.to(device), err.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            prediction, geometry_valid = model.predict_observation(
                uv,
                exposure_id,
                wavelengths,
                weights,
                psf=psf_device,
                psf_offset_chunk=psf_offset_chunk,
                checkpoint_psf_chunks=checkpoint_psf_chunks,
            )
            data_loss, diagnostics = weighted_robust_loss(
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
        if step == 1 or step % log_every == 0 or step == steps:
            record = TrainingRecord(
                step=step,
                loss=float(loss.detach()),
                data_loss=float(data_loss.detach()),
                rmse=float(diagnostics["rmse"]),
                mae=float(diagnostics["mae"]),
                alignment_regularization=float(alignment_reg.detach()),
                background_regularization=float(background_reg.detach()),
                seconds=time.perf_counter() - started,
            )
            records.append(record)
            print(asdict(record), flush=True)
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step,
                intensity_scale=intensity_scale,
                canvas=canvas,
                pools=pools,
                bandpass_wavelength=bandpass_wavelength,
                bandpass_weight=bandpass_weight,
                args=checkpoint_args,
            )
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=steps,
        intensity_scale=intensity_scale,
        canvas=canvas,
        pools=pools,
        bandpass_wavelength=bandpass_wavelength,
        bandpass_weight=bandpass_weight,
        args=checkpoint_args,
    )
    return records


def _union_latent_for_xy(
    model: WCSAwareMultiWavelengthINR,
    canvas: UnionCanvas,
    xy: Tensor,
    *,
    anchor_exposure: int,
    device: torch.device,
) -> Tensor:
    target_wcs = WCS(canvas.header).to(device)
    if model.encoding == "fourier":
        raw = model.position_encoder(xy)
    else:
        ra, dec = target_wcs.pixel_to_world(xy[:, 0].double(), xy[:, 1].double())
        raw = model.position_encoder(torch.stack((ra, dec), dim=-1))
    exposure_id = torch.full(
        (xy.shape[0],), anchor_exposure, dtype=torch.long, device=device
    )
    return model.trunk(model.feature_alignment(raw, exposure_id))


@torch.no_grad()
def render_union_band(
    model: WCSAwareMultiWavelengthINR,
    canvas: UnionCanvas,
    bandpass_wavelength: Tensor,
    bandpass_weight: Tensor,
    representative_exposure: int,
    *,
    anchor_exposure: int,
    device: torch.device,
    tile_pixels: int,
) -> Tensor:
    model.eval()
    height, width = canvas.shape
    output = torch.empty(height * width, dtype=torch.float32, device=device)
    wavelength = bandpass_wavelength[representative_exposure].to(device)
    weight = bandpass_weight[representative_exposure].to(device)
    for start in range(0, output.numel(), tile_pixels):
        stop = min(start + tile_pixels, output.numel())
        index = torch.arange(start, stop, device=device)
        y = torch.div(index, width, rounding_mode="floor")
        x = index - y * width
        xy = torch.stack((x, y), -1).float()
        latent = _union_latent_for_xy(
            model,
            canvas,
            xy,
            anchor_exposure=anchor_exposure,
            device=device,
        )
        valid = torch.ones(xy.shape[0], dtype=torch.bool, device=device)
        value = torch.zeros(xy.shape[0], dtype=torch.float32, device=device)
        for sample_index in range(wavelength.numel()):
            wave = torch.full_like(value, float(wavelength[sample_index]))
            value += weight[sample_index] * model.decode(latent, wave, valid)
        output[start:stop] = value
    return output.reshape(height, width)


@torch.no_grad()
def render_union_decoder_slice(
    model: WCSAwareMultiWavelengthINR,
    canvas: UnionCanvas,
    wavelength: float,
    *,
    anchor_exposure: int,
    device: torch.device,
    tile_pixels: int,
) -> Tensor:
    """Render the direct decoder output S_theta(x, y, wavelength).

    This is the monochromatic decoder value before bandpass integration,
    background addition, and any explicit PSF operation.
    """
    model.eval()
    height, width = canvas.shape
    output = torch.empty(height * width, dtype=torch.float32, device=device)
    wave_value = float(wavelength)
    for start in range(0, output.numel(), tile_pixels):
        stop = min(start + tile_pixels, output.numel())
        index = torch.arange(start, stop, device=device)
        y = torch.div(index, width, rounding_mode="floor")
        x = index - y * width
        xy = torch.stack((x, y), -1).float()
        latent = _union_latent_for_xy(
            model,
            canvas,
            xy,
            anchor_exposure=anchor_exposure,
            device=device,
        )
        valid = torch.ones(xy.shape[0], dtype=torch.bool, device=device)
        wave = torch.full((xy.shape[0],), wave_value, dtype=torch.float32, device=device)
        output[start:stop] = model.decode(latent, wave, valid)
    return output.reshape(height, width)


__all__ = [
    "ExposurePool",
    "TrainingRecord",
    "build_union_coverage_mask",
    "discover_exposures",
    "load_exposure_pool",
    "normalize_pools",
    "render_union_band",
    "render_union_decoder_slice",
    "robust_intensity_scale",
    "seed_everything",
    "train_model",
]
