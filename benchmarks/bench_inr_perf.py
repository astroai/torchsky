"""INR field model training throughput and render-quality benchmarks.

Measures training throughput (steps/sec, pixels/sec) on CPU and CUDA,
and compares INR rendered quality against coadd baselines (including
SWarp when available) using synthetic truth data.

Usage (pytest)::

    pixi run -e test -- pytest tests/test_inr_perf.py -v

Usage (standalone)::

    pixi run -e test -- python benchmarks/bench_inr_perf.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import torchsky  # noqa: E402
from torchsky import WCS  # noqa: E402
from torchsky.device import default_compute_device  # noqa: E402
from torchsky.map import (  # noqa: E402
    FieldObservation,
    ImageObservation,
    coadd_image_plane,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    render_field_observation,
)
from torchsky.map.field.types import FieldModelKind  # noqa: E402

try:
    import bench_map_compat as _map_compat  # noqa: E402
except Exception:
    _map_compat = None

# ── Analytic truth generation ─────────────────────────────────────────


def _analytic_sky(
    ra_deg: torch.Tensor,
    dec_deg: torch.Tensor,
    *,
    center_ra_deg: float,
    center_dec_deg: float,
) -> torch.Tensor:
    """Sum of Gaussians + ripple + slope (same pattern as bench_map_ground_truth)."""
    dra = (ra_deg - float(center_ra_deg)) * math.cos(
        math.radians(float(center_dec_deg))
    )
    ddec = dec_deg - float(center_dec_deg)
    g1 = 1.4 * torch.exp(-((dra + 0.09) ** 2 + (ddec - 0.04) ** 2) / 0.0018)
    g2 = 0.9 * torch.exp(-((dra - 0.06) ** 2 + (ddec + 0.05) ** 2) / 0.0010)
    g3 = 0.6 * torch.exp(-((dra + 0.01) ** 2 + (ddec + 0.10) ** 2) / 0.00055)
    ripple = 0.12 * torch.sin(55.0 * dra) * torch.cos(48.0 * ddec)
    slope = 0.25 + 0.7 * (dra + 0.18) + 0.35 * (ddec - 0.03)
    return (slope + g1 + g2 + g3 + ripple).to(dtype=torch.float32)


def build_truth_image(header: dict[str, Any]) -> torch.Tensor:
    """Render the analytic sky model onto a WCS grid."""
    wcs = WCS(header)
    height = int(header["NAXIS2"])
    width = int(header["NAXIS1"])
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    ra_deg, dec_deg = wcs.pixel_to_world(x, y)
    return _analytic_sky(
        ra_deg.to(dtype=torch.float32),
        dec_deg.to(dtype=torch.float32),
        center_ra_deg=float(header["CRVAL1"]),
        center_dec_deg=float(header["CRVAL2"]),
    )


def _gaussian_blur(image: torch.Tensor, sigma_pix: float) -> torch.Tensor:
    """Separable 1D Gaussian blur."""
    if sigma_pix <= 0.0:
        return image
    radius = max(1, int(math.ceil(3.0 * float(sigma_pix))))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=image.device)
    kernel = torch.exp(-0.5 * (x / float(sigma_pix)) ** 2)
    kernel = kernel / kernel.sum()
    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)
    out = image.unsqueeze(0).unsqueeze(0)
    out = torch.nn.functional.conv2d(out, kernel_x, padding=(0, kernel.numel() // 2))
    out = torch.nn.functional.conv2d(out, kernel_y, padding=(kernel.numel() // 2, 0))
    return out.squeeze(0).squeeze(0)


# ── Benchmark case definitions ────────────────────────────────────────


@dataclass(frozen=True)
class INRPerfSpec:
    """One INR training / render benchmark case."""

    name: str
    description: str
    model_kind: FieldModelKind
    steps: int = 80
    lr: float = 5.0e-3
    patch_size: int = 64
    patches_per_obs: int = 2
    obs_per_step: int = 2
    target_pixels: tuple[int, int] = (256, 256)
    num_observation_images: int = 4
    image_size: tuple[int, int] = (128, 128)
    synthetic_seed: int = 42
    spectral_min: float = 5000.0
    spectral_max: float = 15000.0
    transmission_center: float = 9000.0
    transmission_sigma: float = 500.0
    psf_sigma_pix: float = 2.0
    noise_sigma: float = 0.01
    cd_scale_deg: float = 0.01


DEFAULT_PERF_CASES: tuple[INRPerfSpec, ...] = (
    INRPerfSpec(
        name="sky_lambda_feature_field_fast",
        description="SkyLambdaFeatureField — 256×256 target, 4×128² observations",
        model_kind="sky_lambda_feature_field",
        steps=80,
    ),
    INRPerfSpec(
        name="sky_encoder_resmlp_fast",
        description="SkyEncoderResMLPField — 256×256 target, 4×128² observations",
        model_kind="sky_encoder_resmlp",
        steps=80,
    ),
    INRPerfSpec(
        name="hashgrid_spectral_field_fast",
        description="HashGridSpectralField — 256×256 target, 4×128² observations",
        model_kind="hashgrid_spectral_field",
        steps=80,
    ),
    INRPerfSpec(
        name="sine_implicit_baseline",
        description="SineImplicitField baseline — 80-step fit",
        model_kind="sine_implicit",
        steps=80,
    ),
)


# ── Synthetic observation generation ──────────────────────────────────


def _make_tangent_header(
    *,
    ra: float = 180.0,
    dec: float = 0.0,
    naxis1: int = 128,
    naxis2: int = 128,
    cd_scale: float = 0.01,
) -> dict[str, Any]:
    """Minimal TAN WCS header for synthetic observations."""
    return {
        "NAXIS": 2,
        "NAXIS1": naxis1,
        "NAXIS2": naxis2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (naxis1 / 2.0) + 0.5,
        "CRPIX2": (naxis2 / 2.0) + 0.5,
        "CRVAL1": ra,
        "CRVAL2": dec,
        "CD1_1": -cd_scale,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": cd_scale,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }


def _render_truth_to_observation(
    truth_header: dict[str, Any],
    truth_image: torch.Tensor,
    obs_header: dict[str, Any],
    obs_shape: tuple[int, int],
    *,
    psf_sigma_pix: float,
    noise_sigma: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Project truth onto an observation grid, blur, and add noise."""
    obs_wcs = WCS(obs_header)
    truth_wcs = WCS(truth_header)
    h, w = obs_shape
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float64),
        torch.arange(w, dtype=torch.float64),
        indexing="ij",
    )
    ra_deg, dec_deg = obs_wcs.pixel_to_world(x, y)

    # Sample truth image at observation pixel centers
    src_h, src_w = int(truth_image.shape[0]), int(truth_image.shape[1])
    px, py = truth_wcs.world_to_pixel(
        ra_deg.reshape(-1).to(torch.float64),
        dec_deg.reshape(-1).to(torch.float64),
    )
    px = px.reshape(h, w).to(dtype=torch.float32, device=device)
    py = py.reshape(h, w).to(dtype=torch.float32, device=device)

    # Normalize to [-1, 1] for grid_sample
    px_norm = (2.0 * px / max(src_w - 1, 1)) - 1.0
    py_norm = (2.0 * py / max(src_h - 1, 1)) - 1.0
    grid = torch.stack([px_norm.reshape(-1), py_norm.reshape(-1)], dim=-1).view(
        1, px_norm.numel(), 1, 2
    )
    sampled = torch.nn.functional.grid_sample(
        truth_image.view(1, 1, src_h, src_w).to(device),
        grid.to(device),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(h, w)

    in_bounds = (
        (px >= 0.0) & (px <= float(src_w - 1)) & (py >= 0.0) & (py <= float(src_h - 1))
    )
    signal = torch.where(in_bounds, sampled, torch.zeros_like(sampled))

    # Apply PSF blur
    signal = _gaussian_blur(signal, psf_sigma_pix)

    # Add noise
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    noisy = signal + noise_sigma * torch.randn(
        signal.shape, generator=gen, dtype=torch.float32, device=device
    )
    return noisy


def build_perf_observations(
    spec: INRPerfSpec,
    *,
    device: torch.device | None = None,
) -> list[FieldObservation]:
    """Create synthetic FieldObservation list with analytic truth-based images."""
    dev = device or default_compute_device()

    # Build truth on a common target grid
    th, tw = spec.target_pixels
    truth_header = _make_tangent_header(
        ra=180.0,
        dec=0.0,
        naxis1=tw,
        naxis2=th,
        cd_scale=max(spec.cd_scale_deg * 0.5, 0.002),
    )
    truth_image = build_truth_image(truth_header).to(device=dev)

    observations: list[FieldObservation] = []
    for i in range(spec.num_observation_images):
        h, w = spec.image_size
        obs_header = _make_tangent_header(
            ra=180.0 + float(i) * 0.02,
            dec=float(i) * 0.015,
            naxis1=w,
            naxis2=h,
            cd_scale=spec.cd_scale_deg,
        )
        image = _render_truth_to_observation(
            truth_header,
            truth_image,
            obs_header,
            (h, w),
            psf_sigma_pix=spec.psf_sigma_pix,
            noise_sigma=spec.noise_sigma,
            seed=spec.synthetic_seed + i,
            device=dev,
        )
        observations.append(
            FieldObservation(
                image=image,
                header=obs_header,
                spectral_min=spec.spectral_min,
                spectral_max=spec.spectral_max,
                transmission_center=spec.transmission_center,
                transmission_sigma=spec.transmission_sigma,
                psf_sigma_pix=spec.psf_sigma_pix,
            )
        )
    return observations


def estimate_domain_from_observations(
    observations: Sequence[FieldObservation],
):
    """Derive a FieldDomain from the observation collection."""
    return estimate_field_domain(observations, coord_mode="sphere_xyz")


# ── Throughput measurement ────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingThroughput:
    """Training throughput metrics for one benchmark run."""

    case_name: str
    model_kind: str
    device: str
    steps: int
    duration_sec: float
    steps_per_sec: float
    pixels_per_sec: float
    total_pixels_processed: int
    initial_loss: float
    final_loss: float
    loss_improvement_ratio: float
    num_params: int
    peak_memory_bytes: int = 0


def measure_training_throughput(
    spec: INRPerfSpec,
    *,
    device: torch.device | str | None = None,
) -> TrainingThroughput:
    """Run a full fit_field_model benchmark and return throughput metrics."""
    torch.manual_seed(int(spec.synthetic_seed))
    dev = (
        torch.device(device)
        if isinstance(device, str)
        else (device or default_compute_device())
    )

    observations = build_perf_observations(spec, device=dev)
    domain = estimate_domain_from_observations(observations)
    model = make_field_model(spec.model_kind, domain)
    model = model.to(device=dev)

    total_params = sum(p.numel() for p in model.parameters())

    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    result = fit_field_model(
        observations,
        model,
        steps=spec.steps,
        lr=spec.lr,
        training_patch_size=spec.patch_size,
        training_patches_per_observation=spec.patches_per_obs,
        observations_per_step=min(len(observations), spec.obs_per_step),
        device=dev,
    )
    if dev.type == "cuda":
        torch.cuda.synchronize()
    duration = time.perf_counter() - start

    peak_mem = 0
    if dev.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    patch_pixels = spec.patch_size * spec.patch_size
    pixels_per_step = spec.obs_per_step * spec.patches_per_obs * patch_pixels
    total_pixels = spec.steps * pixels_per_step

    return TrainingThroughput(
        case_name=spec.name,
        model_kind=spec.model_kind,
        device=str(dev),
        steps=spec.steps,
        duration_sec=duration,
        steps_per_sec=spec.steps / duration if duration > 0 else 0.0,
        pixels_per_sec=total_pixels / duration if duration > 0 else 0.0,
        total_pixels_processed=total_pixels,
        initial_loss=result.initial_loss,
        final_loss=result.final_loss,
        loss_improvement_ratio=(
            result.initial_loss / result.final_loss
            if result.final_loss > 0
            else float("inf")
        ),
        num_params=total_params,
        peak_memory_bytes=peak_mem,
    )


# ── Render quality comparison ─────────────────────────────────────────


@dataclass(frozen=True)
class RenderQualityComparison:
    """Comparison of INR render vs coadd baseline for one benchmark case."""

    case_name: str
    model_kind: str
    device: str
    target_shape: tuple[int, int]
    inr_render_duration_sec: float
    coadd_duration_sec: float
    inr_psnr_db: float
    coadd_psnr_db: float
    inr_ssim: float
    coadd_ssim: float
    inr_mean: float
    coadd_mean: float
    inr_psnr_vs_truth_db: float = 0.0
    coadd_psnr_vs_truth_db: float = 0.0
    inr_ssim_vs_truth: float = 0.0
    coadd_ssim_vs_truth: float = 0.0
    coadd_library: str = "none"


def _psnr(pred: Tensor, target: Tensor) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-12:
        return float("inf")
    data_range = float(target.max().item() - target.min().item())
    if data_range < 1e-12:
        data_range = 1.0
    return float(20.0 * math.log10(data_range / math.sqrt(mse)))


def _ssim(pred: Tensor, target: Tensor, *, window_size: int = 11) -> float:
    """Structural Similarity Index."""
    C1 = 0.01**2
    C2 = 0.03**2

    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device) / (
        window_size * window_size
    )

    mu_pred = torch.nn.functional.conv2d(
        pred.view(1, 1, *pred.shape), kernel, padding=window_size // 2
    )
    mu_targ = torch.nn.functional.conv2d(
        target.view(1, 1, *target.shape), kernel, padding=window_size // 2
    )

    mu_pred_sq = mu_pred * mu_pred
    mu_targ_sq = mu_targ * mu_targ
    mu_pred_targ = mu_pred * mu_targ

    sigma_pred_sq = (
        torch.nn.functional.conv2d(
            (pred * pred).view(1, 1, *pred.shape), kernel, padding=window_size // 2
        )
        - mu_pred_sq
    )
    sigma_targ_sq = (
        torch.nn.functional.conv2d(
            (target * target).view(1, 1, *target.shape),
            kernel,
            padding=window_size // 2,
        )
        - mu_targ_sq
    )
    sigma_pred_targ = (
        torch.nn.functional.conv2d(
            (pred * target).view(1, 1, *pred.shape), kernel, padding=window_size // 2
        )
        - mu_pred_targ
    )

    ssim_map = ((2.0 * mu_pred_targ + C1) * (2.0 * sigma_pred_targ + C2)) / (
        (mu_pred_sq + mu_targ_sq + C1) * (sigma_pred_sq + sigma_targ_sq + C2)
    )

    return float(ssim_map.mean().item())


def measure_render_quality(
    spec: INRPerfSpec,
    *,
    device: torch.device | str | None = None,
) -> RenderQualityComparison:
    """Compare INR render against coadd baseline for a benchmark case.

    Uses SWarp if available; falls back to Torchsky's native weighted-mean coadd.
    Both are compared against the synthetic ground truth.
    """
    torch.manual_seed(int(spec.synthetic_seed))
    dev = (
        torch.device(device)
        if isinstance(device, str)
        else (device or default_compute_device())
    )

    # Build truth on the target grid
    th, tw = spec.target_pixels
    truth_header = _make_tangent_header(
        ra=180.0,
        dec=0.0,
        naxis1=tw,
        naxis2=th,
        cd_scale=max(spec.cd_scale_deg * 0.5, 0.002),
    )
    truth_image = build_truth_image(truth_header).to(device=dev)

    # Build observations
    observations = build_perf_observations(spec, device=dev)
    domain = estimate_domain_from_observations(observations)
    model = make_field_model(spec.model_kind, domain)
    model = model.to(device=dev)

    # Train the INR model
    fit_field_model(
        observations,
        model,
        steps=spec.steps,
        lr=spec.lr,
        training_patch_size=spec.patch_size,
        training_patches_per_observation=spec.patches_per_obs,
        observations_per_step=min(len(observations), spec.obs_per_step),
        device=dev,
    )
    model.eval()

    # Create target observation for rendering
    target_obs = FieldObservation(
        image=torch.zeros((th, tw), dtype=torch.float32, device=dev),
        header=truth_header,
        weight=torch.ones((th, tw), dtype=torch.float32, device=dev),
        spectral_min=spec.spectral_min,
        spectral_max=spec.spectral_max,
        transmission_center=spec.transmission_center,
        transmission_sigma=spec.transmission_sigma,
    )

    # INR render
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        inr_render = render_field_observation(model, target_obs, num_spectral_samples=9)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    inr_duration = time.perf_counter() - t0

    # Coadd baseline: try SWarp first, fall back to Torchsky's native coadd.
    coadd_image: torch.Tensor
    coadd_duration: float = 0.0
    coadd_library: str = "none"

    swarp_available = (
        _map_compat is not None and _map_compat._find_swarp_binary() is not None
    )

    with tempfile.TemporaryDirectory(prefix="torchsky_inr_coadd_") as tmp:
        tmpdir = Path(tmp)
        obs_list: list[ImageObservation] = []
        for i in range(spec.num_observation_images):
            h, w = spec.image_size
            obs_header = _make_tangent_header(
                ra=180.0 + float(i) * 0.02,
                dec=float(i) * 0.015,
                naxis1=w,
                naxis2=h,
                cd_scale=spec.cd_scale_deg,
            )
            clean = _render_truth_to_observation(
                truth_header,
                truth_image,
                obs_header,
                (h, w),
                psf_sigma_pix=spec.psf_sigma_pix,
                noise_sigma=0.0,
                seed=spec.synthetic_seed + i,
                device=dev,
            )
            img_path = tmpdir / f"coadd_obs_{i}.fits"
            torchsky.write(str(img_path), clean.cpu(), header=obs_header, overwrite=True)
            obs_list.append(ImageObservation(str(img_path)))

        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_coadd = time.perf_counter()
        if swarp_available:
            try:
                coadd_np, _ = _map_compat._coadd_with_swarp(
                    obs_list,
                    target_header=truth_header,
                    resampling_type="BILINEAR",
                )
                coadd_image = torch.from_numpy(coadd_np).to(device=dev)
                coadd_library = "swarp:bilinear"
            except Exception:
                native = coadd_image_plane(
                    obs_list, target_header=truth_header, device=dev
                )
                coadd_image = native.data
                coadd_library = "torchsky:bilinear"
        else:
            native = coadd_image_plane(obs_list, target_header=truth_header, device=dev)
            coadd_image = native.data
            coadd_library = "torchsky:bilinear"
        if dev.type == "cuda":
            torch.cuda.synchronize()
        coadd_duration = time.perf_counter() - t_coadd

    # Compute quality metrics vs truth
    inr_psnr_truth = _psnr(inr_render, truth_image)
    coadd_psnr_truth = _psnr(coadd_image, truth_image)
    inr_ssim_truth = _ssim(inr_render, truth_image)
    coadd_ssim_truth = _ssim(coadd_image, truth_image)

    return RenderQualityComparison(
        case_name=spec.name,
        model_kind=spec.model_kind,
        device=str(dev),
        target_shape=(th, tw),
        inr_render_duration_sec=inr_duration,
        coadd_duration_sec=coadd_duration,
        inr_psnr_db=_psnr(inr_render, coadd_image),
        coadd_psnr_db=0.0,
        inr_ssim=_ssim(inr_render, coadd_image),
        coadd_ssim=0.0,
        inr_mean=float(inr_render.mean().item()),
        coadd_mean=float(coadd_image.mean().item()),
        inr_psnr_vs_truth_db=inr_psnr_truth,
        coadd_psnr_vs_truth_db=coadd_psnr_truth,
        inr_ssim_vs_truth=inr_ssim_truth,
        coadd_ssim_vs_truth=coadd_ssim_truth,
        coadd_library=coadd_library,
    )


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="INR field model training throughput and render quality benchmarks"
    )
    p.add_argument(
        "--case",
        default=None,
        help="Benchmark a specific case by name (default: all DEFAULT_PERF_CASES)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Tensor device (e.g. cpu, cuda, cuda:1, mps). Default: auto-detect.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override training steps for all cases",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout",
    )
    args = p.parse_args()

    cases = DEFAULT_PERF_CASES
    if args.case is not None:
        cases = tuple(c for c in cases if c.name == args.case)
        if not cases:
            raise SystemExit(f"unknown case {args.case!r}")

    results: list[dict[str, Any]] = []
    for case in cases:
        spec = case
        if args.steps is not None:
            spec = INRPerfSpec(
                name=case.name,
                description=case.description,
                model_kind=case.model_kind,
                steps=args.steps,
                lr=case.lr,
                patch_size=case.patch_size,
                patches_per_obs=case.patches_per_obs,
                obs_per_step=case.obs_per_step,
                target_pixels=case.target_pixels,
                num_observation_images=case.num_observation_images,
                image_size=case.image_size,
            )

        tp = measure_training_throughput(spec, device=args.device)
        rq = measure_render_quality(spec, device=args.device)

        results.append(
            {
                "throughput": {
                    "case": tp.case_name,
                    "model": tp.model_kind,
                    "device": tp.device,
                    "steps": tp.steps,
                    "duration_sec": tp.duration_sec,
                    "steps_per_sec": tp.steps_per_sec,
                    "pixels_per_sec": tp.pixels_per_sec,
                    "total_pixels": tp.total_pixels_processed,
                    "initial_loss": tp.initial_loss,
                    "final_loss": tp.final_loss,
                    "loss_ratio": tp.loss_improvement_ratio,
                    "params": tp.num_params,
                    "peak_mem_bytes": tp.peak_memory_bytes,
                },
                "quality": {
                    "case": rq.case_name,
                    "model": rq.model_kind,
                    "device": rq.device,
                    "inr_render_sec": rq.inr_render_duration_sec,
                    "coadd_duration_sec": rq.coadd_duration_sec,
                    "coadd_library": rq.coadd_library,
                    "inr_psnr_db": rq.inr_psnr_db,
                    "inr_ssim": rq.inr_ssim,
                    "inr_psnr_vs_truth_db": rq.inr_psnr_vs_truth_db,
                    "coadd_psnr_vs_truth_db": rq.coadd_psnr_vs_truth_db,
                    "inr_ssim_vs_truth": rq.inr_ssim_vs_truth,
                    "coadd_ssim_vs_truth": rq.coadd_ssim_vs_truth,
                },
            }
        )

    if args.json:
        json.dump(results, sys.stdout, indent=2)
    else:
        for entry in results:
            tp_data = entry["throughput"]
            rq_data = entry["quality"]
            print(
                f"{tp_data['case']:40s}  {tp_data['device']:6s}  "
                f"{tp_data['steps_per_sec']:8.1f} step/s  "
                f"{tp_data['pixels_per_sec'] / 1e6:8.1f} Mpix/s  "
                f"loss {tp_data['initial_loss']:.4f}→{tp_data['final_loss']:.4f}"
            )
            print(
                f"  {'':40s}  {'':6s}  "
                f"INR render {rq_data['inr_render_sec']:.3f}s  "
                f"PSNR {rq_data['inr_psnr_db']:.1f} dB  SSIM {rq_data['inr_ssim']:.4f}"
            )
            if rq_data.get("coadd_library", "none") != "none":
                print(
                    f"  {'':40s}  {'':6s}  "
                    f"vs truth — INR PSNR {rq_data['inr_psnr_vs_truth_db']:.1f} dB, "
                    f"coadd ({rq_data['coadd_library']}) PSNR "
                    f"{rq_data['coadd_psnr_vs_truth_db']:.1f} dB"
                )


__all__ = [
    "DEFAULT_PERF_CASES",
    "INRPerfSpec",
    "RenderQualityComparison",
    "TrainingThroughput",
    "build_truth_image",
    "build_perf_observations",
    "estimate_domain_from_observations",
    "measure_render_quality",
    "measure_training_throughput",
]

if __name__ == "__main__":
    main()
