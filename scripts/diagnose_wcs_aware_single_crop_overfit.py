#!/usr/bin/env python3
"""Stage-A sanity check: overfit one source-rich native detector crop.

This diagnostic intentionally removes multiband, multi-exposure, PSF, sampling,
and robust-loss complications. It asks whether the current WCS-aware INR can
fit every valid pixel in one small CRF crop at one fixed wavelength.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torchsky.map.wcs_aware_inr import (  # noqa: E402
    WCSAwareMultiWavelengthINR,
    build_union_canvas,
)
from torchsky.map.wcs_aware_training import discover_exposures  # noqa: E402

BAND_CENTER_ANGSTROM = {
    "F090W": 9000.0,
    "F150W": 15000.0,
    "F200W": 20000.0,
    "F277W": 27700.0,
    "F356W": 35600.0,
    "F444W": 44400.0,
}


def _valid_mask(
    hdul: fits.HDUList,
    image: np.ndarray,
    err: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(image) & np.isfinite(err) & (err > 0)
    if "DQ" in hdul:
        valid &= np.asarray(hdul["DQ"].data) == 0
    if "UPDCMASK" in hdul:
        extra = np.asarray(hdul["UPDCMASK"].data)
        valid &= np.isfinite(extra) & (extra == 0)
    return valid


def _choose_crop(
    image: np.ndarray,
    valid: np.ndarray,
    size: int,
) -> tuple[int, int]:
    """Choose a source-rich crop using a local positive-excess score."""

    height, width = image.shape
    if size > height or size > width:
        raise ValueError(f"crop size {size} exceeds image shape {image.shape}")

    half = size // 2
    interior = valid.copy()
    interior[:half, :] = False
    interior[height - (size - half) + 1 :, :] = False
    interior[:, :half] = False
    interior[:, width - (size - half) + 1 :] = False
    if not bool(interior.any()):
        raise RuntimeError("no valid interior pixel can serve as crop center")

    background = float(np.median(image[valid]))
    excess = np.where(valid, np.clip(image - background, 0.0, None), 0.0)
    try:
        from scipy.ndimage import uniform_filter

        score = uniform_filter(excess.astype(np.float32), size=9, mode="nearest")
    except Exception:
        score = excess

    score = np.where(interior, score, -np.inf)
    center_y, center_x = np.unravel_index(int(np.argmax(score)), score.shape)
    x0 = int(np.clip(center_x - half, 0, width - size))
    y0 = int(np.clip(center_y - half, 0, height - size))
    return x0, y0


def _crop_header(header: dict, x0: int, y0: int, size: int) -> dict:
    cropped = dict(header)
    cropped["NAXIS1"] = int(size)
    cropped["NAXIS2"] = int(size)
    if "CRPIX1" in cropped:
        cropped["CRPIX1"] = float(cropped["CRPIX1"]) - float(x0)
    if "CRPIX2" in cropped:
        cropped["CRPIX2"] = float(cropped["CRPIX2"]) - float(y0)
    return cropped


def _robust_scale(image: np.ndarray, valid: np.ndarray) -> float:
    values = image[valid].astype(np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    p95 = float(np.percentile(np.abs(values), 95.0))
    return max(1.4826 * mad, p95, 1.0e-6)


def _loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    err: torch.Tensor,
    *,
    kind: str,
    delta: float,
) -> torch.Tensor:
    residual = prediction - target
    if kind == "uniform_l2":
        return 0.5 * residual.square().mean()

    whitened = residual / err.clamp_min(1.0e-8)
    if kind == "whitened_l2":
        return 0.5 * whitened.square().mean()
    if kind == "huber":
        absolute = whitened.abs()
        rho = torch.where(
            absolute <= delta,
            0.5 * whitened.square(),
            delta * (absolute - 0.5 * delta),
        )
        return rho.mean()
    raise ValueError(kind)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--band",
        choices=tuple(BAND_CENTER_ANGSTROM),
        default="F090W",
    )
    parser.add_argument("--module", choices=("a", "b"), default="b")
    parser.add_argument("--exposure-index", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--spatial-frequencies", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--trunk-depth", type=int, default=4)
    parser.add_argument("--decoder-depth", type=int, default=3)
    parser.add_argument(
        "--output-activation",
        choices=("linear", "softplus"),
        default="linear",
    )
    parser.add_argument(
        "--loss",
        choices=("uniform_l2", "whitened_l2", "huber"),
        default="uniform_l2",
    )
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    selected = discover_exposures(
        args.exposure_dir.expanduser().resolve(),
        input_product="crf",
        bands=[args.band],
        module=args.module,
        max_exposures_per_band=None,
    )
    if not 0 <= args.exposure_index < len(selected):
        raise SystemExit(
            f"exposure-index {args.exposure_index} outside [0, {len(selected) - 1}]"
        )
    exposure_path, band, detector = selected[args.exposure_index]

    with fits.open(exposure_path, memmap=False) as hdul:
        image = np.asarray(hdul["SCI"].data, dtype=np.float32)
        err = np.asarray(hdul["ERR"].data, dtype=np.float32)
        valid = _valid_mask(hdul, image, err)
        header = dict(hdul["SCI"].header)

    scale = _robust_scale(image, valid)
    x0, y0 = _choose_crop(image, valid, args.crop_size)
    size = int(args.crop_size)
    crop_image = image[y0 : y0 + size, x0 : x0 + size]
    crop_err = err[y0 : y0 + size, x0 : x0 + size]
    crop_valid = valid[y0 : y0 + size, x0 : x0 + size]
    if float(crop_valid.mean()) < 0.5:
        raise RuntimeError(
            f"selected crop valid fraction is only {float(crop_valid.mean()):.3f}"
        )

    local_header = _crop_header(header, x0, y0, size)
    canvas = build_union_canvas(
        [local_header],
        pixel_scale_arcsec=None,
        padding_pixels=4,
    )

    yy, xx = np.mgrid[0:size, 0:size]
    uv_numpy = np.stack(
        (xx[crop_valid], yy[crop_valid]),
        axis=-1,
    ).astype(np.float32)
    target_numpy = (crop_image[crop_valid] / scale).astype(np.float32)
    err_numpy = (crop_err[crop_valid] / scale).astype(np.float32)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    uv = torch.from_numpy(uv_numpy).to(device)
    exposure_id = torch.zeros(uv.shape[0], dtype=torch.long, device=device)
    target = torch.from_numpy(target_numpy).to(device)
    err_tensor = torch.from_numpy(err_numpy).to(device)
    wavelength = torch.tensor(
        [[BAND_CENTER_ANGSTROM[band]]],
        dtype=torch.float32,
        device=device,
    )
    bandpass_weight = torch.ones_like(wavelength)

    model = WCSAwareMultiWavelengthINR(
        [local_header],
        canvas,
        encoding="fourier",
        spatial_frequencies=args.spatial_frequencies,
        spectral_frequencies=0,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        trunk_depth=args.trunk_depth,
        decoder_depth=args.decoder_depth,
        anchor_exposure=0,
        learn_background=False,
        output_activation=args.output_activation,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []

    print(
        json.dumps(
            {
                "exposure": str(exposure_path),
                "band": band,
                "detector": detector,
                "crop_origin_xy": [x0, y0],
                "crop_size": size,
                "valid_fraction": float(crop_valid.mean()),
                "intensity_scale": scale,
                "target_physical": {
                    "min": float(crop_image[crop_valid].min()),
                    "median": float(np.median(crop_image[crop_valid])),
                    "max": float(crop_image[crop_valid].max()),
                    "std": float(crop_image[crop_valid].std()),
                },
            },
            indent=2,
        ),
        flush=True,
    )

    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction, geometry_valid = model.predict_observation(
            uv,
            exposure_id,
            wavelength,
            bandpass_weight,
            psf=None,
        )
        if not bool(geometry_valid.all()):
            raise RuntimeError("some crop pixels failed WCS geometry")

        loss = _loss(
            prediction,
            target,
            err_tensor,
            kind=args.loss,
            delta=args.huber_delta,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        loss.backward()

        gradient_square = torch.zeros((), device=device)
        for parameter in model.parameters():
            if parameter.grad is not None:
                gradient_square += parameter.grad.detach().float().square().sum()
        gradient_norm = float(torch.sqrt(gradient_square).cpu())

        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                prediction_physical = prediction.float() * scale
                target_physical = target.float() * scale
                residual_physical = prediction_physical - target_physical
                target_contrast = target_physical - target_physical.median()
                prediction_contrast = (
                    prediction_physical - prediction_physical.median()
                )
                denominator = target_contrast.square().sum().clamp_min(1.0e-20)
                response_slope = float(
                    (target_contrast * prediction_contrast)
                    .sum()
                    .div(denominator)
                    .cpu()
                )
                row: dict[str, float | int] = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "rmse_physical": float(
                        residual_physical.square().mean().sqrt().cpu()
                    ),
                    "mae_physical": float(residual_physical.abs().mean().cpu()),
                    "prediction_min_physical": float(
                        prediction_physical.min().cpu()
                    ),
                    "prediction_median_physical": float(
                        prediction_physical.median().cpu()
                    ),
                    "prediction_max_physical": float(
                        prediction_physical.max().cpu()
                    ),
                    "prediction_std_physical": float(
                        prediction_physical.std().cpu()
                    ),
                    "contrast_response_slope": response_slope,
                    "grad_norm_before_clip": gradient_norm,
                }
                history.append(row)
                print(json.dumps(row), flush=True)

    model.eval()
    with torch.no_grad():
        prediction, _ = model.predict_observation(
            uv,
            exposure_id,
            wavelength,
            bandpass_weight,
            psf=None,
        )
    prediction_physical = prediction.float().cpu().numpy() * scale

    prediction_image = np.full((size, size), np.nan, dtype=np.float32)
    residual_image = np.full((size, size), np.nan, dtype=np.float32)
    prediction_image[crop_valid] = prediction_physical
    residual_image[crop_valid] = prediction_physical - crop_image[crop_valid]

    fits.PrimaryHDU(
        crop_image,
        header=fits.Header(local_header),
    ).writeto(output_dir / "target_crop.fits", overwrite=True)
    fits.PrimaryHDU(
        prediction_image,
        header=fits.Header(local_header),
    ).writeto(output_dir / "prediction_crop.fits", overwrite=True)
    fits.PrimaryHDU(
        residual_image,
        header=fits.Header(local_header),
    ).writeto(output_dir / "residual_crop.fits", overwrite=True)
    fits.PrimaryHDU(
        crop_valid.astype(np.uint8),
        header=fits.Header(local_header),
    ).writeto(output_dir / "valid_mask.fits", overwrite=True)

    summary = {
        "exposure": str(exposure_path),
        "band": band,
        "detector": detector,
        "crop_origin_xy": [x0, y0],
        "crop_size": size,
        "intensity_scale": scale,
        "args": vars(args),
        "final": history[-1],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    torch.save(
        {
            "model": model.state_dict(),
            "header": local_header,
            "canvas_header": canvas.header,
            "canvas_shape": canvas.shape,
            "summary": summary,
        },
        output_dir / "checkpoint.pt",
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir), "final": history[-1]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
