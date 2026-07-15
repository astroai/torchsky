#!/usr/bin/env python3
"""Train the WCS-aware multi-wavelength INR on native SMACS J0723 CRFs.

Defaults match the ARC project layout prepared by
``prepare_smacs0723_multiband_crf.py``:

    inputs:  /arc/projects/ots/jwst_smacs0723/input
    outputs: /arc/projects/ots/jwst_smacs0723/results/wcs_aware_multiband_inr

The data are not coadded before training. Native detector pixels, their WCS,
ERR/DQ masks, and wavelength metadata enter the forward model directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from astropy.io import fits
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchsky.map.wcs_aware_inr import (  # noqa: E402
    BandpassTable,
    PSFKernelBank,
    WCSAwareMultiWavelengthINR,
    build_union_canvas,
    gaussian_psf_kernel,
    weighted_robust_loss,
)

BAND_SPECS = {
    "F090W": (9000.0, 8000.0, 10000.0),
    "F150W": (15000.0, 13000.0, 17000.0),
    "F200W": (20000.0, 17000.0, 23000.0),
    "F277W": (27700.0, 24000.0, 32000.0),
    "F356W": (35600.0, 31000.0, 40000.0),
    "F444W": (44400.0, 38000.0, 50000.0),
}


@dataclass
class ExposurePool:
    path: Path
    band: str
    header: dict[str, Any]
    uv: Tensor
    value: Tensor
    err: Tensor

    @property
    def size(self) -> int:
        return int(self.value.numel())


def _parse_csv(text: str) -> list[str]:
    return [item.strip().upper() for item in text.split(",") if item.strip()]


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_band(path: Path) -> str | None:
    try:
        with fits.open(path, memmap=True) as hdul:
            primary = hdul[0].header
            sci = hdul["SCI"].header if "SCI" in hdul else primary
            for key in ("FILTER", "PUPIL"):
                value = str(sci.get(key) or primary.get(key) or "").strip().upper()
                if value.startswith("F") and value.endswith("W"):
                    return value
    except (OSError, KeyError, ValueError):
        return None
    return None


def discover(input_dir: Path, bands: list[str]) -> list[tuple[Path, str]]:
    selected: list[tuple[Path, str]] = []
    for band in bands:
        band_dir = input_dir / band
        paths = sorted(band_dir.glob("*_crf.fits")) if band_dir.exists() else []
        verified = [(path, _read_band(path)) for path in paths]
        verified = [(path, found) for path, found in verified if found == band]
        if not verified:
            raise FileNotFoundError(
                f"no verified {band} CRFs under {band_dir}; run "
                "scripts/prepare_smacs0723_multiband_crf.py first"
            )
        selected.extend((path.resolve(), band) for path, _ in verified)
    return selected


def _valid_mask(hdul: fits.HDUList, image: np.ndarray, err: np.ndarray) -> np.ndarray:
    valid = np.isfinite(image) & np.isfinite(err) & (err > 0)
    if "DQ" in hdul:
        dq = np.asarray(hdul["DQ"].data)
        valid &= np.isfinite(dq) & (dq == 0)
    if "UPDCMASK" in hdul:
        extra = np.asarray(hdul["UPDCMASK"].data)
        valid &= np.isfinite(extra) & (extra == 0)
    return valid


def load_pool(path: Path, band: str, max_pixels: int, rng: np.random.Generator) -> ExposurePool:
    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(hdul["SCI"].data, dtype=np.float32)
        err = np.asarray(hdul["ERR"].data, dtype=np.float32)
        if image.ndim != 2 or err.shape != image.shape:
            raise ValueError(f"invalid SCI/ERR shapes in {path}")
        valid = np.flatnonzero(_valid_mask(hdul, image, err).reshape(-1))
        if valid.size == 0:
            raise ValueError(f"no valid pixels in {path}")
        if max_pixels > 0 and valid.size > max_pixels:
            valid = rng.choice(valid, size=max_pixels, replace=False)
        y, x = np.unravel_index(valid, image.shape)
        header = dict(hdul["SCI"].header)
        header["NAXIS1"] = int(image.shape[1])
        header["NAXIS2"] = int(image.shape[0])
        return ExposurePool(
            path=path,
            band=band,
            header=header,
            uv=torch.from_numpy(np.stack((x, y), -1).astype(np.float32)),
            value=torch.from_numpy(image.reshape(-1)[valid].copy()),
            err=torch.from_numpy(err.reshape(-1)[valid].copy()),
        )


def intensity_scale(pools: list[ExposurePool]) -> float:
    pieces: list[np.ndarray] = []
    for pool in pools:
        values = pool.value.numpy()
        if values.size > 50000:
            values = values[:: max(1, values.size // 50000)]
        pieces.append(values[np.isfinite(values)])
    merged = np.concatenate(pieces)
    med = float(np.median(merged))
    mad = float(np.median(np.abs(merged - med)))
    p95 = float(np.percentile(np.abs(merged), 95))
    return max(1.4826 * mad, p95, 1e-6)


def normalized_pools(pools: list[ExposurePool], scale: float) -> list[ExposurePool]:
    return [
        ExposurePool(p.path, p.band, p.header, p.uv, p.value / scale, p.err / scale)
        for p in pools
    ]


def build_bandpass(pools: list[ExposurePool], samples: int) -> BandpassTable:
    wavelength: list[np.ndarray] = []
    weight: list[np.ndarray] = []
    for pool in pools:
        _, lo, hi = BAND_SPECS[pool.band]
        wave = np.linspace(lo, hi, samples, dtype=np.float32)
        w = np.ones(samples, dtype=np.float32)
        if samples > 1:
            w[[0, -1]] = 0.5
        wavelength.append(wave)
        weight.append(w / w.sum())
    return BandpassTable(
        torch.from_numpy(np.stack(wavelength)),
        torch.from_numpy(np.stack(weight)),
    ).normalized()


def build_psf(
    pools: list[ExposurePool],
    bandpass: BandpassTable,
    *,
    mode: str,
    size: int,
    fwhm_text: str,
) -> PSFKernelBank | None:
    if mode == "none":
        return None
    mapping: dict[str, float] = {}
    for token in fwhm_text.split(","):
        if token.strip():
            key, value = token.split("=", 1)
            mapping[key.strip().upper()] = float(value)
    kernels: list[Tensor] = []
    for pool in pools:
        if pool.band not in mapping:
            raise ValueError(f"missing --psf-fwhm-pix {pool.band}=value")
        kernel = gaussian_psf_kernel(mapping[pool.band], size)
        kernels.append(kernel[None].repeat(bandpass.wavelength.shape[1], 1, 1))
    return PSFKernelBank(torch.stack(kernels)).normalized()


def sample_batch(
    pools: list[ExposurePool], batch_size: int, generator: torch.Generator
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    each = math.ceil(batch_size / len(pools))
    uv, exp, value, err = [], [], [], []
    for e, pool in enumerate(pools):
        index = torch.randint(0, pool.size, (each,), generator=generator)
        uv.append(pool.uv[index])
        value.append(pool.value[index])
        err.append(pool.err[index])
        exp.append(torch.full((each,), e, dtype=torch.long))
    uv_t = torch.cat(uv)[:batch_size]
    exp_t = torch.cat(exp)[:batch_size]
    value_t = torch.cat(value)[:batch_size]
    err_t = torch.cat(err)[:batch_size]
    return uv_t, exp_t, value_t, err_t, torch.ones_like(value_t, dtype=torch.bool)


def train_variant(
    encoding: str,
    pools: list[ExposurePool],
    scale: float,
    bandpass: BandpassTable,
    psf: PSFKernelBank | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = output_dir / encoding
    run_dir.mkdir(parents=True, exist_ok=True)
    canvas = build_union_canvas(
        [p.header for p in pools],
        pixel_scale_arcsec=args.union_pixel_scale_arcsec,
        padding_pixels=args.union_padding_pixels,
    )
    model = WCSAwareMultiWavelengthINR(
        [p.header for p in pools],
        canvas,
        encoding=encoding,
        spatial_frequencies=args.spatial_frequencies,
        spherical_min_frequency=args.spherical_min_frequency,
        spherical_alignment_rank=args.spherical_alignment_rank,
        spectral_frequencies=args.spectral_frequencies,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        anchor_exposure=args.anchor_exposure,
        learn_background=not args.no_background,
    )
    device = torch.device(args.device)
    model.to(device)
    bandpass = bandpass.to(device)
    psf = psf.to(device) if psf is not None else None

    align_params = list(model.feature_alignment.parameters())
    align_ids = {id(p) for p in align_params}
    main_params = [p for p in model.parameters() if p.requires_grad and id(p) not in align_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": args.learning_rate},
            {"params": align_params, "lr": args.alignment_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 101)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        uv, exposure_id, target, err, mask = sample_batch(pools, args.batch_size, generator)
        uv, exposure_id = uv.to(device), exposure_id.to(device)
        target, err, mask = target.to(device), err.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            prediction, geometry_valid = model.predict_observation(
                uv, exposure_id, bandpass, psf=psf
            )
            data_loss, metrics = weighted_robust_loss(
                prediction, target, err, mask & geometry_valid,
                kind=args.robust_loss,
                delta=args.robust_delta,
                max_relative_weight=args.max_relative_weight,
            )
            align_reg = model.alignment_regularization()
            bg_reg = model.background.square().mean()
            loss = data_loss + args.alignment_regularization * align_reg + args.background_regularization * bg_reg
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": float(step),
                "loss": float(loss.detach()),
                "data_loss": float(data_loss.detach()),
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "align_reg": float(align_reg.detach()),
                "seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(encoding, row, flush=True)
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                run_dir / f"checkpoint_{step:06d}.pt",
            )

    checkpoint = run_dir / "checkpoint_final.pt"
    torch.save(
        {
            "format": "torchsky-wcs-aware-multiband-inr-v1",
            "model": model.state_dict(),
            "intensity_scale": scale,
            "union_header": canvas.header,
            "union_shape": canvas.shape,
            "exposure_paths": [str(p.path) for p in pools],
            "exposure_bands": [p.band for p in pools],
            "args": vars(args),
        },
        checkpoint,
    )
    summary = {
        "encoding": encoding,
        "checkpoint": str(checkpoint),
        "input_exposures": [str(p.path) for p in pools],
        "counts_by_band": {
            band: sum(p.band == band for p in pools) for band in BAND_SPECS
        },
        "union_shape": list(canvas.shape),
        "union_header": canvas.header,
        "intensity_scale": scale,
        "alignment": model.alignment_summary(),
        "background_physical": (model.background.detach().cpu() * scale).tolist(),
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exposure-dir", type=Path,
        default=Path("/arc/projects/ots/jwst_smacs0723/input"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/arc/projects/ots/jwst_smacs0723/results/wcs_aware_multiband_inr"),
    )
    parser.add_argument("--bands", default=",".join(BAND_SPECS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-pixels-per-exposure", type=int, default=200000)
    parser.add_argument("--position-encoding", choices=("fourier", "spherical"), default="fourier")
    parser.add_argument("--run-both-encodings", action="store_true")
    parser.add_argument("--spatial-frequencies", type=int, default=10)
    parser.add_argument("--spherical-min-frequency", type=float, default=1.0)
    parser.add_argument("--spherical-alignment-rank", type=int, default=8)
    parser.add_argument("--spectral-frequencies", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--union-pixel-scale-arcsec", type=float, default=None)
    parser.add_argument("--union-padding-pixels", type=int, default=8)
    parser.add_argument("--anchor-exposure", type=int, default=0)
    parser.add_argument("--bandpass-samples", type=int, default=5)
    parser.add_argument("--psf-mode", choices=("none", "gaussian"), default="none")
    parser.add_argument("--psf-size", type=int, default=7)
    parser.add_argument("--psf-fwhm-pix", default="")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--alignment-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--robust-loss", choices=("huber", "charbonnier"), default="huber")
    parser.add_argument("--robust-delta", type=float, default=0.1)
    parser.add_argument("--max-relative-weight", type=float, default=100.0)
    parser.add_argument("--alignment-regularization", type=float, default=1e-4)
    parser.add_argument("--background-regularization", type=float, default=1e-5)
    parser.add_argument("--no-background", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    bands = _parse_csv(args.bands)
    input_dir = args.exposure_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = discover(input_dir, bands)
    print(f"selected {len(selected)} verified CRF exposures")
    for index, (path, band) in enumerate(selected):
        print(f"[{index:02d}] {band} {path}")
    rng = np.random.default_rng(args.seed)
    raw = [
        load_pool(path, band, args.max_pixels_per_exposure, rng)
        for path, band in selected
    ]
    scale = intensity_scale(raw)
    pools = normalized_pools(raw, scale)
    bandpass = build_bandpass(pools, args.bandpass_samples)
    psf = build_psf(
        pools, bandpass,
        mode=args.psf_mode,
        size=args.psf_size,
        fwhm_text=args.psf_fwhm_pix,
    )
    encodings = ["fourier", "spherical"] if args.run_both_encodings else [args.position_encoding]
    summaries = [
        train_variant(encoding, pools, scale, bandpass, psf, output_dir, args)
        for encoding in encodings
    ]
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
