#!/usr/bin/env python3
"""Render one chosen band from a trained WCS-aware INR checkpoint.

The output is the bandpass-integrated decoder scene on the saved union canvas:

    I_b(x, y) = sum_j w_{b,j} S_theta(x, y, lambda_{b,j}).

It is written in the same physical units as the training SCI arrays after
multiplication by the checkpoint intensity scale. The image is the latent scene
before exposure-specific background addition and before any explicit PSF
operation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torchsky  # noqa: E402
from torchsky.map.wcs_aware_inr import (  # noqa: E402
    UnionCanvas,
    WCSAwareMultiWavelengthINR,
)
from torchsky.map.wcs_aware_training import render_union_band  # noqa: E402


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
        with fits.open(path, memmap=True) as hdul:
            if "SCI" not in hdul or hdul["SCI"].data is None:
                raise ValueError(f"{path} has no valid SCI extension")
            image = hdul["SCI"].data
            if image.ndim != 2:
                raise ValueError(f"{path} SCI is not two-dimensional")
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
    model = WCSAwareMultiWavelengthINR(
        headers,
        canvas,
        encoding=str(args.get("position_encoding", "fourier")),
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


def _auto_coverage_path(checkpoint: Path) -> Path | None:
    candidates = (
        checkpoint.parent / "union_coverage_mask.fits",
        checkpoint.parent.parent / "union_coverage_mask.fits",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--band", required=True, help="For example F090W or F444W")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-pixels", type=int, default=131072)
    parser.add_argument(
        "--coverage-mask",
        type=Path,
        default=None,
        help=(
            "Optional union coverage FITS. When omitted, the script searches next "
            "to the checkpoint and one directory above it."
        ),
    )
    parser.add_argument(
        "--no-coverage-mask",
        action="store_true",
        help="Keep decoder values everywhere on the union canvas, including extrapolation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    band = str(args.band).strip().upper()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.tile_pixels <= 0:
        raise SystemExit("tile-pixels must be positive")

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    exposure_paths = [str(value) for value in saved["exposure_paths"]]
    exposure_bands = [str(value).upper() for value in saved["exposure_bands"]]
    matching = [index for index, value in enumerate(exposure_bands) if value == band]
    if not matching:
        available = ", ".join(dict.fromkeys(exposure_bands))
        raise SystemExit(f"band {band!r} is unavailable; choose from: {available}")
    representative = matching[0]

    headers = _read_headers(exposure_paths)
    canvas = _canvas_from_checkpoint(saved)
    model = _build_model(saved, headers, canvas).to(device).eval()
    wavelengths = torch.as_tensor(saved["bandpass_wavelength"]).float()
    weights = torch.as_tensor(saved["bandpass_weight"]).float()

    image = render_union_band(
        model,
        canvas,
        wavelengths,
        weights,
        representative,
        anchor_exposure=int(saved.get("args", {}).get("anchor_exposure", 0)),
        device=device,
        tile_pixels=int(args.tile_pixels),
    )
    image = image * float(saved["intensity_scale"])

    coverage_path: Path | None = None
    if not args.no_coverage_mask:
        coverage_path = (
            args.coverage_mask.expanduser().resolve()
            if args.coverage_mask is not None
            else _auto_coverage_path(checkpoint)
        )
        if coverage_path is not None:
            coverage = np.asarray(fits.getdata(coverage_path), dtype=bool)
            if coverage.shape != tuple(image.shape):
                raise ValueError(
                    f"coverage shape {coverage.shape} does not match image {tuple(image.shape)}"
                )
            coverage_t = torch.from_numpy(coverage).to(device=image.device)
            image = torch.where(coverage_t, image, torch.zeros_like(image))

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint.parent / f"{band}_decoder_bandpass_scene.fits"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    header = dict(canvas.header)
    header["BAND"] = band
    header["CHKSTEP"] = int(saved.get("step", 0))
    header["INTSCALE"] = float(saved["intensity_scale"])
    header["RENDER"] = "DECODER"
    header["BUNIT"] = str(headers[representative].get("BUNIT", "MJy/sr"))
    header["HISTORY"] = "Bandpass-integrated decoder scene; no background or PSF."
    if coverage_path is not None:
        header["HISTORY"] = f"Outside union coverage set to zero using {coverage_path.name}."

    torchsky.write(
        str(output),
        image.detach().cpu(),
        header=header,
        overwrite=True,
    )
    print(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(saved.get("step", 0)),
            "band": band,
            "representative_exposure": representative,
            "shape": list(image.shape),
            "physical_min": float(image.min().detach().cpu()),
            "physical_max": float(image.max().detach().cpu()),
            "coverage_mask": None if coverage_path is None else str(coverage_path),
            "output": str(output),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
