#!/usr/bin/env python3
"""Export visualisation data from a trained WCS-aware SMACS J0723 INR.

Two products are written beneath ``/arc/projects/ots/jwst_smacs0723/data``:

1. bandpass-integrated union images, one FITS image per band;
2. direct decoder outputs ``S_theta(x, y, lambda_j)`` before bandpass
   integration, background addition, and optional PSF convolution.

Decoder cubes are stored as disk-backed NumPy arrays with shape ``[L,H,W]`` so
rendering does not require holding the full cube in GPU or system memory.
"""

from __future__ import annotations

import argparse
import json
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
from torchsky.map.wcs_aware_training import (  # noqa: E402
    build_union_coverage_mask,
    render_union_band,
)
from torchsky.wcs import WCS  # noqa: E402


def _parse_csv(text: str) -> list[str]:
    return [item.strip().upper() for item in str(text).split(",") if item.strip()]


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
    pixel_scale_arcsec = math.sqrt(abs(determinant)) * 3600.0
    return UnionCanvas(
        header=header,
        shape=(shape[0], shape[1]),
        pixel_scale_arcsec=pixel_scale_arcsec,
    )


def _read_exposure_headers(paths: Sequence[str]) -> list[dict[str, Any]]:
    headers: list[dict[str, Any]] = []
    for text in paths:
        path = Path(text)
        with fits.open(path, memmap=True) as hdul:
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


def _resolve_encoding(
    checkpoint_path: Path,
    checkpoint_args: dict[str, Any],
    requested: str | None,
) -> str:
    if requested is not None:
        return requested
    parent_name = checkpoint_path.parent.name.lower()
    if parent_name in {"fourier", "spherical"}:
        return parent_name
    return str(checkpoint_args.get("position_encoding", "fourier"))


def _build_model(
    saved: dict[str, Any],
    headers: Sequence[dict[str, Any]],
    canvas: UnionCanvas,
    encoding: str,
) -> WCSAwareMultiWavelengthINR:
    args = dict(saved.get("args", {}))
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


@torch.inference_mode()
def _write_decoder_cube(
    model: WCSAwareMultiWavelengthINR,
    canvas: UnionCanvas,
    wavelength: torch.Tensor,
    output_path: Path,
    *,
    anchor_exposure: int,
    intensity_scale: float,
    coverage: torch.Tensor,
    device: torch.device,
    tile_pixels: int,
) -> None:
    if wavelength.ndim != 1 or wavelength.numel() == 0:
        raise ValueError("wavelength must be a non-empty 1D tensor")
    if tile_pixels <= 0:
        raise ValueError("tile_pixels must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = canvas.shape
    cube = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(wavelength.numel()), height, width),
    )

    model.eval()
    model.to(device)
    target_wcs = WCS(canvas.header).to(device)
    coverage_flat = coverage.reshape(-1).to(device=device)
    total_pixels = height * width

    for start in range(0, total_pixels, tile_pixels):
        stop = min(start + tile_pixels, total_pixels)
        index = torch.arange(start, stop, device=device)
        y = torch.div(index, width, rounding_mode="floor")
        x = index - y * width
        xy = torch.stack((x, y), dim=-1).float()

        if model.encoding == "fourier":
            raw = model.position_encoder(xy)
        else:
            ra, dec = target_wcs.pixel_to_world(
                xy[:, 0].double(),
                xy[:, 1].double(),
            )
            raw = model.position_encoder(torch.stack((ra, dec), dim=-1))

        exposure_id = torch.full(
            (xy.shape[0],),
            int(anchor_exposure),
            dtype=torch.long,
            device=device,
        )
        latent = model.trunk(model.feature_alignment(raw, exposure_id))
        valid = torch.ones(xy.shape[0], dtype=torch.bool, device=device)
        covered = coverage_flat[start:stop]

        for sample_index, sample_wavelength in enumerate(wavelength.tolist()):
            wave = torch.full(
                (xy.shape[0],),
                float(sample_wavelength),
                dtype=torch.float32,
                device=device,
            )
            value = model.decode(latent, wave, valid)
            value = torch.where(
                covered,
                value * float(intensity_scale),
                torch.zeros_like(value),
            )
            cube[sample_index].reshape(-1)[start:stop] = (
                value.detach().cpu().numpy().astype(np.float32, copy=False)
            )

    cube.flush()
    del cube


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-output-root",
        type=Path,
        default=Path(
            "/arc/projects/ots/jwst_smacs0723/data/wcs_aware_multiband_inr"
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Subdirectory under data-output-root; defaults to the training run name.",
    )
    parser.add_argument("--encoding", choices=("fourier", "spherical"), default=None)
    parser.add_argument("--bands", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-pixels", type=int, default=131072)
    parser.add_argument("--coverage-tile-rows", type=int, default=128)
    parser.add_argument("--skip-bandpass-integrated", action="store_true")
    parser.add_argument("--skip-decoder", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = dict(saved.get("args", {}))
    encoding = _resolve_encoding(checkpoint_path, checkpoint_args, args.encoding)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    exposure_paths = [str(value) for value in saved["exposure_paths"]]
    exposure_bands = [str(value).upper() for value in saved["exposure_bands"]]
    headers = _read_exposure_headers(exposure_paths)
    canvas = _canvas_from_checkpoint(saved)
    model = _build_model(saved, headers, canvas, encoding).to(device)

    bandpass_wavelength = torch.as_tensor(saved["bandpass_wavelength"]).float()
    bandpass_weight = torch.as_tensor(saved["bandpass_weight"]).float()
    intensity_scale = float(saved["intensity_scale"])
    anchor_exposure = int(checkpoint_args.get("anchor_exposure", 0))

    available_bands = list(dict.fromkeys(exposure_bands))
    selected_bands = _parse_csv(args.bands) if args.bands else available_bands
    unavailable = sorted(set(selected_bands).difference(available_bands))
    if unavailable:
        raise SystemExit("bands absent from checkpoint: " + ", ".join(unavailable))

    inferred_run_name = checkpoint_path.parent.parent.name
    run_name = str(args.run_name or inferred_run_name)
    data_root = args.data_output_root.expanduser().resolve() / run_name
    integrated_root = data_root / "bandpass_integrated" / encoding
    decoder_root = data_root / "decoder" / encoding
    data_root.mkdir(parents=True, exist_ok=True)

    coverage = build_union_coverage_mask(
        headers,
        canvas,
        device=device,
        tile_rows=args.coverage_tile_rows,
    )
    coverage_path = data_root / "union_coverage_mask.fits"
    torchsky.write(
        str(coverage_path),
        coverage.to(torch.uint8).cpu(),
        header=canvas.header,
        overwrite=True,
    )

    outputs: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "encoding": encoding,
        "run_name": run_name,
        "data_root": str(data_root),
        "coverage_mask": str(coverage_path),
        "bandpass_integrated_root": str(integrated_root),
        "decoder_root": str(decoder_root),
        "bands": {},
    }

    for band in selected_bands:
        representative = exposure_bands.index(band)
        wavelength = bandpass_wavelength[representative]
        weight = bandpass_weight[representative]
        band_outputs: dict[str, Any] = {
            "representative_exposure": representative,
            "wavelength_angstrom": wavelength.tolist(),
            "bandpass_weight": weight.tolist(),
        }

        if not args.skip_bandpass_integrated:
            integrated_root.mkdir(parents=True, exist_ok=True)
            image = render_union_band(
                model,
                canvas,
                bandpass_wavelength,
                bandpass_weight,
                representative,
                anchor_exposure=anchor_exposure,
                device=device,
                tile_pixels=args.tile_pixels,
            )
            image = torch.where(
                coverage,
                image * intensity_scale,
                torch.zeros_like(image),
            )
            integrated_path = integrated_root / f"{band}_{encoding}_bandpass.fits"
            torchsky.write(
                str(integrated_path),
                image.cpu(),
                header=canvas.header,
                overwrite=True,
            )
            band_outputs["bandpass_integrated"] = str(integrated_path)

        if not args.skip_decoder:
            band_decoder_root = decoder_root / band
            cube_path = band_decoder_root / f"{band}_{encoding}_decoder_cube.npy"
            _write_decoder_cube(
                model,
                canvas,
                wavelength,
                cube_path,
                anchor_exposure=anchor_exposure,
                intensity_scale=intensity_scale,
                coverage=coverage,
                device=device,
                tile_pixels=args.tile_pixels,
            )
            metadata = {
                "array_path": str(cube_path),
                "array_shape": [int(wavelength.numel()), *canvas.shape],
                "dtype": "float32",
                "axis_order": ["wavelength", "y", "x"],
                "wavelength_unit": "Angstrom",
                "wavelength_angstrom": wavelength.tolist(),
                "bandpass_weight": weight.tolist(),
                "stage": (
                    "decoder output after output activation; before bandpass "
                    "integration, exposure background, and PSF convolution"
                ),
                "anchor_exposure": anchor_exposure,
                "intensity_scale_applied": intensity_scale,
                "coverage_mask": str(coverage_path),
                "union_header": canvas.header,
            }
            metadata_path = band_decoder_root / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            band_outputs["decoder_cube"] = str(cube_path)
            band_outputs["decoder_metadata"] = str(metadata_path)

        outputs["bands"][band] = band_outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest_path = data_root / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
