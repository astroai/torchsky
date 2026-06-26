#!/usr/bin/env python3
"""Render paper-scale exposure-center HybPSFs from HybPSF PC models.

The UPdec-Webb paper uses one centered PSF per exposure on a 1501x1501 grid
with PSR=0.5. HybPSF v1.0.0's public renderer bins its oversampled model back
to detector sampling, so this wrapper keeps the oversampled WebbPSF grid and
adds the fitted PCA residual on that same grid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def _key(path: Path) -> str:
    suffix = "_hybpsf_pc.fits"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def _center_crop(array: np.ndarray, size: int) -> np.ndarray:
    if array.shape[0] < size or array.shape[1] < size:
        raise ValueError(f"cannot crop {array.shape} to {size}x{size}")
    y0 = (array.shape[0] - size) // 2
    x0 = (array.shape[1] - size) // 2
    return array[y0 : y0 + size, x0 : x0 + size]


def _embed_center(target: np.ndarray, stamp: np.ndarray) -> tuple[slice, slice]:
    y0 = (target.shape[0] - stamp.shape[0]) // 2
    x0 = (target.shape[1] - stamp.shape[1]) // 2
    ys = slice(y0, y0 + stamp.shape[0])
    xs = slice(x0, x0 + stamp.shape[1])
    target[ys, xs] += stamp
    return ys, xs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybpsf-root", type=Path, required=True)
    parser.add_argument("--pc-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=1501)
    parser.add_argument("--expected-models", type=int, default=9)
    parser.add_argument(
        "--max-models",
        type=int,
        default=1,
        help="Use 1 for a smoke test; use 0 to render every PC model.",
    )
    parser.add_argument("--center-x", type=float, default=1024.0)
    parser.add_argument("--center-y", type=float, default=1024.0)
    args = parser.parse_args()

    if args.target_size <= 0 or args.target_size % 2 != 1:
        raise ValueError("target-size must be a positive odd integer")
    if "WEBBPSF_PATH" not in os.environ:
        raise RuntimeError("WEBBPSF_PATH is not set")

    sys.path.insert(0, str(args.hybpsf_root.resolve()))
    import webbpsf  # type: ignore
    from HybPSF import HybPSF as implementation  # type: ignore

    implementation.detector["NRCBLONG"] = "NRCB5"
    implementation.gains["NRCBLONG"] = "1.80"

    pc_paths = sorted(args.pc_dir.glob("*_hybpsf_pc.fits"))
    if len(pc_paths) != int(args.expected_models):
        raise RuntimeError(
            f"expected {args.expected_models} PC models, found {len(pc_paths)}"
        )
    if args.max_models > 0:
        pc_paths = pc_paths[: int(args.max_models)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    position = np.asarray([[args.center_x, args.center_y]], dtype=float)

    for index, pc_path in enumerate(pc_paths, start=1):
        key = _key(pc_path)
        print(f"[{index}/{len(pc_paths)}] {key}: rendering...", flush=True)
        with fits.open(pc_path, memmap=False) as hdul:
            pcs = np.asarray(hdul[0].data, dtype=float)
            star_positions = np.asarray(hdul[1].data, dtype=float)
            coefficients = np.asarray(hdul[2].data, dtype=float)
            web_coefficients = np.asarray(hdul[3].data, dtype=float)
            header = hdul[0].header.copy()

        oversampling = int(header["OSAM"])
        if oversampling != 2:
            raise ValueError(f"{pc_path}: expected OSAM=2, got {oversampling}")
        if star_positions.shape[0] < 10:
            raise ValueError(
                f"{pc_path}: at least 10 stars required for cubic Webb scaling"
            )

        polyorder = 3 if star_positions.shape[0] > 10 else 2
        residual = implementation._coeff2psf(
            star_positions,
            coefficients,
            pcs,
            position,
            polyorder,
        )[0]
        web_scale = float(
            implementation._polyfit2d(
                star_positions,
                web_coefficients[:, 0],
                web_coefficients[:, 1],
                position,
                3,
            )[0]
        )

        instrument = webbpsf.NIRCam()
        instrument.filter = str(header["FILTER"])
        instrument.detector = implementation.detector[str(header["DETECTOR"])]
        instrument.load_wss_opd_by_date(str(header["DATE-BEG"]))
        instrument.detector_position = (args.center_x, args.center_y)

        detector_pixels = int(np.ceil(args.target_size / oversampling))
        result = instrument.calc_psf(
            fov_pixels=detector_pixels,
            oversample=oversampling,
        )
        web = np.asarray(result[2].data, dtype=float).T
        web = _center_crop(web, args.target_size)

        # Match the public HybPSF normalization: calibrate WebbPSF flux in
        # the residual support, scale it, add the PCA residual, then enforce
        # non-negativity and unit total flux.
        residual_support = np.zeros_like(web)
        ys, xs = _embed_center(residual_support, residual)
        support_flux = float(web[ys, xs].sum())
        if not np.isfinite(support_flux) or support_flux <= 0:
            raise ValueError(f"{pc_path}: invalid WebbPSF support flux")
        model = web * (web_scale / support_flux)
        _embed_center(model, residual)
        model = np.nan_to_num(model, nan=0.0, posinf=0.0, neginf=0.0)
        model = np.maximum(model, 0.0)
        total = float(model.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"{pc_path}: invalid rendered PSF sum")
        model /= total

        output = args.output_dir / f"{key}_hybpsf_1501.fits"
        out_header = fits.Header()
        out_header["PCFILE"] = pc_path.name
        out_header["FILTER"] = str(header["FILTER"])
        out_header["DETECTOR"] = str(header["DETECTOR"])
        out_header["DATE-BEG"] = str(header["DATE-BEG"])
        out_header["PSFX"] = float(args.center_x)
        out_header["PSFY"] = float(args.center_y)
        out_header["PSR"] = 0.5
        out_header["OVERSAMP"] = oversampling
        out_header["WEBBSCL"] = web_scale
        fits.PrimaryHDU(
            data=model.astype(np.float32),
            header=out_header,
        ).writeto(output, overwrite=True)

        peak = np.unravel_index(int(np.argmax(model)), model.shape)
        rows.append(
            {
                "pc_file": str(pc_path.resolve()),
                "output": str(output.resolve()),
                "shape": list(model.shape),
                "sum": float(model.sum()),
                "peak_yx": [int(peak[0]), int(peak[1])],
                "web_scale": web_scale,
                "stars": int(star_positions.shape[0]),
            }
        )
        print(
            f"[{index}/{len(pc_paths)}] {key}: OK | "
            f"shape {model.shape} | peak {peak}",
            flush=True,
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps({"outputs": rows}, indent=2))
    print(f"Completed {len(rows)} PSFs.")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
