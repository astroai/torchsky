#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

BANDS = ["F090W", "F150W", "F277W", "F444W"]

REFERENCE_NAMES = {
    "F090W": "F090Wb_UPDC-v2_iter0_i2d.fits.fz",
    "F150W": "F150Wb_UPDC-v2_iter0_i2d.fits.fz",
    "F277W": "F277Wb_UPDC-v2_iter0_i2d.fits.fz",
    "F444W": "F444Wb_UPDC-v2_iter0_i2d.fits.fz",
}

CROP_SIZE = 256
STRIDE = 192

# Define tile centers in F090W pixel coordinates.
# These are based on your current F090W center.
F090W_CENTER = (4791.5, 4791.5)

TILES_F090W_PIX = {
    "center": F090W_CENTER,
    "right": (F090W_CENTER[0] + STRIDE, F090W_CENTER[1]),
    "down": (F090W_CENTER[0], F090W_CENTER[1] + STRIDE),
    "right_down": (F090W_CENTER[0] + STRIDE, F090W_CENTER[1] + STRIDE),
}


def get_sci_wcs(path: Path) -> WCS:
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul["SCI"] if "SCI" in hdul else hdul[0]
        return WCS(hdu.header)


def main() -> int:
    updec_dir = DATA_ROOT / "updec_webb"
    out_root = DATA_ROOT / "smacs0723_tiles_wcs"
    out_root.mkdir(parents=True, exist_ok=True)

    # Use F090W as the reference sky grid for defining tile centers.
    ref_path = updec_dir / REFERENCE_NAMES["F090W"]
    ref_wcs = get_sci_wcs(ref_path)

    # Convert F090W tile pixel centers to sky coordinates.
    tile_sky = {}
    for tile, (x, y) in TILES_F090W_PIX.items():
        sky = ref_wcs.pixel_to_world(x, y)
        tile_sky[tile] = {
            "ra_deg": float(sky.ra.deg),
            "dec_deg": float(sky.dec.deg),
            "f090w_pixel_xy": [float(x), float(y)],
        }

    # For each sky tile center, convert RA/Dec into each band's own pixel coordinates.
    all_centers = {}

    for tile, sky_info in tile_sky.items():
        all_centers[tile] = {}

        # Use the SkyCoord object directly from F090W WCS conversion.
        x_ref, y_ref = TILES_F090W_PIX[tile]
        sky = ref_wcs.pixel_to_world(x_ref, y_ref)

        for band in BANDS:
            src_path = updec_dir / REFERENCE_NAMES[band]
            band_wcs = get_sci_wcs(src_path)

            x_band, y_band = band_wcs.world_to_pixel(sky)
            all_centers[tile][band] = {
                "center_x": float(x_band),
                "center_y": float(y_band),
                "ra_deg": float(sky.ra.deg),
                "dec_deg": float(sky.dec.deg),
            }

            out_dir = out_root / tile / band

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "build_smacs0723_minicutout.py"),
                "--bands",
                band,
                "--crop-size",
                str(CROP_SIZE),
                "--center-x",
                str(float(x_band)),
                "--center-y",
                str(float(y_band)),
                "--output-dir",
                str(out_dir),
            ]

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

    centers_path = out_root / "wcs_tile_centers.json"
    centers_path.write_text(
        json.dumps(
            {
                "reference_band": "F090W",
                "crop_size": CROP_SIZE,
                "stride_in_reference_pixels": STRIDE,
                "tile_sky": tile_sky,
                "band_pixel_centers": all_centers,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {centers_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())