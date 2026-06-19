#!/usr/bin/env python3
from pathlib import Path
import json

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

BANDS = ["F090W", "F150W", "F277W", "F444W"]
TILE_ROOT = Path("data/smacs0723_tiles")
OUT_DIR = TILE_ROOT / "qa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CROP_SIZE = 256
STRIDE = 192
OVERLAP = CROP_SIZE - STRIDE


def read_tile(tile: str, band: str):
    path = TILE_ROOT / tile / band / f"smacs0723_{band}_cutout_{CROP_SIZE}.fits"
    if not path.exists():
        raise FileNotFoundError(path)
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
    return data, header, path


def robust_show(ax, image, title):
    finite = np.isfinite(image)
    if not np.any(finite):
        ax.imshow(image, origin="lower")
        ax.set_title(title)
        return

    vmin, vmax = np.nanpercentile(image[finite], [1, 99.5])
    ax.imshow(image, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


summary = {}

for band in BANDS:
    center, hdr, center_path = read_tile("center", band)
    right, _, right_path = read_tile("right", band)
    down, _, down_path = read_tile("down", band)

    # Overlap checks
    cr_overlap = center[:, -OVERLAP:] - right[:, :OVERLAP]
    cd_overlap = center[-OVERLAP:, :] - down[:OVERLAP, :]

    summary[band] = {
        "center": str(center_path),
        "right": str(right_path),
        "down": str(down_path),
        "center_right_overlap_abs_median": float(np.nanmedian(np.abs(cr_overlap))),
        "center_down_overlap_abs_median": float(np.nanmedian(np.abs(cd_overlap))),
        "center_right_overlap_abs_p95": float(np.nanpercentile(np.abs(cr_overlap), 95)),
        "center_down_overlap_abs_p95": float(np.nanpercentile(np.abs(cd_overlap), 95)),
    }

    # Make a larger contiguous mosaic: 448 x 448
    h = CROP_SIZE + STRIDE
    w = CROP_SIZE + STRIDE
    mosaic = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    placements = [
        (center, 0, 0),
        (right, STRIDE, 0),
        (down, 0, STRIDE),
    ]

    for img, x0, y0 in placements:
        valid = np.isfinite(img)
        mosaic[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE][valid] += img[valid]
        weight[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE][valid] += 1.0

    mosaic = np.divide(
        mosaic,
        weight,
        out=np.full_like(mosaic, np.nan),
        where=weight > 0,
    )

    # Update WCS header for mosaic.
    # The mosaic origin is the same as the center tile origin.
    mhdr = hdr.copy()
    mhdr["NAXIS1"] = w
    mhdr["NAXIS2"] = h
    mhdr["BAND"] = band
    mhdr["TILEMODE"] = "center+right+down"
    mhdr["STRIDE"] = STRIDE
    mhdr["OVERLAP"] = OVERLAP

    out_fits = OUT_DIR / f"smacs0723_{band}_contiguous_448.fits"
    fits.PrimaryHDU(data=mosaic, header=mhdr).writeto(out_fits, overwrite=True)

    # Quicklook
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    robust_show(axes[0], center, f"{band} center")
    robust_show(axes[1], right, f"{band} right")
    robust_show(axes[2], down, f"{band} down")
    robust_show(axes[3], mosaic, f"{band} contiguous")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{band}_tiles_and_contiguous.png", dpi=160)
    plt.close(fig)

    # Overlap residual figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    robust_show(axes[0], cr_overlap, f"{band} center-right overlap residual")
    robust_show(axes[1], cd_overlap, f"{band} center-down overlap residual")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{band}_overlap_residuals.png", dpi=160)
    plt.close(fig)

summary_path = OUT_DIR / "adjacent_tile_check_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(f"Wrote QA products to {OUT_DIR}")
print(f"Summary: {summary_path}")