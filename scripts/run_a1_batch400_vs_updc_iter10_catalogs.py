#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


# ============================================================
# Fixed experiment paths
# ============================================================

REPO = Path("/arc/home/Xiaoxi/torchsky")

OUT = Path(
    "/arc/projects/ots/jwst_smacs0723/results/"
    "wcs_aware_multiband_inr/"
    "A1_nerf_UPDCgrid_all_valid_epoch_batch400_400_10k"
)

PREFIX = "A1_nerf_target_canvas_all_valid_epoch"

UPDC_ROOT = Path(
    "/arc/projects/ots/jwst_smacs0723/data"
)

CATALOG_ROOT = OUT / (
    "catalog_crossmatch_all_bands_"
    "A1_batch400_vs_UPDC_iter10"
)

COMPARE_SCRIPT = (
    REPO / "scripts/compare_wcs_aware_source_catalogs.py"
)

COVERAGE_PATH = (
    OUT / f"{PREFIX}_union_coverage_mask.fits"
)

BANDS = (
    "F090W",
    "F150W",
    "F200W",
    "F277W",
    "F356W",
    "F444W",
)

A1_IMAGES = {
    band: OUT / f"{band}_{PREFIX}_bandpass_scene.fits"
    for band in BANDS
}

UPDC_IMAGES = {
    band: UPDC_ROOT / f"{band}b_UPDC-v2_iter10_i2d.fits.fz"
    for band in BANDS
}


# ============================================================
# Catalog settings
#
# These preserve the current preliminary detector settings.
# They are diagnostic, not yet a final science catalog.
# ============================================================

THRESHOLD_SIGMA = 6.0
NPIXELS = 5
MATCH_RADIUS_ARCSEC = 0.05
MATCH_RADIUS_GRID = "0.02,0.03,0.05,0.075"
SNR_BINS = "6,8,10,20,inf"

# Set True only when intentionally testing deblending.
DEBLEND = False
DEBLEND_NLEVELS = 32
DEBLEND_CONTRAST = 0.001

MASK_BLOCK_ROWS = 64
WCS_TOLERANCE_ARCSEC = 1.0e-4


# ============================================================
# FITS helpers
# ============================================================

def read_first_2d(
    path: Path,
    *,
    load_data: bool,
) -> tuple[np.ndarray | None, fits.Header, tuple[int, int]]:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue

            if getattr(hdu.data, "ndim", None) != 2:
                continue

            shape = tuple(int(v) for v in hdu.data.shape)
            header = hdu.header.copy()

            data = (
                np.asarray(hdu.data)
                if load_data
                else None
            )

            return data, header, shape

    raise RuntimeError(f"No 2D image HDU found in {path}")


def validate_inputs() -> None:
    required = [
        COMPARE_SCRIPT,
        COVERAGE_PATH,
        *A1_IMAGES.values(),
        *UPDC_IMAGES.values(),
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        lines = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "The following required files are missing:\n"
            f"{lines}"
        )

    print("All required input files exist.", flush=True)


def grids_match(
    source_shape: tuple[int, int],
    source_wcs: WCS,
    target_shape: tuple[int, int],
    target_wcs: WCS,
) -> tuple[bool, float | None]:
    if source_shape != target_shape:
        return False, None

    height, width = source_shape

    x = np.asarray(
        [
            0,
            width - 1,
            0,
            width - 1,
            (width - 1) / 2,
        ],
        dtype=float,
    )

    y = np.asarray(
        [
            0,
            0,
            height - 1,
            height - 1,
            (height - 1) / 2,
        ],
        dtype=float,
    )

    source_sky = source_wcs.pixel_to_world(x, y)
    target_sky = target_wcs.pixel_to_world(x, y)

    if not isinstance(source_sky, SkyCoord):
        return False, None

    separation = source_sky.separation(target_sky).arcsec
    max_separation = float(np.nanmax(separation))

    return (
        bool(max_separation <= WCS_TOLERANCE_ARCSEC),
        max_separation,
    )


# ============================================================
# Coverage mapping
# ============================================================

coverage_data, coverage_header, coverage_shape = read_first_2d(
    COVERAGE_PATH,
    load_data=True,
)

assert coverage_data is not None

coverage_data = np.asarray(
    np.isfinite(coverage_data) & (coverage_data > 0),
    dtype=np.uint8,
)

coverage_wcs = WCS(coverage_header).celestial


def make_reference_coverage(
    *,
    reference_path: Path,
    destination: Path,
) -> dict[str, object]:
    _, reference_header, reference_shape = read_first_2d(
        reference_path,
        load_data=False,
    )

    reference_wcs = WCS(reference_header).celestial

    same_grid, max_wcs_separation = grids_match(
        coverage_shape,
        coverage_wcs,
        reference_shape,
        reference_wcs,
    )

    if same_grid:
        mapped = coverage_data.copy()
        method = "direct_same_grid"

    else:
        target_height, target_width = reference_shape
        mapped = np.zeros(
            (target_height, target_width),
            dtype=np.uint8,
        )

        x_values = np.arange(
            target_width,
            dtype=np.float64,
        )

        for y0 in range(
            0,
            target_height,
            MASK_BLOCK_ROWS,
        ):
            y1 = min(
                y0 + MASK_BLOCK_ROWS,
                target_height,
            )

            rows = y1 - y0

            xx = np.broadcast_to(
                x_values[None, :],
                (rows, target_width),
            )

            yy = np.broadcast_to(
                np.arange(
                    y0,
                    y1,
                    dtype=np.float64,
                )[:, None],
                (rows, target_width),
            )

            ra, dec = reference_wcs.pixel_to_world_values(
                xx,
                yy,
            )

            source_x, source_y = (
                coverage_wcs.world_to_pixel_values(
                    ra,
                    dec,
                )
            )

            valid = (
                np.isfinite(source_x)
                & np.isfinite(source_y)
                & (source_x >= -0.5)
                & (
                    source_x
                    <= coverage_shape[1] - 0.5
                )
                & (source_y >= -0.5)
                & (
                    source_y
                    <= coverage_shape[0] - 0.5
                )
            )

            source_ix = np.zeros(
                source_x.shape,
                dtype=np.int64,
            )
            source_iy = np.zeros(
                source_y.shape,
                dtype=np.int64,
            )

            source_ix[valid] = np.rint(
                source_x[valid]
            ).astype(np.int64)

            source_iy[valid] = np.rint(
                source_y[valid]
            ).astype(np.int64)

            valid &= (
                (source_ix >= 0)
                & (
                    source_ix
                    < coverage_shape[1]
                )
                & (source_iy >= 0)
                & (
                    source_iy
                    < coverage_shape[0]
                )
            )

            block = np.zeros(
                (rows, target_width),
                dtype=np.uint8,
            )

            block[valid] = coverage_data[
                source_iy[valid],
                source_ix[valid],
            ]

            mapped[y0:y1] = block

        method = "nearest_pixel_wcs_mapping"

    output_header = reference_wcs.to_header(relax=True)

    output_header["MASKSRC"] = (
        str(COVERAGE_PATH),
        "Source coverage mask",
    )
    output_header["MASKMETH"] = (
        method,
        "Coverage mapping method",
    )

    fits.PrimaryHDU(
        mapped,
        header=output_header,
    ).writeto(
        destination,
        overwrite=True,
    )

    return {
        "path": str(destination),
        "mapping_method": method,
        "shape": list(reference_shape),
        "valid_fraction": float(mapped.mean()),
        "same_grid": same_grid,
        "max_grid_separation_arcsec": (
            max_wcs_separation
        ),
    }


# ============================================================
# Summary helpers
# ============================================================

def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def find_snr_bin(
    rows: list[dict],
    lower: float,
    upper: float,
) -> dict | None:
    for row in rows:
        lo = float(row["snr_min_inclusive"])
        hi = float(row["snr_max_exclusive"])

        if lo == lower and hi == upper:
            return row

    return None


# ============================================================
# Main
# ============================================================

def main() -> int:
    validate_inputs()

    CATALOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_mask_dir = (
        CATALOG_ROOT / "reference_masks"
    )
    reference_mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_rows: list[dict] = []
    run_metadata: dict[str, object] = {
        "experiment_directory": str(OUT),
        "coverage_mask": str(COVERAGE_PATH),
        "catalog_root": str(CATALOG_ROOT),
        "configuration": {
            "threshold_sigma": THRESHOLD_SIGMA,
            "npixels": NPIXELS,
            "deblend": DEBLEND,
            "deblend_nlevels": DEBLEND_NLEVELS,
            "deblend_contrast": DEBLEND_CONTRAST,
            "match_radius_arcsec": (
                MATCH_RADIUS_ARCSEC
            ),
            "match_radius_grid_arcsec": (
                MATCH_RADIUS_GRID
            ),
            "snr_bins": SNR_BINS,
        },
        "bands": {},
    }

    for band in BANDS:
        print(
            f"\n{'=' * 72}\n"
            f"{band}\n"
            f"{'=' * 72}",
            flush=True,
        )

        a1_image = A1_IMAGES[band]
        updc_image = UPDC_IMAGES[band]

        band_dir = CATALOG_ROOT / band
        band_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        updc_mask = (
            reference_mask_dir
            / f"{band}_UPDC_iter10_coverage.fits"
        )

        mask_info = make_reference_coverage(
            reference_path=updc_image,
            destination=updc_mask,
        )

        print("A1 image :", a1_image, flush=True)
        print("UPDC     :", updc_image, flush=True)
        print(
            "mask map :",
            mask_info["mapping_method"],
            flush=True,
        )
        print(
            "mask valid fraction:",
            mask_info["valid_fraction"],
            flush=True,
        )
        print(
            "maximum same-pixel WCS separation:",
            mask_info[
                "max_grid_separation_arcsec"
            ],
            "arcsec",
            flush=True,
        )

        command = [
            sys.executable,
            str(COMPARE_SCRIPT),

            "--band",
            band,

            "--image",
            f"A1_batch400={a1_image}",

            "--image",
            f"UPDC_iter10={updc_image}",

            "--coverage",
            f"A1_batch400={COVERAGE_PATH}",

            "--coverage",
            f"UPDC_iter10={updc_mask}",

            "--reference-label",
            "UPDC_iter10",

            "--output-dir",
            str(band_dir),

            "--threshold-sigma",
            str(THRESHOLD_SIGMA),

            "--npixels",
            str(NPIXELS),

            "--match-radius-arcsec",
            str(MATCH_RADIUS_ARCSEC),

            "--match-radius-grid-arcsec",
            MATCH_RADIUS_GRID,

            "--snr-bins",
            SNR_BINS,

            "--overwrite",
        ]

        if DEBLEND:
            command.extend(
                [
                    "--deblend",
                    "--deblend-nlevels",
                    str(DEBLEND_NLEVELS),
                    "--deblend-contrast",
                    str(DEBLEND_CONTRAST),
                ]
            )

        print(
            "\nRunning catalog comparison...",
            flush=True,
        )

        subprocess.run(
            command,
            check=True,
            cwd=REPO,
        )

        summary_path = (
            band_dir
            / "catalog_comparison_summary.json"
        )

        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        images = summary["images"]

        match = summary[
            "matches_to_reference"
        ]["A1_batch400"]

        separation = match["separation_arcsec"]

        source_bins = match["snr_bins"][
            "source_peak_sigma_bins"
        ]
        reference_bins = match["snr_bins"][
            "reference_peak_sigma_bins"
        ]

        a1_6_8 = find_snr_bin(
            source_bins,
            6.0,
            8.0,
        )
        a1_8_10 = find_snr_bin(
            source_bins,
            8.0,
            10.0,
        )
        updc_6_8 = find_snr_bin(
            reference_bins,
            6.0,
            8.0,
        )
        updc_8_10 = find_snr_bin(
            reference_bins,
            8.0,
            10.0,
        )

        row = {
            "band": band,

            "a1_count_full": (
                images["A1_batch400"][
                    "source_count_full"
                ]
            ),
            "updc_count_full": (
                images["UPDC_iter10"][
                    "source_count_full"
                ]
            ),

            "a1_count_common": (
                images["A1_batch400"][
                    "source_count_common"
                ]
            ),
            "updc_count_common": (
                images["UPDC_iter10"][
                    "source_count_common"
                ]
            ),

            "matched_pairs": (
                match["matched_pairs"]
            ),

            "a1_matched_fraction": (
                match["source_matched_fraction"]
            ),

            "updc_recovered_fraction": (
                match[
                    "reference_recovered_fraction"
                ]
            ),

            "source_only": match["source_only"],
            "reference_only": (
                match["reference_only"]
            ),

            "median_separation_arcsec": (
                finite_or_none(
                    separation["median"]
                )
            ),
            "q68_separation_arcsec": (
                finite_or_none(
                    separation["q68"]
                )
            ),
            "q95_separation_arcsec": (
                finite_or_none(
                    separation["q95"]
                )
            ),

            "a1_snr_6_8_count": (
                None
                if a1_6_8 is None
                else a1_6_8["count"]
            ),
            "a1_snr_6_8_matched_fraction": (
                None
                if a1_6_8 is None
                else finite_or_none(
                    a1_6_8["matched_fraction"]
                )
            ),

            "a1_snr_8_10_count": (
                None
                if a1_8_10 is None
                else a1_8_10["count"]
            ),
            "a1_snr_8_10_matched_fraction": (
                None
                if a1_8_10 is None
                else finite_or_none(
                    a1_8_10["matched_fraction"]
                )
            ),

            "updc_snr_6_8_count": (
                None
                if updc_6_8 is None
                else updc_6_8["count"]
            ),
            "updc_snr_6_8_recovered_fraction": (
                None
                if updc_6_8 is None
                else finite_or_none(
                    updc_6_8[
                        "matched_fraction"
                    ]
                )
            ),

            "updc_snr_8_10_count": (
                None
                if updc_8_10 is None
                else updc_8_10["count"]
            ),
            "updc_snr_8_10_recovered_fraction": (
                None
                if updc_8_10 is None
                else finite_or_none(
                    updc_8_10[
                        "matched_fraction"
                    ]
                )
            ),

            "a1_background_median": (
                images["A1_batch400"][
                    "background_median"
                ]
            ),
            "a1_background_rms_median": (
                images["A1_batch400"][
                    "background_rms_median"
                ]
            ),

            "updc_background_median": (
                images["UPDC_iter10"][
                    "background_median"
                ]
            ),
            "updc_background_rms_median": (
                images["UPDC_iter10"][
                    "background_rms_median"
                ]
            ),

            "coverage_mapping_method": (
                mask_info["mapping_method"]
            ),
            "coverage_valid_fraction": (
                mask_info["valid_fraction"]
            ),

            "summary_json": str(summary_path),
        }

        combined_rows.append(row)

        run_metadata["bands"][band] = {
            "a1_image": str(a1_image),
            "updc_image": str(updc_image),
            "coverage": mask_info,
            "catalog_summary": str(summary_path),
        }

    csv_path = (
        CATALOG_ROOT
        / "all_bands_catalog_summary.csv"
    )

    json_path = (
        CATALOG_ROOT
        / "all_bands_catalog_summary.json"
    )

    metadata_path = (
        CATALOG_ROOT
        / "all_bands_run_metadata.json"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                combined_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    json_path.write_text(
        json.dumps(
            combined_rows,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    metadata_path.write_text(
        json.dumps(
            run_metadata,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\n{'=' * 92}\n"
        "ALL-BAND SUMMARY\n"
        f"{'=' * 92}"
    )

    print(
        f"{'band':7s}"
        f"{'A1':>10s}"
        f"{'UPDC':>10s}"
        f"{'match':>9s}"
        f"{'A1 match':>12s}"
        f"{'UPDC rec.':>12s}"
        f"{'median sep':>14s}"
    )

    for row in combined_rows:
        median = row[
            "median_separation_arcsec"
        ]

        median_text = (
            "nan"
            if median is None
            else f'{median:.4f}"'
        )

        print(
            f"{row['band']:7s}"
            f"{row['a1_count_common']:10d}"
            f"{row['updc_count_common']:10d}"
            f"{row['matched_pairs']:9d}"
            f"{row['a1_matched_fraction']:11.2%}"
            f"{row['updc_recovered_fraction']:11.2%}"
            f"{median_text:>14s}"
        )

    print("\nSaved:")
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", metadata_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
