#!/usr/bin/env python3
"""Run the same source detector on intrinsic bandpass fields and UPDC iter10.

No PSF convolution is performed.

For every band, the script:
  1. loads the rendered intrinsic bandpass field;
  2. loads the corresponding UPDC iter10 mosaic;
  3. constructs a common valid footprint;
  4. runs exactly the same Background2D + threshold + segmentation + deblend
     configuration on both images;
  5. writes catalogs, segmentation maps, backgrounds, RMS maps, and summaries.

Defaults match the SMACS0723 A1 NeRF all-valid epoch run:
  threshold        = 5 local background RMS
  npixels          = 5
  connectivity     = 8
  deblend          = enabled
  deblend nlevels  = 32
  deblend contrast = 0.001
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.background import (
    Background2D,
    BiweightLocationBackground,
    MADStdBackgroundRMS,
)
from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources


DEFAULT_BANDS = ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")
DEFAULT_PREFIX = "A1_nerf_target_canvas_all_valid_epoch"
DEFAULT_INTRINSIC_PATTERN = (
    "{band}_" + DEFAULT_PREFIX + "_bandpass_scene.fits"
)
DEFAULT_UPDC_PATTERN = "{band}b_UPDC-v2_iter10_i2d.fits.fz"
DEFAULT_COVERAGE_NAME = DEFAULT_PREFIX + "_union_coverage_mask.fits"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing the rendered intrinsic bandpass FITS files.",
    )
    parser.add_argument(
        "--updc-dir",
        type=Path,
        default=Path("/arc/projects/ots/jwst_smacs0723/data"),
        help="Directory containing the UPDC iter10 FITS files.",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=list(DEFAULT_BANDS),
        help="Bands to process.",
    )
    parser.add_argument(
        "--intrinsic-pattern",
        default=DEFAULT_INTRINSIC_PATTERN,
        help="Intrinsic filename pattern inside run-dir; must contain {band}.",
    )
    parser.add_argument(
        "--updc-pattern",
        default=DEFAULT_UPDC_PATTERN,
        help="UPDC filename pattern inside updc-dir; must contain {band}.",
    )
    parser.add_argument(
        "--intrinsic-coverage",
        type=Path,
        default=None,
        help=(
            "Optional nonzero=valid intrinsic coverage FITS. When omitted, "
            f"RUN_DIR/{DEFAULT_COVERAGE_NAME} is used if present."
        ),
    )
    parser.add_argument(
        "--updc-zero-is-invalid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat zero-valued UPDC pixels as outside the valid footprint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output root. Default: "
            "RUN_DIR/intrinsic_vs_updc_detector_5sigma_deblend"
        ),
    )

    parser.add_argument("--threshold-sigma", type=float, default=5.0)
    parser.add_argument("--npixels", type=int, default=5)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)

    parser.add_argument("--background-box-size", type=int, default=60)
    parser.add_argument("--background-filter-size", type=int, default=5)
    parser.add_argument("--sigma-clip", type=float, default=3.0)
    parser.add_argument("--sigma-clip-maxiters", type=int, default=10)

    parser.add_argument(
        "--deblend",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--deblend-nlevels", type=int, default=32)
    parser.add_argument("--deblend-contrast", type=float, default=0.001)

    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_2d_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            array = np.asarray(hdu.data)
            if array.ndim == 2:
                return np.asarray(array, dtype=np.float64), hdu.header.copy()
    raise ValueError(f"No 2D image HDU found in {path}")


def write_fits(
    path: Path,
    data: np.ndarray,
    header: fits.Header,
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.asarray(data), header=header).writeto(
        path,
        overwrite=overwrite,
    )


def pixel_scale_arcsec(wcs: WCS) -> float:
    try:
        scales = (
            np.asarray(
                proj_plane_pixel_scales(wcs.celestial),
                dtype=float,
            )
            * 3600.0
        )
        if (
            scales.size >= 2
            and np.isfinite(scales[:2]).all()
            and bool((scales[:2] > 0).all())
        ):
            return float(np.sqrt(scales[0] * scales[1]))
    except Exception:
        pass
    return float("nan")


def quantity_values(value: Any) -> np.ndarray:
    return np.asarray(getattr(value, "value", value))


def empty_catalog() -> Table:
    return Table(
        names=(
            "source_id",
            "xcentroid",
            "ycentroid",
            "ra_deg",
            "dec_deg",
            "area_pix2",
            "segment_flux",
            "max_value",
            "local_background",
            "local_background_rms",
            "peak_sigma",
        ),
        dtype=(
            int,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ),
    )


def make_catalog(
    *,
    data_sub: np.ndarray,
    background: np.ndarray,
    background_rms: np.ndarray,
    valid: np.ndarray,
    segment: Any,
    wcs: WCS,
) -> Table:
    if segment is None:
        return empty_catalog()

    catalog = SourceCatalog(
        data_sub,
        segment,
        mask=~valid,
        background=background,
        wcs=wcs,
        progress_bar=False,
    )

    labels = np.asarray(catalog.labels, dtype=np.int64)
    x = quantity_values(catalog.xcentroid).astype(float)
    y = quantity_values(catalog.ycentroid).astype(float)
    area = quantity_values(catalog.area).astype(float)
    segment_flux = quantity_values(catalog.segment_flux).astype(float)
    max_value = quantity_values(catalog.max_value).astype(float)

    ra, dec = wcs.pixel_to_world_values(x, y)
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)

    xi = np.clip(np.rint(x).astype(int), 0, data_sub.shape[1] - 1)
    yi = np.clip(np.rint(y).astype(int), 0, data_sub.shape[0] - 1)
    local_background = background[yi, xi]
    local_rms = background_rms[yi, xi]

    with np.errstate(divide="ignore", invalid="ignore"):
        peak_sigma = max_value / local_rms

    table = Table()
    table["source_id"] = labels
    table["xcentroid"] = x
    table["ycentroid"] = y
    table["ra_deg"] = ra
    table["dec_deg"] = dec
    table["area_pix2"] = area
    table["segment_flux"] = segment_flux
    table["max_value"] = max_value
    table["local_background"] = local_background
    table["local_background_rms"] = local_rms
    table["peak_sigma"] = peak_sigma
    return table


def run_detector(
    *,
    label: str,
    band: str,
    image_path: Path,
    data: np.ndarray,
    header: fits.Header,
    common_valid: np.ndarray,
    output_dir: Path,
    threshold_sigma: float,
    npixels: int,
    connectivity: int,
    background_box_size: int,
    background_filter_size: int,
    sigma_clip_value: float,
    sigma_clip_maxiters: int,
    deblend: bool,
    deblend_nlevels: int,
    deblend_contrast: float,
    overwrite: bool,
) -> dict[str, Any]:
    wcs = WCS(header).celestial
    if wcs.pixel_n_dim != 2 or wcs.world_n_dim != 2:
        raise ValueError(f"{band} {label}: image lacks a 2D celestial WCS")

    sigma_clip = SigmaClip(
        sigma=float(sigma_clip_value),
        maxiters=int(sigma_clip_maxiters),
    )
    bkg = Background2D(
        data,
        box_size=(int(background_box_size), int(background_box_size)),
        filter_size=(int(background_filter_size), int(background_filter_size)),
        mask=~common_valid,
        sigma_clip=sigma_clip,
        bkg_estimator=BiweightLocationBackground(),
        bkgrms_estimator=MADStdBackgroundRMS(),
    )

    background = np.asarray(bkg.background, dtype=np.float64)
    background_rms = np.asarray(bkg.background_rms, dtype=np.float64)
    data_sub = data - background
    threshold = float(threshold_sigma) * background_rms

    segment = detect_sources(
        data_sub,
        threshold,
        npixels=int(npixels),
        connectivity=int(connectivity),
        mask=~common_valid,
    )
    initial_segments = 0 if segment is None else int(segment.nlabels)

    if segment is not None and deblend:
        segment = deblend_sources(
            data_sub,
            segment,
            npixels=int(npixels),
            nlevels=int(deblend_nlevels),
            contrast=float(deblend_contrast),
            connectivity=int(connectivity),
            progress_bar=False,
        )

    final_segments = 0 if segment is None else int(segment.nlabels)
    table = make_catalog(
        data_sub=data_sub,
        background=background,
        background_rms=background_rms,
        valid=common_valid,
        segment=segment,
        wcs=wcs,
    )

    safe_label = label.lower()
    catalog_path = output_dir / f"{band}_{safe_label}_sources.ecsv"
    segmentation_path = (
        output_dir / f"{band}_{safe_label}_segmentation.fits"
    )
    background_path = output_dir / f"{band}_{safe_label}_background.fits"
    rms_path = output_dir / f"{band}_{safe_label}_background_rms.fits"
    background_subtracted_path = (
        output_dir / f"{band}_{safe_label}_background_subtracted.fits"
    )
    summary_path = (
        output_dir / f"{band}_{safe_label}_detection_summary.json"
    )

    table.write(
        catalog_path,
        format="ascii.ecsv",
        overwrite=overwrite,
    )
    write_fits(
        background_path,
        background.astype(np.float32),
        header,
        overwrite=overwrite,
    )
    write_fits(
        rms_path,
        background_rms.astype(np.float32),
        header,
        overwrite=overwrite,
    )
    write_fits(
        background_subtracted_path,
        data_sub.astype(np.float32),
        header,
        overwrite=overwrite,
    )
    if segment is not None:
        write_fits(
            segmentation_path,
            np.asarray(segment.data, dtype=np.int32),
            header,
            overwrite=overwrite,
        )
    elif overwrite:
        segmentation_path.unlink(missing_ok=True)

    summary = {
        "band": band,
        "label": label,
        "image": str(image_path),
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "pixel_scale_arcsec": pixel_scale_arcsec(wcs),
        "common_valid_pixels": int(common_valid.sum()),
        "common_valid_fraction": float(common_valid.mean()),
        "detector": {
            "threshold_sigma": float(threshold_sigma),
            "npixels": int(npixels),
            "connectivity": int(connectivity),
            "background_box_size": int(background_box_size),
            "background_filter_size": int(background_filter_size),
            "sigma_clip": float(sigma_clip_value),
            "sigma_clip_maxiters": int(sigma_clip_maxiters),
            "deblend": bool(deblend),
            "deblend_nlevels": int(deblend_nlevels),
            "deblend_contrast": float(deblend_contrast),
        },
        "initial_segment_count": initial_segments,
        "final_source_count": int(len(table)),
        "final_segment_count": final_segments,
        "background_median": float(
            np.nanmedian(background[common_valid])
        ),
        "background_rms_median": float(
            np.nanmedian(background_rms[common_valid])
        ),
        "products": {
            "catalog": str(catalog_path),
            "segmentation": (
                str(segmentation_path) if segment is not None else None
            ),
            "background": str(background_path),
            "background_rms": str(rms_path),
            "background_subtracted": str(background_subtracted_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = build_parser().parse_args()

    if (
        not math.isfinite(args.threshold_sigma)
        or args.threshold_sigma <= 0
    ):
        raise SystemExit("--threshold-sigma must be finite and positive")
    if args.npixels <= 0:
        raise SystemExit("--npixels must be positive")
    if args.background_box_size <= 0:
        raise SystemExit("--background-box-size must be positive")
    if args.background_filter_size <= 0:
        raise SystemExit("--background-filter-size must be positive")
    if not 0.0 <= args.deblend_contrast <= 1.0:
        raise SystemExit("--deblend-contrast must lie in [0, 1]")

    run_dir = args.run_dir.expanduser().resolve()
    updc_dir = args.updc_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not updc_dir.is_dir():
        raise FileNotFoundError(updc_dir)

    if "{band}" not in args.intrinsic_pattern:
        raise SystemExit("--intrinsic-pattern must contain {band}")
    if "{band}" not in args.updc_pattern:
        raise SystemExit("--updc-pattern must contain {band}")

    if args.intrinsic_coverage is not None:
        coverage_path = args.intrinsic_coverage.expanduser().resolve()
        if not coverage_path.exists():
            raise FileNotFoundError(coverage_path)
    else:
        candidate = run_dir / DEFAULT_COVERAGE_NAME
        coverage_path = candidate if candidate.exists() else None

    coverage_data: np.ndarray | None = None
    if coverage_path is not None:
        coverage_data, _ = read_2d_fits(coverage_path)

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "intrinsic_vs_updc_detector_5sigma_deblend"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    detector_config = {
        "threshold_sigma": float(args.threshold_sigma),
        "npixels": int(args.npixels),
        "connectivity": int(args.connectivity),
        "background_box_size": int(args.background_box_size),
        "background_filter_size": int(args.background_filter_size),
        "sigma_clip": float(args.sigma_clip),
        "sigma_clip_maxiters": int(args.sigma_clip_maxiters),
        "deblend": bool(args.deblend),
        "deblend_nlevels": int(args.deblend_nlevels),
        "deblend_contrast": float(args.deblend_contrast),
    }

    all_summaries: dict[str, Any] = {}

    for raw_band in args.bands:
        band = raw_band.upper()

        intrinsic_path = (
            run_dir / args.intrinsic_pattern.format(band=band)
        )
        updc_path = updc_dir / args.updc_pattern.format(band=band)

        if not intrinsic_path.exists():
            raise FileNotFoundError(intrinsic_path)
        if not updc_path.exists():
            raise FileNotFoundError(updc_path)

        intrinsic, intrinsic_header = read_2d_fits(intrinsic_path)
        updc, updc_header = read_2d_fits(updc_path)

        if intrinsic.shape != updc.shape:
            raise ValueError(
                f"{band}: intrinsic shape {intrinsic.shape} "
                f"!= UPDC shape {updc.shape}"
            )
        if coverage_data is not None and coverage_data.shape != intrinsic.shape:
            raise ValueError(
                f"{band}: coverage shape {coverage_data.shape} "
                f"!= image shape {intrinsic.shape}"
            )

        intrinsic_valid = np.isfinite(intrinsic)
        if coverage_data is not None:
            intrinsic_valid &= (
                np.isfinite(coverage_data) & (coverage_data > 0)
            )

        updc_valid = np.isfinite(updc)
        if args.updc_zero_is_invalid:
            updc_valid &= updc != 0

        common_valid = intrinsic_valid & updc_valid
        if not bool(common_valid.any()):
            raise ValueError(f"{band}: common valid footprint is empty")

        band_dir = output_root / band
        band_dir.mkdir(parents=True, exist_ok=True)

        common_mask_path = band_dir / f"{band}_common_valid_mask.fits"
        write_fits(
            common_mask_path,
            common_valid.astype(np.uint8),
            intrinsic_header,
            overwrite=args.overwrite,
        )

        print(
            f"[{band}] common valid pixels: "
            f"{int(common_valid.sum()):,}",
            flush=True,
        )

        intrinsic_summary = run_detector(
            label="INTRINSIC",
            band=band,
            image_path=intrinsic_path,
            data=intrinsic,
            header=intrinsic_header,
            common_valid=common_valid,
            output_dir=band_dir,
            overwrite=args.overwrite,
            **detector_config,
        )
        print(
            f"[{band}] intrinsic sources: "
            f"{intrinsic_summary['final_source_count']}",
            flush=True,
        )

        updc_summary = run_detector(
            label="UPDC",
            band=band,
            image_path=updc_path,
            data=updc,
            header=updc_header,
            common_valid=common_valid,
            output_dir=band_dir,
            overwrite=args.overwrite,
            **detector_config,
        )
        print(
            f"[{band}] UPDC sources: "
            f"{updc_summary['final_source_count']}",
            flush=True,
        )

        band_summary = {
            "band": band,
            "intrinsic_image": str(intrinsic_path),
            "updc_image": str(updc_path),
            "intrinsic_coverage": (
                None if coverage_path is None else str(coverage_path)
            ),
            "updc_zero_is_invalid": bool(args.updc_zero_is_invalid),
            "common_valid_mask": str(common_mask_path),
            "common_valid_pixels": int(common_valid.sum()),
            "common_valid_fraction": float(common_valid.mean()),
            "detector": detector_config,
            "intrinsic": intrinsic_summary,
            "updc": updc_summary,
            "source_count_difference_intrinsic_minus_updc": (
                int(intrinsic_summary["final_source_count"])
                - int(updc_summary["final_source_count"])
            ),
        }
        band_summary_path = band_dir / f"{band}_comparison_summary.json"
        band_summary_path.write_text(
            json.dumps(band_summary, indent=2),
            encoding="utf-8",
        )
        all_summaries[band] = band_summary

    aggregate = {
        "run_dir": str(run_dir),
        "updc_dir": str(updc_dir),
        "output_root": str(output_root),
        "intrinsic_coverage": (
            None if coverage_path is None else str(coverage_path)
        ),
        "comparison_policy": (
            "same detector configuration and same per-band common valid mask"
        ),
        "detector": detector_config,
        "bands": all_summaries,
    }
    aggregate_path = output_root / "all_bands_detection_summary.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote aggregate summary: {aggregate_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
