#!/usr/bin/env python3
"""Run the existing catalog comparison using one external detection RMS map.

Each image still receives its own Background2D background subtraction:

    data_sub = image - own_background

But every image uses the same externally supplied RMS map for detection:

    threshold = threshold_sigma * shared_rms

The original compare_wcs_aware_source_catalogs.py remains unchanged.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.wcs import WCS
from photutils.background import (
    Background2D,
    BiweightLocationBackground,
    MADStdBackgroundRMS,
)
from photutils.segmentation import (
    SourceCatalog,
    deblend_sources,
    detect_sources,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.compare_wcs_aware_source_catalogs as base  # noqa: E402


_SHARED_RMS_PATH: Path | None = None
_SHARED_RMS: np.ndarray | None = None
_SHARED_RMS_HEADER: fits.Header | None = None
_SAVE_INITIAL_SEGMENTATION = False
_SAVE_DETECTION_THRESHOLD = False


def _wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--threshold-rms-fits",
        type=Path,
        required=True,
        help=(
            "External RMS FITS used to define the detection threshold for every "
            "input image. It must have the same shape and WCS pixel grid."
        ),
    )
    parser.add_argument(
        "--save-initial-segmentation",
        action="store_true",
        help="Save the segmentation before deblending.",
    )
    parser.add_argument(
        "--save-detection-threshold",
        action="store_true",
        help="Save threshold_sigma multiplied by the shared RMS map.",
    )
    return parser


def _parse_wrapper_options(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    parser = _wrapper_parser()
    namespace, remaining = parser.parse_known_args(argv)
    return namespace, remaining


def _read_shared_rms(path: Path) -> tuple[np.ndarray, fits.Header]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if not resolved.is_file():
        raise ValueError(f"shared RMS path is not a file: {resolved}")

    with fits.open(resolved, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue

            array = np.asarray(hdu.data)
            if array.ndim != 2:
                continue

            rms = np.asarray(array, dtype=np.float32)
            header = hdu.header.copy()
            return rms, header

    raise ValueError(f"no two-dimensional image HDU found in {resolved}")


def _get_shared_rms() -> tuple[np.ndarray, fits.Header]:
    global _SHARED_RMS, _SHARED_RMS_HEADER

    if _SHARED_RMS_PATH is None:
        raise RuntimeError("shared RMS path has not been initialized")

    if _SHARED_RMS is None or _SHARED_RMS_HEADER is None:
        _SHARED_RMS, _SHARED_RMS_HEADER = _read_shared_rms(
            _SHARED_RMS_PATH
        )

    return _SHARED_RMS, _SHARED_RMS_HEADER


def _sampled_grid_offset_arcsec(
    image_wcs: WCS,
    rms_wcs: WCS,
    shape: tuple[int, int],
) -> float:
    height, width = shape

    x = np.asarray(
        [
            0.0,
            width - 1.0,
            0.0,
            width - 1.0,
            0.5 * (width - 1.0),
        ],
        dtype=float,
    )
    y = np.asarray(
        [
            0.0,
            0.0,
            height - 1.0,
            height - 1.0,
            0.5 * (height - 1.0),
        ],
        dtype=float,
    )

    image_ra, image_dec = image_wcs.pixel_to_world_values(x, y)
    rms_ra, rms_dec = rms_wcs.pixel_to_world_values(x, y)

    image_sky = SkyCoord(
        np.asarray(image_ra, dtype=float) * u.deg,
        np.asarray(image_dec, dtype=float) * u.deg,
    )
    rms_sky = SkyCoord(
        np.asarray(rms_ra, dtype=float) * u.deg,
        np.asarray(rms_dec, dtype=float) * u.deg,
    )

    offsets = image_sky.separation(rms_sky).arcsec
    return float(np.nanmax(offsets))


def _validate_shared_rms_grid(
    *,
    label: str,
    rms: np.ndarray,
    rms_header: fits.Header,
    image_shape: tuple[int, int],
    image_wcs: WCS,
) -> None:
    if rms.shape != image_shape:
        raise ValueError(
            f"{label}: shared RMS shape {rms.shape} does not match "
            f"image shape {image_shape}"
        )

    rms_wcs = WCS(rms_header).celestial
    if rms_wcs.pixel_n_dim != 2 or rms_wcs.world_n_dim != 2:
        raise ValueError(
            f"{label}: shared RMS FITS does not contain a valid 2D "
            "celestial WCS"
        )

    offset_arcsec = _sampled_grid_offset_arcsec(
        image_wcs,
        rms_wcs,
        image_shape,
    )

    if not math.isfinite(offset_arcsec):
        raise ValueError(
            f"{label}: could not compare image and shared RMS WCS grids"
        )

    if offset_arcsec > 1.0e-3:
        raise ValueError(
            f"{label}: shared RMS is not on the same WCS pixel grid; "
            f"maximum sampled offset is {offset_arcsec:.6g} arcsec"
        )


def _build_catalog_shared_rms(
    *,
    label: str,
    image_path: Path,
    coverage_path: Path | None,
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
) -> base.DetectionResult:
    data, geometry = base._geometry_for_image(
        label=label,
        image_path=image_path,
        coverage_path=coverage_path,
    )
    image_mask = ~geometry.valid_mask

    sigma_clip = SigmaClip(
        sigma=float(sigma_clip_value),
        maxiters=int(sigma_clip_maxiters),
    )

    background_model = Background2D(
        data,
        box_size=(
            int(background_box_size),
            int(background_box_size),
        ),
        filter_size=(
            int(background_filter_size),
            int(background_filter_size),
        ),
        mask=image_mask,
        sigma_clip=sigma_clip,
        bkg_estimator=BiweightLocationBackground(),
        bkgrms_estimator=MADStdBackgroundRMS(),
    )

    own_background = np.asarray(
        background_model.background,
        dtype=np.float64,
    )
    own_background_rms = np.asarray(
        background_model.background_rms,
        dtype=np.float64,
    )

    data_subtracted = data - own_background

    shared_rms, shared_rms_header = _get_shared_rms()
    _validate_shared_rms_grid(
        label=label,
        rms=shared_rms,
        rms_header=shared_rms_header,
        image_shape=data.shape,
        image_wcs=geometry.wcs,
    )

    detection_mask = (
        image_mask
        | ~np.isfinite(shared_rms)
        | (shared_rms <= 0)
    )

    threshold = (
        np.asarray(shared_rms, dtype=np.float32)
        * float(threshold_sigma)
    )

    initial_segment = detect_sources(
        data_subtracted,
        threshold,
        npixels=int(npixels),
        connectivity=int(connectivity),
        mask=detection_mask,
    )

    segment = initial_segment
    if segment is not None and deblend:
        segment = deblend_sources(
            data_subtracted,
            segment,
            npixels=int(npixels),
            nlevels=int(deblend_nlevels),
            contrast=float(deblend_contrast),
            connectivity=int(connectivity),
            progress_bar=False,
        )

    if segment is None:
        table = base._empty_catalog()
        segmentation_array = None
    else:
        catalog = SourceCatalog(
            data_subtracted,
            segment,
            mask=detection_mask,
            background=own_background,
            wcs=geometry.wcs,
            progress_bar=False,
        )

        source_id = np.asarray(catalog.labels, dtype=np.int64)
        x = base._quantity_values(catalog.xcentroid).astype(float)
        y = base._quantity_values(catalog.ycentroid).astype(float)
        area = base._quantity_values(catalog.area).astype(float)
        segment_flux = base._quantity_values(
            catalog.segment_flux
        ).astype(float)
        max_value = base._quantity_values(
            catalog.max_value
        ).astype(float)

        ra, dec = geometry.wcs.pixel_to_world_values(x, y)
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)

        xi = np.clip(
            np.rint(x).astype(int),
            0,
            data.shape[1] - 1,
        )
        yi = np.clip(
            np.rint(y).astype(int),
            0,
            data.shape[0] - 1,
        )

        local_background = own_background[yi, xi]
        local_detection_rms = shared_rms[yi, xi]

        with np.errstate(divide="ignore", invalid="ignore"):
            peak_sigma = max_value / local_detection_rms

        table = Table()
        table["image_label"] = np.repeat(label, len(source_id))
        table["source_id"] = source_id
        table["xcentroid"] = x
        table["ycentroid"] = y
        table["ra_deg"] = ra
        table["dec_deg"] = dec
        table["area_pix2"] = area
        table["segment_flux"] = segment_flux
        table["max_value"] = max_value
        table["local_background"] = local_background

        # This column is now the shared detection RMS, not the image's
        # independently estimated Background2D RMS.
        table["local_background_rms"] = local_detection_rms
        table["peak_sigma"] = peak_sigma

        segmentation_array = np.asarray(
            segment.data,
            dtype=np.int32,
        )

    safe_label = base._safe_label(label)

    catalog_path = (
        output_dir / f"{safe_label}_sources_full.ecsv"
    )
    background_path = (
        output_dir / f"{safe_label}_background.fits"
    )
    own_rms_path = (
        output_dir / f"{safe_label}_background_rms.fits"
    )
    segmentation_path = (
        output_dir / f"{safe_label}_segmentation.fits"
    )
    initial_segmentation_path = (
        output_dir / f"{safe_label}_initial_segmentation.fits"
    )
    threshold_path = (
        output_dir / f"{safe_label}_detection_threshold.fits"
    )

    base._write_table(
        catalog_path,
        table,
        overwrite=overwrite,
    )
    base._write_fits(
        background_path,
        own_background.astype(np.float32),
        geometry.header,
        overwrite=overwrite,
    )
    base._write_fits(
        own_rms_path,
        own_background_rms.astype(np.float32),
        geometry.header,
        overwrite=overwrite,
    )

    if segmentation_array is not None:
        base._write_fits(
            segmentation_path,
            segmentation_array,
            geometry.header,
            overwrite=overwrite,
        )

    if (
        _SAVE_INITIAL_SEGMENTATION
        and initial_segment is not None
    ):
        base._write_fits(
            initial_segmentation_path,
            np.asarray(initial_segment.data, dtype=np.int32),
            geometry.header,
            overwrite=overwrite,
        )

    if _SAVE_DETECTION_THRESHOLD:
        base._write_fits(
            threshold_path,
            threshold.astype(np.float32),
            geometry.header,
            overwrite=overwrite,
        )

    image_valid = geometry.valid_mask
    detection_valid = (
        image_valid
        & np.isfinite(shared_rms)
        & (shared_rms > 0)
    )

    if not bool(detection_valid.any()):
        raise ValueError(
            f"{label}: no valid pixels remain after applying shared RMS"
        )

    own_rms_median = float(
        np.nanmedian(own_background_rms[image_valid])
    )
    detection_rms_median = float(
        np.nanmedian(shared_rms[detection_valid])
    )

    summary: dict[str, Any] = {
        "label": label,
        "image": str(image_path),
        "coverage": (
            None
            if coverage_path is None
            else str(coverage_path)
        ),
        "shape": list(data.shape),
        "pixel_scale_arcsec": base._pixel_scale_arcsec(
            geometry.wcs
        ),
        "valid_fraction": float(image_valid.mean()),
        "source_count_full": int(len(table)),
        "background_median": float(
            np.nanmedian(own_background[image_valid])
        ),
        "background_rms_median": own_rms_median,
        "background_rms_role": (
            "diagnostic_only_not_used_for_detection"
        ),
        "detection_rms_source": str(_SHARED_RMS_PATH),
        "detection_rms_median": detection_rms_median,
        "threshold_sigma": float(threshold_sigma),
        "detection_threshold_median": float(
            threshold_sigma * detection_rms_median
        ),
        "catalog_full": str(catalog_path),
        "background_fits": str(background_path),
        "background_rms_fits": str(own_rms_path),
        "segmentation_fits": (
            str(segmentation_path)
            if segmentation_array is not None
            else None
        ),
        "initial_segmentation_fits": (
            str(initial_segmentation_path)
            if (
                _SAVE_INITIAL_SEGMENTATION
                and initial_segment is not None
            )
            else None
        ),
        "detection_threshold_fits": (
            str(threshold_path)
            if _SAVE_DETECTION_THRESHOLD
            else None
        ),
    }

    return base.DetectionResult(
        geometry=geometry,
        catalog_full=table,
        background=own_background,
        background_rms=shared_rms,
        segmentation=segmentation_array,
        summary=summary,
    )


def main() -> int:
    global _SHARED_RMS_PATH
    global _SAVE_INITIAL_SEGMENTATION
    global _SAVE_DETECTION_THRESHOLD

    wrapper_options, remaining = _parse_wrapper_options(
        sys.argv[1:]
    )

    _SHARED_RMS_PATH = (
        wrapper_options.threshold_rms_fits
        .expanduser()
        .resolve()
    )
    _SAVE_INITIAL_SEGMENTATION = bool(
        wrapper_options.save_initial_segmentation
    )
    _SAVE_DETECTION_THRESHOLD = bool(
        wrapper_options.save_detection_threshold
    )

    if not _SHARED_RMS_PATH.exists():
        raise FileNotFoundError(_SHARED_RMS_PATH)

    # Remove wrapper-only arguments before the original parser runs.
    sys.argv = [sys.argv[0], *remaining]

    # Preserve all original matching, common-footprint, catalog and summary
    # logic, replacing only the per-image detection implementation.
    base._build_catalog = _build_catalog_shared_rms

    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
