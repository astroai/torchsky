#!/usr/bin/env python3
"""Detect and cross-match source catalogs from INR and reference mosaics.

This script is designed for band-by-band comparison of bandpass-integrated INR
reconstructions against UPDC (or another reference mosaic).  Every image is
processed with the same Photutils detection configuration.  Source-count and
cross-match statistics are, by default, restricted to the common sky footprint
of all compared images so differences in mosaic canvas size do not masquerade
as scientific differences.

The default detection threshold is 6 times the local background RMS. Background
estimation uses a 60x60-pixel ``Background2D`` mesh, biweight location, MAD RMS,
and a 5x5 mesh filter.  ``npixels`` and deblending are explicitly recorded
because they materially affect source counts.

Cross-matching is done in celestial coordinates.  All candidate pairs inside the
requested angular radius are generated with Astropy and then reduced to a
one-to-one catalog match by accepting candidate pairs in increasing separation.
Unmatched detections are called ``source_only``/``reference_only`` rather than
false positives/negatives because the reference catalog is not ground truth.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.background import (
    Background2D,
    BiweightLocationBackground,
    MADStdBackgroundRMS,
)
from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources


@dataclass
class ImageGeometry:
    label: str
    path: Path
    shape: tuple[int, int]
    header: fits.Header
    wcs: WCS
    valid_mask: np.ndarray
    coverage_path: Path | None


@dataclass
class DetectionResult:
    geometry: ImageGeometry
    catalog_full: Table
    background: np.ndarray
    background_rms: np.ndarray
    segmentation: np.ndarray | None
    summary: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        metavar="LABEL=FITS",
        help="Input mosaic. Repeat for INR model(s) and the reference image.",
    )
    parser.add_argument(
        "--coverage",
        action="append",
        default=[],
        metavar="LABEL=FITS",
        help="Optional nonzero=valid coverage mask for an image label.",
    )
    parser.add_argument("--reference-label", default="UPDC")
    parser.add_argument("--band", default=None, help="Optional band name stored in outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--threshold-sigma", type=float, default=6.0)
    parser.add_argument("--npixels", type=int, default=5)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--background-box-size", type=int, default=60)
    parser.add_argument("--background-filter-size", type=int, default=5)
    parser.add_argument("--sigma-clip", type=float, default=3.0)
    parser.add_argument("--sigma-clip-maxiters", type=int, default=10)
    parser.add_argument(
        "--deblend",
        action="store_true",
        help="Apply Photutils deblend_sources after initial segmentation.",
    )
    parser.add_argument("--deblend-nlevels", type=int, default=32)
    parser.add_argument("--deblend-contrast", type=float, default=0.001)

    parser.add_argument("--match-radius-arcsec", type=float, default=0.10)
    parser.add_argument(
        "--match-radius-grid-arcsec",
        default="0.05,0.075,0.10,0.15",
        help="Comma-separated radii used only for match-sensitivity diagnostics.",
    )
    parser.add_argument(
        "--snr-bins",
        default="6,8,10,20,inf",
        help="Peak-S/N bin edges used for faint-end cross-match summaries.",
    )
    parser.add_argument(
        "--no-common-footprint",
        action="store_true",
        help="Do not restrict catalogs to the sky footprint shared by every input image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing catalog products already present in output-dir.",
    )
    return parser


def _parse_label_paths(values: list[str], *, kind: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for token in values:
        if "=" not in token:
            raise ValueError(f"{kind} must be LABEL=PATH, got {token!r}")
        label, raw = token.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty label in {token!r}")
        if label in output:
            raise ValueError(f"duplicate {kind} label {label!r}")
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        output[label] = path
    return output


def _parse_float_list(text: str, *, allow_inf: bool = False) -> list[float]:
    values: list[float] = []
    for token in str(text).split(","):
        token = token.strip().lower()
        if not token:
            continue
        value = float("inf") if allow_inf and token in {"inf", "+inf", "infinity"} else float(token)
        if not math.isfinite(value) and not (allow_inf and math.isinf(value) and value > 0):
            raise ValueError(f"invalid numeric list value {token!r}")
        values.append(value)
    if not values:
        raise ValueError("numeric list must not be empty")
    if any(right <= left for left, right in zip(values[:-1], values[1:], strict=True)):
        raise ValueError("numeric list values must be strictly increasing")
    return values


def _read_2d_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            array = np.asarray(hdu.data)
            if array.ndim == 2:
                return np.asarray(array, dtype=np.float64), hdu.header.copy()
    raise ValueError(f"no 2D image HDU found in {path}")


def _pixel_scale_arcsec(wcs: WCS) -> float:
    try:
        scales = np.asarray(proj_plane_pixel_scales(wcs.celestial), dtype=float) * 3600.0
        if scales.size >= 2 and np.isfinite(scales[:2]).all() and bool((scales[:2] > 0).all()):
            return float(np.sqrt(scales[0] * scales[1]))
    except Exception:
        pass
    return float("nan")


def _quantity_values(value: Any) -> np.ndarray:
    raw = getattr(value, "value", value)
    return np.asarray(raw)


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)


def _write_fits(
    path: Path,
    data: np.ndarray,
    header: fits.Header,
    *,
    overwrite: bool,
) -> None:
    fits.PrimaryHDU(np.asarray(data), header=header).writeto(path, overwrite=overwrite)


def _write_table(path: Path, table: Table, *, overwrite: bool) -> None:
    table.write(path, format="ascii.ecsv", overwrite=overwrite)


def _empty_catalog() -> Table:
    return Table(
        names=(
            "image_label",
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
            "U64",
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


def _geometry_for_image(
    *,
    label: str,
    image_path: Path,
    coverage_path: Path | None,
) -> tuple[np.ndarray, ImageGeometry]:
    data, header = _read_2d_fits(image_path)
    wcs = WCS(header).celestial
    if wcs.pixel_n_dim != 2 or wcs.world_n_dim != 2:
        raise ValueError(f"{label}: image does not have a 2D celestial WCS")

    valid = np.isfinite(data)
    if coverage_path is not None:
        coverage, _ = _read_2d_fits(coverage_path)
        if coverage.shape != data.shape:
            raise ValueError(
                f"coverage shape {coverage.shape} != image shape {data.shape} for {label}"
            )
        valid &= np.isfinite(coverage) & (coverage > 0)

    if not bool(valid.any()):
        raise ValueError(f"{label}: no valid pixels remain after masking")
    return data, ImageGeometry(
        label=label,
        path=image_path,
        shape=(int(data.shape[0]), int(data.shape[1])),
        header=header,
        wcs=wcs,
        valid_mask=valid,
        coverage_path=coverage_path,
    )


def _build_catalog(
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
) -> DetectionResult:
    data, geometry = _geometry_for_image(
        label=label,
        image_path=image_path,
        coverage_path=coverage_path,
    )
    mask = ~geometry.valid_mask

    sigma_clip = SigmaClip(
        sigma=float(sigma_clip_value),
        maxiters=int(sigma_clip_maxiters),
    )
    bkg = Background2D(
        data,
        box_size=(background_box_size, background_box_size),
        filter_size=(background_filter_size, background_filter_size),
        mask=mask,
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
        mask=mask,
    )
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

    if segment is None:
        table = _empty_catalog()
        segmentation_array = None
    else:
        catalog = SourceCatalog(
            data_sub,
            segment,
            mask=mask,
            background=background,
            wcs=geometry.wcs,
            progress_bar=False,
        )
        source_id = np.asarray(catalog.labels, dtype=np.int64)
        x = _quantity_values(catalog.xcentroid).astype(float)
        y = _quantity_values(catalog.ycentroid).astype(float)
        area = _quantity_values(catalog.area).astype(float)
        segment_flux = _quantity_values(catalog.segment_flux).astype(float)
        max_value = _quantity_values(catalog.max_value).astype(float)
        ra, dec = geometry.wcs.pixel_to_world_values(x, y)
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)

        xi = np.clip(np.rint(x).astype(int), 0, data.shape[1] - 1)
        yi = np.clip(np.rint(y).astype(int), 0, data.shape[0] - 1)
        local_background = background[yi, xi]
        local_rms = background_rms[yi, xi]
        with np.errstate(divide="ignore", invalid="ignore"):
            peak_sigma = max_value / local_rms

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
        table["local_background_rms"] = local_rms
        table["peak_sigma"] = peak_sigma
        segmentation_array = np.asarray(segment.data, dtype=np.int32)

    safe = _safe_label(label)
    catalog_path = output_dir / f"{safe}_sources_full.ecsv"
    bkg_path = output_dir / f"{safe}_background.fits"
    rms_path = output_dir / f"{safe}_background_rms.fits"
    seg_path = output_dir / f"{safe}_segmentation.fits"

    _write_table(catalog_path, table, overwrite=overwrite)
    _write_fits(bkg_path, background.astype(np.float32), geometry.header, overwrite=overwrite)
    _write_fits(rms_path, background_rms.astype(np.float32), geometry.header, overwrite=overwrite)
    if segmentation_array is not None:
        _write_fits(seg_path, segmentation_array, geometry.header, overwrite=overwrite)

    valid = geometry.valid_mask
    summary = {
        "label": label,
        "image": str(image_path),
        "coverage": None if coverage_path is None else str(coverage_path),
        "shape": list(data.shape),
        "pixel_scale_arcsec": _pixel_scale_arcsec(geometry.wcs),
        "valid_fraction": float(valid.mean()),
        "source_count_full": int(len(table)),
        "background_median": float(np.nanmedian(background[valid])),
        "background_rms_median": float(np.nanmedian(background_rms[valid])),
        "catalog_full": str(catalog_path),
        "background_fits": str(bkg_path),
        "background_rms_fits": str(rms_path),
        "segmentation_fits": str(seg_path) if segmentation_array is not None else None,
    }
    return DetectionResult(
        geometry=geometry,
        catalog_full=table,
        background=background,
        background_rms=background_rms,
        segmentation=segmentation_array,
        summary=summary,
    )


def _positions_valid_in_geometry(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    geometry: ImageGeometry,
) -> np.ndarray:
    x, y = geometry.wcs.world_to_pixel_values(ra_deg, dec_deg)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    inside = (
        finite
        & (x >= 0.0)
        & (x <= geometry.shape[1] - 1)
        & (y >= 0.0)
        & (y <= geometry.shape[0] - 1)
    )
    result = np.zeros(x.shape, dtype=bool)
    if not bool(inside.any()):
        return result

    indices = np.flatnonzero(inside)
    xi = np.clip(np.rint(x[indices]).astype(int), 0, geometry.shape[1] - 1)
    yi = np.clip(np.rint(y[indices]).astype(int), 0, geometry.shape[0] - 1)
    result[indices] = geometry.valid_mask[yi, xi]
    return result


def _restrict_to_common_footprint(
    table: Table,
    geometries: Iterable[ImageGeometry],
) -> Table:
    if len(table) == 0:
        return table.copy(copy_data=True)
    ra = np.asarray(table["ra_deg"], dtype=float)
    dec = np.asarray(table["dec_deg"], dtype=float)
    keep = np.isfinite(ra) & np.isfinite(dec)
    for geometry in geometries:
        keep &= _positions_valid_in_geometry(ra, dec, geometry)
    return table[keep]


def _candidate_pairs(
    source: Table,
    reference: Table,
    *,
    radius_arcsec: float,
) -> list[tuple[int, int, float]]:
    if len(source) == 0 or len(reference) == 0:
        return []
    src = SkyCoord(source["ra_deg"], source["dec_deg"], unit="deg")
    ref = SkyCoord(reference["ra_deg"], reference["dec_deg"], unit="deg")
    # Astropy returns indices into the search-around coordinates first (src),
    # then into the catalog object (ref).
    src_index, ref_index, sep2d, _ = ref.search_around_sky(
        src,
        float(radius_arcsec) * u.arcsec,
    )
    return [
        (int(si), int(ri), float(sep.arcsec))
        for si, ri, sep in zip(src_index, ref_index, sep2d, strict=True)
    ]


def _one_to_one_sky_matches(
    source: Table,
    reference: Table,
    *,
    radius_arcsec: float,
) -> list[tuple[int, int, float]]:
    candidates = sorted(
        _candidate_pairs(source, reference, radius_arcsec=radius_arcsec),
        key=lambda row: (row[2], row[0], row[1]),
    )
    used_source: set[int] = set()
    used_reference: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for source_index, reference_index, separation in candidates:
        if source_index in used_source or reference_index in used_reference:
            continue
        used_source.add(source_index)
        used_reference.add(reference_index)
        matches.append((source_index, reference_index, separation))
    return matches


def _quantile_or_nan(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return float("nan")
    return numerator / denominator


def _pair_table(
    source_label: str,
    reference_label: str,
    source: Table,
    reference: Table,
    matches: list[tuple[int, int, float]],
) -> Table:
    output = Table(
        names=(
            "source_label",
            "reference_label",
            "source_id",
            "reference_source_id",
            "separation_arcsec",
            "source_ra_deg",
            "source_dec_deg",
            "reference_ra_deg",
            "reference_dec_deg",
            "source_segment_flux",
            "reference_segment_flux",
            "segment_flux_ratio_source_over_reference",
            "source_peak_sigma",
            "reference_peak_sigma",
            "source_area_pix2",
            "reference_area_pix2",
        ),
        dtype=(
            "U64", "U64", int, int, float, float, float, float, float,
            float, float, float, float, float, float, float,
        ),
    )
    for source_index, reference_index, separation in matches:
        s = source[source_index]
        r = reference[reference_index]
        sf = float(s["segment_flux"])
        rf = float(r["segment_flux"])
        output.add_row(
            (
                source_label,
                reference_label,
                int(s["source_id"]),
                int(r["source_id"]),
                separation,
                float(s["ra_deg"]),
                float(s["dec_deg"]),
                float(r["ra_deg"]),
                float(r["dec_deg"]),
                sf,
                rf,
                _safe_ratio(sf, rf),
                float(s["peak_sigma"]),
                float(r["peak_sigma"]),
                float(s["area_pix2"]),
                float(r["area_pix2"]),
            )
        )
    return output


def _snr_bin_summary(
    *,
    source: Table,
    reference: Table,
    matches: list[tuple[int, int, float]],
    edges: list[float],
) -> dict[str, list[dict[str, Any]]]:
    source_matched = {row[0] for row in matches}
    reference_matched = {row[1] for row in matches}

    def summarize(table: Table, matched: set[int]) -> list[dict[str, Any]]:
        snr = np.asarray(table["peak_sigma"], dtype=float) if len(table) else np.array([])
        rows: list[dict[str, Any]] = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            choose = np.isfinite(snr) & (snr >= lo) & (snr < hi)
            indices = set(np.flatnonzero(choose).tolist())
            total = len(indices)
            count_matched = len(indices & matched)
            rows.append(
                {
                    "snr_min_inclusive": lo,
                    "snr_max_exclusive": hi,
                    "count": total,
                    "matched": count_matched,
                    "matched_fraction": float(count_matched / total) if total else float("nan"),
                }
            )
        return rows

    return {
        "source_peak_sigma_bins": summarize(source, source_matched),
        "reference_peak_sigma_bins": summarize(reference, reference_matched),
    }


def _match_summary(
    source: Table,
    reference: Table,
    matches: list[tuple[int, int, float]],
    *,
    radius_arcsec: float,
    snr_edges: list[float],
) -> dict[str, Any]:
    source_matched = {row[0] for row in matches}
    reference_matched = {row[1] for row in matches}
    separations = np.asarray([row[2] for row in matches], dtype=float)
    flux_ratios = np.asarray(
        [
            _safe_ratio(
                float(source[si]["segment_flux"]),
                float(reference[ri]["segment_flux"]),
            )
            for si, ri, _ in matches
        ],
        dtype=float,
    )
    return {
        "match_radius_arcsec": float(radius_arcsec),
        "source_count": int(len(source)),
        "reference_count": int(len(reference)),
        "matched_pairs": int(len(matches)),
        "source_matched_fraction": (
            float(len(source_matched) / len(source)) if len(source) else float("nan")
        ),
        "reference_recovered_fraction": (
            float(len(reference_matched) / len(reference)) if len(reference) else float("nan")
        ),
        "source_only": int(len(source) - len(source_matched)),
        "reference_only": int(len(reference) - len(reference_matched)),
        "separation_arcsec": {
            "median": _quantile_or_nan(separations, 0.50),
            "q68": _quantile_or_nan(separations, 0.68),
            "q95": _quantile_or_nan(separations, 0.95),
            "max": float(np.nanmax(separations)) if separations.size else float("nan"),
        },
        "segment_flux_ratio_source_over_reference": {
            "median": _quantile_or_nan(flux_ratios, 0.50),
            "q16": _quantile_or_nan(flux_ratios, 0.16),
            "q84": _quantile_or_nan(flux_ratios, 0.84),
        },
        "snr_bins": _snr_bin_summary(
            source=source,
            reference=reference,
            matches=matches,
            edges=snr_edges,
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    if not math.isfinite(args.threshold_sigma) or args.threshold_sigma <= 0:
        raise SystemExit("--threshold-sigma must be finite and positive")
    if args.npixels <= 0:
        raise SystemExit("--npixels must be positive")
    if not math.isfinite(args.match_radius_arcsec) or args.match_radius_arcsec <= 0:
        raise SystemExit("--match-radius-arcsec must be finite and positive")

    radius_grid = _parse_float_list(args.match_radius_grid_arcsec)
    if any(radius <= 0 for radius in radius_grid):
        raise SystemExit("all match-radius-grid values must be positive")
    snr_edges = _parse_float_list(args.snr_bins, allow_inf=True)
    if len(snr_edges) < 2:
        raise SystemExit("--snr-bins requires at least two edges")

    images = _parse_label_paths(args.image, kind="image")
    coverages = _parse_label_paths(args.coverage, kind="coverage")
    unknown_coverage = sorted(set(coverages).difference(images))
    if unknown_coverage:
        raise SystemExit(
            "coverage labels without matching images: " + ", ".join(unknown_coverage)
        )
    if args.reference_label not in images:
        raise SystemExit(
            f"reference label {args.reference_label!r} is not among image labels {list(images)}"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, DetectionResult] = {}
    for label, image_path in images.items():
        print(f"detecting {label}: {image_path}", flush=True)
        result = _build_catalog(
            label=label,
            image_path=image_path,
            coverage_path=coverages.get(label),
            output_dir=output_dir,
            threshold_sigma=args.threshold_sigma,
            npixels=args.npixels,
            connectivity=args.connectivity,
            background_box_size=args.background_box_size,
            background_filter_size=args.background_filter_size,
            sigma_clip_value=args.sigma_clip,
            sigma_clip_maxiters=args.sigma_clip_maxiters,
            deblend=args.deblend,
            deblend_nlevels=args.deblend_nlevels,
            deblend_contrast=args.deblend_contrast,
            overwrite=args.overwrite,
        )
        results[label] = result
        print(f"{label}: {len(result.catalog_full)} full-footprint sources", flush=True)

    geometries = [result.geometry for result in results.values()]
    catalogs: dict[str, Table] = {}
    image_summaries: dict[str, Any] = {}
    for label, result in results.items():
        if args.no_common_footprint:
            common = result.catalog_full.copy(copy_data=True)
        else:
            common = _restrict_to_common_footprint(result.catalog_full, geometries)
        common_path = output_dir / f"{_safe_label(label)}_sources_common.ecsv"
        _write_table(common_path, common, overwrite=args.overwrite)
        catalogs[label] = common
        summary = dict(result.summary)
        summary["source_count_common"] = int(len(common))
        summary["catalog_common"] = str(common_path)
        summary["common_fraction_of_detected"] = (
            float(len(common) / len(result.catalog_full))
            if len(result.catalog_full)
            else float("nan")
        )
        image_summaries[label] = summary
        print(f"{label}: {len(common)} common-footprint sources", flush=True)

    reference = catalogs[args.reference_label]
    match_summaries: dict[str, Any] = {}
    all_pair_tables: list[Table] = []

    for label, source in catalogs.items():
        if label == args.reference_label:
            continue

        matches = _one_to_one_sky_matches(
            source,
            reference,
            radius_arcsec=args.match_radius_arcsec,
        )
        source_matched = sorted({row[0] for row in matches})
        reference_matched = sorted({row[1] for row in matches})
        source_only_index = sorted(set(range(len(source))).difference(source_matched))
        reference_only_index = sorted(set(range(len(reference))).difference(reference_matched))

        pair_table = _pair_table(
            label,
            args.reference_label,
            source,
            reference,
            matches,
        )
        safe = _safe_label(label)
        safe_ref = _safe_label(args.reference_label)
        pair_path = output_dir / f"{safe}_xmatch_{safe_ref}.ecsv"
        matched_source_path = output_dir / f"{safe}_matched_to_{safe_ref}.ecsv"
        source_only_path = output_dir / f"{safe}_only_vs_{safe_ref}.ecsv"
        reference_only_path = output_dir / f"{safe_ref}_only_vs_{safe}.ecsv"

        _write_table(pair_path, pair_table, overwrite=args.overwrite)
        _write_table(matched_source_path, source[source_matched], overwrite=args.overwrite)
        _write_table(source_only_path, source[source_only_index], overwrite=args.overwrite)
        _write_table(reference_only_path, reference[reference_only_index], overwrite=args.overwrite)
        all_pair_tables.append(pair_table)

        sensitivity: list[dict[str, Any]] = []
        for radius in radius_grid:
            radius_matches = _one_to_one_sky_matches(
                source,
                reference,
                radius_arcsec=radius,
            )
            source_ids = {row[0] for row in radius_matches}
            reference_ids = {row[1] for row in radius_matches}
            sensitivity.append(
                {
                    "radius_arcsec": radius,
                    "matched_pairs": len(radius_matches),
                    "source_matched_fraction": (
                        float(len(source_ids) / len(source)) if len(source) else float("nan")
                    ),
                    "reference_recovered_fraction": (
                        float(len(reference_ids) / len(reference))
                        if len(reference)
                        else float("nan")
                    ),
                }
            )

        summary = _match_summary(
            source,
            reference,
            matches,
            radius_arcsec=args.match_radius_arcsec,
            snr_edges=snr_edges,
        )
        summary.update(
            {
                "reference_label": args.reference_label,
                "pair_catalog": str(pair_path),
                "matched_source_catalog": str(matched_source_path),
                "source_only_catalog": str(source_only_path),
                "reference_only_catalog": str(reference_only_path),
                "match_radius_sensitivity": sensitivity,
            }
        )
        match_summaries[label] = summary

    combined_pair_path = output_dir / "all_cross_matches.ecsv"
    if all_pair_tables:
        combined = vstack(all_pair_tables, metadata_conflicts="silent")
    else:
        combined = Table()
    _write_table(combined_pair_path, combined, overwrite=args.overwrite)

    configuration = {
        "band": None if args.band is None else str(args.band).upper(),
        "threshold_sigma": float(args.threshold_sigma),
        "npixels": int(args.npixels),
        "connectivity": int(args.connectivity),
        "background": {
            "class": "Photutils Background2D",
            "box_size": int(args.background_box_size),
            "filter_size": int(args.background_filter_size),
            "estimator": "BiweightLocationBackground",
            "rms_estimator": "MADStdBackgroundRMS",
            "sigma_clip": float(args.sigma_clip),
            "sigma_clip_maxiters": int(args.sigma_clip_maxiters),
        },
        "deblend": bool(args.deblend),
        "deblend_nlevels": int(args.deblend_nlevels),
        "deblend_contrast": float(args.deblend_contrast),
        "reference_label": args.reference_label,
        "match_policy": "all pairs within radius, then greedy closest one-to-one",
        "match_radius_arcsec": float(args.match_radius_arcsec),
        "match_radius_grid_arcsec": radius_grid,
        "snr_bins": snr_edges,
        "common_sky_footprint": not bool(args.no_common_footprint),
        "important_note": (
            "segment_flux is a Photutils segmentation measurement used here for "
            "diagnostics; it is not claimed to reproduce SExtractor MAG_AUTO. "
            "source_only/reference_only means unmatched to the comparison catalog, "
            "not false-positive/false-negative relative to ground truth."
        ),
    }
    summary = {
        "configuration": configuration,
        "images": image_summaries,
        "matches_to_reference": match_summaries,
        "all_cross_matches": str(combined_pair_path),
    }
    summary_path = output_dir / "catalog_comparison_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
