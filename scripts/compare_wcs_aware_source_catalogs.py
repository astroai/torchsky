#!/usr/bin/env python3
"""Extract and compare source catalogs from INR/UPDC mosaic products.

The same Photutils segmentation pipeline is applied to every input image.
The default detection threshold is 6 times the local background RMS, matching
UPdec-Webb's faint-source detection test.  Background estimation uses a
60x60-pixel Background2D grid with biweight location, a MAD RMS estimator, and
a 5x5 mesh filter.  Parameters are exposed on the CLI and recorded in the
summary for reproducibility.

Example
-------
python scripts/compare_wcs_aware_source_catalogs.py \
  --image A0=/path/F090W_fourier_union_scene.fits \
  --image A1=/path/F090W_A1_nerf_literature_ablation_bandpass_scene.fits \
  --image A2=/path/F090W_A2_siren_literature_ablation_bandpass_scene.fits \
  --image UPDC=/path/F090W_updc_iter0.fits \
  --coverage A0=/path/union_coverage_mask.fits \
  --coverage A1=/path/A1_nerf_literature_ablation_union_coverage_mask.fits \
  --coverage A2=/path/A2_siren_literature_ablation_union_coverage_mask.fits \
  --reference-label UPDC \
  --output-dir /path/catalog_comparison
"""

from __future__ import annotations

import argparse
import csv
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        metavar="LABEL=FITS",
        help="Input mosaic. Repeat for A0/A1/A2/UPDC.",
    )
    parser.add_argument(
        "--coverage",
        action="append",
        default=[],
        metavar="LABEL=FITS",
        help="Optional nonzero=valid coverage mask for an image label.",
    )
    parser.add_argument("--reference-label", default="UPDC")
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
        help="Apply Photutils deblend_sources after the initial detection.",
    )
    parser.add_argument("--deblend-nlevels", type=int, default=32)
    parser.add_argument("--deblend-contrast", type=float, default=0.001)
    parser.add_argument("--match-radius-arcsec", type=float, default=0.10)
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


def _read_2d_fits(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            array = np.asarray(hdu.data)
            if array.ndim == 2:
                return np.asarray(array, dtype=np.float64), hdu.header.copy()
    raise ValueError(f"no 2D image HDU found in {path}")


def _pixel_scale_arcsec(header: fits.Header) -> float:
    try:
        wcs = WCS(header).celestial
        scales = proj_plane_pixel_scales(wcs) * 3600.0
        finite = np.asarray(scales, dtype=float)
        finite = finite[np.isfinite(finite) & (finite > 0)]
        if finite.size:
            return float(np.sqrt(np.prod(finite[:2])))
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
) -> tuple[Table, dict[str, Any]]:
    data, header = _read_2d_fits(image_path)
    finite = np.isfinite(data)
    mask = ~finite

    if coverage_path is not None:
        coverage, _ = _read_2d_fits(coverage_path)
        if coverage.shape != data.shape:
            raise ValueError(
                f"coverage shape {coverage.shape} != image shape {data.shape} for {label}"
            )
        mask |= ~(np.isfinite(coverage) & (coverage > 0))

    valid_fraction = float((~mask).mean())
    if valid_fraction <= 0:
        raise ValueError(f"{label}: no valid pixels remain after masking")

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
    if segment is None:
        table = Table(
            names=(
                "label",
                "source_id",
                "xcentroid",
                "ycentroid",
                "ra_deg",
                "dec_deg",
                "area_pix2",
                "segment_flux",
                "max_value",
                "local_background_rms",
                "peak_sigma",
            ),
            dtype=("U64", int, float, float, float, float, float, float, float, float, float),
        )
    else:
        if deblend:
            segment = deblend_sources(
                data_sub,
                segment,
                npixels=int(npixels),
                nlevels=int(deblend_nlevels),
                contrast=float(deblend_contrast),
                connectivity=int(connectivity),
                progress_bar=False,
            )

        wcs = WCS(header).celestial
        catalog = SourceCatalog(
            data_sub,
            segment,
            mask=mask,
            background=background,
            wcs=wcs,
            progress_bar=False,
        )
        source_id = np.asarray(catalog.labels, dtype=np.int64)
        x = _quantity_values(catalog.xcentroid).astype(float)
        y = _quantity_values(catalog.ycentroid).astype(float)
        area = _quantity_values(catalog.area).astype(float)
        segment_flux = _quantity_values(catalog.segment_flux).astype(float)
        max_value = _quantity_values(catalog.max_value).astype(float)
        ra, dec = wcs.pixel_to_world_values(x, y)
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)

        xi = np.clip(np.rint(x).astype(int), 0, data.shape[1] - 1)
        yi = np.clip(np.rint(y).astype(int), 0, data.shape[0] - 1)
        local_rms = background_rms[yi, xi]
        with np.errstate(divide="ignore", invalid="ignore"):
            peak_sigma = max_value / local_rms

        table = Table()
        table["label"] = np.repeat(label, len(source_id))
        table["source_id"] = source_id
        table["xcentroid"] = x
        table["ycentroid"] = y
        table["ra_deg"] = ra
        table["dec_deg"] = dec
        table["area_pix2"] = area
        table["segment_flux"] = segment_flux
        table["max_value"] = max_value
        table["local_background_rms"] = local_rms
        table["peak_sigma"] = peak_sigma

    safe = _safe_label(label)
    catalog_path = output_dir / f"{safe}_sources_6sigma.ecsv"
    bkg_path = output_dir / f"{safe}_background.fits"
    rms_path = output_dir / f"{safe}_background_rms.fits"
    seg_path = output_dir / f"{safe}_segmentation.fits"

    table.write(catalog_path, format="ascii.ecsv", overwrite=overwrite)
    _write_fits(bkg_path, background.astype(np.float32), header, overwrite=overwrite)
    _write_fits(rms_path, background_rms.astype(np.float32), header, overwrite=overwrite)
    if segment is not None:
        _write_fits(seg_path, segment.data.astype(np.int32), header, overwrite=overwrite)

    summary = {
        "label": label,
        "image": str(image_path),
        "coverage": None if coverage_path is None else str(coverage_path),
        "shape": list(data.shape),
        "pixel_scale_arcsec": _pixel_scale_arcsec(header),
        "valid_fraction": valid_fraction,
        "source_count": int(len(table)),
        "background_median": float(np.nanmedian(background[~mask])),
        "background_rms_median": float(np.nanmedian(background_rms[~mask])),
        "catalog": str(catalog_path),
        "background_fits": str(bkg_path),
        "background_rms_fits": str(rms_path),
        "segmentation_fits": str(seg_path) if segment is not None else None,
    }
    return table, summary


def _greedy_sky_matches(
    source: Table,
    reference: Table,
    *,
    radius_arcsec: float,
) -> list[tuple[int, int, float]]:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if len(source) == 0 or len(reference) == 0:
        return []

    src = SkyCoord(source["ra_deg"], source["dec_deg"], unit="deg")
    ref = SkyCoord(reference["ra_deg"], reference["dec_deg"], unit="deg")
    src_index, ref_index, sep2d, _ = ref.search_around_sky(
        src,
        float(radius_arcsec) * u.arcsec,
    )

    candidates = sorted(
        (
            (int(si), int(ri), float(sep.arcsec))
            for si, ri, sep in zip(src_index, ref_index, sep2d, strict=True)
        ),
        key=lambda row: row[2],
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


def main() -> int:
    args = build_parser().parse_args()
    if not math.isfinite(args.threshold_sigma) or args.threshold_sigma <= 0:
        raise SystemExit("--threshold-sigma must be finite and positive")
    if args.npixels <= 0:
        raise SystemExit("--npixels must be positive")
    if not math.isfinite(args.match_radius_arcsec) or args.match_radius_arcsec <= 0:
        raise SystemExit("--match-radius-arcsec must be finite and positive")

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

    tables: dict[str, Table] = {}
    image_summaries: dict[str, Any] = {}
    for label, image_path in images.items():
        print(f"detecting {label}: {image_path}", flush=True)
        table, summary = _build_catalog(
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
        tables[label] = table
        image_summaries[label] = summary
        print(f"{label}: {len(table)} sources", flush=True)

    reference = tables[args.reference_label]
    match_summaries: dict[str, Any] = {}
    match_csv = output_dir / "catalog_matches_to_reference.csv"
    if match_csv.exists() and not args.overwrite:
        raise FileExistsError(match_csv)

    with match_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
                "source_peak_sigma",
                "reference_peak_sigma",
            ],
        )
        writer.writeheader()

        for label, table in tables.items():
            if label == args.reference_label:
                continue
            matches = _greedy_sky_matches(
                table,
                reference,
                radius_arcsec=args.match_radius_arcsec,
            )
            for source_index, reference_index, separation in matches:
                source_row = table[source_index]
                reference_row = reference[reference_index]
                writer.writerow(
                    {
                        "source_label": label,
                        "reference_label": args.reference_label,
                        "source_id": int(source_row["source_id"]),
                        "reference_source_id": int(reference_row["source_id"]),
                        "separation_arcsec": separation,
                        "source_ra_deg": float(source_row["ra_deg"]),
                        "source_dec_deg": float(source_row["dec_deg"]),
                        "reference_ra_deg": float(reference_row["ra_deg"]),
                        "reference_dec_deg": float(reference_row["dec_deg"]),
                        "source_segment_flux": float(source_row["segment_flux"]),
                        "reference_segment_flux": float(reference_row["segment_flux"]),
                        "source_peak_sigma": float(source_row["peak_sigma"]),
                        "reference_peak_sigma": float(reference_row["peak_sigma"]),
                    }
                )

            source_matched = {row[0] for row in matches}
            reference_matched = {row[1] for row in matches}
            separations = np.asarray([row[2] for row in matches], dtype=float)
            match_summaries[label] = {
                "reference_label": args.reference_label,
                "match_radius_arcsec": float(args.match_radius_arcsec),
                "source_count": int(len(table)),
                "reference_count": int(len(reference)),
                "matched_pairs": int(len(matches)),
                "source_matched_fraction": (
                    float(len(source_matched) / len(table)) if len(table) else float("nan")
                ),
                "reference_recovered_fraction": (
                    float(len(reference_matched) / len(reference))
                    if len(reference)
                    else float("nan")
                ),
                "source_unmatched": int(len(table) - len(source_matched)),
                "reference_unmatched": int(len(reference) - len(reference_matched)),
                "separation_arcsec_median": (
                    float(np.median(separations)) if separations.size else float("nan")
                ),
                "separation_arcsec_q95": (
                    float(np.quantile(separations, 0.95)) if separations.size else float("nan")
                ),
            }

    configuration = {
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
        "match_radius_arcsec": float(args.match_radius_arcsec),
        "note": (
            "UPdec-Webb states that the faint-source detect_sources threshold was "
            "fixed at 6 times background sigma. npixels/deblend details are not "
            "fully specified there, so this script records them explicitly."
        ),
    }
    summary = {
        "configuration": configuration,
        "images": image_summaries,
        "matches_to_reference": match_summaries,
        "matches_csv": str(match_csv),
    }
    summary_path = output_dir / "catalog_comparison_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
