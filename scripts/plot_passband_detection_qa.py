#!/usr/bin/env python3
"""Cross-match two passband detection catalogs and overlay them on a reference FITS.

Passband A detections are drawn in red and passband B detections in blue.
The script writes a QA image plus matched/A-only/B-only ECSV catalogs, DS9
region files, and a compact JSON summary.  It consumes the ECSV products from
``compare_wcs_aware_source_catalogs.py``; it does not rerun source detection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-a", type=Path, required=True)
    parser.add_argument("--catalog-b", type=Path, required=True)
    parser.add_argument("--label-a", default="passband_A")
    parser.add_argument("--label-b", default="passband_B")
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-radius-arcsec", type=float, default=0.10)
    parser.add_argument("--ra-column", default="ra_deg")
    parser.add_argument("--dec-column", default="dec_deg")
    parser.add_argument("--snr-column", default="peak_sigma")
    parser.add_argument("--min-snr-a", type=float, default=None)
    parser.add_argument("--min-snr-b", type=float, default=None)
    parser.add_argument("--marker-size", type=float, default=28.0)
    parser.add_argument("--line-matches", action="store_true")
    parser.add_argument(
        "--plot-mode",
        choices=("all", "matched", "unmatched"),
        default="all",
        help="Which detections to draw. Cross-match outputs are always complete.",
    )
    parser.add_argument(
        "--display-percent",
        type=float,
        default=99.5,
        help="Central percentile interval used for the grayscale stretch.",
    )
    parser.add_argument(
        "--max-display-pixels",
        type=int,
        default=3000,
        help="Downsample only the displayed reference image to this maximum side.",
    )
    parser.add_argument("--crop-center-ra-deg", type=float, default=None)
    parser.add_argument("--crop-center-dec-deg", type=float, default=None)
    parser.add_argument("--crop-width-arcsec", type=float, default=None)
    parser.add_argument("--crop-height-arcsec", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)


def _read_first_2d(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data)
            if data.ndim == 2:
                return np.asarray(data, dtype=np.float64), hdu.header.copy()
    raise ValueError(f"no 2D image HDU found in {path}")


def _load_catalog(
    path: Path,
    *,
    ra_column: str,
    dec_column: str,
    snr_column: str,
    min_snr: float | None,
) -> Table:
    table = Table.read(path)
    missing = [name for name in (ra_column, dec_column) if name not in table.colnames]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    ra = np.asarray(table[ra_column], dtype=float)
    dec = np.asarray(table[dec_column], dtype=float)
    keep = np.isfinite(ra) & np.isfinite(dec)

    if min_snr is not None:
        if snr_column not in table.colnames:
            raise ValueError(
                f"{path} has no {snr_column!r} column required by --min-snr"
            )
        snr = np.asarray(table[snr_column], dtype=float)
        keep &= np.isfinite(snr) & (snr >= float(min_snr))

    output = table[keep].copy(copy_data=True)
    output["qa_row_index"] = np.flatnonzero(keep)
    return output


def _candidate_pairs(
    catalog_a: Table,
    catalog_b: Table,
    *,
    ra_column: str,
    dec_column: str,
    radius_arcsec: float,
) -> list[tuple[int, int, float]]:
    if len(catalog_a) == 0 or len(catalog_b) == 0:
        return []

    coord_a = SkyCoord(
        np.asarray(catalog_a[ra_column], dtype=float),
        np.asarray(catalog_a[dec_column], dtype=float),
        unit="deg",
    )
    coord_b = SkyCoord(
        np.asarray(catalog_b[ra_column], dtype=float),
        np.asarray(catalog_b[dec_column], dtype=float),
        unit="deg",
    )

    # Astropy returns indices into the search-around coordinates first (A),
    # then into the catalog object (B), matching the existing catalog script.
    index_a, index_b, separation, _ = coord_b.search_around_sky(
        coord_a,
        float(radius_arcsec) * u.arcsec,
    )
    return [
        (int(ia), int(ib), float(sep.arcsec))
        for ia, ib, sep in zip(index_a, index_b, separation, strict=True)
    ]


def _one_to_one_matches(
    catalog_a: Table,
    catalog_b: Table,
    *,
    ra_column: str,
    dec_column: str,
    radius_arcsec: float,
) -> list[tuple[int, int, float]]:
    candidates = sorted(
        _candidate_pairs(
            catalog_a,
            catalog_b,
            ra_column=ra_column,
            dec_column=dec_column,
            radius_arcsec=radius_arcsec,
        ),
        key=lambda row: (row[2], row[0], row[1]),
    )

    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for index_a, index_b, separation in candidates:
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        matches.append((index_a, index_b, separation))

    return matches


def _pair_table(
    label_a: str,
    label_b: str,
    catalog_a: Table,
    catalog_b: Table,
    matches: list[tuple[int, int, float]],
    *,
    ra_column: str,
    dec_column: str,
) -> Table:
    output = Table(
        names=(
            "label_a",
            "label_b",
            "index_a",
            "index_b",
            "separation_arcsec",
            "ra_a_deg",
            "dec_a_deg",
            "ra_b_deg",
            "dec_b_deg",
        ),
        dtype=("U64", "U64", int, int, float, float, float, float, float),
    )

    for index_a, index_b, separation in matches:
        output.add_row(
            (
                label_a,
                label_b,
                int(index_a),
                int(index_b),
                float(separation),
                float(catalog_a[index_a][ra_column]),
                float(catalog_a[index_a][dec_column]),
                float(catalog_b[index_b][ra_column]),
                float(catalog_b[index_b][dec_column]),
            )
        )

    return output


def _write_table(path: Path, table: Table, *, overwrite: bool) -> None:
    table.write(path, format="ascii.ecsv", overwrite=overwrite)


def _write_ds9_regions(
    path: Path,
    table: Table,
    *,
    ra_column: str,
    dec_column: str,
    color: str,
    radius_arcsec: float,
    label: str,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)

    lines = [
        "# Region file format: DS9 version 4.1",
        (
            "global color="
            f"{color} dashlist=8 3 width=1 font='helvetica 10 normal roman' "
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 "
            "include=1 source=1"
        ),
        "fk5",
    ]

    for index, row in enumerate(table):
        ra = float(row[ra_column])
        dec = float(row[dec_column])
        lines.append(
            f'circle({ra:.10f},{dec:.10f},{radius_arcsec:.4f}") '
            f'# color={color} text={{{label}:{index}}}'
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _display_image(
    data: np.ndarray,
    wcs: WCS,
    *,
    max_display_pixels: int,
) -> tuple[np.ndarray, WCS, int]:
    longest = max(data.shape)
    stride = max(1, math.ceil(longest / max(1, int(max_display_pixels))))
    if stride == 1:
        return data, wcs, stride

    display = data[::stride, ::stride]
    display_wcs = wcs.slice(
        (slice(None, None, stride), slice(None, None, stride))
    )
    return display, display_wcs, stride


def _marker_indices(
    mode: str,
    size: int,
    matched: set[int],
) -> np.ndarray:
    if mode == "matched":
        return np.asarray(sorted(matched), dtype=int)
    if mode == "unmatched":
        return np.asarray(sorted(set(range(size)).difference(matched)), dtype=int)
    return np.arange(size, dtype=int)


def _apply_crop(
    ax: Any,
    display_wcs: WCS,
    *,
    center_ra_deg: float | None,
    center_dec_deg: float | None,
    width_arcsec: float | None,
    height_arcsec: float | None,
) -> None:
    values = (center_ra_deg, center_dec_deg, width_arcsec)
    if all(value is None for value in values) and height_arcsec is None:
        return
    if any(value is None for value in values):
        raise ValueError(
            "crop requires --crop-center-ra-deg, --crop-center-dec-deg, "
            "and --crop-width-arcsec"
        )

    height = float(height_arcsec if height_arcsec is not None else width_arcsec)
    center = SkyCoord(float(center_ra_deg), float(center_dec_deg), unit="deg")
    x_center, y_center = display_wcs.world_to_pixel(center)

    scales = np.abs(np.linalg.det(display_wcs.pixel_scale_matrix)) ** 0.5 * 3600.0
    if not np.isfinite(scales) or scales <= 0:
        raise ValueError("could not determine reference-image pixel scale")

    half_width_pixels = 0.5 * float(width_arcsec) / scales
    half_height_pixels = 0.5 * height / scales
    ax.set_xlim(x_center - half_width_pixels, x_center + half_width_pixels)
    ax.set_ylim(y_center - half_height_pixels, y_center + half_height_pixels)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> int:
    args = build_parser().parse_args()

    if not math.isfinite(args.match_radius_arcsec) or args.match_radius_arcsec <= 0:
        raise SystemExit("--match-radius-arcsec must be finite and positive")
    if not 0 < args.display_percent <= 100:
        raise SystemExit("--display-percent must be in (0, 100]")

    catalog_a_path = args.catalog_a.expanduser().resolve()
    catalog_b_path = args.catalog_b.expanduser().resolve()
    reference_path = args.reference_image.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for path in (catalog_a_path, catalog_b_path, reference_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_a = _safe_label(args.label_a)
    safe_b = _safe_label(args.label_b)

    catalog_a = _load_catalog(
        catalog_a_path,
        ra_column=args.ra_column,
        dec_column=args.dec_column,
        snr_column=args.snr_column,
        min_snr=args.min_snr_a,
    )
    catalog_b = _load_catalog(
        catalog_b_path,
        ra_column=args.ra_column,
        dec_column=args.dec_column,
        snr_column=args.snr_column,
        min_snr=args.min_snr_b,
    )

    matches = _one_to_one_matches(
        catalog_a,
        catalog_b,
        ra_column=args.ra_column,
        dec_column=args.dec_column,
        radius_arcsec=args.match_radius_arcsec,
    )
    matched_a = {row[0] for row in matches}
    matched_b = {row[1] for row in matches}
    only_a = sorted(set(range(len(catalog_a))).difference(matched_a))
    only_b = sorted(set(range(len(catalog_b))).difference(matched_b))

    pair_table = _pair_table(
        args.label_a,
        args.label_b,
        catalog_a,
        catalog_b,
        matches,
        ra_column=args.ra_column,
        dec_column=args.dec_column,
    )

    pair_path = output_dir / f"{safe_a}_xmatch_{safe_b}.ecsv"
    a_only_path = output_dir / f"{safe_a}_only_vs_{safe_b}.ecsv"
    b_only_path = output_dir / f"{safe_b}_only_vs_{safe_a}.ecsv"
    a_matched_path = output_dir / f"{safe_a}_matched_to_{safe_b}.ecsv"
    b_matched_path = output_dir / f"{safe_b}_matched_to_{safe_a}.ecsv"

    _write_table(pair_path, pair_table, overwrite=args.overwrite)
    _write_table(a_only_path, catalog_a[only_a], overwrite=args.overwrite)
    _write_table(b_only_path, catalog_b[only_b], overwrite=args.overwrite)
    _write_table(
        a_matched_path,
        catalog_a[sorted(matched_a)],
        overwrite=args.overwrite,
    )
    _write_table(
        b_matched_path,
        catalog_b[sorted(matched_b)],
        overwrite=args.overwrite,
    )

    marker_radius = max(0.03, float(args.match_radius_arcsec))
    _write_ds9_regions(
        output_dir / f"{safe_a}_detections.reg",
        catalog_a,
        ra_column=args.ra_column,
        dec_column=args.dec_column,
        color="red",
        radius_arcsec=marker_radius,
        label=args.label_a,
        overwrite=args.overwrite,
    )
    _write_ds9_regions(
        output_dir / f"{safe_b}_detections.reg",
        catalog_b,
        ra_column=args.ra_column,
        dec_column=args.dec_column,
        color="blue",
        radius_arcsec=marker_radius,
        label=args.label_b,
        overwrite=args.overwrite,
    )

    data, header = _read_first_2d(reference_path)
    reference_wcs = WCS(header).celestial
    display, display_wcs, stride = _display_image(
        data,
        reference_wcs,
        max_display_pixels=args.max_display_pixels,
    )

    finite = np.isfinite(display)
    if not bool(finite.any()):
        raise ValueError("reference image contains no finite pixels")
    interval = PercentileInterval(float(args.display_percent))
    vmin, vmax = interval.get_limits(display[finite])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        clipped = SigmaClip(sigma=3.0, maxiters=5)(display[finite])
        values = np.asarray(clipped.compressed(), dtype=float)
        vmin, vmax = float(np.min(values)), float(np.max(values))
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())

    figure = plt.figure(figsize=(12, 10))
    ax = figure.add_subplot(111, projection=display_wcs)
    ax.imshow(display, origin="lower", cmap="gray", norm=norm, interpolation="nearest")

    index_a_plot = _marker_indices(args.plot_mode, len(catalog_a), matched_a)
    index_b_plot = _marker_indices(args.plot_mode, len(catalog_b), matched_b)

    transform = ax.get_transform("world")
    if index_a_plot.size:
        ax.scatter(
            np.asarray(catalog_a[args.ra_column][index_a_plot], dtype=float),
            np.asarray(catalog_a[args.dec_column][index_a_plot], dtype=float),
            transform=transform,
            s=float(args.marker_size),
            facecolors="none",
            edgecolors="red",
            linewidths=0.8,
            marker="o",
            label=f"{args.label_a}: {index_a_plot.size}",
        )
    if index_b_plot.size:
        ax.scatter(
            np.asarray(catalog_b[args.ra_column][index_b_plot], dtype=float),
            np.asarray(catalog_b[args.dec_column][index_b_plot], dtype=float),
            transform=transform,
            s=float(args.marker_size) * 0.8,
            color="blue",
            linewidths=0.8,
            marker="+",
            label=f"{args.label_b}: {index_b_plot.size}",
        )

    if args.line_matches and matches:
        for index_a, index_b, _ in matches:
            ax.plot(
                [
                    float(catalog_a[index_a][args.ra_column]),
                    float(catalog_b[index_b][args.ra_column]),
                ],
                [
                    float(catalog_a[index_a][args.dec_column]),
                    float(catalog_b[index_b][args.dec_column]),
                ],
                transform=transform,
                color="yellow",
                alpha=0.35,
                linewidth=0.5,
            )

    _apply_crop(
        ax,
        display_wcs,
        center_ra_deg=args.crop_center_ra_deg,
        center_dec_deg=args.crop_center_dec_deg,
        width_arcsec=args.crop_width_arcsec,
        height_arcsec=args.crop_height_arcsec,
    )

    ax.coords.grid(color="white", alpha=0.12, linestyle=":")
    ax.coords[0].set_axislabel("Right ascension (ICRS)")
    ax.coords[1].set_axislabel("Declination (ICRS)")
    ax.set_title(
        f"{args.reference_label}: {args.label_a} (red) vs {args.label_b} (blue)\n"
        f"one-to-one matches={len(matches)}, radius={args.match_radius_arcsec:.3f}\""
    )
    ax.legend(loc="upper right", framealpha=0.85)
    figure.tight_layout()

    png_path = output_dir / f"{safe_a}_vs_{safe_b}_overlay.png"
    pdf_path = output_dir / f"{safe_a}_vs_{safe_b}_overlay.pdf"
    if not args.overwrite:
        for path in (png_path, pdf_path):
            if path.exists():
                raise FileExistsError(path)
    figure.savefig(png_path, dpi=args.dpi)
    figure.savefig(pdf_path)
    plt.close(figure)

    separations = np.asarray([row[2] for row in matches], dtype=float)
    summary: dict[str, Any] = {
        "catalog_a": str(catalog_a_path),
        "catalog_b": str(catalog_b_path),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "reference_image": str(reference_path),
        "reference_label": args.reference_label,
        "match_policy": "all pairs within radius, then greedy closest one-to-one",
        "match_radius_arcsec": float(args.match_radius_arcsec),
        "count_a": int(len(catalog_a)),
        "count_b": int(len(catalog_b)),
        "matched_pairs": int(len(matches)),
        "a_matched_fraction": float(len(matched_a) / len(catalog_a))
        if len(catalog_a)
        else None,
        "b_matched_fraction": float(len(matched_b) / len(catalog_b))
        if len(catalog_b)
        else None,
        "a_only": int(len(only_a)),
        "b_only": int(len(only_b)),
        "separation_arcsec": {
            "median": _finite_or_none(np.median(separations))
            if separations.size
            else None,
            "q68": _finite_or_none(np.quantile(separations, 0.68))
            if separations.size
            else None,
            "q95": _finite_or_none(np.quantile(separations, 0.95))
            if separations.size
            else None,
            "max": _finite_or_none(np.max(separations))
            if separations.size
            else None,
        },
        "plot": {
            "mode": args.plot_mode,
            "reference_display_stride": int(stride),
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
        "products": {
            "pair_catalog": str(pair_path),
            "a_only_catalog": str(a_only_path),
            "b_only_catalog": str(b_only_path),
            "a_matched_catalog": str(a_matched_path),
            "b_matched_catalog": str(b_matched_path),
            "a_regions": str(output_dir / f"{safe_a}_detections.reg"),
            "b_regions": str(output_dir / f"{safe_b}_detections.reg"),
        },
    }

    summary_path = output_dir / "passband_detection_qa_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
