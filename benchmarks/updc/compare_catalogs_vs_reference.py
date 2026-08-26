#!/usr/bin/env python3
"""QA a candidate catalog against the UPDC iter10-derived reference catalog.

Matching goes exclusively through ``torchsky.catalogs.crossmatch_sky`` with
``find="all"`` followed by deterministic maximum-cardinality one-to-one
association. Reports matched /
candidate-only / reference-only counts, separation statistics, and flux-ratio
statistics overall and per reference SNR bin. Results go to stdout plus an
optional JSON file (``--out``).

FITS tables are read through ``torchfits``. ECSV uses Astropy's text-table
reader when installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

BENCH_DIR = Path(__file__).resolve().parents[1]
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

SNR_BIN_EDGES = (5.0, 10.0, 20.0, 50.0)


def _require_astropy():
    try:
        from astropy.table import Table
    except ImportError as exc:
        raise RuntimeError(
            "compare_catalogs_vs_reference requires astropy "
            "(pip install 'torchsky[catalog-qa]')"
        ) from exc
    return Table


def _column(table, name: str) -> np.ndarray:
    names = (
        list(table.column_names)
        if hasattr(table, "column_names")
        else list(table.colnames)
    )
    if name not in names:
        raise ValueError(
            f"column {name!r} not found; available: {names}"
        )
    column = table[name]
    if hasattr(column, "to_numpy"):
        column = column.to_numpy(zero_copy_only=False)
    return np.array(column, dtype=np.float64, copy=True).reshape(-1)


def _read_catalog(path: Path, *, ra_col: str, dec_col: str, flux_col: str):
    if path.suffix.lower() in {".fits", ".fit", ".fts", ".fz"}:
        import torchfits

        table = torchfits.table.read(str(path), hdu=1)
    else:
        table = _require_astropy().read(str(path))
    return {
        "ra_deg": _column(table, ra_col),
        "dec_deg": _column(table, dec_col),
        "flux": _column(table, flux_col),
    }


def _maximum_cardinality_pairs(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve radius candidates into a deterministic one-to-one association."""
    all_left = result.left_index.detach().cpu().numpy().astype(np.int64, copy=False)
    all_right = result.right_index.detach().cpu().numpy().astype(np.int64, copy=False)
    all_sep = result.separation_arcsec.detach().cpu().numpy()
    adjacency: dict[int, list[tuple[int, float]]] = {}
    separation_by_pair: dict[tuple[int, int], float] = {}
    for left, right, separation in zip(all_left, all_right, all_sep, strict=True):
        left_i, right_i, separation_f = int(left), int(right), float(separation)
        adjacency.setdefault(left_i, []).append((right_i, separation_f))
        separation_by_pair[(left_i, right_i)] = separation_f
    for candidates in adjacency.values():
        candidates.sort(key=lambda item: (item[1], item[0]))

    # Augmenting paths give maximum cardinality. Trying the least-separated
    # sources and edges first makes ties deterministic and favors local pairs.
    left_order = sorted(
        adjacency,
        key=lambda left: (adjacency[left][0][1], left),
    )
    left_match: dict[int, int] = {}
    right_match: dict[int, int] = {}
    for root in left_order:
        if root in left_match:
            continue
        queue = [root]
        seen_left = {root}
        seen_right: set[int] = set()
        parent_right: dict[int, int] = {}
        free_right: int | None = None
        for left in queue:
            for right, _ in adjacency[left]:
                if right in seen_right:
                    continue
                seen_right.add(right)
                parent_right[right] = left
                owner = right_match.get(right)
                if owner is None:
                    free_right = right
                    break
                if owner not in seen_left:
                    seen_left.add(owner)
                    queue.append(owner)
            if free_right is not None:
                break
        if free_right is None:
            continue
        right = free_right
        while True:
            left = parent_right[right]
            previous_right = left_match.get(left)
            left_match[left] = right
            right_match[right] = left
            if previous_right is None:
                break
            right_match.pop(previous_right)
            right = previous_right

    pairs = sorted(left_match.items())
    left = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    right = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    separation = np.asarray(
        [separation_by_pair[pair] for pair in pairs], dtype=np.float64
    )
    return left, right, separation


def _ratio_stats(ratio: np.ndarray) -> dict[str, float] | None:
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        return None
    return {
        "median": float(np.median(finite)),
        "q16": float(np.percentile(finite, 16)),
        "q84": float(np.percentile(finite, 84)),
    }


def compare_catalogs(
    candidate_ra: np.ndarray,
    candidate_dec: np.ndarray,
    candidate_flux: np.ndarray,
    reference_ra: np.ndarray,
    reference_dec: np.ndarray,
    reference_flux: np.ndarray,
    *,
    radius_arcsec: float,
    reference_snr: np.ndarray | None = None,
) -> dict[str, Any]:
    from torchsky.catalogs.core import crossmatch_sky

    result = crossmatch_sky(
        torch.as_tensor(candidate_ra, dtype=torch.float64),
        torch.as_tensor(candidate_dec, dtype=torch.float64),
        torch.as_tensor(reference_ra, dtype=torch.float64),
        torch.as_tensor(reference_dec, dtype=torch.float64),
        radius_arcsec=float(radius_arcsec),
        find="all",
    )
    left, right, separation = _maximum_cardinality_pairs(result)

    n_candidate = int(candidate_ra.size)
    n_reference = int(reference_ra.size)
    matched = int(left.size)
    report: dict[str, Any] = {
        "n_candidate": n_candidate,
        "n_reference": n_reference,
        "matched": matched,
        "candidate_only": n_candidate - matched,
        "reference_only": n_reference - int(np.unique(right).size),
        "radius_arcsec": float(radius_arcsec),
        "association": "maximum_cardinality_one_to_one",
    }
    if matched:
        report["separation_arcsec"] = {
            "median": float(np.median(separation)),
            "p95": float(np.percentile(separation, 95)),
        }
        stats = _ratio_stats(candidate_flux[left] / reference_flux[right])
        report["flux_ratio"] = stats
    else:
        report["separation_arcsec"] = None
        report["flux_ratio"] = None

    snr_bins: list[dict[str, Any]] = []
    if reference_snr is not None and matched:
        snr_matched = reference_snr[right]
        edges = (-np.inf, *SNR_BIN_EDGES, np.inf)
        labels = ["<5", "5-10", "10-20", "20-50", ">=50"]
        for lo, hi, label in zip(edges[:-1], edges[1:], labels):
            sel = (snr_matched >= lo) & (snr_matched < hi)
            entry: dict[str, Any] = {"snr_bin": label, "matched": int(sel.sum())}
            stats = _ratio_stats(
                candidate_flux[left[sel]] / reference_flux[right[sel]]
            )
            if stats is not None:
                entry["flux_ratio"] = stats
            snr_bins.append(entry)
    report["snr_bins"] = snr_bins
    return report


def detect_candidates_photutils(
    image_paths: list[Path],
    *,
    fwhm_pixels: float,
    detection_sigma: float,
):
    """Detect sources in SCI images with photutils DAOStarFinder."""
    try:
        from photutils.detection import DAOStarFinder
    except ImportError as exc:
        print(
            "photutils detection requires the 'catalog-qa' extras group "
            "(pip install 'torchsky[catalog-qa]')",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    ras: list[float] = []
    decs: list[float] = []
    fluxes: list[float] = []
    import torchfits

    from torchsky.wcs import WCS

    for path in image_paths:
        image, header = torchfits.read(
            str(path), hdu="SCI", return_header=True, mmap=False
        )
        data = image.detach().cpu().numpy().astype(np.float32, copy=False)
        finite = np.isfinite(data)
        background = float(np.median(data[finite]))
        noise = float(1.4826 * np.median(np.abs(data[finite] - background)))
        finder = DAOStarFinder(
            fwhm=float(fwhm_pixels),
            threshold=float(detection_sigma) * noise,
        )
        sources = finder(data - background)
        if sources is None or len(sources) == 0:
            continue
        world = WCS(header).pixel_to_world(
            torch.as_tensor(np.asarray(sources["xcentroid"], dtype=np.float64)),
            torch.as_tensor(np.asarray(sources["ycentroid"], dtype=np.float64)),
            origin=0,
        )
        ras.extend(world[0].detach().cpu().reshape(-1).tolist())
        decs.extend(world[1].detach().cpu().reshape(-1).tolist())
        fluxes.extend(
            np.asarray(sources["flux"], dtype=np.float64).reshape(-1).tolist()
        )
    return (
        np.asarray(ras, dtype=np.float64),
        np.asarray(decs, dtype=np.float64),
        np.asarray(fluxes, dtype=np.float64),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, help="candidate catalog (FITS/ECSV)")
    parser.add_argument("--reference", type=Path, required=True, help="reference catalog (FITS/ECSV)")
    parser.add_argument("--ra-col", default="ra")
    parser.add_argument("--dec-col", default="dec")
    parser.add_argument("--flux-col", default="flux")
    parser.add_argument("--snr-col", default=None, help="reference SNR column for binned ratios")
    parser.add_argument("--radius-arcsec", type=float, default=0.25)
    parser.add_argument(
        "--detect-images",
        type=Path,
        default=None,
        metavar="DIR",
        help="detect candidates from SCI FITS images in DIR instead of --candidate",
    )
    parser.add_argument("--fwhm-pixels", type=float, default=3.0)
    parser.add_argument("--detection-sigma", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    reference = _read_catalog(
        args.reference,
        ra_col=args.ra_col,
        dec_col=args.dec_col,
        flux_col=args.flux_col,
    )

    if args.detect_images is not None:
        image_paths = sorted(args.detect_images.glob("*.fits"))
        if not image_paths:
            raise SystemExit(f"no FITS images found in {args.detect_images}")
        cand_ra, cand_dec, cand_flux = detect_candidates_photutils(
            image_paths,
            fwhm_pixels=args.fwhm_pixels,
            detection_sigma=args.detection_sigma,
        )
        candidate_source: Any = args.detect_images
    elif args.candidate is not None:
        candidate = _read_catalog(
            args.candidate,
            ra_col=args.ra_col,
            dec_col=args.dec_col,
            flux_col=args.flux_col,
        )
        cand_ra, cand_dec, cand_flux = (
            candidate["ra_deg"],
            candidate["dec_deg"],
            candidate["flux"],
        )
        candidate_source = args.candidate
    else:
        raise SystemExit("either --candidate or --detect-images is required")

    reference_snr = None
    if args.snr_col is not None:
        if args.reference.suffix.lower() in {".fits", ".fit", ".fts", ".fz"}:
            import torchfits

            reference_table = torchfits.table.read(str(args.reference), hdu=1)
        else:
            reference_table = _require_astropy().read(str(args.reference))
        reference_snr = _column(reference_table, args.snr_col)

    report = compare_catalogs(
        cand_ra,
        cand_dec,
        cand_flux,
        reference["ra_deg"],
        reference["dec_deg"],
        reference["flux"],
        radius_arcsec=args.radius_arcsec,
        reference_snr=reference_snr,
    )
    report["candidate_source"] = str(candidate_source)
    report["reference"] = str(args.reference)

    summary = (
        f"matched={report['matched']} "
        f"candidate_only={report['candidate_only']} "
        f"reference_only={report['reference_only']}"
    )
    if report.get("separation_arcsec"):
        summary += (
            f" sep_med={report['separation_arcsec']['median']:.4f}\""
            f" sep_p95={report['separation_arcsec']['p95']:.4f}\""
        )
    if report.get("flux_ratio"):
        summary += f" flux_ratio_med={report['flux_ratio']['median']:.4f}"
    print(summary)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
