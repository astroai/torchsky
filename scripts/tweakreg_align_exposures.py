#!/usr/bin/env python3
"""Align JWST exposures with the official jwst ``TweakRegStep``.

Thin CLI wrapper over the JWST pipeline: inputs are never overwritten; every
corrected copy is written to ``--output-dir`` as
``{index:03d}_{stem}_{fitgeometry}_tweakreg.fits`` — the naming consumed by
``benchmarks/bench_map_smacs0723_real.py`` via the ``*_rshift_tweakreg.fits``
glob — with source catalogs saved as ECSV under ``<output-dir>/catalogs/``.

Requires the optional ``tweakreg`` extras group (jwst); import it with
``pip install 'torchsky[tweakreg]'``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _output_name(index: int, stem: str, fitgeometry: str) -> str:
    """Output file name for one aligned exposure (matches *_rshift_tweakreg glob)."""
    return f"{index:03d}_{stem}_{fitgeometry}_tweakreg.fits"


def _validate_sci_wcs(path: Path) -> None:
    """Fail fast when the saved SCI FITS-WCS has a non-finite footprint."""
    from astropy.io import fits as astro_fits
    from astropy.wcs import WCS as AstropyWCS

    try:
        header = dict(astro_fits.getheader(str(path), "SCI"))
        footprint = AstropyWCS(header).calc_footprint()
    except Exception as exc:
        raise RuntimeError(f"invalid saved TweakReg WCS in {path}") from exc
    if (
        footprint is None
        or footprint.size == 0
        or not bool(np.isfinite(footprint).all())
    ):
        raise RuntimeError(f"invalid saved TweakReg WCS in {path}")


def _band_and_detector(path: Path) -> tuple[str, str]:
    """Best-effort (band, detector) tags from the CAL header or filename."""
    band = ""
    detector = ""
    try:
        from astropy.io import fits as astro_fits

        with astro_fits.open(str(path), memmap=False) as hdul:
            head = hdul[0].header
            band = str(head.get("FILTER", "") or head.get("PUPIL", ""))
            detector = str(head.get("DETECTOR", ""))
    except Exception:
        pass
    if not detector:
        parts = path.stem.split("_")
        detector = parts[4] if len(parts) > 4 else ""
    return band, detector


def run_tweakreg_alignment(
    selected,
    *,
    output_dir: Path,
    fitgeometry: str = "rshift",
    snr_threshold: float = 10.0,
    minobj: int = 15,
    searchrad: float = 2.0,
    separation: float = 1.0,
    tolerance: float = 0.7,
    nclip: int = 3,
    sigma: float = 3.0,
):
    """Run TweakReg once across all selected exposures and save corrected copies.

    ``selected`` is a sequence of ``(path, band, detector)`` triples. Returns
    ``(corrected, products_info)`` where corrected mirrors ``selected`` with
    aligned destinations.
    """
    if fitgeometry not in {"shift", "rshift", "rscale", "general"}:
        raise ValueError("invalid TweakReg fitgeometry")
    if len(selected) < 2:
        raise ValueError("TweakReg alignment requires at least two exposures")

    try:
        from jwst.datamodels import ModelLibrary
        from jwst.tweakreg import TweakRegStep
    except ImportError as exc:
        raise RuntimeError(
            "tweakreg alignment requires the jwst pipeline environment"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep source catalogs. They are extremely useful when TweakReg reports
    # too few/mismatched sources, and they are not used as science catalogs.
    catalog_dir = output_dir / "catalogs"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    # Lazy ModelLibrary over paths: inputs stay closed until borrowed and are
    # never modified in place.
    inputs = ModelLibrary([str(path) for path, _, _ in selected])

    try:
        aligned = TweakRegStep.call(
            inputs,
            fitgeometry=fitgeometry,
            snr_threshold=snr_threshold,
            minobj=minobj,
            searchrad=searchrad,
            separation=separation,
            tolerance=tolerance,
            nclip=nclip,
            sigma=sigma,
            starfinder="iraf",
            use_custom_catalogs=False,
            save_catalogs=True,
            catalog_format="ecsv",
            output_dir=str(catalog_dir),
            save_results=False,
        )

        if len(aligned) != len(selected):
            raise RuntimeError(
                "TweakReg returned a different number of exposures: "
                f"{len(aligned)} != {len(selected)}"
            )

        corrected = []
        products = []

        # jwst >= 1.20 returns a closed ModelLibrary: borrow only inside the
        # context manager and shelve without modifications.
        with aligned:
            for index, (source, band, detector) in enumerate(selected):
                model = aligned.borrow(index)
                try:
                    destination = output_dir / _output_name(
                        index, source.stem, fitgeometry
                    )
                    model.save(str(destination), overwrite=True)
                    status = str(
                        getattr(getattr(model.meta, "cal_step", None), "tweakreg", "UNKNOWN")
                    )
                    # Validate the exact FITS-WCS representation consumers read.
                    _validate_sci_wcs(destination)
                    corrected.append((destination.resolve(), band, detector))
                    products.append(
                        {
                            "source": str(source),
                            "corrected": str(destination.resolve()),
                            "band": band,
                            "detector": detector,
                            "status": status,
                        }
                    )
                finally:
                    aligned.shelve(model, index, modify=False)
                    del model

        failed = [p for p in products if str(p["status"]).upper() != "COMPLETE"]
        if failed:
            names = ", ".join(Path(p["source"]).name for p in failed[:5])
            raise RuntimeError(
                f"TweakReg did not complete for: {names}. "
                f"Inspect source catalogs in {catalog_dir}"
            )

        return corrected, {
            "fitgeometry": fitgeometry,
            "catalog_dir": str(catalog_dir.resolve()),
            "products": products,
        }
    finally:
        inputs.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="CAL/CRF exposures to align (never overwritten)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fitgeometry", default="rshift",
                        choices=["shift", "rshift", "rscale", "general"])
    parser.add_argument("--snr-threshold", type=float, default=10.0)
    parser.add_argument("--minobj", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        print(f"missing input files: {', '.join(map(str, missing[:5]))}", file=sys.stderr)
        return 1
    selected = [(path, *_band_and_detector(path)) for path in args.inputs]
    try:
        corrected, info = run_tweakreg_alignment(
            selected,
            output_dir=args.output_dir,
            fitgeometry=args.fitgeometry,
            snr_threshold=args.snr_threshold,
            minobj=args.minobj,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    info["corrected"] = [str(path) for path, _, _ in corrected]
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
