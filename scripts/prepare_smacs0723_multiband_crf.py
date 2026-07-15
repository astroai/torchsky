#!/usr/bin/env python3
"""Audit, download, and organize all-band SMACS J0723 NIRCam CRF inputs.

Default layout::

    /arc/projects/ots/jwst_smacs0723/
    ├── input/
    │   ├── F090W/*.fits
    │   ├── F150W/*.fits
    │   ├── F200W/*.fits
    │   ├── F277W/*.fits
    │   ├── F356W/*.fits
    │   ├── F444W/*.fits
    │   └── crf_inventory.json
    └── results/

Existing verified CRF files found below the project root are adopted into the
input layout with symlinks by default. Missing public products are queried and
downloaded directly from MAST using the repository's existing downloader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from download_jwst_smacs0723_mast import (  # noqa: E402
    download_product,
    select_products,
)

ALL_BANDS = ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")
DEFAULT_PROJECT_ROOT = Path("/arc/projects/ots/jwst_smacs0723")


def _parse_csv(value: str) -> list[str]:
    return [item.strip().upper() for item in str(value).split(",") if item.strip()]


def _header_value(primary: fits.Header, science: fits.Header, key: str) -> str:
    return str(science.get(key) or primary.get(key) or "").strip()


def inspect_crf(path: Path) -> dict[str, Any] | None:
    """Return verified CRF metadata, or ``None`` when the file is unsuitable."""

    if not path.name.lower().endswith("_crf.fits"):
        return None
    try:
        with fits.open(path, memmap=False) as hdul:
            if "SCI" not in hdul or "ERR" not in hdul:
                return None
            primary = hdul[0].header
            science = hdul["SCI"].header
            band = _header_value(primary, science, "FILTER").upper()
            pupil = _header_value(primary, science, "PUPIL").upper()
            if not (band.startswith("F") and band.endswith("W")):
                band = pupil
            detector = _header_value(primary, science, "DETECTOR").lower()
            image = hdul["SCI"].data
            err = hdul["ERR"].data
            if image is None or err is None or image.ndim != 2 or err.shape != image.shape:
                return None
            return {
                "path": str(path.resolve()),
                "filename": path.name,
                "band": band,
                "detector": detector,
                "shape": [int(image.shape[0]), int(image.shape[1])],
                "size_bytes": int(path.stat().st_size),
            }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def detector_matches(metadata: dict[str, Any], *, band: str, module: str) -> bool:
    detector = str(metadata["detector"]).lower()
    module = str(module).lower()
    if module == "any":
        return True
    if band in {"F277W", "F356W", "F444W"}:
        return detector == f"nrc{module}long"
    return detector in {f"nrc{module}{index}" for index in range(1, 5)}


def scan_existing(
    project_root: Path,
    *,
    input_dir: Path,
    results_dir: Path,
    bands: list[str],
    module: str,
) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    found: dict[str, list[tuple[Path, dict[str, Any]]]] = {band: [] for band in bands}
    for path in project_root.rglob("*_crf.fits"):
        resolved = path.resolve()
        if input_dir in resolved.parents or results_dir in resolved.parents:
            continue
        metadata = inspect_crf(path)
        if metadata is None:
            continue
        band = str(metadata["band"]).upper()
        if band not in found or not detector_matches(metadata, band=band, module=module):
            continue
        found[band].append((path, metadata))
    for band in found:
        found[band].sort(key=lambda item: item[0].name)
    return found


def adopt_file(source: Path, destination: Path, *, mode: str, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return
        destination.unlink()
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "move":
        shutil.move(str(source), str(destination))
    else:
        raise ValueError("adopt_mode must be symlink, copy, or move")


def audit_input(input_dir: Path, *, bands: list[str], module: str) -> dict[str, list[dict[str, Any]]]:
    inventory: dict[str, list[dict[str, Any]]] = {band: [] for band in bands}
    for band in bands:
        band_dir = input_dir / band
        if not band_dir.exists():
            continue
        for path in sorted(band_dir.glob("*_crf.fits")):
            metadata = inspect_crf(path)
            if metadata is None:
                continue
            if str(metadata["band"]).upper() != band:
                continue
            if not detector_matches(metadata, band=band, module=module):
                continue
            inventory[band].append(metadata)
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--bands", default=",".join(ALL_BANDS))
    parser.add_argument("--module", choices=("a", "b", "any"), default="b")
    parser.add_argument("--proposal-id", default="2736")
    parser.add_argument("--target-name", default="SMACS-J0723.3-7327")
    parser.add_argument("--max-exposures-per-band", type=int, default=100)
    parser.add_argument(
        "--adopt-mode",
        choices=("symlink", "copy", "move"),
        default="symlink",
        help="How to place already-downloaded verified CRFs into input/<band>.",
    )
    parser.add_argument("--no-adopt-existing", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail if any requested band remains empty.")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    input_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir is not None
        else project_root / "input"
    )
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir is not None
        else project_root / "results"
    )
    bands = _parse_csv(args.bands)
    unsupported = sorted(set(bands).difference(ALL_BANDS))
    if unsupported:
        parser.error("unsupported bands: " + ", ".join(unsupported))

    input_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    for band in bands:
        (input_dir / band).mkdir(parents=True, exist_ok=True)

    if not args.no_adopt_existing:
        existing = scan_existing(
            project_root,
            input_dir=input_dir,
            results_dir=results_dir,
            bands=bands,
            module=args.module,
        )
        for band, items in existing.items():
            for source, _ in items:
                destination = input_dir / band / source.name
                adopt_file(
                    source,
                    destination,
                    mode=args.adopt_mode,
                    overwrite=bool(args.overwrite),
                )

    before_download = audit_input(input_dir, bands=bands, module=args.module)
    print("Verified CRF inventory before download:")
    for band in bands:
        print(f"  {band}: {len(before_download[band])}")

    queried: dict[str, list[str]] = {band: [] for band in bands}
    if not args.no_download:
        for band in bands:
            products = select_products(
                band=band,
                subgroup="CRF",
                max_exposures=int(args.max_exposures_per_band),
                module=args.module,
                proposal_id=str(args.proposal_id),
                target_name=str(args.target_name),
            )
            queried[band] = [product.product_filename for product in products]
            existing_names = {entry["filename"] for entry in before_download[band]}
            missing_products = [
                product for product in products if product.product_filename not in existing_names
            ]
            print(
                f"{band}: MAST listed {len(products)} CRFs; "
                f"downloading {len(missing_products)} missing files"
            )
            for product in missing_products:
                destination = download_product(
                    product,
                    input_dir / band,
                    overwrite=bool(args.overwrite),
                )
                metadata = inspect_crf(destination)
                if metadata is None or str(metadata["band"]).upper() != band:
                    raise RuntimeError(f"downloaded file failed CRF verification: {destination}")
                if not detector_matches(metadata, band=band, module=args.module):
                    raise RuntimeError(
                        f"downloaded detector does not match module {args.module}: {destination}"
                    )

    inventory = audit_input(input_dir, bands=bands, module=args.module)
    missing_bands = [band for band in bands if not inventory[band]]
    manifest = {
        "project_root": str(project_root),
        "input_dir": str(input_dir),
        "results_dir": str(results_dir),
        "module": args.module,
        "bands": bands,
        "counts": {band: len(inventory[band]) for band in bands},
        "files": inventory,
        "mast_products": queried,
        "missing_bands": missing_bands,
    }
    manifest_path = input_dir / "crf_inventory.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Final verified CRF inventory:")
    for band in bands:
        print(f"  {band}: {len(inventory[band])}")
        for entry in inventory[band]:
            print(f"    {entry['filename']}  detector={entry['detector']}")
    print(f"inventory: {manifest_path}")
    print(f"results directory: {results_dir}")

    if args.strict and missing_bands:
        raise RuntimeError("no verified CRF files for: " + ", ".join(missing_bands))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
