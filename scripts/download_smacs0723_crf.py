#!/usr/bin/env python3
"""Download SMACS J0723 JWST/NIRCam F277W NRCBLONG CRF exposures from MAST."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from astroquery.mast import Observations
from astropy.io import fits


def _is_f277w_nrcblong_crf(path: Path) -> bool:
    """Verify downloaded product by opening FITS headers."""
    try:
        with fits.open(path, memmap=False) as hdul:
            primary = hdul[0].header
            sci = hdul["SCI"].header if "SCI" in hdul else primary

            filt = str(sci.get("FILTER") or primary.get("FILTER") or "").upper()
            pupil = str(sci.get("PUPIL") or primary.get("PUPIL") or "").upper()
            detector = str(
                sci.get("DETECTOR") or primary.get("DETECTOR") or ""
            ).lower()

            is_f277w = (filt == "F277W") or (pupil == "F277W")
            is_nrcblong = detector == "nrcblong"
            return is_f277w and is_nrcblong
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-id", default="2736")
    parser.add_argument("--target", default="SMACS J0723")
    parser.add_argument("--output-dir", type=Path, default=Path("data/jwst_smacs0723"))
    parser.add_argument("--expected", type=int, default=9)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    obs = Observations.query_criteria(
        obs_collection="JWST",
        proposal_id=str(args.proposal_id),
        dataproduct_type="image",
        instrument_name="NIRCAM/IMAGE",
        target_name="SMACS*",
    )

    products = Observations.get_product_list(obs)

    # First filter by filename. This is the most reliable way to get CRF files.
    crf = products[
        [
            str(name).lower().endswith("_crf.fits")
            for name in products["productFilename"]
        ]
    ]

    # Keep F277W and NRCBLONG by filename first, then verify by FITS header.
    # JWST filenames usually contain detector, e.g. nrcblong,
    # but not necessarily filter name such as F277W.
    # So: first keep NRCBLONG CRF by filename; verify F277W later from FITS header.
    candidates = crf[
        [
        "nrcblong" in str(name).lower()
        for name in crf["productFilename"]
        ]
    ]

    if len(candidates) == 0:
        print("Available CRF filenames:")
        for name in crf["productFilename"]:
            print("  ", name)
        raise RuntimeError(
            "No NRCBLONG *_crf.fits products found. "
            "The MAST query may be too restrictive, or CRF products are not listed."
        )

    manifest = Observations.download_products(
        candidates,
        download_dir=str(args.output_dir),
        extension="fits",
    )

    final_paths: list[Path] = []
    for row in manifest:
        local = Path(str(row["Local Path"]))
        if not local.exists():
            continue

        dest = args.output_dir / local.name
        if dest.exists() and not args.overwrite:
            final_paths.append(dest)
            continue

        shutil.move(str(local), str(dest))
        final_paths.append(dest)

    verified = [p for p in sorted(final_paths) if _is_f277w_nrcblong_crf(p)]

    print(f"Downloaded candidate CRF files: {len(final_paths)}")
    print(f"Verified F277W/NRCBLONG CRF files: {len(verified)}")
    for p in verified:
        print(p)

    if args.expected and len(verified) != int(args.expected):
        raise RuntimeError(
            f"Expected {args.expected} verified CRF exposures, got {len(verified)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())