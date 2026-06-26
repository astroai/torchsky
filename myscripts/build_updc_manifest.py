#!/usr/bin/env python3
"""Pair corrected JWST exposures with exposure-specific HybPSF FITS files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from astropy.io import fits


def _filter_name(path: Path) -> str:
    with fits.open(path, memmap=True) as hdul:
        for header in (hdul["SCI"].header, hdul[0].header):
            for key in ("FILTER", "PUPIL"):
                value = str(header.get(key, "")).strip().upper()
                if value.startswith("F") and value.endswith("W"):
                    return value
    return ""


def _exposure_key(path: Path) -> str:
    name = path.name
    for suffix in (
        "_cal_1overf_completed.fits",
        "_cal_1overf.fits",
        "_cal.fits",
        ".fits",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _find_psf(psf_dir: Path, key: str) -> Path:
    candidates = sorted(psf_dir.glob(f"{key}*hybpsf*.fits"))
    if not candidates:
        candidates = sorted(psf_dir.glob(f"{key}*psf*.fits"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one PSF for {key!r}, found "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure-dir", type=Path, required=True)
    parser.add_argument("--psf-dir", type=Path, required=True)
    parser.add_argument("--target-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band", default="F277W")
    parser.add_argument("--module-token", default="nrcblong")
    parser.add_argument(
        "--exposure-glob",
        default="*_cal_1overf_completed.fits",
    )
    parser.add_argument("--expected-exposures", type=int, default=9)
    args = parser.parse_args()

    band = str(args.band).upper()
    module_token = str(args.module_token).lower()
    paths = [
        path
        for path in sorted(args.exposure_dir.glob(args.exposure_glob))
        if module_token in path.name.lower() and _filter_name(path) == band
    ]
    if args.expected_exposures and len(paths) != int(args.expected_exposures):
        raise RuntimeError(
            f"expected {args.expected_exposures} {band} exposures, found {len(paths)}"
        )

    observations = []
    for exposure in paths:
        key = _exposure_key(exposure)
        psf = _find_psf(args.psf_dir, key)
        observations.append(
            {
                "exposure": str(exposure.resolve()),
                "science_hdu": "SCI",
                "psf": str(psf.resolve()),
                "psf_hdu": 0,
            }
        )

    payload = {
        "target_reference": str(args.target_reference.resolve()),
        "metadata": {
            "band": band,
            "module_token": module_token,
            "input_stage": "Stage-2 cal + 1/f correction + abnormal-pixel replacement",
            "psf_stage": "exposure-specific double-sampled HybPSF",
        },
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output} with {len(observations)} observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
