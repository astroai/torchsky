#!/usr/bin/env python3
"""Pair corrected JWST exposures with their exposure-specific HybPSF files.

Ported from the UPDC experiment branch; reads filenames, not FITS contents.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def exposure_key(path: Path) -> str:
    for suffix in (
        "_cal_1overf_completed_dqfix.fits",
        "_cal_1overf_completed.fits",
        "_cal_completed_dqfix.fits",
        "_cal_completed.fits",
        "_cal_1overf.fits",
        "_cal.fits",
        ".fits",
    ):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def psf_key(path: Path) -> str:
    for suffix in ("_hybpsf_1501.fits", "_hybpsf.fits", ".fits"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure-dir", type=Path, required=True)
    parser.add_argument("--exposure-glob", default="*.fits")
    parser.add_argument("--psf-dir", type=Path, required=True)
    parser.add_argument("--psf-glob", default="*_hybpsf_1501.fits")
    parser.add_argument("--target-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band", default="")
    parser.add_argument("--expected-exposures", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.expected_exposures is not None and args.expected_exposures <= 0:
        parser.error("--expected-exposures must be positive when provided")
    if not args.target_reference.is_file():
        parser.error(f"missing target reference file: {args.target_reference}")

    exposures = sorted(args.exposure_dir.glob(args.exposure_glob))
    psfs = sorted(args.psf_dir.glob(args.psf_glob))
    if not exposures:
        parser.error("no exposures match the requested directory and glob")
    for paths, label in ((exposures, "exposures"), (psfs, "PSFs")):
        if (
            args.expected_exposures is not None
            and len(paths) != args.expected_exposures
        ):
            parser.error(
                f"expected {args.expected_exposures} {label}, found {len(paths)}"
            )
        for path in paths:
            if not path.is_file():
                parser.error(f"missing {label} file: {path}")

    psf_by_key: dict[str, Path] = {}
    for psf in psfs:
        key = psf_key(psf)
        if key in psf_by_key:
            parser.error(f"duplicate PSF key {key}: {psf_by_key[key]} and {psf}")
        psf_by_key[key] = psf

    observations = []
    exposure_keys: set[str] = set()
    for exposure in exposures:
        key = exposure_key(exposure)
        if key in exposure_keys:
            parser.error(f"duplicate exposure key {key}")
        exposure_keys.add(key)
        psf = psf_by_key.get(key)
        if psf is None:
            parser.error(f"missing PSF for exposure {exposure.name} (key {key})")
        observations.append(
            {"exposure": str(exposure.resolve()), "psf": str(psf.resolve())}
        )

    payload = {
        "band": args.band,
        "target_reference": str(args.target_reference.resolve()),
        "observations": observations,
    }
    if args.output.resolve() in {
        path.resolve() for path in [args.target_reference, *exposures, *psfs]
    }:
        parser.error("output must not replace an input file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
            if args.overwrite:
                os.replace(temporary, args.output)
            else:
                # Publish the complete file while atomically refusing an existing path.
                os.link(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)

    print(f"wrote {args.output}")
    print(f"paired {len(observations)} exposure/PSF pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
