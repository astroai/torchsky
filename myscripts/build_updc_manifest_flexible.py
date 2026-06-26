#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def exposure_key(path: Path) -> str:
    name = path.name
    suffixes = [
        "_cal_1overf_completed_dqfix.fits",
        "_cal_1overf_completed.fits",
        "_cal_1overf.fits",
        "_cal.fits",
        ".fits",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def psf_key(path: Path) -> str:
    name = path.name
    suffixes = [
        "_hybpsf_1501.fits",
        "_hybpsf.fits",
        ".fits",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure-dir", type=Path, required=True)
    parser.add_argument("--exposure-glob", default="*.fits")
    parser.add_argument("--psf-dir", type=Path, required=True)
    parser.add_argument("--psf-glob", default="*_hybpsf_1501.fits")
    parser.add_argument("--target-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band", default="")
    parser.add_argument("--expected-exposures", type=int, default=0)
    args = parser.parse_args()

    exposures = sorted(args.exposure_dir.glob(args.exposure_glob))
    psfs = sorted(args.psf_dir.glob(args.psf_glob))

    if args.expected_exposures and len(exposures) != args.expected_exposures:
        raise RuntimeError(f"expected {args.expected_exposures} exposures, found {len(exposures)}")
    if args.expected_exposures and len(psfs) != args.expected_exposures:
        raise RuntimeError(f"expected {args.expected_exposures} PSFs, found {len(psfs)}")

    psf_by_key = {}
    for psf in psfs:
        key = psf_key(psf)
        if key in psf_by_key:
            raise RuntimeError(f"duplicate PSF key {key}: {psf_by_key[key]} and {psf}")
        psf_by_key[key] = psf

    observations = []
    missing = []
    for exposure in exposures:
        key = exposure_key(exposure)
        psf = psf_by_key.get(key)
        if psf is None:
            missing.append((exposure.name, key))
            continue
        observations.append(
            {
                "exposure": str(exposure.resolve()),
                "psf": str(psf.resolve()),
            }
        )

    if missing:
        print("Available PSF keys:")
        for key in sorted(psf_by_key):
            print("  ", key)
        print("\nMissing exposure keys:")
        for name, key in missing:
            print("  ", name, "=>", key)
        raise RuntimeError(f"missing PSF for {len(missing)} exposure(s)")

    payload = {
        "band": args.band,
        "target_reference": str(args.target_reference.resolve()),
        "observations": observations,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {args.output}")
    print(f"paired {len(observations)} exposure/PSF pairs")
    for row in observations:
        print(Path(row["exposure"]).name, "<->", Path(row["psf"]).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
