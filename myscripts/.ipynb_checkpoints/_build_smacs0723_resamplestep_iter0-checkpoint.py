#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jwst
import stpipe
from astropy.io import fits
from jwst.resample import ResampleStep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure-dir", type=Path, required=True)
    parser.add_argument("--exposure-glob", default="*_completed_dqfix.fits")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="_F277W_resamplestep_iter0")
    parser.add_argument("--band", default="F277W")
    parser.add_argument("--expected-exposures", type=int, default=0)
    parser.add_argument("--pixfrac", type=float, default=1.0)
    parser.add_argument("--pixel-scale-ratio", type=float, default=0.5)
    parser.add_argument("--kernel", default="square")
    parser.add_argument("--weight-type", default="ivm")
    args = parser.parse_args()

    exposures = sorted(args.exposure_dir.glob(args.exposure_glob))
    if args.expected_exposures and len(exposures) != args.expected_exposures:
        raise RuntimeError(
            f"expected {args.expected_exposures} exposures, found {len(exposures)}"
        )
    if not exposures:
        raise RuntimeError(f"no exposures found in {args.exposure_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    asn = {
        "asn_type": "image3",
        "asn_rule": "DMS_Level3_Base",
        "version_id": "local-resamplestep",
        "program": "02736",
        "products": [
            {
                "name": args.output_name,
                "members": [
                    {"expname": str(path.resolve()), "exptype": "science"}
                    for path in exposures
                ],
            }
        ],
    }

    asn_path = args.output_dir / f"{args.output_name}_asn.json"
    asn_path.write_text(json.dumps(asn, indent=2), encoding="utf-8")

    print("JWST:", jwst.__version__)
    print("stpipe:", getattr(stpipe, "__version__", "unknown"))
    print("ASN:", asn_path)
    print("n exposures:", len(exposures))
    for path in exposures:
        print("  ", path.name)

    result = ResampleStep.call(
        str(asn_path),
        output_dir=str(args.output_dir),
        output_file=args.output_name,
        save_results=True,
        pixfrac=args.pixfrac,
        kernel=args.kernel,
        pixel_scale_ratio=args.pixel_scale_ratio,
        weight_type=args.weight_type,
    )

    if result is not None:
        result.close()

    candidates = sorted(args.output_dir.glob(f"{args.output_name}*"))
    print("\nOutput candidates:")
    for path in candidates:
        if path.suffix.lower() in {".fits", ".fz"} or path.name.endswith(".fits"):
            try:
                with fits.open(path, memmap=False) as hdul:
                    shape = None if hdul[1].data is None else hdul[1].data.shape
                print(" ", path, "SCI shape:", shape)
            except Exception as exc:
                print(" ", path, "FITS read failed:", exc)
        else:
            print(" ", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
