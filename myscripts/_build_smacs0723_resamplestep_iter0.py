#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jwst
import stpipe
from astropy.io import fits
from jwst.resample import ResampleStep


def read_reference_grid(path: Path) -> dict:
    with fits.open(path, memmap=False) as hdul:
        ext = "SCI" if "SCI" in hdul else 1
        data = hdul[ext].data
        hdr = hdul[ext].header

        ny, nx = data.shape
        pixel_scale_arcsec = abs(float(hdr["CDELT1"])) * 3600.0

        return {
            "path": str(path),
            "shape_yx": [int(ny), int(nx)],
            "output_shape_xy": [int(nx), int(ny)],
            "pixel_scale_arcsec": pixel_scale_arcsec,
            "crpix_0based_xy": [
                float(hdr["CRPIX1"]) - 1.0,
                float(hdr["CRPIX2"]) - 1.0,
            ],
            "crval_xy": [
                float(hdr["CRVAL1"]),
                float(hdr["CRVAL2"]),
            ],
            "pc": {
                "PC1_1": hdr.get("PC1_1"),
                "PC1_2": hdr.get("PC1_2"),
                "PC2_1": hdr.get("PC2_1"),
                "PC2_2": hdr.get("PC2_2"),
            },
        }


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
    parser.add_argument("--reference-grid", type=Path, default=None)
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
        "asn_pool": f"local_smacs0723_{args.band.lower()}",
        "asn_id": "local",
        "version_id": "local-resamplestep",
        "program": "02736",
        "products": [
            {
                "name": args.output_name,
                "members": [
                    {"expname": path.name, "exptype": "science"}
                    for path in exposures
                ],
            }
        ],
    }

    asn_path = args.exposure_dir / f"{args.output_name}_asn.json"
    asn_path.write_text(json.dumps(asn, indent=2), encoding="utf-8")

    print("JWST:", jwst.__version__)
    print("stpipe:", getattr(stpipe, "__version__", "unknown"))
    print("ASN:", asn_path)
    print("n exposures:", len(exposures))
    for path in exposures:
        print("  ", path.name)

    step = ResampleStep()
    step.pixfrac = args.pixfrac
    step.kernel = args.kernel
    step.weight_type = args.weight_type
    step.output_dir = str(args.output_dir)
    step.output_file = args.output_name
    step.save_results = False

    reference_grid = None
    if args.reference_grid is not None:
        reference_grid = read_reference_grid(args.reference_grid)

        step.output_shape = reference_grid["output_shape_xy"]
        step.pixel_scale = reference_grid["pixel_scale_arcsec"]
        step.crpix = reference_grid["crpix_0based_xy"]
        step.crval = reference_grid["crval_xy"]

        print("\nUsing reference output grid:")
        print(" reference:", reference_grid["path"])
        print(" target shape y,x:", reference_grid["shape_yx"])
        print(" output_shape x,y:", step.output_shape)
        print(" pixel_scale arcsec:", step.pixel_scale)
        print(" crpix 0-based x,y:", step.crpix)
        print(" crval x,y:", step.crval)
        print(" reference PC:", reference_grid["pc"])
    else:
        step.pixel_scale_ratio = args.pixel_scale_ratio
        print("\nUsing automatic ResampleStep grid:")
        print(" pixel_scale_ratio:", step.pixel_scale_ratio)

    result = step.run(str(asn_path))

    out_path = args.output_dir / f"{args.output_name}_i2d.fits"
    if result is None:
        raise RuntimeError("ResampleStep returned None; no output model produced")

    result.save(str(out_path))
    print("Saved:", out_path)

    result.close()

    metadata = {
        "band": args.band,
        "asn": str(asn_path),
        "output": str(out_path),
        "n_exposures": len(exposures),
        "pixfrac": args.pixfrac,
        "kernel": args.kernel,
        "weight_type": args.weight_type,
        "reference_grid": reference_grid,
    }
    metadata_path = args.output_dir / f"{args.output_name}_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Metadata:", metadata_path)

    candidates = sorted(args.output_dir.glob(f"{args.output_name}*"))
    print("\nOutput candidates:")
    for path in candidates:
        if path.suffix.lower() in {".fits", ".fz"} or path.name.endswith(".fits"):
            try:
                with fits.open(path, memmap=False) as hdul:
                    ext = "SCI" if "SCI" in hdul else 1
                    shape = None if hdul[ext].data is None else hdul[ext].data.shape
                print(" ", path, "SCI shape:", shape)
            except Exception as exc:
                print(" ", path, "FITS read failed:", exc)
        else:
            print(" ", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())