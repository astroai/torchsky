#!/usr/bin/env python
"""Propagate a solved WCS across all extensions of a MEF file, or merge
SCAMP-refined headers back into a MEF.

Mode: propagate
  Apply CD + distortion (TPV by default, SIP optional) from a solved WCS to all
  HDUs in a MEF, recomputing each HDU's CRVAL from its own CRPIX.

Mode: merge-scamp
  Apply per-extension .head files from SCAMP back into the MEF.

Examples
--------
  # Propagate from astrometry.net .wcs file:
  python scripts/apply_wcs_to_mef.py propagate \\
      --mef raw.fits.fz \\
      --solved-wcs ccd19.solved.wcs \\
      --output propagated.fits

  # Propagate from a reference HDU within the MEF:
  python scripts/apply_wcs_to_mef.py propagate \\
      --mef raw.fits.fz \\
      --ref-hdu 19 \\
      --output propagated.fits

  # Merge SCAMP per-extension headers:
  python scripts/apply_wcs_to_mef.py merge-scamp \\
      --mef raw.fits.fz \\
      --scamp-prefix scamp/raw \\
      --output calibrated.fits
"""

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torchfits
from torchfits.header_parser import FastHeaderParser

from torchsky.wcs import WCS


def _read_external_header(path: str) -> dict[str, Any]:
    """Parse the FITS-card text emitted by astrometry.net or SCAMP."""
    raw = Path(path).read_text(encoding="ascii")
    if "\n" in raw or "\r" in raw:
        cards = [line[:80].ljust(80) for line in raw.splitlines() if line.strip()]
    else:
        cards = [raw[i : i + 80].ljust(80) for i in range(0, len(raw), 80)]
    cleaned = [
        card for card in cards if card[:8].strip() not in {"", "CONTINUE", "END"}
    ]
    return FastHeaderParser.parse_header_string("".join(cleaned))


def _read_wcs_file(path: str) -> dict[str, Any]:
    """Read an astrometry.net .wcs text header into a plain mapping."""
    return _read_external_header(path)


def _read_scamp_header(path: str) -> dict[str, Any]:
    """Read a SCAMP .head text header into a plain mapping."""
    return _read_external_header(path)


def _propagate_hdu(
    solved_wcs: WCS,
    solved_header: dict,
    header: Mapping[str, Any],
    hdu_index: int,
    distortion: str = "tpv",
    ref_crpix: tuple[float, float] | None = None,
    max_crpix_offset: float = float("inf"),
) -> dict[str, Any]:
    """Propagate the solved WCS to one HDU, returning updated header."""
    updated = dict(header)

    crpix1 = float(updated.get("CRPIX1", 1.0))
    crpix2 = float(updated.get("CRPIX2", 1.0))

    wcs_hdr = solved_wcs.to_header()

    # Override CD matrix
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        if key in wcs_hdr:
            updated[key] = wcs_hdr[key]

    # Override CTYPE, stripping -SIP suffix and optionally adding -TPV
    for key in ("CTYPE1", "CTYPE2"):
        if key in wcs_hdr:
            ctype = str(wcs_hdr[key])
            if ctype.endswith("-SIP"):
                ctype = ctype[:-4]
            if distortion == "tpv":
                has_pv = any(k.startswith("PV") for k in solved_header)
                if has_pv and not ctype.endswith("-TPV"):
                    ctype = ctype + "-TPV"
            updated[key] = ctype

    # Remove PC / CDELT / CROTA if present (they conflict with CD)
    for key in list(updated.keys()):
        if key.startswith("PC") or key in ("CROTA1", "CROTA2"):
            del updated[key]

    # If this CCD is far from the reference, drop distortion entirely
    _use_distortion = True
    if max_crpix_offset < float("inf") and ref_crpix is not None:
        import math as _math

        _dx = crpix1 - ref_crpix[0]
        _dy = crpix2 - ref_crpix[1]
        if _math.sqrt(_dx * _dx + _dy * _dy) > max_crpix_offset:
            _use_distortion = False

    # Distortion keywords
    if not _use_distortion:
        # Pure TAN: ensure CTYPE has no -SIP/-TPV suffix
        for key in ("CTYPE1", "CTYPE2"):
            ctype = str(updated.get(key, ""))
            for suffix in ("-SIP", "-TPV"):
                if ctype.endswith(suffix):
                    ctype = ctype[:-4]
            updated[key] = ctype
        # Strip all distortion keywords
        for key in list(updated.keys()):
            if key.startswith(("A_", "B_", "AP_", "BP_", "PV")):
                del updated[key]
    elif distortion == "sip":
        # Copy SIP coefficients
        for key in list(updated.keys()):
            if key.startswith(("A_", "B_", "AP_", "BP_")):
                del updated[key]
        for key, value in solved_header.items():
            if key.startswith(("A_", "B_", "AP_", "BP_")):
                updated[key] = value
        # Remove PV when using SIP
        for key in list(updated.keys()):
            if key.startswith("PV"):
                del updated[key]
    else:
        # tpv (default): copy PV coefficients, discard SIP
        for key in list(updated.keys()):
            if key.startswith(("A_", "B_", "AP_", "BP_")):
                del updated[key]
        for key in list(updated.keys()):
            if key.startswith("PV"):
                del updated[key]
        for key, value in solved_header.items():
            if key.startswith("PV"):
                updated[key] = value

    # Copy EQUINOX / RADESYS / MJD-OBS etc from solved if present
    for key in ("EQUINOX", "RADESYS", "RADECSYS", "MJD-OBS", "DATE-OBS"):
        if key in solved_header:
            updated[key] = solved_header[key]

    # Compute new CRVAL: evaluate solved WCS at this CCD's CRPIX
    ra, dec = solved_wcs.pixel_to_world(crpix1, crpix2, origin=1)
    updated["CRVAL1"] = ra.item()
    updated["CRVAL2"] = dec.item()

    return updated


def cmd_propagate(args: argparse.Namespace) -> None:
    mef_path = Path(args.mef)
    output_path = Path(args.output)

    # Load the solved WCS
    if args.solved_wcs:
        print(f"Reading solved WCS from {args.solved_wcs}")
        solved_header = _read_wcs_file(args.solved_wcs)
        solved_wcs = WCS(solved_header)
        ref_desc = f"file {args.solved_wcs}"
    elif args.ref_hdu is not None:
        print(f"Reading MEF {mef_path}")
        with torchfits.open(str(mef_path)) as hdul:
            solved_header = dict(hdul[args.ref_hdu].header)
        solved_wcs = WCS(solved_header)
        ref_desc = f"HDU {args.ref_hdu}"
    else:
        print("error: specify --solved-wcs or --ref-hdu", file=sys.stderr)
        sys.exit(1)

    cd11 = solved_wcs.wcs_params.get("CD1_1")
    cd11_str = f"{cd11:.12g}" if isinstance(cd11, (int, float)) else str(cd11)
    print(
        f"Solved WCS: CTYPE={solved_wcs.wcs_params.get('CTYPE1', '?')}, "
        f"CD1_1={cd11_str}, "
        f"has_sip={solved_wcs.sip is not None}, "
        f"has_tpv={solved_wcs.tpv is not None}"
    )

    # Reference CRPIX for distortion-approximation gating
    ref_crpix = (
        float(solved_header.get("CRPIX1", 1.0)),
        float(solved_header.get("CRPIX2", 1.0)),
    )
    max_crpix_offset = float(getattr(args, "max_crpix_offset", float("inf")))
    if max_crpix_offset < float("inf"):
        print(
            f"Distortion gating: max_crpix_offset={max_crpix_offset:.0f} "
            f"(CCDs farther than {max_crpix_offset:.0f} pixels from ref CRPIX"
            f" ({ref_crpix[0]:.1f}, {ref_crpix[1]:.1f}) get pure-TAN)"
        )

    # Open MEF and update each extension
    print(f"Propagating WCS from {ref_desc} to all HDUs in {mef_path}")
    with torchfits.open(str(mef_path)) as hdul:
        image_hdus = [
            i
            for i, hdu in enumerate(hdul)
            if isinstance(hdu, torchfits.TensorHDU)
            and int(hdu.header.get("NAXIS1", 0)) > 0
            and int(hdu.header.get("NAXIS2", 0)) > 0
        ]
        print(f"Found {len(image_hdus)} image HDUs: {image_hdus}")

        for idx in image_hdus:
            hdu = hdul[idx]
            updated_header = _propagate_hdu(
                solved_wcs,
                solved_header,
                hdu.header,
                idx,
                distortion=args.distortion,
                ref_crpix=ref_crpix,
                max_crpix_offset=max_crpix_offset,
            )
            for key in list(hdu.header):
                if key not in updated_header:
                    del hdu.header[key]
            for key, value in updated_header.items():
                if hdu.header.get(key) != value:
                    hdu.header[key] = value
            print(
                f"  HDU {idx}: CRPIX=({float(hdu.header['CRPIX1']):.2f}, "
                f"{float(hdu.header['CRPIX2']):.2f}), "
                f"CRVAL=({float(hdu.header['CRVAL1']):.8f}, "
                f"{float(hdu.header['CRVAL2']):.8f})"
            )

        print(f"Writing {output_path}")
        torchfits.write(str(output_path), hdul, overwrite=True)

    print("Done.")


def cmd_merge_scamp(args: argparse.Namespace) -> None:
    mef_path = Path(args.mef)
    output_path = Path(args.output)
    scamp_prefix = args.scamp_prefix

    print(f"Reading MEF {mef_path}")
    with torchfits.open(str(mef_path)) as hdul:
        image_hdus = [
            i
            for i, hdu in enumerate(hdul)
            if isinstance(hdu, torchfits.TensorHDU)
            and int(hdu.header.get("NAXIS1", 0)) > 0
            and int(hdu.header.get("NAXIS2", 0)) > 0
        ]
        print(f"Found {len(image_hdus)} image HDUs: {image_hdus}")

        for idx in image_hdus:
            head_path = f"{scamp_prefix}_{idx}.head"
            if not Path(head_path).exists():
                print(f"  WARNING: {head_path} not found, skipping HDU {idx}")
                continue

            scamp_header = _read_scamp_header(head_path)
            hdu = hdul[idx]
            old_crval = (hdu.header.get("CRVAL1", "?"), hdu.header.get("CRVAL2", "?"))

            # Apply SCAMP header (overlay onto existing header)
            for key, value in scamp_header.items():
                if key not in {"HISTORY", "COMMENT"} and hdu.header.get(key) != value:
                    hdu.header[key] = value

            print(
                f"  HDU {idx}: applied {head_path}, "
                f"CRVAL old=({old_crval[0]}, {old_crval[1]}) "
                f"new=({hdu.header.get('CRVAL1', '?'):.8f}, {hdu.header.get('CRVAL2', '?'):.8f})"
            )

        print(f"Writing {output_path}")
        torchfits.write(str(output_path), hdul, overwrite=True)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propagate or merge WCS solutions across MEF extensions"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # propagate
    p_prop = sub.add_parser(
        "propagate", help="Propagate WCS from solved reference to all HDUs"
    )
    p_prop.add_argument("--mef", required=True, help="Input MEF file (.fits, .fits.fz)")
    p_prop.add_argument("--solved-wcs", help="Solved .wcs file from astrometry.net")
    p_prop.add_argument(
        "--ref-hdu", type=int, help="Reference HDU index within the MEF"
    )
    p_prop.add_argument(
        "--distortion",
        choices=["sip", "tpv"],
        default="tpv",
        help="Distortion format (default: tpv; SWarp requires tpv)",
    )
    p_prop.add_argument(
        "--max-crpix-offset",
        type=float,
        default=float("inf"),
        help="Fall back to pure-TAN if CCD CRPIX offset from reference "
        "exceeds this value (default: no fallback)",
    )
    p_prop.add_argument("--output", "-o", required=True, help="Output MEF file")
    p_prop.set_defaults(func=cmd_propagate)

    # merge-scamp
    p_merge = sub.add_parser(
        "merge-scamp", help="Merge SCAMP per-extension .head files into MEF"
    )
    p_merge.add_argument("--mef", required=True, help="Input MEF file")
    p_merge.add_argument(
        "--scamp-prefix",
        required=True,
        help="Path prefix for SCAMP .head files (e.g. 'scamp/raw' -> raw_1.head)",
    )
    p_merge.add_argument("--output", "-o", required=True, help="Output MEF file")
    p_merge.set_defaults(func=cmd_merge_scamp)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
