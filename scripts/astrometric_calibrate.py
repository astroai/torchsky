#!/usr/bin/env python
"""Orchestrate astrometric calibration + coadd for MEF exposures.

Pipeline:
  1. solve-field on each exposure's reference CCD
  2. Propagate the solved WCS to all MEF extensions
  3. (Optional) SCAMP refinement via SExtractor + scamp
  4. torchsky coadd on the calibrated exposures

Examples
--------
  # Solve + propagate + coadd:
  python scripts/astrometric_calibrate.py solve \\
      exposures/*.fits.fz \\
      --solved-dir solved \\
      --ref-hdu 19

  # Solve + propagate + coadd all in one:
  python scripts/astrometric_calibrate.py full \\
      exposures/*.fits.fz \\
      --solved-dir solved \\
      --output calibrated_mosaic.fits \\
      --ref-hdu 19

  # Solve + SCAMP + coadd (best quality):
  python scripts/astrometric_calibrate.py full \\
      exposures/*.fits.fz \\
      --solved-dir solved \\
      --output calibrated_mosaic.fits \\
      --scamp
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torchfits


def _pixi_exec_solve_field() -> list[str]:
    """Return the pixi exec prefix for running solve-field."""
    import platform

    machine = platform.machine()
    if machine == "arm64":
        return [
            "pixi",
            "exec",
            "--platform",
            "osx-64",
            "--channel",
            "conda-forge",
            "-s",
            "astrometry",
            "--",
            "solve-field",
        ]
    return [
        "pixi",
        "exec",
        "--channel",
        "conda-forge",
        "-s",
        "astrometry",
        "--",
        "solve-field",
    ]


def _find_tool(name: str) -> str:
    """Find a tool on PATH, or return empty string."""
    path = shutil.which(name)
    if path is None:
        return ""
    return path


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command and print it."""
    print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, check=False, **kwargs)


def _extract_ccd(mef_path: str, hdu_index: int, out_path: str) -> bool:
    """Extract a single CCD from a MEF to a separate FITS file."""
    try:
        with torchfits.open(mef_path) as hdul:
            if hdu_index < 0 or hdu_index >= len(hdul):
                print(
                    f"  HDU {hdu_index} out of range (max {len(hdul) - 1})",
                    file=sys.stderr,
                )
                return False
            hdu = hdul[hdu_index]
            if (
                not isinstance(hdu, torchfits.TensorHDU)
                or int(hdu.header.get("NAXIS1", 0)) <= 0
                or int(hdu.header.get("NAXIS2", 0)) <= 0
            ):
                print(f"  HDU {hdu_index} is not an image HDU", file=sys.stderr)
                return False
            tensor = hdu.to_tensor()
            extracted = torchfits.HDUList(
                [torchfits.TensorHDU(data=tensor, header=hdu.header)]
            )
            torchfits.write(out_path, extracted, overwrite=True)
            return True
    except Exception as e:
        print(f"  Error extracting HDU {hdu_index}: {e}", file=sys.stderr)
        return False


def cmd_solve(args: argparse.Namespace) -> int:
    """Run solve-field + propagate for each exposure."""
    solve_prefix = _pixi_exec_solve_field()
    # Quick check that solve-field is accessible
    check = subprocess.run(solve_prefix + ["--version"], capture_output=True, text=True)
    if check.returncode != 0:
        print("error: solve-field not accessible via pixi exec", file=sys.stderr)
        print(
            "Install astrometry index files: pixi exec --platform osx-64 --channel conda-forge -s astrometry -- solve-field --default-index",
            file=sys.stderr,
        )
        return 1

    exposures = list(map(str, args.exposures))
    if not exposures:
        print("error: no exposure files specified", file=sys.stderr)
        return 1

    solved_dir = Path(args.solved_dir)
    solved_dir.mkdir(parents=True, exist_ok=True)

    print(f"Solving {len(exposures)} exposures with solve-field ...")
    for i, mef_path in enumerate(exposures):
        base = Path(mef_path).stem.replace(".fits", "").replace(".fits", "")
        print(f"\n[{i + 1}/{len(exposures)}] {mef_path}")

        # Extract reference CCD to temp file
        ref_ccd_path = str(solved_dir / f"{base}_ref_ccd.fits")
        if not _extract_ccd(mef_path, args.ref_hdu, ref_ccd_path):
            print(f"  Skipping {mef_path}", file=sys.stderr)
            continue

        # Run solve-field
        solve_out = str(solved_dir / base)
        solve_cmd = solve_prefix + [
            "--no-plots",
            "--no-remove-lines",
            "--crpix-center",
            "--scale-units",
            "arcsecperpix",
            "--scale-low",
            str(args.scale_low),
            "--scale-high",
            str(args.scale_high),
            "--tweak-order",
            str(args.tweak_order),
            "--cpulimit",
            str(args.cpulimit),
            ref_ccd_path,
            "-o",
            solve_out,
        ]
        result = _run(solve_cmd)
        if result.returncode != 0:
            print(
                f"  solve-field failed (exit {result.returncode}), skipping",
                file=sys.stderr,
            )
            continue

        # Locate the .wcs file
        wcs_candidates = list(Path(solved_dir).glob(f"{base}_ref_ccd.wcs"))
        if not wcs_candidates:
            # solve-field may name it differently
            wcs_candidates = list(Path(solved_dir).glob(f"{base}*.wcs"))
        if not wcs_candidates:
            print(f"  WARNING: no .wcs file found for {base}", file=sys.stderr)
            wcs_path = f"{solve_out}.wcs"
            if Path(wcs_path).exists():
                wcs_candidates = [Path(wcs_path)]
        wcs_path = str(wcs_candidates[0]) if wcs_candidates else ""
        if not wcs_path or not Path(wcs_path).exists():
            print(
                "  WARNING: .wcs file not found, skipping propagation", file=sys.stderr
            )
            continue

        # Propagate
        propagated_path = str(solved_dir / f"{base}_propagated.fits")
        prop_cmd = [
            sys.executable,
            str(Path(__file__).parent / "apply_wcs_to_mef.py"),
            "propagate",
            "--mef",
            mef_path,
            "--solved-wcs",
            wcs_path,
            "--output",
            propagated_path,
        ]
        result = _run(prop_cmd)
        if result.returncode != 0:
            print(
                f"  WCS propagation failed (exit {result.returncode})", file=sys.stderr
            )
            continue

        # Clean up extracted CCD
        Path(ref_ccd_path).unlink(missing_ok=True)

    print("\nDone solving.")
    return 0


def cmd_full(args: argparse.Namespace) -> int:
    """Solve + optionally SCAMP + coadd."""
    # Step 1: solve
    solve_args = argparse.Namespace(
        exposures=args.exposures,
        solved_dir=args.solved_dir,
        ref_hdu=args.ref_hdu,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
        tweak_order=args.tweak_order,
        cpulimit=args.cpulimit,
    )
    rc = cmd_solve(solve_args)
    if rc != 0:
        return rc

    solved_dir = Path(args.solved_dir)
    propagated = sorted(solved_dir.glob("*_propagated.fits"))
    if not propagated:
        print("error: no propagated exposures found", file=sys.stderr)
        return 1

    # Step 2: optional SCAMP refinement
    if args.scamp:
        sex = _find_tool("sex")
        scamp = _find_tool("scamp")
        if not sex or not scamp:
            print("error: --scamp requires both sex and scamp on PATH", file=sys.stderr)
            print("  pixi run -e astrometry -- ... provides both", file=sys.stderr)
            return 1

        scamp_dir = solved_dir / "scamp"
        scamp_dir.mkdir(parents=True, exist_ok=True)

        print("\nRunning SExtractor on each propagated exposure ...")
        catalogs = []
        for prop_path in propagated:
            base = prop_path.stem
            cat_path = str(scamp_dir / f"{base}.cat")
            catalogs.append(cat_path)

            # Run sex on each extension
            with torchfits.open(str(prop_path)) as hdul:
                ext_indices = [
                    i
                    for i, hdu in enumerate(hdul)
                    if isinstance(hdu, torchfits.TensorHDU)
                    and int(hdu.header.get("NAXIS1", 0)) > 0
                    and int(hdu.header.get("NAXIS2", 0)) > 0
                ]

            for ext_idx in ext_indices:
                ext_cat = str(scamp_dir / f"{base}_ext{ext_idx}.cat")
                sex_cmd = [
                    sex,
                    f"{prop_path}[{ext_idx}]",
                    "-c",
                    args.sex_config,
                    "-CATALOG_NAME",
                    ext_cat,
                    "-CATALOG_TYPE",
                    "FITS_LDAC",
                ]
                result = _run(sex_cmd)
                if result.returncode != 0:
                    print(f"  sex failed on {prop_path}[{ext_idx}]", file=sys.stderr)

        # Gather all LDAC catalogs
        ldac_cats = sorted(scamp_dir.glob("*_ext*.cat"))
        if not ldac_cats:
            print("error: no SExtractor catalogs produced", file=sys.stderr)
            return 1

        print(f"\nRunning SCAMP on {len(ldac_cats)} catalogs ...")
        scamp_cmd = [
            scamp,
            *[str(c) for c in ldac_cats],
            "-c",
            args.scamp_config,
            "-ASTREF_CATALOG",
            "GAIA-DR3",
            "-POSITION_MAXERR",
            str(args.scamp_maxerr),
            "-POSITION_TOPOUT_MAXERR_ARCSEC",
            str(args.scamp_maxerr),
            "-DISTORT_DEGREES",
            str(args.scamp_distort_degree),
            "-CROSSID_RADIUS",
            str(args.scamp_crossid_radius),
            "-SOLUTION_METHOD",
            "RELATIVE",
            "-MOSAIC_TYPE",
            "FIX_FOCAL_PLANE",
        ]
        result = _run(scamp_cmd)
        if result.returncode != 0:
            print(f"SCAMP failed (exit {result.returncode})", file=sys.stderr)
            return 1

        # Merge SCAMP headers back into propagated files
        print("\nMerging SCAMP headers ...")
        for prop_path in propagated:
            base = prop_path.stem
            head_prefix = str(scamp_dir / base)

            merge_cmd = [
                sys.executable,
                str(Path(__file__).parent / "apply_wcs_to_mef.py"),
                "merge-scamp",
                "--mef",
                str(prop_path),
                "--scamp-prefix",
                head_prefix,
                "--output",
                str(solved_dir / f"{base}_calibrated.fits"),
            ]
            _run(merge_cmd)

        calibrated = sorted(solved_dir.glob("*_calibrated.fits"))
        coadd_inputs = calibrated if calibrated else propagated
    else:
        coadd_inputs = propagated

    # Step 3: torchsky coadd
    if args.output:
        print(f"\nCoadding {len(coadd_inputs)} exposures with torchsky ...")
        coadd_cmd = [
            sys.executable,
            "-c",
            "from torchsky.cli import main; main()",
            "map",
            str(args.output),
            *[str(f) for f in coadd_inputs],
            "--hdu",
            "0",
            "--kernel",
            args.kernel,
            "--stack",
            args.stack,
        ]
        if args.background_mode:
            coadd_cmd.extend(["--background-mode", args.background_mode])
        if args.device:
            coadd_cmd.extend(["--device", args.device])

        result = _run(coadd_cmd)
        if result.returncode != 0:
            print(f"torchsky coadd failed (exit {result.returncode})", file=sys.stderr)
            return result.returncode

        print(f"\nWrote calibrated coadd to {args.output}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Astrometric calibration pipeline for MEF exposures"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- solve ---
    p_solve = sub.add_parser("solve", help="Solve-field + propagate for each exposure")
    p_solve.add_argument("exposures", nargs="+", help="Exposure MEF files")
    p_solve.add_argument(
        "--solved-dir", default="solved", help="Output directory for solved files"
    )
    p_solve.add_argument(
        "--ref-hdu", type=int, default=19, help="Reference CCD HDU index"
    )
    p_solve.add_argument(
        "--scale-low",
        type=float,
        default=0.184,
        help="solve-field scale low (arcsec/pix)",
    )
    p_solve.add_argument(
        "--scale-high",
        type=float,
        default=0.190,
        help="solve-field scale high (arcsec/pix)",
    )
    p_solve.add_argument(
        "--tweak-order", type=int, default=3, help="SIP distortion order"
    )
    p_solve.add_argument(
        "--cpulimit", type=int, default=120, help="solve-field CPU time limit (seconds)"
    )
    p_solve.set_defaults(func=cmd_solve)

    # --- full ---
    p_full = sub.add_parser("full", help="Solve + SCAMP + coadd")
    p_full.add_argument("exposures", nargs="+", help="Exposure MEF files")
    p_full.add_argument("--output", "-o", help="Output coadd FITS file")
    p_full.add_argument(
        "--solved-dir", default="solved", help="Working directory for solved files"
    )
    p_full.add_argument(
        "--ref-hdu", type=int, default=19, help="Reference CCD HDU index"
    )
    p_full.add_argument(
        "--scale-low",
        type=float,
        default=0.184,
        help="solve-field scale low (arcsec/pix)",
    )
    p_full.add_argument(
        "--scale-high",
        type=float,
        default=0.190,
        help="solve-field scale high (arcsec/pix)",
    )
    p_full.add_argument(
        "--tweak-order", type=int, default=3, help="SIP distortion order"
    )
    p_full.add_argument(
        "--cpulimit", type=int, default=120, help="solve-field CPU time limit (seconds)"
    )

    # Coadd options
    p_full.add_argument("--kernel", default="lanczos3", help="Resampling kernel")
    p_full.add_argument("--stack", default="weighted_mean", help="Stacking method")
    p_full.add_argument(
        "--background-mode",
        choices=("none", "per-exposure", "overlap"),
        help="Background policy passed to torchsky map",
    )
    p_full.add_argument("--device", help="Torch device (cpu, cuda, mps)")

    # SCAMP options
    p_full.add_argument(
        "--scamp", action="store_true", help="Run SCAMP refinement after solving"
    )
    p_full.add_argument(
        "--sex-config", default="default.sex", help="SExtractor config file"
    )
    p_full.add_argument(
        "--scamp-config", default="default.scamp", help="SCAMP config file"
    )
    p_full.add_argument(
        "--scamp-maxerr",
        type=float,
        default=1.0,
        help="SCAMP position max error (arcsec)",
    )
    p_full.add_argument(
        "--scamp-distort-degree",
        type=int,
        default=5,
        help="SCAMP distortion polynomial degree",
    )
    p_full.add_argument(
        "--scamp-crossid-radius",
        type=float,
        default=45,
        help="SCAMP cross-ID radius (arcsec)",
    )
    p_full.set_defaults(func=cmd_full)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
