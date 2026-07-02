#!/usr/bin/env python3
#make residual fits
from pathlib import Path
import argparse

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def read_2d_hdu(path, hdu_spec="auto"):
    """
    Read a 2D image HDU from FITS/FITS.fz.

    hdu_spec:
      - "auto": prefer SCI, otherwise first 2D HDU
      - integer string, e.g. "0", "1"
      - extension name, e.g. "SCI"
    """
    path = Path(path)

    with fits.open(path, memmap=False) as hdul:
        if hdu_spec != "auto":
            if hdu_spec.isdigit():
                hdu = hdul[int(hdu_spec)]
            else:
                hdu = hdul[hdu_spec]

            if hdu.data is None or hdu.data.ndim != 2:
                raise RuntimeError(f"HDU {hdu_spec} in {path} is not a 2D image.")

            return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()

        # Prefer SCI extension
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", None) == 2:
                if hdu.header.get("EXTNAME", "").upper() == "SCI":
                    return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()

        # Otherwise first 2D image
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", None) == 2:
                return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()

    raise RuntimeError(f"No 2D image HDU found in {path}")


def finite_median(x):
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.nan
    return np.nanmedian(x[mask])


def wcs_signature(header):
    """
    Lightweight WCS signature for deciding whether two images are likely on the same grid.
    This is intentionally conservative.
    """
    keys = [
        "CTYPE1", "CTYPE2",
        "CRVAL1", "CRVAL2",
        "CRPIX1", "CRPIX2",
        "CDELT1", "CDELT2",
        "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "PC1_1", "PC1_2", "PC2_1", "PC2_2",
        "CUNIT1", "CUNIT2",
    ]
    return tuple(header.get(k, None) for k in keys)


def need_reproject(data1, header1, data2, header2):
    if data1.shape != data2.shape:
        return True
    if wcs_signature(header1) != wcs_signature(header2):
        return True
    return False


def do_reproject(data, source_header, target_header, target_shape, order="bilinear"):
    try:
        from reproject import reproject_interp
    except ImportError as exc:
        raise ImportError(
            "The package 'reproject' is required for reprojection. "
            "Install it with: pixi add -e <env> reproject"
        ) from exc

    source_wcs = WCS(source_header)
    target_wcs = WCS(target_header)

    reproj_data, footprint = reproject_interp(
        (data, source_wcs),
        target_wcs,
        shape_out=target_shape,
        order=order,
        return_footprint=True,
    )

    return reproj_data.astype(np.float32), footprint.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Generate residual FITS: residual = input1 - input2."
    )

    parser.add_argument("--input1", required=True, help="First input FITS file.")
    parser.add_argument("--input2", required=True, help="Second input FITS file.")
    parser.add_argument("--output", required=True, help="Output residual FITS path.")

    parser.add_argument(
        "--hdu1",
        default="auto",
        help='HDU for input1: "auto", integer index, or extension name. Default: auto.',
    )
    parser.add_argument(
        "--hdu2",
        default="auto",
        help='HDU for input2: "auto", integer index, or extension name. Default: auto.',
    )

    parser.add_argument(
        "--reproject",
        choices=["auto", "yes", "no"],
        default="auto",
        help=(
            "Whether to reproject before subtraction. "
            "auto = reproject if shape/WCS differs. Default: auto."
        ),
    )

    parser.add_argument(
        "--target-grid",
        choices=["input1", "input2"],
        default="input2",
        help=(
            "Output grid when reprojecting. "
            "Default: input2, so input1 is reprojected to input2 grid."
        ),
    )

    parser.add_argument(
        "--order",
        default="bilinear",
        choices=["nearest-neighbor", "bilinear", "biquadratic", "bicubic"],
        help="Interpolation order for reproject_interp. Default: bilinear.",
    )

    parser.add_argument(
        "--median-match",
        action="store_true",
        help="Subtract each image's finite median before residual.",
    )

    parser.add_argument(
        "--nan-outside-overlap",
        action="store_true",
        help="Set pixels outside reprojection footprint to NaN.",
    )

    args = parser.parse_args()

    input1_path = Path(args.input1)
    input2_path = Path(args.input2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading input1: {input1_path}")
    data1, header1 = read_2d_hdu(input1_path, args.hdu1)

    print(f"Reading input2: {input2_path}")
    data2, header2 = read_2d_hdu(input2_path, args.hdu2)

    print(f"Input1 shape: {data1.shape}")
    print(f"Input2 shape: {data2.shape}")

    if args.median_match:
        med1 = finite_median(data1)
        med2 = finite_median(data2)
        print(f"Median match enabled:")
        print(f"  input1 median = {med1}")
        print(f"  input2 median = {med2}")
        data1 = data1 - med1
        data2 = data2 - med2

    if args.reproject == "yes":
        use_reproject = True
    elif args.reproject == "no":
        use_reproject = False
    else:
        use_reproject = need_reproject(data1, header1, data2, header2)

    print(f"Reproject: {use_reproject}")

    footprint = None

    if use_reproject:
        if args.target_grid == "input2":
            print("Reprojecting input1 onto input2 grid...")
            data1_on_target, footprint = do_reproject(
                data1,
                header1,
                header2,
                data2.shape,
                order=args.order,
            )
            target_header = header2.copy()
            target_data1 = data1_on_target
            target_data2 = data2

        else:
            print("Reprojecting input2 onto input1 grid...")
            data2_on_target, footprint = do_reproject(
                data2,
                header2,
                header1,
                data1.shape,
                order=args.order,
            )
            target_header = header1.copy()
            target_data1 = data1
            target_data2 = data2_on_target

    else:
        if data1.shape != data2.shape:
            raise RuntimeError(
                "Input shapes differ but --reproject no was selected. "
                "Use --reproject yes or --reproject auto."
            )

        target_header = header1.copy()
        target_data1 = data1
        target_data2 = data2

    residual = target_data1 - target_data2

    valid = (
        np.isfinite(target_data1)
        & np.isfinite(target_data2)
        & np.isfinite(residual)
    )

    if use_reproject and args.nan_outside_overlap and footprint is not None:
        valid &= footprint > 0

    residual = residual.astype(np.float32)
    residual[~valid] = np.nan

    print("Residual stats on valid pixels:")
    print(f"  valid pixels = {int(valid.sum())}")
    print(f"  finite frac  = {float(np.isfinite(residual).mean())}")
    print(f"  median       = {float(np.nanmedian(residual))}")
    print(f"  std          = {float(np.nanstd(residual))}")

    out_header = target_header.copy()
    out_header["EXTNAME"] = "RESIDUAL"
    out_header["HISTORY"] = "Residual generated by make_residual_fits.py"
    out_header["HISTORY"] = f"input1 = {input1_path}"
    out_header["HISTORY"] = f"input2 = {input2_path}"
    out_header["HISTORY"] = "residual = input1 - input2"
    out_header["HISTORY"] = f"reproject = {use_reproject}"
    out_header["HISTORY"] = f"target_grid = {args.target_grid}"
    if args.median_match:
        out_header["HISTORY"] = "Each input median was subtracted before residual."

    hdus = [
        fits.PrimaryHDU(),
        fits.ImageHDU(
            data=residual,
            header=out_header,
            name="RESIDUAL",
        ),
        fits.ImageHDU(
            data=target_data1.astype(np.float32),
            header=target_header,
            name="INPUT1_ON_GRID",
        ),
        fits.ImageHDU(
            data=target_data2.astype(np.float32),
            header=target_header,
            name="INPUT2_ON_GRID",
        ),
        fits.ImageHDU(
            data=valid.astype(np.uint8),
            header=target_header,
            name="VALIDMASK",
        ),
    ]

    if footprint is not None:
        hdus.append(
            fits.ImageHDU(
                data=footprint.astype(np.float32),
                header=target_header,
                name="FOOTPRINT",
            )
        )

    fits.HDUList(hdus).writeto(output_path, overwrite=True)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()