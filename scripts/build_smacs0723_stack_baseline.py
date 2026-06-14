#!/usr/bin/env python3
"""Build a WCS-harmonized multiband stack baseline from SMACS J0723 cutouts.

Run ``scripts/build_smacs0723_minicutout.py`` first. Use the Pixi ``test``
environment so the FITS extension and comparison data dependencies are present.
This script reads that manifest, reprojects each cutout onto the first cutout's
WCS, and writes a small spectral FITS cube for comparison with neural-field
reconstructions.

Examples::

  pixi run -e test -- python scripts/build_smacs0723_stack_baseline.py --list-bands
  pixi run -e test -- python scripts/build_smacs0723_stack_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torchsky.map import ImageObservation, coadd_image_plane_cube  # noqa: E402

FILTER_BANDPASS_ANGSTROM: dict[str, tuple[float, float, float]] = {
    # Approximate NIRCam wide-filter support for tutorial metadata only.
    "F090W": (9000.0, 8000.0, 10000.0),
    "F150W": (15000.0, 13000.0, 17000.0),
    "F200W": (20000.0, 17000.0, 23000.0),
    "F277W": (27700.0, 24000.0, 32000.0),
    "F356W": (35600.0, 31000.0, 40000.0),
    "F444W": (44400.0, 39000.0, 50000.0),
}


def _resolve_manifest_path(path_text: str, *, manifest_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    root_path = (ROOT / path).resolve()
    if root_path.exists():
        return root_path
    return (manifest_dir / path).resolve()


def _write_nonuniform_cube_fits(
    path: str,
    cube: Any,
    *,
    spatial_header: dict,
    spectral_axis: Any,
    overwrite: bool = True,
) -> dict:
    from astropy.io import fits
    from torchsky.spectral import _strip_axis3_and_naxis_keys

    data = cube.cpu().numpy()
    clean_hdr = _strip_axis3_and_naxis_keys(spatial_header)
    header = fits.Header(clean_hdr)
    header["NAXIS"] = 3
    header["NAXIS1"] = data.shape[2]
    header["NAXIS2"] = data.shape[1]
    header["NAXIS3"] = data.shape[0]

    header["CTYPE3"] = str(spectral_axis.type)
    header["CUNIT3"] = str(spectral_axis.unit)

    vals = spectral_axis.values.detach().cpu().numpy()
    header["CRPIX3"] = 1.0
    header["CRVAL3"] = float(vals[0])
    if len(vals) >= 2:
        header["CDELT3"] = float(vals[1] - vals[0])
    else:
        header["CDELT3"] = 1.0

    for i, val in enumerate(vals):
        header[f"WAVE{i}"] = float(val)

    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=overwrite)
    return dict(header)


def _observations_from_manifest(manifest_path: Path) -> list[ImageObservation]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing {manifest_path}; run scripts/build_smacs0723_minicutout.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observations: list[ImageObservation] = []
    for item in manifest.get("outputs", []):
        band = str(item["band"]).upper()
        if band not in FILTER_BANDPASS_ANGSTROM:
            raise ValueError(f"unknown band {band!r}; update FILTER_BANDPASS_ANGSTROM")
        wavelength, spectral_min, spectral_max = FILTER_BANDPASS_ANGSTROM[band]
        observations.append(
            ImageObservation(
                path=str(
                    _resolve_manifest_path(
                        str(item["path"]), manifest_dir=manifest_path.parent
                    )
                ),
                wavelength=wavelength,
                spectral_min=spectral_min,
                spectral_max=spectral_max,
                transmission_center=wavelength,
                transmission_sigma=0.5 * (spectral_max - spectral_min) / 2.355,
                band_name=band,
            )
        )
    if not observations:
        raise ValueError(f"manifest {manifest_path} has no outputs")
    return observations


def _default_manifest_path() -> Path:
    return ROOT / "data" / "smacs0723_minicutout" / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_default_manifest_path(),
        help="Path to the minicutout manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output FITS cube path; default is next to the manifest.",
    )
    parser.add_argument(
        "--resampling-kernel",
        default="bilinear",
        choices=(
            "nearest",
            "bilinear",
            "drizzle_point",
            "drizzle_square",
            "spectral_nonuniform_fast",
        ),
        help=(
            "Reprojection kernel. Start with bilinear for debugging; use "
            "drizzle_square as the torchsky drizzle-like baseline."
        ),
    )
    parser.add_argument(
        "--drizzle-pixfrac",
        type=float,
        default=1.0,
        help="Pixel fraction for drizzle_* kernels.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--list-bands",
        action="store_true",
        help="List bands with approximate wavelength metadata and exit.",
    )
    args = parser.parse_args()

    if args.list_bands:
        for band, (center, lower, upper) in sorted(FILTER_BANDPASS_ANGSTROM.items()):
            print(f"{band} center={center:g}A range=[{lower:g}, {upper:g}]A")
        return 0

    manifest_path = args.manifest.expanduser().resolve()
    try:
        observations = _observations_from_manifest(manifest_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    result = coadd_image_plane_cube(
        observations,
        resampling_kernel=args.resampling_kernel,
        drizzle_pixfrac=float(args.drizzle_pixfrac),
        device=args.device,
    )
    output = args.output or (manifest_path.parent / "smacs0723_stack_baseline.fits")
    output.parent.mkdir(parents=True, exist_ok=True)
    header = _write_nonuniform_cube_fits(
        str(output),
        result.data,
        spatial_header=result.header,
        spectral_axis=result.spectral_axis,
        overwrite=True,
    )
    summary: dict[str, Any] = {
        "output": str(output),
        "source_manifest": str(manifest_path),
        "bands": [obs.band_name for obs in observations],
        "wavelength_angstrom": [
            float(obs.wavelength) for obs in observations if obs.wavelength is not None
        ],
        "shape": list(result.data.shape),
        "naxis": int(header["NAXIS"]),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
