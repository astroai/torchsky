#!/usr/bin/env python3
"""Build small multi-band SMACS J0723 cutouts from UPdec JWST mosaic references.

The repo's public SMACS path uses **JWST NIRCam** UPdec ``*_i2d.fits.fz`` products
under ``data/updec_webb/``. There is no bundled HST ACS pipeline here; this script
produces a compact **multi-wavelength JWST** cutout suitable for the undergrad
tutorial notebook (fusion / hyperspectral themes).

Outputs go to ``data/smacs0723_minicutout/`` (or ``$TORCHSKY_DATA/smacs0723_minicutout/``)
alongside a ``manifest.json`` with SHA-256 checksums. FITS files are **not** committed.

Examples::

  pixi run -e default -- python scripts/build_smacs0723_minicutout.py --download
  pixi run -e default -- python scripts/build_smacs0723_minicutout.py --crop-size 256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BANDS = ("F090W", "F150W", "F277W", "F444W")

REFERENCE_NAMES: dict[str, str] = {
    "F090W": "F090Wb_UPDC-v2_iter0_i2d.fits.fz",
    "F150W": "F150Wb_UPDC-v2_iter0_i2d.fits.fz",
    "F277W": "F277Wb_UPDC-v2_iter0_i2d.fits.fz",
    "F444W": "F444Wb_UPDC-v2_iter0_i2d.fits.fz",
}


def _data_root() -> Path:
    raw = os.environ.get("TORCHSKY_DATA", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (ROOT / "data").resolve()


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip().upper() for part in value.split(",") if part.strip())


def _run_download_references(bands: tuple[str, ...]) -> None:
    script = ROOT / "scripts" / "download_updec_webb.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--bands",
            ",".join(bands),
            "--iterations",
            "0",
            "--output-dir",
            str(_data_root() / "updec_webb"),
        ],
        check=True,
    )


def _center_cutout_2d(
    data: np.ndarray,
    header: Any,
    *,
    crop_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = int(data.shape[0]), int(data.shape[1])
    cy, cx = h // 2, w // 2
    half = int(crop_size) // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(h, y0 + crop_size)
    x1 = min(w, x0 + crop_size)
    patch = np.asarray(data[y0:y1, x0:x1], dtype=np.float32)
    hdr = header.copy()
    hdr["NAXIS1"] = int(patch.shape[1])
    hdr["NAXIS2"] = int(patch.shape[0])
    if "CRPIX1" in hdr:
        hdr["CRPIX1"] = float(hdr["CRPIX1"]) - float(x0)
    if "CRPIX2" in hdr:
        hdr["CRPIX2"] = float(hdr["CRPIX2"]) - float(y0)
    return patch, dict(hdr)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bands",
        default=",".join(DEFAULT_BANDS),
        help="Comma-separated JWST / UPdec bands (default: four-band ladder).",
    )
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Run scripts/download_updec_webb.py for the requested bands (iter 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: $TORCHSKY_DATA/smacs0723_minicutout or data/smacs0723_minicutout.",
    )
    args = parser.parse_args()

    bands = _parse_csv(args.bands)
    data_root = _data_root()
    out_dir = args.output_dir or (data_root / "smacs0723_minicutout")
    out_dir.mkdir(parents=True, exist_ok=True)
    updec_dir = data_root / "updec_webb"

    if args.download:
        _run_download_references(bands)

    sys.path.insert(0, str(ROOT / "src"))
    import torch

    import torchsky

    manifest: dict[str, Any] = {
        "bands": list(bands),
        "crop_size": int(args.crop_size),
        "outputs": [],
    }

    from astropy.io import fits

    for band in bands:
        if band not in REFERENCE_NAMES:
            raise SystemExit(f"unknown band {band!r}; known {sorted(REFERENCE_NAMES)}")
        src = updec_dir / REFERENCE_NAMES[band]
        if not src.is_file():
            raise SystemExit(
                f"missing reference {src}; run with --download or place UPdec file manually."
            )
        with fits.open(src, memmap=False) as hdul:
            hdu = hdul["SCI"] if "SCI" in hdul else hdul[0]
            data = np.asarray(hdu.data, dtype=np.float32)
            patch, hdr = _center_cutout_2d(
                data, hdu.header, crop_size=int(args.crop_size)
            )
        hdr["BAND"] = (band, "Approximate NIRCam filter name")
        out_path = out_dir / f"smacs0723_{band}_cutout_{args.crop_size}.fits"
        tensor = torch.from_numpy(patch)
        torchsky.write(str(out_path), tensor, header=hdr, overwrite=True)
        manifest["outputs"].append(
            {
                "band": band,
                "path": str(out_path.relative_to(ROOT)),
                "sha256": _sha256(out_path),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest['outputs'])} cutouts under {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
