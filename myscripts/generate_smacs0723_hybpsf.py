#!/usr/bin/env python3
"""Generate one centered, double-sampled HybPSF for every exposure.

This is a thin, auditable wrapper around the public ``linngithub0/HybPSF``
implementation. Run that project's ``gen_cat.py`` first so each exposure has
the required PC FITS file. By default this script discovers and pairs files
from ``--exposure-dir`` and ``--pc-dir``. ``--spec`` is available only when
the automatic filename matching is insufficient.

[
  {
    "exposure": "..._nrcblong_cal_1overf_completed.fits",
    "pc_file": "..._nrcblong_cal_1overf_PC.fits",
    "x": 1024.0,
    "y": 1024.0
  }
]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


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


def _load_specs(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("specification must be a non-empty JSON list")
    return payload


def _center_xy(path: Path) -> tuple[float, float]:
    with fits.open(path, memmap=True) as hdul:
        height, width = hdul["SCI"].data.shape
    return (float(width - 1) / 2.0, float(height - 1) / 2.0)


def _discover_specs(
    exposure_dir: Path,
    pc_dir: Path,
    *,
    exposure_glob: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for exposure in sorted(exposure_dir.glob(exposure_glob)):
        key = _exposure_key(exposure)
        candidates = sorted(pc_dir.glob(f"{key}*_PC.fits"))
        if not candidates:
            # HybPSF's gen_cat.py commonly receives the pre-completion 1/f
            # filename, so also match after dropping the completion suffix.
            short_key = key.replace("_completed", "")
            candidates = sorted(pc_dir.glob(f"{short_key}*_PC.fits"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one PC file for {exposure.name}, found "
                f"{[path.name for path in candidates]}"
            )
        x, y = _center_xy(exposure)
        rows.append(
            {
                "exposure": str(exposure.resolve()),
                "pc_file": str(candidates[0].resolve()),
                "x": x,
                "y": y,
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"no exposures matching {exposure_glob!r} under {exposure_dir}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybpsf-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--exposure-dir", type=Path)
    parser.add_argument("--pc-dir", type=Path)
    parser.add_argument(
        "--exposure-glob",
        default="*_cal_1overf_completed.fits",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-ng",
        type=int,
        default=1501,
        help="Paper-scale PSF stamp diameter; must be odd.",
    )
    parser.add_argument(
        "--oversampling",
        type=int,
        default=2,
        choices=(1, 2),
    )
    args = parser.parse_args()

    if args.target_ng <= 0 or args.target_ng % 2 == 0:
        raise ValueError("target-ng must be a positive odd integer")
    sys.path.insert(0, str(args.hybpsf_root.resolve()))
    from psf_fit import get_webPSF  # type: ignore

    if args.spec is not None:
        specs = _load_specs(args.spec)
    else:
        if args.exposure_dir is None or args.pc_dir is None:
            parser.error(
                "provide --spec, or provide both --exposure-dir and --pc-dir"
            )
        specs = _discover_specs(
            args.exposure_dir,
            args.pc_dir,
            exposure_glob=str(args.exposure_glob),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in specs:
        exposure = Path(str(row["exposure"])).expanduser().resolve()
        pc_file = Path(str(row["pc_file"])).expanduser().resolve()
        x = float(row.get("x", 1024.0))
        y = float(row.get("y", 1024.0))
        if not exposure.is_file():
            raise FileNotFoundError(exposure)
        if not pc_file.is_file():
            raise FileNotFoundError(pc_file)

        models = get_webPSF(
            PCfile=str(pc_file),
            psfpos=np.asarray([[x, y]], dtype=np.float64),
            Target_Ng=int(args.target_ng),
            osam=int(args.oversampling),
        )
        psf = np.asarray(models[0], dtype=np.float32)
        if psf.ndim != 2:
            raise ValueError(f"HybPSF returned invalid shape {psf.shape}")
        psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
        psf = np.maximum(psf, 0.0)
        total = float(psf.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"HybPSF for {exposure.name} has invalid sum")
        psf /= total

        output = args.output_dir / f"{_exposure_key(exposure)}_hybpsf.fits"
        header = fits.Header()
        header["EXPOSURE"] = exposure.name
        header["PCFILE"] = pc_file.name
        header["PSFX"] = x
        header["PSFY"] = y
        header["OVERSAMP"] = int(args.oversampling)
        header["TARGETNG"] = int(args.target_ng)
        fits.PrimaryHDU(data=psf, header=header).writeto(output, overwrite=True)
        rows.append(
            {
                "exposure": str(exposure),
                "pc_file": str(pc_file),
                "output": str(output),
                "shape": list(psf.shape),
                "position_xy": [x, y],
            }
        )
        print(f"wrote {output}")

    summary_path = args.output_dir / "hybpsf_summary.json"
    summary_path.write_text(json.dumps({"outputs": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
