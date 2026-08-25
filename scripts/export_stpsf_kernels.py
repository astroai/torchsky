#!/usr/bin/env python3
"""Export per-exposure WebbPSF kernels for native-pixel field training.

For every input CAL/CRF exposure, computes the detector-sampled WebbPSF
kernel for its band and detector, unit-sum normalizes it, and saves a float32
``.npy`` named ``{band}_{detector}_{index:03d}_psf.npy`` in ``--output-dir``.

Consumer contract (``src/torchsky/map/field/types.py``):
``PSFSpec(kernel2d=torch.from_numpy(np.load(path)))`` with a 2-D odd-sized,
non-negative kernel.

Requires the optional ``stpsf`` extras group; install with
``pip install 'torchsky[stpsf]'``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

NIRCAM_SHORT_WAVE_BANDS = frozenset(
    {
        "F070W", "F090W", "F115W", "F140M", "F150W", "F162M",
        "F164N", "F182M", "F187N", "F200W", "F210M", "F212N",
    }
)


def _band_and_detector(path: Path) -> tuple[str, str]:
    """(band, detector) tags from the exposure header or filename."""
    band = ""
    detector = ""
    try:
        from astropy.io import fits as astro_fits

        with astro_fits.open(str(path), memmap=False) as hdul:
            head = hdul[0].header
            band = str(head.get("FILTER", "") or head.get("PUPIL", ""))
            detector = str(head.get("DETECTOR", ""))
    except Exception:
        pass
    if not detector:
        parts = path.stem.split("_")
        detector = parts[4] if len(parts) > 4 else ""
    return band, detector


def _detector_pixel_scale_arcsec(band: str) -> float:
    """Nominal NIRCam detector pixel scale (arcsec) per wavelength channel."""
    return 0.031 if band.upper() in NIRCAM_SHORT_WAVE_BANDS else 0.063


def export_stpsf_kernels(
    inputs: list[Path],
    *,
    output_dir: Path,
    kernel_size: int = 33,
    oversample: int = 2,
):
    """Compute, unit-sum normalize, and save one [K, K] kernel per exposure."""
    try:
        import stpsf
    except ImportError as exc:
        raise RuntimeError(
            "stpsf kernel export requires the stpsf environment "
            "(pip install 'torchsky[stpsf]')"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    half = int(kernel_size) // 2
    size = 2 * half + 1
    destinations: list[Path] = []
    last_band = ""
    for index, path in enumerate(inputs):
        band, detector = _band_and_detector(path)
        if not band or not detector:
            raise ValueError(
                f"{path}: cannot determine FILTER/DETECTOR for PSF export"
            )
        last_band = band
        nircam = stpsf.NIRCam()
        nircam.filter = band.upper()
        nircam.detector = detector.upper()
        psf = nircam.calc_psf(
            fov_pixels=size,
            oversample=int(oversample),
        )
        # Detector-sampled plane: the convolution kernel consumers apply.
        data = np.asarray(psf["DET_DIST"].data, dtype=np.float64)
        if data.shape != (size, size):
            center = np.array(data.shape) // 2
            data = data[
                center[0] - half : center[0] - half + size,
                center[1] - half : center[1] - half + size,
            ]
        total = float(data.sum())
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError(f"{band}/{detector}: non-finite PSF kernel")
        destination = output_dir / f"{band.lower()}_{detector.lower()}_{index:03d}_psf.npy"
        np.save(destination, (data / total).astype(np.float32))
        destinations.append(destination)
    return {
        "destinations": destinations,
        "kernel_shape": (size, size),
        "pixel_scale_arcsec": _detector_pixel_scale_arcsec(last_band),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="CAL/CRF exposures (read-only)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-size", type=int, default=33,
                        help="odd kernel side length K in detector pixels")
    parser.add_argument("--oversample", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        print(f"missing input files: {', '.join(map(str, missing[:5]))}", file=sys.stderr)
        return 1
    try:
        info = export_stpsf_kernels(
            args.inputs,
            output_dir=args.output_dir,
            kernel_size=args.kernel_size,
            oversample=args.oversample,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"wrote {len(info['destinations'])} kernels shape={info['kernel_shape']} "
        f"pixel_scale={info['pixel_scale_arcsec']}\""
    )
    for destination in info["destinations"]:
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
