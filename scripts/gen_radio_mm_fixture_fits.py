#!/usr/bin/env python3
"""Write a tiny committed radio/mm-style FITS for integration tests (deterministic)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import torchsky

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    out = ROOT / "tests" / "data" / "radio_mm_stub.fits"
    out.parent.mkdir(parents=True, exist_ok=True)
    size = 16
    yy, xx = np.mgrid[0:size, 0:size]
    data = np.exp(
        -((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / 6.0
    ).astype(np.float32)
    hdr = {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": 187.5,
        "CRVAL2": -45.0,
        "CDELT1": -0.008,
        "CDELT2": 0.008,
        "BUNIT": "JY/BEAM",
        "BMAJ": 0.015,
        "BMIN": 0.012,
        "BPA": 22.5,
        "RESTFRQ": 2.30538e11,
        "SPECSYS": "LSRK",
    }
    torchsky.write(str(out), torch.from_numpy(data), header=hdr, overwrite=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
