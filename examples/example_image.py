"""Write and read a small image through the torchsky/torchfits boundary."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torchfits

import torchsky


def main() -> None:
    image = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 4.5,
        "CRPIX2": 4.5,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -1.0 / 3600.0,
        "CDELT2": 1.0 / 3600.0,
    }

    with tempfile.TemporaryDirectory(prefix="torchsky-image-example-") as directory:
        path = Path(directory) / "image.fits"
        torchfits.write(path, image, header=header)
        loaded, loaded_header = torchsky.read(str(path), return_header=True)

    print(f"image shape: {tuple(loaded.shape)}")
    print(f"pixel (0, 0): {loaded[0, 0].item():.1f}")
    print(f"WCS axes: {loaded_header['CTYPE1']}, {loaded_header['CTYPE2']}")


if __name__ == "__main__":
    main()
