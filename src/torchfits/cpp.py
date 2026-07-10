from __future__ import annotations

from pathlib import Path
from astropy.io import fits


def read_header_dict(path: str | Path, hdu: int = 0) -> dict:
    with fits.open(path, memmap=False) as hdul:
        return dict(hdul[hdu].header)
