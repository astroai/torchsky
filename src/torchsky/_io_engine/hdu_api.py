"""HDU/header access helpers for root FITS I/O."""

from __future__ import annotations

import os
from typing import Callable, Optional, Union

import torch

from ..hdu import HDUList, Header
from ..wcs import WCS


def open_hdulist(path: str, mode: str = "r") -> HDUList:
    """Open a FITS file for reading/writing."""
    if mode == "r" and not os.path.exists(path):
        raise FileNotFoundError(f"FITS file not found: {path}")

    try:
        return HDUList.fromfile(path, mode)
    except PermissionError:
        raise PermissionError(f"Permission denied accessing file: {path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to open FITS file '{path}': {exc}") from exc


def get_header(
    path: str,
    hdu: Union[int, str, None] = None,
    *,
    autodetect_hdu: Callable[[str, int], int],
) -> Header:
    """Get the header of a FITS file."""
    import torchsky.cpp as cpp

    if hdu is None or (isinstance(hdu, str) and hdu.strip().lower() == "auto"):
        hdu = autodetect_hdu(path, 16)

    if isinstance(hdu, str):
        if hasattr(cpp, "resolve_hdu_name_cached"):
            try:
                hdu = int(cpp.resolve_hdu_name_cached(path, hdu))
                return Header(cpp.read_header_dict(path, hdu))
            except Exception:
                pass

        for i in range(100):  # Fallback scan
            try:
                header_data = cpp.read_header_dict(path, i)
                if not header_data:
                    break
                header = Header(header_data)
                if header.get("EXTNAME") == hdu:
                    return header
            except Exception:
                break
        raise ValueError(f"HDU '{hdu}' not found")

    return Header(cpp.read_header_dict(path, hdu))


def get_wcs(
    path: str,
    hdu: Union[int, str, None] = "auto",
    device: Optional[Union[str, torch.device]] = None,
    *,
    get_header_func: Callable[[str, Union[int, str, None]], Header],
) -> WCS:
    """Build a WCS object directly from a FITS file/header HDU."""
    wcs = WCS(get_header_func(path, hdu))
    if device is not None:
        wcs = wcs.to(torch.device(device))
    return wcs
